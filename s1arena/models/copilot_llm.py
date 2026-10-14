"""Chat LLMs served through a GitHub Copilot subscription (official Copilot SDK), with verbalized probabilities.

Spec: copilot:<Copilot model id>[@effort=<reasoning effort>], e.g. copilot:claude-sonnet-5.5 or
copilot:gpt-5.5@effort=high. List the ids, prices and efforts with `uv run python -m s1arena.models.copilot_llm`.

Every call opens a fresh Copilot session whose system prompt is replaced by render.SYSTEM, which has no tools, and
whose user turn is sent without the date header Copilot adds, so the model receives exactly the messages that llm:
models receive (render.messages) rather than Copilot's agent prompt. Replies are parsed like llm: replies. Copilot
exposes no temperature or max_tokens (it sends temperature 1 and its own output cap); effort sets the reasoning
effort (default: the model's own). Claude models always think through Copilot: claude-haiku-4.5 with a fixed
1024-token budget, newer ones adaptively (efforts start at low); GPT models accept effort=none. So copilot: runs
are not the same configuration as llm: runs of the same model. Usage is billed in Copilot AI credits and reported
as cost_usd at 1 AIC = $0.01. Every reply is checked against the model Copilot reports for each model call: a reply
served by another model (Copilot's auto mode, a model switch, a fusion of models) is a failed call, retried like a
rate limit, and the serving model is recorded as usage["served_model"].

Needs the copilot extra (uv sync --inexact --extra copilot) and a Copilot CLI that is logged in (`copilot login`); the SDK
drives the CLI found via COPILOT_CLI_PATH or PATH. With COPILOT_RUNTIME_URL (host:port) it instead connects to a
Copilot runtime started elsewhere, on a machine where Copilot is logged in
(`COPILOT_CONNECTION_TOKEN=<secret> copilot --headless --port <port>`, e.g. reached through an SSH tunnel), with
COPILOT_CONNECTION_TOKEN as the shared secret. A client whose connection to its runtime broke is replaced, and a
failed call is retried up to 8 times with backoff (Copilot's rate limit resets within a minute).
"""

from __future__ import annotations

import asyncio
import os
import random
import shutil
import tempfile
import threading
import time

from .. import render
from ..registry import register_model
from ..schema import DecisionRequest, ModelResponse
from .base import ModelAdapter
from .gateway_llm import GatewayLLM

USD_PER_AIC = 0.01
TIMEOUT_S = 600.0  # one reply, including reasoning
# session events that mean another model took over (Copilot's auto mode, a model change); the call then fails
MODEL_SWITCH_EVENTS = {"SessionModelChangeData", "AutoModeSwitchCompletedData", "SessionAutoModeResolvedData"}

_loop: asyncio.AbstractEventLoop | None = None
_client = None  # copilot.CopilotClient, shared by all threads; lives on _loop
_lock = threading.Lock()


def _call(coro, timeout: float | None = None):
    """Run a coroutine on the background loop that owns the Copilot client, from any thread."""
    return asyncio.run_coroutine_threadsafe(coro, _loop).result(timeout)


def client():
    global _loop, _client
    with _lock:
        if _client is None:
            # optional dependency, imported only when a copilot model is used
            from copilot import CopilotClient
            from copilot.client import RuntimeConnection

            url = os.environ.get("COPILOT_RUNTIME_URL")
            if not url and not os.environ.get("COPILOT_CLI_PATH") and shutil.which("copilot"):
                os.environ["COPILOT_CLI_PATH"] = shutil.which("copilot")
            if _loop is None:
                _loop = asyncio.new_event_loop()
                threading.Thread(target=_loop.run_forever, name="copilot-sdk", daemon=True).start()

            async def start():
                if url:  # a runtime started elsewhere; its own working directory is what sessions see
                    token = os.environ.get("COPILOT_CONNECTION_TOKEN") or None
                    c = CopilotClient(connection=RuntimeConnection.for_uri(url, connection_token=token))
                else:
                    c = CopilotClient(working_directory=tempfile.mkdtemp(prefix="s1arena-copilot-"))  # sees no repo files
                await c.start()
                return c

            _client = _call(start(), 120)
        return _client


def _drop(failed) -> None:
    """Forget a client whose connection to its runtime broke (a remote runtime or a tunnel restarted), once."""
    global _client
    with _lock:
        if _client is not failed:
            return  # another thread already replaced it
        _client = None
    try:
        _call(failed.stop(), 30)
    except Exception:
        pass


_models: list | None = None


def list_models() -> list:
    """Copilot's model list, fetched once per process and retried with backoff (the endpoint is rate-limited too)."""
    global _models
    for attempt in range(7):
        if _models is not None:
            return _models
        try:
            _models = _call(client().list_models(), 120)
        except Exception:
            if attempt == 6:
                raise
            time.sleep(min(30.0, 2**attempt) * (0.5 + random.random()))
    return _models


@register_model
class CopilotLLM(ModelAdapter):
    kind = "copilot"
    native_probs = False
    probability_source = "verbalized"
    reparse = GatewayLLM.reparse  # same reply format, same parser

    def __init__(self, spec: str, retries: int = 4) -> None:
        self.copilot_model, _, opts = spec.partition("@")
        options = dict(kv.split("=", 1) for kv in opts.split(",") if kv)
        unknown = set(options) - {"effort"}
        if unknown:
            raise ValueError(f"unknown copilot option(s) {sorted(unknown)}; known: effort")
        self.effort = options.get("effort")
        self.retries = max(retries, 8)  # rides out Copilot's rate limit ("reset in under a minute") and reconnects
        self._checked = False

    @property
    def model_id(self) -> str:
        return f"copilot/{self.copilot_model}" + (f"@effort={self.effort}" if self.effort else "")

    def token_prices(self) -> tuple[float, float] | None:
        """(input, output) USD per token from Copilot's model list (listed in AI credits per million tokens)."""
        for m in list_models():
            prices = m.billing.token_prices if m.id == self.copilot_model and m.billing else None
            if prices and prices.input_price is not None:
                return prices.input_price * USD_PER_AIC / 1e6, (prices.output_price or 0) * USD_PER_AIC / 1e6
        return None

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        system, user = (m["content"] for m in render.messages(request))
        start = time.perf_counter()
        try:
            self._check_model()
            text, usage, latency = self._ask(system, user)
        except Exception as error:
            latency = (time.perf_counter() - start) * 1000
            return ModelResponse(error=f"{type(error).__name__}: {str(error)[:300]}", latency_ms=latency)
        response = ModelResponse(latency_ms=latency, usage=usage, cost_usd=usage["aic"] * USD_PER_AIC, raw_text=text)
        return self.reparse(response, request)

    def _check_model(self) -> None:
        """Fail fast on a model id or effort this account cannot use, instead of retrying every call."""
        if self._checked:
            return
        models = {m.id: m for m in list_models()}
        if self.copilot_model not in models:
            raise ValueError(f"model {self.copilot_model!r} not available to this Copilot account; known: {sorted(models)}")
        efforts = models[self.copilot_model].supported_reasoning_efforts or []
        if self.effort and self.effort not in efforts:
            raise ValueError(f"effort {self.effort!r} not supported by {self.copilot_model}; supported: {efforts}")
        self._checked = True

    def _ask(self, system: str, user: str) -> tuple[str, dict, float]:
        """One reply, retried with backoff and jitter when the call fails (rate limits, timeouts, transport).

        Returns (text, usage, latency_ms); the latency is that of the successful attempt, from sending the message
        to the complete reply, without session setup / teardown or retry waits.
        """
        for attempt in range(self.retries + 1):
            c = None
            try:
                c = client()
                text, usage, latency = _call(self._session(c, system, user), TIMEOUT_S + 60)
                return text, {**usage, "attempts": attempt + 1}, latency
            except Exception:
                if c is not None and getattr(c, "_state", "connected") != "connected":
                    _drop(c)  # the next attempt connects afresh
                if attempt == self.retries:
                    raise
                time.sleep(min(30.0, 2**attempt) * (0.5 + random.random()))

    async def _session(self, c, system: str, user: str) -> tuple[str, dict, float]:
        from copilot.generated.rpc import PermissionDecisionReject
        from copilot.session_events import AssistantUsageData

        usage: list[AssistantUsageData] = []
        switches: list[str] = []

        def on_event(event) -> None:
            if isinstance(event.data, AssistantUsageData):
                usage.append(event.data)
            elif type(event.data).__name__ in MODEL_SWITCH_EVENTS:
                switches.append(type(event.data).__name__)

        session = await c.create_session(
            model=self.copilot_model,
            reasoning_effort=self.effort,
            system_message={"mode": "replace", "content": system},
            available_tools=[],
            disabled_mcp_servers=["github-mcp-server", "githubiq"],  # built-in MCP servers: never used, rate-limited
            # a one-turn session needs no session store, checkpoints, memory or file tracking (less work per call)
            enable_session_store=False,
            infinite_sessions={"enabled": False},
            memory={"enabled": False},
            enable_file_change_tracking=False,
            enable_skills=False,
            skip_custom_instructions=True,
            hooks={"on_user_prompt_transformed": lambda hook, ctx: {"modifiedTransformedPrompt": hook["prompt"]}},
            on_permission_request=lambda req, inv: PermissionDecisionReject(feedback="tools are disabled"),
            on_event=on_event,
        )
        try:
            sent = time.perf_counter()
            event = await session.send_and_wait(user, timeout=TIMEOUT_S)
            latency = (time.perf_counter() - sent) * 1000
        finally:
            await session.disconnect()
            await c.delete_session(session.session_id)  # sessions are kept on disk otherwise
        if event is None:
            raise RuntimeError("Copilot session ended without a reply")
        # the reply must come from the requested model alone: never Copilot's auto mode, a switch or a fusion of models
        served = sorted({u.model for u in usage if u.model})
        if served != [self.copilot_model] or switches or any(u.is_auto or u.fusion for u in usage):
            raise RuntimeError(f"reply served by {served or 'no reported model'}"
                               f"{' after ' + ', '.join(switches) if switches else ''}, not {self.copilot_model}")
        output = sum(u.output_tokens or 0 for u in usage)
        reasoning = sum(u.reasoning_tokens or 0 for u in usage)
        ttft = usage[0].time_to_first_token if usage else None
        return event.data.content or "", {
            "served_model": served[0],  # as the Copilot runtime reports it per model call; checked above
            "input_tokens": sum(u.input_tokens or 0 for u in usage),
            "output_tokens": output,  # everything generated, thinking included
            "reasoning_tokens": reasoning,  # hidden thinking, never part of the reply
            "answer_tokens": output - reasoning,  # the reply itself
            "cache_read_tokens": sum(u.cache_read_tokens or 0 for u in usage),
            "model_calls": len(usage),
            "aic": sum(u.copilot_usage.total_nano_aiu for u in usage if u.copilot_usage) / 1e9,
            # model-side timing as measured by the Copilot runtime; latency_ms adds ~1 s of Copilot's own overhead
            "api_duration_ms": sum(u.duration.total_seconds() * 1000 for u in usage if u.duration),
            "ttft_ms": ttft.total_seconds() * 1000 if ttft else None,  # not reported for every model (e.g. GPT)
        }, latency


if __name__ == "__main__":
    print(f"{'model':24s} {'AIC/M in':>9s} {'AIC/M out':>9s}  efforts")
    for m in list_models():
        prices = m.billing.token_prices if m.billing else None
        print(f"{m.id:24s} {getattr(prices, 'input_price', None) or '-':>9} {getattr(prices, 'output_price', None) or '-':>9}"
              f"  {','.join(m.supported_reasoning_efforts or [])}")
