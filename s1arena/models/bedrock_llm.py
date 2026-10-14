"""OpenAI models served by Amazon Bedrock (Converse API, Bedrock API key), with verbalized probabilities.

Spec: bedrock:<model or inference profile id>[@effort=<reasoning effort>], e.g. bedrock:us.openai.gpt-6-luna@effort=none
(GPT-6 Luna and GPT-5.6 Luna are served through cross-region inference profiles only: us.* or global.*).

The model receives the same two messages as llm: models (render.messages: the system prompt as Converse's system block,
the user turn as one text block) and its reply is parsed like llm: replies; maxTokens is 4000 as for llm:. These models
take no temperature, so they sample at their default; effort is sent as reasoning.effort. So bedrock: runs are stored
and labeled separately (bedrock/<id>) from llm: and copilot: runs of the same model.

Authentication: AWS_API_KEY in .env, a Bedrock API key, sent as a bearer token to bedrock-runtime in BEDROCK_REGION
(default us-east-1). Throttling (HTTP 429), server errors and OpenAI's input moderation flags (HTTP 400 "flagged as
potentially violating our usage policy", which hits some game and agent states at random: the same request passes on
another try) are retried with backoff (8 retries); other HTTP errors come back as "bedrock HTTP <status>", so 401-404
count as account refusals in a sweep and 400 as a rejected request. Bedrock
returns no cost: cost_usd is computed from PRICES, OpenAI's list prices for these models (Bedrock's GovCloud price list
has GPT-5.6 Luna at 1.32x them, with the same cache ratios; commercial regions are not listed yet).
"""

from __future__ import annotations

import os
import random
import threading
import time

import httpx

from .. import render
from ..env import load_env
from ..registry import register_model
from ..schema import DecisionRequest, ModelResponse
from .base import ModelAdapter
from .gateway_llm import GatewayLLM

RETRY_STATUS = {408, 429, 500, 502, 503, 504}
# USD per million tokens: input, output, cache read, cache write (keyed by the model id inside the profile id). Bedrock
# caches prompts by itself: a new prompt is billed as a cache write (1.25x input), a repeated prefix as a cache read (0.1x).
PRICES = {
    "openai.gpt-6-luna": (0.10, 0.50, 0.01, 0.125),
    "openai.gpt-5.6-luna": (0.20, 1.20, 0.02, 0.25),
}

_client: httpx.Client | None = None
_lock = threading.Lock()


def client() -> httpx.Client:
    global _client
    with _lock:
        if _client is None:
            load_env()
            key = os.environ.get("AWS_API_KEY")
            if not key:
                raise RuntimeError("AWS_API_KEY (a Bedrock API key) is not set; add it to .env")
            _client = httpx.Client(
                base_url=f"https://bedrock-runtime.{os.environ.get('BEDROCK_REGION', 'us-east-1')}.amazonaws.com",
                timeout=httpx.Timeout(300.0, connect=20.0),
                limits=httpx.Limits(max_connections=256, max_keepalive_connections=128),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
        return _client


@register_model
class BedrockLLM(ModelAdapter):
    kind = "bedrock"
    native_probs = False
    probability_source = "verbalized"
    reparse = GatewayLLM.reparse  # same reply format, same parser

    def __init__(self, spec: str, max_tokens: int = 4000, retries: int = 8) -> None:
        self.bedrock_model, _, opts = spec.partition("@")
        options = dict(kv.split("=", 1) for kv in opts.split(",") if kv)
        unknown = set(options) - {"effort"}
        if unknown:
            raise ValueError(f"unknown bedrock option(s) {sorted(unknown)}; known: effort")
        self.effort = options.get("effort")
        self.max_tokens, self.retries = max_tokens, retries

    @property
    def model_id(self) -> str:
        return f"bedrock/{self.bedrock_model}" + (f"@effort={self.effort}" if self.effort else "")

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        system, user = (m["content"] for m in render.messages(request))
        start = time.perf_counter()
        try:
            text, usage, latency = self.chat(system, user)
        except Exception as error:
            return ModelResponse(error=f"{type(error).__name__}: {str(error)[:300]}", latency_ms=(time.perf_counter() - start) * 1000)
        response = ModelResponse(latency_ms=latency, usage=usage, cost_usd=self._cost(usage), raw_text=text)
        return self.reparse(response, request)

    def chat(self, system: str, user: str) -> tuple[str, dict, float]:
        """One reply to a system prompt and a user turn: (text, usage, latency_ms); raises after the retries."""
        body = {"system": [{"text": system}], "messages": [{"role": "user", "content": [{"text": user}]}],
                "inferenceConfig": {"maxTokens": self.max_tokens}}
        if self.effort:
            body["additionalModelRequestFields"] = {"reasoning": {"effort": self.effort}}
        out, attempts, latency = self._converse(body)
        content = out["output"]["message"]["content"]
        u = out.get("usage") or {}
        usage = {
            "input_tokens": u.get("inputTokens", 0),
            "output_tokens": u.get("outputTokens", 0),  # reasoning included, if any
            "cache_read_tokens": u.get("cacheReadInputTokens", 0),
            "cache_write_tokens": u.get("cacheWriteInputTokens", 0),
            "reasoning_blocks": sum(1 for c in content if "reasoningContent" in c),
            "stop_reason": out.get("stopReason"),
            "server_latency_ms": (out.get("metrics") or {}).get("latencyMs"),
            "attempts": attempts,
        }
        return "".join(c["text"] for c in content if "text" in c), usage, latency

    def _cost(self, usage: dict) -> float | None:
        prefix, _, rest = self.bedrock_model.partition(".")
        prices = PRICES.get(rest if prefix in ("us", "eu", "apac", "global") else self.bedrock_model)  # profile -> model
        if not prices:
            return None
        p_in, p_out, p_read, p_write = prices
        return (usage["input_tokens"] * p_in + usage["output_tokens"] * p_out + usage["cache_read_tokens"] * p_read
                + usage["cache_write_tokens"] * p_write) / 1e6

    def _converse(self, body: dict) -> tuple[dict, int, float]:
        """One Converse call, retried with backoff and jitter on throttling, server and transport errors.

        Returns (reply, attempts, latency_ms of the successful attempt, without retry waits)."""
        for attempt in range(self.retries + 1):
            sent = time.perf_counter()
            try:
                r = client().post(f"/model/{self.bedrock_model}/converse", json=body)
            except httpx.TransportError as error:
                if attempt == self.retries:
                    raise RuntimeError(f"bedrock unreachable: {error}") from error
            else:
                if r.status_code == 200:
                    return r.json(), attempt + 1, (time.perf_counter() - sent) * 1000
                flagged = r.status_code == 400 and "flagged as potentially violating" in r.text  # input moderation, random
                if (r.status_code not in RETRY_STATUS and not flagged) or attempt == self.retries:
                    raise RuntimeError(f"bedrock HTTP {r.status_code}: {r.text[:300]}")
            time.sleep(min(30.0, 2**attempt) * (0.5 + random.random()))
        raise AssertionError("unreachable")
