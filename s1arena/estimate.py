"""Cost estimates before running a model, so LLM runs can be budgeted.

Two estimates per task:
- price-based: input tokens (reference tokenizer, on what the model is actually sent) times the
  gateway list price, plus a rough allowance for the JSON answer;
- measured: mean cost per call observed in cached runs of the same model (e.g. a smoke test),
  times the number of calls. Prefer this one when it exists.
"""

from __future__ import annotations

import json

from . import gateway, render, runner
from .schema import DecisionRequest

_PRICES: dict[str, tuple[float, float]] | None = None


def prices(model_id: str) -> tuple[float, float] | None:
    """(input, output) USD per token from the gateway's model list."""
    global _PRICES
    if _PRICES is None:
        data = gateway.client().get(gateway.BASE_URL + "/v1/models").json()["data"]
        _PRICES = {}
        for m in data:
            p = m.get("pricing") or {}
            if "input" in p:
                _PRICES[m["id"]] = (float(p["input"]), float(p.get("output") or 0))
    return _PRICES.get(model_id)


def _tokens(text: str) -> int:
    from .tasks.base import approx_tokens  # shares the reference tokenizer

    return approx_tokens(DecisionRequest(state={"t": text}, questions={}))


def call_tokens(model, request: DecisionRequest) -> tuple[int, int]:
    """(input, output) tokens for one call."""
    if model.kind == "jev":
        return _tokens(request.model_dump_json()), 0
    prompt = "\n".join(m["content"] for m in render.messages(request))
    answer = sum(8 + 6 * len(q.option_ids()) for q in request.questions.values())
    return _tokens(prompt), answer


def estimate_task(model, task: str, items, k: int) -> dict:
    tin = tout = 0
    for item in items:
        a, b = call_tokens(model, item.request)
        tin, tout = tin + a, tout + b
    calls = len(items) * k
    row = {"task": task, "items": len(items), "calls": calls, "input_tokens": tin * k, "output_tokens": tout * k}
    price = model.token_prices() if hasattr(model, "token_prices") else prices(model.model_id)  # copilot: own list
    row["price_based_usd"] = (tin * price[0] + tout * price[1]) * k if price else None
    observed = [r["response"]["cost_usd"] for r in runner.read_runs(model.slug, task) if r["response"].get("cost_usd") is not None]
    row["measured_usd"] = sum(observed) / len(observed) * calls if observed else None
    row["measured_from_calls"] = len(observed)
    return row


def to_json(rows: list[dict]) -> str:
    return json.dumps(rows, indent=1)
