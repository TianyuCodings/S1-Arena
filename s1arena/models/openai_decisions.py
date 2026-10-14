"""OpenAI's Decisions API (POST /v1/decisions, public beta), with native probabilities.

Spec: decisions[:<model>], default gpt-6-luna (the only model the API serves). Bedrock's OpenAI-compatible endpoint has
the route too, but refuses openai.gpt-6-luna ("not supported in the Decisions API", checked 2026-10-07), so this adapter
calls api.openai.com.

The API takes typed questions like Jev: boolean is sent as "predicate" (answer: P(true)), choice as "choice" with
{value, description} per option, score as "score" with {label, description} per level, where the label is the level id
"0".."n-1" as in the LLM prompt (render.messages). The state goes as text input, the JSON that LLMs see under "Input:".
A choice or score answer comes back as {value, probability} pairs, which become the distribution over the option ids;
an answer of type "refusal" is a final missing answer (error "rejected: refused ...").

Authentication: OPENAI_API_KEY in .env. Rate limits (429), server errors and transport errors are retried with backoff
(8 retries); other HTTP errors come back as "decisions HTTP <status>", so 400 counts as a rejected request. The API
returns no cost: cost_usd is the list price, $0.10 per million input tokens (output is free; cached input is billed at
the full price here, an upper bound).
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from typing import Any

import httpx

from ..env import load_env
from ..registry import register_model
from ..schema import Answer, DecisionRequest, ModelResponse, Question, normalize
from .base import ModelAdapter
from .local_lm import REJECTED

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
PRICE_INPUT = 0.10  # USD per million input tokens

_client: httpx.Client | None = None
_lock = threading.Lock()


def client() -> httpx.Client:
    global _client
    with _lock:
        if _client is None:
            load_env()
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is not set; add it to .env")
            _client = httpx.Client(
                base_url="https://api.openai.com/v1",
                timeout=httpx.Timeout(300.0, connect=20.0),
                limits=httpx.Limits(max_connections=256, max_keepalive_connections=128),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
        return _client


def to_decisions(name: str, question: Question) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "predicate" if question.type == "boolean" else question.type,
                               "name": name, "instructions": question.instructions}
    if question.type == "choice":
        payload["choices"] = [{"value": o, "description": d} for o, d in question.criteria.items()]
    elif question.type == "score":
        payload["levels"] = [{"label": str(i), "description": d} for i, d in enumerate(question.criteria)]
    return payload


def from_decisions(answer: dict[str, Any], question: Question) -> dict[str, float]:
    if question.type == "boolean":
        p = float(answer["probability"])
        return {"true": p, "false": 1.0 - p}
    probs = {str(p["value"]): float(p["probability"]) for p in answer["probabilities"]}
    return normalize(probs, question.option_ids())


@register_model
class OpenAIDecisions(ModelAdapter):
    kind = "decisions"
    native_probs = True
    probability_source = "native (OpenAI Decisions API)"

    def __init__(self, model: str = "gpt-6-luna", retries: int = 8) -> None:
        self.model, self.retries = model, retries

    @property
    def model_id(self) -> str:
        return f"openai-decisions/{self.model}"

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        body = {"model": self.model, "input": json.dumps(request.state, ensure_ascii=False, indent=1),
                "questions": [to_decisions(name, q) for name, q in request.questions.items()]}
        start = time.perf_counter()
        try:
            out, attempts, latency = self._post(body)
        except Exception as error:  # recorded, scored as a failed response
            return ModelResponse(error=f"{type(error).__name__}: {str(error)[:300]}", latency_ms=(time.perf_counter() - start) * 1000)
        by_name = {a.get("name"): a for a in out.get("answers", [])}
        answers, missing, refused = {}, [], []
        for name, question in request.questions.items():
            answer = by_name.get(name)
            if answer is None:
                missing.append(name)
            elif answer.get("type") == "refusal":
                refused.append(name)
            else:
                answers[name] = Answer(probs=from_decisions(answer, question))
        usage = {**(out.get("usage") or {}), "attempts": attempts}
        error = (f"missing answers: {missing}" if missing
                 else REJECTED + f"refused: {refused}" if refused else None)
        return ModelResponse(answers=answers, latency_ms=latency, usage=usage,
                             cost_usd=usage.get("input_tokens", 0) * PRICE_INPUT / 1e6, error=error)

    def _post(self, body: dict) -> tuple[dict, int, float]:
        """One call, retried with backoff and jitter on rate limits, server and transport errors.

        Returns (reply, attempts, latency_ms of the successful attempt, without retry waits)."""
        for attempt in range(self.retries + 1):
            sent = time.perf_counter()
            try:
                r = client().post("/decisions", json=body)
            except httpx.TransportError as error:
                if attempt == self.retries:
                    raise RuntimeError(f"decisions unreachable: {error}") from error
            else:
                if r.status_code == 200:
                    return r.json(), attempt + 1, (time.perf_counter() - sent) * 1000
                if r.status_code not in RETRY_STATUS or attempt == self.retries:
                    raise RuntimeError(f"decisions HTTP {r.status_code}: {r.text[:300]}")
            time.sleep(min(30.0, 2**attempt) * (0.5 + random.random()))
        raise AssertionError("unreachable")
