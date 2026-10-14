"""OpenRouter's Decisions API (POST /api/alpha/decisions), with native probabilities.

Spec: ordecisions[:<model>], default microsoft/microsoft-decision-1 (Microsoft-Decision-1, served by Azure; the reply
names the dated version, kept in usage["served_model"]).

The API takes Jev's request format: a state and named typed questions. Boolean is sent as "noul" without criteria (the
API accepts that, as Jev's gateway does), choice with criteria {option id: description}, score with the ordered list
of level descriptions (answers keyed "0".."n-1"). The state goes as given (object), as for Jev. A boolean answer is
{noul: P(true)}; choice and score answers carry {probabilities}, which become the distribution over the option ids.

Authentication: OPENROUTER_API_KEY in .env. Rate limits (429), server errors and transport errors are retried with
backoff (8 retries); other HTTP errors come back as "decisions HTTP <status>", so 400 (e.g. an input longer than the
32,768-token context) counts as a rejected request. cost_usd is the cost OpenRouter reports in usage.cost.
"""

from __future__ import annotations

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

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}

_client: httpx.Client | None = None
_lock = threading.Lock()


def client() -> httpx.Client:
    global _client
    with _lock:
        if _client is None:
            load_env()
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY is not set; add it to .env")
            _client = httpx.Client(
                base_url="https://openrouter.ai/api/alpha",
                timeout=httpx.Timeout(300.0, connect=20.0),
                limits=httpx.Limits(max_connections=256, max_keepalive_connections=128),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
        return _client


def to_openrouter(question: Question) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "noul" if question.type == "boolean" else question.type,
                               "instructions": question.instructions}
    if question.criteria is not None:
        payload["criteria"] = question.criteria
    return payload


def from_openrouter(answer: dict[str, Any] | None, question: Question) -> dict[str, float] | None:
    if not answer:
        return None
    if question.type == "boolean":
        if answer.get("noul") is None:
            return None
        p = float(answer["noul"])
        return {"true": p, "false": 1.0 - p}
    probs = answer.get("probabilities")
    if not probs:
        return None
    return normalize({str(k): float(v) for k, v in probs.items()}, question.option_ids())


@register_model
class OpenRouterDecisions(ModelAdapter):
    kind = "ordecisions"
    native_probs = True
    probability_source = "native (OpenRouter Decisions API)"

    def __init__(self, model: str = "microsoft/microsoft-decision-1", retries: int = 8) -> None:
        self.model, self.retries = model, retries

    @property
    def model_id(self) -> str:
        return f"openrouter-decisions/{self.model}"

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        body = {"model": self.model, "state": request.state,
                "questions": {name: to_openrouter(q) for name, q in request.questions.items()}}
        start = time.perf_counter()
        try:
            out, attempts, latency = self._post(body)
        except Exception as error:  # recorded, scored as a failed response
            return ModelResponse(error=f"{type(error).__name__}: {str(error)[:300]}", latency_ms=(time.perf_counter() - start) * 1000)
        answers, missing = {}, []
        for name, question in request.questions.items():
            probs = from_openrouter((out.get("answers") or {}).get(name), question)
            if probs is None:
                missing.append(name)
            else:
                answers[name] = Answer(probs=probs)
        usage = {**(out.get("usage") or {}), "attempts": attempts, "served_model": out.get("model"),
                 "provider": out.get("provider")}
        cost = usage.get("cost")
        return ModelResponse(answers=answers, latency_ms=latency, usage=usage,
                             cost_usd=float(cost) if cost is not None else None,
                             error=f"missing answers: {missing}" if missing else None)

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
