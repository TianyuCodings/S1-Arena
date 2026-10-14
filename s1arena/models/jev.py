"""TypeSafe's Jev through the AI Gateway evaluation-model endpoint."""

from __future__ import annotations

import time
from typing import Any

from .. import gateway
from ..registry import register_model
from ..schema import Answer, DecisionRequest, ModelResponse, Question, normalize
from .base import ModelAdapter


def to_jev(question: Question) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": question.type, "instructions": question.instructions}
    if question.criteria is not None:
        payload["criteria"] = question.criteria
    return payload


def from_jev(answer: dict[str, Any] | None, question: Question) -> dict[str, float] | None:
    options = question.option_ids()
    if not answer:
        return None
    if question.type == "boolean":
        p = float(answer.get("probability", 0.5))
        return {"true": p, "false": 1.0 - p}
    probs = answer.get("probabilities")
    if probs:
        return normalize({str(k): v for k, v in probs.items()}, options)
    if question.type == "choice" and answer.get("choice") in options:
        return {o: float(o == answer["choice"]) for o in options}
    if question.type == "score" and "score" in answer:
        level = str(min(len(options) - 1, max(0, round(float(answer["score"])))))
        return {o: float(o == level) for o in options}
    return None


@register_model
class Jev(ModelAdapter):
    kind = "jev"
    native_probs = True
    probability_source = "native"

    def __init__(self, model_id: str = "typesafe-ai/jev") -> None:
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        body = {"state": request.state, "questions": {n: to_jev(q) for n, q in request.questions.items()}}
        headers = {"ai-evaluation-model-specification-version": "4", "ai-model-id": self._model_id}
        start = time.perf_counter()
        try:
            out = gateway.post("/v4/ai/evaluation-model", body, headers=headers)
        except Exception as error:  # recorded, scored as a failed response
            return ModelResponse(error=f"{type(error).__name__}: {str(error)[:300]}", latency_ms=(time.perf_counter() - start) * 1000)
        latency = (time.perf_counter() - start) * 1000
        answers, missing = {}, []
        for name, question in request.questions.items():
            probs = from_jev(out.get("answers", {}).get(name), question)
            if probs is None:
                missing.append(name)
            else:
                answers[name] = Answer(probs=probs)
        meta = out.get("providerMetadata") or {}
        cost = (meta.get("gateway") or {}).get("marketCost")
        return ModelResponse(
            answers=answers,
            latency_ms=latency,
            usage=out.get("usage"),
            cost_usd=float(cost) if cost not in (None, "") else None,
            error=f"missing answers: {missing}" if missing else None,
        )
