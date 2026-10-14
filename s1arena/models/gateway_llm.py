"""Any chat LLM on the AI Gateway (OpenAI-compatible endpoint), with verbalized probabilities."""

from __future__ import annotations

import time

from .. import gateway, render
from ..registry import register_model
from ..schema import Answer, DecisionRequest, ModelResponse, normalize
from .base import ModelAdapter


@register_model
class GatewayLLM(ModelAdapter):
    kind = "llm"
    native_probs = False
    probability_source = "verbalized"

    def __init__(self, model_id: str, temperature: float | None = 1.0, max_tokens: int = 4000) -> None:
        self._model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def model_id(self) -> str:
        return self._model_id

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        body = {"model": self._model_id, "messages": render.messages(request), "max_tokens": self.max_tokens}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        start = time.perf_counter()
        try:
            out = gateway.post("/v1/chat/completions", body)
            text = out["choices"][0]["message"].get("content") or ""
        except Exception as error:
            latency = (time.perf_counter() - start) * 1000
            return ModelResponse(error=f"{type(error).__name__}: {str(error)[:300]}", latency_ms=latency)
        latency = (time.perf_counter() - start) * 1000
        usage = out.get("usage") or {}
        cost = usage.get("cost") if isinstance(usage, dict) else None
        response = ModelResponse(
            latency_ms=latency,
            usage=usage,
            cost_usd=float(cost) if isinstance(cost, (int, float, str)) and str(cost) else None,
            raw_text=text,
        )
        return self.reparse(response, request)

    def reparse(self, response: ModelResponse, request: DecisionRequest) -> ModelResponse:
        """Fill answers from the stored reply text; an unparsable reply or a missing question is an error."""
        try:
            parsed = render.parse(response.raw_text or "", request)
        except Exception as error:
            return response.model_copy(update={"answers": {}, "error": f"{type(error).__name__}: {str(error)[:300]}"})
        answers, missing = {}, []
        for name, question in request.questions.items():
            probs = parsed.get(name)
            if probs:
                answers[name] = Answer(probs=normalize(probs, question.option_ids()))
            else:
                missing.append(name)
        return response.model_copy(update={"answers": answers, "error": f"missing answers: {missing}" if missing else None})
