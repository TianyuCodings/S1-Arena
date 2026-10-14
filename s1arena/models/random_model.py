"""Random distributions. Only for testing the pipeline; never reported on the leaderboard."""

from __future__ import annotations

import hashlib
import random

from ..registry import register_model
from ..schema import Answer, DecisionRequest, ModelResponse
from .base import ModelAdapter


@register_model
class RandomModel(ModelAdapter):
    kind = "random"
    native_probs = True
    probability_source = "random"

    @property
    def model_id(self) -> str:
        return "test/random"

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        seed = hashlib.sha256((request.model_dump_json() + str(sample)).encode()).hexdigest()
        rng = random.Random(seed)
        answers = {}
        for name, question in request.questions.items():
            weights = [rng.random() for _ in question.option_ids()]
            total = sum(weights)
            answers[name] = Answer(probs={o: w / total for o, w in zip(question.option_ids(), weights)})
        return ModelResponse(answers=answers, latency_ms=0.0, cost_usd=0.0)
