"""Template: a model adapter.

Copy this file to s1arena/models/<your_module>.py, rename the class and the `kind`, and replace decide(). Discovery
imports every module under s1arena/models/. Then check it on three toy requests and run it:

    uv run s1arena check-model example
    uv run s1arena run --model example --leaf basic_cls/intent --limit 20
    uv run s1arena score --models example --leaf basic_cls/intent --limit 20

A model receives exactly what it may see (a DecisionRequest) and returns one probability distribution per question over
that question's option ids. Metrics read only these distributions, so every model is scored the same way. Leave a
question out (or set `error`) when the model cannot answer it: the question then scores 0.
"""

from __future__ import annotations

import time

from s1arena.models.base import ModelAdapter
from s1arena.registry import register_model
from s1arena.schema import Answer, DecisionRequest, ModelResponse


@register_model
class ExampleModel(ModelAdapter):
    kind = "example"  # the spec prefix: `example` or `example:<argument>`
    native_probs = True  # True if the model returns probabilities itself (False for verbalized ones)
    probability_source = "uniform baseline"  # how probabilities are obtained; shown on the leaderboard

    def __init__(self, name: str = "uniform") -> None:  # `example:<name>` passes <name> here
        self.name = name

    @property
    def model_id(self) -> str:
        """The id stored with every response and shown on the leaderboard; runs are cached under it."""
        return f"example/{self.name}"

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        started = time.perf_counter()
        # Replace this with a call to your model. request.state is the input; for each question, question.type is
        # "choice", "boolean" or "score", question.instructions says what to decide, and question.option_ids() are
        # the ids to return probabilities for (question.criteria holds their descriptions).
        answers = {}
        for name, question in request.questions.items():
            options = question.option_ids()
            answers[name] = Answer(probs={o: 1 / len(options) for o in options})
        latency = (time.perf_counter() - started) * 1000
        return ModelResponse(answers=answers, latency_ms=latency, cost_usd=0.0)
