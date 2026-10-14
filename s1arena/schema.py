"""The item format every task produces and every model consumes.

A benchmark item is a DecisionRequest (what the model sees) plus gold answers
and metadata (what the model never sees). Question types mirror the typed
decisions of System-One models:

- choice:  pick one option; criteria maps option id -> description
- boolean: probability that a statement is true; option ids are "true"/"false"
- score:   ordered levels; criteria is the ordered list, option ids are "0".."n-1"

Multi-label tasks use several boolean questions in one request.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "0.1"
MAX_OPTIONS = 255
MAX_LEVELS = 10  # score questions: Jev's API accepts at most 10 levels
BOOLEAN_OPTIONS = ("true", "false")

QuestionType = Literal["choice", "boolean", "score"]


class Question(BaseModel):
    type: QuestionType
    instructions: str
    criteria: dict[str, str] | list[str] | None = None

    @model_validator(mode="after")
    def _check(self) -> "Question":
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or not 2 <= len(self.criteria) <= MAX_OPTIONS:
                raise ValueError(f"choice questions need 2..{MAX_OPTIONS} options given as a dict")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= MAX_LEVELS:
                raise ValueError(f"score questions need an ordered list of 2..{MAX_LEVELS} levels")
        elif self.criteria is not None:
            raise ValueError("boolean questions take no criteria; put clarifications in the instructions")
        return self

    def option_ids(self) -> list[str]:
        if self.type == "choice":
            return list(self.criteria)
        if self.type == "score":
            return [str(i) for i in range(len(self.criteria))]
        return list(BOOLEAN_OPTIONS)


class DecisionRequest(BaseModel):
    """Everything the model is allowed to see."""

    state: dict[str, Any]
    questions: dict[str, Question]


class Gold(BaseModel):
    """A hard label, a soft label distribution, or both."""

    label: str | None = None
    dist: dict[str, float] | None = None

    @model_validator(mode="after")
    def _check(self) -> "Gold":
        if self.label is None and self.dist is None:
            raise ValueError("gold needs a label or a distribution")
        if self.dist is not None:
            total = sum(self.dist.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"gold distribution sums to {total}, not 1")
        return self

    def hard(self) -> str:
        if self.label is not None:
            return self.label
        return max(self.dist, key=self.dist.get)

    def target(self, options: list[str]) -> dict[str, float]:
        """Target distribution for proper scoring (Brier): soft if available, else one-hot."""
        if self.dist is not None:
            return {o: float(self.dist.get(o, 0.0)) for o in options}
        return {o: float(o == self.label) for o in options}


class Item(BaseModel):
    schema_version: str = SCHEMA_VERSION
    id: str = ""
    leaf: str
    task: str
    request: DecisionRequest
    gold: dict[str, Gold]
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "Item":
        for name, gold in self.gold.items():
            if name not in self.request.questions:
                raise ValueError(f"gold for unknown question {name!r}")
            options = self.request.questions[name].option_ids()
            if gold.label is not None and gold.label not in options:
                raise ValueError(f"gold label {gold.label!r} is not an option of {name!r}")
            if gold.dist is not None and not set(gold.dist) <= set(options):
                raise ValueError(f"gold distribution of {name!r} has unknown options")
        if not self.id:
            self.id = content_id(self)
        return self


def content_id(item: Item) -> str:
    """Stable id derived from content, so any change to an item changes its id."""
    payload = json.dumps(
        {
            "task": item.task,
            "request": item.request.model_dump(),
            "gold": {k: v.model_dump() for k, v in item.gold.items()},
        },
        ensure_ascii=False,
    )  # key order is kept on purpose: presenting options in another order is another item
    return item.task.replace("/", ".") + ":" + hashlib.sha256(payload.encode()).hexdigest()[:12]


class Answer(BaseModel):
    """A model's answer to one question: a distribution over that question's option ids."""

    probs: dict[str, float]


class ModelResponse(BaseModel):
    answers: dict[str, Answer] = Field(default_factory=dict)
    latency_ms: float | None = None
    usage: dict[str, Any] | None = None
    cost_usd: float | None = None
    error: str | None = None
    raw_text: str | None = None  # the model's reply as written, kept for auditing (LLMs)


def normalize(probs: dict[str, float], options: list[str]) -> dict[str, float]:
    """Restrict to known options, clip negatives, renormalize; uniform if nothing usable."""
    clean = {o: max(0.0, float(probs.get(o, 0.0) or 0.0)) for o in options}
    total = sum(clean.values())
    if total <= 0:
        return {o: 1.0 / len(options) for o in options}
    return {o: p / total for o, p in clean.items()}
