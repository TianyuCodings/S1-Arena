"""Base class for model adapters.

To add a model family, subclass ModelAdapter in a new module under
s1arena/models/, set `kind`, implement decide(), and decorate the class with
@register_model. decide() receives exactly what the model may see (a
DecisionRequest) and returns a probability distribution per question over that
question's option ids. Metrics only ever read those distributions, so every
model is scored the same way.
"""

from __future__ import annotations

from ..schema import DecisionRequest, ModelResponse


class ModelAdapter:
    kind: str = ""  # registry key used in model specs, e.g. "jev" or "llm"
    native_probs: bool = False  # True if the model returns probabilities itself
    probability_source: str = ""  # how probabilities are obtained, shown on the leaderboard

    @property
    def model_id(self) -> str:
        raise NotImplementedError

    @property
    def slug(self) -> str:
        return self.model_id.replace("/", "__").replace(":", "_")

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        raise NotImplementedError

    def reparse(self, response: ModelResponse, request: DecisionRequest) -> ModelResponse | None:
        """Re-read a stored response with the current parser (models that answer in free text).

        Returns None when the model has nothing to re-read, e.g. structured APIs such as Jev.
        """
        return None
