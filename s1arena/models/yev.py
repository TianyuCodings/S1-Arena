"""yev decision models (Yev-Decision-1), run locally through the yev package.

yev is a System-One model: Qwen3.5-4B-Base with LoRA adapters and decision heads, asked through its
system_one(state, questions) call, which takes the request format of Jev. A boolean question is sent as yev's "noul"
type, choice and score questions as they are, and the state as the dict it is (yev writes a dict as JSON). Every
question is read with the state as its own sequence, and the answer is the softmax of the heads' logits divided by
the export's fitted temperature for the question type: "probabilities" for choice and score, "noul" = P(true) for
boolean. Nothing is sampled, so the sample index of a job is ignored.

Context: a question with its state is read 4,096 tokens at a time. A longer state is read as overlapping windows that
cover all of it, and their distributions are averaged (windows="mean"), so no input is cut or refused.

decide_many answers the requests that ask the same questions (most items of a task) together through
system_one_batch, `chunk` states per call, and the rest one at a time. Padding a batch moves probabilities by bf16
rounding noise.

Spec: yev:<Hugging Face id or export directory>[@key=value,...]; keys: rev (repository revision, part of the model
id), micro_batch (question sequences per forward pass, default 8), chunk (states per batched call, default 64).
Needs the yev package (pip install "yev[inference] @ git+https://huggingface.co/HallD/yev"), imported only when a
model is first used, so specs can be listed without it.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from ..registry import register_model
from ..schema import Answer, DecisionRequest, ModelResponse, Question
from .base import ModelAdapter
from .jev import to_jev
from .local_lm import REJECTED, parse_spec


def to_yev(question: Question) -> dict[str, Any]:
    return {**to_jev(question), "type": "noul" if question.type == "boolean" else question.type}


def from_yev(response: Any, name: str, question: Question) -> dict[str, float]:
    if question.type == "boolean":
        p = float(response.nouls[name].noul)
        return {"true": p, "false": 1.0 - p}
    answer = response.choices[name] if question.type == "choice" else response.scores[name]
    return {str(option): float(p) for option, p in answer.probabilities.items()}


@register_model
class Yev(ModelAdapter):
    kind = "yev"
    native_probs = True
    probability_source = "native (temperature-scaled softmax over the decision heads of a local model)"

    def __init__(self, arg: str = "HallD/Yev-Decision-1") -> None:
        self.hf_id, opts = parse_spec(arg)
        if set(opts) - {"rev", "micro_batch", "chunk"}:
            raise ValueError(f"unknown option in {arg!r}")
        self.rev = opts.get("rev")
        self.micro_batch = int(opts.get("micro_batch", 8))
        self.chunk = int(opts.get("chunk", 64))
        self._client = None
        self._lock = threading.Lock()  # closed-loop environments call decide() from several threads

    @property
    def model_id(self) -> str:
        return self.hf_id + (f"@rev={self.rev}" if self.rev else "")

    def _load(self) -> None:
        if self._client is None:
            from yev import YevClient

            self._client = YevClient(self.hf_id, revision=self.rev, micro_batch=self.micro_batch, lazy=False)

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        return self.decide_many([(request, sample)])[0]

    def decide_many(self, jobs: list[tuple[DecisionRequest, int]]) -> list[ModelResponse]:
        """Answer the requests in groups that ask the same questions; the sample number is not used."""
        with self._lock:
            self._load()
            started = time.perf_counter()
            groups: dict[str, list[int]] = {}
            for index, (request, _) in enumerate(jobs):
                key = json.dumps({name: q.model_dump() for name, q in request.questions.items()})
                groups.setdefault(key, []).append(index)
            responses: list[ModelResponse | None] = [None] * len(jobs)
            for indexes in groups.values():
                for start in range(0, len(indexes), self.chunk):
                    part = indexes[start:start + self.chunk]
                    for index, response in zip(part, self._answer([jobs[i][0] for i in part])):
                        responses[index] = response
            per_job = round((time.perf_counter() - started) * 1000 / max(1, len(jobs)), 1)
        for response in responses:  # as for lm: models, latency_ms stays empty: realtime environments charge the
            if response.usage is not None:  # latency of a single request, which a batch does not measure
                response.usage["batch_ms_per_request"] = per_job
        return responses

    def _answer(self, requests: list[DecisionRequest]) -> list[ModelResponse]:
        """Requests with the same questions in one system_one_batch call; one at a time if that call fails."""
        from yev import YevError

        questions = requests[0].questions
        try:
            out = self._client.system_one_batch([r.state for r in requests], {n: to_yev(q) for n, q in questions.items()},
                                                windows="mean")
        except Exception as error:  # noqa: BLE001 - recorded as the request's error, as the other adapters do
            if len(requests) > 1:  # find the request that fails; the others are still answered
                return [response for request in requests for response in self._answer([request])]
            final = REJECTED if isinstance(error, YevError) else ""  # a refused request is final, e.g. out of memory is not
            return [ModelResponse(error=f"{final}{type(error).__name__}: {str(error)[:300]}")]
        return [ModelResponse(answers={name: Answer(probs=from_yev(response, name, q)) for name, q in questions.items()},
                              usage={"input_tokens": response.usage.input_tokens}) for response in out]
