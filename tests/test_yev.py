"""The yev adapter against a stand-in for the yev package: no weights are downloaded."""

import sys
import types
from types import SimpleNamespace

from s1arena.models.yev import Yev
from s1arena.schema import DecisionRequest

CALLS = []


class YevError(Exception):
    pass


class FakeClient:
    """Answers as yev does: P(true) 0.8 for a noul, the first option for a choice, the top level for a score."""

    def __init__(self, model, revision=None, micro_batch=8, lazy=True):
        self.model = model

    def system_one_batch(self, states, questions, windows=None):
        CALLS.append(len(states))
        if any(state.get("text") == "refused" for state in states):
            raise YevError("State is required.")
        if any(state.get("text") == "crash" for state in states):
            raise RuntimeError("CUDA out of memory")
        answers = {"nouls": {}, "choices": {}, "scores": {}}
        for name, question in questions.items():
            if question["type"] == "noul":
                answers["nouls"][name] = SimpleNamespace(noul=0.8)
            elif question["type"] == "choice":
                options = list(question["criteria"])
                answers["choices"][name] = SimpleNamespace(probabilities={o: float(o == options[0]) for o in options})
            else:
                levels = range(len(question["criteria"]))
                answers["scores"][name] = SimpleNamespace(probabilities={level: float(level == levels[-1]) for level in levels})
        return [SimpleNamespace(**answers, usage=SimpleNamespace(input_tokens=12)) for _ in states]


sys.modules["yev"] = types.SimpleNamespace(YevClient=FakeClient, YevError=YevError)
QUESTIONS = {
    "spam": {"type": "boolean", "instructions": "Is this spam?"},
    "topic": {"type": "choice", "instructions": "Topic?", "criteria": {"sport": "Sport", "news": "News"}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]},
}


def request(text, questions=QUESTIONS):
    return DecisionRequest(state={"text": text}, questions=questions)


def test_answers_are_yevs_distributions_over_the_option_ids():
    response = Yev("HallD/Yev-Decision-1@rev=abc").decide(request("Win a prize"))
    assert response.answers["spam"].probs == {"true": 0.8, "false": 0.19999999999999996}
    assert response.answers["topic"].probs == {"sport": 1.0, "news": 0.0}
    assert response.answers["urgency"].probs == {"0": 0.0, "1": 0.0, "2": 1.0}
    assert response.usage["input_tokens"] == 12 and "batch_ms_per_request" in response.usage
    assert response.error is None


def test_requests_with_the_same_questions_share_a_call_and_a_failure_stays_with_its_request():
    model = Yev("HallD/Yev-Decision-1@chunk=2")
    CALLS.clear()
    other = {"spam": QUESTIONS["spam"]}
    responses = model.decide_many([(request("a"), 0), (request("b"), 0), (request("c"), 0), (request("d", other), 0)])
    assert CALLS == [2, 1, 1] and all(r.error is None for r in responses)
    CALLS.clear()
    refused, ok, crashed = model.decide_many([(request("refused"), 0), (request("fine"), 0), (request("crash"), 0)])
    assert CALLS == [2, 1, 1, 1]
    assert refused.error.startswith("rejected: YevError") and not refused.answers
    assert ok.error is None and ok.answers["spam"].probs["true"] == 0.8
    assert crashed.error.startswith("RuntimeError")   # not final: retried on the next run


def test_the_revision_is_part_of_the_model_id():
    assert Yev("HallD/Yev-Decision-1@rev=abc,micro_batch=16").model_id == "HallD/Yev-Decision-1@rev=abc"
    assert Yev().model_id == "HallD/Yev-Decision-1"
