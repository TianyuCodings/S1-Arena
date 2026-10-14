import math
import random

from s1arena import metrics
from s1arena.schema import DecisionRequest, Gold, Item, Question


def make_items(n=4):
    q = Question(type="choice", instructions="Pick.", criteria={"a": "A", "b": "B"})
    return [Item(leaf="basic_cls/test", task="basic_cls/test/toy", request=DecisionRequest(state={"i": i}, questions={"q": q}),
                 gold={"q": Gold(label="a" if i % 2 == 0 else "b")}) for i in range(n)]


def row(item, probs=None, sample=0, error=None):
    answers = {} if probs is None else {"q": {"probs": probs}}
    return {"item_id": item.id, "sample": sample, "response": {"answers": answers, "error": error, "latency_ms": 10.0, "cost_usd": 0.001}}


def test_perfect_answers():
    items = make_items()
    rows = [row(i, {i.gold["q"].label: 1.0}) for i in items]
    result = metrics.score(items, rows)
    assert result["accuracy"] == 1.0 and result["brier"] == 0.0 and result["ece"] == 0.0
    assert result["latency_p50_ms"] == 10.0 and math.isclose(result["cost_per_1k_usd"], 1.0)


def test_unanswered_question_scores_zero():
    items = make_items(2)  # gold a, then b
    rows = [row(items[0], None, error="HTTP 400: input too long"), row(items[1], {"a": 0.0, "b": 0.0})]
    result = metrics.score(items, rows)
    assert result["accuracy"] == 0.0  # wrong, even though a uniform guess could hit the gold label
    assert result["unanswered"] == 2 and result["unanswered_rate"] == 1.0
    assert math.isclose(result["brier"], 0.5)  # a uniform distribution over two options
    assert result["error_rate"] == 0.5


def test_majority_vote_over_samples():
    items = make_items(1)  # gold a
    rows = [row(items[0], {"a": 0.9, "b": 0.1}, 0), row(items[0], {"a": 0.4, "b": 0.6}, 1), row(items[0], {"a": 0.8, "b": 0.2}, 2)]
    result = metrics.score(items, rows)
    assert math.isclose(result["accuracy"], 2 / 3) and result["maj"] == 1.0


def test_ece_does_not_depend_on_the_order_of_ties():
    rng = random.Random(0)
    confidences = [1.0] * 30 + [0.7] * 20 + [round(rng.random(), 2) for _ in range(50)]
    correct = [float(rng.random() < 0.6) for _ in confidences]
    reference = metrics.expected_calibration_error(confidences, correct)
    for seed in range(5):
        order = list(range(len(confidences)))
        random.Random(seed).shuffle(order)
        shuffled = metrics.expected_calibration_error([confidences[i] for i in order], [correct[i] for i in order])
        assert math.isclose(shuffled, reference, abs_tol=1e-12)


def test_ece_of_calibrated_answers():
    assert metrics.expected_calibration_error([0.5] * 10, [1.0, 0.0] * 5) == 0.0
