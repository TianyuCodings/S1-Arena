"""Leaderboard metrics, computed only from per-question probability distributions.

- accuracy (avg@k): mean over samples, then over questions, of argmax == gold
- maj@k: majority vote of the k argmax answers (ties broken by summed probability)
- Brier: squared distance between the predicted distribution and the target
  (one-hot for hard labels, the annotator distribution for soft labels)
- ECE: top-label calibration error with equal-mass bins, pooled over all answers
- se: standard error of the question-level accuracy (items are the unit of sampling)
- NDCG (ranking tasks, TaskAdapter.rank_key): items of one group ranked by the expected level of
  their score question, against the gold levels; mean over groups with at least two distinct levels
- pair accuracy (tasks with items_per_record > 1): share of pairs whose items are all answered correctly
- cost per million calls, for batch-style use
A question without an answer scores 0: the call failed, the reply could not be used, the reply leaves the question out,
or the model refused the input (for example as longer than its context). It counts as wrong for accuracy, maj@k and
pair accuracy; for Brier, ECE and NDCG it is a uniform distribution. Such questions are reported as unanswered, and
their calls as errors.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from .schema import Item, normalize


def expected_calibration_error(confidences: list[float], correct: list[float], bins: int = 10) -> float:
    """Top-option calibration error with equal-mass bins.

    Answers with the same confidence are interchangeable, so each takes the mean correctness of its group. The result
    then does not depend on the order of tied answers. Verbalized confidences tie often (many are exactly 1.0), and a
    sort that orders ties arbitrarily put them into the bins in an order that changed from one machine to another."""
    if not confidences:
        return float("nan")
    order = np.argsort(confidences, kind="stable")
    conf = np.asarray(confidences, dtype=float)[order]
    hit = np.asarray(correct, dtype=float)[order]
    _, start, counts = np.unique(conf, return_index=True, return_counts=True)  # groups of equal confidence
    hit = np.repeat(np.add.reduceat(hit, start) / counts, counts)
    total = len(conf)
    return float(sum(len(s) / total * abs(conf[s].mean() - hit[s].mean()) for s in np.array_split(np.arange(total), bins) if len(s)))


def score(items: list[Item], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_item: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_item[row["item_id"]].append(row)

    acc_q, maj_q, brier, conf, hit = [], [], [], [], []
    latencies, costs = [], []
    responses = errors = answers = unanswered = 0
    for item in items:
        runs = sorted(by_item.get(item.id, []), key=lambda r: r["sample"])
        if not runs:
            continue
        for run in runs:
            response = run["response"]
            responses += 1
            errors += bool(response.get("error"))
            if response.get("latency_ms") is not None:
                latencies.append(response["latency_ms"])
            if response.get("cost_usd") is not None:
                costs.append(response["cost_usd"])
        for name, gold in item.gold.items():
            options = item.request.questions[name].option_ids()
            target, truth = gold.target(options), gold.hard()
            sample_hits, votes, weight = [], defaultdict(int), defaultdict(float)
            for run in runs:
                probs, answered = _answer(run, name, options)
                pred = max(options, key=lambda o: probs[o]) if answered else None  # no answer: wrong, whatever the gold is
                answers += 1
                unanswered += not answered
                sample_hits.append(float(answered and pred == truth))
                brier.append(sum((probs[o] - target[o]) ** 2 for o in options))
                conf.append(probs[pred] if answered else 1 / len(options))
                hit.append(float(answered and pred == truth))
                votes[pred] += 1
                weight[pred] += probs[pred] if answered else 0.0
            acc_q.append(float(np.mean(sample_hits)))
            majority = max(votes, key=lambda o: (votes[o], weight[o]))
            maj_q.append(float(majority is not None and majority == truth))

    n = len(acc_q)
    if n == 0:
        return {"n_questions": 0}
    acc = float(np.mean(acc_q))
    return {
        "n_questions": n,
        "samples": responses / max(1, len({r["item_id"] for r in rows})),
        "accuracy": acc,
        "accuracy_se": float(np.std(acc_q, ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "maj": float(np.mean(maj_q)),
        "brier": float(np.mean(brier)),
        "ece": expected_calibration_error(conf, hit),
        "error_rate": errors / max(1, responses),
        "answers": answers,  # question x sample
        "unanswered": unanswered,  # of them, those without an answer; each scores 0
        "unanswered_rate": unanswered / max(1, answers),
        "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "cost_per_1k_usd": float(np.mean(costs) * 1000) if costs else None,
    }


def _answer(row: dict | None, name: str, options: list[str]) -> tuple[dict[str, float], bool]:
    """The distribution a response gives for a question, and whether it gives one. Without an answer (a failed or
    refused call, a question the reply leaves out, no probability on any option) the distribution is uniform."""
    answer = ((row or {}).get("response", {}).get("answers") or {}).get(name)
    probs = (answer or {}).get("probs") or {}
    answered = any(float(probs.get(o, 0.0) or 0.0) > 0 for o in options)
    return (normalize(probs, options) if answered else {o: 1 / len(options) for o in options}), answered


def _probs(row: dict | None, name: str, options: list[str]) -> dict[str, float]:
    return _answer(row, name, options)[0]


def ndcg(items: list[Item], rows: list[dict[str, Any]], rank_key: str) -> dict[str, Any]:
    """Mean NDCG over groups: items ranked by the model's expected level (averaged over samples) against gold levels."""
    by_item: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_item[row["item_id"]].append(row)
    where, key = rank_key.split(":", 1)
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for item in items:
        runs = by_item.get(item.id)
        score_questions = [n for n, q in item.request.questions.items() if q.type == "score" and n in item.gold]
        if not runs or not score_questions:
            continue
        name = score_questions[0]
        options = item.request.questions[name].option_ids()
        target = item.gold[name].target(options)
        gold_level = sum(i * target[o] for i, o in enumerate(options))
        predicted = float(np.mean([sum(i * _probs(r, name, options)[o] for i, o in enumerate(options)) for r in runs]))
        group = item.meta.get(key) if where == "meta" else item.request.state.get(key)
        groups[str(group)].append((predicted, gold_level))
    values = []
    for members in groups.values():
        if len(members) < 2 or len({g for _, g in members}) < 2:
            continue
        gain = lambda ranked: sum((2 ** g - 1) / math.log2(i + 2) for i, (_, g) in enumerate(ranked))
        ideal = gain(sorted(members, key=lambda m: -m[1]))
        values.append(gain(sorted(members, key=lambda m: -m[0])) / ideal if ideal > 0 else 0.0)
    return {"ndcg": float(np.mean(values)) if values else None, "ndcg_groups": len(values)}


def pair_accuracy(items: list[Item], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Share of pairs (items sharing meta["pair"]) whose questions are all answered correctly, per sample."""
    by_key = {(r["item_id"], r["sample"]): r for r in rows}
    samples = sorted({r["sample"] for r in rows})
    groups: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        if "pair" in item.meta:
            groups[item.meta["pair"]].append(item)
    hits = []
    for members in groups.values():
        if len(members) < 2:
            continue
        for s in samples:
            ok = True
            for item in members:
                row = by_key.get((item.id, s))
                if row is None:
                    ok = None
                    break
                for name, gold in item.gold.items():
                    options = item.request.questions[name].option_ids()
                    probs, answered = _answer(row, name, options)
                    ok = ok and answered and max(options, key=lambda o: probs[o]) == gold.hard()
            if ok is not None:
                hits.append(float(ok))
    return {"pair_accuracy": float(np.mean(hits)) if hits else None, "pairs": len({k for k, m in groups.items() if len(m) >= 2})}


def task_metrics(task, items: list[Item], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """score() plus the metrics a task declares: NDCG for ranking tasks, pair accuracy for paired items."""
    result = score(items, rows)
    if result.get("n_questions"):
        if getattr(task, "rank_key", None):
            result.update(ndcg(items, rows, task.rank_key))
        if getattr(task, "items_per_record", 1) > 1:
            result.update(pair_accuracy(items, rows))
        if result.get("cost_per_1k_usd") is not None:
            result["cost_per_1m_usd"] = result["cost_per_1k_usd"] * 1000
    return result
