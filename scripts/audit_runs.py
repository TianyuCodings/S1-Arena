"""Audit cached runs before summarizing them: completeness, failures and the settings each adapter logs.

For every model and every task under the given leaves, against the built items:
- every item has a final answer for every sample below k (no missing item, no call left to retry);
- the answers cover every question, each with a distribution over that question's options that sums to 1;
- failures by kind (prompt longer than the context, options that do not fit, content filter, unusable reply, ...),
  and the questions they leave to be scored as uniform distributions;
- what the adapters log: thinking tokens (none allowed at think=0, effort "none"), the probability mass on the allowed
  labels (lm:, oss:, dgemma:), the model and effort that served each Copilot reply, its metered AIU and tokens, and
  states cut to fit an encoder (laya:).
Item ids are content hashes (s1arena/schema.py), so an answer cached under a current id was given to the current item.

Usage:
  python scripts/audit_runs.py --models lm:Qwen/Qwen3-8B copilot:gpt-6-sol@effort=none --leaf basic_cls know \
      [--k 1] [--limit 0] [--out audit.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s1arena import dataio, runner  # noqa: E402
from s1arena.env import SUMMARY  # noqa: E402
from s1arena.registry import make_model, tasks_under  # noqa: E402


def failure_kind(row: dict) -> str:
    response = row["response"]
    error = response["error"]
    if not runner.is_final(row):
        return "no reply (a new run retries it)"
    for needle, kind in (("longer than", "prompt longer than the context"), ("max_length_exceeded", "prompt longer than the context"),
                         ("fit in max_len", "options do not fit the input"),
                         ("content filter", "blocked by the content filter"),
                         ("authorization error", "refused (reported as an authorization error)")):
        if needle in error:
            return kind
    return "unusable reply" if response.get("raw_text") is not None else error[:60]


def stats(values: list[float]) -> dict | None:
    if not values:
        return None
    v = np.asarray(values, float)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p99": float(np.percentile(v, 99)), "max": float(v.max()),
            "share_above_0": float(np.mean(v > 0)), "total": float(v.sum())}


def audit(spec: str, data: dict[str, list], k: int) -> dict:
    model = make_model(spec)
    c, fail, by_task = Counter(), Counter(), defaultdict(Counter)
    think, mass, top, latency, served, finish, tokens = [], [], [], [], Counter(), Counter(), Counter()
    batch_ms = 0.0  # local models: each answer logs its batch's wall-clock time divided by the batch's requests
    for task, items in data.items():
        expected = {i.id: i for i in items}
        rows = runner.read_runs(model.slug, task)
        latest = {(r["item_id"], r["sample"]): r for r in runner.latest_rows(model.slug, task)}
        c["rows"] += len(rows)
        c["rows of other samples"] += sum(r["sample"] >= k for r in rows)
        c["rows of items not in the data"] += sum(r["item_id"] not in expected for r in rows)
        c["rows with another model id"] += sum(r.get("model") != model.model_id for r in rows)
        errors = Counter((r["item_id"], r["sample"]) for r in rows if r["response"].get("error"))
        for key, n in errors.items():  # earlier failed calls of an item: answered later, or followed by its final failure
            if key in latest and not latest[key]["response"].get("error"):
                c["failed calls answered later"] += n
            else:
                c["failed calls before a final failure"] += n - 1
        for item in items:
            questions = item.request.questions
            for s in range(k):
                row = latest.get((item.id, s))
                if row is None:
                    c["missing answers"] += 1
                    by_task[task]["missing"] += 1
                    continue
                response, usage = row["response"], row["response"].get("usage") or {}
                answers = response.get("answers") or {}
                c["answers"] += 1
                c["questions"] += len(questions)
                c["questions scored as uniform"] += sum(q not in answers for q in questions)
                if response.get("error"):
                    kind = failure_kind(row)
                    fail[kind] += 1
                    by_task[task][kind] += 1
                elif set(answers) != set(questions):
                    c["successful answers missing a question"] += 1
                for name, answer in answers.items():
                    probs = answer.get("probs") or {}
                    if name not in questions or set(probs) != set(questions[name].option_ids()):
                        c["distributions over other options"] += 1
                    elif any(not math.isfinite(p) or p < 0 for p in probs.values()) or abs(sum(probs.values()) - 1) > 1e-3:
                        c["distributions that are not normalized"] += 1
                    else:
                        top.append(max(probs.values()))
                if response.get("error") and not usage:
                    continue
                if "think_tokens" in usage:
                    think.append(usage["think_tokens"] or 0)
                    allowed = getattr(model, "think", 1) != 0 and getattr(model, "effort", "low") != "none"
                    c["thinking tokens where none are allowed"] += not allowed and bool(usage["think_tokens"])
                mass += list((usage.get("candidate_mass") or {}).values())
                c["states cut to fit the input"] += bool(usage.get("truncated"))
                batch_ms += usage.get("batch_ms_per_request") or 0
                if model.kind == "copilot":
                    served[f"{usage.get('model')}@{usage.get('reasoning_effort')}"] += 1
                    c["served by another model or effort"] += (usage.get("model"), usage.get("reasoning_effort")) != (model.copilot_id, model.effort)
                    finish[str(usage.get("finish_reason"))] += 1
                    for key in ("input_tokens", "output_tokens", "think_tokens", "cache_read_tokens", "aiu"):
                        tokens[key] += usage.get(key) or 0
                    if response.get("latency_ms"):
                        latency.append(response["latency_ms"])
    out = {"spec": spec, "model": model.model_id, "complete": not c["missing answers"] and not fail["no reply (a new run retries it)"],
           "counts": dict(c), "failures": dict(fail), "failures_by_task": {t: dict(v) for t, v in sorted(by_task.items())},
           "top_probability": {"mean": float(np.mean(top)), "share_at_least_0.999": float(np.mean(np.asarray(top) >= 0.999))} if top else None,
           "think_tokens": stats(think),
           "label_mass": {"mean": float(np.mean(mass)), "share_below_0.5": float(np.mean(np.asarray(mass) < 0.5))} if mass else None,
           "batch_hours": batch_ms / 3.6e6 if batch_ms else None}
    if model.kind == "copilot":
        out |= {"served": dict(served), "finish_reason": dict(finish), "usage_totals": dict(tokens),
                "latency_ms": {"p50": float(np.median(latency)), "p95": float(np.percentile(latency, 95))} if latency else None}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--leaf", nargs="+", default=[""], help="task prefixes, e.g. basic_cls know")
    parser.add_argument("--k", type=int, default=1, help="samples every item must have")
    parser.add_argument("--limit", type=int, default=0, help="first N items of every source (0 = all)")
    parser.add_argument("--out", default="", help="file name under results/summary/")
    args = parser.parse_args()

    tasks = [t for prefix in args.leaf for t in tasks_under(prefix)]
    data = {t.name: dataio.load_task(t)[: args.limit or None] for t in tasks}
    print(f"{len(tasks)} tasks, {sum(map(len, data.values()))} items, "
          f"{sum(len(i.request.questions) for items in data.values() for i in items)} questions; not built: {[t for t, v in data.items() if not v]}")
    report = {}
    for spec in args.models:
        a = audit(spec, data, args.k)
        report[a["model"]] = a
        print(f"\n{a['model']}: {'complete' if a['complete'] else 'INCOMPLETE'}")
        print("  " + "; ".join(f"{key}: {value:,}" for key, value in a["counts"].items() if value or key in ("answers", "missing answers")))
        print(f"  failures: {a['failures'] or 0}")
        for key in ("think_tokens", "label_mass", "top_probability", "batch_hours", "served", "usage_totals", "latency_ms"):
            if a.get(key):
                print(f"  {key}: {a[key]}")
    if args.out:
        SUMMARY.mkdir(parents=True, exist_ok=True)
        (SUMMARY / args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {SUMMARY / args.out}")


if __name__ == "__main__":
    main()
