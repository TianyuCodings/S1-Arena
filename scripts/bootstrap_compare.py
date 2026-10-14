"""Confidence intervals and robustness views for `s1arena compare`, from the cached runs (sample 0).

Every question is scored exactly as s1arena.metrics.score scores it (checked task by task); a leaf is the mean over its
tasks, and a domain or the overall score the mean over leaves, as in `s1arena compare`. From those:
- a paired bootstrap over items, stratified by task (the same resampled items for every model): 95% intervals of
  accuracy and Brier per model, the interval of each model's overall rank, and for every pair of models the share of
  replicates in which the first one is better;
- the common subset: the items that every model answered without a failure, rescored (a task left without items is
  dropped), optionally ignoring the failures of some models (--common-ignore); and each model on its own answered
  items. Both show how much scoring failed answers as wrong moves a model.

Usage:
  python scripts/bootstrap_compare.py --models lm:Qwen/Qwen3-8B copilot:gpt-6-sol@effort=none --leaf basic_cls know \
      [--b 1000] [--limit 0] [--common-ignore SPEC ...] [--out bootstrap.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s1arena import dataio, metrics, runner  # noqa: E402
from s1arena.env import SUMMARY  # noqa: E402
from s1arena.registry import make_model, tasks_under  # noqa: E402


def score_task(task, items, models) -> dict:
    """Per question and model: hit, Brier and top-label confidence; per item and model: whether the answer failed."""
    rows = [{r["item_id"]: r for r in runner.latest_rows(m.slug, task.name) if r["sample"] == 0} for m in models]
    hit, brier, conf, qitem = [], [], [], []
    failed = np.array([[bool(rows[k][i.id]["response"].get("error")) for i in items] for k in range(len(models))])
    for j, item in enumerate(items):
        for name, gold in item.gold.items():
            options = item.request.questions[name].option_ids()
            target, truth = gold.target(options), gold.hard()
            h, b, c = [], [], []
            for k in range(len(models)):
                probs, answered = metrics._answer(rows[k][item.id], name, options)  # no answer: wrong, uniform for Brier
                pred = max(options, key=lambda o: probs[o])
                h.append(float(answered and pred == truth))
                b.append(sum((probs[o] - target[o]) ** 2 for o in options))
                c.append(probs[pred] if answered else 1 / len(options))
            hit.append(h), brier.append(b), conf.append(c), qitem.append(j)
    out = {"leaf": task.leaf, "n": len(items), "hit": np.array(hit).T, "brier": np.array(brier).T, "conf": np.array(conf).T,
           "qitem": np.array(qitem), "failed": failed}
    for k in range(len(models)):  # the project's scorer gives the same numbers
        ref = metrics.score(items, list(rows[k].values()))
        assert abs(ref["accuracy"] - out["hit"][k].mean()) < 1e-9 and abs(ref["brier"] - out["brier"][k].mean()) < 1e-9, (task.name, k)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--leaf", nargs="+", default=[""], help="task prefixes, e.g. basic_cls know")
    parser.add_argument("--limit", type=int, default=0, help="first N items of every source (0 = all)")
    parser.add_argument("--b", type=int, default=1000, help="bootstrap replicates")
    parser.add_argument("--common-ignore", nargs="*", default=[], help="models whose failures do not shrink a second common subset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="", help="file name under results/summary/")
    args = parser.parse_args()

    models = [make_model(s) for s in args.models]
    names = [m.model_id for m in models]
    ignore = {make_model(s).model_id for s in args.common_ignore}
    tasks = [t for prefix in args.leaf for t in tasks_under(prefix)]
    per_task = {}
    for t in tasks:
        items = dataio.load_task(t)[: args.limit or None]
        if items:
            per_task[t.name] = score_task(t, items, models)
    leaves = sorted({d["leaf"] for d in per_task.values()})
    scopes = {"overall": leaves} | {d: [l for l in leaves if l.split("/")[0] == d] for d in sorted({l.split("/")[0] for l in leaves})}
    print(f"scored {len(per_task)} tasks, {len(leaves)} leaves, {len(models)} models", flush=True)

    def aggregate(task_values: dict) -> dict:
        by_leaf: dict[str, list] = {}
        for name, v in task_values.items():
            by_leaf.setdefault(per_task[name]["leaf"], []).append(v)
        leaf_mean = {l: np.mean(vs, axis=0) for l, vs in by_leaf.items()}
        return {s: np.mean([leaf_mean[l] for l in ls if l in leaf_mean], axis=0) for s, ls in scopes.items()}

    def point(keep: dict | None = None) -> dict:
        """Accuracy, Brier and ECE per scope and model, on every item or on the items `keep` marks per task."""
        acc, bri, ece = {}, {}, {}
        for name, d in per_task.items():
            mask = np.ones(len(d["qitem"]), bool) if keep is None else keep[name][d["qitem"]]
            if mask.any():
                acc[name], bri[name] = d["hit"][:, mask].mean(axis=1), d["brier"][:, mask].mean(axis=1)
                ece[name] = np.array([metrics.expected_calibration_error(list(d["conf"][k, mask]), list(d["hit"][k, mask]))
                                      for k in range(len(models))])
        a, b, e = aggregate(acc), aggregate(bri), aggregate(ece)
        return {s: {n: {"accuracy": float(a[s][k]), "brier": float(b[s][k]), "ece": float(e[s][k])} for k, n in enumerate(names)} for s in scopes}

    rng = np.random.default_rng(args.seed)
    boot_acc, boot_bri = {}, {}
    for name, d in per_task.items():  # the same resampled items for every model
        w = rng.multinomial(d["n"], np.full(d["n"], 1 / d["n"]), size=args.b).astype(float)[:, d["qitem"]]
        boot_acc[name] = w @ d["hit"].T / w.sum(axis=1, keepdims=True)
        boot_bri[name] = w @ d["brier"].T / w.sum(axis=1, keepdims=True)
    ba, bb = aggregate(boot_acc), aggregate(boot_bri)
    pct = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    ci = {s: {n: {"accuracy": pct(ba[s][:, k]), "brier": pct(bb[s][:, k])} for k, n in enumerate(names)} for s in scopes}
    rank_acc = (-ba["overall"]).argsort(axis=1).argsort(axis=1) + 1
    rank_bri = bb["overall"].argsort(axis=1).argsort(axis=1) + 1
    rank_ci = {n: {"accuracy": [int(x) for x in pct(rank_acc[:, k])], "brier": [int(x) for x in pct(rank_bri[:, k])]} for k, n in enumerate(names)}
    better = lambda x: {a: {b: float(np.mean(x[:, i] > x[:, j])) for j, b in enumerate(names) if j != i} for i, a in enumerate(names)}  # noqa: E731
    p_better = {"accuracy": better(ba["overall"]), "brier": better(-bb["overall"])}  # lower Brier is better

    views = {}
    for label, skip in (("common", set()), ("common_ignoring", ignore)):
        if label == "common_ignoring" and not ignore:
            continue
        rows = [k for k, n in enumerate(names) if n not in skip]
        keep = {name: ~d["failed"][rows].any(axis=0) for name, d in per_task.items()}
        views[label] = {"ignoring": sorted(skip), "items": int(sum(m.sum() for m in keep.values())),
                        "tasks_dropped": [n for n, m in keep.items() if not m.any()], "point": point(keep)}
    own = {n: point({name: ~d["failed"][k] for name, d in per_task.items()}) for k, n in enumerate(names)}
    views["own_answered"] = {"point": {s: {n: own[n][s][n] for n in names} for s in scopes}}

    full = point()
    out = {"models": names, "b": args.b, "items": int(sum(d["n"] for d in per_task.values())), "scopes": scopes, "point": full,
           "ci95": ci, "rank_ci95": rank_ci, "p_better": p_better, "views": views}
    for n in sorted(names, key=lambda n: -full["overall"][n]["accuracy"]):
        c, r = ci["overall"][n], rank_ci[n]["accuracy"]
        print(f"{n:<64} accuracy {full['overall'][n]['accuracy']:.3f} [{c['accuracy'][0]:.3f}, {c['accuracy'][1]:.3f}] rank {r[0]}-{r[1]}  "
              f"Brier {full['overall'][n]['brier']:.3f} [{c['brier'][0]:.3f}, {c['brier'][1]:.3f}]  "
              f"common {views['common']['point']['overall'][n]['accuracy']:.3f}  own {views['own_answered']['point']['overall'][n]['accuracy']:.3f}")
    print({k: (v["items"], v["tasks_dropped"]) for k, v in views.items() if "items" in v})
    if args.out:
        SUMMARY.mkdir(parents=True, exist_ok=True)
        (SUMMARY / args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {SUMMARY / args.out}")


if __name__ == "__main__":
    main()
