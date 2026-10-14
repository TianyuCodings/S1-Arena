"""Run a local language model (s1arena/models/local_lm.py) on many tasks as one large batch.

`s1arena run` sends one task at a time; a local model is much faster when vLLM sees the requests of all tasks
together. Rows go to results/runs/<model slug>/<task>.jsonl exactly as `s1arena run` writes them, so `s1arena
compare` and `score` read them unchanged. Cached final rows are skipped, so the script can be re-run after a crash.

Usage (on a CUDA machine with vLLM):
  python scripts/run_lm_batch.py --model "lm:Qwen/Qwen3-8B@think=512" --limit 2 [--leaf PREFIX] [--k 1] [--latency 50]

--latency N also answers N requests one at a time and writes their wall-clock latency to
results/summary/latency_<model slug>.json (a batch hides the latency of a single decision).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s1arena import dataio, runner  # noqa: E402
from s1arena.env import ROOT  # noqa: E402
from s1arena.registry import make_model, tasks_under  # noqa: E402
from s1arena.schema import ModelResponse  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--leaf", default="")
    parser.add_argument("--limit", type=int, default=0, help="first N items of every source (0 = all)")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=4000, help="requests per decide_many call")
    parser.add_argument("--latency", type=int, default=0, help="also time N single requests")
    args = parser.parse_args()

    model = make_model(args.model)
    jobs, probe = [], []
    for task in tasks_under(args.leaf):
        items = dataio.load_task(task)[: args.limit or None]
        done = {(r["item_id"], r["sample"]) for r in runner.read_runs(model.slug, task.name) if runner.is_final(r)}
        jobs += [(task.name, item, s) for item in items for s in range(args.k) if (item.id, s) not in done]
        probe += items[:1]
    print(f"{model.model_id}: {len(jobs)} requests over {len({t for t, _, _ in jobs})} tasks", flush=True)

    started = time.time()
    for start in range(0, len(jobs), args.chunk):
        chunk = jobs[start:start + args.chunk]
        try:
            responses = model.decide_many([(item.request, s) for _, item, s in chunk])
        except Exception as error:  # recorded like failed calls, retried on the next run
            responses = [ModelResponse(error=f"{type(error).__name__}: {str(error)[:300]}") for _ in chunk]
        by_task = defaultdict(list)
        for (task, item, s), response in zip(chunk, responses):
            by_task[task].append({"item_id": item.id, "sample": s, "model": model.model_id, "response": response.model_dump()})
        for task, rows in by_task.items():
            path = runner.run_path(model.slug, task)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as out:
                out.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        errors = sum(bool(r.error) for r in responses)
        print(f"  {start + len(chunk)}/{len(jobs)} requests, {errors} errors in this chunk, {time.time() - started:.0f}s", flush=True)

    if args.latency:
        step = max(1, len(probe) // args.latency)
        sample = probe[::step][: args.latency]
        times = []
        for item in sample:
            t0 = time.time()
            response = model.decide_many([(item.request, 0)])[0]
            if not response.error:
                times.append((time.time() - t0) * 1000)
        out = {"model": model.model_id, "requests": len(times), "median_ms": float(np.median(times)) if times else None,
               "p90_ms": float(np.percentile(times, 90)) if times else None, "mean_ms": float(np.mean(times)) if times else None}
        path = ROOT / "results" / "summary" / f"latency_{model.slug}.json"
        path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print("latency:", out, flush=True)


if __name__ == "__main__":
    main()
