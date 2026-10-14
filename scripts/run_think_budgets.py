"""Run a thinklm: or thinkdgemma: model at several thinking budgets, from one thought per question
(s1arena/models/thinking_lm.py).

Each budget gets its own run, results/runs/<slug of <kind>:<model>@think=<B>>/<task>.jsonl. The rows are written as
scripts/run_lm_batch.py writes them, so `s1arena compare` and `score` read every budget as a model of its own.
A request is sent again only while some budget lacks its final row, so the script can be re-run after a crash.

Usage (on a CUDA machine with vLLM):
  python scripts/run_think_budgets.py --model Qwen/Qwen3-8B --budgets 128 256 512 1024 2048 [--kind thinkdgemma]
      [--opts tp=2] [--leaf basic_cls know ...] [--limit N] [--shard I/N]

--shard I/N takes every N-th task from the I-th (0-based), to split one model over several processes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s1arena import dataio, runner  # noqa: E402
from s1arena.registry import make_model, tasks_under  # noqa: E402
from s1arena.schema import ModelResponse  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face id, e.g. Qwen/Qwen3-8B")
    parser.add_argument("--budgets", type=int, nargs="+", required=True)
    parser.add_argument("--kind", default="thinklm", choices=["thinklm", "thinkdgemma"])
    parser.add_argument("--opts", default="", help="more spec options, e.g. tp=2")
    parser.add_argument("--leaf", nargs="+", default=[""], help="task prefixes, e.g. basic_cls know")
    parser.add_argument("--limit", type=int, default=0, help="first N items of every source (0 = all)")
    parser.add_argument("--shard", default="0/1", help="I/N: every N-th task from the I-th")
    parser.add_argument("--chunk", type=int, default=4000, help="requests per call")
    args = parser.parse_args()

    budgets = sorted(set(args.budgets))
    opts = "," + args.opts if args.opts else ""
    models = {b: make_model(f"{args.kind}:{args.model}@think={b}{opts}") for b in budgets}
    engine = models[budgets[-1]]  # its thoughts, up to the largest budget, serve every budget
    index, count = (int(x) for x in args.shard.split("/"))
    tasks = sorted({t.name: t for leaf in args.leaf for t in tasks_under(leaf)}.values(), key=lambda t: t.name)
    jobs, done = [], {}
    for task in tasks[index::count]:
        items = dataio.load_task(task)[: args.limit or None]
        for b, m in models.items():
            done[b, task.name] = {r["item_id"] for r in runner.read_runs(m.slug, task.name)
                                  if r["sample"] == 0 and runner.is_final(r)}
        jobs += [(task.name, item) for item in items if not all(item.id in done[b, task.name] for b in budgets)]
    print(f"{args.model} at budgets {budgets}: {len(jobs)} requests over {len({t for t, _ in jobs})} tasks", flush=True)

    started = time.time()
    for start in range(0, len(jobs), args.chunk):
        chunk = jobs[start:start + args.chunk]
        try:
            out = engine.decide_budgets([(item.request, 0) for _, item in chunk], budgets)
        except Exception as error:  # recorded like failed calls, retried on the next run
            failed = [ModelResponse(error=f"{type(error).__name__}: {str(error)[:300]}") for _ in chunk]
            out = {b: failed for b in budgets}
        for b, responses in out.items():
            by_task = defaultdict(list)
            for (task, item), response in zip(chunk, responses):
                if item.id not in done[b, task]:
                    by_task[task].append({"item_id": item.id, "sample": 0, "model": models[b].model_id,
                                          "response": response.model_dump()})
            for task, rows in by_task.items():
                path = runner.run_path(models[b].slug, task)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        errors, elapsed = sum(bool(r.error) for r in out[budgets[-1]]), time.time() - started
        print(f"  {start + len(chunk)}/{len(jobs)} requests, {errors} errors in this chunk, {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
