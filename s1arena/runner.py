"""Run a model over built items. Responses are cached per (model, item, sample),
so re-running only fills in what is missing and new models only cost their own calls."""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .env import RUNS
from .schema import Item, ModelResponse


BATCH = 2000  # requests per decide_many call (local models)


def run_path(model_slug: str, task: str) -> Path:
    return RUNS / model_slug / (task.replace("/", "__") + ".jsonl")


def read_runs(model_slug: str, task: str) -> list[dict]:
    path = run_path(model_slug, task)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# 4xx statuses that reject the request itself (e.g. an input longer than the model's context). Other 4xx statuses
# (401 / 403 auth, 402 an empty gateway balance, 404, 408, 429) say nothing about the item and are retried.
REQUEST_REJECTED = (400, 413, 422)


def is_final(row: dict) -> bool:
    """A success, a reply that could not be used (its questions score 0; see reparse_task), or a request the model
    rejects outright (see REQUEST_REJECTED).

    Only calls that got no reply because of transport, account, rate-limit or server errors are retried, so a
    model is never re-sampled because its first answer failed.
    """
    response = row["response"]
    error = response.get("error")
    if not error or response.get("raw_text") is not None or error.startswith("rejected: "):  # local models: input too long
        return True
    status = re.search(r"HTTP (\d{3})", error)
    return bool(status) and int(status.group(1)) in REQUEST_REJECTED


def run_task(model, task: str, items: list[Item], k: int = 1, workers: int = 8,
             budget_usd: float | None = None) -> int:
    """Query the model k times per item for anything not already cached; returns calls made.

    With budget_usd, calls not yet started are cancelled once this run's spend reaches it (calls in flight finish).
    """
    path = run_path(model.slug, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["item_id"], r["sample"]) for r in read_runs(model.slug, task) if is_final(r)}
    jobs = [(item, s) for item in items for s in range(k) if (item.id, s) not in done]
    if not jobs:
        return 0
    started = time.time()
    if hasattr(model, "decide_many"):  # local models answer a whole batch at once
        return sum(run_tasks_batched(model, [(task, items)], k).values())
    lock = threading.Lock()
    made, spent = 0, 0.0
    with path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(model.decide, item.request, s): (item, s) for item, s in jobs}
        for future in as_completed(futures):
            if future.cancelled():
                continue
            item, s = futures[future]
            try:
                response = future.result()
            except Exception as error:  # a crash is recorded like any other failed call
                response = ModelResponse(error=f"{type(error).__name__}: {str(error)[:300]}")
            row = {"item_id": item.id, "sample": s, "model": model.model_id, "response": response.model_dump()}
            with lock:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
            made += 1
            spent += response.cost_usd or 0.0
            if budget_usd is not None and spent >= budget_usd:
                cancelled = sum(f.cancel() for f in futures)
                if cancelled:
                    print(f"  {task}: budget reached (${spent:.2f}), {cancelled} calls not started", flush=True)
                budget_usd = None
            if made % 100 == 0 or made == len(jobs):
                print(f"  {task}: {made}/{len(jobs)} calls, {time.time() - started:.0f}s", flush=True)
    return made


def run_tasks_batched(model, tasks: list[tuple[str, list[Item]]], k: int = 1) -> dict[str, int]:
    """Local models: pool the uncached calls of many tasks into shared decide_many batches of BATCH requests, so
    small tasks do not each pay for a half-empty batch. Rows go to each task's own cache file; returns calls per task."""
    jobs = []
    for task, items in tasks:
        done = {(r["item_id"], r["sample"]) for r in read_runs(model.slug, task) if is_final(r)}
        jobs += [(task, item, s) for item in items for s in range(k) if (item.id, s) not in done]
    calls: dict[str, int] = {}
    started = time.time()
    for start in range(0, len(jobs), BATCH):
        chunk = jobs[start:start + BATCH]
        try:
            responses = model.decide_many([(item.request, s) for _, item, s in chunk])
        except Exception as error:  # a crashed batch is recorded like failed calls and retried next time
            responses = [ModelResponse(error=f"{type(error).__name__}: {str(error)[:300]}") for _ in chunk]
        rows: dict[str, list[str]] = {}
        for (task, item, s), response in zip(chunk, responses):
            row = {"item_id": item.id, "sample": s, "model": model.model_id, "response": response.model_dump()}
            rows.setdefault(task, []).append(json.dumps(row, ensure_ascii=False) + "\n")
            calls[task] = calls.get(task, 0) + 1
        for task, lines in rows.items():
            path = run_path(model.slug, task)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as out:
                out.writelines(lines)
        print(f"  batch {start + len(chunk)}/{len(jobs)} calls over {len(rows)} tasks, {time.time() - started:.0f}s", flush=True)
    return calls


def reparse_task(model, task: str, items: list[Item]) -> int:
    """Re-read the stored replies of failed calls with the current parser; no new calls.

    A fixed row is appended with "reparsed": true, so latest_rows picks it over the failure.
    """
    requests = {item.id: item.request for item in items}
    fixed = []
    for row in latest_rows(model.slug, task):
        response = row["response"]
        if row["item_id"] not in requests or not response.get("error") or not response.get("raw_text"):
            continue
        new = model.reparse(ModelResponse(**response), requests[row["item_id"]])
        if new is not None and not new.error:
            fixed.append({**row, "response": new.model_dump(), "reparsed": True})
    if fixed:
        with run_path(model.slug, task).open("a", encoding="utf-8") as out:
            for row in fixed:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(fixed)


def _rank(row: dict) -> int:
    return 2 if not row["response"].get("error") else 1 if is_final(row) else 0


def latest_rows(model_slug: str, task: str) -> list[dict]:
    """One row per (item, sample): the most recent successful response, else the most recent final failure (a reply
    that could not be used or a request the model rejects, see is_final), else the most recent failure. A transport
    error recorded after a final row does not hide it: run_task never calls such an item again."""
    best: dict[tuple, dict] = {}
    for row in read_runs(model_slug, task):
        key = (row["item_id"], row["sample"])
        if key not in best or _rank(row) >= _rank(best[key]):
            best[key] = row
    return list(best.values())
