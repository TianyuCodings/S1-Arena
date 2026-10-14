"""Quick checks for a new task or a new model adapter (`s1arena check-task`, `s1arena check-model`)."""

from __future__ import annotations

import itertools
import time
from typing import Any

from . import render
from .registry import TASKS, discover, make_model
from .schema import DecisionRequest, Item, Question, content_id

# Three small requests that cover the three question types; a model must answer all of them.
TOY_REQUESTS = {
    "choice": DecisionRequest(
        state={"review": "The package arrived two weeks late and the box was crushed. Support never answered."},
        questions={"sentiment": Question(type="choice", instructions="What is the overall sentiment of the review?",
                                         criteria={"positive": "The customer is satisfied.",
                                                   "neutral": "The customer is neither satisfied nor unsatisfied.",
                                                   "negative": "The customer is unsatisfied."})},
    ),
    "boolean": DecisionRequest(
        state={"message": "Hi, I was charged twice for order 1182. Please send the second payment back to my card."},
        questions={"refund": Question(type="boolean", instructions="The customer asks for money back.")},
    ),
    "score": DecisionRequest(
        state={"ticket": "The checkout page returns error 500 for every customer since 9:00. We lose all sales."},
        questions={"urgency": Question(type="score", instructions="How urgent is this ticket?",
                                       criteria=["Not urgent: it can wait for a week.", "Somewhat urgent: fix it this week.",
                                                 "Urgent: fix it today.", "Critical: fix it now."])},
    ),
}


def find_task(name: str):
    """The task with this name, or the only task whose name starts with it."""
    discover()
    if name in TASKS:
        return TASKS[name]()
    matches = [n for n in TASKS if n.startswith(name.rstrip("/") + "/")]
    if len(matches) != 1:
        raise SystemExit(f"{name!r} matches {len(matches)} tasks; give one task name (see `s1arena tasks`)")
    return TASKS[matches[0]]()


def check_task(name: str, n: int = 5, max_records: int = 200) -> tuple[Any, list[Item], list[str]]:
    """Load the first records of a task, convert up to n of them and validate every item. Returns (task, items,
    problems); a problem is a string that says what to fix."""
    task = find_task(name)
    problems = []
    card = task.card()
    for key in ("name", "leaf", "source", "url", "license", "criteria_source"):
        if not card.get(key):
            problems.append(f"the class attribute {key!r} is empty")
    if task.tier not in ("A", "B", "C"):
        problems.append(f"tier must be A, B or C, not {task.tier!r}")
    if not task.name.startswith(task.leaf + "/") or task.leaf.count("/") != 1:
        problems.append(f"name {task.name!r} must be <domain>/<leaf>/<source> and leaf {task.leaf!r} must be <domain>/<leaf>")
    items: list[Item] = []
    for record in itertools.islice((r for r in task.load() if r is not None), max_records):
        try:
            produced = task.to_items(record, task.record_rng(record, 0))
            task.label_of(record)
        except Exception as error:
            problems.append(f"a record failed: {type(error).__name__}: {error}")
            continue
        for item in produced:
            try:
                Item.model_validate_json(item.model_dump_json())  # every schema rule again, as a reader checks it
            except Exception as error:
                problems.append(f"{item.id}: invalid item: {error}")
                continue
            if item.task != task.name or item.leaf != task.leaf:
                problems.append(f"{item.id}: item task/leaf differ from the task's; build items with self.item(...)")
            if content_id(item) != item.id:
                problems.append(f"{item.id}: the id does not match the content")
            items.append(item)
        if len(items) >= n:
            break
    if not items:
        problems.append("no items: load() yields no records, or to_item() returns None for every record")
    return task, items[:n], problems


def check_model(spec: str) -> list[dict[str, Any]]:
    """Run a model on the toy requests and check each distribution. One row per request: kind, latency, problems."""
    model = make_model(spec)
    rows = []
    for kind, request in TOY_REQUESTS.items():
        started = time.perf_counter()
        try:
            response = model.decide(request, 0)
        except Exception as error:
            rows.append({"kind": kind, "latency_ms": None, "answers": {}, "problems": [f"decide() raised {type(error).__name__}: {error}"]})
            continue
        wall = (time.perf_counter() - started) * 1000
        problems = [f"error: {response.error}"] if response.error else []
        for name, question in request.questions.items():
            answer = response.answers.get(name)
            if answer is None:
                problems.append(f"no answer for question {name!r}")
                continue
            options = question.option_ids()
            unknown = set(answer.probs) - set(options)
            if unknown:
                problems.append(f"{name!r}: unknown option ids {sorted(unknown)}; allowed: {options}")
            if any(p < 0 for p in answer.probs.values()):
                problems.append(f"{name!r}: negative probability")
            total = sum(answer.probs.get(o, 0.0) for o in options)
            if abs(total - 1) > 1e-3:
                problems.append(f"{name!r}: probabilities sum to {total:.4f}, not 1")
        rows.append({"kind": kind, "latency_ms": response.latency_ms if response.latency_ms is not None else wall,
                     "answers": {k: v.probs for k, v in response.answers.items()}, "problems": problems})
    return rows


def rendered(request: DecisionRequest) -> str:
    """The request as the shared LLM prompt shows it."""
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in render.messages(request))
