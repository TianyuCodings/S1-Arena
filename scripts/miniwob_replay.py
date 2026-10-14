"""Check that every MiniWoB++ task stays solvable in harness version 2, where actions land at grid coordinates.

Replays the successful version-1 episodes of the given models (results/episodes/<model>/loop__web__miniwob__<task>.jsonl,
where each decision chose one of the page's numbered elements) in the current harness. The element is found on the
same page with the version-1 numbering and becomes the grid cell at the centre of its visible box, with the same
operation and text. When the element is out of view in a scroll box, scroll actions toward it come first; when the
centre cell does not reach the element (something else is on top there, or an option inside it, as on a list box),
the nearest cell inside the box that does is used; a click on an option of a closed drop-down list becomes typing the
option's text into the list. A replay succeeds when the page ends the episode with
a positive reward within the task's decision budget.

Usage (on a machine with Chrome and the miniwob package):
    python scripts/miniwob_replay.py --episodes results/episodes --out miniwob_replay.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from s1arena.envs.loop.web import AREA_H, AREA_W, CELL, NO_OP  # noqa: E402
from s1arena.envs.loop.web_page import CONTROL_TAGS, SCROLL_BOX_JS, classes, element_id, short  # noqa: E402
from s1arena.registry import envs_under  # noqa: E402
from s1arena.schema import MAX_OPTIONS  # noqa: E402

MODELS = ["typesafe-ai__jev", "openai__gpt-4.1-mini", "anthropic__claude-haiku-4.5"]
MAX_SCROLLS = 12  # scroll actions tried to bring one element into view

# Does point (x, y) reach element i of the latest snapshot: the element itself, or something inside it that passes the
# click on (not an option, which a click selects: a list box's centre lies on one of its options)?
REACHES_JS = r"""
var el = window.__jvNodes[arguments[0]], hit = document.elementFromPoint(arguments[1], arguments[2]);
return !!hit && (hit === el || (el.contains(hit) && hit.tagName !== 'OPTION'));
"""
# The innermost scroll box around element i, and whether the element lies above (-1) or below (1) its visible part.
SCROLL_TO_JS = SCROLL_BOX_JS + r"""
var el = window.__jvNodes[arguments[0]], r = el.getBoundingClientRect();
for (var p = el.parentElement; p && p !== document.body; p = p.parentElement) {
  if (scrollBox(p)) return [window.__jvNodes.indexOf(p), r.top < p.getBoundingClientRect().top + p.clientTop ? -1 : 1];
}
return null;
"""
BACKGROUND_JS = "var hit = document.elementFromPoint(arguments[0], arguments[1]); return !hit || ['BODY', 'HTML'].indexOf(hit.tagName) >= 0 || hit.id === 'wrap' || hit.id === 'area';"


class OverBudget(Exception):
    pass


def v1_targets(nodes: list[dict]) -> list[int]:
    """Snapshot indices of the elements version 1 numbered [1..n]: visible text, form controls, links and buttons, identifiable leaves."""
    def numbered(node: dict) -> bool:
        if node.get("text") or node["tag"] in CONTROL_TAGS or node.get("editable"):
            return True
        return node.get("leaf", False) and bool(element_id(node) or classes(node) or node.get("title") or node.get("aria-label") or node.get("alt"))
    return [i for i, node in enumerate(nodes) if numbered(node)][:MAX_OPTIONS]


def cell_at(x: float, y: float) -> tuple[int, int]:
    return min(max(int(y // CELL) + 1, 1), AREA_H // CELL), min(max(int(x // CELL) + 1, 1), AREA_W // CELL)


def centre(cell: tuple[int, int]) -> tuple[float, float]:
    return (cell[1] - 0.5) * CELL, (cell[0] - 0.5) * CELL


class Replay:
    def __init__(self, env_cls, seed: int):
        self.env, self.budget = env_cls(), env_cls.max_decisions
        self.env.reset(seed)
        self.decisions = 0
        self.notes: collections.Counter = collections.Counter()
        self.unreachable: list[str] = []

    def act(self, operation: str | None, cell: tuple[int, int] | None = None, text_id: str | None = None) -> None:
        """One version-2 decision, checked against the options of the current request (None: no action)."""
        if self.decisions >= self.budget:
            raise OverBudget
        questions = self.env.observe().questions
        if operation is None:
            choice = {name: NO_OP for name in questions}
        else:
            choice = {"row": str(cell[0]), "column": str(cell[1])}
            if "operation" in questions:
                choice["operation"] = operation
            elif operation != "click":
                raise ValueError(f"{operation} is not possible on this page")
            if "text" in questions:
                choice["text"] = text_id or "t1"  # ignored by a click
            for name, option in choice.items():
                if option != NO_OP and option not in questions[name].criteria:  # NO_OP: the runner's default for an unanswered question
                    raise ValueError(f"{name}={option} is not an option")
        self.env.act(choice)
        self.decisions += 1

    def reaching_cell(self, i: int) -> tuple[int, int] | None:
        """The cell at the centre of element i's visible box if it reaches the element, else the nearest cell in the box that does."""
        x, y, w, h = self.env.nodes[i]["box"]
        middle = cell_at(x + w / 2, y + h / 2)
        inside = [(r, c) for r in range(1, AREA_H // CELL + 1) for c in range(1, AREA_W // CELL + 1)
                  if x <= (c - 0.5) * CELL < x + w and y <= (r - 0.5) * CELL < y + h]
        inside.sort(key=lambda rc: (rc[0] - middle[0]) ** 2 + (rc[1] - middle[1]) ** 2)
        for cell in [middle] + [rc for rc in inside if rc != middle]:
            if self.env.driver.execute_script(REACHES_JS, i, *centre(cell)):
                if cell != middle:
                    self.notes["off-centre cell"] += 1
                return cell
        return None

    def cell_for(self, i: int) -> tuple[int, int]:
        """A cell that reaches element i, or the centre of its box when none does (the action then misses it)."""
        cell = None if self.env.nodes[i].get("offscreen") else self.reaching_cell(i)
        if cell is None:
            self.notes["unreachable"] += 1
            self.unreachable.append(short(self.env.nodes[i]))
            x, y, w, h = self.env.nodes[i]["box"]
            cell = cell_at(x + w / 2, y + h / 2)
        return cell

    def step(self, choice: dict[str, str]) -> None:
        """Play one version-1 decision in version 2."""
        env = self.env
        element, operation, text_id = choice.get("element", NO_OP), choice.get("operation", "click"), choice.get("text")
        if NO_OP in (element, operation) or (operation == "type" and text_id in (None, NO_OP)):
            self.act(None)
            return
        n = int(element)
        targets = v1_targets(env.nodes)
        if n > len(targets):  # version 1's "page background" option
            cell = next((r, c) for r in range(1, AREA_H // CELL + 1) for c in range(1, AREA_W // CELL + 1)
                        if env.driver.execute_script(BACKGROUND_JS, *centre((r, c))))
            self.notes["background"] += 1
            self.act("click", cell)
            return
        i = targets[n - 1]
        if "box" not in env.nodes[i]:  # an option of a closed drop-down list: type its text into the list
            select = max(k for k in range(i) if env.nodes[k]["tag"] == "select")
            self.notes["drop-down typed"] += 1
            self.act("type", self.cell_for(select), f"t{env.texts.index(env.nodes[i]['text']) + 1}")
            return
        cell = None
        for _ in range(MAX_SCROLLS + 1):
            cell = None if env.nodes[i].get("offscreen") else self.reaching_cell(i)
            if cell is not None:
                break
            around = env.driver.execute_script(SCROLL_TO_JS, i)
            if around is None:
                break
            box, direction = around
            self.act("scroll down" if direction > 0 else "scroll up", self.cell_for(box))
            self.notes["scroll"] += 1
            i = v1_targets(env.nodes)[n - 1]
        self.act(operation, cell or self.cell_for(i), text_id)


def load(episodes: Path, models: list[str], tasks: set[str]) -> list[tuple[str, str, dict]]:
    """The latest successful version-1 paused episode per (model, task, seed)."""
    jobs = []
    for model in models:
        for path in sorted((episodes / model).glob("loop__web__miniwob__*.jsonl")):
            task = path.stem.removeprefix("loop__web__miniwob__")
            if tasks and task not in tasks:
                continue
            latest = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line) if line.strip() else {}
                if row.get("env_version", 1) == 1 and row.get("mode") == "paused" and "outcome" in row:
                    latest[row["seed"]] = row
            jobs += [(model, task, row) for seed, row in sorted(latest.items()) if row["outcome"].get("success")]
    return jobs


def replay(job: tuple[str, str, dict], env_classes: dict[str, type]) -> dict:
    model, task, row = job
    env_cls = env_classes[f"loop/web/miniwob/{task}"]
    out = {"model": model, "task": task, "seed": row["seed"], "v1_decisions": row["decisions"], "budget": env_cls.max_decisions}
    run = None
    try:
        run = Replay(env_cls, row["seed"])
        try:
            for step in row["trace"]:
                if run.env.done():
                    break
                run.step(step["c"])
        except OverBudget:
            run.notes["over budget"] += 1
        outcome = run.env.outcome()
        out.update(success=bool(outcome["success"]), score=outcome["score"], decisions=run.decisions, notes=dict(run.notes), actions=outcome["actions"],
                   unreachable=run.unreachable)
    except Exception as error:
        out.update(success=False, error=f"{type(error).__name__}: {str(error)[:300]}", notes=dict(run.notes) if run else {})
    finally:
        if run is not None:
            run.env.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=Path, default=ROOT / "results" / "episodes", help="folder with one sub-folder of episode files per model")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None, help="write every replay as JSON here")
    args = parser.parse_args()
    env_classes = {cls.name: cls for cls in envs_under("loop/web/miniwob")}
    jobs = load(args.episodes, args.models, set(args.tasks))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda job: replay(job, env_classes), jobs))
    if args.out:
        args.out.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    by_task = collections.defaultdict(list)
    for result in results:
        by_task[result["task"]].append(result)
    print(f"{'task':30} {'replayed':>9}  notes")
    for task in sorted(by_task):
        rows = by_task[task]
        notes = collections.Counter()
        for result in rows:
            notes.update(result.get("notes", {}))
        errors = sum("error" in result for result in rows)
        print(f"{task:30} {sum(r['success'] for r in rows):>4} / {len(rows):<3} {dict(notes) or ''}" + (f" errors={errors}" if errors else ""))
    print(f"{'all':30} {sum(r['success'] for r in results):>4} / {len(results)}")


if __name__ == "__main__":
    main()
