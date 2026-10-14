"""Reading and writing built items (one JSONL file per task)."""

from __future__ import annotations

import json
from pathlib import Path

from .env import DATA, DATA_LOCAL
from .schema import Item


def task_path(task: str, redistributable: bool = True) -> Path:
    return (DATA if redistributable else DATA_LOCAL) / f"{task}.jsonl"


def write_items(path: Path, items: list[Item]) -> None:
    """Write via a temporary file, so a run reading the task meanwhile never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")
    tmp.replace(path)


def read_items(path: Path) -> list[Item]:
    with path.open(encoding="utf-8") as f:
        return [Item.model_validate_json(line) for line in f if line.strip()]


def load_task(task) -> list[Item]:
    """Built items for a task adapter instance; empty if it has not been built yet."""
    path = task_path(task.name, task.redistributable)
    return read_items(path) if path.exists() else []


def write_manifest(root: Path, rows: list[dict], known: set[str] | None = None) -> None:
    """Merge rows into root/manifest.json by task name; entries of tasks not rebuilt are kept,
    except tasks that are no longer registered (when `known` is given)."""
    path = root / "manifest.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    new = {row["name"]: row for row in rows}
    merged = [new.pop(row["name"], row) for row in old] + list(new.values())
    merged = [row for row in merged if known is None or row["name"] in known]
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
