"""The Hugging Face dataset of built requests: one row per request, one config per domain, split "test".

`s1arena download` reads it into data/<task>.jsonl; scripts/export_hf_dataset.py writes it. Sources that do not allow
redistribution are not in it: they are in a private dataset of the same layout (HF_DATASET_RESTRICTED, read into
data_local/ by people who have access), and `s1arena build --local` rebuilds them from their loaders for everybody else.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from . import dataio
from .env import HF_DATASET, load_env
from .schema import DecisionRequest, Gold, Item, content_id

COLUMNS = ("id", "task", "leaf", "request", "gold", "source", "license", "tier", "url", "meta")


def item_to_row(item: Item, card: dict[str, Any] | None = None) -> dict[str, str]:
    """A built request as a flat row; request, gold and the full meta are JSON strings. The card of a registered task
    gives the current source, license, tier and url."""
    meta = dict(item.meta)
    if card:
        meta.update({k: card[k] for k in ("source", "license", "tier", "url") if card.get(k)})
    return {
        "id": item.id,
        "task": item.task,
        "leaf": item.leaf,
        "request": item.request.model_dump_json(),
        "gold": json.dumps({name: gold.model_dump() for name, gold in item.gold.items()}, ensure_ascii=False),
        "source": str(meta.get("source", "")),
        "license": str(meta.get("license", "")),
        "tier": str(meta.get("tier", "")),
        "url": str(meta.get("url", "")),
        "meta": json.dumps(meta, ensure_ascii=False),
    }


def row_to_item(row: dict[str, Any]) -> Item:
    """The inverse of item_to_row. The id must match the content, so a changed request is never used by mistake."""
    item = Item(
        id=row["id"],
        leaf=row["leaf"],
        task=row["task"],
        request=DecisionRequest.model_validate_json(row["request"]),
        gold={name: Gold(**gold) for name, gold in json.loads(row["gold"]).items()},
        meta=json.loads(row["meta"]),
    )
    if content_id(item) != item.id:
        raise ValueError(f"{item.id}: the content does not match the id")
    return item


def download(repo: str = HF_DATASET, domains: list[str] | None = None, revision: str | None = None,
             restricted: bool = False) -> dict[str, int]:
    """Write every request of the dataset (or of some domains) to data/<task>.jsonl, or to data_local/<task>.jsonl for
    the restricted dataset. Returns requests per task."""
    load_env()
    from datasets import get_dataset_config_names, load_dataset

    counts: dict[str, int] = {}
    configs = get_dataset_config_names(repo, revision=revision)
    for config in [c for c in configs if c in domains] if domains else configs:
        by_task: dict[str, list[Item]] = defaultdict(list)
        for row in load_dataset(repo, config, split="test", revision=revision):
            by_task[row["task"]].append(row_to_item(row))
        for task, items in sorted(by_task.items()):
            dataio.write_items(dataio.task_path(task, redistributable=not restricted), items)
            counts[task] = len(items)
    return counts
