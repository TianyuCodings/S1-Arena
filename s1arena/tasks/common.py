"""Helpers shared by task adapters: data access and label formatting."""

from __future__ import annotations

import re
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from ..env import ROOT, load_env

CACHE = ROOT / ".cache" / "downloads"


def hf_rows(repo: str, config: str | None = None, split: str = "test", revision: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
    """Rows of a Hugging Face split, with ClassLabel ints replaced by their names.

    revision="refs/convert/parquet" reads the Hub's parquet conversion of script-based repos;
    other keyword arguments (e.g. data_files) go to datasets.load_dataset.
    """
    load_env()
    from datasets import ClassLabel, load_dataset

    ds = load_dataset(repo, config, split=split, revision=revision, **kwargs)
    names = {k: f.names for k, f in ds.features.items() if isinstance(f, ClassLabel)}
    rows = []
    for row in ds:
        for key, labels in names.items():
            row[key + "_name"] = labels[row[key]] if row[key] is not None and row[key] >= 0 else None
        rows.append(row)
    return rows


def fetch(url: str, name: str | None = None) -> Path:
    """Download once into .cache/downloads and return the local path."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (name or url.rsplit("/", 1)[-1].replace("%20", "_"))
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "s1arena"})
        with urllib.request.urlopen(request, timeout=600) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp.rename(path)
    return path


def humanize(label: str) -> str:
    """'card_arrival' -> 'card arrival'; 'MeanOfTransportation' -> 'mean of transportation'."""
    label = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", label)
    return re.sub(r"[_\-.]+", " ", label).strip().lower()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
