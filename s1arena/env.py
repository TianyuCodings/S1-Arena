"""Paths, settings and secrets. Secrets come from .env (never committed).

The working folder holds data/, data_local/, results/ and .env. It is the repository root when s1arena runs from a
checkout, else the current folder; set S1ARENA_HOME to use another folder.
"""

from __future__ import annotations

import os
from pathlib import Path


def _root() -> Path:
    if os.environ.get("S1ARENA_HOME"):
        return Path(os.environ["S1ARENA_HOME"]).expanduser().resolve()
    checkout = Path(__file__).resolve().parent.parent
    return checkout if (checkout / "pyproject.toml").exists() else Path.cwd()


ROOT = _root()
DATA = ROOT / "data"
DATA_LOCAL = ROOT / "data_local"  # built requests from sources that do not allow redistribution (never shared)
RUNS = ROOT / "results" / "runs"  # raw model responses
SUMMARY = ROOT / "results" / "summary"
# The Hugging Face dataset that holds the built requests of every redistributable source (`s1arena download`).
HF_DATASET = os.environ.get("S1ARENA_HF_DATASET", "C-Tianyu/S1-Arena")
# A private dataset with the requests of the sources that do not allow redistribution, for the people who have access;
# everybody else rebuilds them with `s1arena build --local`.
HF_DATASET_RESTRICTED = os.environ.get("S1ARENA_HF_DATASET_RESTRICTED", "C-Tianyu/S1-Arena-restricted")

_loaded = False


def load_env(path: Path | None = None) -> None:
    """Read KEY=VALUE lines into os.environ without overriding existing variables."""
    global _loaded
    if _loaded:
        return
    path = path or Path(os.environ.get("S1ARENA_ENV_FILE", ROOT / ".env"))
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    if "HF_API_KEY" in os.environ:
        os.environ.setdefault("HF_TOKEN", os.environ["HF_API_KEY"])
    _loaded = True
