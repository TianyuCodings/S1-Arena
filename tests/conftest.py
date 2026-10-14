"""Every test runs in a temporary working folder, so no test reads or writes the real data/ or results/."""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

os.environ["S1ARENA_HOME"] = tempfile.mkdtemp(prefix="s1arena-test-")  # before s1arena.env is imported
os.environ.setdefault("S1ARENA_ENV_FILE", os.path.join(os.environ["S1ARENA_HOME"], ".env"))

import pytest  # noqa: E402

TEMPLATES = Path(__file__).resolve().parent.parent / "templates"


def load_template(name: str):
    """Import a file of templates/ once (it registers its class on import)."""
    module = f"s1arena_template_{name}"
    if module not in sys.modules:
        spec = importlib.util.spec_from_file_location(module, TEMPLATES / f"{name}.py")
        sys.modules[module] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[module])
    return sys.modules[module]


@pytest.fixture
def template():
    return load_template
