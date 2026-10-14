"""Plugin registries. Adding a task or a model means adding one module; nothing else changes.

Tasks live under s1arena/tasks/, closed-loop environments under s1arena/envs/, models under
s1arena/models/. Every module in those packages is imported on discovery, so a decorated class
registers itself.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

TASKS: dict[str, type] = {}
MODELS: dict[str, type] = {}
ENVS: dict[str, type] = {}
_discovered = False


def register_task(cls: type) -> type:
    if not cls.name or cls.name in TASKS:
        raise ValueError(f"task name missing or duplicated: {cls.name!r}")
    TASKS[cls.name] = cls
    return cls


def register_env(cls: type) -> type:
    if not cls.name or cls.name in ENVS:
        raise ValueError(f"environment name missing or duplicated: {cls.name!r}")
    ENVS[cls.name] = cls
    return cls


def register_model(cls: type) -> type:
    if not cls.kind or cls.kind in MODELS:
        raise ValueError(f"model kind missing or duplicated: {cls.kind!r}")
    MODELS[cls.kind] = cls
    return cls


def discover() -> None:
    global _discovered
    if _discovered:
        return
    import s1arena.envs
    import s1arena.models
    import s1arena.tasks

    for package in (s1arena.tasks, s1arena.envs, s1arena.models):
        for info in pkgutil.walk_packages(package.__path__, package.__name__ + ".", onerror=lambda name: None):
            try:
                importlib.import_module(info.name)
            except Exception as error:  # one broken module must not hide every other task
                print(f"warning: skipped {info.name} ({type(error).__name__}: {error})", file=sys.stderr)
    _discovered = True


def envs_under(prefix: str) -> list[type]:
    """Environment classes whose name equals the prefix or starts with it."""
    discover()
    prefix = prefix.strip("/")
    return [ENVS[n] for n in sorted(ENVS) if not prefix or n == prefix or n.startswith(prefix + "/")]


def tasks_under(prefix: str) -> list:
    """Instantiate every task whose name equals the prefix or starts with it."""
    discover()
    prefix = prefix.strip("/")
    return [TASKS[n]() for n in sorted(TASKS) if not prefix or n == prefix or n.startswith(prefix + "/")]


def make_model(spec: str):
    """Build a model from a spec such as 'jev', 'llm:openai/gpt-4.1-mini' or 'random'."""
    discover()
    kind, _, arg = spec.partition(":")
    if kind not in MODELS:
        raise KeyError(f"unknown model kind {kind!r}; known: {sorted(MODELS)}")
    return MODELS[kind](arg) if arg else MODELS[kind]()
