"""Sweep for environments that play each seed in its own Docker image (e.g. loop/coding_controller).

An environment takes part when its class has `image(seed) -> str`. The SWE-bench images are 2.5-7 GB each, so a
sweep over hundreds of instances cannot keep them all on disk. The Docker sweep therefore goes seed by seed:

1. pull the seed's image (unless it is already present),
2. play that seed for every model of the plan (models in parallel, through the same cached runner as `play`),
3. remove the image again if this sweep pulled it (`keep_images = true` keeps it),

while the images of the next seeds are pulled in the background (`prefetch`). A seed whose episodes are all cached
for every model is skipped without pulling anything, so the sweep resumes where it stopped. Models stop on gateway
account refusals and on their budget as in s1arena/sweep.py.

Plan keys (in the sweep plan, optional): [docker] parallel_seeds (seeds played at once, default 1; every episode
runs its own container with up to 8 GB), prefetch (images pulled ahead, default 1), keep_images (default false),
min_free_gb (do not pull below this much free space in Docker's root directory, default 30).
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
import tomllib
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .env import load_env
from .envs import runner as env_runner
from .registry import make_model
from .sweep import Plan, _Log, _spent, load_plan, plan_envs


def docker_envs(plan: Plan) -> list[type]:
    return [cls for cls in plan_envs(plan) if hasattr(cls, "image")]


def _docker(*args: str, timeout: float = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, errors="replace", timeout=timeout)


def docker_available() -> str | None:
    """None if the Docker daemon answers, else the reason it does not."""
    try:
        out = _docker("info", "--format", "{{.ServerVersion}}", timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"{type(error).__name__}: {error}"
    return None if out.returncode == 0 else out.stderr.strip()[-300:]


def has_image(image: str) -> bool:
    return _docker("image", "inspect", image, timeout=60).returncode == 0


def free_gb() -> float | None:
    out = _docker("info", "--format", "{{.DockerRootDir}}", timeout=30)
    root = out.stdout.strip()
    try:
        return shutil.disk_usage(root).free / 1e9 if out.returncode == 0 and root else None
    except OSError:  # e.g. Docker Desktop, whose root is inside its VM
        return None


class Images:
    """Pulls images ahead of use and removes the ones this sweep pulled once they are done."""

    def __init__(self, prefetch: int, keep: bool, min_free_gb: float, log) -> None:
        self.pool = ThreadPoolExecutor(max_workers=max(1, prefetch))
        self.pending: dict[str, Future] = {}
        self.pulled: set[str] = set()
        self.keep, self.min_free_gb, self.log = keep, min_free_gb, log
        self.lock = threading.Lock()

    def _pull(self, image: str) -> str | None:
        if has_image(image):
            return None
        free = free_gb()
        if free is not None and free < self.min_free_gb:
            return f"only {free:.0f} GB free in Docker's root directory (min_free_gb = {self.min_free_gb:.0f})"
        started = time.time()
        for attempt in range(3):
            out = _docker("pull", "--quiet", image)
            if out.returncode == 0:
                with self.lock:
                    self.pulled.add(image)
                self.log(f"pulled {image} in {time.time() - started:.0f}s")
                return None
            time.sleep(10 * (attempt + 1))
        return f"pull failed: {out.stderr.strip()[-300:]}"

    def request(self, image: str) -> None:
        with self.lock:
            if image not in self.pending:
                self.pending[image] = self.pool.submit(self._pull, image)

    def wait(self, image: str) -> str | None:
        self.request(image)
        return self.pending[image].result()

    def release(self, image: str) -> None:
        with self.lock:
            self.pending.pop(image, None)
            ours = image in self.pulled and not self.keep
            self.pulled.discard(image)
        if ours:
            _docker("rmi", image, timeout=600)

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        for image in list(self.pulled):
            self.release(image)


class ModelState:
    def __init__(self, plan: Plan, mp, envs: list[type]) -> None:
        self.mp = mp
        self.model = make_model(mp.spec)
        self.names = [cls.name for cls in envs]
        self.start_spend = _spent(self.model.slug, self.names, episodes=True)
        self.stopped: str | None = None
        self.episodes = 0

    def over_budget(self) -> bool:
        if self.mp.budget_usd is None or self.stopped:
            return bool(self.stopped)
        spent = _spent(self.model.slug, self.names, episodes=True) - self.start_spend
        if spent >= self.mp.budget_usd:
            self.stopped = f"budget reached (${spent:.2f} of ${self.mp.budget_usd:.2f})"
        return bool(self.stopped)


def _cached(plan: Plan, state: ModelState, cls, seed: int) -> bool:
    return any(r["seed"] == seed and "outcome" in r
               for r in env_runner.latest_episodes(state.model.slug, cls.name, "paused", plan.cap(cls), None, cls.version))


def run(plan_path: str | Path, model_specs: list[str] | None = None) -> list[dict]:
    load_env()
    plan = load_plan(plan_path)
    options = tomllib.loads(Path(plan_path).read_text(encoding="utf-8")).get("docker", {})
    if model_specs:
        plan.models = [m for m in plan.models if m.spec in model_specs]
    plan.models = [m for m in plan.models if m.envs]
    log = _Log("docker")
    reason = docker_available()
    if reason:
        log(f"Docker is not available here, nothing to do ({reason})")
        return []
    envs = docker_envs(plan)
    if not envs:
        log("no Docker environments in this plan")
        return []
    states = [ModelState(plan, mp, envs) for mp in plan.models]
    images = Images(options.get("prefetch", 1), options.get("keep_images", False), options.get("min_free_gb", 30), log)
    parallel = max(1, options.get("parallel_seeds", 1))

    # every (environment, seed) that some model still needs, in order
    work = [(cls, seed) for cls in envs for seed in plan.seeds(cls)]

    def todo(cls, seed) -> list[ModelState]:
        return [s for s in states if not s.over_budget() and not _cached(plan, s, cls, seed)]

    def play_seed(cls, seed: int) -> None:
        players = todo(cls, seed)
        if not players:
            return
        image = cls.image(seed)
        problem = images.wait(image)
        if problem:
            log(f"{cls.name} seed {seed}: skipped, {problem}")
            return
        try:
            with ThreadPoolExecutor(max_workers=len(players)) as pool:
                futures = {pool.submit(env_runner.run_env, s.model, cls, [seed], "paused", 0.0, 1, plan.max_decisions or None): s
                           for s in players}
                for future, state in futures.items():
                    state.episodes += future.result()
            for state in players:  # an episode missing after the run was refused by the gateway (not recorded)
                if not _cached(plan, state, cls, seed) and not state.stopped:
                    rows = env_runner.read_episodes(state.model.slug, cls.name)
                    if not any(r.get("seed") == seed for r in rows):
                        state.stopped = "gateway refused during an episode"
        finally:
            images.release(image)

    started = time.time()
    try:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            running: list[Future] = []
            for i, (cls, seed) in enumerate(work):
                if not todo(cls, seed):
                    continue
                for ahead_cls, ahead_seed in work[i + 1: i + 1 + options.get("prefetch", 1)]:
                    if todo(ahead_cls, ahead_seed):
                        images.request(ahead_cls.image(ahead_seed))
                running.append(pool.submit(play_seed, cls, seed))
                if len(running) >= parallel:
                    running.pop(0).result()
            for future in running:
                future.result()
    finally:
        images.close()
    log(f"finished in {time.time() - started:.0f}s")
    reports = [{"model": s.model.model_id, "episodes": s.episodes, "stopped": s.stopped} for s in states]
    for r in reports:
        log(f"{r['model']:<40} episodes={r['episodes']:<5} {r['stopped'] or 'complete'}")
    return reports
