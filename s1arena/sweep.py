"""Run every model on every static task and closed-loop environment of a sweep plan, and report coverage.

A plan is a TOML file (see sweeps/full.toml). All work goes through the cached runners (runner.run_task and
envs.runner.run_env), so a sweep can be stopped and restarted at any time and only fills in what is missing.

Per model the sweep
- runs static tasks first (realtime environments charge the model's median latency from its static runs),
- stops that model when the gateway refuses calls for account reasons (e.g. HTTP 402, an empty balance),
- stops that model once its spend in this sweep reaches its budget_usd (checked within each static task and between environments).
Models run in parallel, one thread each; the other models keep going when one stops.
"""

from __future__ import annotations

import json
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import dataio, runner
from .env import SUMMARY, load_env
from .envs import runner as env_runner
from .registry import envs_under, make_model, tasks_under


@dataclass
class ModelPlan:
    spec: str
    k: int = 1
    workers: int = 8
    env_workers: int = 4
    budget_usd: float | None = None
    static: bool = True
    envs: bool = True
    latency_ms: float | None = None  # realtime latency; default: the model's median over its static runs


@dataclass
class Plan:
    name: str
    tasks: list[str] = field(default_factory=lambda: [""])
    envs: list[str] = field(default_factory=lambda: [""])
    models: list[ModelPlan] = field(default_factory=list)
    limit: int = 0  # first N items per task (0 = all)
    episodes: int = 0  # seeds 0..N-1 per environment (0 = the environment's full set)
    max_decisions: int = 0  # decision cap per episode (0 = the environment's own)
    shard: tuple[int, int] = (0, 1)  # (index, count): this worker takes every count-th task and environment

    def seeds(self, cls) -> list[int]:
        return list(range(min(self.episodes, cls.episodes) if self.episodes else cls.episodes))

    def cap(self, cls) -> int:
        return self.max_decisions or cls.max_decisions


def load_plan(path: str | Path) -> Plan:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {})
    models = [ModelPlan(**{**defaults, **m}) for m in raw.get("models", [])]
    return Plan(name=raw.get("name", Path(path).stem), tasks=raw.get("tasks", [""]), envs=raw.get("envs", [""]),
                models=models, limit=raw.get("limit", 0), episodes=raw.get("episodes", 0), max_decisions=raw.get("max_decisions", 0))


def _unique(groups):
    seen = {}
    for group in groups:
        for x in group:
            seen.setdefault(x.name, x)
    return list(seen.values())


def plan_tasks(plan: Plan) -> list:
    index, count = plan.shard
    return _unique(tasks_under(prefix) for prefix in plan.tasks)[index::count]


def plan_envs(plan: Plan) -> list[type]:
    index, count = plan.shard
    return _unique(envs_under(prefix) for prefix in plan.envs)[index::count]


def _spent(model_slug: str, names: list[str], episodes: bool = False) -> float:
    total = 0.0
    for name in names:
        rows = env_runner.read_episodes(model_slug, name) if episodes else runner.read_runs(model_slug, name)
        for row in rows:
            cost = row.get("cost_usd") if episodes else row["response"].get("cost_usd")
            total += cost or 0.0
    return total


def _account_refused(model_slug: str, task: str) -> str | None:
    for row in runner.latest_rows(model_slug, task):
        error = row["response"].get("error")
        if error and env_runner.ACCOUNT_ERROR.search(error):
            return error[:200]
    return None


def _env_mode_latency(model, cls, latency_ms: float | None = None) -> tuple[str, float] | None:
    if cls.mode != "realtime":
        return "paused", 0.0
    if latency_ms is not None:
        return "realtime", latency_ms
    measured = SUMMARY / f"latency_{model.slug}.json"  # local models: single-request latency from run_lm_batch.py
    if measured.exists():  # (latencies in their batched runs are amortized over the batch)
        median = json.loads(measured.read_text(encoding="utf-8")).get("median_ms")
        if median:
            return "realtime", round(float(median), -1)
    try:
        return "realtime", env_runner.model_latency_ms(model)
    except ValueError:  # no static runs yet to fix the latency from
        return None


class _Log:
    lock = threading.Lock()

    def __init__(self, model_id: str):
        self.model_id = model_id

    def __call__(self, message: str) -> None:
        with self.lock:
            print(f"[{self.model_id}] {message}", flush=True)


def run_model(plan: Plan, mp: ModelPlan) -> dict:
    model = make_model(mp.spec)
    log = _Log(model.model_id)
    tasks, envs = plan_tasks(plan), plan_envs(plan)
    task_names, env_names = [t.name for t in tasks], [c.name for c in envs]
    start_spend = _spent(model.slug, task_names) + _spent(model.slug, env_names, episodes=True)
    report = {"model": model.model_id, "calls": 0, "episodes": 0, "stopped": None}

    def remaining() -> float | None:
        if mp.budget_usd is None:
            return None
        spent = _spent(model.slug, task_names) + _spent(model.slug, env_names, episodes=True) - start_spend
        return mp.budget_usd - spent

    def over_budget() -> bool:
        left = remaining()
        if left is not None and left <= 0:
            spent = mp.budget_usd - left
            report["stopped"] = f"budget reached (${spent:.2f} of ${mp.budget_usd:.2f})"
            return True
        return False

    if mp.static and hasattr(model, "decide_many"):  # local models: all tasks' calls pooled into shared batches
        loaded = [(t.name, dataio.load_task(t)[: plan.limit or None]) for t in tasks]
        calls = runner.run_tasks_batched(model, [(n, items) for n, items in loaded if items], k=mp.k)
        report["calls"] += sum(calls.values())
        for name, n in calls.items():
            log(f"{name}: {n} calls")
    elif mp.static:
        for task in tasks:
            if over_budget():
                break
            items = dataio.load_task(task)[: plan.limit or None]
            if not items:
                continue
            calls = runner.run_task(model, task.name, items, k=mp.k, workers=mp.workers, budget_usd=remaining())
            report["calls"] += calls
            if calls:
                log(f"{task.name}: {calls} calls")
                refused = _account_refused(model.slug, task.name)
                if refused:
                    report["stopped"] = f"gateway refused: {refused}"
                    break
    if mp.envs and not report["stopped"]:
        for cls in envs:
            if over_budget():
                break
            if hasattr(cls, "image"):  # one Docker image per seed: played by `s1arena sweep-docker`
                log(f"{cls.name}: skipped, Docker images per seed (run `s1arena sweep-docker`)")
                continue
            mode_latency = _env_mode_latency(model, cls, mp.latency_ms)
            if mode_latency is None:
                log(f"{cls.name}: skipped, no static runs to fix a realtime latency")
                continue
            mode, latency = mode_latency
            try:  # a simulator missing on this machine is skipped, not written into the cache as crashed episodes
                probe = cls()
                try:
                    probe.reset(0)
                finally:
                    probe.close()
            except Exception as error:
                log(f"{cls.name}: skipped, cannot start here ({type(error).__name__}: {str(error)[:150]})")
                continue
            try:
                played = env_runner.run_env(model, cls, plan.seeds(cls), mode, latency, workers=mp.env_workers,
                                            max_decisions=plan.max_decisions or None)
            except Exception as error:  # e.g. a simulator not installed on this machine; the other environments go on
                log(f"{cls.name}: skipped ({type(error).__name__}: {str(error)[:200]})")
                continue
            report["episodes"] += played
            if played:
                log(f"{cls.name}: {played} episodes")
                done = {r["seed"] for r in env_runner.latest_episodes(model.slug, cls.name, mode, plan.cap(cls),
                                                                     latency if mode == "realtime" else None, cls.version)}
                # run_env does not record refused episodes, so a seed still missing right after a run means the
                # gateway refused (or the episode crashed); one probe call tells the two apart
                missing = len(done & set(plan.seeds(cls))) < len(plan.seeds(cls))
                if missing and _refused(model, cls):
                    report["stopped"] = "gateway refused during episodes"
                    break
    log(f"done: {report['calls']} calls, {report['episodes']} episodes" + (f", stopped: {report['stopped']}" if report["stopped"] else ""))
    return report


def _refused(model, cls) -> bool:
    """One real call on the environment's first request; True if the gateway refuses it for account reasons."""
    env = cls()
    try:
        env.reset(0)
        request = env.observe_agent(list(env.active_agents())[0]) if env.agents else env.observe()
    except Exception:
        return False
    finally:
        env.close()
    error = model.decide(request, 0).error
    return bool(error and env_runner.ACCOUNT_ERROR.search(error))


def run(plan: Plan, parallel_models: int = 0) -> list[dict]:
    load_env()
    started = time.time()
    with ThreadPoolExecutor(max_workers=parallel_models or max(1, len(plan.models))) as pool:
        reports = list(pool.map(lambda mp: run_model(plan, mp), plan.models))
    print(f"sweep {plan.name}: finished in {time.time() - started:.0f}s")
    for r in reports:
        print(f"  {r['model']:<40} calls={r['calls']:<7} episodes={r['episodes']:<5} {r['stopped'] or 'complete'}")
    return reports


def status(plan: Plan) -> dict:
    """Coverage per model and domain: final static calls / planned calls, finished episodes / planned episodes."""
    tasks, envs = plan_tasks(plan), plan_envs(plan)
    items = {t.name: [i.id for i in dataio.load_task(t)[: plan.limit or None]] for t in tasks}
    table: dict[str, dict[str, list[int]]] = {}
    for mp in plan.models:
        model = make_model(mp.spec)
        cells = table.setdefault(model.model_id, {})
        if mp.static:
            for task in tasks:
                ids = set(items[task.name])
                done = sum(1 for r in runner.latest_rows(model.slug, task.name)
                           if r["item_id"] in ids and r["sample"] < mp.k and runner.is_final(r))
                cell = cells.setdefault(task.name.split("/")[0], [0, 0])
                cell[0] += done
                cell[1] += len(ids) * mp.k
        if mp.envs:
            for cls in envs:
                mode_latency = _env_mode_latency(model, cls, mp.latency_ms)
                seeds = set(plan.seeds(cls))
                done = 0
                if mode_latency:
                    mode, latency = mode_latency
                    done = sum(1 for r in env_runner.latest_episodes(model.slug, cls.name, mode, plan.cap(cls),
                                                                     latency if mode == "realtime" else None, cls.version)
                               if r["seed"] in seeds and "outcome" in r)
                cell = cells.setdefault("env:" + cls.name.split("/")[0], [0, 0])
                cell[0] += done
                cell[1] += len(seeds)
    return table


def print_status(table: dict) -> None:
    groups = sorted({g for cells in table.values() for g in cells})
    names = list(table)
    print(f"{'':<22}" + "".join(f"{n.split('/')[-1][:20]:>22}" for n in names))
    for g in groups:
        row = []
        for n in names:
            done, total = table[n].get(g, (0, 0))
            row.append(f"{done}/{total} ({done / total:.0%})" if total else "-")
        print(f"{g:<22}" + "".join(f"{c:>22}" for c in row))
    for n in names:
        done = sum(c[0] for c in table[n].values())
        total = sum(c[1] for c in table[n].values())
        print(f"{n}: {done}/{total} ({done / total:.0%})" if total else f"{n}: nothing planned")


def write_status(plan: Plan, table: dict) -> Path:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    path = SUMMARY / f"sweep_{plan.name}_status.json"
    path.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    return path


def report(plan: Plan) -> dict:
    """Results of a plan from the caches (no calls): static accuracy per leaf (mean over its tasks) and overall
    (mean over leaves, so leaves with more sources do not weigh more), and per-environment scores."""
    from . import metrics

    tasks, envs = plan_tasks(plan), plan_envs(plan)
    models = [(mp, make_model(mp.spec)) for mp in plan.models]
    static: dict[str, dict[str, float]] = {}  # leaf -> model -> accuracy
    asked: dict[str, list[int]] = {}  # model -> [answers asked for (question x sample), those without an answer]
    for task in tasks:
        items = dataio.load_task(task)[: plan.limit or None]
        keep = {i.id for i in items}
        per_model: dict[str, list[float]] = {}
        for mp, model in models:
            if not mp.static:
                continue
            rows = [r for r in runner.latest_rows(model.slug, task.name) if r["item_id"] in keep and r["sample"] < mp.k]
            if rows:
                ids = {r["item_id"] for r in rows}
                res = metrics.task_metrics(task, [i for i in items if i.id in ids], rows)
                if res.get("n_questions"):
                    static.setdefault(task.leaf, {}).setdefault(model.model_id, []).append(res["accuracy"])
                    cell = asked.setdefault(model.model_id, [0, 0])
                    cell[0] += res["answers"]
                    cell[1] += res["unanswered"]
    static = {leaf: {m: sum(v) / len(v) for m, v in cell.items()} for leaf, cell in static.items()}
    envs_table: dict[str, dict[str, dict]] = {}  # env -> model -> summary
    for cls in envs:
        for mp, model in models:
            if not mp.envs:
                continue
            mode_latency = _env_mode_latency(model, cls, mp.latency_ms)
            if not mode_latency:
                continue
            mode, latency = mode_latency
            seeds = set(plan.seeds(cls))
            rows = [r for r in env_runner.latest_episodes(model.slug, cls.name, mode, plan.cap(cls),
                                                          latency if mode == "realtime" else None, cls.version)
                    if r["seed"] in seeds]
            summary = env_runner.summarize(rows)
            if summary.get("episodes"):
                envs_table.setdefault(cls.name, {})[model.model_id] = {"mode": mode, "score_name": cls.score_name, **summary}
    overall = {}
    for _, model in models:
        leaves = [cell[model.model_id] for cell in static.values() if model.model_id in cell]
        played = [cell[model.model_id] for cell in envs_table.values() if model.model_id in cell]
        decisions = [s["decisions_mean"] * s["episodes"] for s in played]
        failed = [(s["failed_decision_rate"] or 0.0) * d for s, d in zip(played, decisions)]
        answers, unanswered = asked.get(model.model_id, [0, 0])
        overall[model.model_id] = {"static_accuracy": sum(leaves) / len(leaves) if leaves else None, "leaves": len(leaves),
                                   "envs": len(played),  # env scores have different scales: no cross-env mean
                                   "static_answers": answers, "static_unanswered": unanswered,
                                   "env_decisions": round(sum(decisions)), "env_failed_decisions": round(sum(failed))}
    return {"plan": plan.name, "models": [m.model_id for _, m in models], "overall": overall, "static": static, "envs": envs_table}


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.3f}"


def _share(part: int, whole: int) -> str:
    return "-" if not whole else f"{part:,} of {whole:,} ({part / whole:.1%})"


def report_markdown(rep: dict) -> str:
    names = rep["models"]
    head = lambda first: [f"| {first} | " + " | ".join(f"`{n}`" for n in names) + " |", "|---" * (len(names) + 1) + "|"]
    lines = [f"# Sweep `{rep['plan']}`", "", "Generated by `s1arena sweep-report` from cached results; `-` = not run.", "",
             "A static question without an answer scores 0: the call failed, the reply could not be used, or the model "
             "refused the input (for example as longer than its context). An environment decision without an answer "
             "takes the environment's default action.", "",
             "## Overall", "",
             "| Model | Static accuracy (mean over leaves) | Static questions without an answer | Leaves "
             "| Environments played | Environment decisions without an answer |",
             "|---|---|---|---|---|---|"]
    for n in names:
        o = rep["overall"][n]
        lines.append(f"| `{n}` | {_fmt(o['static_accuracy'])} | {_share(o['static_unanswered'], o['static_answers'])} | {o['leaves']} "
                     f"| {o['envs']} | {_share(o['env_failed_decisions'], o['env_decisions'])} |")
    lines += ["", "## Static accuracy per leaf", "", *head("Leaf")]
    for leaf in sorted(rep["static"]):
        lines.append(f"| {leaf} | " + " | ".join(_fmt(rep["static"][leaf].get(n)) for n in names) + " |")
    lines += ["", "## Environments (mean score; success rate in brackets)", "", *head("Environment")]
    for env in sorted(rep["envs"]):
        cells = []
        for n in names:
            s = rep["envs"][env].get(n)
            cells.append("-" if not s else _fmt(s["score_mean"]) + (f" ({s['success_rate']:.2f})" if s.get("success_rate") is not None else ""))
        lines.append(f"| {env} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(plan: Plan, rep: dict) -> tuple[Path, Path]:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    js, md = SUMMARY / f"sweep_{plan.name}_report.json", SUMMARY / f"sweep_{plan.name}_report.md"
    js.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    md.write_text(report_markdown(rep), encoding="utf-8")
    return js, md
