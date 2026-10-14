"""Loaders and text renderings of recorded agent runs with outcomes, shared by step/progress_success
and step/completion: Terminal-Bench 2.0 leaderboard runs of the Terminus-2 agent, tau2-bench telecom
simulations, and SWE-rebench (July 2026) agent runs. Where two leaves read the same source, they use
disjoint halves of its tasks (split by a hash of the task name), so no task appears in both.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor

from ..common import fetch
from ._sources import hub_file


def task_half(task: str) -> int:
    """0 -> step/progress_success, 1 -> step/completion."""
    return int(hashlib.sha256(f"step-split:{task}".encode()).hexdigest(), 16) % 2


def pick(key: str, n: int) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % n


# ---------- Terminal-Bench 2.0 (Terminus-2 agent) ----------

TB2_REPO = "harborframework/terminal-bench-2-leaderboard"
TB2_REVISION = "572b2614be2c0cb2527e14f5b1e4026f1072e6c1"
TB2_ROOT = "submissions/terminal-bench/2.0"
_CONFIRM = re.compile(r"\s*Are you sure you want to mark the task as complete\?.*$", re.S)


def tb2_trials() -> list[dict]:
    """Every Terminus-2 trial of the pinned leaderboard revision with its reward, read from the job-level result files."""
    from huggingface_hub import HfApi

    api = HfApi()
    out = []
    for sub in sorted(x.path for x in api.list_repo_tree(TB2_REPO, path_in_repo=TB2_ROOT, repo_type="dataset", revision=TB2_REVISION)):
        name = sub.rsplit("/", 1)[-1]
        if not name.lower().startswith(("terminus2__", "terminus-2__")):
            continue
        for job in sorted(x.path for x in api.list_repo_tree(TB2_REPO, path_in_repo=sub, repo_type="dataset", revision=TB2_REVISION)):
            if job.endswith(".yaml"):
                continue
            with open(hub_file(TB2_REPO, f"{job}/result.json", TB2_REVISION), encoding="utf-8") as f:
                stats = json.load(f)["stats"]["evals"]
            for ev in stats.values():
                errors = {t: kind for kind, trials in (ev.get("exception_stats") or {}).items() for t in trials}
                for reward, trials in ((ev.get("reward_stats") or {}).get("reward") or {}).items():
                    for trial in trials:
                        out.append({"model": name.split("__", 1)[1], "job": job, "trial": trial, "task": trial.rsplit("__", 1)[0],
                                    "reward": float(reward), "exception": errors.get(trial, "")})
    return out


def tb2_trajectory(trial: dict) -> dict | None:
    """The trial's ATIF trajectory, or None if the submission has none for it; rate limits are retried with backoff."""
    import time

    from huggingface_hub.errors import EntryNotFoundError

    for attempt in range(6):
        try:
            with open(hub_file(TB2_REPO, f"{trial['job']}/{trial['trial']}/agent/trajectory.json", TB2_REVISION), encoding="utf-8") as f:
                return json.load(f)
        except EntryNotFoundError:
            return None
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** (attempt + 1))
    return None


def tb2_trajectories(trials: list[dict], workers: int = 8) -> list[dict | None]:
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(tb2_trajectory, trials))


def tb2_task(trajectory: dict) -> str:
    first = str(trajectory["steps"][0].get("message") or "")
    text = first.split("Task Description:", 1)[1] if "Task Description:" in first else first
    return text.split("Current terminal state:", 1)[0].strip()


def tb2_agent_steps(trajectory: dict) -> list[dict]:
    return [s for s in trajectory["steps"][1:] if s.get("source") == "agent"]


def tb2_marks(step: dict) -> bool:
    return any(c.get("function_name") == "mark_task_complete" for c in step.get("tool_calls") or [])


def tb2_render(steps: list[dict], drop_confirmation: bool = False) -> str:
    """Agent turns (analysis and plan, keystrokes, terminal output) as text."""
    return "\n\n".join(tb2_blocks(steps, drop_confirmation))


def tb2_blocks(steps: list[dict], drop_confirmation: bool = False) -> list[str]:
    blocks = []
    for n, s in enumerate(steps, 1):
        lines = [f"Turn {n}", str(s.get("message") or "").strip()]
        keys = [str((c.get("arguments") or {}).get("keystrokes", "")) for c in s.get("tool_calls") or [] if c.get("function_name") == "bash_command"]
        if keys:
            lines.append("Keystrokes sent:\n" + "".join(keys).rstrip())
        if tb2_marks(s):
            lines.append("(The agent marked the task as complete.)")
        output = "\n".join(str(r.get("content") or "") for r in (s.get("observation") or {}).get("results") or [])
        if drop_confirmation:
            output = _CONFIRM.sub("", output)
        if output.strip():
            lines.append("Terminal:\n" + output.strip())
        blocks.append("\n".join(lines))
    return blocks


# ---------- tau2-bench (telecom) ----------

TAU2_SHA = "b7ea9074c1cba482b30687fecdb5c8425fd6f619"  # commit of sierra-research/tau2-bench
TAU2_DATA = f"https://raw.githubusercontent.com/sierra-research/tau2-bench/{TAU2_SHA}/data/tau2/results/final"


def tau2_file(name: str) -> dict:
    with open(fetch(f"{TAU2_DATA}/{name}", f"tau2_{TAU2_SHA[:8]}_{name}"), encoding="utf-8") as f:
        return json.load(f)


def tau2_render(messages: list[dict]) -> str:
    """Conversation with the agent's and the user's tool calls and results (telecom users operate their own phone)."""
    lines = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role in ("assistant", "user"):
            who = "Agent" if role == "assistant" else "User"
            if content:
                lines.append(f"{who}: {content}")
            for call in m.get("tool_calls") or []:
                lines.append(f"{who} calls {call.get('name')}({json.dumps(call.get('arguments'), ensure_ascii=False)})")
        elif role == "tool":
            who = "user's" if m.get("requestor") == "user" else "agent's"
            lines.append(f"Result of the {who} tool call: {content}")
    return "\n\n".join(lines)


# ---------- SWE-rebench (July 2026) ----------

SWR_REPO = "ibragim-bad/swe_rebench_07_2026_trajectories"
SWR_REVISION = "cdae27cdd16673f0c682871ad55d325f24cc7020"


def swr_shard(participant: str, run: int) -> list[dict]:
    with gzip.open(hub_file(SWR_REPO, f"trajectories/{participant}/run_{run}.jsonl.gz", SWR_REVISION), "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def swr_task(record: dict) -> str:
    users = [p for p in record.get("prompts") or [] if p.get("role") == "user" and p.get("content")]
    return str(users[0]["content"]).strip() if users else ""


def swr_render(events: list[dict]) -> str:
    return "\n\n".join(line for line in swr_lines(events) if line)


def swr_lines(events: list[dict]) -> list[str]:
    """One text line per event (empty for events that are not shown), so line i belongs to event i."""
    return [_swr_line(e) for e in events]


def _swr_line(e: dict) -> str:
    kind = e.get("type")
    if kind == "tool_call":
        return f"Agent calls {e.get('tool')}: {json.dumps(e.get('input'), ensure_ascii=False)}"
    if kind == "tool_result":
        status = f" ({e.get('status')})" if e.get("status") and e.get("status") != "success" else ""
        return f"Result{status}: {e.get('content')}"
    if kind in ("assistant_message", "final_message") and e.get("content"):
        return f"Agent: {e.get('content')}"
    return ""


