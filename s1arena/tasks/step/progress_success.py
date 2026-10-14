"""Leaf: progress / eventual success (进度 / 最终能否成功).

A prefix of a recorded agent run is shown (the run is not over); will the run succeed? Gold is the
run's recorded outcome, decided by the benchmark's own checks: the task's tests (Terminal-Bench 2.0),
the environment assertions (tau2-bench) or the repository's tests (SWE-rebench). This is the
hard-label version: the soft label planned for this leaf (success share over repeated continuations
from the same prefix) needs runs from our own harness.

The prefix ends at a turn boundary after a third, a half or two thirds of the run's turns (chosen by
a hash), always before the agent claims completion. There is no length limit: each model is limited by its own context.
"""

from __future__ import annotations

import json

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from . import _runs as runs
from ._sources import longest_fitting

LEAF = "step/progress_success"
FRACTIONS = (1 / 3, 1 / 2, 2 / 3)


def cut(n: int, key: str) -> int:
    """Number of turns shown out of n: a third, a half or two thirds (at least 1, at most n - 1)."""
    fraction = FRACTIONS[runs.pick(key, len(FRACTIONS))]
    return min(n - 1, max(1, round(fraction * n)))


# ---------- Terminal-Bench 2.0 ----------

TB2_QUESTION = ("An AI agent (Terminus-2, driving a Linux terminal) is working on the task in the state; its first turns are shown "
                "and the run is not finished. The run succeeds if, when it ends, the task's tests pass. Will this run succeed?")


@register_task
class TB2Progress(TaskAdapter):
    name = f"{LEAF}/terminal_bench_2"
    leaf = LEAF
    source = "Terminal-Bench 2.0 leaderboard runs of the Terminus-2 agent with eight models (Merrill et al., 2026)"
    url = "https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard"
    license = "Apache-2.0"
    tier = "C"
    criteria_source = "Written from the Terminal-Bench success criterion (the task's tests pass)"
    notes = ("Runs of Terminus-2 with Claude Opus 4.6, GPT-5.3-Codex, GLM-4.7/5, Kimi K2.5, MiniMax M2.5, DeepSeek-V3.2 and "
             "GPT-OSS-20B on the 89 Terminal-Bench 2.0 tasks, five trials each (pinned leaderboard revision). Gold is the verifier's "
             "recorded reward. Only half of the tasks are used here; the other half serves step/completion. Runs that ended in an "
             "infrastructure error are dropped; agent timeouts count as failures. The agent's hidden reasoning is not shown.")

    def load(self):
        trials = [t for t in runs.tb2_trials() if runs.task_half(t["task"]) == 0 and t["exception"] in ("", "AgentTimeoutError")]
        rows = []
        for trial, trajectory in zip(trials, runs.tb2_trajectories(trials)):
            if not trajectory:
                continue
            steps = runs.tb2_agent_steps(trajectory)
            mark = next((i for i, s in enumerate(steps) if runs.tb2_marks(s)), len(steps))
            if mark < 2:
                continue
            k, task_text = cut(mark, trial["trial"]), runs.tb2_task(trajectory)
            blocks = runs.tb2_blocks(steps[:k])
            k = longest_fitting(blocks, k, task_text)
            if k < 1:
                continue
            rows.append({**trial, "task_text": task_text, "prefix": "\n\n".join(blocks[:k]), "shown": k, "turns_before_claim": mark})
        return rows

    def label_of(self, record):
        return f"{record['model']}|{record['reward']}"

    def to_item(self, record, rng):
        state = {"task": record["task_text"], "agent": f"Terminus-2 with {record['model']}", "turns_so_far": record["prefix"]}
        gold = Gold(label="true" if record["reward"] >= 1 else "false")
        return self.item(state, {"will_succeed": Question(type="boolean", instructions=TB2_QUESTION)}, {"will_succeed": gold},
                         task_name=record["task"], trial=record["trial"], model=record["model"], turns_shown=record["shown"],
                         turns_before_claim=record["turns_before_claim"])


# ---------- tau2-bench telecom ----------

TAU2_FILES = {
    "claude-3-7-sonnet": "claude-3-7-sonnet-20250219_telecom_default_gpt-4.1-2025-04-14_4trials.json",
    "gpt-4.1": "gpt-4.1-2025-04-14_telecom_default_gpt-4.1-2025-04-14_4trials.json",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14_telecom_base_gpt-4.1-2025-04-14_4trials.json",
    "o4-mini": "o4-mini-2025-04-16_telecom_default_gpt-4.1-2025-04-14_4trials.json",
}
TAU2_QUESTION = ("A customer-service agent, who must follow the telecom policy in the state, is helping a customer (a simulated user who "
                 "operates their own phone); the conversation so far is shown and it is not finished. The conversation succeeds if, "
                 "when it ends, the evaluator's checks pass: the customer's phone and account are in the state the task requires "
                 "(and the required actions were taken). Will this conversation succeed?")


@register_task
class Tau2Progress(TaskAdapter):
    name = f"{LEAF}/tau2_telecom"
    leaf = LEAF
    source = "tau2-bench telecom domain, published simulations of four agent models with a GPT-4.1 user simulator (Barres et al., 2025)"
    url = "https://github.com/sierra-research/tau2-bench"
    license = "MIT"
    tier = "C"
    criteria_source = "Written from the tau2-bench reward definition (environment assertions and required actions)"
    notes = ("Telecom technical-support tasks (a domain new in tau2-bench, not in tau-bench), four trials per task for Claude 3.7 "
             "Sonnet, GPT-4.1, GPT-4.1-mini and o4-mini agents. Gold is the recorded reward (1 = success), computed from environment "
             "assertions on the user's device and account plus required actions; no LLM-judged criteria are involved. The prefix "
             "ends just before an agent turn, at the same turn for all trials of a (model, task); trials whose conversations are "
             "identical up to the cut form one item, whose soft label is the success share of those continuations (the leaf's "
             "planned soft label, available where trials share a prefix) and whose hard label is the majority outcome. The user "
             "simulator's hidden instructions are not shown. Only half of the tasks are used here; the other half serves step/completion.")

    def load(self):
        from collections import defaultdict

        rows = []
        for model, name in TAU2_FILES.items():
            data = runs.tau2_file(name)
            policy = data["info"]["environment_info"]["policy"]
            by_task: dict[str, list] = defaultdict(list)
            for sim in data["simulations"]:
                info = sim.get("reward_info") or {}
                if runs.task_half(sim["task_id"]) != 0 or info.get("reward") not in (0, 0.0, 1, 1.0):
                    continue
                if "NL_ASSERTION" in (info.get("reward_basis") or []):
                    continue
                turns = [i for i, m in enumerate(sim["messages"]) if m.get("role") == "assistant"]
                if len(turns) >= 3:
                    by_task[sim["task_id"]].append((sim, turns))
            for task, sims in sorted(by_task.items()):
                # one cut per (model, task), so that trials whose conversations agree up to it share the prefix
                k = cut(min(len(t) for _, t in sims), f"{model}|{task}")
                groups: dict[str, list] = defaultdict(list)
                for sim, turns in sorted(sims, key=lambda x: x[0]["trial"]):
                    groups[runs.tau2_render(sim["messages"][:turns[k]])].append(sim)
                for text, members in groups.items():
                    outcomes = [float(m["reward_info"]["reward"]) for m in members]
                    rows.append({"model": model, "task": task, "trials": [m["trial"] for m in members], "policy": policy, "prefix": text,
                                 "outcomes": outcomes, "reward": outcomes[0], "shown": k})
        return rows

    def label_of(self, record):
        return f"{record['model']}|{record['reward']}"

    def to_item(self, record, rng):
        state = {"policy": record["policy"], "conversation_so_far": record["prefix"]}
        outcomes = record["outcomes"]
        share = sum(o >= 1 for o in outcomes) / len(outcomes)
        # several trials with this exact prefix: their success share is a soft label; the hard label is the majority
        # (a tie goes to the first trial's outcome)
        label = "true" if share > 0.5 or (share == 0.5 and outcomes[0] >= 1) else "false"
        gold = Gold(label=label, dist={"true": share, "false": 1 - share} if len(outcomes) > 1 else None)
        return self.item(state, {"will_succeed": Question(type="boolean", instructions=TAU2_QUESTION)}, {"will_succeed": gold},
                         task_id=record["task"], trials=record["trials"], agent_model=record["model"], agent_turns_shown=record["shown"])


# ---------- SWE-rebench ----------

SWR_PARTICIPANTS = ("claude-code", "codex", "cursor", "deepseek-v4-pro", "fable-5", "glm-5.2", "gpt-5.6-luna", "gpt-5.6-sol", "grok-4.5",
                    "mimo-v2.5-pro", "minimax-m3", "opus-5", "qwen3.5-35b-a3b", "qwen3.6-27b", "qwen3.6-35b-a3b", "sonnet-5")
SWR_RUNS = (0, 1)
SWR_QUESTION = ("A coding agent is fixing the GitHub issue in the state inside the repository; its first tool calls and their results "
                "are shown and the run is not finished. The run succeeds if the agent's final patch resolves the issue: the tests the "
                "maintainers wrote for the issue pass and the repository's existing tests still pass. Will this run succeed?")


@register_task
class SWERebenchProgress(TaskAdapter):
    name = f"{LEAF}/swe_rebench_2607"
    leaf = LEAF
    source = "SWE-rebench July 2026 leaderboard trajectories: 111 fresh issues, agentic systems and models in a common scaffold (Nebius, 2026)"
    url = "https://huggingface.co/datasets/ibragim-bad/swe_rebench_07_2026_trajectories"
    license = ("Other: the release does not relicense third-party content; task text and repository content stay under their "
               "repositories' licenses, agent outputs under the agents' and model providers' terms")
    tier = "C"
    criteria_source = "Written from the SWE-bench definition of a resolved task"
    notes = ("Runs 0 and 1 of 15 participants (Claude Code, Codex and 13 models in a common scaffold); Junie (no tool events) is "
             "left out. Gold is the recorded evaluation (resolved or not). The "
             "prefix ends after a tool result; long tool results were already truncated by the release. A different release from "
             "the SWE-rebench patch data used in code/patch_correctness (July 2026 tasks, other agents).")

    def load(self):
        rows = []
        for participant in SWR_PARTICIPANTS:
            for run in SWR_RUNS:
                for r in runs.swr_shard(participant, run):
                    evaluation = r.get("evaluation") or {}
                    if evaluation.get("status") not in ("resolved", "failed"):
                        continue
                    events = [e for e in r.get("events") or [] if e.get("type") != "prompt"]
                    results = [i for i, e in enumerate(events) if e.get("type") == "tool_result"]
                    if len(results) < 3:
                        continue
                    k, issue = cut(len(results), r["trajectory_id"]), runs.swr_task(r)
                    lines = runs.swr_lines(events[: results[k - 1] + 1])
                    n_lines = longest_fitting(lines, len(lines), issue)
                    k = sum(1 for i in results[:k] if i < n_lines)  # tool results whose line is inside the kept part
                    if k < 1:
                        continue
                    text = "\n\n".join(line for line in lines[: results[k - 1] + 1] if line)
                    rows.append({"id": r["trajectory_id"], "participant": participant, "run": run, "repo": r.get("repository"),
                                 "instance": r.get("instance_id"), "issue": issue, "prefix": text,
                                 "resolved": bool(evaluation.get("resolved")), "shown": k, "calls": len(results),
                                 "agent": json.dumps({k2: v for k2, v in (r.get("participant") or {}).items() if k2 in ("scaffold", "model")})})
        return rows

    def label_of(self, record):
        return f"{record['participant']}|{record['resolved']}"

    def to_item(self, record, rng):
        state = {"repository": record["repo"], "issue": record["issue"], "agent": record["agent"], "steps_so_far": record["prefix"]}
        gold = Gold(label="true" if record["resolved"] else "false")
        return self.item(state, {"will_succeed": Question(type="boolean", instructions=SWR_QUESTION)}, {"will_succeed": gold},
                         trajectory_id=record["id"], instance_id=record["instance"], tool_results_shown=record["shown"],
                         tool_results=record["calls"])
