"""Leaf: declaring completion (何时宣布完成).

The agent has just declared its task complete. Is the task actually complete? This tests
premature "done" declarations. Gold is the benchmark's own check of the final state, recorded with
the run: the task's tests (Terminal-Bench 2.0) or the environment assertions (tau2-bench). The runs
come from the halves of those benchmarks' tasks that step/progress_success does not use.
"""

from __future__ import annotations

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from . import _runs as runs
from ._sources import fits

LEAF = "step/completion"

# ---------- Terminal-Bench 2.0 ----------

TB2_QUESTION = ("An AI agent (Terminus-2, driving a Linux terminal) has worked on the task in the state and has just marked it as "
                "complete; all its turns are shown. The task is complete only if the task's tests pass on the final state of the "
                "machine. Is the task actually complete?")


@register_task
class TB2Completion(TaskAdapter):
    name = f"{LEAF}/terminal_bench_2"
    leaf = LEAF
    source = "Terminal-Bench 2.0 leaderboard runs of the Terminus-2 agent with eight models (Merrill et al., 2026)"
    url = "https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard"
    license = "Apache-2.0"
    tier = "C"
    criteria_source = "Written from the Terminal-Bench success criterion (the task's tests pass)"
    sampling = "balanced"  # premature claims are the point of the leaf
    notes = ("Terminus-2 asks the agent to confirm a completion claim ('Are you sure...?') and ends the run on a second claim. Items "
             "are runs whose last two turns are such a claim and its bare confirmation, so the machine was graded exactly as it was "
             "at the claim; the state is every turn up to the claim (the confirmation prompt is removed). Gold is the verifier's "
             "recorded reward. Only the half of the tasks not used by step/progress_success; runs with infrastructure errors are "
             "dropped. Sampling is balanced between complete and premature claims (about a third of the "
             "claims are premature).")

    def load(self):
        trials = [t for t in runs.tb2_trials() if runs.task_half(t["task"]) == 1 and t["exception"] == ""]
        rows = []
        for trial, trajectory in zip(trials, runs.tb2_trajectories(trials)):
            if not trajectory:
                continue
            steps = runs.tb2_agent_steps(trajectory)
            if len(steps) < 3 or not (runs.tb2_marks(steps[-1]) and runs.tb2_marks(steps[-2])):
                continue
            if any(c.get("function_name") == "bash_command" for c in steps[-1].get("tool_calls") or []):
                continue  # the confirmation turn itself changed the machine
            text, task_text = runs.tb2_render(steps[:-1], drop_confirmation=True), runs.tb2_task(trajectory)
            if not fits(text, task_text):
                continue
            rows.append({**trial, "task_text": task_text, "turns": text, "n_turns": len(steps) - 1})
        return rows

    def label_of(self, record):
        return str(record["reward"])

    def to_item(self, record, rng):
        state = {"task": record["task_text"], "agent": f"Terminus-2 with {record['model']}", "turns": record["turns"]}
        gold = Gold(label="true" if record["reward"] >= 1 else "false")
        return self.item(state, {"actually_complete": Question(type="boolean", instructions=TB2_QUESTION)}, {"actually_complete": gold},
                         task_name=record["task"], trial=record["trial"], model=record["model"], turns=record["n_turns"])


# ---------- tau2-bench telecom, solo mode ----------

TAU2_SOLO_FILES = {
    ("gpt-4.1", "telecom"): "gpt-4.1-2025-04-14_telecom_no-user_gpt-4.1-2025-04-14_4trials.json",
    ("gpt-4.1", "telecom-workflow"): "gpt-4.1-2025-04-14_telecom-workflow_no-user_gpt-4.1-2025-04-14_4trials.json",
    ("o4-mini", "telecom"): "o4-mini-2025-04-16_telecom_no-user_gpt-4.1-2025-04-14_4trials.json",
    ("o4-mini", "telecom-workflow"): "o4-mini-2025-04-16_telecom-workflow_no-user_gpt-4.1-2025-04-14_4trials.json",
}
TAU2_QUESTION = ("A telecom support agent works alone on the ticket in the state (there is no live user: it operates the customer's "
                 "phone and account with tools itself) under the policy given. Its tool calls and their results are shown; it has just "
                 "stopped, declaring the ticket handled. The task is complete only if the customer's phone and account are now in the "
                 "state the ticket requires (the evaluator's environment assertions pass, and required actions were taken). Is the task "
                 "actually complete?")


@register_task
class Tau2SoloCompletion(TaskAdapter):
    name = f"{LEAF}/tau2_telecom_solo"
    leaf = LEAF
    source = "tau2-bench telecom domain in solo mode (no user), published simulations of GPT-4.1 and o4-mini agents (Barres et al., 2025)"
    url = "https://github.com/sierra-research/tau2-bench"
    license = "MIT"
    tier = "C"
    criteria_source = "Written from the tau2-bench reward definition (environment assertions and required actions)"
    sampling = "balanced"  # premature claims are the point of the leaf, and only about one claim in eight is premature here
    notes = ("In solo mode the agent receives a ticket and acts on the phone itself, ending the run with a stop message. Runs that "
             "end in a transfer to a human agent are not completion claims and are dropped. Gold is the recorded reward (1 = the "
             "environment assertions pass). Both the default and the workflow-policy variants of the domain are used, four trials "
             "per task; only the half of the tasks not used by step/progress_success. Identical repeated runs are kept once. Sampling is "
             "balanced between complete and premature claims (about one claim in eight is premature).")

    def load(self):
        rows, seen = [], set()
        for (model, variant), name in TAU2_SOLO_FILES.items():
            data = runs.tau2_file(name)
            policy = data["info"]["environment_info"]["policy"]
            tickets = {t["id"]: t.get("ticket") or "" for t in data.get("tasks") or []}
            for sim in data["simulations"]:
                reward = (sim.get("reward_info") or {}).get("reward")
                if runs.task_half(sim["task_id"]) != 1 or reward not in (0, 0.0, 1, 1.0) or sim.get("termination_reason") != "agent_stop":
                    continue
                calls = [c.get("name") for m in sim["messages"] if m.get("role") == "assistant" for c in m.get("tool_calls") or []]
                if not calls or "transfer_to_human_agents" in calls or not tickets.get(sim["task_id"]):
                    continue
                actions = runs.tau2_render(sim["messages"])
                key = (model, variant, sim["task_id"], actions)
                if key in seen:  # temperature-0 agents often repeat a run exactly
                    continue
                seen.add(key)
                rows.append({"model": model, "variant": variant, "task": sim["task_id"], "trial": sim["trial"], "policy": policy,
                             "ticket": tickets[sim["task_id"]], "actions": actions, "reward": float(reward)})
        return rows

    def label_of(self, record):
        return str(record["reward"])

    def to_item(self, record, rng):
        state = {"policy": record["policy"], "ticket": record["ticket"], "agent_actions": record["actions"]}
        gold = Gold(label="true" if record["reward"] >= 1 else "false")
        return self.item(state, {"actually_complete": Question(type="boolean", instructions=TAU2_QUESTION)}, {"actually_complete": gold},
                         task_id=record["task"], trial=record["trial"], agent_model=record["model"], policy_variant=record["variant"])
