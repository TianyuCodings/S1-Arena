"""Leaf: agent trajectory judge (Agent 轨迹判官).

AgentRewardBench: expert annotations of web-agent trajectories (success, side effects,
optimality, looping). tau-bench: customer-service agent conversations whose success was
decided by execution (the final database state and required outputs are compared with
the ground truth).
"""

from __future__ import annotations

import csv
import json

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch

LEAF = "judge/agent_trajectory"
ARB = "McGill-NLP/agent-reward-bench"
ARB_JUDGE = "gpt-4o-mini-noscreen"  # a text-only judge configuration (accessibility tree, no screenshots): its prompt is our state
ARB_CONTEXT = ("You are given the goal of a web navigation task, the actions taken by the agent with the URL and the agent's "
               "reasoning at each step, and the accessibility tree of the last page (no screenshots). ")
# AgentRewardBench judge prompt questions (and the annotators' questions), verbatim.
ARB_QUESTIONS = {
    "success": ("trajectory_success", "Was the sequence of actions below successful in achieving the goal?", {"Successful": "true", "Unsuccessful": "false"}),
    "side_effect": ("trajectory_side_effect", "Did the agent perform unnecessary actions that could lead to unintended side effects?", {"Yes": "true", "No": "false"}),
    "looping": ("trajectory_looping", "Did the agent loop through a sequence of actions that did not make progress towards the goal?", {"Yes": "true", "No": "false"}),
}
ARB_OPTIMALITY = ["1. Complete Failure", "2. Suboptimal", "3. Somewhat Optimal", "4. Completely Optimal"]
# Terms of use from the AgentRewardBench dataset card, verbatim; they require derivatives to carry them, so every item does.
ARB_TERMS = (
    "By downloading this Dataset, you agree to comply with the following terms of use: - Restrictions: You agree not to use "
    "the Dataset in any way that is unlawful or would infringe upon the rights of others. - Acknowledgment: By using the "
    "Dataset, you acknowledge that the Dataset may contain data derived from third-party sources, and you agree to abide by "
    "any additional terms and conditions that may apply to such third-party data. - Fair Use Declaration: The Dataset may be "
    "used for research if it constitutes \"fair use\" under copyright laws within your jurisdiction. You are responsible for "
    "ensuring your use complies with applicable laws. Derivatives must also include the terms of use above."
)


def arb_download(path: str) -> str:
    from huggingface_hub import hf_hub_download

    from ...env import load_env

    load_env()
    return hf_hub_download(ARB, path, repo_type="dataset")


@register_task
class AgentRewardBench(TaskAdapter):
    name = f"{LEAF}/agentrewardbench"
    leaf = LEAF
    source = "AgentRewardBench, expert-annotated web agent trajectories, test split (Lù et al., 2025)"
    url = "https://huggingface.co/datasets/McGill-NLP/agent-reward-bench"
    license = ("AgentRewardBench terms of use: research use under fair use; derivatives must include these terms "
               "(no license file; trajectories contain third-party web content)")
    tier = "A"
    criteria_source = "The four questions and answer choices verbatim from the AgentRewardBench judge prompt, which mirror the expert annotation form"
    notes = ("Trajectories of GPT-4o, Claude 3.7 Sonnet, Llama-3.3-70B and Qwen2.5-VL-72B agents on WebArena, VisualWebArena, "
             "AssistantBench and WorkArena test tasks. The state is the text of the paper's text-only judge prompt (goal, "
             "actions with URL and reasoning, last accessibility tree). Hard labels from the primary annotator; for the "
             "trajectories annotated twice, the soft label is the two annotators' distribution. "
             "Each item's meta carries the dataset's terms of use, as they require.")

    def load(self):
        with open(arb_download("data/splits.csv"), encoding="utf-8") as f:
            test = {r["task_id"] for r in csv.DictReader(f) if r["split"] == "test"}
        with open(arb_download("data/annotations.csv"), encoding="utf-8") as f:
            annotations = list(csv.DictReader(f))
        groups: dict[tuple, list] = {}
        for a in annotations:  # the first annotation of a trajectory is the primary one (agent_reward_bench/judge/utils.py)
            groups.setdefault((a["benchmark"], a["model_name"], a["task_id"]), []).append(a)
        out = []
        for (benchmark, model, task_id), anns in groups.items():
            if task_id.replace(".improved", "").replace(".resized", "") not in test:
                continue
            if any(a[column] == "Unsure" for a in anns for column in ("trajectory_success", "trajectory_side_effect", "trajectory_optimality", "trajectory_looping")):
                continue
            out.append({"benchmark": benchmark, "model": model, "task_id": task_id, "annotations": [{k: a[k] for k in a if k.startswith("trajectory_")} for a in anns]})
        return out

    def label_of(self, record):
        return f"{record['benchmark']}|{record['annotations'][0]['trajectory_success']}"

    def state(self, record) -> dict:
        path = arb_download(f"judgments/{record['benchmark']}/{record['model']}/{ARB_JUDGE}/{record['task_id']}.json")
        with open(path, encoding="utf-8") as f:
            messages = json.load(f)["chat_messages"]
        parts = [p["text"] for p in messages[1]["content"] if p.get("type") == "text"]
        goal = next(p for p in parts if p.startswith("The user goal is:"))
        actions = next(p for p in parts if p.startswith("The agent performed the following actions:"))
        tree = next((p for p in parts if p.startswith("The last accessibility tree is:")), "")
        return {"goal": goal.removeprefix("The user goal is:").strip(), "actions": actions, "last_accessibility_tree": tree}

    def to_item(self, record, rng):
        anns = record["annotations"]
        questions, gold = {}, {}
        for key, (column, text, mapping) in ARB_QUESTIONS.items():
            questions[key] = Question(type="boolean", instructions=ARB_CONTEXT + text)
            votes = [mapping[a[column]] for a in anns]
            share = votes.count("true") / len(votes)
            gold[key] = Gold(label=votes[0], dist={"true": share, "false": 1 - share} if len(votes) > 1 else None)
        questions["optimality"] = Question(type="score", instructions=ARB_CONTEXT + "Did the agent perform the task optimally, by only performing necessary actions and avoiding unnecessary ones?", criteria=ARB_OPTIMALITY)
        levels = [str(ARB_OPTIMALITY.index(a["trajectory_optimality"])) for a in anns]
        dist = {str(i): levels.count(str(i)) / len(levels) for i in range(len(ARB_OPTIMALITY))} if len(levels) > 1 else None
        gold["optimality"] = Gold(label=levels[0], dist=dist)
        meta = {"benchmark": record["benchmark"], "agent": record["model"].removeprefix("GenericAgent-"), "original_id": record["task_id"],
                "annotators": len(anns), "terms_of_use": ARB_TERMS}
        return self.item(self.state(record), questions, gold, **meta)


TAU_QUESTION = (
    "The conversation is between a customer-service agent, who must follow the domain policy, and a user whose goal is "
    "given below. Did the agent successfully complete the user's task? It counts as completed only if, by the end of the "
    "conversation, the agent has made exactly the database changes (bookings, cancellations, modifications, exchanges, "
    "returns, refunds and so on) that the user's request calls for and the policy allows, and has given the user any "
    "information the task requires."
)


def render_tau(traj: list[dict]) -> str:
    lines = []
    for m in traj:
        role, content = m.get("role"), m.get("content")
        if role == "system":
            continue
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            if content and str(content) != "None":
                lines.append(f"Agent: {content}")
            calls = m.get("tool_calls") or []
            for call in calls if isinstance(calls, list) else []:
                function = call.get("function", {})
                lines.append(f"Agent calls tool {function.get('name')}({function.get('arguments')})")
        elif role == "tool":
            lines.append(f"Tool result ({m.get('name')}): {content}")
    return "\n\n".join(lines)


@register_task
class TauBench(TaskAdapter):
    name = f"{LEAF}/tau_bench"
    leaf = LEAF
    source = "tau-bench historical trajectories of GPT-4o and Claude 3.5 Sonnet (new) in the airline and retail domains (Yao et al., 2024)"
    url = "https://github.com/sierra-research/tau-bench"
    license = "MIT"
    tier = "C"
    criteria_source = "Written from the tau-bench success criterion (final database state and required outputs match the ground truth)"
    notes = ("Gold is the execution-based reward recorded with each trajectory (1 = success). Several trials per task. The "
             "user's goal is the instruction given to the simulated user. tau2-bench later corrected some airline tasks, so a "
             "few airline labels may rest on flawed ground truth.")
    FILES = {("airline", "gpt-4o"): "gpt-4o-airline.json", ("retail", "gpt-4o"): "gpt-4o-retail.json",
             ("airline", "claude-3.5-sonnet-new"): "sonnet-35-new-airline.json", ("retail", "claude-3.5-sonnet-new"): "sonnet-35-new-retail.json"}

    def load(self):
        out = []
        for (domain, model), name in self.FILES.items():
            path = fetch(f"https://raw.githubusercontent.com/sierra-research/tau-bench/main/historical_trajectories/{name}", f"tau_bench_{name}")
            with path.open(encoding="utf-8") as f:
                for r in json.load(f):
                    traj = r.get("traj") or []
                    if not traj or traj[0].get("role") != "system" or r.get("reward") not in (0, 0.0, 1, 1.0):
                        continue
                    out.append({"domain": domain, "model": model, "task_id": r["task_id"], "trial": r["trial"], "reward": int(r["reward"]),
                                "policy": traj[0]["content"], "goal": r["info"]["task"]["instruction"], "conversation": render_tau(traj)})
        return out

    def label_of(self, record):
        return f"{record['domain']}|{record['model']}|{record['reward']}"

    def to_item(self, record, rng):
        question = Question(type="boolean", instructions=TAU_QUESTION)
        state = {"policy": record["policy"], "user_goal": record["goal"], "conversation": record["conversation"]}
        gold = Gold(label="true" if record["reward"] == 1 else "false")
        return self.item(state, {"success": question}, {"success": gold}, domain=record["domain"], agent=record["model"], task_id=record["task_id"], trial=record["trial"])
