"""PlanBench (Valmeekam et al., NeurIPS 2023 Datasets and Benchmarks): Blocksworld and Logistics.

Blocksworld is the classic robot-arm domain: one hand picks up, puts down, stacks and unstacks
blocks. States, goals and plans are given in PlanBench's own English templates, and its
domain description (the list of actions and restrictions) is reused verbatim. Three uses:
- failure type: PlanBench's plan-verification task, whose gold is VAL's verdict (valid, an
  action with an unmet precondition at step k, or an unmet goal);
- task success: plans written by 13 language models for PlanBench's plan-generation task, with
  PlanBench's VAL check of each plan;
- next skill: the next action on a shortest plan, computed by breadth-first search over all
  Blocksworld states (only states where exactly one applicable action is optimal).
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from collections import deque

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch

RAW = "https://raw.githubusercontent.com/karthikv792/LLMs-Planning/fc638a1aff7df3fe7a1a1d289fa2c04cc24dc284/"
URL = "https://github.com/karthikv792/LLMs-Planning"
LICENSE = "MIT"
COLORS = {"a": "red", "b": "blue", "c": "orange", "d": "yellow", "e": "white", "f": "magenta", "g": "black", "h": "cyan",
          "i": "green", "j": "violet", "k": "silver", "l": "gold"}  # PlanBench's encoded_objects


@functools.lru_cache(maxsize=None)
def planbench_file(path: str) -> dict:
    with open(fetch(RAW + path, "planbench_" + path.replace("/", "_")), encoding="utf-8") as f:
        return json.load(f)


def _order(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def split_query(query: str) -> dict:
    """Domain description, and the last problem of a PlanBench prompt: initial conditions, goal and plan (if any)."""
    rules = query.split("[STATEMENT]")[0].strip()
    last = query.split("[STATEMENT]")[-1]
    init = re.search(r"As initial conditions I have that, (.+?)\.\s*\n", last, re.S)
    goal = re.search(r"My goal is to have that (.+?)\.\s*\n", last, re.S)
    plan = re.search(r"\[PLAN\]\n(.*?)\[PLAN END\]", last, re.S)
    return {"rules": rules, "init": init.group(1).strip() if init else "", "goal": goal.group(1).strip() if goal else "",
            "plan": [line.strip() for line in plan.group(1).splitlines() if line.strip()] if plan else []}


# ---------- a Blocksworld model (PlanBench semantics) ----------

def facts_from_text(text: str) -> frozenset:
    """'the red block is clear, ... and the yellow block is on the table' -> {('clear','red'), ...}."""
    facts = set()
    for part in re.split(r", | and ", text):
        part = part.strip()
        m = re.fullmatch(r"the (\w+) block is on top of the (\w+) block", part)
        if m:
            facts.add(("on", m.group(1), m.group(2)))
            continue
        m = re.fullmatch(r"the (\w+) block is on the table", part)
        if m:
            facts.add(("ontable", m.group(1)))
            continue
        m = re.fullmatch(r"the (\w+) block is clear", part)
        if m:
            facts.add(("clear", m.group(1)))
            continue
        if part == "the hand is empty":
            facts.add(("handempty",))
            continue
        m = re.fullmatch(r"the hand is currently holding (\w+) block", part)
        if m:
            facts.add(("holding", m.group(1)))
            continue
        raise ValueError(f"unknown fact: {part!r}")
    return frozenset(facts)


def blocks_of(facts) -> list[str]:
    return sorted({x for f in facts for x in f[1:]})


def applicable(state: frozenset, blocks: list[str]) -> list[tuple]:
    out = []
    for x in blocks:
        if ("handempty",) in state and ("clear", x) in state and ("ontable", x) in state:
            out.append(("pick-up", x))
        if ("holding", x) in state:
            out.append(("put-down", x))
        for y in blocks:
            if x == y:
                continue
            if ("holding", x) in state and ("clear", y) in state:
                out.append(("stack", x, y))
            if ("on", x, y) in state and ("clear", x) in state and ("handempty",) in state:
                out.append(("unstack", x, y))
    return out


def apply(state: frozenset, action: tuple) -> frozenset:
    s = set(state)
    kind, x = action[0], action[1]
    if kind == "pick-up":
        s -= {("ontable", x), ("clear", x), ("handempty",)}
        s.add(("holding", x))
    elif kind == "put-down":
        s -= {("holding", x)}
        s |= {("ontable", x), ("clear", x), ("handempty",)}
    elif kind == "stack":
        y = action[2]
        s -= {("holding", x), ("clear", y)}
        s |= {("on", x, y), ("clear", x), ("handempty",)}
    else:
        y = action[2]
        s -= {("on", x, y), ("clear", x), ("handempty",)}
        s |= {("holding", x), ("clear", y)}
    return frozenset(s)


def action_text(action: tuple) -> str:
    kind, x = action[0], action[1]
    if kind == "pick-up":
        return f"pick up the {x} block"
    if kind == "put-down":
        return f"put down the {x} block"
    if kind == "stack":
        return f"stack the {x} block on top of the {action[2]} block"
    return f"unstack the {x} block from on top of the {action[2]} block"


def goal_distances(init: frozenset, goal: frozenset, blocks: list[str]) -> dict[frozenset, int]:
    """Length of a shortest plan to the goal from every state reachable from init (forward search, then
    breadth-first search backwards from the states that satisfy the goal)."""
    succ, seen, queue = {}, {init}, deque([init])
    while queue:
        s = queue.popleft()
        succ[s] = [apply(s, a) for a in applicable(s, blocks)]
        for t in succ[s]:
            if t not in seen:
                seen.add(t)
                queue.append(t)
    pred: dict = {s: [] for s in succ}
    for s, ts in succ.items():
        for t in ts:
            pred[t].append(s)
    dist = {s: 0 for s in succ if goal <= s}
    queue = deque(dist)
    while queue:
        t = queue.popleft()
        for s in pred[t]:
            if s not in dist:
                dist[s] = dist[t] + 1
                queue.append(s)
    return dist


def pddl_to_text(line: str) -> str | None:
    """'(unstack b c)' -> 'unstack the blue block from on top of the orange block'."""
    parts = line.strip().strip("()").split()
    if not parts or parts[0] not in ("pick-up", "put-down", "stack", "unstack") or any(p not in COLORS for p in parts[1:]):
        return None
    if len(parts) != (3 if parts[0] in ("stack", "unstack") else 2):
        return None
    return action_text((parts[0], *[COLORS[p] for p in parts[1:]]))


# ---------- failure type: PlanBench plan verification (VAL verdicts) ----------

VERIFICATION = {"blocksworld": "plan-bench/prompts/blocksworld/task_3_plan_verification.json",
                "logistics": "plan-bench/prompts/logistics/task_3_plan_verification.json"}


def verdict(text: str) -> tuple[str, int | None]:
    if "plan is valid" in text:
        return "valid", None
    m = re.search(r"action at step (\d+) has (?:an )?unmet precondition", text)
    if m:
        return "precondition", int(m.group(1))
    if "unmet goal condition" in text:
        return "goal", None
    raise ValueError(f"unknown verdict: {text[:80]}")


@register_task
class PlanBenchVerification(TaskAdapter):
    name = "embodied/failure_type/planbench_verification"
    leaf = "embodied/failure_type"
    source = "PlanBench plan verification task, Blocksworld and Logistics (Valmeekam et al., 2023)"
    url = URL
    license = LICENSE
    tier = "B"
    criteria_source = "PlanBench domain descriptions and problem statements verbatim; the options restate VAL's three verdicts (valid, unmet precondition at step k, unmet goal)"
    notes = ("Plans for PlanBench problems, some valid and some broken; the gold is VAL's verdict as released with the task. "
             "One question joins the failure type and the step: the plan is valid, it fails at step k because an action's "
             "precondition does not hold, or every action runs but the goal is not reached. The one-shot example of the "
             "original prompt is left out.")

    def load(self):
        rows = []
        for domain, path in VERIFICATION.items():
            for inst in planbench_file(path)["instances"]:
                parts = split_query(inst["query"])
                kind, step = verdict(inst["ground_truth_plan"])
                if not parts["plan"] or (step is not None and step > len(parts["plan"])):
                    continue
                rows.append({"domain": domain, "id": inst["instance_id"], "kind": kind, "step": step, **parts})
        return rows

    def label_of(self, record):
        return f"{record['domain']}|{record['kind']}"

    def to_item(self, record, rng):
        options = {"valid": "Valid: every action can be carried out when its turn comes, and the goal holds at the end",
                   "goal_not_reached": "Goal not reached: every action can be carried out, but the goal does not hold at the end"}
        for i, step in enumerate(record["plan"], 1):
            options[f"fails_at_step_{i}"] = f"Fails at step {i} ({step}): a precondition of this action does not hold"
        gold = {"valid": "valid", "goal": "goal_not_reached"}.get(record["kind"]) or f"fails_at_step_{record['step']}"
        question = Question(type="choice", criteria=options, instructions=(
            "Verify the plan: carry out its actions in order from the initial conditions, following the restrictions on the "
            "actions. Is the plan valid, and if not, how does it fail?"))
        state = {"domain_description": record["rules"], "initial_conditions": record["init"], "goal": record["goal"],
                 "plan": "\n".join(f"{i}. {s}" for i, s in enumerate(record["plan"], 1))}
        return self.item(state, {"verdict": question}, {"verdict": Gold(label=gold)}, domain=record["domain"],
                         original_id=record["id"], failure=record["kind"])


# ---------- task success: plans written by language models, checked by VAL ----------

LLM_RESULTS = {  # model -> results file (prompting as released)
    "gpt-3.5-turbo-instruct": "gpt-3.5-turbo-instruct/task_1_plan_generation.json",
    "gpt-4-turbo": "gpt-4-turbo_chat/task_1_plan_generation.json",
    "gpt-4o": "gpt-4o_chat/task_1_plan_generation_zero_shot.json",
    "claude-3-opus": "claude-3-opus/task_1_plan_generation.json",
    "claude-3.5-sonnet": "claude-3.5-sonnet_aws/task_1_plan_generation_zero_shot.json",
    "gemini-1.5-pro": "gemini-1.5-pro/task_1_plan_generation.json",
    "gemini-1.5-flash": "gemini-1.5-flash/task_1_plan_generation_zero_shot.json",
    "llama-3.1-405b": "llama-3.1-405b_aws/task_1_plan_generation.json",
    "llama-3-70b": "llama3-70b-8192_groq/task_1_plan_generation_zero_shot.json",
    "o1-mini": "o1-mini_chat/task_1_plan_generation_zero_shot.json",
    "o1-preview": "o1-preview_chat/task_1_plan_generation_zero_shot.json",
    "deepseek-r1": "deepseek-r1-api/task_1_plan_generation_zero_shot.json",
    "qwq": "qwen-qwq/task_1_plan_generation.json",
}


@register_task
class PlanBenchLLMPlans(TaskAdapter):
    name = "embodied/task_success/planbench_llm_plans"
    leaf = "embodied/task_success"
    source = "PlanBench Blocksworld plan generation: plans written by 13 language models with VAL checks (Valmeekam et al., 2023-2025)"
    url = URL
    license = LICENSE
    tier = "B"
    criteria_source = "PlanBench domain description and problem statement verbatim; question written; gold is PlanBench's VAL check of the plan"
    notes = ("For each of the 500 Blocksworld problems, the plan of one model (chosen by hash among those that returned a "
             "non-empty plan) as PlanBench extracted it from the reply, rendered in PlanBench's action wording. Gold is "
             "PlanBench's 'llm_correct', which runs VAL on the extracted plan. The models range from about 6% of plans correct "
             "(Gemini 1.5 Flash, GPT-3.5 Instruct) to 99% (DeepSeek-R1); GPT-4's released files carry no VAL result and are not used.")

    def load(self):
        by_model = {m: {i["instance_id"]: i for i in planbench_file("llm_planning_analysis/results/blocksworld/" + f)["instances"]}
                    for m, f in LLM_RESULTS.items()}
        rows = []
        for iid in sorted({i for plans in by_model.values() for i in plans}):
            for model in sorted(LLM_RESULTS, key=lambda m: _order(f"{iid}:{m}")):
                inst = by_model[model].get(iid)
                if not inst or "llm_correct" not in inst:
                    continue
                plan = inst.get("extracted_llm_plan") or ""
                lines = plan if isinstance(plan, list) else plan.splitlines()
                steps = [pddl_to_text(line) for line in lines if line.strip()]
                if not steps or any(s is None for s in steps):
                    continue
                parts = split_query(inst["query"])
                if not parts["init"] or not parts["goal"]:
                    continue
                rows.append({"id": iid, "model": model, "rules": parts["rules"], "init": parts["init"], "goal": parts["goal"],
                             "plan": steps, "correct": bool(inst["llm_correct"])})
                break
        return rows

    def label_of(self, record):
        return str(record["correct"])

    def to_item(self, record, rng):
        question = Question(type="boolean", instructions=(
            "A language model wrote this plan for the problem. Does the plan achieve the goal? It does only if every action is "
            "allowed by the restrictions when it is carried out and the goal holds after the last action."))
        state = {"domain_description": record["rules"], "initial_conditions": record["init"], "goal": record["goal"],
                 "plan": "\n".join(f"{i}. {s}" for i, s in enumerate(record["plan"], 1))}
        return self.item(state, {"plan_succeeds": question}, {"plan_succeeds": Gold(label="true" if record["correct"] else "false")},
                         original_id=record["id"], model=record["model"])


# ---------- next skill: the next action on a shortest plan (breadth-first search) ----------

@register_task
class BlocksworldNextAction(TaskAdapter):
    name = "embodied/next_skill/planbench_blocksworld"
    leaf = "embodied/next_skill"
    source = "PlanBench Blocksworld problems, next action on a shortest plan (problems from Valmeekam et al., 2023; answers by exhaustive search)"
    url = URL
    license = LICENSE
    tier = "C"
    criteria_source = "PlanBench domain description verbatim; question written; gold from breadth-first search over all states"
    notes = ("Each item shows a PlanBench problem and the actions done so far (a prefix of a shortest plan); the options are "
             "all actions allowed in the current state and the gold is the only one that keeps the plan shortest. States "
             "where two or more actions are equally good are not used. Checked by an independent solver.")

    def load(self):
        rows = []
        for inst in planbench_file("plan-bench/prompts/blocksworld/task_1_plan_generation.json")["instances"]:
            parts = split_query(inst["query"])
            if not parts["init"] or not parts["goal"]:
                continue
            init, goal = facts_from_text(parts["init"]), facts_from_text(parts["goal"])
            blocks = blocks_of(init)
            dist = goal_distances(init, goal, blocks)
            if not dist.get(init):
                continue
            path, state, candidates = [], init, []
            while dist[state]:
                allowed = applicable(state, blocks)
                best = [a for a in allowed if dist.get(apply(state, a)) == dist[state] - 1]
                if len(best) == 1 and len(allowed) >= 2:
                    candidates.append((list(path), best[0], dist[state]))
                a = sorted(best)[0]
                path.append(a)
                state = apply(state, a)
            if not candidates:
                continue
            prefix, answer, remaining = candidates[int(_order(str(inst["instance_id"]))[:8], 16) % len(candidates)]
            rows.append({"id": inst["instance_id"], "rules": parts["rules"], "init": parts["init"], "goal": parts["goal"],
                         "done": [action_text(a) for a in prefix], "answer": action_text(answer), "remaining": remaining,
                         "blocks": len(blocks)})
        return rows

    def label_of(self, record):
        return f"{record['blocks']} blocks|{min(record['remaining'], 6)}"

    def to_item(self, record, rng):
        init = facts_from_text(record["init"])
        blocks = blocks_of(init)
        state = init
        for step in record["done"]:
            state = apply(state, next(a for a in applicable(state, blocks) if action_text(a) == step))
        actions = sorted(action_text(a) for a in applicable(state, blocks))
        options = {f"action_{i + 1}": a for i, a in enumerate(actions)}
        gold = next(k for k, a in options.items() if a == record["answer"])
        question = Question(type="choice", criteria=options, instructions=(
            "The actions done so far were carried out from the initial conditions. Which action should come next to reach the "
            "goal in as few actions as possible?"))
        done = "\n".join(f"{i}. {s}" for i, s in enumerate(record["done"], 1)) or "(none yet)"
        state_text = {"domain_description": record["rules"], "initial_conditions": record["init"], "goal": record["goal"],
                      "actions_done_so_far": done}
        return self.item(state_text, {"next_action": question}, {"next_action": Gold(label=gold)},
                         original_id=record["id"], remaining_actions=record["remaining"], blocks=record["blocks"])
