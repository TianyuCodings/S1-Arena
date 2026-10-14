"""Leaf: failure attribution (失败归因).

A failed agent run is shown in full; the questions ask which agent and which step caused the
failure, and what kind of error it was. Gold comes from human annotation: Who&When (the agent
and the decisive error step in failed multi-agent logs) and AgentErrorBench (the critical
failure step, the agent module it belongs to and its error type, annotated by experts). TRAIL
(expert span-level error annotation) is gated.
"""

from __future__ import annotations

import json
import re

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ._sources import drive_file, drive_folder, fits, hub_file

LEAF = "step/failure_attribution"

# ---------- Who&When ----------

# Adapted from the all-at-once prompt of Who&When (Automated_FA/Lib/utils.py, all_at_once): the two questions verbatim,
# the output-format sentences removed and the step numbering stated explicitly.
WW_CONTEXT = ("You are analyzing a multi-agent conversation history when solving a real world problem. The problem, its correct "
              "answer and the conversation are given; the agents did not reach the correct answer. Each entry of the conversation "
              "is a step, numbered from 0 as shown. ")
WW_AGENT = ("Which agent made a mistake that should be directly responsible for the wrong solution to the real world problem? "
            "If there are no agents that make obvious mistakes, decide one single agent in your mind.")
WW_STEP = "In which step did the mistake agent first make the mistake that is directly responsible for the wrong solution?"


def ww_agent(name: str) -> str:
    """Magentic-One roles such as 'Orchestrator (thought)' or 'Orchestrator (-> WebSurfer)' belong to the Orchestrator."""
    return str(name or "").split(" (")[0].strip()


@register_task
class WhoAndWhen(TaskAdapter):
    name = f"{LEAF}/who_and_when"
    leaf = LEAF
    source = "Who&When, failed multi-agent logs on GAIA and AssistantBench with annotated failure agent and step (Zhang et al., 2025)"
    url = "https://huggingface.co/datasets/Kevin355/Who_and_When"
    license = "MIT (GitHub repository ag2ai/Agents_Failure_Attribution; the Hugging Face card states no license)"
    tier = "B"
    criteria_source = "Questions verbatim from the Who&When all-at-once attribution prompt (output format removed; step numbering stated)"
    notes = ("Both subsets: 126 logs of CaptainAgent-built (algorithm-generated) teams and 58 Magentic-One (hand-crafted) logs. "
             "Annotators marked the agent responsible for the failure and the step of its decisive mistake. The correct answer to "
             "the problem is shown, as in the paper's with-ground-truth setting. Magentic-One roles are merged by agent "
             "('Orchestrator (thought)' -> Orchestrator). Logs whose annotated step is spoken by another agent than the annotated "
             "agent are dropped (9 logs).")

    def load(self):
        import pandas as pd

        rows = []
        for subset, speaker_key, answer_key in (("Algorithm-Generated", "name", "ground_truth"), ("Hand-Crafted", "role", "groundtruth")):
            frame = pd.read_parquet(hub_file("Kevin355/Who_and_When", f"{subset}.parquet"))
            for r in frame.to_dict("records"):
                history = [dict(e) for e in r["history"]]
                speakers = [ww_agent(e.get(speaker_key)) for e in history]
                step = int(r["mistake_step"])
                agents = {s.lower(): s for s in speakers if s and s.lower() != "human"}
                truth = agents.get(ww_agent(r["mistake_agent"]).lower())
                if not truth or not 0 <= step < len(history) or speakers[step].lower() != truth.lower():
                    continue
                steps = [f"Step {i} - {speakers[i]}: {str(e.get('content') or '').strip()}" for i, e in enumerate(history)]
                if not fits(*steps):
                    continue
                rows.append({"subset": subset, "id": r["question_ID"], "problem": r["question"], "answer": str(r[answer_key]),
                             "steps": steps, "speakers": speakers, "agents": sorted(set(agents.values())), "agent": truth, "step": step})
        return rows

    def label_of(self, record):
        return record["subset"]

    def to_item(self, record, rng):
        agents = {a: f"{a}" for a in record["agents"]}
        steps = {str(i): f"Step {i} ({s})" for i, s in enumerate(record["speakers"])}
        questions = {"step": Question(type="choice", instructions=WW_CONTEXT + WW_STEP, criteria=steps)}
        gold = {"step": Gold(label=str(record["step"]))}
        if len(agents) >= 2:  # a log with a single agent besides the human asks only for the step
            questions = {"agent": Question(type="choice", instructions=WW_CONTEXT + WW_AGENT, criteria=agents), **questions}
            gold = {"agent": Gold(label=record["agent"]), **gold}
        state = {"problem": record["problem"], "correct_answer": record["answer"], "conversation": "\n\n".join(record["steps"])}
        return self.item(state, questions, gold, subset=record["subset"], original_id=record["id"], n_steps=len(record["steps"]))


# ---------- AgentErrorBench ----------

AEB_ROOT = "1bQe6dQA85pktT63YnKIKJDTVaH3O3Vpu"  # public Google Drive folder linked from github.com/ulab-uiuc/AgentDebug
# Module and error-type definitions verbatim from AgentDebug (detector/error_definitions.py).
AEB_TYPES = {
    "memory": {
        "over_simplification": "Agent oversimplifies complex information from previous N steps, ignoring details and key factors, leading to decisions based on partial or oversimplified summaries",
        "memory_retrieval_failure": "Relevant information exists in agent memory but fails to be retrieved when needed",
        "hallucination": "Agent \"recalls\" events that never happened, object states never observed, or actions never executed, and uses these as basis for reasoning. Agent fills in missing parts without solid memory basis, generating false memory information",
    },
    "reflection": {
        "progress_misjudge": "Agent incorrectly evaluates progress toward completing the overall task goal, being either overly optimistic or pessimistic",
        "outcome_misinterpretation": "Agent correctly executes an action but incorrectly interprets the direct result or environment feedback from that action",
        "causal_misattribution": "Agent correctly identifies a failure phenomenon but attributes it to the wrong cause",
        "hallucination": "Agent believes it performed actions that never actually occurred",
    },
    "planning": {
        "constraint_ignorance": "Planning ignores task constraints, not considering resource limits (time, budget, space) or other relevant restrictions",
        "impossible_action": "Agent plans to execute an action or subtask that is fundamentally impossible under current physical or logical conditions. May include decomposition problems",
        "inefficient_plan": "Agent creates plan that can theoretically complete task but is extremely inefficient, lengthy, or illogical",
    },
    "action": {
        "misalignment": "Generated specific action completely contradicts the intention stated in current plan module (wrong action choice)",
        "invalid_action": "Uses action that does not exist",
        "format_error": "Generated action has invalid format causing parse failure",
        "parameter_error": "Action parameters are unreasonable or incorrectly chosen",
    },
    "system": {
        "step_limit": "Agent executes reasonably but fails due to reaching system maximum step limit",
        "tool_execution_error": "External tool or API called by agent returns error or exhibits unpredictable behavior",
        "llm_limit": "Agent response limitations cause failure",
        "environment_error": "Simulation environment itself has bugs, network issues cause unexpected behavior against expected rules",
    },
    "others": {"others": "All remaining problems not previously defined or discussed"},
}
AEB_MODULES = {
    "memory": "Memory: the agent's recall and summary of earlier observations and actions (errors: over-simplification, memory retrieval failure, hallucinated memories).",
    "reflection": "Reflection: the agent's assessment of the last action's outcome and of its progress (errors: progress misjudgment, outcome misinterpretation, causal misattribution, hallucinated actions).",
    "planning": "Planning: the agent's plan for the next step (errors: constraint ignorance, impossible action, inefficient plan).",
    "action": "Action: the concrete action the agent emits (errors: misalignment with the plan, invalid action, format error, parameter error).",
    "system": "System: failures outside the agent's reasoning (errors: step limit reached, tool execution error, LLM limit, environment error).",
    "others": "Others: all remaining problems not covered by the modules above.",
}
# Adapted from AgentDebug's critical-error identification prompt (detector/critical_error_detection.py).
AEB_CONTEXT = ("The agent below failed its task. Each step shows the observation the agent received and the agent's output "
               "(memory, reflection, plan and action; the first step has only a plan and an action). Identify the CRITICAL ERROR: the earliest and most important error that led to "
               "the task failure, i.e. the error that, if we could go back in time and fix only that error, would most likely have "
               "let the whole trajectory succeed. Step 1 has no memory or reflection, since there is no history yet. System errors "
               "(step limit, tool execution error, LLM limit, environment error) can also be the critical error, e.g. when the agent "
               "acted reasonably but hit the step limit. ")
AEB_TYPO = {"plan": "planning", "plan_inefficient": "inefficient_plan"}
_OBS_FIRST = re.compile(r"[Yy]our current observation is:\s*(.*?)\n\s*(?:Your task is to:.*?\n)?Your admissible actions", re.S)
_OBS_NEXT = re.compile(r"You are now at step \d+ and your current observation is:\s*(.*?)\nYour admissible actions", re.S)
_OBS_GAIA = re.compile(r"You are now at step \d+\.\s*\nCurrent Observation:\s*(.*?)\n\nAvailable Tools:", re.S)
_TASK = re.compile(r"(?:Your task is(?: to)?:|Task:)\s*(.*?)\n", re.S)


def aeb_steps(messages: list[dict], env: str) -> tuple[str, list[str]]:
    """Task text and one rendered block per agent step (observation, admissible actions, agent output)."""
    task_match = _TASK.search(messages[0]["content"])
    task = task_match.group(1).strip() if task_match else ""
    blocks = []
    turns = [m for m in messages if m["role"] in ("user", "assistant")]
    for t in range(0, len(turns) - 1, 2):
        user, agent = turns[t]["content"], turns[t + 1]["content"]
        if turns[t]["role"] != "user" or turns[t + 1]["role"] != "assistant":
            break
        n = t // 2 + 1
        pattern = _OBS_GAIA if env == "gaia" else (_OBS_FIRST if n == 1 else _OBS_NEXT)
        match = pattern.search(user)
        observation = match.group(1).strip() if match else ""
        if not observation:
            observation = "(none: the task has just started)" if n == 1 else "(empty)"
        # the admissible-action lists are left out: they add thousands of tokens per step and are not needed to attribute the failure
        blocks.append(f"Step {n}\nObservation: {observation}\nAgent output:\n{agent.strip()}")
    return task, blocks


@register_task
class AgentErrorBench(TaskAdapter):
    name = f"{LEAF}/agenterrorbench"
    leaf = LEAF
    source = "AgentErrorBench, failed ALFWorld, WebShop and GAIA runs with expert root-cause annotation (Zhu et al., 2025, AgentDebug)"
    url = "https://github.com/ulab-uiuc/AgentDebug"
    license = "MIT (repository license; the data is linked from the repository README)"
    tier = "B"
    criteria_source = ("Adapted from AgentDebug's critical-error identification prompt; module and error-type definitions verbatim "
                       "from AgentDebug's error definitions")
    notes = ("200 failed runs of GPT-4o, Llama-3.3-70B and Qwen3-8B agents (100 ALFWorld, 50 WebShop, 50 GAIA) with a "
             "memory / reflection / planning / action loop; expert annotators marked the critical failure step, its module and "
             "error type (Cohen's kappa 0.55). The error-type question is asked only when the annotated type is one of the taxonomy's "
             "types (a few labels are empty; 'plan' is read as planning).")

    def load(self):
        root = dict(drive_folder(AEB_ROOT))
        labels = dict(drive_folder(root["Label"]))
        runs = dict(drive_folder(root["Original_Failure_Trajectory"]))
        rows = []
        for env, label_file, folder in (("alfworld", "alfworld_labels.json", "ALFWorld"), ("webshop", "webshop_labels.json", "WebShop"),
                                        ("gaia", "gaia_labels.json", "GAIA")):
            with open(drive_file(labels[label_file], f"agenterrorbench_{label_file}"), encoding="utf-8") as f:
                annotations = json.load(f)
            files = dict(drive_folder(runs[folder]))
            for a in annotations:
                name = a["trajectory_id"] + ".json"
                if name not in files:
                    continue
                with open(drive_file(files[name], f"agenterrorbench_{env}_{name}"), encoding="utf-8") as f:
                    run = json.load(f)
                task, blocks = aeb_steps(run["messages"], env)
                module = AEB_TYPO.get(a["critical_failure_module"].strip().lower(), a["critical_failure_module"].strip().lower())
                step = int(a["critical_failure_step"])
                if module not in AEB_MODULES or not 1 <= step <= len(blocks) or not task or not fits(task, *blocks):
                    continue
                kind = ""
                for note in a.get("step_annotations") or []:
                    if note.get("step") == step and isinstance(note.get(a["critical_failure_module"]), dict):
                        raw = str(note[a["critical_failure_module"]].get("failure_type") or "").strip().lower()
                        kind = AEB_TYPO.get(raw, raw)
                kind = kind if kind in AEB_TYPES[module] else ""
                rows.append({"env": env, "id": a["trajectory_id"], "llm": a["LLM"], "task": task, "steps": blocks,
                             "step": step, "module": module, "type": kind})
        return rows

    def label_of(self, record):
        return f"{record['env']}|{record['module']}"

    def to_item(self, record, rng):
        steps = {str(i): f"Step {i}" for i in range(1, len(record["steps"]) + 1)}
        questions, gold = {}, {}
        if len(steps) >= 2:  # a run that ended after one step asks only for the module and type
            questions["critical_step"] = Question(type="choice", instructions=AEB_CONTEXT + "At which step does the critical error occur?", criteria=steps)
            gold["critical_step"] = Gold(label=str(record["step"]))
        questions["critical_module"] = Question(type="choice", instructions=AEB_CONTEXT + "In which module of the agent does the critical error occur?",
                                                criteria=dict(AEB_MODULES))
        gold["critical_module"] = Gold(label=record["module"])
        if record["type"]:
            types = {f"{m}.{t}": f"{m} / {t}: {d}" for m, group in AEB_TYPES.items() for t, d in group.items()}
            questions["error_type"] = Question(type="choice", instructions=AEB_CONTEXT + "Which type of error is the critical error?", criteria=types)
            gold["error_type"] = Gold(label=f"{record['module']}.{record['type']}")
        state = {"task": record["task"], "outcome": "The agent failed the task.", "trajectory": "\n\n".join(record["steps"])}
        return self.item(state, questions, gold, environment=record["env"], agent_model=record["llm"], original_id=record["id"])


# ---------- TRAIL ----------

TRAIL_REPO = "PatronusAI/TRAIL"
# TRAIL's error taxonomy with the definitions given in its evaluation prompt (benchmarking/run_eval.py), verbatim.
TRAIL_CATEGORIES = {
    "Language-only": "Reasoning error / hallucination: language-only hallucination",
    "Tool-related": "Reasoning error / hallucination: tool-related (fabricating tool outputs/capabilities)",
    "Poor Information Retrieval": "Reasoning error / information processing: poor information retrieval (tried to find information that was not relevant to the task)",
    "Tool Output Misinterpretation": "Reasoning error / information processing: tool output misinterpretation (made assumptions about the tool output or used the tool output in an incorrect context)",
    "Incorrect Problem Identification": "Reasoning error / decision making: incorrect problem identification (misunderstood the overall task or the local task)",
    "Tool Selection Errors": "Reasoning error / decision making: tool selection errors (used the wrong tool for the task)",
    "Formatting Errors": "Reasoning error / output generation: formatting errors (errors with formatting and execution of code or structuring of output in a specific format)",
    "Instruction Non-compliance": "Reasoning error / output generation: instruction non-compliance (failed to perform the task provided and instead did something else)",
    "Tool Definition Issues": "System execution error / configuration: tool definition issues (the tool was not defined correctly by the user or contains some errors that make it inconsistent with its description)",
    "Environment Setup Errors": "System execution error / configuration: environment setup errors (includes permission problems and inability to access resources or API keys)",
    "Rate Limiting": "System execution error / API issues: rate limiting (like 429)",
    "Authentication Errors": "System execution error / API issues: authentication errors (like 401/403)",
    "Service Errors": "System execution error / API issues: service errors (like 500)",
    "Resource Not Found": "System execution error / API issues: resource not found (like 404)",
    "Resource Exhaustion": "System execution error / resource management: resource exhaustion (includes memory overflow)",
    "Timeout Issues": "System execution error / resource management: timeout issues (the system took too long to respond)",
    "Context Handling Failures": "Planning and coordination error / context management: context handling failures (includes window overflow and state tracking or forgetting important context)",
    "Resource Abuse": "Planning and coordination error / context management: resource abuse (called the tool excessively due to memory issues)",
    "Goal Deviation": "Planning and coordination error / task management: goal deviation (the system deviated from the task or the subtask)",
    "Task Orchestration": "Planning and coordination error / task management: task orchestration (includes subtask coordination between agents and progress monitoring)",
}
TRAIL_ALIASES = {"context handling failure": "Context Handling Failures", "task orchestration errors": "Task Orchestration",
                 "task orchestration error": "Task Orchestration", "tool selection": "Tool Selection Errors", "formatting error": "Formatting Errors",
                 "instruction non complience": "Instruction Non-compliance"}
TRAIL_MIN_COUNT = 5  # categories annotated fewer times in the whole dataset are not asked (they would be 'no' almost everywhere)
# Adapted from TRAIL's evaluation prompt: the task (find all errors of the taxonomy in the trace) turned into one yes/no question per category.
TRAIL_QUESTION = ("Below is the execution trace of an LLM agent system (agents, LLM calls with the new messages each one received and its "
                  "output, tool calls with their inputs and outputs). Expert annotators went through every span of the trace and marked "
                  "each error by the taxonomy's final subcategory. Does the trace contain at least one error of this category: {definition}?")


def trail_category(name: str) -> str | None:
    key = " ".join(str(name).strip().split()).lower()
    exact = {c.lower(): c for c in TRAIL_CATEGORIES}
    return exact.get(key) or TRAIL_ALIASES.get(key)


def trail_json(text: str):
    """Some annotation files have trailing commas."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", text))


def render_trail(trace: dict) -> str:
    """The trace in execution order; every distinct LLM input message is shown once, where it first appears, and a tool
    output that an input message repeats verbatim is replaced there by a pointer to the tool span above."""
    lines, seen, outputs = [], set(), []

    def walk(span, depth):
        a = span.get("span_attributes") or {}
        kind, name = a.get("openinference.span.kind", ""), span.get("span_name", "")
        pad = "  " * min(depth, 6)
        if kind == "AGENT":
            task = a.get("input.value", "")
            try:
                task = json.loads(task).get("task", task)
            except (json.JSONDecodeError, AttributeError):
                pass
            lines.append(f"{pad}[agent {name} starts] task: {task}")
        elif kind == "LLM":
            i = 0
            while f"llm.input_messages.{i}.message.content" in a:
                content = str(a[f"llm.input_messages.{i}.message.content"])
                if content not in seen:
                    seen.add(content)
                    for tool, output in outputs:
                        if output in content:
                            content = content.replace(output, f"[the output of the {tool} call shown above]")
                    lines.append(f"{pad}[LLM input, {a.get(f'llm.input_messages.{i}.message.role', '')}] {content}")
                i += 1
            replies = [str(a[k]) for k in sorted(a) if k.startswith("llm.output_messages.") and k.endswith(".message.content")]
            lines.append(f"{pad}[LLM output, {a.get('llm.model_name', '')}] {' '.join(replies)}")
        elif kind == "TOOL":
            tool, output = a.get("tool.name", name), str(a.get("output.value", "") or "")
            lines.append(f"{pad}[tool {tool}] input: {a.get('input.value', '')} output: {output}")
            if len(output.strip()) >= 200:
                outputs.append((tool, output.strip()))
        elif kind == "CHAIN":
            lines.append(f"{pad}[{name}]")
        for child in span.get("child_spans") or []:
            walk(child, depth + 1)
        if kind == "AGENT":
            lines.append(f"{pad}[agent {name} returns] {a.get('output.value', '')}")

    for span in trace.get("spans") or []:
        walk(span, 0)
    return "\n".join(lines)


@register_task
class TRAIL(TaskAdapter):
    name = f"{LEAF}/trail"
    leaf = LEAF
    source = "TRAIL, expert-annotated error taxonomy over GAIA and SWE-bench agent traces (Deshpande et al., 2025)"
    url = "https://huggingface.co/datasets/PatronusAI/TRAIL"
    license = "MIT, gated: users agree not to reshare the dataset outside a gated or private repository"
    tier = "B"
    criteria_source = "Category names and definitions verbatim from TRAIL's evaluation prompt taxonomy; question adapted to one yes/no per category"
    redistributable = False
    notes = ("148 OpenTelemetry traces: o3-mini smolagents (OpenDeepResearch) on GAIA and a Claude 3.7 Sonnet CodeAct agent on "
             "SWE-bench Lite. Four expert annotators marked every error by span, category and impact, with four review rounds. One item "
             "per trace, one yes/no question per error category (multi-label); categories annotated fewer than five times in the whole "
             "dataset are not asked. The trace is rendered as text in execution order, showing each distinct LLM input message once. "
             "Gated on Hugging Face (accept the terms with the account whose token is used); items stay local.")

    def load(self):
        from collections import Counter

        from huggingface_hub import HfApi

        from ...env import load_env

        load_env()
        files = HfApi().list_repo_files(TRAIL_REPO, repo_type="dataset")
        rows = []
        for split, folder, labels in (("gaia", "GAIA", "processed_annotations_gaia"), ("swe_bench", "SWE Bench", "processed_annotations_swe_bench")):
            for path in sorted(f for f in files if f.startswith(folder + "/") and f.endswith(".json")):
                base = path.split("/")[-1]
                if f"{labels}/{base}" not in files:
                    continue
                with open(hub_file(TRAIL_REPO, path), encoding="utf-8") as f:
                    trace = json.load(f)
                with open(hub_file(TRAIL_REPO, f"{labels}/{base}"), encoding="utf-8") as f:
                    annotation = trail_json(f.read())
                found = [trail_category(e.get("category", "")) for e in annotation.get("errors") or []]
                text = render_trail(trace)
                if not fits(text):
                    continue
                rows.append({"split": split, "id": base.removesuffix(".json"), "trace": text,
                             "categories": sorted({c for c in found if c})})
        counts = Counter(c for r in rows for c in r["categories"])
        self.asked = [c for c in TRAIL_CATEGORIES if counts[c] >= TRAIL_MIN_COUNT]
        return rows

    def label_of(self, record):
        return f"{record['split']}|{len(record['categories']) > 2}"

    def to_item(self, record, rng):
        questions, gold = {}, {}
        for category in self.asked:
            key = category.lower().replace("-", "_").replace(" ", "_")
            questions[key] = Question(type="boolean", instructions=TRAIL_QUESTION.format(definition=TRAIL_CATEGORIES[category]))
            gold[key] = Gold(label="true" if category in record["categories"] else "false")
        return self.item({"trace": record["trace"]}, questions, gold, split=record["split"], original_id=record["id"])
