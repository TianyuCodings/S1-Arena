"""Leaf: process reward (过程奖励).

A step-by-step solution is shown; the question is which step is the first erroneous one
(or none). Gold comes from human step annotations: PRM800K raters, ProcessBench experts
and MR-Ben annotators. know/math_verification uses ProcessBench's final-answer
correctness; this leaf uses ProcessBench's first-error-step label instead.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch, hf_rows

LEAF = "judge/process_reward"

# PRM800K labeling instructions, phase 2 ("Rating Solution Steps"): definition of a Bad step, verbatim.
PRM800K_INSTRUCTIONS = (
    "The model's solution to the math problem is split into steps. Grade the steps in order and find the first Bad step. "
    "A Bad option is one that confidently says something incorrect, is off-topic/weird, leads the solution into a clear "
    "dead-end, or is not explained clearly enough for a human to follow along with (even if it is correct). If the option "
    "has a final answer, it should not only be correct, but also clearly follow from the rest of the reasoning. Which step "
    "is the first Bad step? Choose \"none\" if no step is Bad."
)


@register_task
class PRM800K(TaskAdapter):
    name = f"{LEAF}/prm800k"
    leaf = LEAF
    source = "PRM800K, phase 2 test split (Lightman et al., 2024)"
    url = "https://github.com/openai/prm800k"
    license = "MIT"
    tier = "B"
    criteria_source = "Definition of a Bad step verbatim from the PRM800K phase-2 labeling instructions; question adapted"
    notes = ("Full pre-generated GPT-4 solutions to MATH test problems; raters labelled steps until the first negative (Bad) "
             "step, so gold is that step, or none when the solution was labelled to the end without one. Solutions the rater "
             "gave up on are dropped, and so are the few solutions labelled twice with different answers.")

    def load(self):
        path = fetch("https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/data/phase2_test.jsonl", "prm800k_phase2_test.jsonl")
        groups = defaultdict(list)
        with path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                finish, question = r["label"]["finish_reason"], r["question"]
                steps, labelled = question["pre_generated_steps"], r["label"]["steps"]
                if finish not in ("found_error", "solution") or not steps or len(labelled) > len(steps):
                    continue
                ratings = []
                for i, step in enumerate(labelled):
                    match = [c["rating"] for c in step["completions"] or [] if c["text"] == steps[i]]
                    ratings.append(match[0] if match else None)
                if None in ratings:
                    continue
                first = next((i for i, x in enumerate(ratings) if x == -1), None)
                if (finish == "found_error") != (first == len(labelled) - 1) or (finish == "solution" and len(labelled) != len(steps)):
                    continue
                groups[(question["problem"], tuple(steps))].append("none" if first is None else f"step_{first + 1}")
        return [{"problem": p, "steps": list(s), "label": labels[0]} for (p, s), labels in groups.items() if len(set(labels)) == 1]

    def label_of(self, record):
        return "none" if record["label"] == "none" else "error"

    def to_item(self, record, rng):
        options = {f"step_{i}": f"Step {i}" for i in range(1, len(record["steps"]) + 1)}
        options["none"] = "No step is Bad"
        solution = "\n".join(f"Step {i}: {s.strip()}" for i, s in enumerate(record["steps"], 1))
        question = Question(type="choice", instructions=PRM800K_INSTRUCTIONS, criteria=options)
        return self.item({"problem": record["problem"], "solution": solution}, {"first_bad_step": question}, {"first_bad_step": Gold(label=record["label"])}, steps=len(record["steps"]))


# ProcessBench critique template (code/templates/critique_template.txt), output-format sentence removed.
PROCESSBENCH_INSTRUCTIONS = (
    "The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0). Your "
    "task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return "
    "the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "
    "\"not found\")."
)


@register_task
class ProcessBenchFirstError(TaskAdapter):
    name = f"{LEAF}/processbench"
    leaf = LEAF
    source = "ProcessBench, earliest-error paragraph (Zheng et al., 2025)"
    url = "https://huggingface.co/datasets/Qwen/ProcessBench"
    license = "Apache-2.0"
    tier = "A"
    criteria_source = "ProcessBench critique template (code/templates/critique_template.txt), output-format sentence removed"
    notes = ("GSM8K, MATH, OlympiadBench and Omni-MATH solutions by Qwen and Llama models with expert-annotated earliest "
             "error. The same solutions appear in know/math_verification/processbench, which asks about final-answer "
             "correctness instead.")

    def load(self):
        rows = []
        for split in ("gsm8k", "math", "olympiadbench", "omnimath"):
            rows += [dict(r, split=split) for r in hf_rows("Qwen/ProcessBench", None, split) if -1 <= int(r["label"]) < len(r["steps"])]
        return rows

    def label_of(self, record):
        return f"{record['split']}|{int(record['label']) == -1}"

    def to_item(self, record, rng):
        options = {"-1": "-1: no paragraph contains an error (not found)"}
        options.update({str(i): f"Paragraph {i}" for i in range(len(record["steps"]))})
        solution = "\n\n".join(f"<paragraph_{i}>\n{s}\n</paragraph_{i}>" for i, s in enumerate(record["steps"]))
        question = Question(type="choice", instructions=PROCESSBENCH_INSTRUCTIONS, criteria=options)
        return self.item({"math_problem": record["problem"], "solution": solution}, {"earliest_error": question},
                         {"earliest_error": Gold(label=str(int(record["label"])))}, split=record["split"], generator=record["generator"], original_id=record["id"])


# MR-Ben evaluation prompt (script/utils.py): task and field definitions, verbatim; output-format part removed.
MRBEN_INSTRUCTIONS = (
    "Following is a question/solution pair in subject {subject}. Your task is to examine the solutions step by step. "
    "First Error Step: For every step it can either be correct, neutral or incorrect. Correct steps are those that possess "
    "sound logic and correct computation and lead to the correct answer. Neutral steps are those step that are explanatory, "
    "exploring or focusing on background illustration. They have no obvious mistakes but is not very clear if they lead to "
    "the correct answer. Incorrect steps are those with factual errors, computation errors or understanding/logic errors. "
    "These steps might or might not detour the reasoning path to incorrect answers. We need to single out the first step "
    "that comes with above errors or lead to incorrect answers. Which step is the first error step? Choose \"N/A\" if the "
    "solution is correct (it correctly answers the question with justifiable reasoning and selects the correct option)."
)
_STEP = re.compile(r"^\s*Step\s*(\d+)\s*[:.]", re.IGNORECASE)


@register_task
class MRBen(TaskAdapter):
    name = f"{LEAF}/mr_ben"
    leaf = LEAF
    source = "MR-Ben meta-reasoning benchmark, all subjects except coding (Zeng et al., 2024)"
    url = "https://huggingface.co/datasets/Randolphzeng/Mr-Ben"
    license = "MIT"
    tier = "B"
    criteria_source = "MR-Ben evaluation prompt (script/utils.py): first-error-step definitions verbatim, task sentence adapted to a single choice, output format removed"
    notes = ("Multiple-choice questions in physics, chemistry, biology, medicine, math (MMLU subjects) and logic (LogiQA) "
             "answered by GPT-3.5, Claude 2 and Mistral-medium; solution correctness and first error step annotated by "
             "humans. Coding is left out because its first errors are marked as code lines, not steps.")
    SUBJECTS = ("biology", "chemistry", "logic", "math", "medicine", "physics")

    def load(self):
        from huggingface_hub import hf_hub_download

        from ...env import load_env

        load_env()
        out = []
        for subject in self.SUBJECTS:
            with open(hf_hub_download("Randolphzeng/Mr-Ben", f"{subject}.json", repo_type="dataset"), encoding="utf-8") as f:
                for r in json.load(f):
                    lines = [s.strip() for s in r["Model_Solution_Steps"] if s and s.strip()]
                    numbers = [int(m.group(1)) for m in (_STEP.match(s) for s in lines) if m]
                    if not numbers or numbers != list(range(1, len(numbers) + 1)):
                        continue
                    if r["Model_Solution_Correctness"] == "correct":
                        label = "none"
                    elif r["Model_Solution_Correctness"] == "incorrect" and str(r["Model_Solution_First_Error_Step"]).isdigit():
                        label = str(int(r["Model_Solution_First_Error_Step"]))
                        if int(label) not in numbers:
                            continue
                    else:
                        continue
                    out.append({"uuid": r["Question_UUID"], "subject": r["Subject"], "file": subject, "question": r["Question"],
                                "options": r.get("Options") or "", "solution": "\n".join(lines), "steps": len(numbers), "label": label, "model": r["Sampled_Model"]})
        return out

    def label_of(self, record):
        return f"{record['file']}|{record['label'] == 'none'}"

    def to_item(self, record, rng):
        options = {str(i): f"Step {i}" for i in range(1, record["steps"] + 1)}
        options["none"] = "N/A: the solution is correct"
        question = Question(type="choice", instructions=MRBEN_INSTRUCTIONS.format(subject=record["subject"]), criteria=options)
        state = {"question": record["question"], "options": record["options"], "step_by_step_solution": record["solution"]}
        return self.item(state, {"first_error_step": question}, {"first_error_step": Gold(label=record["label"])},
                         subject=record["subject"], solution_model=record["model"], original_id=record["uuid"])
