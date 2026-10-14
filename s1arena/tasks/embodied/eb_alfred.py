"""EB-ALFRED episodes (EmbodiedBench, Yang et al., ICML 2025): did the robot complete the household task?

EmbodiedBench's trajectory release holds 3,326 episodes of 11 vision-language models acting in
the EB-ALFRED simulator (AI2-THOR) with high-level skills such as "find a Mug" or "put down the
object in hand". Every executed skill comes with the simulator's feedback, and every episode
with the simulator's own success check of the task's goal conditions. Only the text of the
record is used: the instruction, the skills and their validity rules (verbatim from the
EmbodiedBench prompt), and the executed skills with feedback. The models' own image
descriptions and reasoning are left out.
"""

from __future__ import annotations

import functools
import json

from ...env import load_env
from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter

REPO, REV = "EmbodiedBench/EB-Alfred_trajectory_dataset", "720b5f5683b9ecbb9418f8c5998d686cde3a1397"
SUBSETS = ("base", "common_sense", "complex_instruction", "long_horizon")  # the spatial and visual-appearance subsets need the images


@functools.lru_cache(maxsize=1)
def episodes() -> tuple[dict, ...]:
    load_env()
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(REPO, "eb-alfred_dataset_multi_step.json", repo_type="dataset", revision=REV), encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for ep in data:
        if ep["eval_set"] not in SUBSETS or any(not step for step in ep["trajectory"]):
            continue
        steps = [(s["action"][1], s["env_feedback"]) for step in ep["trajectory"] for s in step["executable_plan"]]
        if not steps:
            continue
        prompt = ep["input"]
        rules = prompt[prompt.index("## Action Descriptions and Validity Rules"): prompt.index("## The available action id")].strip()
        rules = rules.replace("## Action Descriptions and Validity Rules", "").strip()
        out.append({"model": ep["model_name"].split("_vlm")[0], "subset": ep["eval_set"], "episode": ep["episode_id"],
                    "instruction": ep["instruction"].strip(), "rules": rules, "steps": steps, "success": bool(ep["success"])})
    return tuple(out)


@register_task
class EBAlfredSuccess(TaskAdapter):
    name = "embodied/task_success/eb_alfred"
    leaf = "embodied/task_success"
    source = "EB-ALFRED trajectory dataset of EmbodiedBench: 11 vision-language models in AI2-THOR (Yang et al., ICML 2025)"
    url = "https://huggingface.co/datasets/EmbodiedBench/EB-Alfred_trajectory_dataset"
    license = "Not stated"
    tier = "B"
    criteria_source = "Skill descriptions and validity rules verbatim from the EmbodiedBench prompt; question written; gold is the simulator's success check"
    notes = ("Episodes of the base, common-sense, complex-instruction and long-horizon subsets (the spatial and visual-appearance "
             "subsets refer to what the images show). The record lists every executed skill with the simulator's feedback; "
             "episodes end when the goal conditions hold, when the model stops, or after the step limit, so failed episodes "
             "are longer on average (22 against 11 skills). The dataset card states no license.")

    def load(self):
        return list(episodes())

    def label_of(self, record):
        return f"{record['success']}|{record['subset']}"

    def to_item(self, record, rng):
        lines = [f"{i}. {action} -> {feedback}" for i, (action, feedback) in enumerate(record["steps"], 1)]
        state = {"skills": record["rules"], "instruction": record["instruction"], "executed_skills_with_feedback": "\n".join(lines)}
        question = Question(type="boolean", instructions=(
            "A robot in a simulated home was given the instruction and carried out the skills listed, in order; after each skill "
            "the simulator reported whether it could be executed. When the record ends, has the robot completed the task?"))
        gold = Gold(label="true" if record["success"] else "false")
        return self.item(state, {"task_completed": question}, {"task_completed": gold},
                         model=record["model"], subset=record["subset"], episode=record["episode"], skills=len(record["steps"]))
