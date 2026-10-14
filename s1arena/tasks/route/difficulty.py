"""Leaf: difficulty / reasoning-effort estimation (难度 / 推理力度估计).

A score question with five difficulty levels defined by the solve rate: the share of a fixed
pool of models whose automatically scored answer to this request was correct. Gold is the level
that contains the measured solve rate.
"""

from __future__ import annotations

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import hf_rows
from . import outcomes as oc

LEAF = "route/difficulty"
LEVELS = [
    "Very easy: at least 80% of the models solve it",
    "Easy: 60% to 80% of the models solve it",
    "Medium: 40% to 60% of the models solve it",
    "Hard: 20% to 40% of the models solve it",
    "Very hard: fewer than 20% of the models solve it",
]


def level(solve_rate: float) -> int:
    for i, bound in enumerate((0.8, 0.6, 0.4, 0.2)):
        if solve_rate >= bound - 1e-9:
            return i
    return 4


class SolveRate(TaskAdapter):
    leaf = LEAF
    tier = "C"
    criteria_source = "Written; levels are fixed bands of the measured solve rate"
    POOL = ""  # which models the solve rate is over and how they were scored

    def rows(self) -> list[dict]:
        """Records with 'id', 'group', 'prompt' and 'rate' (share of models that solved it)."""
        raise NotImplementedError

    def load(self):
        return [dict(r, level=level(r["rate"])) for r in self.rows()]

    def label_of(self, record):
        return f"{record['level']}|{record['group']}"

    def to_item(self, record, rng):
        question = Question(type="score", instructions=f"How hard is this request for language models? {self.POOL}", criteria=LEVELS)
        gold = Gold(label=str(record["level"]))
        return self.item({"request": record["prompt"]}, {"difficulty": question}, {"difficulty": gold},
                         group=record["group"], solve_rate=round(record["rate"], 4), original_id=record["id"])


@register_task
class EmbedLLMDifficulty(SolveRate):
    name = f"{LEAF}/embedllm"
    source = "EmbedLLM correctness matrix, test questions (Zhuang et al., 2024)"
    url = oc.EMBEDLLM_URL
    license = "Apache-2.0 (card); source benchmarks keep their own licenses"
    notes = ("Solve rate over 112 open-weight models (2023 to early 2024) evaluated with the Open LLM Leaderboard harness. "
             "Only MMLU, MedMCQA, LogiQA and GSM8K questions, whose prompts show everything the models were scored on; left out: "
             "PIQA (used in know/commonsense), GPQA (its authors ask that examples not be posted in plain text), and sets whose "
             "prompts omit the answer options or whose scoring is broken (ASDiv).")
    POOL = ("Difficulty is the share of 112 open-weight language models (released 2023 to early 2024) whose answer was scored "
            "correct in a standard evaluation harness: the likelihood-ranked option for multiple choice, the final number for math word problems.")

    def rows(self):
        return [{"id": r["id"], "group": r["group"], "prompt": r["prompt"], "rate": sum(r["score"].values()) / len(r["score"])}
                for r in oc.embedllm(("mmlu", "medmcqa", "logiqa", "gsm8k")) if len(r["score"]) == 112]


@register_task
class BigCodeBenchDifficulty(SolveRate):
    name = f"{LEAF}/bigcodebench"
    source = "BigCodeBench-Instruct tasks with leaderboard solve rates (Zhuo et al., 2025)"
    url = "https://huggingface.co/datasets/bigcode/bigcodebench-solve-rate"
    license = "Apache-2.0"
    notes = ("Solve rate as published by the BigCodeBench team for the instruct split (share of leaderboard models whose greedy "
             "solution passes all tests). Tasks no model solved (211 of 1,140) are left out: many fail for environment or test reasons.")
    POOL = ("Difficulty is the share of the models on the BigCodeBench leaderboard (open and API models, 2024 to 2025) whose "
            "generated code passes all of the task's unit tests.")

    def rows(self):
        rates = {r["task_id"]: r["solve_rate"] for r in hf_rows("bigcode/bigcodebench-solve-rate", None, "instruct", revision="18a48547eade70496554e69b01dac52ad26bb297")}
        tasks = hf_rows("bigcode/bigcodebench", None, "v0.1.4", revision="b74c0d0bf70d2c0bc459be537895cca163007f1a")
        return [{"id": t["task_id"], "group": "bigcodebench", "prompt": t["instruct_prompt"].strip(), "rate": rates[t["task_id"]] / 100.0}
                for t in tasks if rates.get(t["task_id"], 0) > 0]

    def label_of(self, record):
        return str(record["level"])


@register_task
class LiveBenchDifficulty(SolveRate):
    name = f"{LEAF}/livebench"
    source = "LiveBench code generation, code completion, typo fixing and Connections puzzles (White et al., 2025)"
    url = oc.LIVEBENCH_URL
    license = oc.LIVEBENCH_LICENSE
    notes = ("Solve rate over every model on the LiveBench leaderboard that answered the question (at least 40; frontier and "
             "open models from 2023 to 2025), scored without an LLM judge. The license of the LiveBench question sets is not stated.")
    POOL = ("Difficulty is the share of the models on the LiveBench leaderboard (about 40 to 180 models from 2023 to 2025) "
            "whose output was scored fully correct: code passing the hidden tests, an exactly restored text, or a fully solved puzzle.")

    def rows(self):
        return [{"id": r["id"], "group": r["group"], "prompt": r["prompt"], "rate": sum(r["score"].values()) / len(r["score"])}
                for r in oc.livebench() if len(r["score"]) >= 40]
