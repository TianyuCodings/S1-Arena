"""Leaf: does answering need retrieval (是否需要检索)?

Only the time-sensitivity part of the adaptive-retrieval decision is covered: whether the correct answer can
change over time, so that a model with a fixed training cutoff must look it up. Model-specific gold (answering
with and without retrieval) is left open, see the report.
"""

from __future__ import annotations

import csv
import json

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch

LEAF = "ret/retrieval_need"
# The FreshQA snapshot of April 21, 2026 (the dataset is re-released as dated Google Sheets).
FRESHQA_SHEET = "https://docs.google.com/spreadsheets/d/1_8mi-yuK30mvoDJu1KQXD6ODem7MKMcIgVAwDSzJkjM/export?format=csv"
# Category definitions verbatim from Vu et al. (2024), Figure 1 caption.
FRESHQA_TYPES = {
    "never-changing": "never-changing, in which the answer almost never changes",
    "slow-changing": "slow-changing, in which the answer typically changes over the course of several years",
    "fast-changing": "fast-changing, in which the answer typically changes within a year or less",
}


@register_task
class FreshQA(TaskAdapter):
    name = f"{LEAF}/freshqa"
    leaf = LEAF
    source = "FreshQA, test questions, snapshot of 2026-04-21 (Vu et al., 2024)"
    url = "https://github.com/freshllms/freshqa"
    license = "Apache-2.0"
    tier = "A"
    criteria_source = "Question categories and their definitions verbatim from Vu et al. (2024), Figure 1"
    notes = ("Human-labelled fact type of each question: answers that change within a year or over several years need retrieval "
             "of current information; never-changing ones do not on that account. Includes FreshQA's false-premise questions, which "
             "also carry a fact type. Answers, sources, effective year and review schedule are withheld.")

    def load(self):
        with fetch(FRESHQA_SHEET, "freshqa_2026-04-21.csv").open(encoding="utf-8") as f:
            rows = list(csv.reader(f))
        header = next(i for i, r in enumerate(rows) if r and r[0] == "id")
        records = [dict(zip(rows[header], r)) for r in rows[header + 1:] if len(r) > 7 and r[0].strip()]
        return [r for r in records if r["split"] == "TEST" and r["fact_type"] in FRESHQA_TYPES and r["question"].strip()]

    def label_of(self, record):
        return f"{record['fact_type']}|{record['false_premise']}"

    def to_item(self, record, rng):
        question = Question(
            type="choice",
            instructions=("Questions can be divided by how often their correct answer changes; a question whose answer can change "
                          "needs up-to-date information retrieved at answering time. Which category does this question belong to?"),
            criteria=shuffled_options(FRESHQA_TYPES, rng),
        )
        gold = Gold(label=record["fact_type"])
        return self.item({"question": record["question"].strip()}, {"answer_change": question}, {"answer_change": gold},
                         original_id=record["id"], false_premise=record["false_premise"], num_hops=record["num_hops"])


@register_task
class SituatedQATemporal(TaskAdapter):
    name = f"{LEAF}/situatedqa_temporal"
    leaf = LEAF
    source = "SituatedQA, temporal context-dependent question identification, test (Zhang and Choi, 2021)"
    url = "https://github.com/mikejqzhang/SituatedQA"
    license = "Not stated (SituatedQA annotations); questions from WebQuestions (CC BY 4.0) and MS MARCO (non-commercial research terms)"
    tier = "B"
    criteria_source = "Adapted from the SituatedQA task definition: does the answer depend on the temporal context in which the question is asked"
    sampling = "balanced"
    notes = ("Majority of three crowdworkers (yes/maybe/no; split or 'maybe' majorities were discarded by the authors). Only the "
             "WebQuestions and MS MARCO questions of the test split are used: the NQ-Open ones overlap with "
             "know/factual_judgment/nq_open. Sampled evenly over (label, source) cells; about one question in five is time-dependent.")

    def load(self):
        url = "https://raw.githubusercontent.com/mikejqzhang/SituatedQA/master/data/id_data/temp.test.jsonl"
        with fetch(url, "situatedqa_id_temp.test.jsonl").open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return [r for r in rows if r["source_dataset"] in ("webq", "marco") and r["is_dependent"] in ("yes", "no")]

    def label_of(self, record):
        return f"{record['is_dependent']}|{record['source_dataset']}"

    def to_item(self, record, rng):
        question = Question(
            type="boolean",
            instructions=("Does the answer to this question depend on when it is asked, so that it has different correct answers at "
                          "different points in time and must be looked up for the present?"),
        )
        gold = Gold(label="true" if record["is_dependent"] == "yes" else "false")
        return self.item({"question": record["question"].strip()}, {"time_dependent": question}, {"time_dependent": gold},
                         original_id=str(record["id"]), source_dataset=record["source_dataset"])
