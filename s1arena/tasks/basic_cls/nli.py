"""Leaf: natural language inference (自然语言推理)."""

from __future__ import annotations

import json

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch, hf_rows

LEAF = "basic_cls/nli"
CHAOS = "https://raw.githubusercontent.com/jsbaan/calibration-on-disagreement-data/main/data/chaosNLI_v1.0/"

# ChaosNLI crowdworker instructions (Nie et al., 2020, Figure 1), verbatim
CHAOS_INSTRUCTIONS = (
    "Given a context, a statement can be either: Definitely correct (Entailment); or Definitely incorrect "
    "(Contradiction); or Neither (Neutral). Your goal is to choose the correct category for a given pair of "
    "context and statement."
)
CHAOS_OPTIONS = {"entailment": "Definitely correct", "contradiction": "Definitely incorrect", "neutral": "Neither"}
SHORT = {"e": "entailment", "n": "neutral", "c": "contradiction"}


@register_task
class ChaosNLI(TaskAdapter):
    name = f"{LEAF}/chaosnli"
    leaf = LEAF
    source = "ChaosNLI, SNLI and MNLI parts, 100 annotations per example (Nie et al., 2020)"
    url = "https://github.com/easonnie/ChaosNLI"
    license = "CC BY-NC 4.0"
    tier = "A"
    criteria_source = "Crowdworker instructions verbatim from Nie et al. (2020), Figure 1"
    notes = "Official download is gone; files come from the checksum-matched mirror jsbaan/calibration-on-disagreement-data."

    def load(self):
        rows = []
        for part in ("snli", "mnli_m"):
            with fetch(CHAOS + f"chaosNLI_{part}.jsonl", f"chaosNLI_{part}.jsonl").open(encoding="utf-8") as f:
                rows += [dict(json.loads(line), part=part) for line in f if line.strip()]
        return rows

    def label_of(self, record):
        return record["majority_label"]

    def to_item(self, record, rng):
        dist = dict(zip(("entailment", "neutral", "contradiction"), record["label_dist"]))
        total = sum(dist.values())
        dist = {k: v / total for k, v in dist.items()}
        question = Question(type="choice", instructions=CHAOS_INSTRUCTIONS, criteria=shuffled_options(CHAOS_OPTIONS, rng))
        state = {"context": record["example"]["premise"], "statement": record["example"]["hypothesis"]}
        gold = Gold(label=SHORT[record["majority_label"]], dist=dist)
        return self.item(state, {"relation": question}, {"relation": gold}, original_id=record["uid"], part=record["part"])


@register_task
class ANLIR3(TaskAdapter):
    name = f"{LEAF}/anli_r3"
    leaf = LEAF
    source = "Adversarial NLI, round 3 test (Nie et al., 2020)"
    url = "https://huggingface.co/datasets/facebook/anli"
    license = "CC BY-NC 4.0"
    tier = "B"
    criteria_source = "ChaosNLI crowdworker instructions reused for ANLI's premise-hypothesis pairs"

    def load(self):
        return hf_rows("facebook/anli", "plain_text", "test_r3")

    def label_of(self, record):
        return record["label_name"]

    def to_item(self, record, rng):
        question = Question(type="choice", instructions=CHAOS_INSTRUCTIONS, criteria=shuffled_options(CHAOS_OPTIONS, rng))
        state = {"context": record["premise"], "statement": record["hypothesis"]}
        return self.item(state, {"relation": question}, {"relation": Gold(label=record["label_name"])}, original_id=record["uid"])
