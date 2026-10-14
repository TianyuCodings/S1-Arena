"""Leaf: toxicity and hate speech (毒性 / 仇恨言论)."""

from __future__ import annotations

import csv

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch, hf_rows

LEAF = "basic_cls/toxicity"


@register_task
class CivilComments(TaskAdapter):
    name = f"{LEAF}/civil_comments"
    leaf = LEAF
    source = "Civil Comments / Jigsaw Unintended Bias (Borkan et al., 2019)"
    url = "https://huggingface.co/datasets/google/civil_comments"
    license = "CC0"
    tier = "A"
    criteria_source = "Rater instructions verbatim from conversationai crowdsourcing_annotation_schemes/toxicity_with_subattributes.md"
    sampling = "balanced"
    notes = "Gold is the fraction of raters who rated the comment Toxic or Very Toxic; sampled evenly across four bands of that fraction."

    BANDS = ((0.0, "0.0-0.2"), (0.2, "0.2-0.5"), (0.5, "0.5-0.8"), (0.8, "0.8-1.0"))

    def load(self):
        return [r for r in hf_rows("google/civil_comments", None, "test") if r["text"].strip()]

    def label_of(self, record):
        return [name for low, name in self.BANDS if record["toxicity"] >= low][-1]

    def to_item(self, record, rng):
        p = float(record["toxicity"])
        question = Question(
            type="boolean",
            instructions="Is this comment toxic or very toxic? Very Toxic - a very hateful, aggressive, disrespectful comment "
            "or otherwise very likely to make a user leave a discussion or give up on sharing their perspective. Toxic - a "
            "rude, disrespectful, unreasonable comment or otherwise somewhat likely to make a user leave a discussion or give "
            "up on sharing their perspective. Slightly toxic, hard to say and not toxic comments count as false.",
        )
        gold = Gold(label="true" if p >= 0.5 else "false", dist={"true": p, "false": 1.0 - p})
        return self.item({"comment": record["text"]}, {"toxic": question}, {"toxic": gold}, band=self.label_of(record))


@register_task
class DynaHate(TaskAdapter):
    name = f"{LEAF}/dynahate"
    leaf = LEAF
    source = "Dynamically Generated Hate Speech Dataset v0.2.3 (Vidgen et al., 2021)"
    url = "https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset"
    license = "CC BY 4.0"
    tier = "A"
    criteria_source = "Hate definition verbatim from the DynaHate annotation codebook (after Warner and Hirschberg, 2012)"

    def load(self):
        path = fetch(
            "https://raw.githubusercontent.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset/main/"
            "Dynamically%20Generated%20Hate%20Dataset%20v0.2.3.csv",
            "dynahate_v0.2.3.csv",
        )
        with path.open(encoding="utf-8") as f:
            return [r for r in csv.DictReader(f) if r["split"] == "test"]

    def label_of(self, record):
        return f"{record['label']}|{record['type']}"

    def to_item(self, record, rng):
        question = Question(
            type="boolean",
            instructions="Is this text hateful? Hate is defined as \"abusive speech targeting specific group characteristics, "
            "such as ethnic origin, religion, gender, or sexual orientation\".",
        )
        gold = Gold(label="true" if record["label"] == "hate" else "false")
        return self.item({"text": record["text"]}, {"hateful": question}, {"hateful": gold}, original_id=record["acl.id"], hate_type=record["type"], round=record["round"])
