"""Leaf: intent detection with out-of-scope requests (意图识别)."""

from __future__ import annotations

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import hf_rows, humanize

LEAF = "basic_cls/intent"


@register_task
class CLINC150(TaskAdapter):
    name = f"{LEAF}/clinc150"
    leaf = LEAF
    source = "CLINC150, plus variant with out-of-scope queries (Larson et al., 2019)"
    url = "https://huggingface.co/datasets/clinc/clinc_oos"
    license = "CC BY 3.0"
    tier = "C"
    criteria_source = "Written from the intent names; out-of-scope option written"

    def load(self):
        rows = hf_rows("clinc/clinc_oos", "plus", "test")
        names = sorted({r["intent_name"] for r in rows})
        self.options = {n: ("None of the listed intents applies; the request is out of scope." if n == "oos" else humanize(n)) for n in names}
        return rows

    def label_of(self, record):
        return record["intent_name"]

    def to_item(self, record, rng):
        question = Question(
            type="choice",
            instructions="Which intent does this request to a virtual assistant express? Choose oos if none of the listed intents applies.",
            criteria=shuffled_options(self.options, rng),
        )
        return self.item({"request": record["text"]}, {"intent": question}, {"intent": Gold(label=record["intent_name"])})


@register_task
class MassiveEN(TaskAdapter):
    name = f"{LEAF}/massive_en"
    leaf = LEAF
    source = "MASSIVE, en-US (FitzGerald et al., 2022)"
    url = "https://huggingface.co/datasets/mteb/amazon_massive_intent"
    license = "CC BY 4.0"
    tier = "C"
    criteria_source = "Written from the scenario_intent label names"

    def load(self):
        rows = hf_rows("mteb/amazon_massive_intent", "en", "test")
        train = hf_rows("mteb/amazon_massive_intent", "en", "train")
        labels = sorted({r["label_text"] for r in rows + train})
        self.options = {}
        for label in labels:
            scenario, _, intent = label.partition("_")
            self.options[label] = f"scenario: {humanize(scenario)}; intent: {humanize(intent) or humanize(scenario)}"
        return rows

    def label_of(self, record):
        return record["label_text"]

    def to_item(self, record, rng):
        question = Question(type="choice", instructions="Which intent does this request to a voice assistant express?", criteria=shuffled_options(self.options, rng))
        return self.item({"request": record["text"]}, {"intent": question}, {"intent": Gold(label=record["label_text"])}, original_id=record["id"])
