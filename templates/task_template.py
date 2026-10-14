"""Template: one static benchmark source as an S1-Arena task.

Copy this file to s1arena/tasks/<domain>/<your_module>.py, rename the class, fill in the class attributes and replace
load(), label_of() and to_item(). Discovery imports every module under s1arena/tasks/, so nothing else changes.
Then check it and build it:

    uv run s1arena check-task basic_cls/example/my_dataset
    uv run s1arena build --leaf basic_cls/example/my_dataset --n 500

The inline records below make the template runnable as it is; a real task reads its evaluation split, for example
with `hf_rows("org/dataset", "config", "test")` from s1arena.tasks.common.
"""

from __future__ import annotations

from s1arena.registry import register_task
from s1arena.schema import Gold, Question
from s1arena.tasks.base import TaskAdapter, shuffled_options

LABELS = {  # option id -> description the model sees
    "positive": "The reviewer is satisfied with the product.",
    "neutral": "The reviewer is neither satisfied nor unsatisfied.",
    "negative": "The reviewer is unsatisfied with the product.",
}
RECORDS = [
    {"id": 1, "text": "Works exactly as described and the battery lasts all day.", "label": "positive"},
    {"id": 2, "text": "It does the job. Nothing special.", "label": "neutral"},
    {"id": 3, "text": "Stopped working after two days and the seller ignores me.", "label": "negative"},
    {"id": 4, "text": "Great value for the price, I bought a second one.", "label": "positive"},
    {"id": 5, "text": "Arrived on time; I have not used it much yet.", "label": "neutral"},
    {"id": 6, "text": "The screen cracked on the first drop. Do not buy.", "label": "negative"},
]


@register_task
class MyDataset(TaskAdapter):
    name = "basic_cls/example/my_dataset"  # <domain>/<leaf>/<source>; the domain must exist (see `s1arena tasks`)
    leaf = "basic_cls/example"  # tasks of one leaf are scored together; their requests are split evenly
    source = "My review dataset (Author et al., 2026)"  # as you would cite it
    url = "https://huggingface.co/datasets/org/my-dataset"
    license = "CC BY 4.0"  # the source's license; set redistributable = False if it does not allow sharing
    tier = "C"  # A: the source's own instructions verbatim; B: adapted; C: newly written
    criteria_source = "Written from the label names"  # where the instructions and option descriptions come from
    redistributable = True

    def load(self):
        """Yield the raw records of the evaluation split."""
        return RECORDS

    def label_of(self, record):
        """The key for stratified sampling, usually the gold label."""
        return record["label"]

    def to_item(self, record, rng):
        """One record as one request: the state the model reads, the questions it answers, and the gold answers."""
        question = Question(type="choice", instructions="What is the sentiment of the review?",
                            criteria=shuffled_options(LABELS, rng))  # a reproducible random option order
        return self.item({"review": record["text"]}, {"sentiment": question},
                         {"sentiment": Gold(label=record["label"])}, original_id=record["id"])
