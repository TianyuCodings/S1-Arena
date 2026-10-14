"""Leaf: RAG faithfulness / hallucination (RAG 忠实度 / 幻觉).

Is the generated text supported by the context the generator was given? RAGTruth has
human word-level hallucination spans, turned into one boolean per response sentence;
BEGIN has human attribution labels for knowledge-grounded dialogue responses.
"""

from __future__ import annotations

import csv
import json
import re

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch

LEAF = "judge/rag_faithfulness"

# RAGTruth hallucination taxonomy (Niu et al., 2024, Section 3.1), condensed into one question.
RAGTRUTH_QUESTION = (
    "Is this sentence of the response fully supported by the source content given in the task prompt? It is not supported "
    "if it contradicts or departs from the provided information, altering its meaning (conflict), or if it includes "
    "information not substantiated by the provided information, including inferred details, insights or assumptions, even "
    "when they may be true in the real world (baseless information). Sentence {i}: \"{sentence}\""
)
_BOUNDARY = re.compile(r"[.!?][\"”’')\]]*(\s+)(?=[\"“'(\[]?[A-Z0-9])")


def sentence_spans(text: str, min_chars: int = 20) -> list[tuple[int, int]]:
    """Character spans of sentences: lines are split at sentence-final punctuation; fragments shorter
    than min_chars (list markers, headings) are merged into the following sentence of the same line."""
    spans = []
    for line in re.finditer(r"[^\n]+", text):
        pieces, start = [], 0
        for boundary in _BOUNDARY.finditer(line.group()):  # the gap (group 1) separates two sentences
            pieces.append((start, boundary.start(1)))
            start = boundary.end(1)
        pieces.append((start, len(line.group())))
        merged = []
        for s, e in pieces:
            if merged and merged[-1][1] - merged[-1][0] < min_chars:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        for s, e in merged:
            chunk = line.group()[s:e]
            lead = len(chunk) - len(chunk.lstrip())
            s, e = line.start() + s + lead, line.start() + s + len(chunk.rstrip())
            if e > s and re.search(r"\w", text[s:e]):
                spans.append((s, e))
    return spans


@register_task
class RAGTruth(TaskAdapter):
    name = f"{LEAF}/ragtruth"
    leaf = LEAF
    source = "RAGTruth, test split: QA (MS MARCO), data-to-text (Yelp) and news summarization (CNN/DM, Recent News) (Niu et al., 2024)"
    url = "https://github.com/ParticleMedia/RAGTruth"
    license = "MIT"
    tier = "B"
    criteria_source = "Adapted from the RAGTruth hallucination taxonomy (evident/subtle conflict, evident/subtle baseless information)"
    notes = ("One boolean per response sentence: a sentence is unsupported if it overlaps any human-annotated hallucination "
             "span (including spans marked implicit_true, i.e. true but absent from the context, which RAGTruth counts as "
             "baseless). Sentences are split deterministically at sentence-final punctuation and line breaks. Responses from "
             "GPT-4, GPT-3.5, Llama-2-7B/13B/70B-chat and Mistral-7B-Instruct; responses flagged as truncated or as incorrect refusals are dropped.")

    def load(self):
        base = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/"
        sources = {}
        with fetch(base + "source_info.jsonl", "ragtruth_source_info.jsonl").open(encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                sources[s["source_id"]] = s
        out = []
        with fetch(base + "response.jsonl", "ragtruth_response.jsonl").open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["split"] != "test" or r["quality"] != "good" or r["source_id"] not in sources:
                    continue
                spans = sentence_spans(r["response"])
                if not spans:
                    continue
                marked = [(l["start"], l["end"]) for l in r["labels"]]
                supported = [not any(ls < e and le > s for ls, le in marked) for s, e in spans]
                source = sources[r["source_id"]]
                out.append({"id": r["id"], "task_type": source["task_type"], "prompt": source["prompt"], "response": r["response"],
                            "sentences": [r["response"][s:e] for s, e in spans], "supported": supported, "model": r["model"]})
        return out

    def label_of(self, record):
        return f"{record['task_type']}|{all(record['supported'])}"

    def to_item(self, record, rng):
        questions, gold = {}, {}
        for i, (sentence, ok) in enumerate(zip(record["sentences"], record["supported"]), 1):
            questions[f"sentence_{i}"] = Question(type="boolean", instructions=RAGTRUTH_QUESTION.format(i=i, sentence=sentence))
            gold[f"sentence_{i}"] = Gold(label="true" if ok else "false")
        state = {"task_prompt": record["prompt"], "response": record["response"]}
        return self.item(state, questions, gold, task_type=record["task_type"], response_model=record["model"], original_id=record["id"])


# BEGIN taxonomy of response types (Dziri et al., 2022, Section 3.1), condensed.
BEGIN_OPTIONS = {
    "fully_attributable": "Fully attributable: the response conveys information that can be completely supported by the provided knowledge.",
    "not_attributable": ("Not fully attributable: the response contains at least some information that cannot be verified given "
                         "the knowledge, regardless of whether it is true in the real world; this includes hallucinations, "
                         "statements contradicting the knowledge, and off-topic information external to it."),
    "generic": ("Generic: the response is general enough to fit a large number of possible contexts (e.g. \"I don't know "
                "about that\", \"Hello there!\"); it is vague and does not provide new information."),
}
BEGIN_LABELS = {"Fully attributable": "fully_attributable", "Not fully attributable": "not_attributable", "Generic": "generic"}


@register_task
class BEGIN(TaskAdapter):
    name = f"{LEAF}/begin"
    leaf = LEAF
    source = "BEGIN, knowledge-grounded dialogue responses, test split (Dziri et al., 2022)"
    url = "https://github.com/google/BEGIN-dataset"
    license = "CC BY 4.0 (WoW and CMU-DoG parts); CDLA-Sharing 1.0 (Topical-Chat part)"
    tier = "B"
    criteria_source = "Category definitions adapted from the BEGIN taxonomy of response types (Dziri et al., 2022, Section 3.1)"
    notes = "Responses of GPT-2, T5, CTRL-Dialog and DoHA fine-tuned on Wizard of Wikipedia, CMU-DoG and Topical-Chat; human attribution labels."
    FILES = {"wow": "wow/begin_test_wow.tsv", "cmu": "cmu-dog/begin_test_cmu.tsv", "tc": "topicalchat/begin_test_tc.tsv"}

    def load(self):
        out = []
        for key, path in self.FILES.items():
            local = fetch(f"https://raw.githubusercontent.com/google/BEGIN-dataset/main/{path}", f"begin_test_{key}.tsv")
            with local.open(encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    if r["begin_label"] in BEGIN_LABELS and r["response"].strip() and r["knowledge"].strip():
                        out.append(r)
        return out

    def label_of(self, record):
        return f"{record['data_source']}|{record['begin_label']}"

    def to_item(self, record, rng):
        question = Question(
            type="choice",
            instructions="Which category describes the response with respect to the knowledge it should be grounded in?",
            criteria=shuffled_options(BEGIN_OPTIONS, rng),
        )
        state = {"knowledge": record["knowledge"], "previous_turn": record["message"], "response": record["response"]}
        gold = Gold(label=BEGIN_LABELS[record["begin_label"]])
        return self.item(state, {"attribution": question}, {"attribution": gold}, data_source=record["data_source"], response_model=record["model_name"].lower())
