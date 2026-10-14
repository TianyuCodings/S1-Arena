"""Leaf: summary consistency (摘要一致性).

Is a summary (or each summary sentence) factually consistent with its source document?
All labels are human judgments: SummEdits (edited summaries labelled consistent or
inconsistent), TofuEval (expert sentence labels on topic-focused meeting summaries) and
FaithBench (annotated hallucinations in summaries by ten modern LLMs).
"""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch, hf_rows

LEAF = "judge/summary_consistency"

# TofuEval zero-shot prompts (Tang et al., 2024, Appendix), verbatim definitions.
SUMMARY_LEVEL = ("Determine if the summary is factually consistent with the document provided above. A summary is factually "
                 "consistent with the document if all information in the summary can be entailed (either stated or implied) by the document.")
SENTENCE_LEVEL = ("Determine if the sentence is factually consistent with the document provided above. A sentence is factually "
                  "consistent with the document if it can be entailed (either stated or implied) by the document. Sentence {i} of the summary: \"{sentence}\"")


@register_task
class SummEdits(TaskAdapter):
    name = f"{LEAF}/summedits"
    leaf = LEAF
    source = "SummEdits, 10 domains (Laban et al., 2023)"
    url = "https://huggingface.co/datasets/Salesforce/summedits"
    license = "CC BY 4.0"
    tier = "B"
    criteria_source = "Summary-level consistency definition from the TofuEval prompts; SummEdits annotators labelled each edit as consistent or inconsistent with the same notion"
    notes = ("Seed summaries verified as consistent were edited (edits proposed by an LLM) and each edited summary was labelled "
             "consistent or inconsistent by a human annotator; borderline cases were removed by the authors. The LLM-assigned "
             "edit types are not used.")

    def load(self):
        from huggingface_hub import hf_hub_download

        from ...env import load_env

        load_env()
        path = hf_hub_download("Salesforce/summedits", "summedits.json", repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        return [r for r in rows if r.get("label") in (0, 1) and r.get("doc") and r.get("summary")]

    def label_of(self, record):
        return f"{record['domain']}|{record['label']}"

    def to_item(self, record, rng):
        question = Question(type="boolean", instructions=SUMMARY_LEVEL)
        gold = Gold(label="true" if record["label"] == 1 else "false")
        return self.item({"document": record["doc"], "summary": record["summary"]}, {"consistent": question}, {"consistent": gold},
                         domain=record["domain"], original_id=record["id"])


@register_task
class TofuEvalMeetingBank(TaskAdapter):
    name = f"{LEAF}/tofueval_meetingbank"
    leaf = LEAF
    source = "TofuEval, MeetingBank part, test split (Tang et al., 2024)"
    url = "https://github.com/amazon-science/tofueval"
    license = "MIT-0 (annotations); MeetingBank transcripts CC BY-NC-SA 4.0"
    tier = "A"
    criteria_source = "Sentence-level consistency definition verbatim from the TofuEval zero-shot prompt (Tang et al., 2024, Appendix)"
    notes = ("Topic-focused summaries of city-council meeting transcripts by six LLMs; each summary sentence was labelled by "
             "expert linguistic annotators, and every sentence is one boolean question. Transcripts come from "
             "lytang/MeetingBank-transcript. The MediaSum part is left out because its transcripts need a separate large download.")

    def load(self):
        base = "https://raw.githubusercontent.com/amazon-science/tofueval/main/"
        with fetch(base + "document_ids_dev_test_split.json", "tofueval_split.json").open(encoding="utf-8") as f:
            test_ids = set(json.load(f)["test"]["meetingbank"])
        docs = {r["meeting_id"]: r["source"] for r in hf_rows("lytang/MeetingBank-transcript", None, "test") if r["meeting_id"] in test_ids}
        path = fetch(base + "factual_consistency/meetingbank_factual_eval_test.csv", "tofueval_meetingbank_test.csv")
        summaries = defaultdict(list)
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["doc_id"] in docs and r["sent_label"] in ("yes", "no"):
                    summaries[(r["doc_id"], r["topic"], r["model_name"])].append(r)
        out = []
        for (doc_id, topic, model), sentences in summaries.items():
            sentences.sort(key=lambda r: int(r["sent_idx"]))
            out.append({"doc_id": doc_id, "topic": topic, "model": model, "document": docs[doc_id],
                        "sentences": [s["summ_sent"].strip() for s in sentences], "labels": [s["sent_label"] == "yes" for s in sentences]})
        return out

    def label_of(self, record):
        return str(all(record["labels"]))

    def to_item(self, record, rng):
        questions, gold = {}, {}
        for i, (sentence, ok) in enumerate(zip(record["sentences"], record["labels"]), 1):
            questions[f"sentence_{i}"] = Question(type="boolean", instructions=SENTENCE_LEVEL.format(i=i, sentence=sentence))
            gold[f"sentence_{i}"] = Gold(label="true" if ok else "false")
        state = {"document": record["document"], "summary_topic": record["topic"], "summary": " ".join(record["sentences"])}
        return self.item(state, questions, gold, original_id=record["doc_id"], summarizer=record["model"])


@register_task
class FaithBench(TaskAdapter):
    name = f"{LEAF}/faithbench"
    leaf = LEAF
    source = "FaithBench, summaries by 10 modern LLMs (Bao et al., 2025)"
    url = "https://github.com/vectara/FaithBench"
    license = "CC BY-NC-SA 4.0"
    tier = "B"
    criteria_source = "Summary-level consistency definition from the TofuEval prompts; FaithBench labels map Consistent to true and Unwanted hallucination to false"
    notes = ("FaithBench keeps samples on which hallucination detectors disagree, so they are hard; annotators often disagree "
             "too. Only samples whose annotations agree are used: every annotator found no hallucination (Consistent) or the "
             "mildest label given was Unwanted. Samples labelled Benign or Questionable are left out.")

    def load(self):
        path = fetch("https://raw.githubusercontent.com/vectara/FaithBench/main/FaithBench.csv", "faithbench.csv")
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(io.StringIO(f.read())))
        out = []
        for i, r in enumerate(rows):
            if r["worst-label"] == r["best-label"] and r["worst-label"] in ("Consistent", "Unwanted"):
                out.append({"index": i, "source": r["source"], "summary": r["summary"].strip(), "summarizer": r["LLM"], "consistent": r["worst-label"] == "Consistent"})
        return out

    def label_of(self, record):
        return str(record["consistent"])

    def to_item(self, record, rng):
        question = Question(type="boolean", instructions=SUMMARY_LEVEL)
        gold = Gold(label="true" if record["consistent"] else "false")
        return self.item({"document": record["source"], "summary": record["summary"]}, {"consistent": question}, {"consistent": gold},
                         summarizer=record["summarizer"], original_index=record["index"])
