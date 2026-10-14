"""Leaf: is the retrieved context sufficient to answer the question (RAG 上下文是否充分)?

know/answerability already uses SQuAD 2.0 and MuSiQue, so this leaf uses other data: web search results judged
by MS MARCO editors, movie plots from DuoRC's ParaphraseRC, and IIRC questions whose evidence from linked
articles is kept or removed.
"""

from __future__ import annotations

import json
import random
import re

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch, hf_rows

LEAF = "ret/context_sufficiency"


@register_task
class MSMarcoQnA(TaskAdapter):
    name = f"{LEAF}/ms_marco_qna"
    leaf = LEAF
    source = "MS MARCO question answering v2.1, dev set (Bajaj et al., 2018)"
    url = "https://huggingface.co/datasets/microsoft/ms_marco"
    license = "MS MARCO terms: non-commercial research use"
    tier = "A"
    criteria_source = "MS MARCO novice task definition: predict whether the question can be answered based only on the information contained in the provided passages"
    notes = ("Bing queries with the ~10 passages Bing retrieved for them; human editors either wrote an answer from the passages or "
             "marked 'No Answer Present' (45% of the dev set). The passages the editors used (is_selected) and their URLs are withheld.")

    def load(self):
        return hf_rows("microsoft/ms_marco", "v2.1", "validation")

    def answerable(self, record) -> bool:
        return not (len(record["answers"]) == 1 and record["answers"][0].strip() == "No Answer Present.")

    def label_of(self, record):
        return f"{self.answerable(record)}|{record['query_type']}"

    def to_item(self, record, rng):
        passages = [p.strip() for p in record["passages"]["passage_text"] if p.strip()]
        if not passages or not record["answers"]:
            return None
        question = Question(
            type="boolean",
            instructions="Can the question be answered based only on the information contained in the provided passages?",
        )
        state = {"question": record["query"].strip(), "passages": "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))}
        gold = Gold(label="true" if self.answerable(record) else "false")
        return self.item(state, {"sufficient": question}, {"sufficient": gold}, query_id=record["query_id"], query_type=record["query_type"])


@register_task
class DuoRCParaphrase(TaskAdapter):
    name = f"{LEAF}/duorc_paraphrase"
    leaf = LEAF
    source = "DuoRC ParaphraseRC, test (Saha et al., 2018)"
    url = "https://huggingface.co/datasets/ibm-research/duorc"
    license = "MIT"
    tier = "C"
    criteria_source = "Written; gold is ParaphraseRC's no-answer flag, set when the worker reading this plot found no answer in it"
    sampling = "balanced"
    notes = ("Questions were written from a movie's Wikipedia plot and answered by other workers from its longer IMDb plot, which is "
             "the context shown; the plot often lacks the asked detail (15% of test questions). Sampled half and half so that "
             "insufficient contexts are not a small minority.")

    def load(self):
        return [r for r in hf_rows("ibm-research/duorc", "ParaphraseRC", "test") if r["plot"].strip() and r["question"].strip()]

    def label_of(self, record):
        return str(not record["no_answer"])

    def to_item(self, record, rng):
        question = Question(type="boolean", instructions="Does the plot contain enough information to answer the question about the movie?")
        state = {"movie_title": record["title"].strip(), "plot": record["plot"].strip(), "question": record["question"].strip()}
        gold = Gold(label="false" if record["no_answer"] else "true")
        return self.item(state, {"sufficient": question}, {"sufficient": gold}, question_id=record["question_id"])


def norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


@register_task
class IIRCEvidenceRemoved(TaskAdapter):
    name = f"{LEAF}/iirc"
    leaf = LEAF
    source = "IIRC, incomplete information reading comprehension, test (Ferguson et al., 2020)"
    url = "https://allenai.org/data/iirc"
    license = "CC BY 4.0"
    tier = "B"
    criteria_source = "Adapted from the IIRC setting: the Wikipedia paragraph alone lacks information that must come from the linked articles"
    items_per_record = 2
    notes = ("Each answerable question (span, value or yes/no answer) appears twice with the same Wikipedia paragraph and one or two "
             "snippets from other linked articles (evidence annotated for other questions on that paragraph): once with the annotated "
             "evidence from the question's own linked articles added (sufficient) and once without it (insufficient). Pairs are kept "
             "only if a span answer occurs in the sufficient context and neither an answer span from a linked article nor a "
             "computed value occurs in the insufficient one.")

    def load(self):
        with fetch("https://iirc-dataset.s3.us-west-2.amazonaws.com/iirc_test.json", "iirc_test.json").open(encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for p_index, passage in enumerate(data):
            questions = passage["questions"]
            for q_index, q in enumerate(questions):
                kind = q["answer"]["type"]
                evidence = [(c["passage"], c["text"].strip()) for c in q["context"] if c["passage"] != "main" and c["text"].strip()]
                if kind == "none" or not evidence:
                    continue
                own = {a.lower() for a, _ in evidence} | {a.lower() for a in q.get("question_links") or []}
                others = sorted({(c["passage"], c["text"].strip()) for q2 in questions if q2 is not q for c in q2["context"]
                                 if c["passage"] != "main" and c["passage"].lower() not in own and c["text"].strip()})
                if not others:
                    continue
                rng = random.Random(f"{self.name}:{p_index}:{q_index}")
                distractors = rng.sample(others, min(2, len(others)))
                spans = q["answer"].get("answer_spans") or []
                full = norm(" ".join([passage["text"]] + [t for _, t in evidence + distractors]))
                partial = norm(" ".join([passage["text"]] + [t for _, t in distractors]))
                if kind == "span" and not any(norm(s["text"]) and norm(s["text"]) in full for s in spans):
                    continue
                # answers found in a linked article, and computed values, must not be readable from the reduced context
                absent = [norm(s["text"]) for s in spans if s.get("passage") != "main"]
                if kind == "value":
                    absent.append(norm(str(q["answer"]["answer_value"])))
                if any(a and re.search(rf"\b{re.escape(a)}\b", partial) for a in absent):
                    continue
                rows.append({"id": f"{p_index}:{q_index}", "title": passage["title"], "paragraph": passage["text"].strip(), "question": q["question"].strip(),
                             "evidence": evidence, "distractors": distractors, "kind": kind})
        return rows

    def label_of(self, record):
        return record["kind"]

    def context(self, record, snippets, rng) -> str:
        snippets = list(snippets)
        rng.shuffle(snippets)
        parts = [f"[1] {record['title']}: {record['paragraph']}"]
        parts += [f"[{i + 2}] {article}: {text}" for i, (article, text) in enumerate(snippets)]
        return "\n\n".join(parts)

    def to_items(self, record, rng):
        question = Question(
            type="boolean",
            instructions=("Does the context contain all the information needed to answer the question? The first passage is a "
                          "Wikipedia paragraph; the others are snippets from articles it links to. Use only the context."),
        )
        items = []
        for sufficient, snippets in ((True, record["evidence"] + record["distractors"]), (False, record["distractors"])):
            state = {"question": record["question"], "context": self.context(record, snippets, rng)}
            gold = Gold(label="true" if sufficient else "false")
            items.append(self.item(state, {"sufficient": question}, {"sufficient": gold}, original_id=record["id"], answer_type=record["kind"],
                                   variant="evidence_kept" if sufficient else "evidence_removed"))
        return items
