"""Leaf: escalate to a human or a stronger LLM (升级到人工或 LLM).

A cheap model has drafted an answer; the classifier decides whether to hand the request on.
Gold follows the downstream outcome. Escalating to a stronger model is right only when that
model answers correctly and the draft is wrong (AutoMix's "needy" case; when both fail,
escalating costs more and fixes nothing). Escalating to a human expert, who is taken to be
right, is correct exactly when the draft is wrong.
"""

from __future__ import annotations

import functools
import json
import re
import zipfile

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch, hf_rows
from . import outcomes as oc

LEAF = "route/escalation"


# ---------- AutoMix: Llama-2-13B drafts with self-verification, Llama-2-70B as the larger model ----------

AUTOMIX_ZIP = "https://drive.usercontent.google.com/download?id=1dhyt7UuYumk9Gae9eJ_mpTVrLeSTuRht&export=download&confirm=t"
OPTION = re.compile(r"\(([A-D])\)\s*(.+?)\s*(?=\n \([A-D]\)|$)", re.S)


@functools.lru_cache(maxsize=1)
def automix_quality() -> tuple[dict, ...]:
    """QuALITY validation rows whose answer text and answer letter agree (about 96%)."""
    rows = []
    with zipfile.ZipFile(fetch(AUTOMIX_ZIP, "automix_data.zip")) as z, z.open("automix_release_with_decision.jsonl") as f:
        for line in f:
            if b'"quality"' not in line:
                continue
            r = json.loads(line)
            if r["dataset"] != "quality" or r["split"] != "val":
                continue
            options = {k: v.strip() for k, v in OPTION.findall(r["question"])}
            if options.get(r["correct_option"]) != str(r["output"]).strip():
                continue  # the released letter contradicts the released answer text
            rows.append({"id": r["pid"], "document": r["base_ctx"].strip(), "question": r["question"].strip(),
                         "draft": str(r["llama13b_pred_ans"]).strip(), "p_ver": float(r["p_ver_13b"]), "category": r["category"]})
    return tuple(rows)


@register_task
class AutoMixQuality(TaskAdapter):
    name = f"{LEAF}/automix_quality"
    leaf = LEAF
    source = "AutoMix cascade outputs on QuALITY, validation (Aggarwal et al., 2024)"
    url = "https://github.com/automix-llm/automix"
    license = "Apache-2.0 (AutoMix); QuALITY CC BY 4.0"
    tier = "B"
    criteria_source = "Adapted from AutoMix's routing categories: route to the larger model when it does better than the small one"
    notes = ("Gold is AutoMix's 'needy' category (70B right, 13B wrong); 'good', 'hopeless' and 13B-better cases are 'do not "
             "escalate'. The self-verification score is AutoMix's own confidence signal for the draft. Rows whose released "
             "answer letter contradicts the released answer text (about 4%) are left out. The data is a 252 MB zip on Google Drive.")

    def load(self):
        return list(automix_quality())

    def label_of(self, record):
        return record["category"]

    def to_item(self, record, rng):
        instructions = ("A small model (Llama-2-13B chat) answered this multiple-choice question about the document. Should the "
                        "question be escalated to the larger model (Llama-2-70B chat)? Escalate only when the larger model would "
                        "answer correctly and the small model's draft is wrong.")
        state = {"document": record["document"], "question": record["question"], "draft_answer": record["draft"],
                 "self_verification": f"{record['p_ver']:.2f} (share of the small model's own verification samples that judged the draft correct)"}
        gold = Gold(label="true" if record["category"] == "NEEDY" else "false")
        return self.item(state, {"escalate": Question(type="boolean", instructions=instructions)}, {"escalate": gold},
                         category=record["category"], original_id=record["id"])


# ---------- LiveBench: GPT-4o mini drafts, o1-preview as the stronger model ----------

LB_DRAFT, LB_STRONG = "gpt-4o-mini-2024-07-18", "o1-preview-2024-09-12"


@register_task
class LiveBenchEscalation(TaskAdapter):
    name = f"{LEAF}/livebench"
    leaf = LEAF
    source = "LiveBench model answers and judgments: GPT-4o mini drafts, o1-preview as the stronger model (White et al., 2025)"
    url = "https://huggingface.co/datasets/livebench/model_answer"
    license = oc.LIVEBENCH_LICENSE
    tier = "C"
    criteria_source = "Written; gold combines LiveBench's objective scores of both models on the same question"
    notes = ("Code generation, code completion, typo fixing and Connections questions of June-July 2024 with a stored GPT-4o mini "
             "answer. Escalate = GPT-4o mini scored 0 and o1-preview scored 1.")

    def load(self):
        answers = {}
        for r in hf_rows("livebench/model_answer", None, "leaderboard", revision="bc43a286b08243d6dcfa89ba2e906283e9adac9c"):
            if r["model_id"] == LB_DRAFT and r["choices"]:
                answers[r["question_id"]] = r["choices"][0]["turns"][-1]
        out = []
        for r in oc.livebench():
            if r["id"] in answers and LB_DRAFT in r["score"] and LB_STRONG in r["score"]:
                needy = r["score"][LB_DRAFT] < 1 and r["score"][LB_STRONG] >= 1
                out.append(dict(r, draft=answers[r["id"]], label="true" if needy else "false",
                                case=f"draft {'right' if r['score'][LB_DRAFT] >= 1 else 'wrong'}, strong {'right' if r['score'][LB_STRONG] >= 1 else 'wrong'}"))
        return out

    def label_of(self, record):
        return f"{record['case']}|{record['group']}"

    def to_item(self, record, rng):
        instructions = ("GPT-4o mini wrote the draft answer to this request. Should the request be escalated to the stronger and "
                        "costlier o1-preview? Escalate only when o1-preview would get it right and the draft is wrong. "
                        + oc.LIVEBENCH_PROTOCOL)
        state = {"request": record["prompt"], "draft_answer": record["draft"]}
        return self.item(state, {"escalate": Question(type="boolean", instructions=instructions)}, {"escalate": Gold(label=record["label"])},
                         case=record["case"], group=record["group"], original_id=record["id"])


# ---------- RouterArena: GPT-4o mini drafts, a human expert as the fallback ----------

@register_task
class RouterArenaEscalation(TaskAdapter):
    name = f"{LEAF}/routerarena_human"
    leaf = LEAF
    source = "RouterArena cached GPT-4o mini replies, full split (Lu et al., 2025)"
    url = oc.ROUTERARENA_URL
    license = oc.ROUTERARENA_LICENSE
    tier = "C"
    criteria_source = "Written; gold is RouterArena's automatic score of the draft (escalate when it is wrong)"
    notes = ("Escalation to a human expert, who is taken to answer correctly, so gold = the draft is scored wrong. Only "
             "requests scored 0/1. The authors ask that RouterArena be used for evaluation only.")

    def load(self):
        return [dict(r, draft=r["outcome"]["gpt-4o-mini"]["answer"], label="true" if r["outcome"]["gpt-4o-mini"]["score"] < 1 else "false")
                for r in oc.routerarena().values() if "gpt-4o-mini" in r["outcome"]]

    def label_of(self, record):
        return f"{record['label']}|{record['family']}"

    def to_item(self, record, rng):
        instructions = ("A low-cost model (GPT-4o mini) wrote the draft answer to this request. Should the request be escalated "
                        "to a human expert instead of returning the draft? Escalate when the draft's final answer is wrong; the "
                        "expert is assumed to answer correctly. The draft is scored automatically against the reference answer "
                        "(option letter, exact match, math answer check or code tests).")
        state = {"request": record["prompt"], "draft_answer": record["draft"]}
        return self.item(state, {"escalate": Question(type="boolean", instructions=instructions)}, {"escalate": Gold(label=record["label"])},
                         family=record["family"], original_id=record["id"])
