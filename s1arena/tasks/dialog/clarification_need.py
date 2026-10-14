"""Leaf: answer directly or ask a clarifying question first (该直接回答还是先澄清)."""

from __future__ import annotations

import json
import zipfile

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch, hf_rows

LEAF = "dialog/clarification_need"


@register_task
class AmbigQA(TaskAdapter):
    name = f"{LEAF}/ambigqa"
    leaf = LEAF
    source = "AmbigQA / AmbigNQ, validation (Min et al., 2020)"
    url = "https://huggingface.co/datasets/sewon/ambig_qa"
    license = "CC BY-SA 3.0"
    tier = "B"
    criteria_source = "Adapted from the AmbigQA definition of ambiguity: a question with multiple plausible interpretations, each with a different answer"
    notes = ("Each question was annotated by one to three crowdworkers as having a single answer or multiple interpretations; soft label = "
             "share of annotators who found multiple interpretations (when there are two or more), hard label = majority. Questions "
             "with a tied vote are dropped. The questions are those of NQ-open validation, which know/factual_judgment also uses.")

    def load(self):
        out = []
        for r in hf_rows("sewon/ambig_qa", "light", "validation"):
            types = r["annotations"]["type"]
            multiple = sum(t == "multipleQAs" for t in types)
            if not types or 2 * multiple == len(types):
                continue
            out.append({"id": r["id"], "question": r["question"], "votes": multiple, "annotators": len(types)})
        return out

    def label_of(self, record):
        return f"{2 * record['votes'] > record['annotators']}|{record['annotators']}"

    def to_item(self, record, rng):
        question = Question(
            type="boolean",
            instructions="A user asks an assistant this question. Is the question ambiguous, i.e. does it have more than one plausible "
            "interpretation, each leading to a different answer (for example, it leaves open which event, time, entity or aspect is "
            "meant), so that the assistant should ask a clarifying question before answering?",
        )
        share = record["votes"] / record["annotators"]
        dist = {"true": share, "false": 1.0 - share} if record["annotators"] >= 2 else None
        gold = Gold(label="true" if 2 * record["votes"] > record["annotators"] else "false", dist=dist)
        return self.item({"question": record["question"]}, {"ambiguous": question}, {"ambiguous": gold}, original_id=record["id"], annotators=record["annotators"])


SHARC_OPTIONS = {
    "yes": "Answer the user's question with Yes: the rule text, the user's scenario and the conversation so far determine that the answer is yes.",
    "no": "Answer the user's question with No: the rule text, the user's scenario and the conversation so far determine that the answer is no.",
    "irrelevant": "Reply that the question is irrelevant: the rule text does not answer this question.",
    "ask_follow_up": "Ask the user a follow-up question first: the rule text applies, but the scenario and the conversation so far do not yet give enough information to determine the answer.",
}


@register_task
class ShARC(TaskAdapter):
    name = f"{LEAF}/sharc"
    leaf = LEAF
    source = "ShARC conversational machine reading with rules, test (Saeidi et al., 2018)"
    url = "https://sharc-data.github.io/"
    license = "CC BY-SA 3.0"
    tier = "B"
    criteria_source = "Task description adapted from the ShARC homepage and the four outcomes of the ShARC decision task (Yes, No, Irrelevant, or a follow-up question)"
    notes = ("Rule texts are from UK and US government websites; crowdworkers wrote the questions, scenarios, follow-up questions and "
             "answers. The test split was released in February 2024. Each utterance is one decision point of a dialogue.")

    def load(self):
        path = fetch("https://sharc-data.github.io/data/sharc1-official.zip", "sharc1-official.zip")
        with zipfile.ZipFile(path) as zf, zf.open("sharc1-official/json/sharc_test.json") as f:
            return json.load(f)

    @staticmethod
    def outcome(record) -> str:
        return {"Yes": "yes", "No": "no", "Irrelevant": "irrelevant"}.get(record["answer"], "ask_follow_up")

    def label_of(self, record):
        return f"{self.outcome(record)}|{min(len(record['history']), 2)}"

    def to_item(self, record, rng):
        question = Question(
            type="choice",
            instructions="An assistant answers users' questions using a supporting rule text. The question is often underspecified, in "
            "the sense that it does not provide enough information to be answered directly; the assistant can use the rule text to infer "
            "what needs to be asked in order to determine the final answer. What should the assistant do next?",
            criteria=shuffled_options(SHARC_OPTIONS, rng),
        )
        history = "\n".join(f"Assistant: {h['follow_up_question']}\nUser: {h['follow_up_answer']}" for h in record["history"])
        state = {"rule_text": record["snippet"], "user_scenario": record["scenario"], "user_question": record["question"], "follow_up_conversation": history}
        gold = Gold(label=self.outcome(record))
        return self.item(state, {"next_step": question}, {"next_step": gold}, original_id=record["utterance_id"], source_url=record["source_url"])
