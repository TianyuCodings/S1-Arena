"""Leaf: step correctness (这一步对不对).

Two candidate next steps from the same point of a tool-using agent's run: one correct and one
plausible but wrong. The question is which one is correct; every pair is asked in both orders.
Gold: BFCL's multi-turn ground truth versus a model turn that BFCL's execution checks reject
(TRBench), and a human-corrected ReAct step or action plan versus the model's original one that
annotators marked wrong (ToolComp).
"""

from __future__ import annotations

import json
import re

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ._sources import hub_file
from .action_critic import trb_context, trb_names, trb_rows

LEAF = "step/step_correctness"
PAIR = {"A": "Response A is the correct one.", "B": "Response B is the correct one."}

# ---------- TRBench-BFCL multi-turn ----------

# Adapted from TRBench's critique prompt: task sentence and evaluation criteria, with the output format removed.
TRB_QUESTION = (
    "Given the conversation history (tools, earlier turns and tool results) and two candidate assistant responses for the current "
    "user turn, exactly one of which is correct, decide which response is correct. Criteria: the available tools must be fully "
    "and appropriately leveraged to meet the requirements; tool call names must be valid, correct and complete; tool call arguments "
    "must be valid, correct and complete; fabrication, including information not provided by the user, conflicting with user input "
    "or not derived from the tools, is wrong; repetitive or unnecessary tool calls are wrong; the calls must leave the environment "
    "in the state the user asked for."
)
TRB_PAIRS_PER_TASK = 2


@register_task
class TRBenchMultiTurnPairs(TaskAdapter):
    name = f"{LEAF}/trbench_multi_turn"
    leaf = LEAF
    source = "TRBench-BFCL, multi-turn base category (Li et al., 2025, ToolRM), built on BFCL v3 multi-turn"
    url = "https://huggingface.co/datasets/RioLee/TRBench-BFCL"
    license = "Apache-2.0 (Hugging Face card and LICENSE file; the GitHub README says CC BY-NC-SA 4.0)"
    tier = "B"
    criteria_source = "Adapted from TRBench's critique prompt (task sentence and evaluation criteria)"
    items_per_record = 2
    notes = ("One user turn of a BFCL multi-turn task (file system, trading, travel, messaging and other simulated APIs), with the "
             "earlier turns and tool results. Correct: BFCL's ground-truth calls for the turn; wrong: the calls of a frontier model "
             "(Gemini 2.5 Pro, Claude 3.7 Sonnet, GPT-4o, o1, DeepSeek-R1, xLAM-2-70B, Qwen3-32B) that fail BFCL's execution checks "
             "(wrong final environment state or wrong execution results). Empty model turns are dropped, at most two pairs per task "
             "are used, and dotted function names forced into underscores by function-calling APIs are restored. Each pair is asked "
             "in both orders.")

    def load(self):
        groups: dict[str, list] = {}
        for r in trb_rows(("multi_turn_base",)):
            if r["error_type"].endswith("empty_turn_model_response") or not str(r["reject_response"]).strip():
                continue
            groups.setdefault(r["raw_id"], []).append(r)
        rows = []
        for rid, pairs in sorted(groups.items()):
            pairs.sort(key=lambda r: (r["reject_response_from"], len(r["chat_history"])))
            for r in pairs[:TRB_PAIRS_PER_TASK]:
                rows.append({"id": rid, "chat": [dict(m) for m in r["chat_history"]], "correct": r["chosen_response"],
                             "wrong": r["reject_response"], "from": r["reject_response_from"], "error": r["error_type"]})
        return rows

    def label_of(self, record):
        return record["error"]

    def to_items(self, record, rng):
        tools, conversation = trb_context(record["chat"])
        responses = {"correct": trb_names(record["correct"], tools), "wrong": trb_names(record["wrong"], tools)}
        question = Question(type="choice", instructions=TRB_QUESTION, criteria=dict(PAIR))
        items = []
        for first in ("correct", "wrong"):
            second = "wrong" if first == "correct" else "correct"
            state = {"tools": tools, "conversation": conversation, "response_a": responses[first], "response_b": responses[second]}
            gold = Gold(label="A" if first == "correct" else "B")
            items.append(self.item(state, {"correct_response": question}, {"correct_response": gold}, original_id=record["id"],
                                   order="correct_first" if first == "correct" else "correct_second", wrong_from=record["from"],
                                   checker_error=record["error"]))
        return items


# ---------- ToolComp ----------

TC_REPO = "vaskarnath/toolcomp_process_supervision_eval"
TC_DEFAULT_DATE = "Tuesday, September 03, 2024"  # the date ToolComp's prompts use when a prompt has no historical date
# Adapted from ToolComp's pairwise LLM-as-judge prompts (prompts/llm_as_judge.py); output format removed, 'tie' dropped since
# one candidate was marked wrong by the annotators and the other is their correction.
TC_REACT = (
    "You are an expert judge of tool calls. A ReAct assistant (Thought, Action, Action Input, then an Observation from the tool) "
    "answers the question with the tools listed in the state, using information accurate up to the current date given there; the "
    "'finish' action ends the task with the final answer. Given the question, the action plan and the history of past actions "
    "(assume they are optimal), judge the two candidate next ReAct steps and decide which one is better in progressing to the "
    "final answer. Try not to penalize a candidate step for formatting; select based on which candidate step is better in "
    "progressing to the final answer."
)
TC_PLAN = (
    "You are an expert planner of tool calls. Given the question and the tools listed in the state, judge which of the two action "
    "plans is better suited to answer the question, i.e. which one is better at outlining the correct steps to get to the final "
    "answer. An action plan should not make specific tool calls or try to answer the question directly; it lists the high-level "
    "steps and the tool to use for each."
)
_ESCAPED = re.compile(r"\\([_*#\[\]`~>])")


def tc_text(value) -> str:
    """Field text with the traces of hand editing removed: markdown escapes, extra whitespace, JSON spacing."""
    text = _ESCAPED.sub(r"\1", str((value or {}).get("text") or "")).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return json.dumps(parsed, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        pass
    text = re.sub(r"^\{\s+", "{", re.sub(r"\s+\}$", "}", text))  # hand-typed '{ "query": ... }' (JSON with raw newlines does not parse)
    return re.sub(r"[ \t]+", " ", re.sub(r"\s*\n\s*", "\n", text))


def tc_step(step: dict, observation: bool = True) -> str:
    parts = [f"Thought: {tc_text(step.get('thought'))}", f"Action: {tc_text(step.get('action'))}",
             f"Action Input: {tc_text(step.get('action_input'))}"]
    if observation:
        parts.append(f"Observation: {tc_text(step.get('observation'))}")
    return "\n".join(parts)


@register_task
class ToolCompPairs(TaskAdapter):
    name = f"{LEAF}/toolcomp"
    leaf = LEAF
    source = "ToolComp process-supervision pairs, human-corrected versus original ReAct steps and action plans (Nath et al., 2025)"
    url = "https://huggingface.co/datasets/vaskarnath/toolcomp_process_supervision_eval"
    license = "Not stated (the sibling repository vaskarnath/toolcomp is Apache-2.0)"
    tier = "B"
    criteria_source = "Adapted from ToolComp's pairwise LLM-as-judge prompts for ReAct steps and action plans"
    items_per_record = 2
    notes = ("Multi-tool questions (search, Wolfram Alpha, Python, weather, stocks and more) answered by a FireFunction-v1 ReAct "
             "agent. Human annotators labelled every step; where a step was wrong they rewrote it to be correct and the run "
             "continued from the correction. The pair is the rewrite (correct) versus the original step (wrong), for ReAct steps "
             "and for initial action plans, with each candidate's observation as in ToolComp's judge prompt. Hand-editing traces "
             "(markdown escapes, JSON spacing, trailing whitespace) are normalized away, and pairs that are identical after "
             "normalization are dropped. Each pair is asked in both orders.")

    def load(self):
        import pandas as pd
        from huggingface_hub import HfApi

        files = [f for f in HfApi().list_repo_files(TC_REPO, repo_type="dataset") if f.endswith(".parquet")]
        rows = []
        for f in sorted(files):
            for i, r in enumerate(pd.read_parquet(hub_file(TC_REPO, f)).to_dict("records")):
                history = [dict(h) for h in (json.loads(r["history"]) if isinstance(r["history"], str) else r["history"])]
                pref, dis = dict(r["preferred"]), dict(r["dispreferred"])
                plan_pair = bool((pref.get("action plan") or {}).get("text"))
                if plan_pair:
                    good, bad = tc_text(pref.get("action plan")), tc_text(dis.get("action plan"))
                else:
                    good, bad = tc_step(pref), tc_step(dis)
                    if tc_step(pref, False) == tc_step(dis, False):
                        continue
                if not good or not bad or good == bad:
                    continue
                date = r.get("historical_date")
                date = date if isinstance(date, str) and date.strip() and date != "NaN" else TC_DEFAULT_DATE
                plan = next((tc_text(h.get("action plan")) for h in history if (h.get("action plan") or {}).get("text")), "")
                past = [tc_step(h) for h in history if (h.get("action") or {}).get("text")]
                rows.append({"key": f"{f}:{i}", "kind": "plan" if plan_pair else "react", "question": r["prompt"], "tools": r["tools"],
                             "date": date, "plan": plan, "past": past, "good": good, "bad": bad})
        return rows

    def label_of(self, record):
        return record["kind"]

    def to_items(self, record, rng):
        plan = record["kind"] == "plan"
        question = Question(type="choice", instructions=TC_PLAN if plan else TC_REACT,
                            criteria={"A": f"Candidate {'action plan' if plan else 'step'} A is better.",
                                      "B": f"Candidate {'action plan' if plan else 'step'} B is better."})
        base = {"question": record["question"], "tools": f"{record['tools']} and finish (ends the task with the final answer)",
                "current_date": record["date"]}
        if not plan:
            base["action_plan"] = record["plan"]
            base["history_of_past_actions"] = "\n\n".join(record["past"]) or "(no past actions yet)"
        items = []
        for first in ("good", "bad"):
            second = "bad" if first == "good" else "good"
            label = "action_plan" if plan else "step"
            state = {**base, f"candidate_{label}_a": record[first], f"candidate_{label}_b": record[second]}
            gold = Gold(label="A" if first == "good" else "B")
            items.append(self.item(state, {"better": question}, {"better": gold}, kind=record["kind"], original_key=record["key"],
                                   order="correct_first" if first == "good" else "correct_second"))
        return items
