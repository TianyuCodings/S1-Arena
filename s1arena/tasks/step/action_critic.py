"""Leaf: action critic (动作执行前的正确性判断).

Before a proposed action is executed, is it correct? One boolean question per item; the proposed
action's result is never shown. Gold: human step labels (AgentProcessBench), the expert
demonstration versus a model-proposed alternative (WebRewardBench on Mind2Web), and BFCL's
ground-truth call versus a model's call that BFCL's AST checker rejects (TRBench).
"""

from __future__ import annotations

import hashlib
import json
import re

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ._sources import fits, hub_file, hub_parquet_columns

LEAF = "step/action_critic"


def _parity(key: str) -> bool:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2 == 0


def render_messages(messages: list[dict]) -> str:
    """Chat messages with tool calls and tool results as plain text."""
    lines = []
    for m in messages:
        role, content = str(m.get("role") or "").lower(), str(m.get("content") or "").strip()
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            if content:
                lines.append(f"Assistant: {content}")
            for call in m.get("tool_calls") or []:
                fn = call.get("function") or call
                lines.append(f"Assistant calls {fn.get('name')}({fn.get('arguments')})")
        elif role == "tool":
            lines.append(f"Tool result ({m.get('name', 'tool')}): {content}")
    return "\n\n".join(lines)


# ---------- AgentProcessBench ----------

# Adapted from the AgentProcessBench annotation guide (annotation_platform/ANNOTATION_GUIDE_v1.md, v1.1): the core criterion
# and the definitions of a Positive and a Negative step, condensed; the step's own tool result is not shown.
APB_QUESTION = (
    "A tool-using agent is working on the task in the state. Judge the proposed next step (one assistant message) before its "
    "result is known. Core criterion: does this step make the task closer to being correctly completed? A step is correct if it "
    "is correct, reasonable and materially moves the task forward: the right tool at the right time, parameters that satisfy the "
    "schema and the task's constraints, correct understanding and use of earlier tool outputs, a necessary clarification, or an "
    "effective correction of an earlier error. It is incorrect if it introduces errors, deviates from the goal or harms the chance "
    "of completing the task: a wrong tool or wrong parameters, misreading earlier tool outputs, fabrication or unsupported "
    "assumptions, redundant repetition without new information, continuing on a wrong premise, violating constraints or format, "
    "or drifting off topic. A final answer is correct only if it is right and in the required format. Is the proposed step correct?"
)


@register_task
class AgentProcessBenchCritic(TaskAdapter):
    name = f"{LEAF}/agentprocessbench"
    leaf = LEAF
    source = "AgentProcessBench, HotpotQA and GAIA (text) subsets, human step labels (2026)"
    url = "https://huggingface.co/datasets/LulaCola/AgentProcessBench"
    license = "MIT"
    tier = "B"
    criteria_source = "Adapted from the AgentProcessBench annotation guide v1.1 (core criterion and the Positive / Negative definitions)"
    notes = ("Trajectories of tool-using agents answering HotpotQA and GAIA questions; every assistant message was labelled +1 / 0 / -1 "
             "by human annotators (89% agreement). The annotation platform showed DeepSeek-V3.2, GPT-5.2 and Claude 4.5 Sonnet labels "
             "as guidance, so the human labels are LLM-assisted. Items are steps up to and including the first -1 (later steps inherit "
             "errors by the guide's propagation rule); neutral steps are dropped; +1 is 'correct', -1 'incorrect'. The state is the "
             "conversation before the step; the step's tool result is not shown. The BFCL and tau2-bench subsets are not used "
             "(they overlap other sources).")
    SUBSETS = ("hotpotqa", "gaia_dev")

    def load(self):
        rows = []
        for subset in self.SUBSETS:
            with open(hub_file("LulaCola/AgentProcessBench", f"{subset}/test.jsonl"), encoding="utf-8") as f:
                trajectories = [json.loads(line) for line in f if line.strip()]
            for t in trajectories:
                labels = {int(k): int(v) for k, v in (t.get("step_labels") or {}).items()}
                for index in sorted(labels):
                    if labels[index] == 0:
                        continue
                    message = t["messages"][index]
                    if message.get("role") != "assistant":
                        continue
                    if fits(json.dumps(t["messages"][:index + 1], ensure_ascii=False), json.dumps(t.get("tools") or [], ensure_ascii=False)):
                        rows.append({"subset": subset, "id": t["total_index"], "index": index, "label": labels[index],
                                     "messages": t["messages"][:index + 1], "tools": t.get("tools") or []})
                    if labels[index] == -1:
                        break
        return rows

    def label_of(self, record):
        return f"{record['subset']}|{record['label']}"

    def to_item(self, record, rng):
        messages = record["messages"]
        tools = [t.get("function", t) for t in record["tools"]]
        state = {"tools": json.dumps(tools, ensure_ascii=False), "conversation_so_far": render_messages(messages[:-1]),
                 "proposed_step": render_messages(messages[-1:])}
        gold = Gold(label="true" if record["label"] == 1 else "false")
        return self.item(state, {"correct": Question(type="boolean", instructions=APB_QUESTION)}, {"correct": gold},
                         subset=record["subset"], trajectory=record["id"], message_index=record["index"])


# ---------- TRBench-BFCL (single-turn, parallel calls) ----------

TRB_REPO, TRB_FILE = "RioLee/TRBench-BFCL", "trbench_test_no_think.parquet"
# Adapted from the evaluation criteria in TRBench's critique prompt ('prompt' field).
TRB_QUESTION = (
    "An assistant with the tools in the state proposes the response below for the user's request, before any call is executed. "
    "Judge it by these criteria: the available tools must be fully and appropriately leveraged to meet the requirements; tool call "
    "names must be valid, correct and complete; tool call arguments must be valid, correct and complete; information must not be "
    "fabricated, conflict with the user's input or come from nowhere; repetitive or unnecessary calls are wrong. Is the proposed "
    "response correct?"
)
_TOOLS = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.S)


def trb_rows(sources: tuple[str, ...]) -> list[dict]:
    import pandas as pd

    frame = pd.read_parquet(hub_file(TRB_REPO, TRB_FILE))
    frame = frame[frame["task_source"].isin(sources)]
    return frame.to_dict("records")


def trb_context(chat_history) -> tuple[str, str]:
    """(tool list, conversation) from TRBench's chat history; the system message holds the tools."""
    messages = [dict(m) for m in chat_history]
    tools = ""
    if messages and str(messages[0].get("role")).lower() == "system":
        found = _TOOLS.findall(messages[0]["content"])  # the prompt also mentions an empty '<tools></tools>'
        tools = max(found, key=len) if found and max(found, key=len) else messages[0]["content"]
        messages = messages[1:]
    return tools, render_messages(messages)


_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def trb_names(response: str, tools: str) -> str:
    """Restore dotted function names that function-calling APIs force to underscores ('spotify_play' -> 'spotify.play'), as
    BFCL's checker does before comparing; otherwise the underscore would mark model responses."""
    names = set(re.findall(r'"name":\s*"([^"]+)"', tools))
    alias = {n.replace(".", "_"): n for n in names if "." in n}

    def fix(match):
        try:
            call = json.loads(match.group(1))
        except json.JSONDecodeError:
            return match.group(0)
        if isinstance(call, dict) and call.get("name") not in names and call.get("name") in alias:
            call["name"] = alias[call["name"]]
            return "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>"
        return match.group(0)

    return _CALL.sub(fix, response)


@register_task
class TRBenchCritic(TaskAdapter):
    name = f"{LEAF}/trbench_parallel"
    leaf = LEAF
    source = "TRBench-BFCL, single-turn parallel and parallel-multiple categories (Li et al., 2025, ToolRM)"
    url = "https://huggingface.co/datasets/RioLee/TRBench-BFCL"
    license = "Apache-2.0 (Hugging Face card and LICENSE file; the GitHub README says CC BY-NC-SA 4.0)"
    tier = "B"
    criteria_source = "Adapted from the evaluation criteria of TRBench's critique prompt"
    notes = ("BFCL v3 requests that need several calls in one turn. A correct proposal is BFCL's ground-truth call set; an incorrect "
             "one is a call set from Gemini 2.5 Pro, Claude 3.7 Sonnet, GPT-4o, o1, DeepSeek-R1, xLAM-2-70B or Qwen3-32B that BFCL's "
             "AST checker rejects (wrong number of calls or no matching call). One item per BFCL request, showing the ground truth "
             "or one rejected response (chosen by a hash of the request id, about half each). Dotted function names that the "
             "models' function-calling APIs turned into underscores are restored, as BFCL's checker does, so that the name style does "
             "not give the label away; string-value mismatches (often borderline) and unparsable outputs are left out. BFCL's simple, multiple and live "
             "simple/multiple categories are not used (they appear in step/tool_arguments and route/tool_selection).")
    SOURCES = ("parallel", "parallel_multiple", "live_parallel", "live_parallel_multiple")

    def load(self):
        groups: dict[str, dict] = {}
        for r in trb_rows(self.SOURCES):
            if r["error_type"].startswith("ast_decoder") or r["error_type"] == "value_error:string":
                continue
            g = groups.setdefault(r["raw_id"], {"id": r["raw_id"], "source": r["task_source"], "chat": r["chat_history"],
                                                "gold": r["chosen_response"], "rejected": {}})
            g["rejected"].setdefault(r["reject_response_from"], (r["reject_response"], r["error_type"]))
        rows = []
        for rid, g in sorted(groups.items()):
            show_gold = _parity(f"trbench:{rid}")
            model = sorted(g["rejected"])[int(hashlib.sha256(rid.encode()).hexdigest(), 16) % len(g["rejected"])]
            response, error = (g["gold"], "") if show_gold else g["rejected"][model]
            rows.append({"id": rid, "source": g["source"], "chat": [dict(m) for m in g["chat"]], "response": response,
                         "correct": show_gold, "from": "ground truth" if show_gold else model, "error": error})
        return rows

    def label_of(self, record):
        return f"{record['source']}|{record['correct']}"

    def to_item(self, record, rng):
        tools, conversation = trb_context(record["chat"])
        state = {"tools": tools, "conversation": conversation, "proposed_response": trb_names(record["response"], tools)}
        gold = Gold(label="true" if record["correct"] else "false")
        return self.item(state, {"correct": Question(type="boolean", instructions=TRB_QUESTION)}, {"correct": gold},
                         category=record["source"], original_id=record["id"], response_from=record["from"], checker_error=record["error"])


# ---------- WebRewardBench (Mind2Web test part) ----------

WRB_REPO = "WebShepherd/WebRewardBench"
WRB_COLUMNS = ["intent", "start_url", "action_history", "text_observation", "chosen", "rejected", "task_id", "step_id", "annotation_id", "source_name"]
WRB_VALID = {"click", "fill", "select_option", "hover", "scroll", "go_back", "go_forward", "goto", "press", "keyboard_press",
             "send_msg_to_user", "noop", "new_tab"}
_ACTION = re.compile(r"^\s*(\w+)\((.*)\)\s*$", re.S)
_BID = re.compile(r"^\s*['\"]([^'\"]+)['\"]")
# Written for this task from the BrowserGym action space that WebRewardBench uses.
WRB_QUESTION = (
    "A web agent is completing the task in the state on a real website. The state gives the page's accessibility tree (elements "
    "have ids in brackets), the previous actions and the proposed next action (BrowserGym syntax: click(id), fill(id, text), "
    "select_option(id, option), hover(id), scroll(dx, dy), go_back(), send_msg_to_user(text) and so on). Is the proposed action the "
    "correct next step toward completing the task?"
)


def ax_tree(text: str) -> tuple[dict[str, str], dict[str, str | None]]:
    """Line and parent element id of every element id in an indented accessibility tree."""
    lines, parents, stack = {}, {}, []  # stack of (depth, id)
    for line in text.split("\n"):
        depth = len(line) - len(line.lstrip("\t"))
        match = re.match(r"\s*\[([^\]]+)\]", line)
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if match:
            bid = match.group(1)
            lines[bid] = line.strip()
            parents[bid] = stack[-1][1] if stack else None
            stack.append((depth, bid))
    return lines, parents


def _element(action: str) -> tuple[str, str | None]:
    match = _ACTION.match(action or "")
    if not match:
        return "", None
    kind, args = match.group(1), match.group(2)
    bid = _BID.match(args)
    return kind, (bid.group(1) if bid and kind not in ("scroll", "goto", "send_msg_to_user", "press", "keyboard_press") else None)


def _related(a: str, b: str, parents: dict) -> bool:
    """a and b are the same element or one contains the other."""
    for x, y in ((a, b), (b, a)):
        node = x
        while node is not None:
            if node == y:
                return True
            node = parents.get(node)
    return False


def webrewardbench_rows() -> list[dict]:
    from huggingface_hub import HfApi

    from ...env import load_env

    load_env()
    files = sorted(f for f in HfApi().list_repo_files(WRB_REPO, repo_type="dataset") if f.endswith(".parquet"))
    return hub_parquet_columns(WRB_REPO, files, WRB_COLUMNS, "webrewardbench_text")  # the screenshots are not read


def webrewardbench_tasks() -> set[str]:
    """Mind2Web annotation ids in WebRewardBench (step/next_action leaves these tasks out)."""
    return {r["annotation_id"] for r in webrewardbench_rows() if str(r["source_name"]).startswith("mind2web")}


@register_task
class WebRewardBenchCritic(TaskAdapter):
    name = f"{LEAF}/webrewardbench"
    leaf = LEAF
    source = "WebRewardBench, Mind2Web test steps with expert and model-proposed actions (Chae et al., 2025, Web-Shepherd)"
    url = "https://huggingface.co/datasets/WebShepherd/WebRewardBench"
    license = "Not stated (WebRewardBench); built on the Mind2Web test set (CC BY 4.0), whose authors ask not to redistribute it"
    tier = "C"
    criteria_source = "Written for this task from the BrowserGym action space"
    redistributable = False
    notes = ("707 Mind2Web test states (cross-task, cross-website, cross-domain), each with the human demonstrator's action and four "
             "actions sampled from GPT-4o-mini and Qwen2.5 models that differ from it. A 'correct' item shows the demonstrated action, "
             "an 'incorrect' item one sampled action (chosen by a hash of the state, about half each). Sampled actions on the same "
             "element as the demonstration, or on an element that contains or is contained in it, are dropped (they may do the same "
             "thing), as are actions on ids missing from the page and invalid action types; the sampled actions were not checked by "
             "people, so an 'incorrect' label means 'differs from the expert', as in Mind2Web's own metrics. The agents' written "
             "thoughts are not shown. Previous actions are cut at the current step (the release stores the whole demonstration). "
             "The WebArena part is not used. Loader only: Mind2Web's authors ask that the test data not be redistributed.")

    def load(self):
        rows = []
        for r in webrewardbench_rows():
            if not str(r["source_name"]).startswith("mind2web") or not r["chosen"]:
                continue
            lines, parents = ax_tree(r["text_observation"] or "")
            gold = r["chosen"][0]["action"].strip()
            gold_kind, gold_bid = _element(gold)
            if gold_kind not in WRB_VALID or (gold_bid is not None and gold_bid not in lines):
                continue
            wrong = []
            for cand in r["rejected"] or []:
                action = cand["action"].strip()
                kind, bid = _element(action)
                if kind not in WRB_VALID or action == gold or action in wrong:
                    continue
                if bid is not None and (bid not in lines or (gold_bid is not None and _related(bid, gold_bid, parents))):
                    continue
                wrong.append(action)
            key = f"{r['annotation_id']}:{r['step_id']}"
            show_gold = _parity(f"webrewardbench:{key}") or not wrong
            proposed = gold if show_gold else wrong[int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(wrong)]
            kind, bid = _element(proposed)
            rows.append({"split": r["source_name"], "key": key, "intent": r["intent"], "website": r["start_url"],
                         "history": list(r["action_history"] or [])[: int(r["step_id"])], "tree": r["text_observation"],
                         "action": proposed, "element": lines.get(bid, "") if bid else "", "correct": show_gold})
        return rows

    def label_of(self, record):
        return f"{record['split']}|{record['correct']}"

    def to_item(self, record, rng):
        proposed = record["action"] + (f"\nTarget element: {record['element']}" if record["element"] else "")
        state = {"task": record["intent"], "website": record["website"],
                 "previous_actions": "\n".join(record["history"]) if record["history"] else "None",
                 "page_accessibility_tree": record["tree"], "proposed_action": proposed}
        gold = Gold(label="true" if record["correct"] else "false")
        return self.item(state, {"correct": Question(type="boolean", instructions=WRB_QUESTION)}, {"correct": gold},
                         split=record["split"], state_key=record["key"])
