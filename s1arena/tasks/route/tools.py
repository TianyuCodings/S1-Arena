"""Leaf: tool / skill selection (工具 / 技能选择).

A choice among the available functions plus a "none" option for requests that no available
function can serve. Gold comes from the source's ground-truth call (BFCL's expert-written
answers; MetaTool's generation target) or from construction (irrelevant function sets; the
right tool removed from the list).
"""

from __future__ import annotations

import ast
import hashlib
import json
import re

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch

LEAF = "route/tool_selection"

# ---------- BFCL ----------

BFCL_SHA = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"  # commit of ShishirPatil/gorilla, BFCL v4 data
BFCL_DATA = f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{BFCL_SHA}/berkeley-function-call-leaderboard/bfcl_eval/data"
# Adapted from BFCL's default system prompt: "... Based on the question, you will need to make one or more function/tool
# calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the
# parameters required by the function, also point it out."
BFCL_INSTRUCTIONS = ("You are given a request and a set of possible functions. Based on the request, choose the one function "
                     "that should be called. If none of the functions can be used, or the request lacks the parameters the "
                     "function requires, choose 'none'.")
NONE_OPTION = "No function: none of the available functions can be used for this request, so no function should be called."


def _jsonl(url: str, name: str) -> list[dict]:
    with open(fetch(url, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _describe(function: dict) -> str:
    """'Description. Parameters: city (string, required): The city; days (integer, optional): ...'"""
    params = function.get("parameters") or {}
    required = set(params.get("required") or [])
    parts = []
    for pname, spec in (params.get("properties") or {}).items():
        kind = spec.get("type", "any")
        if spec.get("enum"):
            kind += ", one of " + "/".join(str(e) for e in spec["enum"])
        parts.append(f"{pname} ({kind}, {'required' if pname in required else 'optional'}): {str(spec.get('description', '')).strip()}")
    text = str(function.get("description", "")).strip()
    return text + (" Parameters: " + "; ".join(parts) if parts else " No parameters.")


class BFCLSelection(TaskAdapter):
    leaf = LEAF
    url = "https://gorilla.cs.berkeley.edu/leaderboard.html"
    license = "Apache-2.0"
    tier = "B"
    criteria_source = "Adapted from BFCL's default function-calling system prompt; option texts are the original function documentation"
    CALL: tuple[str, ...] = ()  # categories with exactly one ground-truth call
    NONE: tuple[str, ...] = ()  # categories where no function should be called

    def load(self):
        rows = []
        for category in self.CALL + self.NONE:
            questions = _jsonl(f"{BFCL_DATA}/BFCL_v4_{category}.json", f"bfcl_{BFCL_SHA[:8]}_{category}.jsonl")
            answers = {}
            if category in self.CALL:
                for a in _jsonl(f"{BFCL_DATA}/possible_answer/BFCL_v4_{category}.json", f"bfcl_{BFCL_SHA[:8]}_{category}_answer.jsonl"):
                    answers[a["id"].split("-")[0]] = a["ground_truth"]  # ids differ after the first '-' in a few rows
            for q in questions:
                functions = q.get("function") or []
                names = [f["name"] for f in functions]
                if not functions or len(set(names)) != len(names) or "none" in names:
                    continue
                if category in self.CALL:
                    truth = answers.get(q["id"].split("-")[0]) or []
                    if len(truth) != 1 or next(iter(truth[0])) not in names:
                        continue
                    gold = next(iter(truth[0]))
                else:
                    gold = "none"
                rows.append({"id": q["id"], "category": category, "messages": q["question"][0], "functions": functions, "gold": gold})
        return rows

    def label_of(self, record):
        return f"{record['category']}|{'none' if record['gold'] == 'none' else 'call'}"

    def to_item(self, record, rng):
        messages = record["messages"]
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user") if any(m["role"] == "user" for m in messages) else len(messages) - 1
        state = {"request": str(messages[last_user]["content"]).strip()}
        earlier = [m for i, m in enumerate(messages) if i != last_user]
        if earlier:
            state["earlier_messages"] = "\n\n".join(f"{m['role']}: {str(m['content']).strip()}" for m in earlier)
        options = {f["name"]: _describe(f) for f in record["functions"]}
        options = shuffled_options(options, rng)
        options["none"] = NONE_OPTION
        question = Question(type="choice", instructions=BFCL_INSTRUCTIONS, criteria=options)
        return self.item(state, {"function": question}, {"function": Gold(label=record["gold"])}, category=record["category"])


@register_task
class BFCLExpert(BFCLSelection):
    name = f"{LEAF}/bfcl_expert"
    source = "Berkeley Function Calling Leaderboard v4, expert-written 'multiple' and 'irrelevance' categories (Patil et al., 2025)"
    notes = ("Multiple: one correct function among 2-4; irrelevance: the only function offered does not fit. Ground truth written "
             "by the BFCL team. The 'none' option is always listed last.")
    CALL = ("multiple",)
    NONE = ("irrelevance",)


@register_task
class BFCLLive(BFCLSelection):
    name = f"{LEAF}/bfcl_live"
    source = "Berkeley Function Calling Leaderboard v4, user-contributed 'live_multiple' and 'live_irrelevance' categories (Patil et al., 2025)"
    notes = ("Real user queries sent to BFCL's hosted endpoint in 2024, with up to 37 functions; ground-truth calls written by the "
             "BFCL team. BFCL counts a request that lacks a required argument as irrelevant; a spot check of 30 live_irrelevance "
             "items found about 3 that look mislabeled. The 'none' option is always listed last.")
    CALL = ("live_multiple",)
    NONE = ("live_irrelevance",)


# ---------- MetaTool ----------

METATOOL_SHA = "35e81bb7576826e980c80fed8f8c0a2b4a1e6fbb"
METATOOL_DATA = f"https://raw.githubusercontent.com/HowieHwong/MetaTool/{METATOOL_SHA}/dataset/tmp_dataset"
# Verbatim from MetaTool's tool-selection prompt (Huang et al., 2024).
METATOOL_INSTRUCTIONS = ("Your current task is to choose the appropriate tool to solve the user's query based on their question. "
                         "If there is a tool in the list that is applicable to this query, please return the name of the tool "
                         "(you can only choose one tool). If there isn't, please return 'None.'")
TOOL_LINE = re.compile(r"^\d+\. tool name: (.*?), tool description: (.*)$", re.M)
# General-purpose web search, browsing and automation tools: when one of them is still listed, "None" is debatable.
GENERAL_TOOLS = {"Zapier", "web_scraper", "web_requests", "WebsiteTool", "internetSearch", "search",
                 "MixerBox_WebSearchG_web_search", "total_query_meta_search_engine", "metaphor_search_api"}


def _clean_description(text: str) -> str:
    text = text.strip()
    if text.startswith("["):
        try:
            parts = ast.literal_eval(text)
            if isinstance(parts, list):
                return " ".join(dict.fromkeys(str(p).strip() for p in parts))
        except (ValueError, SyntaxError):
            pass
    return text


@register_task
class MetaToolSelection(TaskAdapter):
    name = f"{LEAF}/metatool"
    leaf = LEAF
    source = "MetaTool ToolE, tool selection with and without the right tool in the list (Huang et al., 2024)"
    url = "https://github.com/HowieHwong/MetaTool"
    license = "MIT"
    tier = "A"
    criteria_source = "MetaTool's tool-selection prompt, verbatim; option texts are the original tool descriptions"
    notes = ("Queries were written by ChatGPT for a given plugin tool and then checked by hand; the label is that tool (by "
             "construction), or 'None' in the variant where the tool and its closest neighbours were removed from the list "
             "(subtask 3). Each query is used in one variant only, chosen by a hash of the query. 'None' variants whose list still "
             "holds a general-purpose web search, browsing or automation tool (e.g. Zapier) are not used, since there 'None' is "
             "debatable. Some queries are vague.")

    def load(self):
        variants = {}
        for subtask, removed in (("Task2-Subtask1", False), ("Task2-Subtask3", True)):
            with open(fetch(f"{METATOOL_DATA}/{subtask}.json", f"metatool_{METATOOL_SHA[:8]}_{subtask}.json"), encoding="utf-8") as f:
                for r in json.load(f):
                    block = r["action_prompt"].split("[List of Tools with Names and Descriptions Start]")[1].split("[List of Tools with Names and Descriptions End]")[0]
                    tools = {name.strip(): _clean_description(desc) for name, desc in TOOL_LINE.findall(block)}
                    gold = "none" if removed else r["tool"]
                    if removed == (r["tool"] in tools) or "none" in tools or (removed and GENERAL_TOOLS & set(tools)):
                        continue
                    variants.setdefault(r["query"].strip(), {})[removed] = {"query": r["query"].strip(), "tools": tools, "gold": gold, "tool": r["tool"], "variant": "tool_removed" if removed else "tool_listed"}
        rows = []
        for query, pair in sorted(variants.items()):
            want = int(hashlib.sha256(query.encode()).hexdigest(), 16) % 2 == 1
            rows.append(pair.get(want) or next(iter(pair.values())))
        return rows

    def label_of(self, record):
        return record["variant"]

    def to_item(self, record, rng):
        options = shuffled_options(dict(record["tools"]), rng)
        options["none"] = "None: no tool in the list is applicable to this query."
        question = Question(type="choice", instructions=METATOOL_INSTRUCTIONS, criteria=options)
        return self.item({"query": record["query"]}, {"tool": question}, {"tool": Gold(label=record["gold"])}, variant=record["variant"], original_tool=record["tool"])
