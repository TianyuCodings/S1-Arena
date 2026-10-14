"""Leaf: tool-call argument values (工具调用参数，可枚举).

The function to call is fixed; the questions ask which value to pass for its parameters, one
choice question per parameter. Candidate values are enumerated from the state: values in the
request, the schema's enum values, values returned by earlier calls, or leaving an optional
parameter unset. Gold comes from the source's ground-truth call: BFCL's possible-answer lists
(the values its AST checker accepts) and ComplexFuncBench's human-annotated call paths. Every
question has exactly one acceptable option; a candidate that the checker would also accept is
dropped.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch
from ._sources import fits

LEAF = "step/tool_arguments"
UNSET = "unset"

# ---------- helpers ----------


def kind(value) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "num"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return "other"


def standardize(text: str) -> str:
    """BFCL's string normalization (ast_checker.standardize_string): drop spaces and ,./-_*^, lowercase."""
    return re.sub(r"[ \,\.\/\-\_\*\^]", "", text).lower().replace("'", '"')


def same(a, b) -> bool:
    """Equality as BFCL's AST checker sees it: numbers by value, strings after normalization, recursively."""
    if kind(a) != kind(b):
        return False
    if kind(a) == "num":
        return float(a) == float(b)
    if kind(a) == "str":
        return standardize(a) == standardize(b)
    if kind(a) == "list":
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if kind(a) == "dict":
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    return a == b


def literal(value, code: bool = False) -> str:
    """How a value is shown: a JSON literal, or the raw expression for Java / JavaScript arguments."""
    if code and isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def option_id(text: str, taken: set[str]) -> str:
    """The value itself as option id when short; otherwise a truncated form with a hash suffix."""
    oid = text if len(text) <= 60 else text[:48] + "…" + hashlib.sha1(text.encode()).hexdigest()[:6]
    while oid in taken:
        oid += "'"
    taken.add(oid)
    return oid


# ---------- BFCL ----------

BFCL_SHA = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"  # commit of ShishirPatil/gorilla (BFCL v4 data), as in route/tool_selection
BFCL_DATA = f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{BFCL_SHA}/berkeley-function-call-leaderboard/bfcl_eval/data"
NUMBER = re.compile(r"(?<![\w.-])-?\d+(?:\.\d+)?(?![\w])")
# Adapted from BFCL's default system prompt ("You are an expert in composing functions. You are given a question and a set
# of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the
# purpose."), narrowed to one parameter of a call whose function is already chosen.
BFCL_QUESTION = ("You are an expert in composing functions. The request in the state is served by exactly one call to the "
                 "function `{fn}`, documented in the state. Which value should that call pass for the parameter `{param}`? {values}")
VALUES_JSON = "Values are written as JSON literals."
VALUES_CODE = "Values are written as {language} expressions."
UNSET_SENTENCE = " Choose 'unset' if the parameter should be left out, so that the function's default applies."
SCHEMA_KINDS = {"integer": "num", "int": "num", "float": "num", "number": "num", "double": "num", "long": "num",
                "string": "str", "str": "str", "boolean": "bool", "bool": "bool", "array": "list", "list": "list", "tuple": "list"}


def _jsonl(url: str, name: str) -> list[dict]:
    with open(fetch(url, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def bfcl_questions(record: dict) -> tuple[dict, dict]:
    """One choice question per parameter that has at least two candidate values and exactly one acceptable one."""
    function, truth, code = record["function"], record["truth"], record["code"]
    params = (function.get("parameters") or {}).get("properties") or {}
    required = set((function.get("parameters") or {}).get("required") or [])
    canonical = {p: next((v for v in accepted if v != ""), UNSET) for p, accepted in truth.items()}
    numbers = []
    for m in NUMBER.findall(record["request"]):
        numbers.append(float(m) if "." in m else int(m))
    questions, gold = {}, {}
    for param, accepted in truth.items():
        if param not in params:
            continue
        correct = canonical[param]
        spec = params[param]
        want = kind(correct) if correct != UNSET else SCHEMA_KINDS.get(str(spec.get("type", "")).lower())
        if want not in ("num", "str", "bool", "list"):  # nested dicts hold per-key answer lists, not values; 'any' has no kind
            continue
        pool = [] if correct == UNSET else [correct]
        others = [v for q, v in canonical.items() if q != param and v != UNSET]
        extra = list(spec.get("enum") or []) + others
        if want == "num":
            extra += numbers
        if want == "bool":
            extra += [True, False]
        for value in extra:
            if kind(value) == want and not any(same(value, v) for v in pool):
                pool.append(value)
        acceptable = lambda v: any(same(v, a) for a in accepted if a != "")  # noqa: E731
        options = [v for v in pool if v == correct or not acceptable(v)]
        unset_ok = "" in accepted
        if correct == UNSET:
            options = [v for v in pool if not acceptable(v)]
        offer_unset = param not in required and (correct == UNSET or not unset_ok)
        if len(options) + offer_unset < 2 or (correct == UNSET and not offer_unset):
            continue
        taken: set[str] = set()
        criteria = {}
        for value in options:
            text = literal(value, code)
            criteria[option_id(text, taken)] = f"Pass {text}"
        if offer_unset:
            criteria[option_id(UNSET, taken)] = f"Leave `{param}` unset: do not pass it, so that the function's default applies."
        label = UNSET if correct == UNSET else next(k for k, v in criteria.items() if v == f"Pass {literal(correct, code)}")
        values = VALUES_CODE.format(language=code) if code else VALUES_JSON
        text = BFCL_QUESTION.format(fn=function["name"], param=param, values=values) + (UNSET_SENTENCE if offer_unset else "")
        questions[param] = (text, criteria)
        gold[param] = label
    return questions, gold


class BFCLArguments(TaskAdapter):
    leaf = LEAF
    url = "https://gorilla.cs.berkeley.edu/leaderboard.html"
    license = "Apache-2.0"
    tier = "B"
    criteria_source = ("Adapted from BFCL's default function-calling system prompt; candidate values enumerated from the request, "
                       "the parameter's enum and the call's other arguments; gold from BFCL's possible-answer lists")
    CATEGORIES: dict[str, str | None] = {}  # category -> language whose expressions the values are written in (None: JSON)

    def load(self):
        rows = []
        for category, code in self.CATEGORIES.items():
            questions = _jsonl(f"{BFCL_DATA}/BFCL_v4_{category}.json", f"bfcl_{BFCL_SHA[:8]}_{category}.jsonl")
            answers = {a["id"]: a["ground_truth"] for a in _jsonl(f"{BFCL_DATA}/possible_answer/BFCL_v4_{category}.json",
                                                                   f"bfcl_{BFCL_SHA[:8]}_{category}_answer.jsonl")}
            for q in questions:
                truth = answers.get(q["id"]) or []
                functions = q.get("function") or []
                if len(truth) != 1 or len(functions) != 1 or next(iter(truth[0])) != functions[0]["name"]:
                    continue
                messages = q["question"][0]
                users = [i for i, m in enumerate(messages) if m["role"] == "user"]
                if not users:
                    continue
                last = users[-1]
                record = {"id": q["id"], "category": category, "code": code, "function": functions[0],
                          "request": str(messages[last]["content"]).strip(),
                          "earlier": [f"{m['role']}: {str(m['content']).strip()}" for i, m in enumerate(messages) if i != last],
                          "truth": truth[0][functions[0]["name"]]}
                if bfcl_questions(record)[0]:
                    rows.append(record)
        return rows

    def label_of(self, record):
        return record["category"]

    def to_item(self, record, rng):
        specs, gold = bfcl_questions(record)
        questions = {p: Question(type="choice", instructions=text, criteria=shuffled_options(criteria, rng)) for p, (text, criteria) in specs.items()}
        state = {"request": record["request"]}
        if record["earlier"]:
            state["earlier_messages"] = "\n\n".join(record["earlier"])
        state["function"] = json.dumps(record["function"], ensure_ascii=False, indent=1)
        return self.item(state, questions, {p: Gold(label=g) for p, g in gold.items()}, category=record["category"], original_id=record["id"])


@register_task
class BFCLSimpleArguments(BFCLArguments):
    name = f"{LEAF}/bfcl_simple"
    source = "Berkeley Function Calling Leaderboard v4, expert-written 'simple' categories in Python, Java and JavaScript (Patil et al., 2025)"
    notes = ("One request per BFCL item, one question per parameter of the (given) function. Candidates: the ground-truth value, "
             "the call's other argument values of the same type, numbers in the request, the parameter's enum values, and 'unset' "
             "for optional parameters; any candidate the BFCL AST checker would also accept (string normalization, int/float "
             "equality, or 'unset' where omission is allowed) is removed. Parameters with fewer than two candidates are not asked. "
             "Java and JavaScript values are shown as source expressions, as BFCL expects them.")
    CATEGORIES = {"simple_python": None, "simple_java": "Java", "simple_javascript": "JavaScript"}


@register_task
class BFCLLiveSimpleArguments(BFCLArguments):
    name = f"{LEAF}/bfcl_live_simple"
    source = "Berkeley Function Calling Leaderboard v4, user-contributed 'live_simple' category (Patil et al., 2025)"
    notes = ("Real user requests sent to BFCL's hosted endpoint in 2024 (several languages), each served by one call; ground-truth "
             "possible answers written by the BFCL team. Candidates and filtering as in bfcl_simple.")
    CATEGORIES = {"live_simple": None}


# ---------- ComplexFuncBench ----------

CFB_REPO = "zai-org/ComplexFuncBench"
CFB_QUESTION = ("The assistant is working on the user's request with the functions documented in the state; the conversation so "
                "far, with every function call and its result, is shown. Its next step is a single call to `{fn}`. Which value "
                "should that call pass for the argument `{arg}`? The candidates are the `{key}` values in the earlier function "
                "result where the correct value appears.")
MAX_OPTIONS = 40


def _scalars(value, key=None, out=None) -> list:
    out = [] if out is None else out
    if isinstance(value, dict):
        for k, v in value.items():
            _scalars(v, k, out)
    elif isinstance(value, list):
        for v in value:
            _scalars(v, key, out)
    elif not isinstance(value, bool) and value is not None:
        out.append((key, value))
    return out


def render_cfb(conversation: list[dict]) -> str:
    lines = []
    for m in conversation:
        if m["role"] == "user":
            lines.append(f"User: {m['content']}")
        elif m["role"] == "assistant":
            if m.get("content"):
                lines.append(f"Assistant: {m['content']}")
            for call in m.get("function_call") or []:
                lines.append(f"Assistant calls {call['name']}({json.dumps(call['arguments'], ensure_ascii=False)})")
        elif m["role"] == "observation":
            results = m["content"] if isinstance(m["content"], list) else [m["content"]]
            for result in results:
                lines.append(f"Function result: {json.dumps(result, ensure_ascii=False)}")
    return "\n\n".join(lines)


@register_task
class ComplexFuncBenchArguments(TaskAdapter):
    name = f"{LEAF}/complexfuncbench"
    leaf = LEAF
    source = "ComplexFuncBench, multi-step Booking.com function calling with recorded API results (Zhong et al., 2025)"
    url = "https://huggingface.co/datasets/zai-org/ComplexFuncBench"
    license = "Apache-2.0"
    tier = "C"
    criteria_source = "Written for this task; gold is the argument value in the human-annotated function-call path"
    notes = ("Hotels, flights, car rental, attractions and cross-domain requests. Golden call paths were drafted with GPT-4o, "
             "corrected by senior annotators on 100 templates and generalised to 1,000 samples by annotators. One item is a step "
             "where the assistant makes a single call after at least one function result; it asks for every argument whose "
             "annotated value appears in an earlier function result (IDs, tokens, slugs, coordinates, sort options, dates offered by "
             "the API), with the other values under the same key in that result as candidates (at most 40). Integer arguments, "
             "values the user typed verbatim and arguments with only one candidate are not asked; steps with parallel calls are "
             "left out.")

    def load(self):
        from huggingface_hub import hf_hub_download

        from ...env import load_env

        load_env()
        path = hf_hub_download(CFB_REPO, "ComplexFuncBench.jsonl", repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]
        rows = []
        for sample in samples:
            conversation = sample["conversations"]
            results: list = []  # every function result so far, in order
            for index, message in enumerate(conversation):
                if message["role"] == "observation":
                    results += message["content"] if isinstance(message["content"], list) else [message["content"]]
                    continue
                calls = message.get("function_call") or []
                if message["role"] != "assistant" or len(calls) != 1 or not results:
                    continue
                call, asked = calls[0], {}
                said = " ".join(str(m.get("content") or "") for m in conversation[:index] if m["role"] == "user")
                for arg, value in call["arguments"].items():
                    # strings and decimals only (small integers such as guest counts match unrelated numbers in the results),
                    # and not values the user typed verbatim (those are not taken from a result)
                    if isinstance(value, bool) or not isinstance(value, (str, float)) or str(value) in said:
                        continue
                    source = next((r for r in results if any(v == value for _, v in _scalars(r))), None)
                    if source is None:
                        continue
                    keys = Counter(k for k, v in _scalars(source) if v == value)
                    key = keys.most_common(1)[0][0]
                    pool = list(dict.fromkeys(v for k, v in _scalars(source) if k == key))
                    pool = [value] + [v for v in pool if v != value and literal(v) != literal(value)][: MAX_OPTIONS - 1]
                    if len(pool) >= 2 and key is not None:
                        asked[arg] = {"key": key, "value": value, "pool": pool}
                if asked and fits(render_cfb(conversation[:index]), json.dumps(sample["functions"], ensure_ascii=False), limit=29_000):
                    rows.append({"id": sample["id"], "step": index, "domain": sample["id"].rsplit("-", 1)[0], "function": call["name"],
                                 "functions": sample["functions"], "history": conversation[:index], "asked": asked})
        return rows

    def label_of(self, record):
        return f"{record['domain']}|{record['function']}"

    def to_item(self, record, rng):
        questions, gold = {}, {}
        for arg, spec in record["asked"].items():
            taken: set[str] = set()
            criteria = {option_id(literal(v), taken): f"Pass {literal(v)}" for v in spec["pool"]}
            label = next(k for k, text in criteria.items() if text == f"Pass {literal(spec['value'])}")
            text = CFB_QUESTION.format(fn=record["function"], arg=arg, key=spec["key"])
            questions[arg] = Question(type="choice", instructions=text, criteria=shuffled_options(criteria, rng))
            gold[arg] = Gold(label=label)
        functions = [{k: f[k] for k in ("name", "description", "parameters") if k in f} for f in record["functions"]]
        state = {"functions": json.dumps(functions, ensure_ascii=False), "conversation": render_cfb(record["history"])}
        return self.item(state, questions, gold, domain=record["domain"], original_id=record["id"], step=record["step"], function=record["function"])
