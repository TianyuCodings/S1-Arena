"""Leaf: output prediction (输出预测) as multiple choice.

The correct option is the output recorded by executing the function (CRUXEval, LiveCodeBench
execution). Wrong options are, first, outputs that models actually predicted and that the
benchmarks' execution-based scorers marked wrong (the most frequent ones), and, when there are
not enough of those, small type-preserving edits of the true output. Every option is a Python
literal; options are compared by value with ast.literal_eval, so no option equals the true
output (1 and 1.0 count as equal and are never both present).
"""

from __future__ import annotations

import ast
import json
import random
import zipfile
from collections import Counter, defaultdict

from ...env import load_env
from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch, hf_rows

LEAF = "code/output_prediction"
LETTERS = "ABCD"
N_OPTIONS = 4
CRUX_ZIP = "https://raw.githubusercontent.com/facebookresearch/cruxeval/main/samples/evaluation_results.zip"
LCB_EXEC = "Gemini-Pro-1.5 (May)/Scenario.codeexecution_10_0.2_eval_all.json"
# CRUXEval-O prompt (Gu et al., 2024), also used by LiveCodeBench's execution scenario.
INSTRUCTIONS = (
    "You are given a Python function and an assertion containing an input to the function. Which literal completes "
    "the assertion, i.e. is the output when executing the provided code on the given input?"
)


def literal(text: str):
    """The value of a Python literal, or a sentinel when the text is not one."""
    try:
        return ast.literal_eval(text.strip())
    except Exception:
        return NotALiteral


class NotALiteral:
    pass


def same(a, b) -> bool:
    """Equality as the benchmarks' assert statements see it (so True == 1 and 1 == 1.0)."""
    try:
        return bool(a == b)
    except Exception:
        return False


def edits(value, rng: random.Random) -> list:
    """Small, type-preserving changes of a value (candidates for wrong options)."""
    out = []
    if isinstance(value, bool):
        out = [not value]
    elif isinstance(value, int):
        out = [value + 1, value - 1, value * 2 if value else 2, -value if value else 1, value + 10]
    elif isinstance(value, float):
        out = [value + 1.0, value * 2 if value else 1.0, -value if value else 0.5, round(value + 0.5, 2)]
    elif isinstance(value, str):
        if value:
            out = [value[::-1], value[:-1], value[1:], value + value[-1], value.swapcase(), value.upper(), value + " ", value.replace(value[0], "", 1)]
        else:
            out = [" ", "a", "0"]
    elif isinstance(value, (list, tuple)):
        kind = type(value)
        items = list(value)
        if items:
            out = [kind(items[::-1]), kind(items[:-1]), kind(items[1:]), kind(items + items[-1:])]
            i = rng.randrange(len(items))
            for changed in edits(items[i], rng)[:2]:
                out.append(kind(items[:i] + [changed] + items[i + 1:]))
        else:
            out = [kind([0]), kind([""]), None]
    elif isinstance(value, dict):
        keys = list(value)
        if keys:
            k = keys[rng.randrange(len(keys))]
            out = [{x: v for x, v in value.items() if x != k}]
            for changed in edits(value[k], rng)[:2]:
                out.append({**value, k: changed})
        else:
            out = [{"": 0}, None]
    elif isinstance(value, set):
        out = [set(), value | {0}] if value else [{0}, None]
    elif value is None:
        out = [0, "", False, []]
    rng.shuffle(out)
    return out


def build_options(true_text: str, wrong_counts: dict, seed: str):
    """The true output plus three distinct wrong values (most frequent model mistakes first, then edits),
    as repr() literals with the true one first; None when that is not possible."""
    truth = literal(true_text)
    if truth is NotALiteral:
        return None
    chosen, kinds = [], []
    for text, _ in sorted(wrong_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        value = literal(text)
        if value is NotALiteral or same(value, truth) or any(same(value, c) for c in chosen):
            continue
        chosen.append(value)
        kinds.append("model")
        if len(chosen) == N_OPTIONS - 1:
            break
    for value in edits(truth, random.Random(seed)):
        if len(chosen) == N_OPTIONS - 1:
            break
        if same(value, truth) or any(same(value, c) for c in chosen):
            continue
        chosen.append(value)
        kinds.append("edit")
    texts = [repr(v) for v in [truth] + chosen]
    if len(chosen) < N_OPTIONS - 1 or len(set(texts)) != len(texts):
        return None
    return {"texts": texts, "kinds": kinds}


def choice_item(task, record, rng, assertion: str):
    """Shuffle the prepared options (true output first in record['options']) and build the item."""
    texts = record["options"]["texts"]
    order = list(range(len(texts)))
    rng.shuffle(order)
    options = {LETTERS[i]: texts[j] for i, j in enumerate(order)}
    question = Question(type="choice", instructions=INSTRUCTIONS, criteria=options)
    state = {"code": record["code"], "assertion": assertion}
    return task.item(state, {"output": question}, {"output": Gold(label=LETTERS[order.index(0)])}, original_id=record["id"], distractors=record["options"]["kinds"])


@register_task
class CRUXEvalOutput(TaskAdapter):
    name = f"{LEAF}/cruxeval"
    leaf = LEAF
    source = "CRUXEval-O, output prediction for 800 short Python functions (Gu et al., 2024)"
    url = "https://github.com/facebookresearch/cruxeval"
    license = "MIT"
    tier = "B"
    criteria_source = "Adapted from the CRUXEval-O prompt; wrong options are model predictions from the CRUXEval release that its execution-based scorer marked wrong, topped up with small edits of the true output"
    notes = (
        "Wrong options come from the non-chain-of-thought output-prediction samples of the released models "
        "(evaluation_results.zip in the CRUXEval repository), most frequent first. Options are shown as Python repr literals."
    )

    def load(self):
        rows = hf_rows("cruxeval-org/cruxeval", None, "test")
        wrong = defaultdict(Counter)
        with zipfile.ZipFile(fetch(CRUX_ZIP, "cruxeval_evaluation_results.zip")) as zf:
            for name in sorted(zf.namelist()):
                if not name.endswith("_output.json") or "+cot" in name:
                    continue
                result = json.loads(zf.read(name))
                for sample, predictions in result["raw_generations"].items():
                    for text, ok in zip(predictions, result["raw_scored_generations"][sample]):
                        if not ok and text.strip():
                            wrong[sample][text.strip()] += 1
        out = []
        for r in rows:
            options = build_options(r["output"], wrong[r["id"]], f"{self.name}:{r['id']}")
            if options:
                out.append(dict(r, options=options))
        return out

    def label_of(self, record):
        return type(literal(record["output"])).__name__

    def to_item(self, record, rng):
        return choice_item(self, record, rng, f"assert f({record['input']}) == ??")


@register_task
class LiveCodeBenchExecution(TaskAdapter):
    name = f"{LEAF}/livecodebench_execution"
    leaf = LEAF
    source = "LiveCodeBench code execution v2, LeetCode solutions with recorded outputs (Jain et al., 2024)"
    url = "https://huggingface.co/datasets/livecodebench/execution-v2"
    license = "Not stated precisely (the Hugging Face card says 'cc'); LiveCodeBench code and submissions MIT"
    tier = "B"
    criteria_source = "Adapted from the CRUXEval-O prompt used by LiveCodeBench; wrong options are Gemini-Pro-1.5 predictions that LiveCodeBench marked wrong, topped up with small edits of the true output"
    notes = "Problems from LeetCode contests of May 2023 to 2024; the programs are correct solutions and the outputs come from running them."

    def load(self):
        load_env()
        from huggingface_hub import hf_hub_download

        rows = hf_rows("livecodebench/execution-v2", None, "test")
        with open(hf_hub_download("livecodebench/submissions", LCB_EXEC, repo_type="dataset"), encoding="utf-8") as f:
            graded = {g["id"]: g for g in json.load(f)}
        out = []
        for r in rows:
            wrong = Counter()
            g = graded.get(r["id"])
            if g:
                for text, ok in zip(g["pred_list"], g["graded_list"]):
                    if not ok and str(text).strip():
                        wrong[str(text).strip()] += 1
            options = build_options(r["output"], wrong, f"{self.name}:{r['id']}")
            if options:
                out.append(dict(r, options=options))
        return out

    def label_of(self, record):
        return f"{record['difficulty']}|{type(literal(record['output'])).__name__}"

    def to_item(self, record, rng):
        return choice_item(self, record, rng, f"assert {record['input']} == ??")
