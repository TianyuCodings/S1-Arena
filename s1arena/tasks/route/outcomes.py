"""Recorded per-model outcomes shared by the routing leaves.

Each loader returns one record per request with the prompt the models saw and each
model's automatically scored outcome (1 = correct). Sources are pinned to fixed revisions
so rebuilding gives the same items:

- RouterBench, zero-shot file (Hu et al., 2024): 11 models, one reply each, scored 0/1.
- RouterArena cached results (Lu et al., 2025): 3 models on the sub_10 split, GPT-4o mini on all.
- LiveBench judgments (White et al., 2025): objective scores of many models, no LLM judge.
- EmbedLLM correctness matrix (Zhuang et al., 2024): 112 open models on 3,000 test questions.
"""

from __future__ import annotations

import ast
import functools
import json
import pickle
from collections import defaultdict
from typing import Any

from ...env import load_env
from ..common import fetch, hf_rows

# ---------- helpers ----------

def hf_file(repo: str, filename: str, revision: str) -> str:
    """Local path of one file of a Hugging Face dataset repo at a pinned revision."""
    load_env()
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision)


class _PandasOnlyUnpickler(pickle.Unpickler):
    """Unpickles a pandas DataFrame but refuses every other global, so a pickle cannot run code."""

    ALLOWED = {
        ("pandas.core.frame", "DataFrame"), ("pandas.core.internals.managers", "BlockManager"),
        ("pandas._libs.internals", "_unpickle_block"), ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.base", "Index"), ("pandas.core.indexes.range", "RangeIndex"),
        ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.numeric", "_frombuffer"), ("builtins", "slice"),
    }

    def find_class(self, module: str, name: str) -> Any:
        module = module.replace("numpy.core.", "numpy._core.")
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")
        return super().find_class(module, name)


def _first_text(value: Any) -> str:
    """RouterBench stores prompts and replies as the repr of a list of strings."""
    if isinstance(value, str) and value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return "\n".join(str(p) for p in parsed)
        except (ValueError, SyntaxError):
            pass
    return "" if value is None else str(value)


# ---------- RouterBench ----------

ROUTERBENCH = "withmartian/routerbench"
ROUTERBENCH_REV = "784021482c3f320c6619ed4b3bb3b41a21424fcb"
ROUTERBENCH_URL = "https://huggingface.co/datasets/withmartian/routerbench"
ROUTERBENCH_LICENSE = "Not stated (dataset card); code repository MIT"
# Evals with letter-matched 0/1 scores. Left out: GSM8K (correct replies are scored 0.25-0.75), MBPP (prompts
# lack the function signature the tests call), MT-Bench (GPT-4 judge), the small OpenAI-evals sets, and
# WinoGrande (its validation set is already used in know/commonsense).
ROUTERBENCH_GROUPS = ("mmlu", "hellaswag", "arc-challenge")
ROUTERBENCH_MODELS = {  # model id -> description shown to the classifier
    "gpt-4-1106-preview": "GPT-4 Turbo preview (OpenAI, gpt-4-1106-preview, November 2023)",
    "claude-v2": "Claude 2 (Anthropic, claude-v2, 2023)",
    "gpt-3.5-turbo-1106": "GPT-3.5 Turbo (OpenAI, gpt-3.5-turbo-1106, November 2023)",
    "claude-instant-v1": "Claude Instant 1 (Anthropic, claude-instant-v1, 2023)",
    "zero-one-ai/Yi-34B-Chat": "Yi-34B-Chat (01.AI, open weights, 34B parameters, November 2023)",
    "mistralai/mixtral-8x7b-chat": "Mixtral 8x7B Instruct (Mistral AI, open-weight mixture of experts, December 2023)",
}
ROUTERBENCH_PROTOCOL = ("RouterBench sent the prompt once, zero-shot, to each model and scored the reply 1 if it gives "
                        "the correct option letter and 0 otherwise.")


@functools.lru_cache(maxsize=1)
def routerbench() -> tuple[dict[str, Any], ...]:
    """Requests of the kept evals with each model's 0/1 score, reply and cost."""
    with open(hf_file(ROUTERBENCH, "routerbench_0shot.pkl", ROUTERBENCH_REV), "rb") as f:
        frame = _PandasOnlyUnpickler(f).load()
    models = [c for c in frame.columns if f"{c}|model_response" in frame.columns]
    rows = []
    for rec in frame.to_dict("records"):
        group = "mmlu" if rec["eval_name"].startswith("mmlu") else rec["eval_name"]
        if group not in ROUTERBENCH_GROUPS:
            continue
        rows.append({
            "id": rec["sample_id"], "group": group, "eval": rec["eval_name"], "prompt": _first_text(rec["prompt"]).strip(),
            "score": {m: float(rec[m]) for m in models},
            "reply": {m: _first_text(rec[f"{m}|model_response"]).strip() for m in models},
            "cost": {m: float(rec[f"{m}|total_cost"]) for m in models},
        })
    return tuple(rows)


@functools.lru_cache(maxsize=None)
def routerbench_mean_cost(model: str) -> float:
    rows = routerbench()
    return sum(r["cost"][model] for r in rows) / len(rows)


# ---------- RouterArena ----------

ROUTERARENA_SHA = "cff9659dd6a3a07815f4842fcdadea5a597a1df1"  # GitHub commit of RouteWorks/RouterArena
ROUTERARENA_URL = "https://github.com/RouteWorks/RouterArena"
ROUTERARENA_LICENSE = "Apache-2.0 (RouterArena repository); source datasets keep their own licenses"
ROUTERARENA_MODELS = {
    "gemini-2.0-flash-001": "Gemini 2.0 Flash (Google, gemini-2.0-flash-001, February 2025)",
    "gpt-4o-mini": "GPT-4o mini (OpenAI, July 2024)",
    "claude-3-haiku-20240307": "Claude 3 Haiku (Anthropic, claude-3-haiku-20240307, March 2024)",
}
# Metrics that score a reply 0 or 1 against the reference; METEOR-scored translation and NarrativeQA items are left out.
ROUTERARENA_BINARY = {"mcq_accuracy", "exact_match", "code_accuracy", "superglue_exact_match", "math_metric", "chess_accuracy", "superglue_clozetest"}
ROUTERARENA_PROTOCOL = ("RouterArena sent the prompt once to each model and scored the final answer automatically against "
                        "the reference answer (option letter, exact match, math answer check or code tests).")


@functools.lru_cache(maxsize=1)
def routerarena() -> dict[str, dict[str, Any]]:
    """global index -> prompt, dataset family and per-model {score, answer, cost} for binary-scored requests."""
    out: dict[str, dict[str, Any]] = {}
    for model in ROUTERARENA_MODELS:
        url = f"https://raw.githubusercontent.com/RouteWorks/RouterArena/{ROUTERARENA_SHA}/cached_results/{model}.jsonl"
        with open(fetch(url, f"routerarena_{ROUTERARENA_SHA[:8]}_{model}.jsonl"), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                ev = rec.get("evaluation_result") or {}
                if ev.get("metric") not in ROUTERARENA_BINARY or ev.get("score") not in (0, 1, 0.0, 1.0):
                    continue
                row = out.setdefault(rec["global_index"], {"id": rec["global_index"], "prompt": rec["question"].strip(), "outcome": {}})
                row["outcome"][model] = {"score": float(ev["score"]), "answer": str(rec.get("generated_answer") or ""), "cost": ev.get("inference_cost")}
    meta = {r["Global Index"]: r for r in hf_rows("RouteWorks/RouterArena", None, "full", revision="a4a062ce3313b56bb09c042e1bc37b61d34e3bd8")}
    for gi, row in out.items():
        info = meta.get(gi, {})
        row["family"] = gi.rsplit("_", 1)[0].split("_")[0]
        row["domain"] = info.get("Domain") or ""
        row["difficulty"] = info.get("Difficulty") or ""
    return out


# ---------- LiveBench ----------

LIVEBENCH_URL = "https://huggingface.co/datasets/livebench/model_judgment"
LIVEBENCH_LICENSE = "Not stated (dataset cards); code Apache-2.0. Texts from LeetCode/AtCoder via LiveCodeBench, arXiv abstracts and NYT Connections"
LIVEBENCH_REVS = {"model_judgment": "9704e5da7bfbefe75ac1482a13de827127295993", "coding": "a958549fdd8aa57be0a3fafe7b205ffc160ed5f4", "language": "3ada32a2e53d5e04e57fa503384cb85ce9116c40"}
# Tasks scored 0/1 by code tests or exact match; Connections is counted as correct only when every group is right.
LIVEBENCH_TASKS = ("LCB_generation", "coding_completion", "typos", "connections")
LIVEBENCH_PROTOCOL = ("LiveBench scored each model's output automatically against the ground truth: generated code must pass "
                      "the hidden tests, a typo-fixed text must match the original exactly, and a Connections answer counts "
                      "only if every group is right.")


@functools.lru_cache(maxsize=1)
def livebench() -> tuple[dict[str, Any], ...]:
    """Questions of the kept tasks with each judged model's outcome (1 = fully correct)."""
    questions = {}
    for name in ("coding", "language"):
        for r in hf_rows(f"livebench/{name}", None, "test", revision=LIVEBENCH_REVS[name]):
            if r["task"] in LIVEBENCH_TASKS:
                questions[r["question_id"]] = r
    outcome: dict[str, dict[str, float]] = defaultdict(dict)
    for j in hf_rows("livebench/model_judgment", None, "leaderboard", revision=LIVEBENCH_REVS["model_judgment"]):
        if j["question_id"] in questions and j["score"] is not None:
            outcome[j["question_id"]][j["model"]] = 1.0 if float(j["score"]) >= 1.0 else 0.0
    rows = []
    for qid, q in sorted(questions.items()):
        if outcome.get(qid):
            rows.append({"id": qid, "group": q["task"], "prompt": "\n\n".join(q["turns"]).strip(), "score": outcome[qid],
                         "release": str(q.get("livebench_release_date") or "")[:10]})
    return tuple(rows)


# ---------- EmbedLLM ----------

EMBEDLLM = "RZ412/EmbedLLM"
EMBEDLLM_REV = "266ae3e180ccb533c86afc9cfb2f5db99d171aea"
EMBEDLLM_URL = "https://huggingface.co/datasets/RZ412/EmbedLLM"


def _final_question(prompt: str, category: str) -> str:
    """The request without few-shot demonstrations or the trailing 'Answer:' cue."""
    text = prompt.strip()
    if category == "gsm8k" and "Question:" in text:
        text = "Question:" + text.rsplit("Question:", 1)[1]
    if text.endswith("Answer:"):
        text = text[: -len("Answer:")].rstrip()
    return text


@functools.lru_cache(maxsize=1)
def embedllm(categories: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    """Test questions of the given category prefixes with the 0/1 outcome of all 112 models."""
    import pandas as pd

    frame = pd.read_csv(hf_file(EMBEDLLM, "test.csv", EMBEDLLM_REV), usecols=["prompt_id", "model_name", "label", "category", "prompt"])
    frame = frame[frame["category"].map(lambda c: c.split("_")[0] in categories or c in categories)]
    rows = []
    for (pid, category), group in frame.groupby(["prompt_id", "category"], sort=True):
        rows.append({
            "id": int(pid), "category": category, "group": "mmlu" if category.startswith("mmlu") else category,
            "prompt": _final_question(group["prompt"].iloc[0], category),
            "score": {m: float(v) for m, v in zip(group["model_name"], group["label"])},
        })
    return tuple(rows)
