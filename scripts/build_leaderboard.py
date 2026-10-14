"""Write leaderboard/results.json, the numbers of the leaderboard website, from a results summary.

Usage (from the repository root):

    uv run python scripts/build_leaderboard.py path/to/paper_groups.json

The input is the summary of all evaluated models (one row per model): accuracy per domain of the non-agentic tasks,
overall accuracy and Brier score, latency, cost and output tokens per request, and, for models that ran the agentic
tasks, the accuracy on static agent steps and the success rate in the closed-loop environments. The output holds only
these aggregate numbers, no model responses.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "leaderboard" / "results.json"
USD_PER_COPILOT_CREDIT = 0.01
GROUPS = {
    "text": ("Text and knowledge", ["basic_cls", "know", "social"]),
    "applied": ("Applied decisions", ["pro", "ret", "struct", "code", "fore", "dialog", "batch"]),
    "control": ("Evaluation and control", ["judge", "safety", "route"]),
}
DOMAINS = {
    "basic_cls": ("Cls", "Basic text classification"),
    "know": ("Know", "Knowledge and reasoning"),
    "social": ("Social", "Subjective and social judgments"),
    "pro": ("Pro", "Professional domains"),
    "ret": ("Ret", "Retrieval, ranking and matching"),
    "struct": ("Struct", "Structured data, tables and logs"),
    "code": ("Code", "Code"),
    "fore": ("Fore", "Forecasting"),
    "dialog": ("Dialog", "Dialogue and interaction"),
    "batch": ("Batch", "Batch semantic processing"),
    "judge": ("Judge", "LLM-as-judge and evaluators"),
    "safety": ("Safety", "Safety and moderation"),
    "route": ("Route", "Routing and orchestration"),
}
FAMILIES = {  # family label in the summary -> (family, how the probabilities are obtained)
    "System-1 models, native probabilities": ("System-1 model", "native probabilities"),
    "Closed LLMs, verbalized probabilities": ("Closed LLM", "verbalized probabilities"),
    "Open LLMs, probabilities read at the answer token": ("Open LLM", "label probabilities at the answer token"),
}
API_PREFIXES = ("copilot/", "bedrock/", "typesafe-ai/", "openai-decisions/", "openrouter-decisions/")
ACCESS = {"Copilot": "GitHub Copilot", "gateway": "Vercel AI Gateway", "OpenAI": "OpenAI API", "OpenRouter": "OpenRouter"}


def hf_link(model_id: str | None) -> str | None:
    """The Hugging Face page of an open-weight model id (org/name[@setting]); None for models behind an API."""
    if not model_id or model_id.startswith(API_PREFIXES):
        return None
    repo = model_id.split("@", 1)[0]
    return f"https://huggingface.co/{repo}" if repo.count("/") == 1 else None


def cost(row: dict) -> dict | None:
    if row.get("cost") is None:
        return None
    unit = row["cost_unit"]
    if unit == "gpu_s":
        return {"value": round(row["cost"], 1), "unit": "GPU-seconds (H100-equivalent)", "usd": None}
    usd = row["cost"] * (USD_PER_COPILOT_CREDIT if unit == "aiu" else 1.0)
    return {"value": round(usd, 4), "unit": "USD", "usd": round(usd, 4)}


def model_row(row: dict) -> dict:
    family, probabilities = FAMILIES[row["family"]]
    static = row.get("all") is not None
    agentic = row.get("steps") is not None
    where = row.get("latency_where")
    return {
        "name": row["model"],
        "family": family,
        "probabilities": probabilities,
        "reasons": bool(row.get("reasons")),  # writes a reasoning text before it answers
        "ids": {"static": row.get("three_id"), "agentic": row.get("agent_id")},
        "link": hf_link(row.get("three_id") or row.get("agent_id")),
        "accuracy": round(row["all"], 4) if static else None,
        "brier": round(row["brier_all"], 4) if static else None,
        "groups": {g: round(row[g], 4) for g in GROUPS} if static else None,
        "domains": {d: round(v, 4) for d, v in row["domains"].items()} if static else None,
        "latency_ms": round(row["latency_ms"], 1) if row.get("latency_ms") is not None else None,
        "latency_where": "local GPU (" + where + ")" if where in ("H100", "A100") else "network (" + ACCESS.get(where, where) + ")",
        "cost_per_1k_requests": cost(row),
        "output_tokens": round(row["output_tokens"], 1) if row.get("output_tokens") is not None else None,
        "agentic": {
            "step_accuracy": round(row["steps"], 4),
            "loop_success": round(row["loops"], 4) if row.get("loops") is not None else None,
            "steps_unanswered": round(row.get("steps_unanswered") or 0.0, 4),
            "decisions_unanswered": round(row.get("decisions_unanswered") or 0.0, 4),
        } if agentic else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("summary", help="the results summary (paper_groups.json)")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    models = [model_row(r) for r in summary["rows"]]
    ranked = sorted((m for m in models if m["accuracy"] is not None), key=lambda m: -m["accuracy"])
    for rank, m in enumerate(ranked, 1):
        m["rank"] = rank
    for m in models:
        m.setdefault("rank", None)
    models.sort(key=lambda m: (m["rank"] is None, m["rank"] or 0, -(m["agentic"] or {}).get("step_accuracy", 0)))
    out = {
        "benchmark": "S1-Arena",
        "updated": date.today().isoformat(),
        "non_agentic": {
            "requests": summary["items"],
            "domains": [{"id": d, "short": s, "name": n, "group": g} for g, (_, ds) in GROUPS.items() for d in ds for s, n in [DOMAINS[d]]],
            "groups": {g: name for g, (name, _) in GROUPS.items()},
            "accuracy": "mean over the tasks of the 13 domains; a question without an answer is wrong",
            "brier": "squared distance between the predicted distribution and the gold (one-hot or annotator) distribution",
            "latency": "median time of a single request",
            "cost": "per 1,000 requests: USD for hosted models, H100-equivalent GPU-seconds for local models",
        },
        "agentic": {
            "environments": summary["environments"],
            "step_accuracy": "accuracy on static agent single steps",
            "loop_success": "mean success rate over the closed-loop environments that every listed model played and that define success",
        },
        "models": models,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(models)} models ({len(ranked)} ranked) to {args.out}")


if __name__ == "__main__":
    main()
