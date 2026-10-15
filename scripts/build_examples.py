"""Write leaderboard/examples.json: real question examples for the leaderboard website.

Usage (from the repository root, after `uv run s1arena download` and scripts/build_video.py):

    uv run --with highway-env==1.12.1 --with gymnasium==1.3.0 python scripts/build_examples.py

For each domain the script picks up to three requests from data/<task>.jsonl, from different leaves and, where it can,
of different question types. It uses only redistributable sources (never data_local/), skips sources and requests with
offensive or harmful text, and prefers short requests: long state values are cut with "…" and every example stays
under MAX_BYTES. The choice is deterministic: tasks and requests are ordered by a hash of SEED and their name or id,
so the same data always gives the same file. The closed-loop example is the first decision request of the episode in
the site's video (site/media/closed_loop.json: environment and seed), taken through the environment's own reset() and
observe(), with the option the model chose there. scripts/build_site.py embeds the file, so the Pages workflow needs no
data/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from s1arena.env import DATA, ROOT
from s1arena.registry import TASKS, discover, envs_under

OUT = ROOT / "leaderboard" / "examples.json"
SEED = "s1arena-site-examples-1"
PER_DOMAIN = 3
MAX_BYTES = 1500
STATE_CHARS = 560  # total characters of state text shown per example
INSTRUCTION_CHARS = 360
OPTION_CHARS = 110
MAX_OPTIONS = 12
VIDEO = ROOT / "site" / "media" / "closed_loop.json"  # written by scripts/build_video.py

DOMAINS = {  # order and names as in the dataset card (scripts/export_hf_dataset.py)
    "basic_cls": "Basic text classification", "know": "Knowledge and reasoning", "social": "Subjective and social judgments",
    "pro": "Professional domains", "ret": "Retrieval, ranking and matching", "struct": "Structured data, tables and logs",
    "code": "Code", "fore": "Forecasting", "dialog": "Dialogue and interaction", "batch": "Batch semantic processing",
    "judge": "LLM-as-judge and evaluators", "safety": "Safety and moderation", "route": "Routing and orchestration",
    "step": "Agent single steps (static)", "game": "Games (static)", "embodied": "Embodied and robotics (static)",
}
# Sources about toxicity, hate, harm, attacks or unsafe advice never appear on the site; in the safety domain only
# the PII sources do (benign, partly synthetic text).
SKIP_SOURCE = re.compile(r"tox|hate|harm|offens|jailbreak|injection|abuse|sexis|misogyn|racis|insult|profan|moderation|"
                         r"implicit|dices|persuasion|deception|unsafe|safetext|injur|refusal|cosapien|dynabench|red_?team")
SAFETY_ONLY = ("safety/pii/",)
SKIP_TEXT = re.compile(r"\b(fuck|shit|bitch|nigg|fag|retard|cunt|whore|slut|rape|porn|nazi|terror|suicid|kill|murder|"
                       r"assault|abuse|sex|nude|naked|racis|idiot|stupid|hate|damn|drug|weapon|bomb|gun|blood|dead|death|"
                       r"die[sd]?|gay|lesbian|muslim|jew|christian|islam|abortion|trump|biden|hitler)", re.I)


def rank(*parts: str) -> str:
    return hashlib.sha256(":".join((SEED, *parts)).encode()).hexdigest()


def cut(text: str, limit: int) -> str:
    """The text with its line breaks (boards, tables and code keep their layout), cut to `limit` characters."""
    text = "\n".join(line.rstrip() for line in text.strip().splitlines())
    return text if len(text) <= limit else text[: max(1, limit - 1)].rstrip() + "…"


def state_rows(state: dict[str, Any], chars: int = STATE_CHARS) -> list[list[str]]:
    """The state as key/value rows; each value is text or compact JSON. Short values stay whole; the long ones share
    what is left of `chars` equally."""
    values = {k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(", ", ": ")) for k, v in state.items()}
    limits, budget = {}, chars
    for i, (key, value) in enumerate(sorted(values.items(), key=lambda kv: len(kv[1]))):
        limits[key] = max(60, min(len(value), budget // (len(values) - i)))
        budget -= limits[key]
    return [[key, cut(value, limits[key])] for key, value in values.items()]


def gold_view(question: dict, gold: dict) -> dict:
    dist = {str(k): float(v) for k, v in (gold.get("dist") or {}).items() if v}
    label = gold.get("label")
    if label is None and dist:
        label = max(dist, key=dist.get)
    view: dict[str, Any] = {"label": None if label is None else str(label)}
    if dist and max(dist.values()) < 0.999:  # a soft label (annotator shares): show the distribution
        view["dist"] = [[k, round(p, 3)] for k, p in sorted(dist.items(), key=lambda kv: -kv[1])[:6]]
    return view


def options(question: dict) -> tuple[list[list[str]], int]:
    if question["type"] == "boolean":
        return [["true", ""], ["false", ""]], 0
    crit = question["criteria"]
    pairs = list(crit.items()) if isinstance(crit, dict) else [[str(i), text] for i, text in enumerate(crit)]
    shown = [[str(k), cut(str(v or ""), OPTION_CHARS)] for k, v in pairs[:MAX_OPTIONS]]
    return shown, len(pairs) - len(shown)


def example(item: dict, card: dict, domain: str) -> dict | None:
    """One example in the site's format, or None when the request does not fit the card."""
    questions = item["request"]["questions"]
    name, question = next(iter(questions.items()))
    if question["type"] == "choice" and len(question["criteria"]) > MAX_OPTIONS:
        return None
    opts, more = options(question)
    ex = {
        "domain": domain, "task": item["task"], "source": card["source"], "url": card["url"], "license": card["license"],
        "question": name, "type": question["type"], "instructions": cut(question["instructions"], INSTRUCTION_CHARS),
        "state": state_rows(item["request"]["state"]), "options": opts, "more_options": more,
        "more_questions": len(questions) - 1, "gold": gold_view(question, item["gold"].get(name) or {}),
    }
    text = json.dumps(ex, ensure_ascii=False)
    if len(text.encode()) > MAX_BYTES or SKIP_TEXT.search(json.dumps(item["request"], ensure_ascii=False)):
        return None
    if ex["gold"]["label"] is None and "dist" not in ex["gold"]:
        return None
    return ex


def fits(task: str, domain: str, n: int = 2) -> list[dict]:
    """The first `n` requests of the task (in hash order) that fit the card."""
    card = TASKS[task]().card()
    with (DATA / f"{task}.jsonl").open(encoding="utf-8") as f:  # one JSON object per line (\n only, as written)
        items = sorted((json.loads(line) for line in f if line.strip()), key=lambda it: rank(it["id"]))
    out = []
    for item in items:
        if len(out) < n and (ex := example(item, card, domain)) is not None:
            out.append(ex)
    return out


def pick(domain: str) -> list[dict]:
    tasks = []
    for path in sorted((DATA / domain).rglob("*.jsonl")):
        task = path.relative_to(DATA).with_suffix("").as_posix()
        cls = TASKS.get(task)
        if cls is None or not cls.redistributable or SKIP_SOURCE.search(task):
            continue
        if domain == "safety" and not task.startswith(SAFETY_ONLY):
            continue
        tasks.append(task)
    tasks.sort(key=lambda t: rank(t))
    chosen: list[dict] = []
    found: dict[str, list[dict]] = {}
    leaf = lambda task: task.rsplit("/", 1)[0]
    # passes: a new leaf and a new question type, then a new leaf, then any task not chosen yet
    for new_type, new_leaf in ((True, True), (False, True), (False, False)):
        for task in tasks:
            if len(chosen) == PER_DOMAIN:
                return chosen
            if any(ex["task"] == task or (new_leaf and leaf(ex["task"]) == leaf(task)) for ex in chosen):
                continue
            if task not in found:
                found[task] = fits(task, domain)
            if not found[task] or (new_type and any(c["type"] == found[task][0]["type"] for c in chosen)):
                continue
            chosen.append(found[task][0])
    if len(chosen) == 1 and len(found[chosen[0]["task"]]) > 1:  # a domain with one usable source (safety): two requests
        chosen.append(found[chosen[0]["task"]][1])
    return chosen


def closed_loop() -> dict:
    video = json.loads(VIDEO.read_text(encoding="utf-8"))
    cls = next(c for c in envs_under(video["env"]) if c.name == video["env"])
    env = cls()
    try:
        env.reset(video["seed"])
        request = json.loads(env.observe().model_dump_json())
        card = env.card()
    finally:
        env.close()
    name, question = next(iter(request["questions"].items()))
    opts, more = options(question)
    return {
        "domain": "closed_loop", "task": cls.name, "source": card["source"], "url": card["url"], "license": card["license"],
        "question": name, "type": question["type"], "instructions": cut(question["instructions"], INSTRUCTION_CHARS),
        "state": state_rows(request["state"], STATE_CHARS + 100), "options": opts, "more_options": more,
        "more_questions": len(request["questions"]) - 1, "gold": None,
        "scoring": cut(card.get("notes", ""), 300), "seed": video["seed"],
        "chosen": {"model": video["model"], **video["trace"][0]},  # the model's answer to this request in the video
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    discover()
    out = {"seed": SEED, "domains": {d: name for d, name in DOMAINS.items()},
           "examples": {d: pick(d) for d in DOMAINS}, "closed_loop": closed_loop()}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for d, exs in out["examples"].items():
        print(f"{d:<10} {len(exs)}  " + ", ".join(f"{ex['task'].split('/', 1)[1]} ({ex['type']})" for ex in exs))
    print(f"closed loop: {out['closed_loop']['task']}; wrote {args.out} ({Path(args.out).stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
