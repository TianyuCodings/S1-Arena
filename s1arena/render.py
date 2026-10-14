"""One prompt template for every LLM, so no model gets a hand-tuned prompt.

LLMs are asked to verbalize a probability for each option. This is provisional:
the LLM configuration (thinking, logprobs, sampling) is still to be decided.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .schema import DecisionRequest

SYSTEM = (
    "You are a classifier. Read the input and answer every question. "
    "For each question give a probability for the allowed options; the probabilities "
    "of one question must sum to 1. Reply with a single JSON object and nothing else."
)


def messages(request: DecisionRequest) -> list[dict[str, str]]:
    lines = ["Input:", json.dumps(request.state, ensure_ascii=False, indent=1), "", "Questions:"]
    for name, question in request.questions.items():
        lines.append(f'- "{name}" ({question.type}): {question.instructions}')
        if question.type == "choice":
            for option, description in question.criteria.items():
                lines.append(f'    - "{option}": {description}')
        elif question.type == "score":
            for level, description in enumerate(question.criteria):
                lines.append(f'    - "{level}": {description}')
        else:
            lines.append('    - "true": the statement is true')
            lines.append('    - "false": the statement is false')
    lines += [
        "",
        'Reply with JSON of the form {"<question>": {"<option id>": <probability>, ...}, ...}. '
        "Options you leave out get probability 0.",
    ]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def parse(text: str, request: DecisionRequest) -> dict[str, dict[str, float]]:
    """Extract per-question distributions from a model reply.

    Replies may wrap the JSON in code fences, add prose (with braces, e.g. LaTeX) around it,
    or revise an earlier answer; the last top-level JSON object naming a question wins.
    """
    text = text or ""
    wanted = {_key(name) for name in request.questions}
    decoder = json.JSONDecoder()
    obj: dict[str, Any] | None = None
    unnamed: dict[str, Any] | None = None  # one-question requests: a lone distribution under another key
    only = next(iter(request.questions)) if len(request.questions) == 1 else None
    start = text.find("{")
    while start != -1:
        try:
            candidate, end = decoder.raw_decode(text, start)
        except ValueError:
            start = text.find("{", start + 1)
            continue
        if isinstance(candidate, dict) and wanted & {_key(k) for k in candidate}:
            obj = candidate
        elif only and isinstance(candidate, dict) and len(candidate) == 1:
            value = next(iter(candidate.values()))
            options = set(request.questions[only].option_ids())
            if isinstance(value, dict) and value and set(map(str, value)) <= options:
                unnamed = {only: value}
        start = text.find("{", end)
    obj = obj if obj is not None else unnamed
    if obj is None:
        raise ValueError("no JSON object naming the questions in reply")
    # tolerate keys such as '"funniness (score)"' that echo the prompt's formatting
    by_name = {_key(k): v for k, v in obj.items()}
    out: dict[str, dict[str, float]] = {}
    for name in request.questions:
        value = obj.get(name, by_name.get(_key(name)))
        if isinstance(value, dict):
            out[name] = {str(k): float(v) for k, v in value.items() if isinstance(v, (int, float))}
        elif isinstance(value, str):
            out[name] = {value: 1.0}
    return out


def _key(key: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(key)).strip().strip('"').lower()
