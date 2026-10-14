"""Tool arguments as typed questions for the τ² customer-service harness (customer_service.py).

A classifier cannot write free text, so each argument of the chosen tool is asked as a question over candidate
values collected deterministically, in this order (duplicates removed, first occurrence kept):

- the tool's JSON schema: its enum values, its default value, null for a nullable field;
- the ticket: its sentences, the values of its "label: value" fields, quoted phrases, numbers (with or without a
  unit), and tokens shaped like phone numbers, e-mail addresses, IDs and ISO dates;
- the policy: the literal values written in its code spans (for example `mode="4g_5g_preferred"`), its numbers and
  its ISO dates;
- earlier tool results, errors included: every string (up to STRING_MAX characters) and number of a JSON result;
  in a text result the quoted phrases, the shaped tokens above, URLs, numbers and comma-separated lists after a colon.

Only values of the argument's JSON type are offered: strings for a string, numbers for a number, whole numbers for
an integer. A boolean argument is a boolean question; an enum argument a choice over the schema's values; an object
argument is asked field by field (one question per leaf field); a list argument as one boolean question per
candidate ("include this value?"). A choice keeps the first MAX_OPTIONS candidates. An argument with a single
candidate is filled in without asking; one with none gets its default, or null.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ...schema import MAX_OPTIONS, Question

STRING_MAX = 200  # longer strings (long descriptions) are not offered as values
NULL_TEXT = "null (no value)"
_MISSING = object()

PHONE = re.compile(r"(?<![\w-])\d{3}-\d{3}-\d{4}(?![\w-])")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"https?://[^\s'\"<>|]+")
ID = re.compile(r"(?<![\w-])[A-Z]{1,3}\d{3,}(?![\w-])")
DATE = re.compile(r"(?<![\w-])\d{4}-\d{2}-\d{2}(?![\w-])")
# A number standing alone or with a unit ("2GB", "2.0 GB", "$5.00", "80%"); digits inside dates, times, phone
# numbers, IDs or names such as "5G" are not numbers.
NUMBER = re.compile(r"(?<![\w.\-/:$])\$?(\d+(?:\.\d+)?)(?:\s?(?:GB|MB|Mbps|USD|%))?(?![\w\-/:]|\.\d)")
QUOTED = re.compile(r"(?<!\w)'([^'\n]{1,80}?)'(?!\w)|\"([^\"\n]{1,80}?)\"")
LIST_MARKER = re.compile(r"(?m)^\s*\d+\.\s")  # "1. " starting a line numbers a list; it is not a value
CODE = re.compile(r"`([^`\n]+)`")
FIELD = re.compile(r"^([A-Za-z][A-Za-z ]{0,40}):\s*(.+?)\.?$")
LIST_ITEM = re.compile(r"[\w][\w .'/-]{0,40}")


# ---------------------------------------------------------------------------------------------------
# Candidate values.


@dataclass
class Values:
    """Candidate strings and numbers in order of first appearance."""

    strings: list[str] = field(default_factory=list)
    numbers: list[float] = field(default_factory=list)

    def add_string(self, text: Any) -> None:
        if isinstance(text, str):
            text = text.strip()
            if text and len(text) <= STRING_MAX and text not in self.strings:
                self.strings.append(text)

    def add_number(self, value: Any) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if value == value and abs(value) != float("inf") and value not in self.numbers:
                self.numbers.append(value)

    def extend(self, other: Values) -> None:
        for text in other.strings:
            self.add_string(text)
        for value in other.numbers:
            self.add_number(value)


def _numbers(text: str, out: Values) -> None:
    for match in NUMBER.finditer(LIST_MARKER.sub(" ", text)):
        out.add_number(float(match.group(1)))


def _shaped(text: str, out: Values) -> None:
    """Quoted phrases, phone numbers, e-mail addresses, URLs, IDs, ISO dates and numbers found in free text."""
    spans = []
    for pattern in (PHONE, EMAIL, URL, ID, DATE):
        for match in pattern.finditer(text):
            spans.append((match.start(), match.group(0).rstrip(".,;:)")))
    for match in QUOTED.finditer(text):
        spans.append((match.start(), match.group(1) or match.group(2)))
    for _, value in sorted(spans, key=lambda s: s[0]):
        out.add_string(value)
    _numbers(text, out)


def ticket_values(ticket: str) -> Values:
    out = Values()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ticket.strip()) if s.strip()]
    for sentence in sentences:
        out.add_string(sentence)
    for sentence in sentences:
        for segment in re.split(r",\s+", sentence):
            match = FIELD.match(segment.strip())
            if match:
                out.add_string(match.group(2))
    _shaped(ticket, out)
    return out


def policy_values(policy: str) -> Values:
    out = Values()
    for code in CODE.findall(policy):
        for match in QUOTED.finditer(code):
            out.add_string(match.group(1) or match.group(2))
    for match in DATE.finditer(policy):
        out.add_string(match.group(0))
    _numbers(policy, out)
    return out


def output_values(content: str) -> Values:
    """Values in one tool result (the ToolMessage content as the agent receives it)."""
    out = Values()
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        data = None
    if isinstance(data, (dict, list)):
        _walk(data, out)
        return out
    text = str(content)
    _shaped(text, out)
    for line in text.splitlines():
        if ": " not in line:
            continue
        items = [item.strip() for item in re.split(r",\s*", line.rsplit(": ", 1)[1].strip().rstrip("."))]
        if len(items) >= 2 and all(LIST_ITEM.fullmatch(item) for item in items):
            for item in items:
                out.add_string(item)
    return out


def _walk(data: Any, out: Values) -> None:
    if isinstance(data, dict):
        for value in data.values():
            _walk(value, out)
    elif isinstance(data, list):
        for value in data:
            _walk(value, out)
    elif isinstance(data, str):
        out.add_string(data)
    else:
        out.add_number(data)


# ---------------------------------------------------------------------------------------------------
# Arguments from the tool's JSON schema.


@dataclass
class ArgSpec:
    path: tuple[str, ...]  # ("apn_settings", "mmsc_url") for a field of an object argument
    kind: str  # "string", "number", "integer", "boolean", "enum" or "array"
    description: str
    enum: list[Any] | None = None
    default: Any = _MISSING
    nullable: bool = False
    item: ArgSpec | None = None  # the element of an array

    @property
    def name(self) -> str:
        return ".".join(self.path)


def arg_specs(parameters: dict[str, Any]) -> list[ArgSpec]:
    """One spec per argument (per leaf field for object arguments), in the schema's order."""
    defs = parameters.get("$defs", {})
    out: list[ArgSpec] = []
    for name, prop in parameters.get("properties", {}).items():
        _flatten((name,), prop, defs, out)
    return out


def _resolve(prop: dict[str, Any], defs: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Follow $ref and anyOf; returns the schema and whether null is allowed. Among alternatives an enum or an
    object with named fields is preferred to a bare type (e.g. NetworkModePreference over "string")."""
    nullable = False
    while True:
        if "$ref" in prop:
            outer = {k: v for k, v in prop.items() if k != "$ref"}
            prop = {**defs[prop["$ref"].rsplit("/", 1)[-1]], **outer}
            continue
        if "anyOf" in prop:
            outer = {k: v for k, v in prop.items() if k != "anyOf"}
            options = [a for a in prop["anyOf"] if a.get("type") != "null"]
            nullable = nullable or len(options) < len(prop["anyOf"])
            resolved = [_resolve(a, defs)[0] for a in options] or [{"type": "string"}]
            best = next((r for r in resolved if "enum" in r), None) or next((r for r in resolved if r.get("properties")), None) or resolved[0]
            prop = {**best, **outer}
            continue
        return prop, nullable


def _flatten(path: tuple[str, ...], prop: dict[str, Any], defs: dict[str, Any], out: list[ArgSpec]) -> None:
    schema, nullable = _resolve(prop, defs)
    description = prop.get("description") or schema.get("description") or prop.get("title") or schema.get("title") or ""
    default = prop.get("default", schema.get("default", _MISSING))
    kind = schema.get("type")
    if "enum" in schema:
        out.append(ArgSpec(path, "enum", description, enum=list(schema["enum"]), default=default, nullable=nullable))
    elif kind == "object" and schema.get("properties"):
        for key, sub in schema["properties"].items():
            _flatten(path + (key,), sub, defs, out)
    elif kind == "array":
        items, _ = _resolve(schema.get("items", {"type": "string"}), defs)
        item_kind = "enum" if "enum" in items else items.get("type") if items.get("type") in ("string", "number", "integer", "boolean") else "string"
        item = ArgSpec(path, item_kind, description, enum=items.get("enum"))
        out.append(ArgSpec(path, "array", description, default=default, nullable=nullable, item=item))
    elif kind in ("string", "number", "integer", "boolean"):
        out.append(ArgSpec(path, kind, description, default=default, nullable=nullable))
    else:
        out.append(ArgSpec(path, "string", description, default=default, nullable=nullable))


def candidates(spec: ArgSpec, pool: Values) -> list[Any]:
    """Candidate values of an argument, typed and in order, without null."""
    values: list[Any] = []

    def add(value: Any) -> None:
        if value is not None and value is not _MISSING and value not in values:
            values.append(value)

    if spec.default is not _MISSING and spec.kind not in ("array", "boolean"):
        add(spec.default)
    if spec.kind == "enum":
        for value in spec.enum or []:
            add(value)
    elif spec.kind == "boolean":
        values = [True, False]
    elif spec.kind == "string":
        for text in pool.strings:
            add(text)
    elif spec.kind == "number":
        values = [float(v) for v in values if isinstance(v, (int, float))]
        for number in pool.numbers:
            add(float(number))
    elif spec.kind == "integer":
        values = [int(v) for v in values if isinstance(v, (int, float)) and float(v).is_integer()]
        for number in pool.numbers:
            if number.is_integer():
                add(int(number))
    elif spec.kind == "array" and spec.item is not None:
        return candidates(spec.item, pool)
    return values


def _render(value: Any) -> str:
    return NULL_TEXT if value is None else json.dumps(value, ensure_ascii=False)


def _type_text(spec: ArgSpec) -> str:
    kind = {"enum": "one of the values the schema allows", "integer": "whole number"}.get(spec.kind, spec.kind)
    return kind + (", or null" if spec.nullable else "")


def _described(spec: ArgSpec) -> str:
    text = spec.description.strip()
    return f" Argument description: {text}" + ("" if text.endswith((".", "?", "!")) else ".") if text else ""


@dataclass
class ArgForm:
    """The questions for one tool call and how to turn the answers into the call's arguments."""

    tool: str
    questions: dict[str, Question] = field(default_factory=dict)
    options: dict[str, dict[str, Any]] = field(default_factory=dict)  # question -> option id -> value
    fixed: dict[tuple[str, ...], Any] = field(default_factory=dict)  # arguments filled in without asking
    lists: dict[tuple[str, ...], list[tuple[str, Any]]] = field(default_factory=dict)  # list argument -> (question, value)
    paths: dict[str, tuple[str, ...]] = field(default_factory=dict)  # question -> argument path

    def arguments(self, choice: dict[str, str]) -> dict[str, Any] | None:
        """The call's arguments, or None when an answer is missing or unusable."""
        values: dict[tuple[str, ...], Any] = dict(self.fixed)
        for name, question in self.questions.items():
            answer = choice.get(name)
            if answer not in question.option_ids():
                return None
            if name in self.options:
                values[self.paths[name]] = self.options[name][answer]
        for path, items in self.lists.items():
            values[path] = [value for name, value in items if choice.get(name) == "true"]
        out: dict[str, Any] = {}
        for path, value in values.items():
            node = out
            for key in path[:-1]:
                node = node.setdefault(key, {})
            node[path[-1]] = value
        return out


def arg_form(tool: str, specs: list[ArgSpec], pool: Values) -> ArgForm:
    form = ArgForm(tool=tool)
    sources = "values from the tool's schema, the ticket, the policy and earlier tool results"
    for spec in specs:
        if spec.kind == "array":
            items = candidates(spec, pool)[:MAX_OPTIONS]
            form.lists[spec.path] = []
            for k, value in enumerate(items, 1):
                name = f"{spec.name}[{k}]"
                form.questions[name] = Question(type="boolean", instructions=(
                    f'Is {_render(value)} one of the items of the list argument "{spec.name}" of {tool}?{_described(spec)} '
                    f"The candidate items are {sources}."))
                form.lists[spec.path].append((name, value))
            continue
        if spec.kind == "boolean" and not spec.nullable:
            form.questions[spec.name] = Question(type="boolean", instructions=f'Is the argument "{spec.name}" of {tool} true?{_described(spec)}')
            form.options[spec.name] = {"true": True, "false": False}
            form.paths[spec.name] = spec.path
            continue
        values = candidates(spec, pool)
        values = values[: MAX_OPTIONS - 1] if spec.nullable else values[:MAX_OPTIONS]
        if spec.nullable:
            values.append(None)
        if len(values) < 2:
            form.fixed[spec.path] = values[0] if values else (None if spec.default is _MISSING else spec.default)
            continue
        source = "the values the tool's schema allows" if spec.kind == "enum" else sources
        form.options[spec.name] = {f"v{k}": value for k, value in enumerate(values, 1)}
        form.paths[spec.name] = spec.path
        form.questions[spec.name] = Question(
            type="choice",
            instructions=f'Which value to pass as the argument "{spec.name}" of {tool}?{_described(spec)} Type: {_type_text(spec)}. The options are {source}.',
            criteria={f"v{k}": _render(value) for k, value in enumerate(values, 1)},
        )
    return form
