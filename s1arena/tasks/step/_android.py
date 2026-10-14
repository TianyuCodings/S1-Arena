"""AndroidControl helpers: the official test split, the accessibility trees (protobuf text format) and a
text rendering of a screen with numbered interactive elements.

The official TFRecords (50 GB with screenshots) are too large to stream for 1,543 test episodes, so two
public conversions of them are read instead, joined by the official episode ids: ckg/AndroidControlParsed-20k
(accessibility trees as protobuf text and the actions) and HarrytheOrange/parsed_AndroidControl (goals, step
instructions and actions in the original JSON format). Their actions agree for every test step.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

from ..common import fetch

SPLITS_URL = "https://storage.googleapis.com/gresearch/android_control/splits.json"
TREES_REPO = "ckg/AndroidControlParsed-20k"
EPISODES_REPO = "HarrytheOrange/parsed_AndroidControl"

# ---- protobuf text format ----

_TOKEN = re.compile(r'\s*(\{|\}|:|"(?:[^"\\]|\\.)*"|[^\s{}:"]+)', re.S)


def _unescape(quoted: str) -> str:
    """C-style escapes of the text format; non-ASCII text is written as octal UTF-8 bytes."""
    body, out, i = quoted[1:-1], bytearray(), 0
    simple = {"n": 10, "t": 9, "r": 13, '"': 34, "'": 39, "\\": 92, "a": 7, "b": 8, "f": 12, "v": 11, "?": 63}
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt in "01234567":
                j = i + 1
                while j < len(body) and j < i + 4 and body[j] in "01234567":
                    j += 1
                out.append(int(body[i + 1:j], 8) & 0xFF)
                i = j
                continue
            if nxt == "x":
                j = i + 2
                while j < len(body) and j < i + 4 and body[j] in "0123456789abcdefABCDEF":
                    j += 1
                out.append(int(body[i + 2:j], 16))
                i = j
                continue
            out.append(simple.get(nxt, ord(nxt) if ord(nxt) < 128 else 63))
            i += 2
            continue
        out.extend(ch.encode("utf-8"))
        i += 1
    return out.decode("utf-8", errors="replace")


def parse_textproto(text: str) -> dict:
    """Nested dicts; every field maps to a list of values (repeated or not)."""
    tokens = _TOKEN.findall(text)
    root: dict = defaultdict(list)
    stack, i = [root], 0
    while i < len(tokens):
        name = tokens[i]
        i += 1
        if name == "}":
            stack.pop()
            continue
        if i < len(tokens) and tokens[i] == ":":
            i += 1
        if i >= len(tokens):
            break
        if tokens[i] == "{":
            child: dict = defaultdict(list)
            stack[-1][name].append(child)
            stack.append(child)
        else:
            value = tokens[i]
            if value.startswith('"'):
                value = _unescape(value)
            elif value in ("true", "false"):
                value = value == "true"
            elif re.fullmatch(r"-?\d+", value):
                value = int(value)
            elif re.fullmatch(r"-?\d*\.\d+(?:[eE][-+]?\d+)?", value):
                value = float(value)
            stack[-1][name].append(value)
        i += 1
    return root


def _first(node: dict, field: str, default=None):
    values = node.get(field) or []
    return values[0] if values else default


# ---- screens ----


def screen_elements(tree_text: str, width: int = 1080, height: int = 2400) -> list[dict]:
    """Visible nodes with their bounds, label and flags, in reading order (top to bottom, left to right)."""
    forest = parse_textproto(tree_text)
    nodes = []
    for window in forest.get("windows", []):
        kind = str(_first(window, "window_type", ""))
        for tree in window.get("tree", []):
            for node in tree.get("nodes", []):
                b = _first(node, "bounds_in_screen", {}) or {}
                left, top = max(0, _first(b, "left", 0)), max(0, _first(b, "top", 0))
                right, bottom = min(width, _first(b, "right", 0)), min(height, _first(b, "bottom", 0))
                if right <= left or bottom <= top:
                    continue
                own = next((str(_first(node, f)) for f in ("text", "content_description", "hint_text", "tooltip_text")
                            if str(_first(node, f, "") or "").strip()), "")
                nodes.append({
                    "bounds": (left, top, right, bottom), "window": kind,
                    "cls": str(_first(node, "class_name", "")).rsplit(".", 1)[-1],
                    "text": " ".join(own.split()),
                    "clickable": bool(_first(node, "is_clickable", False)), "long": bool(_first(node, "is_long_clickable", False)),
                    "editable": bool(_first(node, "is_editable", False)), "checkable": bool(_first(node, "is_checkable", False)),
                    "checked": bool(_first(node, "is_checked", False)), "selected": bool(_first(node, "is_selected", False)),
                    "scrollable": bool(_first(node, "is_scrollable", False)), "focused": bool(_first(node, "is_focused", False)),
                })
    for n in nodes:
        n["interactive"] = n["clickable"] or n["long"] or n["editable"] or n["checkable"]
        n["area"] = (n["bounds"][2] - n["bounds"][0]) * (n["bounds"][3] - n["bounds"][1])
    texts = [n for n in nodes if n["text"]]
    for n in nodes:  # interactive containers without their own text are named by the text inside them
        if n["interactive"] and not n["text"]:
            l, t, r, b = n["bounds"]
            inner = [x["text"] for x in texts if x is not n and x["bounds"][0] >= l and x["bounds"][1] >= t and x["bounds"][2] <= r and x["bounds"][3] <= b]
            n["label"] = " | ".join(dict.fromkeys(inner))[:120]
        else:
            n["label"] = n["text"][:160]
    unique, seen = [], set()
    for n in sorted(nodes, key=lambda n: (n["bounds"][1], n["bounds"][0], -n["area"])):
        key = (n["bounds"], n["label"], n["interactive"])
        if key in seen or not (n["interactive"] or n["text"]):
            continue
        seen.add(key)
        unique.append(n)
    return unique


def element_at(elements: list[dict], x: float, y: float, interactive_only: bool = True) -> dict | None:
    inside = [e for e in elements if (e["interactive"] or not interactive_only)
              and e["bounds"][0] <= x <= e["bounds"][2] and e["bounds"][1] <= y <= e["bounds"][3]]
    return min(inside, key=lambda e: e["area"]) if inside else None


def describe(e: dict) -> str:
    flags = [f for f, on in (("clickable", e["clickable"]), ("long-clickable", e["long"]), ("editable", e["editable"]),
                             ("checkable", e["checkable"]), ("checked", e["checked"]), ("selected", e["selected"]),
                             ("focused", e["focused"]), ("scrollable", e["scrollable"])) if on]
    label = f" \"{e['label']}\"" if e["label"] else " (no text)"
    return f"{e['cls'] or 'View'}{label}" + (f" [{', '.join(flags)}]" if flags else "")


# ---- data ----


def test_episodes() -> list[dict]:
    """Official AndroidControl test episodes with, per step, the accessibility tree before the action and the action."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    from ...env import load_env

    load_env()
    with open(fetch(SPLITS_URL, "android_control_splits.json"), encoding="utf-8") as f:
        test = set(json.load(f)["test"])
    episodes = {}
    with open(hf_hub_download(EPISODES_REPO, "parsed_android_control.jsonl", repo_type="dataset"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["episode_id"] in test:
                episodes[r["episode_id"]] = {"episode_id": r["episode_id"], "goal": r["goal"], "instructions": r["step_instructions"],
                                             "actions": r["actions"], "screens": {}}
    files = sorted(f for f in HfApi().list_repo_files(TREES_REPO, repo_type="dataset") if f.endswith(".parquet"))
    for name in files:
        table = pq.read_table(hf_hub_download(TREES_REPO, name, repo_type="dataset"))
        ids = table.column("episode_id").to_pylist()
        keep = [i for i, e in enumerate(ids) if e in episodes]
        for row in table.take(keep).to_pylist():
            messages = {m["role"]: m["content"] for m in row["messages"]}
            episodes[row["episode_id"]]["screens"][row["step_id"]] = {"tree": messages.get("screen_description", ""),
                                                                      "app": messages.get("active_application", ""),
                                                                      "action": messages.get("assistant", "")}
    return [episodes[e] for e in sorted(episodes)]


def action_matches(recorded: str, action: dict) -> bool:
    """The conversion's function call agrees with the original JSON action."""
    try:
        call = json.loads(recorded.replace("!FUNCTIONCALL", "", 1))
    except json.JSONDecodeError:
        return False
    return call.get("name") == action.get("action_type") and all(action.get(k) == v for k, v in (call.get("parameters") or {}).items())
