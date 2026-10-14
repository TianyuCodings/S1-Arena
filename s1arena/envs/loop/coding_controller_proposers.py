"""Proposers for loop/coding_controller: the fixed part of the agent that writes candidate next actions.

The proposer belongs to the environment, so it must be a function of the trajectory: the same instance and the same
actions executed so far give the same candidates, for every model under test. Interface:

    class Proposer:
        name, version       # identify the proposer in cache keys and cards
        cacheable           # True for proposers that call a model
        propose(context) -> list[Action]

    context.instance  the SWE-bench row (the scripted proposer also reads the gold patch and the distractors; a model
                      proposer must read only problem_statement, repo and version)
    context.steps     the actions executed so far with their (normalized) outputs; decisions that ran nothing are
                      not part of the trajectory
    context.diff      the current changes against the initial repository

An Action is a shell command, a unified diff to apply, or a revert to the initial state. Submitting is not a
candidate: the environment always offers it.

Caching: proposals of cacheable proposers are stored under .cache/coding_controller/proposals/<proposer>/, keyed by
SHA-256 of (proposer name and version, instance id, and the executed actions as exact text). The key leaves out the
outputs, which are a function of the actions in a fixed container: a model proposer is called at most once per
trajectory, and an output that differs slightly between runs (a flaky test) cannot change the candidates.

ScriptedProposer needs no model: it offers the reference (gold) patch and failing patches from public SWE-bench
submissions as edits, views of the code the edits touch, a run of the repository's existing tests for the issue, the
issue's example code when it has one, and a revert once an edit has been applied.

ChatProposer asks a chat model (OpenAI-compatible chat completions, temperature 0) for K candidates in one call,
through the project's AI Gateway or any other base URL (e.g. a vLLM server). It is not used by a registered
environment yet: which model proposes is an open decision.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from ...env import ROOT

CACHE = ROOT / ".cache" / "coding_controller" / "proposals"


@dataclass(frozen=True)
class Action:
    kind: str  # "command" | "patch" | "revert"
    text: str = ""  # the shell command or the unified diff
    title: str = ""  # one line shown with the action
    tag: str = ""  # the proposer's own label, never shown to the model (scripted: gold, distractor:<system>, view, test, example, revert)

    def key(self) -> list[str]:
        return [self.kind, self.text]

    def to_json(self) -> dict[str, str]:
        return {"kind": self.kind, "text": self.text, "title": self.title, "tag": self.tag}


@dataclass
class Step:
    action: Action
    output: str
    exit_code: int
    seconds: float
    timed_out: bool = False


@dataclass
class Context:
    instance: dict[str, Any]
    steps: list[Step] = field(default_factory=list)
    diff: str = ""


class Proposer:
    name = ""
    version = "1"
    cacheable = False

    def propose(self, context: Context) -> list[Action]:
        raise NotImplementedError

    def ident(self) -> str:
        return f"{self.name}@{self.version}"


def trajectory_key(proposer: Proposer, context: Context) -> str:
    payload = {"proposer": proposer.ident(), "instance": context.instance["instance_id"], "actions": [s.action.key() for s in context.steps]}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


_CACHE_LOCK = threading.Lock()


def propose(proposer: Proposer, context: Context) -> list[Action]:
    """The proposer's candidates for this trajectory, from the cache when the proposer is cacheable."""
    if not proposer.cacheable:
        return proposer.propose(context)
    key = trajectory_key(proposer, context)
    path = CACHE / re.sub(r"[^\w.@-]+", "_", proposer.ident()) / key[:2] / f"{key}.json"
    if path.exists():
        return [Action(**a) for a in json.loads(path.read_text(encoding="utf-8"))["actions"]]
    actions = proposer.propose(context)
    record = {"key": key, "proposer": proposer.ident(), "instance": context.instance["instance_id"], "steps": len(context.steps),
              "actions": [a.to_json() for a in actions], "usage": getattr(proposer, "last_usage", None)}
    with _CACHE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.part")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
    return actions


# ---------------------------------------------------------------------------------------------------
# Patches: parsing helpers shared with the environment.

_SECTION = re.compile(r"^diff --git ", re.M)
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.M)


def patch_sections(patch: str) -> list[str]:
    """The file sections of a unified git diff ('diff --git' blocks)."""
    starts = [m.start() for m in _SECTION.finditer(patch)]
    return [patch[a:b] for a, b in zip(starts, starts[1:] + [len(patch)])]


def section_path(section: str) -> str:
    for line in section.splitlines():
        if line.startswith("+++ b/"):
            return line[6:].strip()
        if line.startswith("--- a/"):
            path = line[6:].strip()
            if path:
                return path
    first = section.splitlines()[0] if section else ""
    m = re.match(r"diff --git a/(\S+) b/(\S+)", first)
    return m.group(2) if m else ""


def patch_files(patch: str) -> list[str]:
    return [section_path(s) for s in patch_sections(patch)]


def patch_size(patch: str) -> tuple[int, int]:
    """Lines added and removed."""
    added = removed = 0
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


# ---------------------------------------------------------------------------------------------------
# Scripted proposer.

VIEW_MARGIN = 10  # lines of context added around each hunk an edit touches
VIEW_MAX_LINES = 80
MAX_VIEWS = 6
EXAMPLE_MAX_CHARS = 2500
_TEST_MARKERS = re.compile(r": '>>>>> Start Test Output'\n(.*?)\n: '>>>>> End Test Output'", re.S)
_CODE_BLOCK = re.compile(r"```[ \t]*(python3?|py|pycon)?[ \t]*\r?\n(.*?)```", re.S | re.I)


def code_views(patches: list[str]) -> list[Action]:
    """Numbered views of the original lines the patches touch: every hunk of every patch with VIEW_MARGIN lines around
    it, overlapping ranges of a file merged, split into views of at most VIEW_MAX_LINES lines; the first MAX_VIEWS in
    file and line order."""
    ranges: dict[str, list[list[int]]] = {}
    for patch in patches:
        for section in patch_sections(patch):
            path = section_path(section)
            for m in _HUNK.finditer(section):
                start, length = int(m.group(1)), int(m.group(2) or 1)
                ranges.setdefault(path, []).append([max(1, start - VIEW_MARGIN), start + max(length, 1) - 1 + VIEW_MARGIN])
    views = []
    for path in sorted(ranges):
        merged: list[list[int]] = []
        for first, last in sorted(ranges[path]):
            if merged and first <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], last)
            else:
                merged.append([first, last])
        for first, last in merged:
            for a in range(first, last + 1, VIEW_MAX_LINES):
                b = min(last, a + VIEW_MAX_LINES - 1)
                views.append(Action("command", f"nl -ba {path} | sed -n '{a},{b}p'", title=f"Show lines {a}-{b} of {path}", tag="view"))
    return views[:MAX_VIEWS]


def existing_tests_command(instance: dict[str, Any]) -> str | None:
    """The instance's own test command (from its eval script), restricted to test files that exist before the
    hidden test patch is applied. None if no such file is left (or the command would run the whole suite)."""
    m = _TEST_MARKERS.search(instance.get("eval_script") or "")
    if not m:
        return None
    command = m.group(1).strip()
    tokens = command.split()
    directives: dict[str, bool] = {}  # directive -> the test file is new in the test patch
    for section in patch_sections(instance.get("test_patch") or ""):
        path = section_path(section)
        new = "\nnew file mode" in section or "\n--- /dev/null" in section
        forms = {path, path[:-3] if path.endswith(".py") else path}
        if instance["repo"] == "django/django" and path.startswith("tests/") and path.endswith(".py"):
            forms.add(path[len("tests/"):-3].replace("/", "."))
        for token in tokens:
            if token in forms:
                directives[token] = directives.get(token, False) or new
    if not directives or all(directives.values()):
        return None
    kept = [t for t in tokens if not directives.get(t, False)]
    return " ".join(kept)


def _echo(line: str) -> str:
    """A console input line that is a bare expression becomes print(expression), since a script does not echo values."""
    try:
        body = ast.parse(line).body
    except SyntaxError:
        return line
    if len(body) == 1 and isinstance(body[0], ast.Expr):
        value = body[0].value
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "print"):
            return f"print({line.strip()})"
    return line


def issue_example(statement: str) -> str | None:
    """The issue's example code: its Python code blocks in order (blocks tagged python/py/pycon, console sessions
    with the prompts removed, the printed results dropped and bare expressions printed, untagged blocks that look like
    code), joined. None if there is no import statement in it or it is longer than EXAMPLE_MAX_CHARS."""
    parts = []
    for m in _CODE_BLOCK.finditer(statement.replace("\r\n", "\n")):
        tag, code = (m.group(1) or "").lower(), m.group(2)
        lines = code.split("\n")
        if any(line.startswith(">>>") for line in lines):  # a console session: keep the input lines only
            code = "\n".join(_echo(line[4:]) if line.startswith(">>> ") and not line[4:5].isspace() else line[4:] if line.startswith("... ") else line[3:]
                             for line in lines if line.startswith((">>>", "...")))
        elif not tag and not re.search(r"^\s*(import|from)\s+\w|^\s*(def|class)\s+\w|^\s*print\(", code, re.M):
            continue  # an untagged block that looks like output or prose
        if code.strip():
            parts.append(code.strip("\n"))
    code = "\n".join(parts)
    if not code or len(code) > EXAMPLE_MAX_CHARS or not re.search(r"^\s*(import|from)\s+\w", code, re.M) or "S1ARENA_EOF" in code:
        return None
    return code


class ScriptedProposer(Proposer):
    """Candidates without a model (see the module docstring). On the initial repository it offers every edit (gold
    and distractors); once an edit is applied it offers a revert instead. Views, the test run and the example run
    are offered until they have been run on the current state of the repository."""

    name = "scripted"
    version = "1"
    cacheable = False

    def propose(self, context: Context) -> list[Action]:
        inst = context.instance
        edits = [Action("patch", inst["clean_patch"], title="Apply this patch", tag="gold")]
        edits += [Action("patch", patch, title="Apply this patch", tag=f"distractor:{system}") for system, patch in inst["distractors"]]
        views = code_views([edit.text for edit in edits])
        runs = []
        tests = existing_tests_command(inst)
        if tests:
            runs.append(Action("command", tests, title="Run the repository's existing tests for this part of the code", tag="test"))
        example = issue_example(inst["problem_statement"])
        if example:
            runs.append(Action("command", f"python - <<'S1ARENA_EOF'\n{example}\nS1ARENA_EOF", title="Run the example code from the issue", tag="example"))

        edited, seen = False, set()  # seen: commands run since the repository last changed
        for step in context.steps:
            if step.action.kind == "patch" and step.exit_code == 0:
                edited, seen = True, set()
            elif step.action.kind == "revert":
                edited, seen = False, set()
            elif step.action.kind == "command":
                seen.add(step.action.text)
        out = list(edits) if not edited else [Action("revert", title="Revert all changes: restore every file to its state at the start", tag="revert")]
        out += [a for a in views + runs if a.text not in seen]
        return out


# ---------------------------------------------------------------------------------------------------
# Chat-model proposer (OpenAI-compatible), for the open decision on the final proposer.

CHAT_SYSTEM = """You are the proposer inside a software engineering agent. The agent is resolving a GitHub issue in a Python repository checked out at /testbed, inside a container with the project's Python environment active and no network access. At every step you propose {k} different candidate next actions; a separate controller picks one of them to run, or submits the current changes as the final answer (you never submit).

Each action is one bash command (it may span several lines, e.g. with a heredoc), run in /testbed with a {timeout}-second limit; its output is shown back to you at the next step. Edit files with commands, for example `sed -i`, `python - <<'EOF' ... EOF`, `cat > path <<'EOF' ... EOF` or `git apply <<'EOF' ... EOF`. Do not modify existing tests. Useful kinds of actions: inspect code, reproduce the issue, edit the code, run tests, check the diff. Make the {k} candidates meaningfully different from each other.

Reply with a JSON object only: {{"candidates": [{{"title": "<one line: what the command does and why>", "command": "<bash command>"}}, ...]}}"""


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit // 3
    return f"{text[:head]}\n[... {len(text) - limit:,} characters not shown ...]\n{text[-(limit - head):]}"


def chat_prompt(context: Context, k: int, timeout: int, history_chars: int = 12_000, output_chars: int = 3_000) -> list[dict[str, str]]:
    """The proposer's messages: the issue, every executed action with its output (newest outputs first within a
    budget) and the current diff. Only the issue, repository and version of the instance are read."""
    inst = context.instance
    blocks, budget = [], history_chars
    for i in reversed(range(len(context.steps))):
        step = context.steps[i]
        text = step.action.text if step.action.kind != "revert" else "(revert all changes)"
        body = f"$ {_clip(text, output_chars)}\n{_clip(step.output, output_chars)}"
        if len(body) <= budget:
            budget -= len(body)
            blocks.append(f"Step {i + 1} (exit code {step.exit_code}{', timed out' if step.timed_out else ''}):\n{body}")
        else:
            budget = 0
            blocks.append(f"Step {i + 1}: $ {text.splitlines()[0][:200] if text else ''} (output not shown)")
    user = (f"Repository: {inst['repo']} (version {inst['version']})\n\nIssue:\n{_clip(inst['problem_statement'], 8000)}\n\n"
            f"Steps so far:\n{chr(10).join(reversed(blocks)) if blocks else 'none yet'}\n\n"
            f"Current changes (git diff against the start):\n{_clip(context.diff, 6000) if context.diff else 'none'}\n\n"
            f"Propose {k} candidate next actions.")
    return [{"role": "system", "content": CHAT_SYSTEM.format(k=k, timeout=timeout)}, {"role": "user", "content": user}]


def parse_candidates(text: str, k: int) -> list[Action]:
    decoder, found, start = json.JSONDecoder(), None, text.find("{")
    while start != -1:
        try:
            obj, end = decoder.raw_decode(text, start)
        except ValueError:
            start = text.find("{", start + 1)
            continue
        if isinstance(obj, dict) and isinstance(obj.get("candidates"), list):
            found = obj["candidates"]
        start = text.find("{", end)
    if found is None:  # no complete candidates list (e.g. a reply broken in its last candidate): keep the well-formed ones
        found, start = [], text.find("{")
        while start != -1:
            try:
                obj, end = decoder.raw_decode(text, start)
            except ValueError:
                start = text.find("{", start + 1)
                continue
            if isinstance(obj, dict) and "command" in obj:
                found.append(obj)
                start = text.find("{", end)
            else:
                start = text.find("{", start + 1)
    out, seen = [], set()
    for item in found or []:
        if not isinstance(item, dict) or not isinstance(item.get("command"), str) or not item["command"].strip():
            continue
        command = item["command"].strip("\n")
        if command in seen:
            continue
        seen.add(command)
        out.append(Action("command", command, title=str(item.get("title") or "").strip().splitlines()[0][:200] if item.get("title") else "", tag="chat"))
    return out[:k]


class ChatProposer(Proposer):
    """K candidates per call from a fixed chat model at temperature 0. base_url=None uses the project's AI Gateway
    (s1arena.gateway); otherwise any OpenAI-compatible server, with the key read from the environment variable
    api_key_env if given."""

    cacheable = True

    def __init__(self, model: str, k: int = 5, timeout: int = 300, base_url: str | None = None, api_key_env: str | None = None,
                 max_tokens: int = 4000, version: str = "1", transport=None) -> None:
        self.model, self.k, self.timeout, self.base_url, self.api_key_env = model, k, timeout, base_url, api_key_env
        self.max_tokens, self.version = max_tokens, version
        self.name = f"chat:{model}:k{k}"
        self.transport = transport  # tests: a function(body) -> response JSON
        self.last_usage: dict[str, Any] | None = None

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.transport is not None:
            return self.transport(body)
        if self.base_url is None:
            from ... import gateway

            return gateway.post("/v1/chat/completions", body)
        import httpx

        headers = {"Authorization": f"Bearer {os.environ[self.api_key_env]}"} if self.api_key_env else {}
        response = httpx.post(self.base_url.rstrip("/") + "/chat/completions", json=body, headers=headers, timeout=600)
        response.raise_for_status()
        return response.json()

    def propose(self, context: Context) -> list[Action]:
        body = {"model": self.model, "messages": chat_prompt(context, self.k, self.timeout), "temperature": 0, "max_tokens": self.max_tokens}
        out = self._post(body)
        self.last_usage = out.get("usage")
        return parse_candidates(out["choices"][0]["message"].get("content") or "", self.k)


class CopilotProposer(ChatProposer):
    """K candidates per call from a model served through GitHub Copilot (s1arena.models.copilot_llm, which checks that
    the requested model answered): ChatProposer's prompt and parsing, but Copilot sets the temperature (1) and the output
    cap itself, so proposals are samples, fixed by the cache once made."""

    def __init__(self, model: str, k: int = 5, timeout: int = 300, effort: str | None = None, version: str = "1") -> None:
        super().__init__(model, k=k, timeout=timeout, version=version)
        self.spec = model + (f"@effort={effort}" if effort else "")
        self.name = f"copilot:{self.spec}:k{k}"
        self._llm = None

    def propose(self, context: Context) -> list[Action]:
        from ...models.copilot_llm import CopilotLLM

        if self._llm is None:
            self._llm = CopilotLLM(self.spec)
            self._llm._check_model()
        system, user = (m["content"] for m in chat_prompt(context, self.k, self.timeout))
        text, self.last_usage, _ = self._llm._ask(system, user)
        return parse_candidates(text, self.k)


class BedrockProposer(ChatProposer):
    """K candidates per call from a model on Amazon Bedrock (s1arena.models.bedrock_llm): ChatProposer's prompt and
    parsing; these models take no temperature, so proposals are samples, fixed by the cache once made."""

    def __init__(self, model: str, k: int = 5, timeout: int = 300, effort: str | None = None, version: str = "1") -> None:
        super().__init__(model, k=k, timeout=timeout, version=version)
        self.spec = model + (f"@effort={effort}" if effort else "")
        self.name = f"bedrock:{self.spec}:k{k}"
        self._llm = None

    def propose(self, context: Context) -> list[Action]:
        from ...models.bedrock_llm import BedrockLLM

        if self._llm is None:
            self._llm = BedrockLLM(self.spec)
        system, user = (m["content"] for m in chat_prompt(context, self.k, self.timeout))
        text, usage, _ = self._llm.chat(system, user)
        self.last_usage = {**usage, "cost_usd": self._llm._cost(usage)}
        return parse_candidates(text, self.k)
