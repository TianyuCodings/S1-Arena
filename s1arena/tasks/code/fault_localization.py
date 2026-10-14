"""Leaf: fault localization (缺陷定位) on SWE-bench issues.

Gold is read off the patch the maintainers merged for the issue:
- file level: the one source file the fix changed, among files that fixes of other issues in the
  same repository changed (half of them from the same directory tree, as hard negatives);
- function level: given that file, the one function or method whose body the fix changed.
Only fixes that touch exactly one file (and, for the function level, one function) are used.
"""

from __future__ import annotations

import ast
import re
import warnings
from collections import defaultdict

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import hf_rows

LEAF = "code/fault_localization"
LETTERS = "ABCDEFGHIJ"
MAX_FILE_CHARS = 60_000


def patched_files(patch: str) -> list[tuple[str, bool]]:
    """(path, existed before the fix) for every file in a unified diff."""
    out = []
    for block in re.split(r"^diff --git ", patch, flags=re.M)[1:]:
        old = re.search(r"^--- (?:\"?a/(.+?)\"?|/dev/null)\s*$", block, re.M)
        new = re.search(r"^\+\+\+ (?:\"?b/(.+?)\"?|/dev/null)\s*$", block, re.M)
        header = re.match(r"\"?a/(.+?)\"? \"?b/", block)
        path = (old and old.group(1)) or (new and new.group(1)) or (header and header.group(1))
        if path:
            out.append((path, not (old and old.group(1) is None)))
    return out


def swe_bench_test(repo: str = "princeton-nlp/SWE-bench", **kwargs) -> list[dict]:
    return hf_rows(repo, split="test", data_files={"test": "data/test-00000-of-00001.parquet"}, verification_mode="no_checks", **kwargs)


def common_depth(a: str, b: str) -> int:
    depth = 0
    for x, y in zip(a.split("/")[:-1], b.split("/")[:-1]):
        if x != y:
            break
        depth += 1
    return depth


class FileLocalization(TaskAdapter):
    """Which file did the merged fix change? Candidates are files changed by fixes of other issues in the same repository."""

    tier = "C"
    criteria_source = "Written; the correct option is the file changed by the merged fix, the others are files changed by fixes of other issues in the same repository"

    def rows(self) -> list[dict]:
        """Issues with repo, problem_statement, patch, instance_id and 'cohort' (issues whose files are the preferred pool)."""
        raise NotImplementedError

    def load(self):
        rows = self.rows()
        pool, by_cohort = defaultdict(set), defaultdict(set)
        for r in rows:
            for path, existed in patched_files(r["patch"]):
                if existed:
                    pool[r["repo"]].add(path)
                    by_cohort[(r["repo"], r["cohort"])].add(path)
        records = []
        for r in rows:
            files = patched_files(r["patch"])
            if len(files) != 1 or not files[0][1]:
                continue
            gold = files[0][0]
            near = sorted(by_cohort[(r["repo"], r["cohort"])] - {gold})
            if len(near) < 9:
                near = sorted(pool[r["repo"]] - {gold})
            if len(near) < 9:
                continue
            records.append({"instance_id": r["instance_id"], "repo": r["repo"], "issue": r["problem_statement"], "gold": gold, "pool": near})
        return records

    def label_of(self, record):
        return record["repo"]

    def to_item(self, record, rng):
        pool = list(record["pool"])
        rng.shuffle(pool)
        pool.sort(key=lambda p: -common_depth(p, record["gold"]))  # stable: random order within a depth
        distractors = pool[:5]
        rest = pool[5:]
        distractors += rng.sample(rest, min(4, len(rest)))
        paths = distractors + [record["gold"]]
        rng.shuffle(paths)
        options = {LETTERS[i]: p for i, p in enumerate(paths)}
        answer = LETTERS[paths.index(record["gold"])]
        question = Question(
            type="choice",
            instructions="Which of these files has to be edited to resolve the issue? Exactly one of them was changed by the fix that was merged.",
            criteria=options,
        )
        state = {"repository": record["repo"], "issue": record["issue"]}
        return self.item(state, {"file": question}, {"file": Gold(label=answer)}, instance_id=record["instance_id"], gold_path=record["gold"])


@register_task
class SWEBenchFile(FileLocalization):
    name = f"{LEAF}/swe_bench_file"
    leaf = LEAF
    source = "SWE-bench test set, file-level localization from the gold patch (Jimenez et al., 2024)"
    url = "https://huggingface.co/datasets/princeton-nlp/SWE-bench"
    license = "MIT (SWE-bench repository)"
    notes = (
        "Issues (2012-2023, 12 Python repositories) whose fix changes exactly one existing source file. Ten candidate "
        "paths: the gold file, up to five files from the closest directories, and random other files, drawn from the gold "
        "patches of other SWE-bench test issues of the same repository and version when possible. Only paths are shown, "
        "as in Agentless-style file localization."
    )

    def rows(self):
        return [dict(r, cohort=r["version"]) for r in swe_bench_test()]


@register_task
class SWEBenchLiveFile(FileLocalization):
    name = f"{LEAF}/swe_bench_live_file"
    leaf = LEAF
    source = "SWE-bench-Live, issues from 2024-2025, file-level localization from the gold patch (Zhang et al., 2025)"
    url = "https://huggingface.co/datasets/SWE-bench-Live/SWE-bench-Live"
    license = "MIT"
    notes = (
        "Continuously collected issues (created 2024-2025, 60 Python repositories after filtering), which limits "
        "contamination. Same construction as swe_bench_file: fixes that change exactly one existing file; ten candidate "
        "paths from the gold patches of other SWE-bench-Live issues of the same repository (closest directories first). "
        "A distractor may not exist at the issue's commit."
    )

    def rows(self):
        rows = hf_rows("SWE-bench-Live/SWE-bench-Live", None, "full", verification_mode="no_checks",
                       data_files={"full": ["data/full-00000-of-00002.parquet", "data/full-00001-of-00002.parquet"]})
        return [dict(r, cohort=str(r["created_at"])[:4]) for r in rows]


def oracle_file(text: str, path: str) -> str | None:
    """A file's content from the SWE-bench oracle prompt, without its line-number prefixes."""
    match = re.search(r"\[start of " + re.escape(path) + r"\]\n(.*?)\n\[end of " + re.escape(path) + r"\]", text, re.S)
    if not match:
        return None
    lines = []
    for number, line in enumerate(match.group(1).split("\n"), 1):
        m = re.match(r"^(\d+) ?(.*)$", line, re.S)
        if not m or int(m.group(1)) != number:
            return None
        lines.append(m.group(2))
    return "\n".join(lines)


def changed_old_lines(patch: str) -> list[int]:
    """Line numbers in the original file that the patch removes, or next to which it inserts."""
    out, old, last = [], None, 0
    for line in patch.split("\n"):
        m = re.match(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
        if m:
            old = int(m.group(1))
            last = old - 1
            continue
        if old is None or line.startswith(("---", "+++")):
            continue
        if line.startswith("-"):
            out.append(old)
            last, old = old, old + 1
        elif line.startswith("+"):
            out.append(max(last, 1))
        elif line.startswith(" ") or line == "":
            last, old = old, old + 1
    return out


def functions(source: str) -> list[tuple[str, int, int, str]]:
    """(qualified name, first line incl. decorators, last line, signature line) of every def."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tree = ast.parse(source)
    lines = source.split("\n")
    found = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                found.append((prefix + child.name, start, child.end_lineno, lines[child.lineno - 1].strip()))
                walk(child, prefix + child.name + ".")
            elif isinstance(child, ast.ClassDef):
                walk(child, prefix + child.name + ".")
            else:
                walk(child, prefix)

    walk(tree, "")
    return found


@register_task
class SWEBenchFunction(TaskAdapter):
    name = f"{LEAF}/swe_bench_function"
    leaf = LEAF
    source = "SWE-bench test set, function-level localization inside the fixed file, oracle file contents (Jimenez et al., 2024)"
    url = "https://huggingface.co/datasets/princeton-nlp/SWE-bench_oracle"
    license = "MIT (SWE-bench repository)"
    tier = "C"
    criteria_source = "Written; the correct option is the function or method containing every line the merged fix changed"
    notes = (
        "Issues whose fix changes one Python file and, within it, lines of a single function or method (innermost "
        "enclosing def). The file is shown as it was before the fix; the options are all functions and methods it "
        "defines (qualified names). Files over 60,000 characters are skipped."
    )

    def load(self):
        records = []
        for r in swe_bench_test("princeton-nlp/SWE-bench_oracle"):
            files = patched_files(r["patch"])
            if len(files) != 1 or not files[0][1] or not files[0][0].endswith(".py"):
                continue
            path = files[0][0]
            source = oracle_file(r["text"], path)
            if source is None or len(source) > MAX_FILE_CHARS:
                continue
            try:
                defs = functions(source)
            except SyntaxError:
                continue
            owners = set()
            for line in changed_old_lines(r["patch"]):
                enclosing = [d for d in defs if d[1] <= line <= d[2]]
                if not enclosing:
                    owners = set()
                    break
                owners.add(max(enclosing, key=lambda d: d[1])[0])
            names = {}
            for name, start, _, signature in defs:
                names.setdefault(name, f"{signature}  (line {start})")
            if len(owners) != 1 or not 2 <= len(names) <= 255:
                continue
            records.append({"instance_id": r["instance_id"], "repo": r["repo"], "issue": r["problem_statement"], "path": path,
                            "source": source, "options": names, "gold": owners.pop()})
        return records

    def label_of(self, record):
        return record["repo"]

    def to_item(self, record, rng):
        question = Question(
            type="choice",
            instructions="Which function or method in this file has to be edited to resolve the issue? The fix that was merged changed only this one.",
            criteria=shuffled_options(dict(record["options"]), rng),
        )
        state = {"repository": record["repo"], "issue": record["issue"], "file_path": record["path"], "file_content": record["source"]}
        return self.item(state, {"function": question}, {"function": Gold(label=record["gold"])}, instance_id=record["instance_id"])
