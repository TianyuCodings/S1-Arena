"""Leaf: test outcome prediction (测试能否通过): given code and a test, will the test pass?

No code is executed here. Outcomes come from the sources' recorded execution: HumanEvalFix
pairs a correct solution with a buggy one that fails the same test suite, and LiveCodeBench
stores, for every generated program that failed, the test it failed on. Both sources are
paired so that the same test (or the same task) appears once with a pass and once with a fail.
"""

from __future__ import annotations

import json

from ...env import load_env
from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import hf_rows

LEAF = "code/test_outcome"

LANGUAGES = {"python": "Python", "js": "JavaScript", "java": "Java", "go": "Go", "cpp": "C++", "rust": "Rust"}


@register_task
class HumanEvalFixOutcome(TaskAdapter):
    name = f"{LEAF}/humanevalfix"
    leaf = LEAF
    source = "HumanEvalFix from HumanEvalPack, correct and buggy solutions with their unit tests, six languages (Muennighoff et al., 2023)"
    url = "https://huggingface.co/datasets/bigcode/humanevalpack"
    license = "MIT"
    tier = "C"
    criteria_source = "Written; the canonical solution passes the test and the buggy solution fails it, as constructed and checked by the HumanEvalFix authors"
    items_per_record = 2
    notes = (
        "Each record is one task in one language, asked twice with the same test: once with the canonical solution "
        "(passes) and once with the manually inserted bug (fails). Bug type and failure symptoms are withheld."
    )

    def load(self):
        rows = []
        for config, language in LANGUAGES.items():
            for r in hf_rows("bigcode/humanevalpack", config, "test"):
                if r["canonical_solution"].strip() != r["buggy_solution"].strip():
                    rows.append(dict(r, language=language))
        return rows

    def label_of(self, record):
        return record["language"]

    def to_items(self, record, rng):
        test = ((record.get("test_setup") or "") + "\n" + record["test"]).strip()
        question = Question(
            type="boolean",
            instructions="Will this test pass when it is run against the code? It passes only if every assertion holds "
            "and the run finishes without an error or a timeout.",
        )
        items = []
        for version, body in (("canonical", record["canonical_solution"]), ("buggy", record["buggy_solution"])):
            state = {"language": record["language"], "code": record["prompt"] + body, "test": test}
            gold = Gold(label="true" if version == "canonical" else "false")
            items.append(self.item(state, {"passes": question}, {"passes": gold}, version=version, original_id=record["task_id"], bug_type=record["bug_type"]))
        return items


LCB_FILE = "Gemini-Pro-1.5 (May)/Scenario.codegeneration_10_0.2_eval_all.json"


@register_task
class LiveCodeBenchOutcome(TaskAdapter):
    name = f"{LEAF}/livecodebench"
    leaf = LEAF
    source = "LiveCodeBench code generation, recorded per-test results of Gemini-Pro-1.5 programs (Jain et al., 2024)"
    url = "https://huggingface.co/datasets/livecodebench/submissions"
    license = "MIT"
    tier = "C"
    criteria_source = "Written; outcomes are LiveCodeBench's recorded test results (wrong answer or runtime error on the shown test; the other program passed every test)"
    items_per_record = 2
    notes = (
        "Problems from LeetCode, AtCoder and Codeforces (May 2023 to 2024). Each record is a test on which one sampled "
        "program failed (wrong answer or runtime error) and another program for the same problem passed all tests; the "
        "test is asked with both programs. Time-limit failures and tests whose logged input was truncated are excluded."
    )

    def load(self):
        load_env()
        from huggingface_hub import hf_hub_download

        with open(hf_hub_download("livecodebench/submissions", LCB_FILE, repo_type="dataset"), encoding="utf-8") as f:
            problems = json.load(f)
        records = []
        for p in problems:
            passing = list(dict.fromkeys(code for code, ok in zip(p["code_list"], p["graded_list"]) if ok and code.strip()))
            if not passing:
                continue
            seen = set()
            for code, ok, meta in zip(p["code_list"], p["graded_list"], p["metadata"]):
                meta = json.loads(meta) if isinstance(meta, str) else meta
                if ok or meta.get("error_code") not in (-2, -4) or not code.strip():
                    continue
                test_input, expected = str(meta.get("inputs", "")), str(meta.get("expected", ""))
                if "(truncated)" in test_input + expected or (code, test_input) in seen:
                    continue
                seen.add((code, test_input))
                records.append({
                    "question_id": p["question_id"], "title": p["question_title"], "platform": p["platform"],
                    "problem": p["question_content"], "starter_code": p.get("starter_code") or "",
                    "input": test_input, "expected": expected, "failing": code, "passing": passing[0],
                    "error": meta.get("error_message"),
                })
        return records

    def label_of(self, record):
        return record["platform"]

    def to_items(self, record, rng):
        functional = bool(record["starter_code"].strip())  # LeetCode-style: the input lines are the call arguments
        if functional:
            text = ("Will this program pass the test? The test calls the solution method with the given arguments (one per "
                    "line, as JSON) and passes if the method returns the expected value without raising an error.")
        else:
            text = ("Will this program pass the test? The test feeds the input to standard input and passes if the program "
                    "prints the expected output (ignoring surrounding whitespace) without an error.")
        question = Question(type="boolean", instructions=text)
        items = []
        for version in ("passing", "failing"):
            state = {"problem": record["problem"], "program": record[version], "test_input": record["input"], "expected_output": record["expected"]}
            gold = Gold(label="true" if version == "passing" else "false")
            items.append(self.item(state, {"passes": question}, {"passes": gold}, version=version, original_id=record["question_id"], platform=record["platform"]))
        return items
