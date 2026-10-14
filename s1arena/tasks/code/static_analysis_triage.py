"""Leaf: static analysis warning triage (静态分析告警分诊): real issue or false positive?

D2A: warnings of the Infer static analyzer on C projects (OpenSSL, libav, nginx, httpd, libtiff),
labelled by differential analysis: an issue reported before a bug-fixing commit that disappears
after it is labelled real (1); issues that are never fixed are labelled false positives (0).
"""

from __future__ import annotations

import ast

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import hf_rows

LEAF = "code/static_analysis_triage"


@register_task
class D2ATrace(TaskAdapter):
    name = f"{LEAF}/d2a"
    leaf = LEAF
    source = "D2A leaderboard, Trace + Code task, dev split: Infer warnings on C projects labelled by differential analysis (Zheng et al., ICSE-SEIP 2021)"
    url = "https://github.com/IBM/D2A"
    license = "Apache-2.0 (as listed by the Hugging Face copy claudios/D2A; the IBM D2A repository is Apache-2.0)"
    tier = "B"
    criteria_source = "Adapted from the D2A leaderboard label definition (1: a real bug report; 0: a false bug report)"
    sampling = "balanced"
    notes = (
        "Balanced sampling: only 2.6% of dev warnings are labelled real, so accuracy on a proportional sample would reward "
        "always answering false. The test split's labels are hidden, so the dev split is used. Labels come from the D2A "
        "auto-labeler (differential analysis of before/after-fix versions plus commit-message heuristics) and are noisy. "
        "The bug URL (it names the commit) is withheld."
    )

    def load(self):
        rows = hf_rows("claudios/D2A", "code_trace", "dev")
        return [r for r in rows if str(r["label"]) in ("0", "1") and r["trace"].strip()]

    def label_of(self, record):
        return str(record["label"])

    def to_item(self, record, rng):
        try:
            functions = ast.literal_eval(record["functions"])
        except (ValueError, SyntaxError):
            functions = []
        others = [f for f in functions if f.strip() and f.strip() != record["bug_function"].strip()]
        state = {"analyzer_report": record["trace"].strip(), "function_with_reported_issue": record["bug_function"]}
        if others:
            state["other_functions_in_trace"] = "\n\n".join(others)
        question = Question(
            type="boolean",
            instructions="The Infer static analyzer reported the issue below, with its bug trace. Is this a real bug (a true "
            "positive) rather than a false alarm?",
        )
        gold = Gold(label="true" if str(record["label"]) == "1" else "false")
        kind = record["trace"].split("error:", 1)[-1].split("\n", 1)[0].strip()
        return self.item(state, {"real_bug": question}, {"real_bug": gold}, original_id=record["id"], bug_type=kind)


OWASP_COMMIT = "20cbf3d11123347e47ed89541e6942836def53f7"
OWASP_ZIP = f"https://codeload.github.com/OWASP-Benchmark/BenchmarkJava/zip/{OWASP_COMMIT}"
# Vulnerability categories as named on the OWASP Benchmark scorecards, with their CWE ids.
OWASP_CATEGORIES = {
    "cmdi": "Command Injection (CWE-78)",
    "crypto": "Weak Cryptographic Algorithm (CWE-327)",
    "hash": "Weak Hashing Algorithm (CWE-328)",
    "ldapi": "LDAP Injection (CWE-90)",
    "pathtraver": "Path Traversal (CWE-22)",
    "securecookie": "Insecure Cookie: a cookie is set without the Secure flag (CWE-614)",
    "sqli": "SQL Injection (CWE-89)",
    "trustbound": "Trust Boundary Violation (CWE-501)",
    "weakrand": "Weak Randomness (CWE-330)",
    "xpathi": "XPath Injection (CWE-643)",
    "xss": "Cross-Site Scripting (CWE-79)",
}


def strip_java_comments(code: str) -> str:
    """Remove // and /* */ comments outside string and char literals, then collapse blank-line runs."""
    out, i, n = [], 0, len(code)
    while i < n:
        c = code[i]
        if c in "\"'":
            j = i + 1
            while j < n and code[j] != c:
                j += 2 if code[j] == "\\" else 1
            out.append(code[i:j + 1])
            i = j + 1
        elif code.startswith("//", i):
            while i < n and code[i] != "\n":
                i += 1
        elif code.startswith("/*", i):
            end = code.find("*/", i + 2)
            i = n if end < 0 else end + 2
        else:
            out.append(c)
            i += 1
    lines = [line.rstrip() for line in "".join(out).split("\n")]
    text = "\n".join(lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip() + "\n"


def java_method(source: str, name: str) -> str | None:
    """The source of the first 'static ... name(' method in a class file (brace matching)."""
    import re

    match = re.search(r"\n(\s*public static [^\n;=]*\b" + re.escape(name) + r"\s*\()", source)
    if not match:
        return None
    start, depth, i = match.start(1), 0, source.index("{", match.start(1))
    while i < len(source):
        depth += {"{": 1, "}": -1}.get(source[i], 0)
        if depth == 0:
            return source[start:i + 1]
        i += 1
    return None


@register_task
class OWASPBenchmark(TaskAdapter):
    name = f"{LEAF}/owasp_benchmark"
    leaf = LEAF
    source = "OWASP Benchmark for Java v1.2: 2,740 servlets, each a true or a false positive for one vulnerability category (OWASP Foundation)"
    url = "https://github.com/OWASP-Benchmark/BenchmarkJava"
    license = "GPL-2.0"
    tier = "C"
    criteria_source = "Written; the finding names the Benchmark's category (scorecard wording and CWE); gold is the 'real vulnerability' column of expectedresults-1.2.csv"
    notes = (
        "Every test case is built to look like a finding of its category to a static analyzer; about half are real. "
        "Comments are stripped (several state which branch is 'safe') and the test class name is replaced by TestCase. "
        "Helper code the verdict can depend on is shown: SeparateClassRequest, the Thing factory classes with "
        "thing.properties, the Utils methods called, and the algorithm entries of benchmark.properties. Published in "
        "2016, so memorisation is possible. Pinned to commit 20cbf3d."
    )

    def load(self):
        import csv
        import re
        import zipfile

        from ..common import fetch

        root = f"BenchmarkJava-{OWASP_COMMIT}/"
        with zipfile.ZipFile(fetch(OWASP_ZIP, f"owasp_benchmark_java_{OWASP_COMMIT[:7]}.zip")) as zf:
            read = lambda path: zf.read(root + path).decode("utf-8")
            expected = [r for r in csv.reader(read("expectedresults-1.2.csv").splitlines()) if r and not r[0].startswith("#")]
            helpers = "src/main/java/org/owasp/benchmark/helpers/"
            utils = read(helpers + "Utils.java")
            separate = strip_java_comments(read(helpers + "SeparateClassRequest.java"))
            things = "\n".join(strip_java_comments(read(helpers + f"{n}.java")) for n in ("ThingInterface", "ThingFactory", "Thing1", "Thing2"))
            properties = [line for line in read("src/main/resources/benchmark.properties").splitlines() if re.match(r"(crypto|hash)Alg\d=", line)]
            records = []
            for test, category, real, cwe in (r[:4] for r in expected):
                code = read(f"src/main/java/org/owasp/benchmark/testcode/{test}.java")
                context, config = [], []
                if "SeparateClassRequest" in code:
                    context.append(separate)
                if "ThingFactory" in code:
                    context.append(things)
                    config.append("thing.properties: thing=Thing2")
                for method in sorted(set(re.findall(r"helpers\.Utils\.(\w+)\s*\(", code))):
                    body = java_method(utils, method)
                    if body:
                        context.append("// Utils." + method + "\n" + strip_java_comments(body))
                if "benchmarkprops" in code:
                    config.append("benchmark.properties: " + "; ".join(properties))
                records.append({"test": test, "category": category.strip(), "real": real.strip() == "true", "cwe": cwe.strip(),
                                "code": strip_java_comments(code).replace(test, "TestCase"),
                                "helpers": "\n\n".join(context), "configuration": "\n".join(config)})
        return [r for r in records if r["category"] in OWASP_CATEGORIES]

    def label_of(self, record):
        return f"{record['category']}|{record['real']}"

    def to_item(self, record, rng):
        state = {"finding": f"{OWASP_CATEGORIES[record['category']]} reported in class TestCase", "servlet_code": record["code"]}
        if record["helpers"]:
            state["helper_code"] = record["helpers"]
        if record["configuration"]:
            state["configuration"] = record["configuration"]
        question = Question(
            type="boolean",
            instructions="A static application security testing tool reported the finding below for this Java servlet. Is it "
            "a real vulnerability (a true positive) rather than a false positive?",
        )
        gold = Gold(label="true" if record["real"] else "false")
        return self.item(state, {"real_vulnerability": question}, {"real_vulnerability": gold}, original_id=record["test"], category=record["category"])
