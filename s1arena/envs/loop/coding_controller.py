"""Coding controller: the model chooses among candidate actions a fixed proposer writes, inside real SWE-bench tasks.

In SWE-bench the actions are free text (shell commands, file edits), so a decision model cannot act alone. Here the
agent is split: a fixed proposer, part of the environment, writes candidate next actions from the trajectory so far,
and the model under test picks which one runs, or submits the current changes. This measures a System-1 model as the
selector and stopping controller of a coding agent.

Tasks: EPISODES instances of SWE-bench Verified (Jimenez et al., 2024; OpenAI's human-validated subset), at dataset
revision DATASET_REVISION, each played in its official evaluation image (swebench/sweb.eval.x86_64.<id>:latest).
Chosen as documented next to EPISODES: balanced across repositories, and each instance validated before use (the
reference patch resolves it, the repository without a patch does not, and every distractor patch fails).

Each decision is one choice question. The state is: fixed instructions, the repository and version, the issue text
verbatim (up to ISSUE_CHARS), the decisions used, every executed action with its output (outputs shortened to
OUTPUT_CHARS with a marker; older outputs dropped beyond HISTORY_CHARS, newest first), and the current changes as a git
diff against the start (up to DIFF_CHARS). The options are the proposer's candidates in a deterministic shuffled
order (numbered 1..n; the order is a function of the instance and the trajectory, so it says nothing about quality)
plus "submit". Command outputs are normalized so the same actions give the same text: ANSI codes and carriage-return
redraws removed, and run times (with pytest's slowest-durations section and SymPy's slow-test lines, whose content
depends on machine load), process ids, memory addresses, date-times, temporary-directory names and SymPy's random seed
masked.

Actions run in the episode's container (coding_controller_exec.py): commands through bash in /testbed with the
project's conda environment, no network and a COMMAND_TIMEOUT-second limit; patches through `git apply`; a revert
restores the initial files. A decision without a usable answer runs nothing (it still counts).

Outcome: the episode ends at "submit" or after max_decisions decisions; either way the change made since the start
is extracted from the container and graded by SWE-bench's own harness logic (swebench 5.0.2) in a fresh container:
resolved = every FAIL_TO_PASS test passes and every PASS_TO_PASS test still passes. Score 1/0, success = resolved.
The outcome also reports how the episode ended, the test counts, the patch's files and hash, the proposer's tags of
the actions taken (e.g. gold, distractor:<system>, view, test), and timings.

Proposer (coding_controller_proposers.py): pluggable. The registered environment uses ScriptedProposer, which needs
no model: it offers the reference patch and DISTRACTORS failing patches from public SWE-bench submissions as edits,
numbered views of the code those edits touch, a run of the repository's existing tests for the issue, the issue's
example code when it has one, and a revert once an edit is applied. Every edit is shown and applied as its code part:
file sections for tests, documentation, changelogs and new files are dropped and git's index lines removed, for the
reference patch and the submissions' patches alike, so they look alike. With this proposer the task is to recognize
the right fix among plausible ones (the submissions' own failed attempts) and to stop once it is in place.

Data: the dataset parquet (6.3 MB, checksum verified) and the distractor patches (from the public S3 bucket of the
SWE-bench experiments repository, checksum verified) are downloaded on first use into .cache/coding_controller/.
Needs Docker, the evaluation images of the chosen instances (about 2.5-7 GB each; the harness never pulls them) and
the swebench package (5.0.2).
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import threading
import time
import urllib.request
from typing import Any

from ...env import ROOT
from ...registry import register_env
from ...schema import DecisionRequest, Question
from ..base import Environment
from .coding_controller_exec import Sandbox, SnapshotError, grade
from .coding_controller_proposers import (Action, BedrockProposer, ChatProposer, Context, CopilotProposer, Proposer, ScriptedProposer, Step, patch_files, patch_sections, patch_size,
                                          propose, section_path, trajectory_key)

DATASET_REPO = "SWE-bench/SWE-bench_Verified"
DATASET_REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"  # 2026-08-16: adds image, eval_script and log_parser for swebench 5
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}/data/test-00000-of-00001.parquet"
DATASET_SHA256 = "030cfd7f2a704c4c0226e7f104c725a3b41230b1d3517f9c915ad7ea5be3fa25"
SUBMISSION_URL = "https://swe-bench-submissions.s3.amazonaws.com/verified/{system}/logs/{instance}/patch.diff"
CACHE = ROOT / ".cache" / "coding_controller"

MAX_DECISIONS = 30
COMMAND_TIMEOUT = 300  # seconds per command (the longest test run of the 40 instances took 132 s under load)
GRADE_TIMEOUT = 1800  # seconds for the grading test run (swebench's default)
ISSUE_CHARS = 8_000
OUTPUT_CHARS = 3_000  # per action and per output in the history
HISTORY_CHARS = 12_000  # actions and outputs shown, newest first
DIFF_CHARS = 6_000
OPTION_CHARS = 5_000  # per option (the scripted proposer's patches all fit)
ORDER_SEED = "s1arena/loop/coding_controller"
SUBMIT, NO_OP = "submit", "none"
DISTRACTORS = 3

# Public SWE-bench Verified submissions (2024-2025, with per-instance patches in the experiments bucket) whose failed
# patches serve as distractors, in order of preference: most instances resolved first (116 to 359 of 500), so the
# distractors are the near misses of strong agents where there are any.
SYSTEMS = (
    "20250807_openhands_gpt5", "20250522_sweagent_claude-4-sonnet-20250514", "20250405_amazon-q-developer-agent-20250405-dev",
    "20250716_openhands_kimi_k2", "20250225_sweagent_claude-3-7-sonnet", "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
    "20241202_agentless-1.5_claude-3.5-sonnet-20241022", "20250520_openhands_devstral_small",
    "20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022", "20250226_swerl_llama3_70b", "20241022_tools_claude-3-5-haiku",
    "20250511_sweagent_lm_32b", "20241028_agentless-1.5_gpt4o", "20240628_autocoderover-v20240620", "20250616_Skywork-SWE-32B",
    "20240620_sweagent_claude3.5sonnet", "20240612_MASAI_gpt4o", "20241016_epam-ai-run-gpt-4o", "20240615_appmap-navie_gpt4o",
    "20240728_sweagent_gpt4o",
)

# Seed -> (instance id, its 3 distractors as (submission, first 16 hex digits of the SHA-256 of the raw patch.diff)).
# Selection (see the notes): per repository a fixed shuffle of the usable instances, validated in that order until 4
# pass; rounds of one instance per repository, then a 5th django and sympy instance for the 2 slots requests could
# not fill (of its 5 usable instances, 2 lack a third failing distractor and 1 takes over 10 minutes to grade).
EPISODES: list[tuple[str, tuple[tuple[str, str], ...]]] = [
    ("astropy__astropy-14508", (
        ("20241029_OpenHands-CodeAct-2.1-sonnet-20241022", "a61710b5401dd5db"),
        ("20241202_agentless-1.5_claude-3.5-sonnet-20241022", "7bb759ce2456906d"),
        ("20250520_openhands_devstral_small", "73896fa87314f4be"))),
    ("django__django-16502", (
        ("20250807_openhands_gpt5", "2af1809bf77e27fe"),
        ("20250522_sweagent_claude-4-sonnet-20250514", "13f31e200f44ca1d"),
        ("20250405_amazon-q-developer-agent-20250405-dev", "4a38a013bae1b6e7"))),
    ("matplotlib__matplotlib-25332", (
        ("20250405_amazon-q-developer-agent-20250405-dev", "78c6624b9722fe0b"),
        ("20250716_openhands_kimi_k2", "ac0e4dcecbaf8efc"),
        ("20241202_agentless-1.5_claude-3.5-sonnet-20241022", "cee6e869d269f1c9"))),
    ("psf__requests-5414", (
        ("20241202_agentless-1.5_claude-3.5-sonnet-20241022", "b923563cb8c4dfd0"),
        ("20250520_openhands_devstral_small", "973bd613b0431134"),
        ("20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022", "b7ac5d7aee808768"))),
    ("pydata__xarray-2905", (
        ("20241029_OpenHands-CodeAct-2.1-sonnet-20241022", "df1f73296343ce3a"),
        ("20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022", "b206747cda2254a5"),
        ("20241028_agentless-1.5_gpt4o", "4c2b0ac036f375ad"))),
    ("pylint-dev__pylint-7080", (
        ("20250522_sweagent_claude-4-sonnet-20250514", "175c2787298039ed"),
        ("20250405_amazon-q-developer-agent-20250405-dev", "726189eb95d6c267"),
        ("20241029_OpenHands-CodeAct-2.1-sonnet-20241022", "04689bb7a379df79"))),
    ("pytest-dev__pytest-7432", (
        ("20241022_tools_claude-3-5-haiku", "f31428f163398be4"),
        ("20250511_sweagent_lm_32b", "bda85eafb3fc304e"),
        ("20240620_sweagent_claude3.5sonnet", "8c754d8662c1456a"))),
    ("scikit-learn__scikit-learn-9288", (
        ("20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022", "10400a40fb7338ce"),
        ("20250511_sweagent_lm_32b", "5bd0ee9136e415db"),
        ("20250616_Skywork-SWE-32B", "a999a6d25bb85e6f"))),
    ("sphinx-doc__sphinx-8035", (
        ("20241029_OpenHands-CodeAct-2.1-sonnet-20241022", "1ad1e4219c2372c1"),
        ("20241202_agentless-1.5_claude-3.5-sonnet-20241022", "327d499a5e92a24c"),
        ("20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022", "f56a29ddbdc8697c"))),
    ("sympy__sympy-15017", (
        ("20250807_openhands_gpt5", "6c614ccf7beaa967"),
        ("20241202_agentless-1.5_claude-3.5-sonnet-20241022", "836651264964110e"),
        ("20250520_openhands_devstral_small", "9129d1b82ea52493"))),
    ("astropy__astropy-13453", (
        ("20241029_OpenHands-CodeAct-2.1-sonnet-20241022", "a151fe322ba711df"),
        ("20250520_openhands_devstral_small", "56e64fb1c698b042"),
        ("20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022", "c14a17db139500c4"))),
    ("django__django-15561", (
        ("20250225_sweagent_claude-3-7-sonnet", "e9331a4d1f8afaa1"),
        ("20250226_swerl_llama3_70b", "ab03e604bee026c7"),
        ("20241016_epam-ai-run-gpt-4o", "5ddc2bc23bd6a03f"))),
    ("matplotlib__matplotlib-23299", (
        ("20250522_sweagent_claude-4-sonnet-20250514", "007a5544056422b3"),
        ("20250405_amazon-q-developer-agent-20250405-dev", "7a0444197030d2eb"),
        ("20250716_openhands_kimi_k2", "c822f342281d1b86"))),
    ("psf__requests-1921", (
        ("20241029_OpenHands-CodeAct-2.1-sonnet-20241022", "469bef19349852b0"),
        ("20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022", "7913acfbf31470a0"),
        ("20241022_tools_claude-3-5-haiku", "25a9bd4a9d0270a5"))),
    ("pydata__xarray-4695", (
        ("20250716_openhands_kimi_k2", "bcd0e22fc362d1e7"),
        ("20250225_sweagent_claude-3-7-sonnet", "489f88ac0f871025"),
        ("20241029_OpenHands-CodeAct-2.1-sonnet-20241022", "35abbb5fde5fb101"))),
    ("pylint-dev__pylint-7277", (
        ("20241202_agentless-1.5_claude-3.5-sonnet-20241022", "9e4927f577e3fe16"),
        ("20250520_openhands_devstral_small", "4d32fe7286e400bb"),
        ("20250616_Skywork-SWE-32B", "a794fa93ab50a11c"))),
    ("pytest-dev__pytest-5262", (
        ("20241029_OpenHands-CodeAct-2.1-sonnet-20241022", "38dbd7a1c7575b25"),
        ("20250226_swerl_llama3_70b", "ab64636a3f302322"),
        ("20241028_agentless-1.5_gpt4o", "3abe7e87dc7ed0c1"))),
    ("scikit-learn__scikit-learn-14087", (
        ("20250716_openhands_kimi_k2", "532bb55271c003cb"),
        ("20241202_agentless-1.5_claude-3.5-sonnet-20241022", "0b8c15814e4488b7"),
        ("20250520_openhands_devstral_small", "228510cb43456c57"))),
    ("sphinx-doc__sphinx-8269", (
        ("20250225_sweagent_claude-3-7-sonnet", "1d6248ce08928e7d"),
        ("20250520_openhands_devstral_small", "8b986a66dbbcd6e4"),
        ("20241016_epam-ai-run-gpt-4o", "e2d206d9492ac75b"))),
    ("sympy__sympy-15345", (
        ("20241202_agentless-1.5_claude-3.5-sonnet-20241022", "93947ae431af6227"),
        ("20250520_openhands_devstral_small", "1b5ed8f07694bdf9"),
        ("20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022", "228ce97e0133e38a"))),
    ("astropy__astropy-14995", (
        ("20241028_agentless-1.5_gpt4o", "a27d0c6de2185e04"),
        ("20250616_Skywork-SWE-32B", "6b87396deb79304b"),
        ("20241016_epam-ai-run-gpt-4o", "b74a3cd7d14eabec"))),
    ("django__django-15503", (
        ("20250807_openhands_gpt5", "2e1b3af2b7947127"),
        ("20250405_amazon-q-developer-agent-20250405-dev", "93130eee66277e36"),
        ("20250716_openhands_kimi_k2", "76420f5cd672a439"))),
    ("matplotlib__matplotlib-25479", (
        ("20250807_openhands_gpt5", "59bea08cfa3ffe1b"),
        ("20250522_sweagent_claude-4-sonnet-20250514", "34cbfe7561fe1842"),
        ("20250405_amazon-q-developer-agent-20250405-dev", "be95a6910366f346"))),
    ("pydata__xarray-3151", (
        ("20240620_sweagent_claude3.5sonnet", "a1061495ba2be3a7"),
        ("20241016_epam-ai-run-gpt-4o", "c0db2ffb7bc1cbdc"),
        ("20240728_sweagent_gpt4o", "68862f8b64b3fe8e"))),
    ("pylint-dev__pylint-4604", (
        ("20250807_openhands_gpt5", "1ded4d15a877fa24"),
        ("20250522_sweagent_claude-4-sonnet-20250514", "f8ee72b06dd8486f"),
        ("20250405_amazon-q-developer-agent-20250405-dev", "0aae81197454f1c8"))),
    ("pytest-dev__pytest-7205", (
        ("20250807_openhands_gpt5", "756f763accbfe084"),
        ("20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022", "d2d852e9d9afe8d6"),
        ("20250226_swerl_llama3_70b", "6653905850278579"))),
    ("scikit-learn__scikit-learn-14983", (
        ("20250716_openhands_kimi_k2", "dedd44ab092eea5f"),
        ("20250225_sweagent_claude-3-7-sonnet", "fb32be7cd3be8cee"),
        ("20241029_OpenHands-CodeAct-2.1-sonnet-20241022", "39984a40d94447b0"))),
    ("sphinx-doc__sphinx-9230", (
        ("20250405_amazon-q-developer-agent-20250405-dev", "bb9991fb7568c9d6"),
        ("20250225_sweagent_claude-3-7-sonnet", "fb38807c80b6f72c"),
        ("20241202_agentless-1.5_claude-3.5-sonnet-20241022", "104eb4b8a60ae5eb"))),
    ("sympy__sympy-24661", (
        ("20250226_swerl_llama3_70b", "3ec7b2d892c4739a"),
        ("20241028_agentless-1.5_gpt4o", "b971e60fd1f40ed8"),
        ("20241016_epam-ai-run-gpt-4o", "a9e9756fb430645a"))),
    ("astropy__astropy-14096", (
        ("20250616_Skywork-SWE-32B", "0b6a15d262cec715"),
        ("20240620_sweagent_claude3.5sonnet", "752ae88c51bf1f9e"),
        ("20241016_epam-ai-run-gpt-4o", "e0abdb749b9f1eeb"))),
    ("django__django-14315", (
        ("20250807_openhands_gpt5", "775bd1e0b8a94a05"),
        ("20250522_sweagent_claude-4-sonnet-20250514", "6e3f16a9b9c91634"),
        ("20250405_amazon-q-developer-agent-20250405-dev", "7906afd5681c1836"))),
    ("matplotlib__matplotlib-24177", (
        ("20250522_sweagent_claude-4-sonnet-20250514", "99773cd5ab50feac"),
        ("20250405_amazon-q-developer-agent-20250405-dev", "bd1ba47099cc3d22"),
        ("20250716_openhands_kimi_k2", "c2595600224d29a6"))),
    ("pydata__xarray-3095", (
        ("20241202_agentless-1.5_claude-3.5-sonnet-20241022", "ccd4717bb2552711"),
        ("20250226_swerl_llama3_70b", "a03b42f1f188f653"),
        ("20241022_tools_claude-3-5-haiku", "2b1b63fc78257e1c"))),
    ("pylint-dev__pylint-6386", (
        ("20250522_sweagent_claude-4-sonnet-20250514", "4d325b8dcb98235a"),
        ("20250405_amazon-q-developer-agent-20250405-dev", "17705d15b8039d92"),
        ("20250716_openhands_kimi_k2", "7c63ac92d1961a50"))),
    ("pytest-dev__pytest-10051", (
        ("20250405_amazon-q-developer-agent-20250405-dev", "cac07c0ce2899fb1"),
        ("20250716_openhands_kimi_k2", "e1ee946cac11564c"),
        ("20250225_sweagent_claude-3-7-sonnet", "5f708a0a695b152b"))),
    ("scikit-learn__scikit-learn-25102", (
        ("20250405_amazon-q-developer-agent-20250405-dev", "8ea6e252c8d6a649"),
        ("20250520_openhands_devstral_small", "9d798a47e096e4a0"),
        ("20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022", "bf14ca356542d947"))),
    ("sphinx-doc__sphinx-9658", (
        ("20250225_sweagent_claude-3-7-sonnet", "07f18639d673a424"),
        ("20241029_OpenHands-CodeAct-2.1-sonnet-20241022", "e9266c9c8def280f"),
        ("20241202_agentless-1.5_claude-3.5-sonnet-20241022", "59d04b004500336d"))),
    ("sympy__sympy-18698", (
        ("20250807_openhands_gpt5", "8d37dfb331b30fee"),
        ("20250522_sweagent_claude-4-sonnet-20250514", "b3388fecf89504b6"),
        ("20250405_amazon-q-developer-agent-20250405-dev", "b03b142ac190e039"))),
    ("django__django-17084", (
        ("20250405_amazon-q-developer-agent-20250405-dev", "296c40d9c0dfe3e0"),
        ("20250716_openhands_kimi_k2", "b2aeef016b5fc1ee"),
        ("20250225_sweagent_claude-3-7-sonnet", "26c656692d831b59"))),
    ("sympy__sympy-23413", (
        ("20250522_sweagent_claude-4-sonnet-20250514", "e432b48bd2679d84"),
        ("20250716_openhands_kimi_k2", "ebc890022ebe2178"),
        ("20241029_OpenHands-CodeAct-2.1-sonnet-20241022", "3e94f9a4ab7f2343"))),
]

INSTRUCTIONS = f"""You control a software engineering agent that is resolving a GitHub issue in a Python repository checked out at /testbed. At every step an assistant proposes candidate actions: shell commands run in the repository (with the project's Python environment active, no network access and a {COMMAND_TIMEOUT}-second limit), patches to apply, or a revert of all changes. You choose one of them to run, or submit the current changes as the final answer. After you submit, or after {MAX_DECISIONS} decisions, the changes are graded with the repository's tests for this issue, which you cannot see: the issue counts as resolved only if the tests that check the fix now pass and the other tests still pass. Long texts are shortened; a marker says how much was left out."""

_LOCK = threading.RLock()
_DATA: dict[str, Any] = {}


# ---------------------------------------------------------------------------------------------------
# Data.

def _fetch(url: str, sha256: str, path) -> bytes:
    """Download once into the cache, checking the checksum (sha256 may be a prefix of the hex digest)."""
    if path.exists():
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest().startswith(sha256):
            return data
    request = urllib.request.Request(url, headers={"User-Agent": "s1arena"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            break
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    if not hashlib.sha256(data).hexdigest().startswith(sha256):
        raise RuntimeError(f"checksum mismatch for {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.{threading.get_ident()}.part")  # concurrent downloads of one file
    tmp.write_bytes(data)
    tmp.replace(path)
    return data


def dataset() -> dict[str, dict[str, Any]]:
    """Every SWE-bench Verified row at the pinned revision, by instance id."""
    with _LOCK:
        if "rows" not in _DATA:
            import io

            import pyarrow.parquet as pq

            data = _fetch(DATASET_URL, DATASET_SHA256, CACHE / f"swe_bench_verified_{DATASET_REVISION[:12]}.parquet")
            _DATA["rows"] = {row["instance_id"]: row for row in pq.read_table(io.BytesIO(data)).to_pylist()}
        return _DATA["rows"]


def submission_patch(system: str, instance_id: str, sha256: str | None = None) -> str:
    """A submission's raw patch for an instance (checked against sha256 when given)."""
    path = CACHE / "submissions" / system / f"{instance_id}.diff"
    url = SUBMISSION_URL.format(system=system, instance=instance_id)
    if sha256 is None:  # selection time: take what the bucket serves
        if not path.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "s1arena"})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return path.read_bytes().decode("utf-8", errors="replace")
    return _fetch(url, sha256, path).decode("utf-8", errors="replace")


# Every test_patch path of SWE-bench Verified matches _TEST_PART and no gold patch path does; no gold patch touches a
# documentation or changelog file (_DOC_PART).
_TEST_PART = re.compile(r"(^|/)(tests?|testing|e2e)(/|$)|(^|/)(test_[^/]*|[^/]*_tests?\.py|tests\.py|conftest\.py)$")
_DOC_PART = re.compile(r"^(docs?|changelog)/|(^|/)whats_?new/|(^|/)(changes|changelog|history|news|authors|contributors)(\.(rst|md|txt))?$|\.(rst|md)$", re.I)


def is_code_section(section: str) -> bool:
    """A file section that edits an existing file of the code: not a new, deleted, renamed or binary file, not a test
    file, not documentation or a changelog."""
    head = section.split("\n@@", 1)[0]
    if re.search(r"^(new file mode|deleted file mode|rename from|copy from|GIT binary patch|Binary files )", head, re.M):
        return False
    if "\n--- /dev/null" in head or "\n+++ /dev/null" in head or "\n@@" not in section:
        return False
    path = section_path(section)
    return bool(path) and not _TEST_PART.search(path) and not _DOC_PART.search(path)


def clean_patch(patch: str) -> str:
    """The code part of a patch: its code sections (is_code_section), without git's index lines."""
    kept = []
    for section in patch_sections(patch):
        if is_code_section(section):
            kept.append(re.sub(r"^index [0-9a-f]+\.\.[0-9a-f]+( \d+)?\n", "", section, count=1, flags=re.M))
    text = "".join(kept)
    return text if not text or text.endswith("\n") else text + "\n"


def episode(seed: int) -> dict[str, Any]:
    """The instance of a seed: its dataset row plus the cleaned reference patch and the cleaned distractors."""
    if not 0 <= seed < len(EPISODES):
        raise ValueError(f"seed {seed} out of range: {len(EPISODES)} episodes")
    instance_id, distractors = EPISODES[seed]
    row = dict(dataset()[instance_id])
    row["clean_patch"] = clean_patch(row["patch"])
    row["distractors"] = [(system, clean_patch(submission_patch(system, instance_id, sha))) for system, sha in distractors]
    return row


# ---------------------------------------------------------------------------------------------------
# Text.

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-Z\\-_]")
MASKS = (
    # which tests make pytest's --durations list (sphinx's tox config asks for it) or SymPy's slow-test lines depends on load
    (re.compile(r"(=+ slowest (?:\d+ )?durations =+)\n.*?(?=\n=+ |\Z)", re.S), r"\1\n(not shown: the durations vary from run to run)"),
    (re.compile(r"^.* - Took \d+(?:\.\d+)? seconds\n?", re.M), ""),
    (re.compile(r"\bpid=\d+"), "pid=<pid>"),
    (re.compile(r"\b(in|took|after|elapsed:?|time:?)\s+\d+(?:\.\d+)?\s*(?:s|sec|secs|seconds|ms)\b(?:\s*\(\d+:\d\d:\d\d\))?", re.I), r"\1 <time>"),
    (re.compile(r"\d+(?:\.\d+)?=setup\[\d+(?:\.\d+)?\]\+cmd\[\d+(?:\.\d+)?\] seconds"), "<time> seconds"),  # tox
    (re.compile(r"\b\d+\.\d+\s*(seconds|secs|sec|ms|s)\b"), r"<time> \1"),
    (re.compile(r"0x[0-9a-fA-F]{6,}"), "0x<address>"),
    (re.compile(r"\b\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d(?:[.,]\d+)?"), "<date-time>"),
    (re.compile(r"(random seed:\s*)\d+"), r"\1<seed>"),
    (re.compile(r"/tmp/tmp[\w-]{6,}"), "/tmp/tmp<random>"),
)


def normalize_output(text: str) -> str:
    text = ANSI.sub("", text or "")
    lines = []
    for line in text.split("\n"):
        if "\r" in line:
            parts = [p for p in line.split("\r") if p]
            line = parts[-1] if parts else ""
        lines.append(line.rstrip())
    text = "\n".join(lines).strip("\n")
    for pattern, replacement in MASKS:
        text = pattern.sub(replacement, text)
    return text


def clip(text: str, limit: int) -> str:
    """Keep the first third and the last two thirds of the budget, cut at line ends, with a marker in between."""
    if len(text) <= limit:
        return text
    head_end = text.rfind("\n", 0, limit // 3)
    head_end = head_end if head_end > limit // 6 else limit // 3
    tail_start = text.find("\n", len(text) - (limit - head_end))
    tail_start = tail_start + 1 if 0 <= tail_start < len(text) - (limit - head_end) // 2 else len(text) - (limit - head_end)
    omitted = tail_start - head_end
    return f"{text[:head_end]}\n[... {omitted:,} characters not shown ...]\n{text[tail_start:]}"


def describe(action: Action) -> str:
    """How an action is shown as an option."""
    if action.kind == "revert":
        return action.title or "Revert all changes: restore every file to its state at the start"
    if action.kind == "patch":
        files = ", ".join(patch_files(action.text)) or "no files"
        added, removed = patch_size(action.text)
        return clip(f"{action.title or 'Apply this patch'} (edits {files}; +{added} -{removed} lines):\n{action.text.rstrip()}", OPTION_CHARS)
    title = action.title.rstrip(".") + ":\n" if action.title else ""
    return clip(f"{title}$ {action.text.rstrip()}", OPTION_CHARS)


# ---------------------------------------------------------------------------------------------------
# Environment.

class CodingController(Environment):
    leaf = "loop/coding_controller"
    url = "https://www.swebench.com/"
    license = ("MIT (SWE-bench, SWE-bench Verified); the repositories under their own licences (BSD, MIT, Apache-2.0...); "
               "distractor patches from the SWE-bench experiments S3 bucket (no licence stated, fetched at run time, not redistributed)")
    tier = "B"
    mode = "paused"
    max_decisions = MAX_DECISIONS
    score_name = "resolved (1) or not (0)"
    done_question = False  # True: a separate yes/no "resolved?" question decides submission (see the notes)

    def make_proposer(self) -> Proposer:
        raise NotImplementedError

    @classmethod
    def instance_ids(cls) -> list[str]:
        return [instance_id for instance_id, _ in EPISODES]

    @classmethod
    def image(cls, seed: int) -> str:
        """The Docker image a seed is played (and graded) in; used by the Docker sweep (s1arena/docker_sweep.py)."""
        return dataset()[EPISODES[seed][0]]["image"]

    # --- episode ---

    def reset(self, seed: int) -> None:
        self.close()
        started = time.perf_counter()
        self.seed = seed
        self.inst = episode(seed)
        self.proposer = self.make_proposer()
        self.sandbox = Sandbox(self.inst["image"], self.inst["instance_id"])
        try:  # the runner calls reset() outside its try/finally, so a failed start must not leave the container behind
            self.sandbox.start()
            self.base = self.sandbox.snapshot()
        except BaseException:
            self.close()
            raise
        self.steps: list[Step] = []  # executed actions: the trajectory the proposer sees
        self.log: list[Step | None] = []  # every decision; None = nothing was run
        self.diff = ""
        self.options: dict[str, Action] | None = None
        self.decisions = 0
        self.submitted = False
        self.digest = hashlib.sha256()
        self.request_hash = ""
        self.timing = {"setup_s": time.perf_counter() - started, "propose_s": 0.0, "act_s": 0.0, "diff_s": 0.0}
        self._outcome: dict[str, Any] | None = None

    def _candidates(self) -> dict[str, Action]:
        if self.options is None:
            started = time.perf_counter()
            context = Context(self.inst, list(self.steps), self.diff)
            actions = propose(self.proposer, context)
            if not actions:  # a proposer that returned nothing: the harness's own neutral fallback
                actions = [Action("command", "git status && git diff --stat", title="Show the current changes", tag="fallback")]
            unique = list({tuple(a.key()): a for a in actions}.values())
            random.Random(f"{ORDER_SEED}/{self.inst['instance_id']}/{trajectory_key(self.proposer, context)}").shuffle(unique)
            self.options = {str(i + 1): a for i, a in enumerate(unique)}
            self.timing["propose_s"] += time.perf_counter() - started
        return self.options

    def _history(self) -> str:
        if not self.log:
            return "none yet"
        blocks, budget = [], HISTORY_CHARS
        for i in reversed(range(len(self.log))):
            step = self.log[i]
            if step is None:
                blocks.append(f"Step {i + 1}: no usable answer; nothing was run.")
                continue
            action = step.action
            if action.kind == "revert":
                header, shown = f"Step {i + 1}: reverted all changes.", ""
            elif action.kind == "patch":
                files = ", ".join(patch_files(action.text))
                header = (f"Step {i + 1}: applied a patch to {files}." if step.exit_code == 0 else
                          f"Step {i + 1}: tried to apply a patch to {files}; git apply failed (exit code {step.exit_code}) and changed nothing.")
                shown = clip(action.text.rstrip(), OUTPUT_CHARS)
            else:
                header = f"Step {i + 1}: ran a command (exit code {step.exit_code}{'; stopped at the time limit' if step.timed_out else ''})."
                shown = "$ " + clip(action.text.rstrip(), OUTPUT_CHARS)
            output = clip(step.output, OUTPUT_CHARS) if step.output else "(no output)"
            body = (shown + "\n" if shown else "") + "Output:\n" + output
            if len(body) <= budget:
                budget -= len(body)
                blocks.append(header + "\n" + body)
            else:
                budget = 0
                blocks.append(header + f" (details not shown: only the newest {HISTORY_CHARS:,} characters of actions and outputs are kept)")
        return "\n\n".join(reversed(blocks))

    def observe(self) -> DecisionRequest:
        options = self._candidates()
        inst = self.inst
        state = {
            "instructions": INSTRUCTIONS,
            "repository": f"{inst['repo']} (version {inst['version']}), checked out at /testbed",
            "issue": clip(inst["problem_statement"].replace("\r\n", "\n").strip(), ISSUE_CHARS),
            "decisions used": f"{self.decisions} of {self.max_decisions}",
            "steps so far": self._history(),
            "current changes (git diff against the start)": clip(self.diff.rstrip(), DIFF_CHARS) if self.diff else "none: the repository is unchanged",
        }
        criteria = {oid: describe(action) for oid, action in options.items()}
        if self.done_question:
            questions = {
                "done": Question(type="boolean", instructions="Do the current changes resolve the issue, so that they should be submitted now as the final answer? (true ends the episode and runs the hidden tests)"),
                "action": Question(type="choice", instructions="If you do not submit: which proposed action do you run next?", criteria=criteria if len(criteria) > 1 else {**criteria, NO_OP: "Run nothing this step"}),
            }
        else:
            criteria[SUBMIT] = "Submit the current changes (shown above) as the final answer: the episode ends and the hidden tests are run"
            questions = {"action": Question(type="choice", instructions="Which action do you take next? Options 1 to "
                                            f"{len(options)} were proposed by the assistant and run in the repository; '{SUBMIT}' ends the task.", criteria=criteria)}
        request = DecisionRequest(state=state, questions=questions)
        self.request_hash = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        return request

    def default_choice(self, request: DecisionRequest) -> dict[str, str]:
        """No usable answer: run nothing (and do not submit)."""
        return {name: ("false" if question.type == "boolean" else NO_OP) for name, question in request.questions.items()}

    def act(self, choice: dict[str, str]) -> None:
        self.decisions += 1
        self.digest.update(self.request_hash.encode())
        self.digest.update(repr(sorted(choice.items())).encode())
        if (choice.get("done") == "true") if self.done_question else (choice.get("action") == SUBMIT):
            self.submitted = True
            return
        action = (self.options or {}).get(choice.get("action", NO_OP))
        if action is None:
            self.log.append(None)
            return
        started = time.perf_counter()
        if action.kind == "patch":
            result = self.sandbox.apply(action.text)
        elif action.kind == "revert":
            result = self.sandbox.revert(self.base)
        else:
            result = self.sandbox.run(action.text, COMMAND_TIMEOUT)
        self.timing["act_s"] += time.perf_counter() - started
        output = normalize_output(result.output)
        if result.timed_out:
            output += f"\n[the command was stopped after {COMMAND_TIMEOUT} seconds]"
        step = Step(action, output, result.exit_code, round(result.seconds, 2), result.timed_out)
        self.steps.append(step)
        self.log.append(step)
        started = time.perf_counter()
        try:
            self.diff = self.sandbox.diff(self.base)
        except SnapshotError as error:
            self.diff = f"(the changes cannot be read: {error})"
        self.timing["diff_s"] += time.perf_counter() - started
        self.options = None

    def done(self) -> bool:
        return self.submitted

    # --- outcome ---

    def outcome(self) -> dict[str, Any]:
        if self._outcome is None:
            self._outcome = self._grade()
        return self._outcome

    def _grade(self) -> dict[str, Any]:
        inst = self.inst
        tags = [(step.action.tag or step.action.kind) if step else "none" for step in self.log]
        info: dict[str, Any] = {
            "instance_id": inst["instance_id"], "repo": inst["repo"], "termination": "submit" if self.submitted else "decision_limit",
            "actions": tags, "commands": sum(1 for s in self.steps if s.action.kind == "command"),
            "edits": sum(1 for s in self.steps if s.action.kind == "patch"), "reverts": sum(1 for s in self.steps if s.action.kind == "revert"),
            "no_ops": sum(1 for s in self.log if s is None), "request_digest": self.digest.hexdigest()[:16], "proposer": self.proposer.ident(),
        }
        try:
            patch = self.sandbox.diff(self.base)
        except SnapshotError as error:  # the actions destroyed the repository: nothing gradable was produced
            return {"success": False, "score": 0.0, "grade_status": "no_patch", "patch_error": str(error)[:300], **info, "timing": self._timing()}
        info.update({"patch_files": patch_files(patch), "patch_chars": len(patch), "patch_sha256": hashlib.sha256(patch.encode()).hexdigest()[:16]})
        result = grade(inst, patch, timeout=GRADE_TIMEOUT)
        self.timing["grade_s"] = result.get("grade_s", 0.0)
        out = {"success": result["resolved"], "score": 1.0 if result["resolved"] else 0.0, "grade_status": result["status"], **info}
        for key in ("fail_to_pass", "pass_to_pass", "apply_command"):
            if key in result:
                out[key] = result[key]
        out["timing"] = self._timing()
        return out

    def _timing(self) -> dict[str, float]:
        return {k: round(v, 2) for k, v in self.timing.items()}

    def close(self) -> None:
        sandbox = getattr(self, "sandbox", None)
        if sandbox is not None:
            sandbox.stop()
            self.sandbox = None

    def card(self) -> dict[str, Any]:
        return {**super().card(), "dataset": f"{DATASET_REPO}@{DATASET_REVISION[:12]}", "instances": self.instance_ids(),
                "command_timeout_s": COMMAND_TIMEOUT, "grade_timeout_s": GRADE_TIMEOUT}


@register_env
class SWEVerifiedScripted(CodingController):
    name = "loop/coding_controller/swebench_verified_scripted"
    source = ("SWE-bench Verified (Jimenez et al., 2024; OpenAI, 2024), official evaluation images and swebench 5.0.2 grading; "
              "candidate actions from a scripted proposer: the reference patch and 3 failed patches of public SWE-bench submissions")
    criteria_source = ("Issue text verbatim from SWE-bench Verified; candidate actions from the scripted proposer (reference patch and "
                       "submissions' patches reduced to their code part, views of the code they touch, the instance's own test command "
                       "on the existing test files, the issue's example code); instructions newly written; resolution by SWE-bench's own tests")
    notes = ("40 SWE-bench Verified instances from 10 repositories: 4 each from astropy, matplotlib, xarray, pylint, pytest, "
             "scikit-learn and sphinx, 5 each from django and sympy, 2 from requests (it has too few usable instances, as have seaborn "
             "and flask), interleaved by repository. Usable: the evaluation image is present, the reference patch is code only and "
             "fits in one option (5,000 characters), the instance's test command covers existing test files, and at least 3 failed "
             "patches of 20 public submissions remain after reduction to code and de-duplication. Validated in the image: the reference "
             "patch applies with git apply and resolves the instance (graded in under 10 minutes), the repository without a patch does "
             "not, and the 3 distractors (the first ones that fail, submissions ordered by instances resolved) apply and fail. Each "
             "decision is one choice among the proposer's candidates (shuffled) plus submit; the shortest resolving play is 2 "
             "decisions (apply the right patch, submit). Ends at submit or after 30 decisions; the change since the start is then "
             "graded as swebench 5.0.2 grades it, in a fresh container. Score 1 if resolved (every FAIL_TO_PASS test passes, every "
             "PASS_TO_PASS test still passes), success = resolved.")
    episodes = len(EPISODES)

    def make_proposer(self) -> Proposer:
        return ScriptedProposer()


@register_env
class SWEVerifiedGPT41Mini(CodingController):
    name = "loop/coding_controller/swebench_verified_gpt41mini"
    source = ("SWE-bench Verified (Jimenez et al., 2024; OpenAI, 2024), official evaluation images and swebench 5.0.2 grading; "
              "candidate actions written by GPT-4.1 mini (temperature 0, 5 per state) through the AI Gateway")
    criteria_source = ("Issue text verbatim from SWE-bench Verified; candidate actions written by a fixed proposer model; instructions "
                       "newly written; resolution by SWE-bench's own tests")
    notes = ("The same 40 instances, decision cap and grading as swebench_verified_scripted, with a model as the proposer: GPT-4.1 mini "
             "at temperature 0 writes 5 candidate next actions (shell commands, patches, or a revert) for every new trajectory state, "
             "and the decision is one choice among them plus submit. Proposals are cached by a hash of the proposer, the instance and "
             "the actions executed so far, so a trajectory seen before is replayed without model calls; a new trajectory state needs "
             "the proposer model. The reference patch is never among the candidates, so the model under test has to steer the proposer "
             "towards a fix by the actions it picks.")
    episodes = len(EPISODES)

    def make_proposer(self) -> Proposer:
        return ChatProposer("openai/gpt-4.1-mini", k=5)


@register_env
class SWEVerifiedGPT56Luna(CodingController):
    name = "loop/coding_controller/swebench_verified_gpt56luna"
    source = ("SWE-bench Verified (Jimenez et al., 2024; OpenAI, 2024), official evaluation images and swebench 5.0.2 grading; "
              "candidate actions written by GPT-5.6 Luna (reasoning effort none, 5 per state) through Amazon Bedrock")
    criteria_source = SWEVerifiedGPT41Mini.criteria_source
    notes = ("swebench_verified_gpt41mini with GPT-5.6 Luna as the proposer, called through Amazon Bedrock (bedrock:us.openai."
             "gpt-5.6-luna@effort=none) instead of GPT-4.1 mini through the AI Gateway: the same 40 instances, prompt, decision cap and "
             "grading. The model takes no temperature, so a state's 5 candidates are a sample, fixed by the proposal cache once drawn "
             "(cached by a hash of the proposer, the instance and the actions executed so far, so a trajectory seen before is "
             "replayed without model calls). The reference patch is never among the candidates, so the model under test has to "
             "steer the proposer towards a fix by the actions it picks.")
    episodes = len(EPISODES)

    def make_proposer(self) -> Proposer:
        return BedrockProposer("us.openai.gpt-5.6-luna", k=5, effort="none")
