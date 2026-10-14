"""Leaf: commit / PR classification (提交 / PR 分类): commit type, and whether a change needs review.

- Conventional commit type of a commit from its message (type prefix removed) and diff; gold is
  the manual annotation of the CCS dataset, with the category definitions of its authors.
- CodeReviewer quality estimation: whether a diff hunk drew a reviewer's comment in real code
  review on GitHub (the recorded outcome), i.e. whether it needed a human reviewer's attention.
"""

from __future__ import annotations

import json
import re

from ...env import load_env
from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import CACHE, hf_rows

LEAF = "code/commit_classification"
MAX_DIFF_CHARS = 60_000

# Category definitions verbatim from the CCS classification prompt (Zeng et al., 2025).
CCS_TYPES = {
    "feat": "Code changes aim to introduce new features to the codebase, encompassing both internal and user-oriented features.",
    "fix": "Code changes aim to fix bugs and faults within the codebase.",
    "perf": "Code changes aim to improve performance, such as enhancing execution speed or reducing memory consumption.",
    "style": "Code changes aim to improve readability without affecting the meaning of the code. This type encompasses aspects "
    "like variable naming, indentation, and addressing linting or code analysis warnings.",
    "refactor": "Code changes aim to restructure the program without changing its behavior, aiming to improve maintainability. "
    "To avoid confusion and overlap, we propose the constraint that this category does not include changes classified as "
    "``perf'' or ``style''. Examples include enhancing modularity, refining exception handling, improving scalability, "
    "conducting code cleanup, and removing deprecated code.",
    "docs": "Code changes that modify documentation or text, such as correcting typos, modifying comments, or updating documentation.",
    "test": "Code changes that modify test files, including the addition or updating of tests.",
    "ci": "Code changes to CI (Continuous Integration) configuration files and scripts, such as configuring or updating CI/CD "
    "scripts, e.g., ``.travis.yml'' and ``.github/workflows''.",
    "build": "Code changes affecting the build system (e.g., Maven, Gradle, Cargo). Change examples include updating "
    "dependencies, configuring build configurations, and adding scripts.",
    "chore": "Code changes for other miscellaneous tasks that do not neatly fit into any of the above categories.",
}
PREFIX = re.compile(r"^(" + "|".join(CCS_TYPES) + r")(\([^)]*\))?!?:\s*", re.I)  # a conventional-commit type prefix


@register_task
class ConventionalCommits(TaskAdapter):
    name = f"{LEAF}/conventional_commits"
    leaf = LEAF
    source = "CCS conventional commit classification dataset, test split (Zeng et al., ICSE 2025)"
    url = "https://huggingface.co/datasets/0x404/ccs_dataset"
    license = "Apache-2.0"
    tier = "A"
    criteria_source = "The ten category definitions and the task wording verbatim from the dataset's classification prompt"
    notes = (
        "Gold is the authors' manual annotation (annotated_type; it agrees with the developer's own conventional-commit "
        "prefix in 85% of the test commits). The commit message is shown with its type prefix removed (masked_commit_message). "
        "Diffs over 60,000 characters are left out."
    )

    def load(self):
        rows = hf_rows("0x404/ccs_dataset", None, "test")
        return [r for r in rows if r["annotated_type"] in CCS_TYPES and len(r["git_diff"]) <= MAX_DIFF_CHARS]

    def label_of(self, record):
        return record["annotated_type"]

    def to_item(self, record, rng):
        question = Question(
            type="choice",
            instructions="Please classify the given commit into one of the ten categories: docs, perf, style, refactor, feat, "
            "fix, test, ci, build, and chore.",
            criteria=shuffled_options(CCS_TYPES, rng),
        )
        message = PREFIX.sub("", record["masked_commit_message"].strip())  # a few masked messages keep a second prefix
        state = {"commit_message": message, "commit_diff": record["git_diff"]}
        gold = Gold(label=record["annotated_type"])
        return self.item(state, {"type": question}, {"type": gold}, sha=record["sha"], developer_type=record["type"])


CODEREVIEWER_LINES = 3000  # the test file is 1.5 GB because each line carries the whole old file; the first lines suffice


@register_task
class CodeReviewerQuality(TaskAdapter):
    name = f"{LEAF}/codereviewer_quality"
    leaf = LEAF
    source = "CodeReviewer diff quality estimation, test split (Li et al., FSE 2022)"
    url = "https://zenodo.org/records/6900648"
    license = "CC BY 4.0 (Zenodo record); code from open-source GitHub projects"
    tier = "B"
    criteria_source = "Adapted from the CodeReviewer task definition: quality estimation predicts whether a diff hunk needs a review comment"
    notes = (
        "Gold is whether the hunk received a reviewer comment in the project's real code review (label 1) or not (label 0). "
        "The review comment itself is withheld. Read from the first 3,000 lines of the test file in the Hugging Face copy "
        "fasterinnerlooper/codereviewer (each line also carries the old file, which is not shown). Hunks over 20,000 "
        "characters are left out."
    )

    def load(self):
        path = CACHE / "codereviewer_quality_test_head.jsonl"
        if not path.exists():
            load_env()
            from huggingface_hub import HfFileSystem

            CACHE.mkdir(parents=True, exist_ok=True)
            lines = []
            with HfFileSystem().open("datasets/fasterinnerlooper/codereviewer/quality/cls-test.jsonl", "rb", block_size=2**22) as f:
                for raw in f:
                    row = json.loads(raw)
                    lines.append(json.dumps({k: row[k] for k in ("id", "proj", "lang", "patch", "y")}, ensure_ascii=False))
                    if len(lines) == CODEREVIEWER_LINES:
                        break
            tmp = path.with_suffix(".part")
            tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            tmp.rename(path)
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return [r for r in rows if r["patch"].strip() and str(r["y"]) in ("0", "1")]

    def label_of(self, record):
        return f"{record['y']}|{record['lang']}"

    def to_item(self, record, rng):
        question = Question(
            type="boolean",
            instructions="In code review, does this diff hunk need a review comment (a reviewer asks for a change or raises a "
            "concern), rather than being accepted as it is?",
        )
        state = {"project": record["proj"], "language": record["lang"], "diff_hunk": record["patch"]}
        gold = Gold(label="true" if str(record["y"]) == "1" else "false")
        return self.item(state, {"needs_comment": question}, {"needs_comment": gold}, original_id=record["id"])
