"""Leaf: patch correctness (补丁是否正确) for SWE-style issue fixing.

Gold is the recorded outcome of running the repository's tests on an agent's final patch
(resolved: the maintainers' fail-to-pass tests now pass and pass-to-pass tests still pass).
Every record holds one resolved and one unresolved patch for the same issue, so a model has to
tell them apart rather than guess from the issue alone.
"""

from __future__ import annotations

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import hf_rows
from ._hub import hub_columns

LEAF = "code/patch_correctness"
MAX_PATCH_CHARS = 20_000
RESOLVED = (
    "A patch resolves the issue when, after it is applied, the tests the maintainers wrote for the issue pass and "
    "the repository's existing tests still pass."
)


def pick(frame, resolved: bool):
    rows = frame[frame["resolved"] == resolved].sort_values("key")
    return rows.iloc[0] if len(rows) else None


def mixed_pairs(patches, statements: dict[str, dict]) -> list[dict]:
    """One resolved and one unresolved patch (distinct, non-empty, not too long) per issue that has both."""
    patches = patches[patches["patch"].str.strip().str.len().between(1, MAX_PATCH_CHARS)]
    records = []
    for instance_id, group in patches.groupby("instance_id", sort=True):
        if instance_id not in statements:
            continue
        good, bad = pick(group, True), pick(group, False)
        if good is None or bad is None or good["patch"].strip() == bad["patch"].strip():
            continue
        records.append({**statements[instance_id], "instance_id": instance_id, "resolved_patch": good["patch"], "unresolved_patch": bad["patch"],
                        "resolved_key": good["key"], "unresolved_key": bad["key"]})
    return records


@register_task
class SWERebenchPatches(TaskAdapter):
    name = f"{LEAF}/swe_rebench"
    leaf = LEAF
    source = "SWE-rebench tasks with OpenHands + Qwen3-Coder-480B final patches and their test results (Badertdinov et al., 2025)"
    url = "https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories"
    license = "CC BY 4.0"
    tier = "C"
    criteria_source = "Written from the SWE-bench definition of a resolved task; gold is the recorded resolved flag"
    items_per_record = 2
    notes = (
        "Only issues created in 2024 or later (SWE-rebench collects new tasks continuously, which limits contamination). "
        "Each record is one issue with one resolved and one unresolved final patch from different agent runs; both are "
        "asked. Only the patch columns of the trajectory file are downloaded; the trajectories themselves are not shown."
    )

    def load(self):
        patches = hub_columns(
            "nebius/SWE-rebench-openhands-trajectories", ["trajectories.parquet"],
            ["trajectory_id", "instance_id", "model_patch", "resolved"], "swe_rebench_openhands_patches",
        ).rename(columns={"model_patch": "patch", "trajectory_id": "key"})
        patches["resolved"] = patches["resolved"] == 1
        tasks = hub_columns(
            "nebius/SWE-rebench", [f"data/test-0000{i}-of-00002.parquet" for i in range(2)],
            ["instance_id", "repo", "problem_statement", "created_at"], "swe_rebench_tasks",
        )
        tasks = tasks[tasks["created_at"].astype(str) >= "2024"]
        statements = {r.instance_id: {"repo": r.repo, "issue": r.problem_statement, "created_at": str(r.created_at)[:10]} for r in tasks.itertuples()}
        return mixed_pairs(patches, statements)

    def label_of(self, record):
        return record["created_at"][:4]

    def to_items(self, record, rng):
        question = Question(type="boolean", instructions="Does this patch resolve the issue? " + RESOLVED)
        items = []
        for version in ("resolved", "unresolved"):
            state = {"repository": record["repo"], "issue": record["issue"], "patch": record[f"{version}_patch"]}
            gold = Gold(label="true" if version == "resolved" else "false")
            items.append(self.item(state, {"resolves": question}, {"resolves": gold}, version=version, instance_id=record["instance_id"],
                                   trajectory_id=record[f"{version}_key"], created_at=record["created_at"]))
        return items


@register_task
class SWEAgentPatchPairs(TaskAdapter):
    name = f"{LEAF}/swe_bench_extra_pairwise"
    leaf = LEAF
    source = "SWE-bench-extra tasks with SWE-agent final patches from Nebius's open-weight agent runs and their test results (Golubev et al., 2024)"
    url = "https://huggingface.co/datasets/nebius/SWE-agent-trajectories"
    license = "CC BY 4.0"
    tier = "C"
    criteria_source = "Written from the SWE-bench definition of a resolved task; gold is the recorded resolved flag"
    items_per_record = 2
    notes = (
        "Pairwise: for one issue, one patch that resolved it and one that did not (runs of Llama-based SWE-agent models); "
        "each pair is asked in both orders. Only the patch columns of the trajectory files are downloaded."
    )

    def load(self):
        patches = hub_columns(
            "nebius/SWE-agent-trajectories", [f"data/train-{i:05d}-of-00012.parquet" for i in range(12)],
            ["instance_id", "model_name", "target", "generated_patch"], "swe_agent_trajectory_patches",
        ).rename(columns={"generated_patch": "patch", "target": "resolved"})
        patches["key"] = [f"{model}:{i:06d}" for i, model in enumerate(patches["model_name"])]
        tasks = hf_rows("nebius/SWE-bench-extra", None, "train")
        statements = {r["instance_id"]: {"repo": r["repo"], "issue": r["problem_statement"], "created_at": str(r["created_at"])[:10]} for r in tasks}
        return mixed_pairs(patches, statements)

    def label_of(self, record):
        return record["created_at"][:4]

    def to_items(self, record, rng):
        question = Question(
            type="choice",
            instructions="Exactly one of the two patches resolves the issue. Which one? " + RESOLVED,
            criteria={"A": "Patch A resolves the issue.", "B": "Patch B resolves the issue."},
        )
        items = []
        for first in ("resolved", "unresolved"):
            second = "unresolved" if first == "resolved" else "resolved"
            state = {"repository": record["repo"], "issue": record["issue"], "patch_a": record[f"{first}_patch"], "patch_b": record[f"{second}_patch"]}
            gold = Gold(label="A" if first == "resolved" else "B")
            items.append(self.item(state, {"better": question}, {"better": gold}, order="resolved_first" if first == "resolved" else "resolved_second",
                                   instance_id=record["instance_id"], created_at=record["created_at"]))
        return items
