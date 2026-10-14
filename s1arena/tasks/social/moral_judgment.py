"""Leaf: moral judgments (道德判断): acceptability, community verdicts and graded agreement.

ETHICS gives a hard acceptability label confirmed by crowd relabelling; Scruples gives
the verdict votes of an online community; Social-Chem-101 gives 50 crowd workers'
graded estimates of how widely a rule of thumb is shared.
"""

from __future__ import annotations

import csv
import io
import json
import tarfile
import zipfile

from ...env import load_env
from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch
from ._common import median_gold, plurality_gold

LEAF = "social/moral_judgment"


# ---------- ETHICS, commonsense morality ----------

@register_task
class EthicsCommonsense(TaskAdapter):
    name = f"{LEAF}/ethics_commonsense"
    leaf = LEAF
    source = "ETHICS commonsense morality, hard test set, short scenarios (Hendrycks et al., 2021)"
    url = "https://huggingface.co/datasets/hendrycks/ethics"
    license = "MIT"
    tier = "B"
    criteria_source = "Adapted from the ETHICS task definition: predict whether, according to commonsense moral judgments, the first-person character clearly should not have done that action"
    notes = ("Short (1-2 sentence) first-person scenarios written on MTurk; labels kept only when at least 4 of 5 "
             "relabelling workers agreed. The long Reddit scenarios are left out (Scruples covers that material).")

    def load(self):
        load_env()
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("hendrycks/ethics", "data/commonsense/test_hard.csv", repo_type="dataset")
        with open(path, encoding="utf-8", newline="") as f:
            return [r for r in csv.DictReader(f) if r["is_short"].strip() == "True" and r["input"].strip()]

    def label_of(self, record):
        return "wrong" if record["label"].strip() == "1" else "not_wrong"

    def to_item(self, record, rng):
        question = Question(
            type="boolean",
            instructions="According to commonsense moral judgments (the moral standards that most people intuitively accept), "
            "should the first-person character clearly not have done what they describe? Answer true if the action is "
            "clearly morally wrong, false if it is not clearly wrong.",
        )
        gold = Gold(label="true" if record["label"].strip() == "1" else "false")
        return self.item({"scenario": record["input"].strip()}, {"clearly_wrong": question}, {"clearly_wrong": gold})


# ---------- Scruples Anecdotes: r/AmItheAsshole community verdicts ----------

SCRUPLES = "https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/data/"
# Classes and meanings from Lourie et al. (2021), Table 2, with the community's initialisms.
SCRUPLES_OPTIONS = {
    "AUTHOR": "The author is in the wrong (“YTA”)",
    "OTHER": "The other party is in the wrong (“NTA”)",
    "EVERYBODY": "Everyone involved is in the wrong (“ESH”)",
    "NOBODY": "Nobody is in the wrong (“NAH”)",
    "INFO": "More information is needed (“INFO”)",
}
SCRUPLES_MIN_VOTES = 10


@register_task
class ScruplesAnecdotes(TaskAdapter):
    name = f"{LEAF}/scruples_anecdotes"
    leaf = LEAF
    source = "Scruples Anecdotes, test split: r/AmItheAsshole posts with the community's verdict votes (Lourie et al., 2021)"
    url = "https://github.com/allenai/scruples"
    license = "Apache-2.0"
    tier = "B"
    criteria_source = "Task name (Who's in the wrong?) and class meanings from Lourie et al. (2021), Table 2, with the subreddit's initialisms"
    notes = (f"Posts with at least {SCRUPLES_MIN_VOTES} verdict votes (top-level comments, extracted by the authors' rules). "
             "Soft label = vote shares; hard label = the released majority label. Posts from November 2018 to April 2019.")

    def load(self):
        path = fetch(SCRUPLES + "anecdotes.tar.gz", "scruples_anecdotes.tar.gz")
        with tarfile.open(path, "r:gz") as archive:
            data = archive.extractfile("anecdotes/test.scruples-anecdotes.jsonl").read().decode("utf-8")
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        return [r for r in rows if sum(r["label_scores"].values()) >= SCRUPLES_MIN_VOTES and r["text"].strip()]

    def label_of(self, record):
        return record["label"]

    def to_item(self, record, rng):
        question = Question(
            type="choice",
            instructions="Who's in the wrong? Members of the r/AmItheAsshole community read this post and voted in the "
            "comments on who was in the wrong. Give the share of voters who chose each verdict.",
            criteria=shuffled_options(SCRUPLES_OPTIONS, rng),
        )
        counts = {key: record["label_scores"].get(key, 0) for key in SCRUPLES_OPTIONS}
        gold = plurality_gold(counts, tie_label=record["label"])
        state = {"title": record["title"].strip(), "text": record["text"].strip()}
        meta = {"original_id": record["id"], "post_type": record["post_type"], "votes": sum(counts.values())}
        return self.item(state, {"verdict": question}, {"verdict": gold}, **meta)


# ---------- Social-Chem-101: anticipated agreement with rules of thumb ----------

# Anticipated-agreement buckets and examples from the annotation interface (Forbes et al., 2020, Appendix A.4.6).
AGREEMENT_LEVELS = [
    "< 1%: almost no one (e.g., “Murdering people is great”)",
    "~5%-25%: people occasionally think this (e.g., “Ghosts exist”)",
    "~50%: controversial, people naturally disagree (e.g., “It's okay to have an abortion”)",
    "~75%-90%: common belief (e.g., “It's okay to try again when you fail”)",
    "> 99%: universally true (e.g., “It's bad to deliberately hurt your pets”)",
]
SOCIAL_CHEM_MIN_ANSWERS = 20


def rot_clause(rot: str) -> str:
    """'It's rude to exclude people.' -> "it's rude to exclude people", for 'What portion ... agree that <rot>?'."""
    text = rot.strip().rstrip(".!?;: ")
    if len(text) > 1 and text[0].isupper() and not text[1].isupper() and not text.startswith("I "):
        text = text[0].lower() + text[1:]
    return text


@register_task
class SocialChemAgreement(TaskAdapter):
    name = f"{LEAF}/social_chem_101"
    leaf = LEAF
    source = "Social-Chem-101, rules of thumb annotated by 50 crowd workers each (analysis split; Forbes et al., 2020)"
    url = "https://github.com/mbforbes/social-chemistry-101"
    license = "CC BY-SA 4.0"
    tier = "A"
    criteria_source = "Worker question 'What portion of people probably agree that ${rot}?' and the five anticipated-agreement buckets verbatim from the dataset documentation and Forbes et al. (2020), Appendix A.4.6"
    notes = ("Soft label = the 50 workers' answers (unanswered ones dropped); hard label = the median bucket. "
             "The authors picked these 400 rules of thumb as the most controversial in the 5-worker annotation, so "
             "disagreement is high by design.")

    def load(self):
        import pandas as pd

        path = fetch("https://storage.googleapis.com/ai2-mosaic-public/projects/social-chemistry/data/social-chem-101.zip", "social-chem-101.zip")
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if n.endswith(".tsv"))
            with archive.open(name) as f:
                frame = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"), sep="\t", usecols=["area", "split", "rot-agree", "situation", "rot", "rot-id"], dtype=str)
        frame = frame[frame["split"] == "analysis"]
        records = []
        for rot_id, group in frame.groupby("rot-id", sort=True):
            answers = [int(float(x)) for x in group["rot-agree"].dropna() if str(x).strip() != ""]
            if len(answers) < SOCIAL_CHEM_MIN_ANSWERS:
                continue
            records.append({"rot_id": rot_id, "rot": group["rot"].iloc[0].strip(), "situation": str(group["situation"].iloc[0]).strip(),
                            "area": group["area"].iloc[0], "answers": sorted(answers)})
        return records

    def label_of(self, record):
        return f"{record['area']}|{record['answers'][(len(record['answers']) - 1) // 2]}"

    def to_item(self, record, rng):
        answers = record["answers"]
        gold = median_gold(answers, len(AGREEMENT_LEVELS))
        question = Question(type="score", instructions=f"What portion of people probably agree that {rot_clause(record['rot'])}?", criteria=AGREEMENT_LEVELS)
        state = {"situation": record["situation"], "rule_of_thumb": record["rot"]}
        meta = {"original_id": record["rot_id"], "area": record["area"], "answers": len(answers)}
        return self.item(state, {"agreement": question}, {"agreement": gold}, **meta)
