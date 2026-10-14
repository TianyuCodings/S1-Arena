"""Leaf: translation quality (翻译质量).

Target: the MQM error-weighted score of Freitag et al. (2021). Every segment was rated by
three expert annotators; each annotator's weighted score is put into one of five bands,
the soft label is the distribution of the three bands and the hard label is the band of
the average score (the official segment score).
"""

from __future__ import annotations

import csv
import re
import tarfile
from collections import defaultdict

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch

LEAF = "judge/translation_quality"
REPO = "https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/main/"
LANGUAGES = {"en": "English", "de": "German", "zh": "Chinese"}

# Severity definitions (Freitag et al., 2021, Table 11) and weights (Table 1), verbatim; bands written by us.
MQM_INSTRUCTIONS = (
    "Rate the translation of the source segment by its MQM (Multidimensional Quality Metrics) error-weighted score. Expert "
    "annotators mark every error in the translation with a category (accuracy, fluency, terminology, style, locale "
    "convention) and a severity. Major: errors that may confuse or mislead the reader due to significant change in meaning "
    "or because they appear in a visible or important part of the content. Minor: errors that don't lead to loss of meaning "
    "and wouldn't confuse or mislead the reader but would be noticed, would decrease stylistic quality, fluency or clarity, "
    "or would make the content less appealing. The score adds 5 per major error, 1 per minor error (0.1 per minor "
    "punctuation error) and 25 for a non-translation, and is capped at 25. Which band does the translation's score fall in? "
    "The segment is shown without its surrounding document."
)
BANDS = [
    "Score of 25 (the maximum): five or more major errors, or a non-translation.",
    "Score from 10 to below 25: typically two to four major errors.",
    "Score from 5 to below 10: typically one major error, or five or more minor errors.",
    "Score above 0 and below 5: minor errors only.",
    "Score of 0: no errors.",
]


def band(score: float) -> int:
    if score >= 25:
        return 0
    if score >= 10:
        return 1
    if score >= 5:
        return 2
    if score > 0:
        return 3
    return 4


def weight(category: str, severity: str) -> float:
    category, severity = category.strip().lower(), severity.strip().lower()
    if category.startswith("non-translation"):
        return 25.0
    if category.startswith(("no-error", "source")) or severity not in ("major", "minor"):
        return 0.0
    if severity == "major":
        return 5.0
    return 0.1 if category == "fluency/punctuation" else 1.0


def strip_marks(text: str) -> str:
    return re.sub(r"</?v>", "", text).strip()


class MQMTask(TaskAdapter):
    leaf = LEAF
    tier = "B"
    criteria_source = "Severity definitions and error weights verbatim from Freitag et al. (2021), Tables 1 and 11; the five score bands are ours"
    sampling = "balanced"
    FILES: dict[str, str] = {}  # language pair -> path in the google/wmt-mqm-human-evaluation repository

    def rows(self, lp: str, path: str):
        raise NotImplementedError

    def load(self):
        out = []
        for lp, path in self.FILES.items():
            segments, raters, skip = {}, defaultdict(lambda: defaultdict(float)), set()
            for r in self.rows(lp, path):
                key = (r["system"], r.get("globalSegId") or r["seg_id"])
                if r["severity"] == "HOTW-test":  # attention check: the rater was shown a corrupted translation
                    skip.add(key)
                    continue
                segments.setdefault(key, (strip_marks(r["source"]), strip_marks(r["target"])))
                raters[key][r["rater"]] += weight(r["category"], r["severity"])
            for key, scores in raters.items():
                if key in skip or len(scores) < 2:
                    continue
                capped = [min(25.0, s) for s in scores.values()]
                source, target = segments[key]
                if source.strip() and target:
                    out.append({"lp": lp, "system": key[0], "seg": key[1], "source": source.strip(), "target": target, "scores": capped})
        return out

    def label_of(self, record):
        return f"{record['lp']}|{band(sum(record['scores']) / len(record['scores']))}"

    def to_item(self, record, rng):
        scores = record["scores"]
        dist = {str(i): 0.0 for i in range(len(BANDS))}
        for s in scores:
            dist[str(band(s))] += 1 / len(scores)
        gold = Gold(label=str(band(sum(scores) / len(scores))), dist=dist)
        src, tgt = record["lp"].split("-")
        state = {"source_language": LANGUAGES[src], "target_language": LANGUAGES[tgt], "source": record["source"], "translation": record["target"]}
        question = Question(type="score", instructions=MQM_INSTRUCTIONS, criteria=BANDS)
        meta = {"language_pair": record["lp"], "system": record["system"], "segment": record["seg"], "mqm_scores": scores}
        return self.item(state, {"mqm_band": question}, {"mqm_band": gold}, **meta)


def read_tsv(lines) -> list[dict]:
    return list(csv.DictReader(lines, delimiter="\t", quoting=csv.QUOTE_NONE))


@register_task
class WMT20MQM(MQMTask):
    name = f"{LEAF}/wmt20_mqm"
    source = "WMT 2020 newstest MQM annotations, en-de and zh-en (Freitag et al., 2021)"
    url = "https://github.com/google/wmt-mqm-human-evaluation"
    license = "Apache-2.0 (annotations); WMT news test data for research use"
    notes = ("Ten systems per language pair including human translations; every segment rated by three professional "
             "translators. Sampled evenly across language pair and score band (bands are very unevenly filled: most "
             "segments have no or only minor errors).")
    FILES = {"en-de": "newstest2020/ende/mqm_newstest2020_ende.tsv", "zh-en": "newstest2020/zhen/mqm_newstest2020_zhen.tsv"}

    def rows(self, lp, path):
        with fetch(REPO + path, "mqm_" + path.rsplit("/", 1)[-1]).open(encoding="utf-8") as f:
            return read_tsv(f)


@register_task
class WMT23MQM(MQMTask):
    name = f"{LEAF}/wmt23_mqm"
    source = "WMT 2023 general MT MQM annotations with three ratings per segment, en-de and zh-en (Kocmi et al., 2023; Google MQM release)"
    url = "https://github.com/google/wmt-mqm-human-evaluation"
    license = "Apache-2.0 (annotations); WMT general test data for research use"
    notes = ("Paragraph-level segments (news, user reviews, social media, speech) re-annotated by three raters each; segments "
             "that carried an attention-check (hands-on-the-wheel) annotation are dropped. Sampled evenly across language "
             "pair and score band.")
    FILES = {"en-de": "generalMT2023/ende/mqm_generalMT2023_ende.3ratingsPerSegment.tsv.tar.gz",
             "zh-en": "generalMT2023/zhen/mqm_generalMT2023_zhen.3ratingsPerSegment.tsv.tar.gz"}

    def rows(self, lp, path):
        with tarfile.open(fetch(REPO + path, "mqm_" + path.rsplit("/", 1)[-1])) as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith(".tsv"))
            with tar.extractfile(member) as f:
                return read_tsv(line.decode("utf-8") for line in f)
