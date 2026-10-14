"""Leaf: language identification, formality and readability (语种 / 文体 / 正式度 / 可读性)."""

from __future__ import annotations

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import hf_rows

LEAF = "basic_cls/language_style"

# WiLI uses Wikipedia language codes; these are not ISO 639-3 and need names by hand.
WIKI_CODES = {
    "zh-yue": "Cantonese",
    "be-tarask": "Belarusian (Taraškievica orthography)",
    "roa-tara": "Tarantino",
    "roa-rup": "Aromanian",
    "nds-nl": "Dutch Low Saxon",
    "map-bms": "Banyumasan",
    "fiu-vro": "Võro",
    "bat-smg": "Samogitian",
    "cbk": "Chavacano",
    "zh-classical": "Classical Chinese",
    "simple": "Simple English",
}


def language_name(code: str) -> str:
    import pycountry

    if code in WIKI_CODES:
        return WIKI_CODES[code]
    lang = pycountry.languages.get(alpha_3=code) or (pycountry.languages.get(alpha_2=code) if len(code) == 2 else None)
    return lang.name if lang else code


@register_task
class WiLI2018(TaskAdapter):
    name = f"{LEAF}/wili2018"
    leaf = LEAF
    source = "WiLI-2018 Wikipedia language identification (Thoma, 2018)"
    url = "https://huggingface.co/datasets/MartinThoma/wili_2018"
    license = "ODbL 1.0"
    tier = "C"
    criteria_source = "Language names from ISO 639-3 (pycountry) and Wikipedia language codes"
    notes = "About 1.2% of test labels are known to be wrong (WiLI-2018-corrected)."

    def load(self):
        rows = hf_rows("MartinThoma/wili_2018", "WiLI-2018 dataset", "test")
        codes = sorted({r["label_name"] for r in rows})
        self.options = {code: f"{language_name(code)} ({code})" for code in codes}
        return rows

    def label_of(self, record):
        return record["label_name"]

    def to_item(self, record, rng):
        question = Question(type="choice", instructions="Which language is this paragraph written in?", criteria=shuffled_options(self.options, rng))
        return self.item({"paragraph": record["sentence"]}, {"language": question}, {"language": Gold(label=record["label_name"])})


@register_task
class PavlickFormality(TaskAdapter):
    name = f"{LEAF}/pavlick_formality"
    leaf = LEAF
    source = "Formality scores (Pavlick and Tetreault, 2016)"
    url = "https://huggingface.co/datasets/osyvokon/pavlick-formality-scores"
    license = "CC BY 3.0"
    tier = "B"
    criteria_source = "Question and scale endpoints verbatim from Pavlick and Tetreault (2016); intermediate levels written"
    notes = "Gold is the mean of 5 ratings rounded to the nearest level; the individual ratings are not in the HF copy."

    LEVELS = [
        "-3: Very Informal",
        "-2: Informal",
        "-1: Somewhat informal",
        "0: Neither formal nor informal",
        "1: Somewhat formal",
        "2: Formal",
        "3: Very Formal",
    ]

    def load(self):
        return hf_rows("osyvokon/pavlick-formality-scores", None, "test")

    def level(self, record) -> int:
        return min(6, max(0, round(float(record["avg_score"])) + 3))

    def label_of(self, record):
        return str(self.level(record))

    def to_item(self, record, rng):
        question = Question(type="score", instructions="How formal do you think this sentence is?", criteria=self.LEVELS)
        gold = Gold(label=str(self.level(record)))
        return self.item({"sentence": record["sentence"]}, {"formality": question}, {"formality": gold}, domain=record["domain"], avg_score=record["avg_score"])


@register_task
class CEFRSP(TaskAdapter):
    name = f"{LEAF}/cefr_sp"
    leaf = LEAF
    source = "CEFR-SP, public part (Arase et al., 2022)"
    url = "https://huggingface.co/datasets/edesaras/CEFR-Sentence-Level-Annotations"
    license = "CC BY-SA 3.0 (Wiki-Auto part) / CC BY-NC-SA 4.0 (SCoRE part)"
    tier = "B"
    criteria_source = "Level descriptions adapted from the CEFR global reading descriptors (Council of Europe)"
    notes = "Soft label from two expert annotators; on disagreement the hard label is the lower level."

    LEVELS = [
        "A1: can be understood by a reader who knows only very short, simple texts with familiar names, words and basic phrases.",
        "A2: can be understood by a reader who handles short, simple texts on familiar matters with high-frequency everyday vocabulary.",
        "B1: can be understood by a reader who handles straightforward factual texts on subjects related to their interests.",
        "B2: can be understood by a reader who reads with a large degree of independence, including articles on contemporary problems.",
        "C1: can be understood by a reader who handles long and complex factual and literary texts and distinctions of style.",
        "C2: can be understood by a reader who handles virtually all forms of written language, including abstract and complex texts.",
    ]

    def load(self):
        return hf_rows("edesaras/CEFR-Sentence-Level-Annotations", None, "test")

    def label_of(self, record):
        return str(min(record["Annotator I"], record["Annotator II"]))

    def to_item(self, record, rng):
        levels = [record["Annotator I"] - 1, record["Annotator II"] - 1]
        if not all(0 <= x < 6 for x in levels):
            return None
        dist = {str(i): levels.count(i) / 2 for i in range(6) if i in levels}
        question = Question(type="score", instructions="Which CEFR level best describes the reading ability needed to understand this sentence?", criteria=self.LEVELS)
        return self.item({"sentence": record["text"]}, {"cefr": question}, {"cefr": Gold(dist=dist)})
