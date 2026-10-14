"""Leaf: multi-label tagging (多标签打标). Each label is one boolean question in the same request."""

from __future__ import annotations

import csv
import zipfile

from ...env import load_env
from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch

LEAF = "basic_cls/multilabel"

# GoEmotions, Appendix A "Emotion Definitions" (Demszky et al., 2020), verbatim.
GOEMOTIONS = {
    "admiration": "Finding something impressive or worthy of respect.",
    "amusement": "Finding something funny or being entertained.",
    "anger": "A strong feeling of displeasure or antagonism.",
    "annoyance": "Mild anger, irritation.",
    "approval": "Having or expressing a favorable opinion.",
    "caring": "Displaying kindness and concern for others.",
    "confusion": "Lack of understanding, uncertainty.",
    "curiosity": "A strong desire to know or learn something.",
    "desire": "A strong feeling of wanting something or wishing for something to happen.",
    "disappointment": "Sadness or displeasure caused by the nonfulfillment of one's hopes or expectations.",
    "disapproval": "Having or expressing an unfavorable opinion.",
    "disgust": "Revulsion or strong disapproval aroused by something unpleasant or offensive.",
    "embarrassment": "Self-consciousness, shame, or awkwardness.",
    "excitement": "Feeling of great enthusiasm and eagerness.",
    "fear": "Being afraid or worried.",
    "gratitude": "A feeling of thankfulness and appreciation.",
    "grief": "Intense sorrow, especially caused by someone's death.",
    "joy": "A feeling of pleasure and happiness.",
    "love": "A strong positive emotion of regard and affection.",
    "nervousness": "Apprehension, worry, anxiety.",
    "optimism": "Hopefulness and confidence about the future or the success of something.",
    "pride": "Pleasure or satisfaction due to one's own achievements or the achievements of those with whom one is closely associated.",
    "realization": "Becoming aware of something.",
    "relief": "Reassurance and relaxation following release from anxiety or distress.",
    "remorse": "Regret or guilty feeling.",
    "sadness": "Emotional pain, sorrow.",
    "surprise": "Feeling astonished, startled by something unexpected.",
}


@register_task
class GoEmotions(TaskAdapter):
    name = f"{LEAF}/goemotions"
    leaf = LEAF
    source = "GoEmotions (Demszky et al., 2020)"
    url = "https://huggingface.co/datasets/google-research-datasets/go_emotions"
    license = "Apache-2.0"
    tier = "A"
    criteria_source = "Emotion definitions verbatim from Demszky et al. (2020), Appendix A; neutral wording from the rater instructions (Section 3)"
    notes = "Soft label = share of raters who selected the emotion (3 or 5 raters); hard label = selected by at least 2 raters (the 'simplified' config)."

    def load(self):
        load_env()
        from datasets import load_dataset

        simple = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
        names = simple.features["labels"].feature.names
        raw = load_dataset("google-research-datasets/go_emotions", "raw", split="train").to_pandas()
        test_ids = set(simple["id"])
        raw = raw[raw["id"].isin(test_ids) & ~raw["example_very_unclear"]]
        counts = raw.groupby("id")[names].sum()
        raters = raw.groupby("id").size()
        rows = []
        for row in simple:
            if row["id"] not in counts.index:
                continue
            n = int(raters[row["id"]])
            rows.append({
                "id": row["id"], "text": row["text"], "labels": [names[i] for i in row["labels"]],
                "share": {e: float(counts.loc[row["id"], e]) / n for e in names}, "raters": n,
            })
        return rows

    def label_of(self, record):
        return record["labels"][0] if record["labels"] else "neutral"

    def to_item(self, record, rng):
        questions, gold = {}, {}
        for emotion, share in record["share"].items():
            if emotion == "neutral":
                text = "Is no emotion clearly expressed in this Reddit comment (neutral)? Raters chose neutral when they were not certain that any emotion was expressed."
            else:
                text = f"Is the emotion \"{emotion}\" expressed in this Reddit comment? {emotion}: {GOEMOTIONS[emotion]} Answer true only if you are reasonably confident it is expressed."
            questions[emotion] = Question(type="boolean", instructions=text)
            share = min(1.0, max(0.0, share))
            gold[emotion] = Gold(label="true" if emotion in record["labels"] else "false", dist={"true": share, "false": 1.0 - share})
        return self.item({"comment": record["text"]}, questions, gold, original_id=record["id"], raters=record["raters"])


# SemEval-2018 Task 1 E-c option glosses (Mohammad et al., 2018, Section 3.4.1), verbatim.
EC_GLOSSES = {
    "anger": "also includes annoyance, rage",
    "anticipation": "also includes interest, vigilance",
    "disgust": "also includes disinterest, dislike, loathing",
    "fear": "also includes apprehension, anxiety, terror",
    "joy": "also includes serenity, ecstasy",
    "love": "also includes affection",
    "optimism": "also includes hopefulness, confidence",
    "pessimism": "also includes cynicism, no confidence",
    "sadness": "also includes pensiveness, grief",
    "surprise": "also includes distraction, amazement",
    "trust": "also includes acceptance, liking, admiration",
}


@register_task
class SemEval2018EC(TaskAdapter):
    name = f"{LEAF}/semeval2018_ec"
    leaf = LEAF
    source = "SemEval-2018 Task 1 Affect in Tweets, emotion classification E-c (Mohammad et al., 2018)"
    url = "https://saifmohammad.com/WebPages/affectintweets.htm"
    license = "Research use only; do not redistribute (NRC)"
    redistributable = False
    tier = "A"
    criteria_source = "Emotion glosses verbatim from Mohammad et al. (2018), Section 3.4.1; task wording from task 5"

    def load(self):
        path = "SemEval2018-Task1-all-data/English/E-c/2018-E-c-En-test-gold.txt"
        zip_path = fetch("https://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip", "ait2018.zip")
        with zipfile.ZipFile(zip_path) as zf, zf.open(path) as f:
            return list(csv.DictReader((line.decode("utf-8") for line in f), delimiter="\t"))

    def label_of(self, record):
        present = [e for e in EC_GLOSSES if record[e] == "1"]
        return present[0] if present else "none"

    def to_item(self, record, rng):
        questions, gold = {}, {}
        for emotion, gloss in EC_GLOSSES.items():
            questions[emotion] = Question(
                type="boolean",
                instructions=f"Is {emotion} ({gloss}) one of the emotions that best represent the mental state of the tweeter?",
            )
            gold[emotion] = Gold(label="true" if record[emotion] == "1" else "false")
        return self.item({"tweet": record["Tweet"]}, questions, gold, original_id=record["ID"])
