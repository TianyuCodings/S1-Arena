"""Leaf: sentiment and emotion (情感 / 情绪)."""

from __future__ import annotations

import csv
import zipfile

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch, hf_rows

LEAF = "basic_cls/sentiment"
AIT_ZIP = "https://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip"


@register_task
class TweetEvalSentiment(TaskAdapter):
    name = f"{LEAF}/tweeteval_sentiment"
    leaf = LEAF
    source = "TweetEval sentiment (SemEval-2017 Task 4A; Barbieri et al., 2020)"
    url = "https://huggingface.co/datasets/cardiffnlp/tweet_eval"
    license = "CC BY 3.0 (card) + Twitter ToS"
    tier = "B"
    criteria_source = "Adapted from the SemEval-2017 Task 4 crowdsourcing instructions (Rosenthal et al., 2017)"

    OPTIONS = {
        "positive": "The tweet expresses an overall positive sentiment or opinion.",
        "negative": "The tweet expresses an overall negative sentiment or opinion.",
        "neutral": "The tweet expresses neither a positive nor a negative sentiment, or is purely factual.",
    }

    def load(self):
        return hf_rows("cardiffnlp/tweet_eval", "sentiment", "test")

    def label_of(self, record):
        return record["label_name"]

    def to_item(self, record, rng):
        question = Question(type="choice", instructions="What overall sentiment does this tweet express?", criteria=shuffled_options(self.OPTIONS, rng))
        return self.item({"tweet": record["text"]}, {"sentiment": question}, {"sentiment": Gold(label=record["label_name"])})


@register_task
class SST5(TaskAdapter):
    name = f"{LEAF}/sst5"
    leaf = LEAF
    source = "Stanford Sentiment Treebank, 5 classes (Socher et al., 2013)"
    url = "https://huggingface.co/datasets/SetFit/sst5"
    license = "Unspecified (research use)"
    tier = "C"
    criteria_source = "Written from the SST label names"

    LEVELS = [
        "Very negative: strongly unfavorable.",
        "Negative: unfavorable.",
        "Neutral: neither favorable nor unfavorable.",
        "Positive: favorable.",
        "Very positive: strongly favorable.",
    ]

    def load(self):
        return hf_rows("SetFit/sst5", None, "test")

    def label_of(self, record):
        return str(record["label"])

    def to_item(self, record, rng):
        question = Question(type="score", instructions="How positive or negative is this sentence from a movie review?", criteria=self.LEVELS)
        return self.item({"sentence": record["text"]}, {"sentiment": question}, {"sentiment": Gold(label=str(record["label"]))})


@register_task
class SemEval2018VOC(TaskAdapter):
    name = f"{LEAF}/semeval2018_voc"
    leaf = LEAF
    source = "SemEval-2018 Task 1 Affect in Tweets, valence ordinal classification (Mohammad et al., 2018)"
    url = "https://saifmohammad.com/WebPages/affectintweets.htm"
    license = "Research use only; do not redistribute (NRC)"
    redistributable = False
    tier = "A"
    criteria_source = "Level descriptions verbatim from the official V-oc gold files; task wording from Mohammad et al. (2018), task 4"

    LEVELS = [
        "-3: very negative emotional state can be inferred",
        "-2: moderately negative emotional state can be inferred",
        "-1: slightly negative emotional state can be inferred",
        "0: neutral or mixed emotional state can be inferred",
        "1: slightly positive emotional state can be inferred",
        "2: moderately positive emotional state can be inferred",
        "3: very positive emotional state can be inferred",
    ]

    def load(self):
        path = "SemEval2018-Task1-all-data/English/V-oc/2018-Valence-oc-En-test-gold.txt"
        with zipfile.ZipFile(fetch(AIT_ZIP, "ait2018.zip")) as zf, zf.open(path) as f:
            rows = list(csv.DictReader((line.decode("utf-8") for line in f), delimiter="\t"))
        return rows

    def label_of(self, record):
        return record["Intensity Class"]

    def to_item(self, record, rng):
        level = self.LEVELS.index(record["Intensity Class"].strip())
        question = Question(
            type="score",
            instructions="Classify the tweet into one of seven ordinal classes, corresponding to various levels of "
            "positive and negative sentiment intensity, that best represents the mental state of the tweeter.",
            criteria=self.LEVELS,
        )
        return self.item({"tweet": record["Tweet"]}, {"valence": question}, {"valence": Gold(label=str(level))}, original_id=record["ID"])
