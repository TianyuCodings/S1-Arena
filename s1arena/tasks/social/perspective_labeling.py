"""Leaf: labels by annotator group (按视角标注).

"How would group X label this?" Each item shows one text and asks one question per
annotator group of a single demographic dimension (e.g. women and men, or five
race/ethnicity groups); the gold of each question is that group's answer shares.
Only datasets that record who gave each rating are used, and a group enters an item
only when enough of its members rated that text.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict

from ...env import load_env
from ...registry import register_task
from ...schema import Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch
from ._common import plurality_gold

LEAF = "social/perspective_labeling"


def group_questions(groups: dict[str, dict[str, int]], spec: dict[str, tuple[str, str]], template: str, options: dict[str, str], rng):
    """One choice question per group, in a fixed group order; gold = that group's answer shares.

    spec maps a group key to (question name, phrase completing "... who <phrase>"). The option order is
    shuffled once per item and shared by all its questions."""
    shown = shuffled_options(options, rng)
    questions, gold = {}, {}
    for group, counts in groups.items():
        name, phrase = spec[group]
        questions[name] = Question(type="choice", instructions=template.format(group=phrase), criteria=dict(shown))
        gold[name] = plurality_gold({key: counts.get(key, 0) for key in options})  # ties broken in the listed (unshuffled) order
    return questions, gold


# ---------- DICES: safety ratings of chatbot conversations by rater demographics ----------

DICES_RAW = "https://raw.githubusercontent.com/google-research-datasets/dices-dataset/main/"
# Raters that failed the authors' quality checks, as listed in the dataset READMEs.
DICES_REMOVED = {
    "350": {"297514565398139", "297515609163939", "297515750682315", "297515617432733", "297541515566649", "297541515769980",
            "297515629971478", "297059995361243", "297541522412126", "297540556928761", "297541321453321", "297540562350921",
            "297540983991638", "297060365288109", "297514543980607", "297515729806999", "297541271027233", "296709611112092",
            "296709543131761"},
    "990": {"296768824962581", "296740172100277", "296729424498056", "296835612486158", "297212198239077", "296740254381555",
            "296740049152799", "297212002905763", "296740761150184", "296708974347177", "296770194464643", "296768399729683"},
}
DICES_OPTIONS = {
    "unsafe": "Unsafe: the rater answered Yes to at least one of the safety questions",
    "unsure": "Unsure: the rater answered Unsure to at least one safety question and Yes to none",
    "safe": "Safe: the rater answered No to every safety question",
}
DICES_VALUES = {"Yes": "unsafe", "Unsure": "unsure", "No": "safe"}
DICES_REASONS = {
    "350": "harmful content, unfair bias, misinformation, political affiliation, or safety policy violations (polarizing topics, endorsements)",
    "990": "harmful content, unfair bias, or misinformation",
}
# Dimension -> {value in the data: (question name, phrase completing "raters who ...")}; the order fixes the question order.
DICES_GENDER = {"Woman": ("women", "identify as women"), "Man": ("men", "identify as men")}
DICES_AGE = {"gen z": ("gen_z", "belong to Generation Z"), "millenial": ("millennials", "are Millennials"),
             "gen x+": ("gen_x_or_older", "belong to Generation X or older")}
DICES_EDUCATION = {"College degree or higher": ("college_degree", "have a college degree or higher"),
                   "High school or below": ("high_school_or_less", "have a high school education or less")}
DICES_GROUPS = {
    "350": {
        "rater_gender": DICES_GENDER,
        "rater_race": {"White": ("white", "identify as White"), "Black/African American": ("black", "identify as Black or African American"),
                       "LatinX, Latino, Hispanic or Spanish Origin": ("latinx", "identify as Latinx, Latino, Hispanic or of Spanish origin"),
                       "Asian/Asian subcontinent": ("asian", "identify as Asian (including the Asian subcontinent)"),
                       "Multiracial": ("multiracial", "identify as multiracial")},
        "rater_age": DICES_AGE,
        "rater_education": DICES_EDUCATION,
    },
    "990": {
        "rater_locale": {"US": ("us", "are based in the United States"), "IN": ("india", "are based in India")},
        "rater_gender": DICES_GENDER,
        "rater_age": DICES_AGE,
        "rater_education": DICES_EDUCATION,
    },
}
DICES_MIN_GROUP = 5


class DICES(TaskAdapter):
    part = ""  # "350" or "990"
    leaf = LEAF
    url = "https://github.com/google-research-datasets/dices-dataset"
    license = "CC BY 4.0"
    tier = "B"
    criteria_source = ("Safety question categories and the Yes / No / Unsure answers with the rule that aggregates them into "
                       "the overall rating (Q_overall), adapted from Aroyo et al. (2023), Sections 3.4 and 4")

    def load(self):
        path = fetch(DICES_RAW + f"{self.part}/diverse_safety_adversarial_dialog_{self.part}.csv", f"dices_{self.part}.csv")
        removed = DICES_REMOVED[self.part]
        conversations: dict[str, dict] = {}
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["rater_id"].strip() in removed or row["Q_overall"] not in DICES_VALUES:
                    continue
                item = conversations.setdefault(row["item_id"], {"context": row["context"], "response": row["response"], "ratings": []})
                item["ratings"].append({**{dim: row[dim] for dim in DICES_GROUPS[self.part]}, "answer": DICES_VALUES[row["Q_overall"]]})
        records = []
        for item_id, item in sorted(conversations.items(), key=lambda kv: int(kv[0])):
            overall = defaultdict(int)
            for rating in item["ratings"]:
                overall[rating["answer"]] += 1
            majority = max(DICES_OPTIONS, key=lambda k: overall.get(k, 0))
            for dimension, phrases in DICES_GROUPS[self.part].items():
                groups = {value: defaultdict(int) for value in phrases}
                for rating in item["ratings"]:
                    if rating[dimension] in groups:
                        groups[rating[dimension]][rating["answer"]] += 1
                if all(sum(c.values()) >= DICES_MIN_GROUP for c in groups.values()):
                    records.append({"item_id": item_id, "context": item["context"], "response": item["response"], "dimension": dimension,
                                    "majority": majority, "groups": {value: dict(c) for value, c in groups.items()}})
        return records

    def conversation(self, context: str) -> str:
        return context.strip()

    def label_of(self, record):
        return f"{record['dimension']}|{record['majority']}"

    def to_item(self, record, rng):
        template = ("Raters read the conversation and answered, for each of several reasons, whether the chatbot's last "
                    f"response is unsafe: {DICES_REASONS[self.part]}. Each rater's answers combine into one overall rating. "
                    "What share of the raters who {group} gave each overall rating?")
        spec = DICES_GROUPS[self.part][record["dimension"]]
        questions, gold = group_questions(record["groups"], spec, template, DICES_OPTIONS, rng)
        state = {"conversation": self.conversation(record["context"]), "last_chatbot_response": record["response"].strip()}
        meta = {"original_id": record["item_id"], "dimension": record["dimension"].removeprefix("rater_"),
                "raters": {spec[g][0]: sum(c.values()) for g, c in record["groups"].items()}}
        return self.item(state, questions, gold, **meta)


@register_task
class DICES350(DICES):
    part = "350"
    name = f"{LEAF}/dices_350"
    source = "DICES-350: adversarial chatbot conversations rated for safety by all 104 raters of a pool balanced by gender and race (Aroyo et al., 2023)"
    notes = ("One item per (conversation, demographic dimension), with one question per group; each question's gold is the "
             "group's shares of the overall rating (Q_overall). The 19 raters that failed the authors' quality checks are "
             "removed, leaving 104. Raters' demographics are self-reported.")


@register_task
class DICES990(DICES):
    part = "990"
    name = f"{LEAF}/dices_990"
    source = "DICES-990: adversarial chatbot conversations, each rated for safety by 60-70 raters balanced by gender and locale (US, India) (Aroyo et al., 2023)"
    notes = ("One item per (conversation, demographic dimension), with one question per group (locale, gender, age, "
             "education); groups with fewer than 5 raters on a conversation are left out. The 12 raters that failed the "
             "authors' quality checks are removed. Speakers A and B are shown as USER and CHATBOT.")

    def conversation(self, context: str) -> str:
        text = context.replace("\\n", "\n").strip()
        text = re.sub(r"(?m)^A:\s?", "USER: ", text)
        return re.sub(r"(?m)^B:\s?", "CHATBOT: ", text)


# ---------- Measuring Hate Speech: the 70-comment reference set rated by hundreds of annotators ----------

# Item wording verbatim from Sachdeva et al. (2022), Appendix A, Table 1. Codes in the data: 0 = no, 1 = unclear, 2 = yes.
MHS_QUESTION = ("Crowd workers in the United States answered this survey item about the comment: \"Does this comment contain "
                "hate speech, defined as 'bias-motivated, hostile and malicious language targeted at a person/group because of "
                "their actual or perceived innate characteristics, especially when the group is unnecessarily labeled'?\" "
                "What share of the annotators who {group} chose each answer?")
MHS_OPTIONS = {"yes": "Yes", "no": "No", "unclear": "Unclear"}
MHS_VALUES = {2: "yes", 0: "no", 1: "unclear"}
MHS_MIN_GROUP = 20
MHS_REFERENCE_MIN = 100  # the reference-set comments are the only ones rated by more than a handful of annotators


def _ideology(value) -> str | None:
    value = str(value or "")
    if value.endswith("liberal"):
        return "liberal"
    if value.endswith("conservative"):
        return "conservative"
    return "moderate" if value == "neutral" else None


def _education(value) -> str | None:
    value = str(value or "")
    if value in {"college_grad_ba", "masters", "professional_degree", "phd"}:
        return "ba"
    return "no_ba" if value in {"some_high_school", "high_school_grad", "some_college", "college_grad_aa"} else None


def _income(value) -> str | None:
    return {"<10k": "low", "10k-50k": "low", "50k-100k": "middle", "100k-200k": "high", ">200k": "high"}.get(str(value or ""))


def _age(value) -> str | None:
    try:
        age = float(value)
    except (TypeError, ValueError):
        return None
    if age != age:  # NaN
        return None
    return "18-29" if age < 30 else ("30-44" if age < 45 else "45+")


# Dimension -> (function of the annotator row returning the group keys it belongs to, {group key: phrase}).
MHS_DIMENSIONS = {
    "gender": (lambda r: [{"female": "women", "male": "men"}.get(r["annotator_gender"])],
               {"women": ("women", "identify as women"), "men": ("men", "identify as men")}),
    "race": (lambda r: [g for g in ("black", "white", "latinx", "asian") if r[f"annotator_race_{g}"]],
             {"black": ("black", "identify as Black (alone or with another race)"),
              "white": ("white", "identify as White (alone or with another race)"),
              "latinx": ("latinx", "identify as Latino/Latina or Hispanic (alone or with another race)"),
              "asian": ("asian", "identify as Asian (alone or with another race)")}),
    "ideology": (lambda r: [_ideology(r["annotator_ideology"])],
                 {"liberal": ("liberal", "describe their politics as liberal (slightly, moderately or extremely)"),
                  "moderate": ("neutral", "describe their politics as neutral"),
                  "conservative": ("conservative", "describe their politics as conservative (slightly, moderately or extremely)")}),
    "religion": (lambda r: [g for g in ("christian", "atheist", "nothing") if r[f"annotator_religion_{g}"]],
                 {"christian": ("christian", "identify as Christian"), "atheist": ("atheist", "identify as atheist"),
                  "nothing": ("no_religion", "report no religion")}),
    "sexuality": (lambda r: [g for g in ("straight", "bisexual", "gay") if r[f"annotator_sexuality_{g}"]],
                  {"straight": ("heterosexual", "identify as heterosexual"), "bisexual": ("bisexual", "identify as bisexual"),
                   "gay": ("gay", "identify as gay")}),
    "education": (lambda r: [_education(r["annotator_educ"])],
                  {"no_ba": ("no_bachelors_degree", "do not have a bachelor's degree"),
                   "ba": ("bachelors_degree_or_higher", "have a bachelor's degree or higher")}),
    "income": (lambda r: [_income(r["annotator_income"])],
               {"low": ("income_under_50k", "have a household income under $50,000"),
                "middle": ("income_50k_to_100k", "have a household income of $50,000 to $100,000"),
                "high": ("income_over_100k", "have a household income over $100,000")}),
    "age": (lambda r: [_age(r["annotator_age"])],
            {"18-29": ("age_18_29", "are 18 to 29 years old"), "30-44": ("age_30_44", "are 30 to 44 years old"),
             "45+": ("age_45_plus", "are 45 or older")}),
}


@register_task
class MeasuringHateSpeech(TaskAdapter):
    name = f"{LEAF}/measuring_hate_speech"
    leaf = LEAF
    source = "Measuring Hate Speech corpus, the 70 reference-set comments rated by 240-815 U.S. crowd workers each (Kennedy et al., 2020; Sachdeva et al., 2022)"
    url = "https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech"
    license = "CC BY 4.0"
    tier = "A"
    criteria_source = "Hate speech survey item and its answers (Yes / No / Unclear) verbatim from Sachdeva et al. (2022), Appendix A, Table 1"
    notes = ("Only the reference set: 70 comments the authors chose as examples of seven levels from supportive speech to "
             f"genocidal hate speech, which every annotator saw a few of. One item per (comment, demographic dimension), "
             f"with one question per group of at least {MHS_MIN_GROUP} annotators; demographics are self-reported. "
             "Race, religion and sexual orientation allow several answers, so those groups overlap.")

    def load(self):
        load_env()
        import pandas as pd
        from huggingface_hub import hf_hub_download

        frame = pd.read_parquet(hf_hub_download("ucberkeley-dlab/measuring-hate-speech", "measuring-hate-speech.parquet", repo_type="dataset"))
        sizes = frame.groupby("comment_id").size()
        frame = frame[frame["comment_id"].isin(sizes[sizes >= MHS_REFERENCE_MIN].index)]
        records = []
        for comment_id, rows in frame.groupby("comment_id", sort=True):
            rows = rows.to_dict("records")
            answers = [MHS_VALUES.get(int(r["hatespeech"])) if r["hatespeech"] == r["hatespeech"] else None for r in rows]
            overall = defaultdict(int)
            for answer in answers:
                if answer:
                    overall[answer] += 1
            majority = max(MHS_OPTIONS, key=lambda k: overall.get(k, 0))
            for dimension, (groups_of, phrases) in MHS_DIMENSIONS.items():
                groups = {key: defaultdict(int) for key in phrases}
                for row, answer in zip(rows, answers, strict=True):
                    if answer:
                        for key in groups_of(row):
                            if key in groups:
                                groups[key][answer] += 1
                groups = {key: dict(c) for key, c in groups.items() if sum(c.values()) >= MHS_MIN_GROUP}
                if len(groups) >= 2:
                    records.append({"comment_id": int(comment_id), "text": str(rows[0]["text"]).strip(), "dimension": dimension,
                                    "majority": majority, "groups": groups})
        return records

    def label_of(self, record):
        return f"{record['dimension']}|{record['majority']}"

    def to_item(self, record, rng):
        spec = MHS_DIMENSIONS[record["dimension"]][1]
        questions, gold = group_questions(record["groups"], spec, MHS_QUESTION, MHS_OPTIONS, rng)
        meta = {"original_id": record["comment_id"], "dimension": record["dimension"],
                "annotators": {spec[g][0]: sum(c.values()) for g, c in record["groups"].items()}}
        return self.item({"comment": record["text"]}, questions, gold, **meta)
