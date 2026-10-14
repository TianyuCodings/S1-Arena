"""Leaf: opinion distributions of countries and groups (群体观点分布).

Each item names a population and a question; the gold is that population's answer
shares (a soft label), with the most common answer as the hard label. Survey answer
options keep the order the survey presented them in, since most are ordered scales;
the Polis vote options are shuffled.
"""

from __future__ import annotations

import ast
import csv
import datetime as dt
import json

from ...env import load_env
from ...registry import register_task
from ...schema import Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch, humanize
from ._common import LETTERS, lettered, plurality_gold

LEAF = "social/opinion_distribution"


def hf_file(repo: str, filename: str) -> str:
    load_env()
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset")


# ---------- GlobalOpinionQA: countries (Pew Global Attitudes, World Values Survey) ----------

SURVEYS = {
    "GAS": "the Pew Research Center's Global Attitudes Survey",
    "WVS": "the World Values Survey, wave 7 (2017-2022)",
}


@register_task
class GlobalOpinionQA(TaskAdapter):
    name = f"{LEAF}/globalopinionqa"
    leaf = LEAF
    source = "GlobalOpinionQA: Pew Global Attitudes Survey and World Values Survey questions with answer shares by country (Durmus et al., 2023)"
    url = "https://huggingface.co/datasets/Anthropic/llm_global_opinions"
    license = "CC BY-NC-SA 4.0"
    tier = "B"
    criteria_source = ("Survey questions and answer options verbatim from the surveys; instruction adapted from the "
                       "cross-national prompt of Durmus et al. (2023), 'How would someone from [country X] respond to this question?'")
    notes = ("One item per (question, country); sampling spreads items over distinct questions, one country each. "
             "The source does not record the survey year of Pew questions. Options keep the survey's order, "
             "including 'DK/Refused' where the survey reported it.")

    def load(self):
        import pandas as pd

        frame = pd.read_csv(hf_file("Anthropic/llm_global_opinions", "data/global_opinions.csv"))
        records = []
        for qid, row in frame.iterrows():
            options = [str(o) for o in ast.literal_eval(row["options"])]
            text = row["selections"]
            selections = ast.literal_eval(text[text.index("{"): text.rindex("}") + 1])
            if not 2 <= len(options) <= len(LETTERS):
                continue
            for country, values in selections.items():
                values = [float(v) for v in values]
                total = sum(values)
                if len(values) != len(options) or abs(total - 1.0) > 0.05 or min(values) < 0:
                    continue  # malformed distribution
                records.append({"qid": int(qid), "survey": row["source"], "question": row["question"], "options": options,
                                "country": country, "shares": [v / total for v in values]})
        return records

    def label_of(self, record):
        return str(record["qid"])  # more questions than items: one country from each sampled question

    def to_item(self, record, rng):
        country = record["country"]
        criteria = lettered(record["options"])
        question = Question(
            type="choice",
            instructions=f"How would someone from {country} respond to this survey question? Give the share of "
            f"respondents in {country} who chose each answer option.",
            criteria=criteria,
        )
        state = {"population": f"Adults in {country} surveyed by {SURVEYS[record['survey']]}", "question": record["question"]}
        gold = plurality_gold(dict(zip(criteria, record["shares"], strict=True)))
        return self.item(state, {"answer": question}, {"answer": gold}, survey=record["survey"], country=country, question_index=record["qid"])


# ---------- SubPOP-Eval: U.S. subpopulations (General Social Survey 2022) ----------

# Attribute names follow OpinionQA (Santurkar et al., 2023), which SubPOP adopts.
SUBPOP_ATTRIBUTES = {
    "CREGION": "region of the United States",
    "REGION": "region of the United States",
    "SEX": "sex",
    "GENDER": "gender",
    "AGE": "age",
    "EDUCATION": "education",
    "RACE": "race or ethnicity",
    "INCOME": "family income",
    "POLPARTY": "political party",
    "POLIDEOLOGY": "political ideology",
    "RELIG": "religion",
}


def _as_list(value):
    """A list field that SubPOP releases either as a list or as its Python string form."""
    return ast.literal_eval(value) if isinstance(value, str) else list(value)


@register_task
class SubPOPEval(TaskAdapter):
    name = f"{LEAF}/subpop_eval"
    leaf = LEAF
    source = "SubPOP-Eval: General Social Survey 2022 questions with answer shares of 22 U.S. subpopulations (Suh et al., 2025)"
    url = "https://huggingface.co/datasets/jjssuh/subpop"
    license = "CC BY-NC-SA 4.0 (gated; research or personal non-commercial use)"
    redistributable = False  # the gate's terms keep all rights with the survey centres; the repo must not become an ungated copy
    tier = "B"
    criteria_source = "GSS question wording and answer options as released in SubPOP; instruction adapted from the SubPOP / OpinionQA group-steering prompts"
    notes = ("Gated on Hugging Face (manual approval). One item per (question, subpopulation); gold = the group's "
             "post-stratification-weighted answer shares as released (refusals excluded). Data: Pew Research Center's American "
             "Trends Panel and the General Social Survey 1972-2022 (NORC). The opinions expressed herein, including any "
             "implications for policy, are those of the author and not of the survey research centers.")

    def load(self):
        with open(hf_file("jjssuh/subpop", "subpop_eval.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        records = []
        for row in rows:
            options = [str(o) for o in _as_list(row["options"])]  # released as strings such as "['Too little', ...]"
            values = [float(v) for v in _as_list(row["responses"])]
            if len(options) == len(values) + 1 and options[-1].strip().lower() == "refused":
                options = options[:-1]  # refusals are excluded from the released shares
            total = sum(values)
            if len(options) != len(values) or not 2 <= len(options) <= len(LETTERS) or total <= 0:
                continue
            records.append({"qkey": row["qkey"], "attribute": row["attribute"], "group": row["group"], "question": row["question"],
                            "options": options, "shares": [v / total for v in values]})
        return records

    def label_of(self, record):
        return f"{record['attribute']}|{record['group']}"

    def to_item(self, record, rng):
        attribute = SUBPOP_ATTRIBUTES.get(str(record["attribute"]).upper(), humanize(str(record["attribute"])))
        criteria = lettered(record["options"])
        population = f"U.S. adults in the 2022 General Social Survey whose {attribute} is: {record['group']}"
        question = Question(
            type="choice",
            instructions="How did people in this group answer the survey question? Give the share of the group's "
            "respondents (survey-weighted, refusals excluded) who chose each answer option.",
            criteria=criteria,
        )
        gold = plurality_gold(dict(zip(criteria, record["shares"], strict=True)))
        meta = {"qkey": record["qkey"], "attribute": record["attribute"], "group": record["group"]}
        return self.item({"population": population, "question": record["question"]}, {"answer": question}, {"answer": gold}, **meta)


# ---------- Polis open data: participants of civic deliberations ----------

POLIS_RAW = "https://raw.githubusercontent.com/compdemocracy/openData/master/"
# English-language conversations of the Polis open data (the Austrian and Taiwanese ones are not in English;
# london.youth.policing has 26 voters). Descriptions follow the repository README and case-study pages.
POLIS = {
    "15-per-hour-seattle": ("a Polis demonstration conversation on the move to a $15/hour minimum wage", "Seattle, Washington, USA"),
    "american-assembly.bowling-green": ("a conversation run by The American Assembly with residents of Bowling Green and Warren County", "Kentucky, USA"),
    "bg2050-volunteers": ("the 'Imagine Bowling Green in 2050' community conversation", "Bowling Green, Kentucky, USA"),
    "brexit-consensus": ("an experimental conversation on whether there can be consensus on Brexit", "United Kingdom"),
    "canadian-electoral-reform": ("a conversation on Canadian electoral reform", "Canada"),
    "football-concussions": ("a Polis demonstration conversation on concussions in American football", "USA"),
    "march-on.operation-marchin-orders": ("a conversation run by March On, a sister organization of the Women's March, on the movement's direction and platform", "USA"),
    "scoop-hivemind.affordable-housing": ("a Scoop HiveMind conversation on housing affordability", "New Zealand"),
    "scoop-hivemind.biodiversity": ("a Scoop and PEP HiveMind conversation on protecting and restoring biodiversity", "New Zealand"),
    "scoop-hivemind.freshwater": ("a Scoop HiveMind conversation on freshwater quality", "New Zealand"),
    "scoop-hivemind.taxes": ("a Scoop HiveMind conversation on tax", "New Zealand"),
    "scoop-hivemind.ubi": ("a Scoop HiveMind conversation on a universal basic income", "New Zealand"),
    "ssis.land-bank-farmland.2rumnecbeh.2021-08-01": ("a community discussion on land use, conservation and the county Land Bank", "San Juan Islands, Washington, USA"),
}
POLIS_OPTIONS = {"agree": "Agree", "disagree": "Disagree", "pass": "Pass / Unsure"}
POLIS_VOTE = {"1": "agree", "-1": "disagree", "0": "pass"}
POLIS_MIN_VOTES = 30


def _polis_rows(conversation: str, filename: str) -> list[dict[str, str]]:
    path = fetch(POLIS_RAW + f"{conversation}/{filename}", f"polis_{conversation}_{filename}")
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _year(timestamp: float) -> int:
    seconds = timestamp / 1000 if timestamp > 1e11 else timestamp  # exports use milliseconds or seconds
    return dt.datetime.fromtimestamp(seconds, dt.timezone.utc).year


def _years(stamps: list[float]) -> str:
    """Years in which the conversation was active (5th to 95th percentile of its votes)."""
    ordered = sorted(stamps)
    first, last = _year(ordered[len(ordered) // 20]), _year(ordered[(len(ordered) * 19) // 20])
    return str(first) if first == last else f"{first}-{last}"


@register_task
class PolisOpenData(TaskAdapter):
    name = f"{LEAF}/polis"
    leaf = LEAF
    source = "Polis open data: statements from 13 English-language public deliberations with participants' agree / disagree / pass votes (The Computational Democracy Project)"
    url = "https://github.com/compdemocracy/openData"
    license = "CC BY 4.0"
    tier = "B"
    criteria_source = "Polis voting options (Agree, Disagree, Pass / Unsure) verbatim; instruction written"
    sampling = "balanced"  # equal share per conversation, so the large conversations do not crowd out the others
    notes = (f"Statements accepted by the moderators with at least {POLIS_MIN_VOTES} voters; each voter's last vote counts. "
             "The population is the self-selected participants who voted on the statement, not a representative sample. "
             "Balanced across conversations.")

    def load(self):
        records = []
        for conversation, (description, place) in POLIS.items():
            summary = {}
            path = fetch(POLIS_RAW + f"{conversation}/summary.csv", f"polis_{conversation}_summary.csv")
            with path.open(encoding="utf-8", newline="") as f:
                for row in csv.reader(f):
                    if len(row) >= 2:
                        summary[row[0]] = row[1]
            latest: dict[tuple[str, str], tuple[float, str]] = {}
            for vote in _polis_rows(conversation, "votes.csv"):
                key = (vote["comment-id"], vote["voter-id"])
                stamp = float(vote["timestamp"])
                if key not in latest or stamp >= latest[key][0]:
                    latest[key] = (stamp, vote["vote"])
            counts: dict[str, dict[str, int]] = {}
            for (comment, _), (_, value) in latest.items():
                option = POLIS_VOTE.get(value.strip())
                if option:
                    counts.setdefault(comment, dict.fromkeys(POLIS_OPTIONS, 0))[option] += 1
            years = _years([stamp for stamp, _ in latest.values()])
            for comment in _polis_rows(conversation, "comments.csv"):
                cid = comment["comment-id"]
                tally = counts.get(cid)
                if comment["moderated"].strip() != "1" or not tally or sum(tally.values()) < POLIS_MIN_VOTES:
                    continue
                records.append({
                    "conversation": conversation, "description": description, "place": place,
                    "topic": summary.get("topic", "").strip(), "years": years,
                    "comment_id": cid, "statement": comment["comment-body"].strip(), "counts": tally,
                })
        return [r for r in records if r["statement"]]

    def label_of(self, record):
        return record["conversation"]

    def to_item(self, record, rng):
        population = f"People who took part in {record['description']} ({record['place']}, {record['years']}) and voted on this statement"
        question = Question(
            type="choice",
            instructions="In this Polis conversation participants wrote short statements and voted on statements written by "
            "others: Agree, Disagree, or Pass / Unsure. What share of the participants who voted on this statement chose each option?",
            criteria=shuffled_options(POLIS_OPTIONS, rng),
        )
        state = {"population": population, "conversation_topic": record["topic"], "statement": record["statement"]}
        gold = plurality_gold(record["counts"])
        meta = {"conversation": record["conversation"], "comment_id": record["comment_id"], "voters": sum(record["counts"].values())}
        return self.item(state, {"vote": question}, {"vote": gold}, **meta)


# ---------- Collective Constitutional AI: a representative U.S. sample voting on rules for an AI chatbot ----------

CCAI_RAW = "https://raw.githubusercontent.com/saffronh/ccai/main/"


@register_task
class CollectiveConstitutionalAI(TaskAdapter):
    name = f"{LEAF}/ccai_polis"
    leaf = LEAF
    source = "Collective Constitutional AI: Polis votes of about 1,000 U.S. adults on rules for an AI chatbot, July 2023 (Anthropic and The Collective Intelligence Project, 2023)"
    url = "https://github.com/saffronh/ccai"
    license = "Not stated"
    tier = "B"
    criteria_source = "Polis voting options (Agree, Disagree, Pass / Unsure) and the conversation prompt 'Help us pick rules for our AI Chatbot!' verbatim; instruction written"
    notes = ("License not stated: the repository has no license file, while Anthropic's announcement describes the data as "
             "openly licensed. "
             f"The authors' cleaned export (clean_comments.csv): 275 moderated-in statements with vote counts recomputed after "
             f"removing spam and low-effort participants; statements with at least {POLIS_MIN_VOTES} votes. Participants were "
             "recruited by PureSpectrum to be representative of U.S. adults by age, gender, income and geography and screened "
             "for familiarity with generative AI. Almost every statement has an Agree majority; the soft label carries the signal.")

    def load(self):
        path = fetch(CCAI_RAW + "clean_comments.csv", "ccai_clean_comments.csv")
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        records = []
        for row in rows:
            counts = {"agree": int(float(row["agrees"])), "disagree": int(float(row["disagrees"])), "pass": int(float(row["passes"]))}
            if row["moderated"].strip() == "1" and sum(counts.values()) >= POLIS_MIN_VOTES and row["comment-body"].strip():
                records.append({"comment_id": row["comment-id"], "statement": row["comment-body"].strip(), "counts": counts})
        return records

    def label_of(self, record):
        return max(record["counts"], key=record["counts"].get)

    def to_item(self, record, rng):
        population = ("About 1,000 U.S. adults recruited in July 2023 to be representative by age, gender, income and "
                      "geography (screened for familiarity with generative AI) who voted on this statement")
        question = Question(
            type="choice",
            instructions="In this Polis conversation participants wrote short statements and voted on statements written by "
            "others: Agree, Disagree, or Pass / Unsure. What share of the participants who voted on this statement chose each option?",
            criteria=shuffled_options(POLIS_OPTIONS, rng),
        )
        state = {"population": population, "conversation_topic": "Help us pick rules for our AI Chatbot!", "statement": record["statement"]}
        gold = plurality_gold(record["counts"])
        return self.item(state, {"vote": question}, {"vote": gold}, comment_id=record["comment_id"], voters=sum(record["counts"].values()))
