"""Leaf: finance decisions (金融).

Financial news sentiment from an investor's viewpoint (Financial PhraseBank), monetary-policy
stance of FOMC sentences (hawkish / dovish / neutral) and credit risk of loan applicants
(South German Credit, the corrected German credit data).
"""

from __future__ import annotations

import zipfile

from ...env import load_env
from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch, hf_rows

LEAF = "pro/finance"


# ---------- Financial PhraseBank: sentiment for an investor ----------

@register_task
class FinancialPhraseBank(TaskAdapter):
    name = f"{LEAF}/financial_phrasebank"
    leaf = LEAF
    source = "Financial PhraseBank v1.0, sentences with majority agreement (Malo et al., 2014)"
    url = "https://huggingface.co/datasets/takala/financial_phrasebank"
    license = "CC BY-NC-SA 3.0"
    tier = "A"
    criteria_source = "Annotation instructions verbatim from the Financial PhraseBank README (addressed to the model instead of the annotators)"
    notes = ("No official split; all 4,838 distinct sentences (majority label of 5-8 annotators with finance background) are eligible, "
             "sampled by label and agreement level (100%, >75%, >66%, >50%, kept in meta). About 2% of sentences contain garbled "
             "non-ASCII company names, as released.")
    FILES = (("all", "Sentences_AllAgree.txt"), ("75", "Sentences_75Agree.txt"), ("66", "Sentences_66Agree.txt"), ("50", "Sentences_50Agree.txt"))
    OPTIONS = {
        "positive": "positive: the news may have a positive influence on the stock price",
        "negative": "negative: the news may have a negative influence on the stock price",
        "neutral": "neutral: no positive or negative influence on the stock price, including sentiment that is not relevant from an economic or financial perspective",
    }

    def load(self):
        load_env()
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("takala/financial_phrasebank", "data/FinancialPhraseBank-v1.0.zip", repo_type="dataset")
        agreement, labels = {}, {}
        with zipfile.ZipFile(path) as zf:
            for level, name in self.FILES:  # from the strictest file down, so each sentence keeps its highest agreement level
                for line in zf.read(f"FinancialPhraseBank-v1.0/{name}").decode("latin-1").splitlines():
                    if "@" not in line:
                        continue
                    sentence, label = (x.strip() for x in line.rsplit("@", 1))
                    agreement.setdefault(sentence, level)
                    labels.setdefault(sentence, set()).add(label)
        return [{"sentence": s, "label": next(iter(l)), "agreement": agreement[s]} for s, l in labels.items() if len(l) == 1]

    def label_of(self, record):
        return f"{record['label']}|{record['agreement']}"

    def to_item(self, record, rng):
        question = Question(
            type="choice",
            instructions="Classify the sentence into a positive, negative or neutral category by considering only the information explicitly "
            "available in the given sentence. Consider the sentence from the view point of an investor only; i.e. whether the news may "
            "have positive, negative or neutral influence on the stock price. Sentences which have a sentiment that is not relevant from "
            "an economic or financial perspective are considered neutral.",
            criteria=shuffled_options(self.OPTIONS, rng),
        )
        return self.item({"sentence": record["sentence"]}, {"sentiment": question}, {"sentiment": Gold(label=record["label"])}, agreement=record["agreement"])


# ---------- FOMC: hawkish / dovish / neutral ----------

FOMC_GUIDE = (
    "Annotation guide, by category of the sentence. Economic status: dovish when inflation decreases, when unemployment increases, "
    "when economic growth is projected as low; hawkish when inflation increases, when unemployment decreases, when economic growth "
    "is projected high, when economic output is higher than potential supply/actual output, when economic slack falls; neutral when "
    "unemployment rate or growth is unchanged, maintained, or sustained. Dollar value change: dovish when the dollar appreciates; "
    "hawkish when the dollar depreciates. Energy/house prices: dovish when oil/energy prices decrease, when house prices decrease; "
    "hawkish when oil/energy prices increase, when house prices increase. Foreign nations: dovish when the US trade deficit "
    "decreases; hawkish when the US trade deficit increases; neutral when relating to a foreign nation's economic or trade policy. "
    "Fed expectations/actions/assets: dovish when the Fed expects subpar inflation, Fed expecting disinflation, narrowing spreads of "
    "treasury bonds, decreases in treasury security yields, and reduction of bank reserves; hawkish when the Fed expects high "
    "inflation, widening spreads of treasury bonds, increase in treasury security yields, increase in TIPS value, increase bank "
    "reserves. Money supply: dovish when money supply is low, M2 increases, increased demand for loans; hawkish when money supply "
    "is high, increased demand for goods, low demand for loans. Key words/phrases: dovish when the stance is \"accommodative\", "
    "indicating a focus on \"maximum employment\" and \"price stability\"; hawkish when indicating a focus on \"price stability\" "
    "and \"sustained growth\"; neutral with the use of phrases \"mixed\", \"moderate\", \"reaffirmed\". Labor: dovish when "
    "productivity increases; hawkish when productivity decreases."
)


@register_task
class FOMCHawkishDovish(TaskAdapter):
    name = f"{LEAF}/fomc_hawkish_dovish"
    leaf = LEAF
    source = "Trillion Dollar Words: FOMC minutes, speeches and press conferences, hawkish-dovish test split (Shah et al., 2023)"
    url = "https://huggingface.co/datasets/gtfintechlab/fomc_communication"
    license = "CC BY-NC 4.0"
    tier = "A"
    criteria_source = "Label definitions (Section 3) and the annotation guide (Appendix C.2, Table 10) of Shah et al. (2023), verbatim; the table is flattened into sentences"
    notes = "Sentences were split at contrasting connectives by the authors; labels by two independent annotators. The year of the sentence is not shown."
    OPTIONS = {
        "dovish": "Dovish: the sentence indicates future monetary policy easing.",
        "hawkish": "Hawkish: the sentence would indicate a future monetary policy tightening.",
        "neutral": "Neutral: mixed sentiment, indicating no change in the monetary policy, or not directly related to monetary policy stance.",
    }
    MAP = {0: "dovish", 1: "hawkish", 2: "neutral"}

    def load(self):  # 'index' and 'orig_index' repeat across the three document types, so the row number identifies an item
        return [dict(r, test_row=i) for i, r in enumerate(hf_rows("gtfintechlab/fomc_communication", None, "test"))]

    def label_of(self, record):
        return self.MAP[int(record["label"])]

    def to_item(self, record, rng):
        question = Question(
            type="choice",
            instructions="Classify the stance of this sentence from Federal Open Market Committee (FOMC) communications on monetary policy. " + FOMC_GUIDE,
            criteria=shuffled_options(self.OPTIONS, rng),
        )
        gold = Gold(label=self.MAP[int(record["label"])])
        return self.item({"sentence": record["sentence"].strip()}, {"stance": question}, {"stance": gold}, year=int(record["year"]), test_row=record["test_row"])


# ---------- South German Credit: credit risk ----------

# Code table of Groemping (2019), shipped with the data (codetable.txt); '...' replaced by the quantity it stands for.
SGC_FIELDS = (  # (column, state key, {code: text} or unit)
    ("laufkont", "checking_account", {"1": "no checking account", "2": "balance below 0 DM", "3": "balance from 0 to under 200 DM", "4": "balance of 200 DM or more, or salary paid into the account for at least 1 year"}),
    ("laufzeit", "duration", " months"),
    ("moral", "credit_history", {"0": "delay in paying off in the past", "1": "critical account/other credits elsewhere", "2": "no credits taken/all credits paid back duly", "3": "existing credits paid back duly till now", "4": "all credits at this bank paid back duly"}),
    ("verw", "purpose", {"0": "others", "1": "car (new)", "2": "car (used)", "3": "furniture/equipment", "4": "radio/television", "5": "domestic appliances", "6": "repairs", "7": "education", "8": "vacation", "9": "retraining", "10": "business"}),
    ("hoehe", "credit_amount", " DM"),
    ("sparkont", "savings", {"1": "unknown/no savings account", "2": "below 100 DM", "3": "100 to under 500 DM", "4": "500 to under 1000 DM", "5": "1000 DM or more"}),
    ("beszeit", "employment_duration", {"1": "unemployed", "2": "less than 1 year", "3": "1 to under 4 years", "4": "4 to under 7 years", "5": "7 years or more"}),
    ("rate", "installment_rate_percent_of_disposable_income", {"1": "35 or more", "2": "25 to under 35", "3": "20 to under 25", "4": "under 20"}),
    ("buerge", "other_debtors", {"1": "none", "2": "co-applicant", "3": "guarantor"}),
    ("wohnzeit", "years_at_present_residence", {"1": "less than 1 year", "2": "1 to under 4 years", "3": "4 to under 7 years", "4": "7 years or more"}),
    ("verm", "most_valuable_property", {"1": "unknown / no property", "2": "car or other", "3": "building society savings agreement/life insurance", "4": "real estate"}),
    ("alter", "age", " years"),
    ("weitkred", "other_installment_plans", {"1": "bank", "2": "stores", "3": "none"}),
    ("wohn", "housing", {"1": "for free", "2": "rent", "3": "own"}),
    ("bishkred", "number_of_credits_at_this_bank", {"1": "1", "2": "2-3", "3": "4-5", "4": "6 or more"}),
    ("beruf", "job", {"1": "unemployed/unskilled - non-resident", "2": "unskilled - resident", "3": "skilled employee/official", "4": "manager/self-employed/highly qualified employee"}),
    ("pers", "people_financially_dependent_on_applicant", {"1": "3 or more", "2": "0 to 2"}),
    ("telef", "telephone", {"1": "no", "2": "yes (under customer name)"}),
)


@register_task
class SouthGermanCredit(TaskAdapter):
    name = f"{LEAF}/south_german_credit"
    leaf = LEAF
    source = "South German Credit (UPDATE), 1,000 consumer credits 1973-1975 (Groemping, 2019; corrected Statlog German Credit)"
    url = "https://archive.ics.uci.edu/dataset/573/south+german+credit+update"
    license = "CC BY 4.0"
    tier = "B"
    criteria_source = "Outcome definition from the UCI variable description ('Has the credit contract been complied with (good) or not (bad)?'); attribute wording from the dataset's code table"
    notes = ("Gold is the recorded outcome of each credit. Bad credits are heavily oversampled (300 bad, 700 good) and there is no "
             "official split, so all 1,000 are eligible. The corrected codes of Groemping (2019) are used instead of the erroneous "
             "Statlog coding. Sex/marital status and foreign-worker status are withheld as protected attributes.")
    OPTIONS = {
        "good": "good: the credit contract was complied with",
        "bad": "bad: the credit contract was not complied with",
    }

    def load(self):
        path = fetch("https://archive.ics.uci.edu/static/public/573/south+german+credit+update.zip", "south_german_credit.zip")
        with zipfile.ZipFile(path) as zf:
            lines = zf.read("SouthGermanCredit.asc").decode("latin-1").split("\n")
        header = lines[0].split()
        return [dict(zip(header, line.split()), row=i) for i, line in enumerate(lines[1:], 1) if line.strip()]

    def label_of(self, record):
        return "good" if record["kredit"] == "1" else "bad"

    def to_item(self, record, rng):
        state = {}
        for column, key, coding in SGC_FIELDS:
            state[key] = coding[record[column]] if isinstance(coding, dict) else record[column] + coding
        question = Question(
            type="choice",
            instructions="A German bank granted this consumer credit in the 1970s. Based on the applicant and loan data, was the credit contract complied with (good credit risk) or not (bad credit risk)?",
            criteria=shuffled_options(self.OPTIONS, rng),
        )
        return self.item(state, {"credit_risk": question}, {"credit_risk": Gold(label=self.label_of(record))}, row=record["row"])
