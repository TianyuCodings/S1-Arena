"""Leaf: persuasiveness and funniness (说服力 / 幽默程度).

The target is the distribution of several people's ratings: New Yorker caption-contest
voters, five pun annotators, the three participants who read each argument in
Anthropic's persuasion study, and five crowd workers per argument pair in UKPConvArg1.
Ordinal scales use the median level as the hard label.
"""

from __future__ import annotations

import csv
import re
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

from ...env import load_env
from ...registry import register_task
from ...schema import Question
from ..base import TaskAdapter
from ..common import fetch
from ._common import median_gold, ordinal_gold, plurality_gold

LEAF = "social/persuasion_humor"


def hf_file(repo: str, filename: str) -> str:
    load_env()
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset")


# ---------- New Yorker Cartoon Caption Contest: crowd ratings of captions ----------

NYCC_LEVELS = ["Unfunny", "Somewhat funny", "Funny"]
NYCC_MIN_VOTES = 40
NYCC_BINS = (1.2, 1.4, 1.6)  # mean-score bins for sampling (1 = unfunny ... 3 = funny)


def _wiki_name(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1].replace("_", " ")


def _caption_ok(caption: str) -> bool:
    words = re.findall(r"[A-Za-z]{2,}", caption)
    odd = sum(1 for ch in caption if ch == "?" or ord(ch) > 0x2FFF)
    return len(words) >= 2 and odd <= max(3, len(caption) // 10)


@register_task
class NewYorkerCaptions(TaskAdapter):
    name = f"{LEAF}/nycc_captions"
    leaf = LEAF
    source = ("New Yorker Cartoon Caption Contest crowd ratings (Jain et al., 2020; Zhang et al., 2024), "
              "cartoons described by the human-written annotations of Hessel et al. (2023), ranking test split")
    url = "https://huggingface.co/datasets/yguooo/newyorker_caption_ranking"
    license = "CC BY-NC 4.0 (ratings); CC BY 4.0 (cartoon descriptions)"
    tier = "B"
    criteria_source = "Rating options of the contest's voting page (Unfunny / Somewhat Funny / Funny) as described by Zhang et al. (2024); instruction written"
    sampling = "balanced"  # equal share per band of mean rating, so funny captions are not swamped by the many unfunny ones
    notes = ("Cartoons are given as text: location, literal description, what is unusual and related Wikipedia entities, "
             "written by annotators for Hessel et al. (2023); contests of their ranking test split that have vote data. "
             f"Captions with at least {NYCC_MIN_VOTES} votes; contests whose export is degenerate (mostly unanimous "
             "'unfunny' or a median under 25 votes) are skipped. Balanced over four bands of mean rating.")

    def cartoons(self) -> dict[int, dict]:
        import pyarrow.parquet as pq

        columns = ["contest_number", "instance_id", "image_location", "image_description", "image_uncanny_description", "entities"]
        table = pq.read_table(hf_file("jmhessel/newyorker_caption_contest", "ranking/test-00000-of-00001.parquet"), columns=columns)
        out: dict[int, dict] = {}
        for row in sorted(table.to_pylist(), key=lambda r: (r["contest_number"], r["instance_id"])):
            out.setdefault(int(row["contest_number"]), row)  # descriptions repeat for every caption pair of a contest
        return out

    def load(self):
        load_env()
        from huggingface_hub import HfApi

        available = {s.rfilename for s in HfApi().dataset_info("yguooo/newyorker_caption_ranking").siblings}
        records = []
        for contest, cartoon in self.cartoons().items():
            filename = f"ranking/source/{contest}.csv"
            if filename not in available:
                continue
            with open(hf_file("yguooo/newyorker_caption_ranking", filename), encoding="utf-8", errors="replace", newline="") as f:
                rows = list(csv.DictReader(f))
            votes = sorted(int(float(r["votes"])) for r in rows)
            unanimous = sum(1 for r in rows if int(float(r["not_funny"])) == int(float(r["votes"]))) / len(rows)
            if unanimous > 0.2 or votes[len(votes) // 2] < 25:
                continue  # degenerate export for this contest
            description = {
                "location": cartoon["image_location"].strip(),
                "description": cartoon["image_description"].strip(),
                "uncanny": cartoon["image_uncanny_description"].strip(),
                "entities": ", ".join(_wiki_name(e) for e in cartoon["entities"] or []),
            }
            for r in rows:
                counts = [int(float(r["not_funny"])), int(float(r["somewhat_funny"])), int(float(r["funny"]))]
                caption = " ".join(r["caption"].split())
                if sum(counts) < NYCC_MIN_VOTES or not _caption_ok(caption):
                    continue
                records.append({"contest": contest, "cartoon": description, "caption": caption, "counts": counts})
        return records

    def label_of(self, record):
        mean = sum((i + 1) * c for i, c in enumerate(record["counts"])) / sum(record["counts"])
        return str(sum(mean >= edge for edge in NYCC_BINS))

    def to_item(self, record, rng):
        cartoon = record["cartoon"]
        state = {
            "cartoon_setting": cartoon["location"],
            "cartoon_description": cartoon["description"],
            "what_is_unusual": cartoon["uncanny"],
            "related_concepts": cartoon["entities"],
            "caption": record["caption"],
        }
        question = Question(
            type="score",
            instructions="The New Yorker's weekly cartoon caption contest lets readers rate submitted captions on its voting "
            "page as Unfunny, Somewhat Funny or Funny. The cartoon is described in words. Give the share of voters who "
            "gave this caption each rating.",
            criteria=list(NYCC_LEVELS),
        )
        gold = ordinal_gold(record["counts"])
        return self.item(state, {"funniness": question}, {"funniness": gold}, contest=record["contest"], votes=sum(record["counts"]))


# ---------- ExPUN: five annotators' funniness ratings of puns ----------

EXPUN_URL = "https://raw.githubusercontent.com/amazon-science/expunations/main/data/expunations_annotated_full.json"
SEMEVAL_PUNS = "https://alt.qcri.org/semeval2017/task7/data/uploads/semeval2017_task7.tar.xz"
# One scale for the ExPUN annotation flow (Sun et al., 2022, Appendix A.1): an annotator who did not understand the
# text, found it offensive, or judged it not to be a joke gave no funniness rating (recorded as 0).
EXPUN_LEVELS = [
    "0: no funniness rating: the annotator did not understand the text, found it offensive or inappropriate, or judged that it is not intended to be a joke",
    "1: very not funny: a joke that is not funny at all, or tries to be funny but does not achieve the intended effect",
    "2: between 1 and 3",
    "3: an average joke that may elicit some chuckles (or groans)",
    "4: between 3 and 5",
    "5: very funny: a good joke that you find humorous and potentially would want to share or tell to others",
]


def _attach_quotes(text: str, mark: str) -> str:
    """Attach free-standing quote marks to their neighbours, alternating opening and closing."""
    out, opening, glue_next = [], True, False
    for token in text.split(" "):
        if re.fullmatch(re.escape(mark) + r"[.,!?;:…]*", token):  # a quote mark, possibly followed by punctuation
            if opening:
                out.append(token)
                glue_next = True
            elif out:
                out[-1] += token
            else:
                out.append(token)
            opening = not opening
        elif glue_next and out:
            out[-1] += token  # the first word after an opening quote
            glue_next = False
        else:
            out.append(token)
    return " ".join(out)


def detokenize(tokens: list[str]) -> str:
    """Rejoin the SemEval-2017 pun tokens ("' ' I ' m out . ' '", "Wal - Mart") into running text."""
    text = " ".join(t.strip() for t in tokens if t and t.strip())
    text = re.sub(r"(^|\s)'\s'(?=\s|$)", r'\1"', text)  # LaTeX-style double quotes, tokenized as ' '
    text = re.sub(r"\s+'\s*(s|t|re|ve|ll|d|m|S|T|RE|VE|LL|D|M)\b", r"'\1", text)  # contractions and possessives
    text = re.sub(r"(?<=\w)\s+-\s+-\s+(?=\w)", " -- ", text)
    text = re.sub(r"(?<=\w)\s+-\s+(?=\w)", "-", text)
    text = re.sub(r"\s+([.,!?;:%…])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = _attach_quotes(_attach_quotes(text, '"'), "'")
    return text.strip()


@register_task
class ExPUN(TaskAdapter):
    name = f"{LEAF}/expun"
    leaf = LEAF
    source = "ExPUN: puns and non-puns from SemEval-2017 Task 7, each rated by five annotators (Sun et al., 2022)"
    url = "https://github.com/amazon-science/expunations"
    license = "CC BY-NC 4.0 (annotations); pun texts CC BY 4.0 or CC BY-NC 4.0 (SemEval-2017 Task 7)"
    tier = "B"
    criteria_source = "Funniness scale and anchor descriptions verbatim from the ExPUN annotation guidelines (Sun et al., 2022, Appendix A.1), with the guidelines' skip conditions folded into level 0"
    notes = ("Soft label = the five annotators' ratings; hard label = the median. Texts are rebuilt from the SemEval-2017 "
             "Task 7 test tokens. Funniness is low overall (mean 1.68 among rated texts).")

    def load(self):
        import json

        annotations = json.loads(fetch(EXPUN_URL, "expunations_annotated_full.json").read_text(encoding="utf-8"))
        texts = {}
        with tarfile.open(fetch(SEMEVAL_PUNS, "semeval2017_task7.tar.xz"), "r:xz") as archive:
            for kind in ("homographic", "heterographic"):
                data = archive.extractfile(f"semeval2017_task7/data/test/subtask1-{kind}-test.xml").read()
                for text in ET.fromstring(data):
                    texts[text.get("id")] = detokenize([w.text or "" for w in text])
        records = []
        for row in annotations:
            ratings = [int(x) for x in row["Funniness (1-5)"] if str(x).strip() != ""]
            if row["ID"] in texts and len(ratings) >= 3 and all(0 <= x <= 5 for x in ratings):
                records.append({"id": row["ID"], "text": texts[row["ID"]], "ratings": ratings})
        return records

    def label_of(self, record):
        return median_gold(record["ratings"], len(EXPUN_LEVELS)).label

    def to_item(self, record, rng):
        question = Question(
            type="score",
            instructions="Annotators read the short text, said whether they understood it, whether it is offensive or "
            "inappropriate, and whether it is intended to be a joke, and then rated how funny the joke is from 1 (very not "
            "funny) to 5 (very funny). Give the share of annotators who gave each rating.",
            criteria=list(EXPUN_LEVELS),
        )
        gold = median_gold(record["ratings"], len(EXPUN_LEVELS))
        return self.item({"text": record["text"]}, {"funniness": question}, {"funniness": gold}, original_id=record["id"], pun_type=record["id"].split("_")[0])


# ---------- Anthropic persuasion study: support shifts of three readers per argument ----------

PERSUASION_LEVELS = [
    "Support decreased (the final rating is lower than the initial rating)",
    "No change",
    "Support increased by 1 point",
    "Support increased by 2 points",
    "Support increased by 3 or more points",
]


def _rating(text: str) -> int:
    return int(str(text).split("-", 1)[0].strip())


def _shift_level(shift: int) -> int:
    return 0 if shift < 0 else min(shift + 1, len(PERSUASION_LEVELS) - 1)


@register_task
class AnthropicPersuasion(TaskAdapter):
    name = f"{LEAF}/anthropic_persuasion"
    leaf = LEAF
    source = "Persuasion dataset: claims with human- and Claude-written arguments, each read by three participants (Durmus et al., 2024)"
    url = "https://huggingface.co/datasets/Anthropic/persuasion"
    license = "CC BY-NC-SA 4.0"
    tier = "B"
    criteria_source = "7-point support scale labels from the dataset; persuasiveness = final minus initial support (Durmus et al., 2024), binned; instruction written"
    notes = ("Soft label = the three participants' support shifts; hard label = the median shift. Arguments were written "
             "by people or by Claude models (the ratings are always human); the control condition (arguments against "
             "indisputable facts) is left out.")

    def load(self):
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        with open(hf_file("Anthropic/persuasion", "persuasion_data.csv"), encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["source"] != "Control":
                    groups[(row["claim"].strip(), row["argument"].strip())].append(row)
        records = []
        for (claim, argument), rows in groups.items():
            shifts = [_rating(r["rating_final"]) - _rating(r["rating_initial"]) for r in rows]
            if len(rows) >= 3 and claim and argument:
                records.append({"claim": claim, "argument": argument, "source": rows[0]["source"], "prompt_type": rows[0]["prompt_type"],
                                "levels": sorted(_shift_level(s) for s in shifts)})
        return records

    def label_of(self, record):
        return f"{'human' if record['source'] == 'Human' else 'model'}|{record['levels'][(len(record['levels']) - 1) // 2]}"

    def to_item(self, record, rng):
        question = Question(
            type="score",
            instructions="Participants rated how much they support the claim on a 7-point scale (1 - Strongly oppose, "
            "2 - Oppose, 3 - Somewhat oppose, 4 - Neither oppose nor support, 5 - Somewhat support, 6 - Support, "
            "7 - Strongly support), then read the argument and rated their support again. Give the share of participants "
            "whose support changed by each amount (final minus initial rating).",
            criteria=list(PERSUASION_LEVELS),
        )
        gold = median_gold(record["levels"], len(PERSUASION_LEVELS))
        meta = {"argument_source": record["source"], "prompt_type": record["prompt_type"], "participants": len(record["levels"])}
        return self.item({"claim": record["claim"], "argument": record["argument"]}, {"support_change": question}, {"support_change": gold}, **meta)


# ---------- UKPConvArg1: which of two arguments is more convincing ----------

UKP_RAW = "https://raw.githubusercontent.com/UKPLab/acl2016-convincing-arguments/master/data/UKPConvArg1-full-XML/"
UKP_FILES = [
    "ban-plastic-water-bottles_no-bad-for-the-economy", "ban-plastic-water-bottles_yes-emergencies-only",
    "christianity-or-atheism-_atheism", "christianity-or-atheism-_christianity",
    "evolution-vs-creation_creation", "evolution-vs-creation_evolution",
    "firefox-vs-internet-explorer_it-has-a-cute-logo-oh-and-extensions-err-add-ons",
    "firefox-vs-internet-explorer_there-s-more-browsers-than-the-ie-firefox-is-an-animal",
    "gay-marriage-right-or-wrong_allowing-gay-marriage-is-right", "gay-marriage-right-or-wrong_allowing-gay-marriage-is-wrong",
    "human-growth-and-development-should-parents-use-spanking-as-an-option-to-discipline-_no",
    "human-growth-and-development-should-parents-use-spanking-as-an-option-to-discipline-_yes",
    "if-your-spouse-committed-murder-and-he-or-she-confided-in-you-would-you-turn-them-in-_no",
    "if-your-spouse-committed-murder-and-he-or-she-confided-in-you-would-you-turn-them-in-_yes",
    "india-has-the-potential-to-lead-the-world-_no-against", "india-has-the-potential-to-lead-the-world-_yes-for",
    "is-it-better-to-have-a-lousy-father-or-to-be-fatherless-_fatherless",
    "is-it-better-to-have-a-lousy-father-or-to-be-fatherless-_lousy-father",
    "is-porn-wrong-_no-is-is-not", "is-porn-wrong-_yes-porn-is-wrong",
    "is-the-school-uniform-a-good-or-bad-idea-_bad", "is-the-school-uniform-a-good-or-bad-idea-_good",
    "personal-pursuit-or-advancing-the-common-good-_advancing-the-comnong-good",
    "personal-pursuit-or-advancing-the-common-good-_personal-pursuit",
    "pro-choice-vs-pro-life_pro-choice", "pro-choice-vs-pro-life_pro-life",
    "should-physical-education-be-mandatory-in-schools-_no-", "should-physical-education-be-mandatory-in-schools-_yes-",
    "tv-is-better-than-books_books", "tv-is-better-than-books_tv",
    "william-farquhar-ought-to-be-honoured-as-the-rightful-founder-of-singapore_no-it-is-raffles-",
    "william-farquhar-ought-to-be-honoured-as-the-rightful-founder-of-singapore_yes-of-course-",
]
# The three answers of the HIT, verbatim from Habernal and Gurevych (2016), Section 3.2, with A1/A2 as A/B.
UKP_OPTIONS = {
    "a": "Argument A is more convincing than argument B (A > B)",
    "b": "Argument A is less convincing than argument B (A < B)",
    "equal": "Argument A and argument B are convincing equally (A = B)",
}
UKP_VALUES = {"a1": "a", "a2": "b", "equal": "equal"}


@register_task
class UKPConvArg(TaskAdapter):
    name = f"{LEAF}/ukpconvarg1"
    leaf = LEAF
    source = "UKPConvArg1, argument pairs from createdebate.com and convinceme.net with five crowd judgments each (Habernal and Gurevych, 2016)"
    url = "https://github.com/UKPLab/acl2016-convincing-arguments"
    license = "CC BY 4.0"
    tier = "A"
    criteria_source = "The HIT's three answers and instructions (do not judge the truth of the proposition; be objective; do not express your position) from Habernal and Gurevych (2016), Section 3.2"
    items_per_record = 2  # each pair in both orders
    notes = ("Full corpus (all 32 debate sides, 'equal' votes kept). Soft label = the workers' votes (five per pair, four for "
             "1% of pairs); pairs whose votes have no single most common answer are left out. Each pair is asked in both orders.")

    def load(self):
        records = []
        for topic in UKP_FILES:
            root = ET.parse(fetch(UKP_RAW + f"{topic}.xml", f"ukpconvarg1_{topic}.xml")).getroot()
            for pair in root:
                votes = Counter(UKP_VALUES[v] for v in (a.findtext("value") for a in pair.find("mTurkAssignments")) if v in UKP_VALUES)
                if sum(votes.values()) < 4:
                    continue
                ranked = votes.most_common()
                if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                    continue  # no single most common answer
                meta = pair.find("debateMetaData")
                records.append({
                    "topic": topic, "pair_id": pair.findtext("id"),
                    "debate": " ".join((meta.findtext("title") or "").split()),
                    "stance": " ".join((pair.find("arg1").findtext("stance") or "").split()),
                    "arg1": (pair.find("arg1").findtext("text") or "").strip(),
                    "arg2": (pair.find("arg2").findtext("text") or "").strip(),
                    "votes": {key: votes.get(key, 0) for key in UKP_OPTIONS},
                })
        return [r for r in records if r["arg1"] and r["arg2"]]

    def label_of(self, record):
        return f"{record['topic']}|{max(record['votes'], key=record['votes'].get)}"

    def to_items(self, record, rng):
        items = []
        for order, (first, second) in (("original", ("arg1", "arg2")), ("swapped", ("arg2", "arg1"))):
            votes = record["votes"] if order == "original" else {"a": record["votes"]["b"], "b": record["votes"]["a"], "equal": record["votes"]["equal"]}
            question = Question(
                type="choice",
                instructions="Both arguments take the same stance in the debate. Which argument is more convincing? Do not "
                "judge the truth of the proposition, be objective, and leave your own position aside. Crowd workers "
                "compared the two arguments; give the share of workers who chose each answer.",
                criteria=dict(UKP_OPTIONS),
            )
            state = {"debate": record["debate"], "stance": record["stance"], "argument_a": record[first], "argument_b": record[second]}
            items.append(self.item(state, {"more_convincing": question}, {"more_convincing": plurality_gold(votes)},
                                   order=order, topic=record["topic"], original_id=record["pair_id"]))
        return items
