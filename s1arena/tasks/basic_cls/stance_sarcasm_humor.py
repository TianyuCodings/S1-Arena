"""Leaf: stance, sarcasm and humor (立场 / 讽刺 / 幽默)."""

from __future__ import annotations

import csv
import io
import re
import zipfile
from collections import Counter

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch, hf_rows

LEAF = "basic_cls/stance_sarcasm_humor"


@register_task
class SemEval2016Stance(TaskAdapter):
    name = f"{LEAF}/semeval2016_stance"
    leaf = LEAF
    source = "SemEval-2016 Task 6 stance detection (Mohammad et al., 2016), via TweetEval"
    url = "http://www.saifmohammad.com/WebPages/StanceDataset.htm"
    license = "Research use only; do not redistribute (NRC)"
    redistributable = False
    tier = "B"
    criteria_source = "Question and option definitions verbatim from the NRC raw-annotation README; NEUTRAL merged into NONE as in the official data"

    TARGETS = {
        "stance_abortion": "Legalization of Abortion",
        "stance_atheism": "Atheism",
        "stance_climate": "Climate Change is a Real Concern",
        "stance_feminist": "Feminist Movement",
        "stance_hillary": "Hillary Clinton",
    }
    OPTIONS = {
        "favor": "We can infer from the tweet that the tweeter supports the target (e.g., directly or indirectly by "
        "supporting someone/something, by opposing or criticizing someone/something opposed to the target, or by "
        "echoing the stance of somebody else)",
        "against": "We can infer from the tweet that the tweeter is against the target (e.g., directly or indirectly by "
        "opposing or criticizing someone/something, by supporting someone/something opposed to the target, or by "
        "echoing the stance of somebody else)",
        "none": "We can infer from the tweet that the tweeter has a neutral stance towards the target (e.g., the tweeter "
        "supports the target entity to some extent, but is also against it to some extent), or none of the above.",
    }

    def load(self):
        rows = []
        for config, target in self.TARGETS.items():
            rows += [dict(r, target=target) for r in hf_rows("cardiffnlp/tweet_eval", config, "test")]
        return rows

    def label_of(self, record):
        return f"{record['target']}|{record['label_name']}"

    def to_item(self, record, rng):
        question = Question(
            type="choice",
            instructions="From reading the tweet, which of the options below is most likely to be true about the tweeter's "
            f"stance or outlook towards the target: {record['target']}?",
            criteria=shuffled_options(self.OPTIONS, rng),
        )
        state = {"target": record["target"], "tweet": record["text"]}
        return self.item(state, {"stance": question}, {"stance": Gold(label=record["label_name"])})


@register_task
class ISarcasmEval(TaskAdapter):
    name = f"{LEAF}/isarcasmeval"
    leaf = LEAF
    source = "iSarcasmEval subtask A, English (Abu Farha et al., 2022)"
    url = "https://github.com/iabufarha/iSarcasmEval"
    license = "MIT"
    tier = "B"
    criteria_source = "Sarcasm definition from the task overview paper (Abu Farha et al., 2022); labels come from the tweets' authors"

    def load(self):
        path = fetch("https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/test/task_A_En_test.csv", "isarcasmeval_task_A_En_test.csv")
        with path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def label_of(self, record):
        return record["sarcastic"]

    def to_item(self, record, rng):
        question = Question(
            type="boolean",
            instructions="Did the author intend this tweet to be sarcastic? Sarcasm is a form of verbal irony that occurs when "
            "there is a discrepancy between the literal and intended meanings of an utterance.",
        )
        gold = Gold(label="true" if record["sarcastic"].strip() == "1" else "false")
        return self.item({"tweet": record["text"]}, {"sarcastic": question}, {"sarcastic": gold})


@register_task
class Humicroedit(TaskAdapter):
    name = f"{LEAF}/humicroedit"
    leaf = LEAF
    source = "Humicroedit, SemEval-2020 Task 7 subtask 1 (Hossain et al., 2020)"
    url = "https://www.cs.rochester.edu/u/nhossain/humicroedit.html"
    license = "Not stated (research data)"
    redistributable = False
    tier = "A"
    criteria_source = "Task wording and 0-3 scale names verbatim from Hossain et al. (2020)"

    LEVELS = ["0: Not funny", "1: Slightly funny", "2: Moderately funny", "3: Funny"]

    def load(self):
        path = fetch("https://www.cs.rochester.edu/u/nhossain/semeval-2020-task-7-dataset.zip", "humicroedit.zip")
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.endswith("subtask-1/test.csv") and "__MACOSX" not in n)
            text = zf.read(name).decode("utf-8")
        rows = []
        for r in csv.DictReader(io.StringIO(text)):
            grades = r["grades"].strip()
            width = 5 if len(grades) <= 5 else 10
            grades = grades.zfill(width)
            if abs(sum(map(int, grades)) / len(grades) - float(r["meanGrade"])) > 0.01:
                continue  # could not recover the individual grades
            rows.append(dict(r, grades=grades))
        return rows

    def label_of(self, record):
        return str(round(float(record["meanGrade"])))

    def to_item(self, record, rng):
        original = record["original"]
        match = re.search(r"<([^/>]*)/>", original)
        if not match:
            return None
        counts = Counter(int(c) for c in record["grades"])
        total = sum(counts.values())
        dist = {str(level): counts.get(level, 0) / total for level in range(4)}
        state = {
            "original_headline": original.replace(match.group(0), match.group(1)),
            "edited_headline": original.replace(match.group(0), record["edit"]),
        }
        question = Question(type="score", instructions="Estimate the funniness of the edited headline on a 0-3 humor scale.", criteria=self.LEVELS)
        return self.item(state, {"funniness": question}, {"funniness": Gold(dist=dist)}, original_id=record["id"], mean_grade=float(record["meanGrade"]))
