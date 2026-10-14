"""Leaf: user satisfaction and frustration (用户满意度 / 挫败感)."""

from __future__ import annotations

import json

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch

LEAF = "dialog/user_satisfaction"

USS_URL = "https://raw.githubusercontent.com/sunnweiwei/user-satisfaction-simulation/master/dataset/"
USS_PARTS = ("MWOZ", "SGD", "ReDial", "CCPE")  # the English parts; JDDC (Chinese e-commerce) is left out
# Sun et al. (2021), Section 4.2, verbatim.
USS_LEVELS = [
    "Very dissatisfied (the system fails to understand and fulfill user's request)",
    "Dissatisfied (the system understands the request but fails to satisfy it in any way)",
    "Normal (the system understands users request and either partially satisfies the request or provides information on how the request can be fulfilled)",
    "Satisfied (the system understands and satisfies the user request, but provides more information than what the user requested or takes extra turns before meeting the request)",
    "Very satisfied (the system understands and satisfies the user request completely and efficiently)",
]


def main_score(ratings: list[int]) -> int:
    """The USS baselines' gold label: the most frequent rating, the lowest one on ties (0-based)."""
    counts = [ratings.count(level) for level in range(5)]
    return counts.index(max(counts))


@register_task
class USS(TaskAdapter):
    name = f"{LEAF}/uss"
    leaf = LEAF
    source = "User Satisfaction Simulation (USS), exchange-level ratings on MultiWOZ, SGD, ReDial and CCPE dialogues (Sun et al., 2021)"
    url = "https://github.com/sunnweiwei/user-satisfaction-simulation"
    license = "Not stated (source dialogues: MultiWOZ MIT, SGD CC BY-SA 4.0, ReDial CC BY 4.0, CCPE CC BY-SA 4.0)"
    tier = "A"
    criteria_source = "Five-level scale verbatim from the annotation instructions in Sun et al. (2021), Section 4.2"
    sampling = "balanced"
    notes = ("Three to seven crowd ratings per exchange; soft label = share of ratings per level, hard label = most frequent rating "
             "(lowest on ties), as in the USS baselines. Annotators rated satisfaction before the user's next utterance, so the input "
             "ends with a system turn and the user's reaction is withheld. Sampled evenly across hard levels because about 85% of "
             "exchanges are rated Normal; levels 1 and 5 are rare. The dataset has no official split (the paper uses 10-fold cross-validation).")

    def load(self):
        records = []
        for part in USS_PARTS:
            text = fetch(USS_URL + f"{part}.txt", f"uss_{part}.txt").read_text(encoding="utf-8")
            for s, session in enumerate(x for x in text.split("\n\n") if x.strip()):
                lines = [line.split("\t") for line in session.split("\n") if line.strip()]
                for i, line in enumerate(lines):
                    role, utterance, _, ratings = (line + ["", "", "", ""])[:4]
                    if role != "USER" or utterance == "OVERALL" or not ratings.strip() or i == 0 or lines[i - 1][0] != "SYSTEM":
                        continue
                    history = "\n".join(f"{'User' if r[0] == 'USER' else 'System'}: {r[1]}" for r in lines[:i])
                    records.append({"part": part, "session": s, "turn": i, "conversation": history,
                                    "ratings": [int(x) - 1 for x in ratings.split(",") if x.strip()]})
        return records

    def label_of(self, record):
        return str(main_score(record["ratings"]))

    def to_item(self, record, rng):
        question = Question(
            type="score",
            instructions="How satisfied is the user with the system at this point of the conversation? Rate user satisfaction "
            "based on the conversation so far, as it stands before the user's next utterance.",
            criteria=USS_LEVELS,
        )
        ratings = record["ratings"]
        dist = {str(level): ratings.count(level) / len(ratings) for level in range(5) if level in ratings}
        gold = Gold(label=str(main_score(ratings)), dist=dist)
        meta = {"part": record["part"], "session": record["session"], "turn": record["turn"], "raters": len(ratings)}
        return self.item({"conversation": record["conversation"]}, {"satisfaction": question}, {"satisfaction": gold}, **meta)


EMOWOZ = "https://zenodo.org/record/6506504/files/"
# EmoWOZ labelling scheme, Table 2 of Feng et al. (2022): label names and the "implication of user" column, verbatim.
EMOWOZ_LABELS = {
    0: ("neutral", "Neutral: describing situations and needs."),
    1: ("fearful", "Fearful, sad, disappointed: encountered a bad event (e.g. robbery)."),
    2: ("dissatisfied", "Dissatisfied, disliking: dissatisfied with the operator's suggestion or mistake."),
    3: ("apologetic", "Apologetic: apologising for causing confusion to the operator."),
    4: ("abusive", "Abusive: insulting the operator when the goal is not fulfilled."),
    5: ("excited", "Excited, happy, anticipating: looking forward to a good event (e.g. birthday party)."),
    6: ("satisfied", "Satisfied, liking, appreciative: satisfied with the operator because the goal is fulfilled."),
}


@register_task
class EmoWOZ(TaskAdapter):
    name = f"{LEAF}/emowoz"
    leaf = LEAF
    source = "EmoWOZ, test split: MultiWOZ and DialMAGE human-machine dialogues (Feng et al., 2022)"
    url = "https://zenodo.org/record/6506504"
    license = "CC BY-NC 4.0"
    tier = "A"
    criteria_source = "Emotion labels and their descriptions verbatim from the EmoWOZ labelling scheme (Feng et al., 2022, Table 2)"
    sampling = "balanced"
    notes = ("Each user utterance was labelled by three crowdworkers; soft label = their shares, hard label = the published final label. "
             "Sampled evenly across the seven labels because 70% of utterances are neutral and the frustration labels "
             "(dissatisfied, abusive) are rare; abusive has only a handful of test items.")

    def load(self):
        with fetch(EMOWOZ + "data-split.json", "emowoz_data_split.json").open(encoding="utf-8") as f:
            split = json.load(f)["test"]
        records = []
        for part, filename in (("multiwoz", "emowoz-multiwoz.json"), ("dialmage", "emowoz-dialmage.json")):
            with fetch(EMOWOZ + filename, f"emowoz_{part}.json").open(encoding="utf-8") as f:
                data = json.load(f)
            for dialogue_id in split[part]:
                log = data[dialogue_id]["log"]
                for i, turn in enumerate(log):
                    if i % 2 or not turn["emotion"]:
                        continue  # even turns are the user's
                    final = [e for e in turn["emotion"] if "emotion" in e]
                    votes = [e["annotation"] for e in turn["emotion"] if "annotation" in e]
                    if not final or final[0]["emotion"] not in EMOWOZ_LABELS:
                        continue
                    history = "\n".join(f"{'User' if j % 2 == 0 else 'System'}: {t['text']}" for j, t in enumerate(log[: i + 1]))
                    records.append({"part": part, "dialogue_id": dialogue_id, "turn": i, "conversation": history,
                                    "label": final[0]["emotion"], "votes": [v for v in votes if v in EMOWOZ_LABELS]})
        return records

    def label_of(self, record):
        return EMOWOZ_LABELS[record["label"]][0]

    def to_item(self, record, rng):
        options = {name: text for name, text in EMOWOZ_LABELS.values()}
        question = Question(
            type="choice",
            instructions="Which emotion does the user express in their last utterance, given the conversation so far?",
            criteria=shuffled_options(options, rng),
        )
        votes = record["votes"]
        dist = {EMOWOZ_LABELS[v][0]: votes.count(v) / len(votes) for v in sorted(set(votes))} if votes else None
        gold = Gold(label=EMOWOZ_LABELS[record["label"]][0], dist=dist)
        meta = {"part": record["part"], "dialogue_id": record["dialogue_id"], "turn": record["turn"]}
        return self.item({"conversation": record["conversation"]}, {"emotion": question}, {"emotion": gold}, **meta)
