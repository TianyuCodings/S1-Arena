"""Leaf: has the speaker finished their turn? (用户说完了吗 / 轮次结束检测)

Built from recorded turn boundaries in transcribed spoken dialogues. A turn is cut either at its
real end (the other speaker takes the floor next) or inside it; the text is lowercased and
punctuation is removed, like a streaming ASR transcript. Each turn yields both items.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch

LEAF = "dialog/end_of_turn"
CONTEXT_TURNS = 8
QUESTION = (
    "Below is a live transcript of a spoken conversation (lowercased, without punctuation). The current speaker is still holding "
    "the floor. Has the current speaker finished their turn, so that the other person should respond now? Answer false if the "
    "speaker is likely to continue talking."
)


def asr_words(text: str) -> list[str]:
    """Lowercased words without punctuation (apostrophes and word-internal hyphens kept)."""
    text = re.sub(r"[^\w'\- ]+", " ", text.lower().replace("_", " "))
    return [w.strip("-'") for w in text.split() if w.strip("-'")]


class EndOfTurn(TaskAdapter):
    tier = "C"
    criteria_source = "Written; gold is where the recorded turn actually ended"
    items_per_record = 2

    def label_of(self, record):
        return f"{record['bucket']}|{'unit' if record['inner_cuts'] else 'none'}"

    def to_items(self, record, rng):
        words = record["words"]
        avoid = set(record.get("avoid", []))  # positions that must not be used as cuts
        allowed = [k for k in range(1, len(words)) if k not in avoid]
        inner = [k for k in record["inner_cuts"] if k in allowed]
        question = Question(type="boolean", instructions=QUESTION)
        variants = [(words, True, "end")]
        if allowed:
            if inner and rng.random() < 0.6:  # a point where a sentence or dialogue-act unit ended but the speaker went on
                variants.append((words[: rng.choice(inner)], False, "unit_boundary"))
            else:
                variants.append((words[: rng.choice(allowed)], False, "random"))
        items = []
        for prefix, done, where in variants:
            state = {"conversation_so_far": record["context"], "current_speaker": record["speaker"], "current_turn_so_far": " ".join(prefix)}
            items.append(self.item(state, {"turn_finished": question}, {"turn_finished": Gold(label="true" if done else "false")},
                                   cut=where, words_shown=len(prefix), turn_words=len(words), conversation=record["conversation"]))
        return items if rng.random() < 0.5 else items[::-1]


def length_bucket(n: int) -> str:
    return "short" if n <= 6 else "medium" if n <= 15 else "long"


@register_task
class TaskmasterWOZ(EndOfTurn):
    name = f"{LEAF}/taskmaster1_woz"
    leaf = LEAF
    source = "Taskmaster-1 spoken two-person (Wizard-of-Oz) dialogues: user turns transcribed from speech (Byrne et al., 2019)"
    url = "https://github.com/google-research-datasets/Taskmaster/tree/master/TM-1-2019"
    license = "CC BY 4.0"
    notes = ("Crowdworkers spoke to what they believed was a voice assistant; their speech was transcribed, the operator's replies were typed. "
             "Only user turns of at least three words that are followed by the assistant are used. Consecutive user utterances form one "
             "turn and the boundaries between them are never used as cut points, since the user may have paused there. There is no "
             "official split for the spoken dialogues.")

    def load(self):
        url = "https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-1-2019/woz-dialogs.json"
        with fetch(url, "taskmaster1_woz-dialogs.json").open(encoding="utf-8") as f:
            dialogues = json.load(f)
        records = []
        for d in dialogues:
            turns = []  # (speaker, [utterance texts])
            for u in d["utterances"]:
                speaker = {"USER": "User", "ASSISTANT": "Assistant"}.get(u.get("speaker"))
                if not speaker or not u.get("text", "").strip():
                    continue
                if turns and turns[-1][0] == speaker:
                    turns[-1][1].append(u["text"])
                else:
                    turns.append((speaker, [u["text"]]))
            for i, (speaker, texts) in enumerate(turns[:-1]):
                if speaker != "User":
                    continue
                words, inner, avoid = [], [], []
                for t, text in enumerate(texts):
                    sentences = [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if asr_words(x)]
                    for j, sentence in enumerate(sentences):
                        words += asr_words(sentence)
                        if j < len(sentences) - 1:
                            inner.append(len(words))  # a sentence ended inside the utterance
                    if t < len(texts) - 1:
                        avoid.append(len(words))  # two user utterances meet: the user may have paused and finished here
                if len(words) < 3:
                    continue
                context = [f"{s}: {' '.join(asr_words(' '.join(x)))}" for s, x in turns[max(0, i - CONTEXT_TURNS):i]]
                records.append({"conversation": d["conversation_id"], "speaker": "User", "context": "\n".join(context),
                                "words": words, "inner_cuts": inner, "avoid": avoid, "bucket": length_bucket(len(words))})
        return records


# Switchboard test conversations of Stolcke et al. (2000), the usual SwDA test split.
SWDA_TEST = {2121, 2131, 2151, 2229, 2335, 2434, 2441, 2461, 2503, 2632, 2724, 2752, 2753, 2836, 2838, 3528, 3756, 3942, 3994}
BACKCHANNEL_ACTS = {"b", "bh", "bk"}  # continuer, backchannel question, response acknowledgement: they do not take the floor


def swda_clean(text: str) -> str:
    text = re.sub(r"<+[^<>]*>+", " ", text)  # <laughter>, <<long pause>>
    text = re.sub(r"\{[A-Z] ", " ", text).replace("}", " ")  # {F uh, } fillers and other markup: keep the words
    text = re.sub(r"\(\(|\)\)|\[|\]|\+|/|--|#|\*\S*", " ", text)
    return text


@register_task
class SwitchboardDA(EndOfTurn):
    name = f"{LEAF}/switchboard"
    leaf = LEAF
    source = "Switchboard Dialog Act Corpus (SwDA), test conversations of Stolcke et al. (2000) (Jurafsky et al., 1997)"
    url = "https://github.com/cgpotts/swda"
    license = "CC BY-NC-SA 3.0"
    notes = ("Telephone conversations between strangers. Backchannels (dialog acts b, bh, bk) and non-verbal units do not take the floor, "
             "so the speaker's turn continues across them; a turn ends when the other speaker produces anything else. Turns the other "
             "speaker interrupted (continued later with '+'), turns ending in an abandoned unit, and turns under three words are left "
             "out. Hard negatives cut where a dialog-act unit ended but the speaker kept talking. Disfluency markup is removed.")

    def load(self):
        path = fetch("https://github.com/cgpotts/swda/raw/master/swda.zip", "swda.zip")
        records = []
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if not name.endswith(".utt.csv") or name.startswith("__MACOSX"):
                    continue
                with zf.open(name) as f:
                    rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))
                if not rows or int(rows[0]["conversation_no"]) not in SWDA_TEST:
                    continue
                records += self.conversation(rows)
        return records

    def conversation(self, rows: list[dict]) -> list[dict]:
        units = []
        for r in rows:
            tag = re.split(r"[\^(@,;]", r["act_tag"].strip())[0]
            words = asr_words(swda_clean(r["text"]))
            if tag in BACKCHANNEL_ACTS or tag == "x" or not words:
                continue
            units.append({"caller": r["caller"], "tag": tag, "words": words})
        turns = []
        for u in units:
            if turns and turns[-1]["caller"] == u["caller"]:
                turns[-1]["units"].append(u)
            else:
                turns.append({"caller": u["caller"], "units": [u]})
        records = []
        for i, turn in enumerate(turns[:-1]):
            interrupted = i + 2 < len(turns) and turns[i + 2]["units"][0]["tag"] == "+"
            if interrupted or turn["units"][-1]["tag"] == "%":
                continue
            words, inner = [], []
            for u in turn["units"]:
                words += u["words"]
                inner.append(len(words))
            if len(words) < 3:
                continue
            context = [f"Speaker {t['caller']}: {' '.join(w for u in t['units'] for w in u['words'])}" for t in turns[max(0, i - CONTEXT_TURNS):i]]
            records.append({"conversation": rows[0]["conversation_no"], "speaker": f"Speaker {turn['caller']}", "context": "\n".join(context),
                            "words": words, "inner_cuts": inner[:-1], "bucket": length_bucket(len(words))})
        return records
