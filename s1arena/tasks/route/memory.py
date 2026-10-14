"""Leaf: memory write and retrieval decisions (记忆写入 / 检索决策).

Write: should the latest dialogue turn be stored in long-term memory? Gold is whether
Multi-Session Chat crowdworkers wrote a memory summary for that turn.
Retrieve: should the next reply draw on stored memories about the user? Gold is whether the
DuLeMon crowdworker playing the bot used one of the user memories in that reply.
"""

from __future__ import annotations

import json
import re
import tarfile
import zipfile

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch

LEAF = "route/memory_decisions"
SPEAKERS = {"bot_0": "Speaker A", "bot_1": "Speaker B"}


@register_task
class MSCMemoryWrite(TaskAdapter):
    name = f"{LEAF}/msc_write"
    leaf = LEAF
    source = "Multi-Session Chat, per-turn memory summaries, test split (Xu et al., 2022)"
    url = "https://parl.ai/projects/msc/"
    license = "Not stated (data); ParlAI code MIT"
    tier = "B"
    criteria_source = ("Adapted from the MSC memory-writing step: 'summarizing any new pertinent information contained in the last "
                       "dialogue turn. This includes the case of deciding that no new information should be stored in the memory.'")
    notes = ("Sessions 1-4 of the persona-summary data; gold = a crowdworker wrote a summary for the turn (about 33% of turns have "
             "none). The memory shown is what was summarized for the same speaker earlier in the session.")

    def load(self):
        rows = []
        with tarfile.open(fetch("https://dl.fbaipublicfiles.com/parlai/msc/msc_v0.1.tar.gz", "msc_v0.1.tar.gz")) as tar:
            for session in (1, 2, 3, 4):
                lines = tar.extractfile(f"msc/msc_personasummary/session_{session}/test.txt").read().decode("utf-8").splitlines()
                for d, line in enumerate(lines):
                    dialog = json.loads(line)["dialog"]
                    memory = {"bot_0": [], "bot_1": []}
                    for t, turn in enumerate(dialog):
                        speaker = turn["id"]
                        if speaker not in memory:
                            break
                        rows.append({"session": session, "dialog": d, "turn": t, "speaker": SPEAKERS[speaker],
                                     "history": [f"{SPEAKERS[x['id']]}: {x['text'].strip()}" for x in dialog[:t]],
                                     "latest": turn["text"].strip(), "memory": list(memory[speaker]),
                                     "label": "true" if (turn.get("persona_text") or "").strip() else "false"})
                        memory[speaker] = list(turn.get("agg_persona_list") or memory[speaker])
        return rows

    def label_of(self, record):
        return f"{record['label']}|{record['session']}"

    def to_item(self, record, rng):
        state = {"conversation_so_far": "\n".join(record["history"]) or "(this is the first turn)",
                 "memory_of_latest_speaker": "\n".join(f"- {m}" for m in record["memory"]) or "(empty)",
                 "latest_turn": f"{record['speaker']}: {record['latest']}"}
        instructions = (f"A dialogue agent keeps a long-term memory of what each speaker reveals about themselves, for use in later "
                        f"conversations. Should the latest turn be written to {record['speaker']}'s memory? Answer true if it contains "
                        "new pertinent information about the speaker (facts, preferences, plans, experiences) that is not yet in the "
                        "memory; answer false if no new information should be stored.")
        return self.item(state, {"write_to_memory": Question(type="boolean", instructions=instructions)},
                         {"write_to_memory": Gold(label=record["label"])}, session=record["session"], turn=record["turn"])


CJK_GAP = re.compile(r"(?<=[^\x00-\x7F])\s+(?=[^\x00-\x7F])")


def _detok(text: str) -> str:
    """DuLeMon is word-segmented with spaces; remove the spaces between Chinese characters."""
    return CJK_GAP.sub("", text.strip())


@register_task
class DuLeMonRetrieve(TaskAdapter):
    name = f"{LEAF}/dulemon_retrieve"
    leaf = LEAF
    source = "DuLeMon, Chinese long-term persona memory dialogues, 'both' test split (Xu et al., 2022)"
    url = "https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2022-DuLeMon"
    license = "Not stated (data); code Apache-2.0"
    tier = "C"
    criteria_source = "Written; gold is the crowdworker's own tag of which stored user memory the bot's reply used"
    notes = ("Chinese dialogues (the word segmentation spaces are removed). The label records what the crowdworker playing the bot "
             "did, so it is somewhat subjective; about 21% of bot replies use a stored memory about the user.")

    def load(self):
        rows = []
        with zipfile.ZipFile(fetch("https://dialogue.bj.bcebos.com/Knover/DuLeMon.zip", "DuLeMon.zip")) as z:
            lines = z.read("DuLeMon/both/test.json").decode("utf-8").splitlines()
        for d, line in enumerate(lines):
            if not line.strip():
                continue
            dialog = json.loads(line)
            turns = [t.split("\t") for t in dialog["conversation"]]
            for t, parts in enumerate(turns):
                if not parts[0].startswith("Bot:") or t == 0:
                    continue
                tags = {x.strip() for x in parts[1].split(",")} if len(parts) > 1 else set()
                history = []
                for prev in turns[:t]:
                    role, _, text = prev[0].partition(":")
                    history.append(f"{'User' if role.strip() == 'Usr' else 'Bot'}: {_detok(text)}")
                rows.append({"dialog": d, "turn": t, "history": history,
                             "user_memory": [_detok(p.split(":", 1)[1]) for p in dialog["user_said_persona"]],
                             "bot_persona": [_detok(p.split(":", 1)[1]) for p in dialog["bot_persona"]],
                             "label": "true" if any(x.startswith("U") for x in tags) else "false"})
        return rows

    def label_of(self, record):
        return record["label"]

    def to_item(self, record, rng):
        state = {"memory_about_user": "\n".join(f"- {m}" for m in record["user_memory"]) or "(empty)",
                 "bot_persona": "\n".join(f"- {m}" for m in record["bot_persona"]),
                 "dialogue_so_far": "\n".join(record["history"])}
        instructions = ("A chatbot has a long-term memory of what the user said about themselves in earlier conversations. Before "
                        "writing its next reply, should the bot retrieve and use one of these stored memories about the user? Answer "
                        "true if the next reply should draw on a stored user memory, false if it should not.")
        return self.item(state, {"retrieve_memory": Question(type="boolean", instructions=instructions)},
                         {"retrieve_memory": Gold(label=record["label"])}, dialog=record["dialog"], turn=record["turn"])
