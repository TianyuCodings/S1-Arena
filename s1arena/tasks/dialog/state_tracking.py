"""Leaf: dialogue state tracking (对话状态追踪).

One item is a conversation up to a user turn plus one frame (service) that is active in that
turn. It asks for the active intent and, for every slot of that intent, the value the user has
specified or accepted so far, chosen from candidates: "none", "dontcare", the schema's possible
values (categorical slots) or the values mentioned in the conversation so far (other slots; a
free-form slot that nobody has mentioned yet is not asked).
SGD and MultiWOZ 2.2 share the same JSON layout.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch, slug

LEAF = "dialog/state_tracking"
NONE_TEXT = "The user has not specified or accepted a value for this slot so far."
DONTCARE_TEXT = "The user has said they have no preference for this slot (any value is fine)."


def norm(value: str) -> str:
    return " ".join(str(value).lower().split())


class SchemaDST(TaskAdapter):
    tier = "B"
    criteria_source = "Service, intent and slot descriptions verbatim from the dataset schema; question wording written"

    def raw(self) -> tuple[list[dict], dict[str, dict]]:
        """Test dialogues and the schema by service name."""
        raise NotImplementedError

    def turn_mentions(self, dialogue: dict, index: int, turn: dict) -> list[tuple[str, str, str]]:
        """(slot, surface value, group key) mentioned in one turn."""
        out = []
        for frame in turn["frames"]:
            for action in frame.get("actions", []):
                canon = action.get("canonical_values") or action["values"]
                for value, key in zip(action["values"], canon):
                    out.append((action["slot"], value, norm(key)))
        return out

    def active_frames(self, turn: dict) -> list[dict]:
        return [f for f in turn["frames"] if f.get("state") and f["state"].get("active_intent", "NONE") != "NONE"]

    def load(self):
        dialogues, schema = self.raw()
        records = []
        for dialogue in dialogues:
            seen: dict[str, dict[str, list[str]]] = defaultdict(dict)  # slot -> group key -> surface values, in order
            lines = []
            for index, turn in enumerate(dialogue["turns"]):
                speaker = "User" if turn["speaker"] == "USER" else "System"
                lines.append(f"{speaker}: {turn['utterance']}")
                for slot, value, key in self.turn_mentions(dialogue, index, turn):
                    group = seen[slot].setdefault(key, [])
                    if value not in group:
                        group.append(value)
                if turn["speaker"] != "USER":
                    continue
                for frame in self.active_frames(turn):
                    service = schema.get(frame["service"])
                    intent = frame["state"]["active_intent"]
                    intents = {i["name"]: i for i in service["intents"]} if service else {}
                    if intent not in intents:
                        continue
                    values = frame["state"]["slot_values"]
                    for slot, vals in values.items():  # state values are mentions too (e.g. accepted offers)
                        key = self.group_of(seen[slot], vals)
                        group = seen[slot].setdefault(key, [])
                        for v in vals:
                            if v not in group and norm(v) not in {norm(g) for g in group}:
                                group.append(v)
                    slots = list(intents[intent]["required_slots"]) + list(intents[intent]["optional_slots"])
                    records.append({
                        "dialogue_id": dialogue["dialogue_id"], "turn": index, "service": frame["service"], "intent": intent,
                        "conversation": "\n".join(lines), "slots": slots, "state": values,
                        "mentions": {s: {k: list(g) for k, g in seen[s].items()} for s in slots if s in seen},
                    })
        self.schema = schema
        return records

    @staticmethod
    def group_of(groups: dict[str, list[str]], values: list[str]) -> str:
        """Key of the mention group that contains any of these surface values, else a new key."""
        wanted = {norm(v) for v in values}
        for key, group in groups.items():
            if wanted & ({norm(g) for g in group} | {key}):
                return key
        return norm(values[0])

    def label_of(self, record):
        filled = sum(1 for s in record["slots"] if s in record["state"])
        return f"{record['service'].split('_')[0].lower()}|{min(filled, 3)}"

    def slot_question(self, record, slot_spec: dict, service: dict, rng: random.Random) -> tuple[Question, Gold] | None:
        slot = slot_spec["name"]
        state = record["state"].get(slot)
        options = {"none": NONE_TEXT, "dontcare": DONTCARE_TEXT}
        gold = "none"
        if slot_spec.get("is_categorical"):
            for v in slot_spec.get("possible_values", []):
                if norm(v) not in ("dontcare", "none"):
                    options.setdefault(norm(v), f"{v}")
            if state:
                value = norm(state[0])
                if value in ("dontcare", "none"):
                    gold = value
                elif value in options:
                    gold = value
                else:
                    return None  # annotated value outside the schema's list
        else:
            groups = record["mentions"].get(slot, {})
            ids = {}
            for key, surfaces in groups.items():
                if key in ("dontcare", "none"):
                    continue
                oid = norm(surfaces[0])
                if oid in options:
                    continue
                shown = " / ".join(f'"{s}"' for s in surfaces)
                options[oid] = f"The value mentioned as {shown}."
                ids[key] = oid
            if state:
                value = norm(state[0])
                if value in ("dontcare", "none"):
                    gold = value
                else:
                    key = self.group_of(groups, state)
                    if key not in ids:
                        return None
                    gold = ids[key]
            if len(options) == 2 and gold == "none":
                return None  # nothing mentioned for this slot: only "none" or "dontcare" would be offered
        instructions = (f"Dialogue state tracking for the service \"{record['service']}\" ({service['description']}). "
                        f"Slot \"{slot}\": {slot_spec['description']}. Based on the conversation so far, which value has the user "
                        "specified or accepted for this slot?")
        return Question(type="choice", instructions=instructions, criteria=shuffled_options(options, rng)), Gold(label=gold)

    def to_item(self, record, rng):
        service = self.schema[record["service"]]
        specs = {s["name"]: s for s in service["slots"]}
        intents = {i["name"]: i["description"] for i in service["intents"]}
        questions, gold = {}, {}
        if len(intents) >= 2:
            questions["active_intent"] = Question(
                type="choice",
                instructions=f"Which intent of the service \"{record['service']}\" ({service['description']}) is the user currently pursuing?",
                criteria=shuffled_options(intents, rng),
            )
            gold["active_intent"] = Gold(label=record["intent"])
        for slot in record["slots"]:
            if slot not in specs:
                continue
            made = self.slot_question(record, specs[slot], service, rng)
            if made is None:
                continue
            questions[slug(slot)], gold[slug(slot)] = made
        if not any(q != "active_intent" for q in questions):
            return None
        meta = {"dialogue_id": record["dialogue_id"], "turn": record["turn"], "service": record["service"]}
        return self.item({"conversation": record["conversation"]}, questions, gold, **meta)


SGD = "https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/master/test/"


@register_task
class SGDTest(SchemaDST):
    name = f"{LEAF}/sgd"
    leaf = LEAF
    source = "Schema-Guided Dialogue (SGD / DSTC8), test split (Rastogi et al., 2020)"
    url = "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue"
    license = "CC BY-SA 4.0"
    notes = ("The test split includes services unseen in training. Options for free-form slots are the values mentioned so far "
             "(surface forms of the same canonical value are one option), plus random values of the same slot from other dialogues "
             "up to three; accepted system offers count as specified.")

    def raw(self):
        dialogues = []
        for i in range(1, 35):
            with fetch(SGD + f"dialogues_{i:03d}.json", f"sgd_test_dialogues_{i:03d}.json").open(encoding="utf-8") as f:
                dialogues += json.load(f)
        with fetch(SGD + "schema.json", "sgd_test_schema.json").open(encoding="utf-8") as f:
            schema = {s["service_name"]: s for s in json.load(f)}
        return dialogues, schema


MWOZ = "https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/"


@register_task
class MultiWOZ22(SchemaDST):
    name = f"{LEAF}/multiwoz22"
    leaf = LEAF
    source = "MultiWOZ 2.2, test split (Zang et al., 2020; Budzianowski et al., 2018)"
    url = "https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2"
    license = "MIT"
    notes = ("Candidate values for free-form slots come from the dialogue acts and states of earlier turns; the state's alternative "
             "spellings of a value form one option. MultiWOZ 2.2 corrected many state errors of 2.1, but some label noise remains.")

    def raw(self):
        dialogues = []
        for i in (1, 2):
            with fetch(MWOZ + f"test/dialogues_{i:03d}.json", f"multiwoz22_test_dialogues_{i:03d}.json").open(encoding="utf-8") as f:
                dialogues += json.load(f)
        with fetch(MWOZ + "schema.json", "multiwoz22_schema.json").open(encoding="utf-8") as f:
            schema = {s["service_name"]: s for s in json.load(f)}
        with fetch(MWOZ + "dialog_acts.json", "multiwoz22_dialog_acts.json").open(encoding="utf-8") as f:
            self.acts = json.load(f)
        return dialogues, schema

    def turn_mentions(self, dialogue, index, turn):
        acts = self.acts.get(dialogue["dialogue_id"], {}).get(str(index), {}).get("dialog_act", {})
        out = []
        for act, pairs in acts.items():
            domain = act.split("-")[0].lower()
            for slot, value in pairs:
                if value in ("?", "none") or slot == "none":
                    continue
                out.append((f"{domain}-{slot}", value, norm(value)))
        for frame in turn["frames"]:
            for span in frame.get("slots", []):
                if "value" in span and isinstance(span["value"], str):
                    out.append((span["slot"], span["value"], norm(span["value"])))
        return out
