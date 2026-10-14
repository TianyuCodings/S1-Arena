"""Leaf: cooperative and social games (合作 / 社交博弈).

Three decisions from games played by people. Diplomacy: is a negotiation message a lie? The
sender marked every message as truthful or deceptive during play. Werewolf: after the first
day's speeches, which players are werewolves? Gold is the role the game assigned. Hanabi: at a
decision point of an expert human game, will playing a given card succeed? Gold is the exact
probability from card counting under the hints received (a solver's belief, used as a soft label).
"""

from __future__ import annotations

import functools
import hashlib
import json
import random
import zipfile

from ...env import load_env
from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch
from . import _hanabi as hb

LEAF = "game/social_games"


def _order(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


# ---------- Diplomacy: deception in negotiation messages ----------

DIPLOMACY = "https://raw.githubusercontent.com/DenisPeskoff/2020_acl_diplomacy/c28c0870a639f6f388c87d4f8eaf685e64bfb1a6/data/"
HISTORY = 20  # earlier messages of the conversation shown


@register_task
class DiplomacyDeception(TaskAdapter):
    name = f"{LEAF}/diplomacy_deception"
    leaf = LEAF
    source = "It Takes Two to Lie: Diplomacy negotiation messages with the senders' own truthfulness labels, test games (Peskov et al., ACL 2020)"
    url = "https://github.com/DenisPeskoff/2020_acl_diplomacy"
    license = "CC BY 4.0"
    tier = "B"
    criteria_source = "Adapted from the annotation in Peskov et al. (2020): senders marked each message they sent as truthful or deceptive"
    sampling = "balanced"
    notes = ("Messages of the two held-out test games (2,741 messages, 8.8% lies). The item shows the message, the 20 earlier "
             "messages of the same two-player conversation, the season and both players' supply-center counts. Equal numbers "
             "of lies and truthful messages are sampled, since lies are rare.")

    def load(self):
        rows = []
        with open(fetch(DIPLOMACY + "test.jsonl", "diplomacy_deception_test.jsonl"), encoding="utf-8") as f:
            for c, line in enumerate(f):
                conv = json.loads(line)
                for i, (text, label) in enumerate(zip(conv["messages"], conv["sender_labels"])):
                    if label not in (True, False) or not text.strip():
                        continue
                    rows.append({"game": conv["game_id"], "conversation": c, "index": i, "label": "lie" if label is False else "truth",
                                 "conv": conv})
        return rows

    def label_of(self, record):
        return record["label"]

    def record_rng(self, record, seed):  # the record carries the whole conversation; key the randomness on the message only
        return random.Random(f"{seed}:{self.name}:{record['conversation']}:{record['index']}")

    def to_item(self, record, rng):
        conv, i = record["conv"], record["index"]
        say = lambda k: f"{conv['seasons'][k]} {conv['years'][k]}, {conv['speakers'][k].title()}: {conv['messages'][k].strip()}"
        earlier = [say(k) for k in range(max(0, i - HISTORY), i)]
        sender, receiver = conv["speakers"][i].title(), conv["receivers"][i].title()
        score, delta = int(conv["game_score"][i]), int(conv["game_score_delta"][i])
        state = {
            "game": (f"Diplomacy, played online by seven players who negotiate in private messages. This is the private conversation "
                     f"between {sender} and {receiver}."),
            "earlier_messages": "\n".join(earlier) if earlier else "(none)",
            "message": f"{conv['seasons'][i]} {conv['years'][i]}, {sender} to {receiver}: {conv['messages'][i].strip()}",
            "supply_centers": f"{sender} holds {score} supply centers and {receiver} holds {score - delta}.",
        }
        question = Question(type="boolean", instructions=(
            "While playing, every sender privately marked each message they sent as truthful or as a lie. Is this message a lie, "
            "according to its sender?"))
        return self.item(state, {"lie": question}, {"lie": Gold(label="true" if record["label"] == "lie" else "false")},
                         game_id=conv["game_id"], message_index=conv["absolute_message_index"][i])


# ---------- Werewolf: who are the werewolves after the first day's speeches ----------

FANLANG_ZIP = "https://drive.usercontent.google.com/download?id=1pw6uIPdjfxssEPELA-U6neejmZ2sIrpe&export=download&confirm=t"
WEREWOLF_POOL = 3000
SETUP = ("A 9-player game of Werewolf on the FanLang app. Roles: 3 werewolves, 3 villagers, a seer, a witch and a hunter; players "
         "do not see each other's roles. Each night the werewolves choose a victim, the seer checks one player's side, the witch "
         "may save the victim once and poison someone once, and the hunter may shoot a player when he dies. During the day "
         "every living player speaks once, then all vote to exile one player. Speeches are automatic transcripts in Chinese.")


@functools.lru_cache(maxsize=1)
def fanlang_games() -> tuple[dict, ...]:
    """Games whose first day has at least eight transcribed speeches before the vote and no werewolf self-exposure."""
    out = []
    with zipfile.ZipFile(fetch(FANLANG_ZIP, "fanlang9_werewolf_data.zip")) as z:
        names = sorted((n for n in z.namelist() if n.endswith(".json")), key=_order)
        for name in names:
            d = json.loads(z.read(name))
            gs, times = d["game_state"], d["task_time"]
            vote = times.get("Day 1 Daytime - Start voting")
            if not vote or "suicide" in gs.get("Day 1 Daytime", {}) or "Voting Result" not in gs.get("Day 1 Daytime", {}):
                continue
            talks = []
            for key, audio in d["audio"].items():
                if not key.startswith("Day 1 Daytime - [") or key not in times or times[key][0] >= vote[0]:
                    continue
                text = (audio.get("text") or "").strip()
                if len(text) < 20:
                    continue
                player = key.split("[")[1].split("]")[0]
                kind = "last words" if "last words" in key else "speech"
                talks.append((times[key][0], player, kind, text))
            if sum(kind == "speech" for _, _, kind, _ in talks) < 8:
                continue
            deaths = gs.get("Day 1 Night", {}).get("Death Message") or []
            out.append({"id": name.rsplit("/", 1)[-1][:-5], "roles": gs["roles"], "deaths": deaths, "result": gs.get("Game Result", ""),
                        "talks": [(p, k, t) for _, p, k, t in sorted(talks)]})
            if len(out) >= WEREWOLF_POOL:
                break
    return tuple(out)


@register_task
class FanLangWerewolf(TaskAdapter):
    name = f"{LEAF}/werewolf_fanlang"
    leaf = LEAF
    source = "FanLang-9: human 9-player Werewolf games with roles, votes and transcribed speeches (Wu et al., 2024)"
    url = "https://github.com/boluoweifenda/werewolf"
    license = "MIT"
    tier = "C"
    criteria_source = "Written; gold is each player's role as assigned by the game"
    notes = ("Chinese speeches transcribed automatically (Paraformer); the dataset's LLM-written speech summaries are not used. "
             "Each item shows the first night's outcome and all first-day speeches before the vote, and asks for every seat "
             "whether that player is a werewolf (3 of 9 are). Games where a werewolf revealed himself on day 1 are left out.")

    def load(self):
        return list(fanlang_games())

    def label_of(self, record):
        return record["result"]

    def to_item(self, record, rng):
        deaths = record["deaths"]
        night = "Nobody died during the first night." if not deaths else f"Player {', '.join(map(str, deaths))} died during the first night."
        lines = [f"[{p}] ({kind}): {text}" for p, kind, text in record["talks"]]
        state = {"setup": SETUP, "first_night": night, "first_day_speeches": "\n".join(lines)}
        questions, gold = {}, {}
        for seat in sorted(record["roles"], key=int):
            questions[f"player_{seat}"] = Question(type="boolean", instructions=(
                f"Is player {seat} a werewolf? Judge from the first night's outcome and the first day's speeches."))
            gold[f"player_{seat}"] = Gold(label="true" if record["roles"][seat] == "Werewolf" else "false")
        return self.item(state, questions, gold, original_id=record["id"], game_result=record["result"])


# ---------- Hanabi: will this card play successfully, from expert human games ----------

AH2AC2_REPO, AH2AC2_REV = "ah2ac2/datasets", "a88a99e40bf78e582fab17ff0dcef6b7937f3c1b"
AH2AC2_FILES = ("2_player_games_train_1k", "2_player_games_val", "3_player_games_train_1k", "3_player_games_val")
BANDS = ((0.0, 0.0, "impossible"), (1e-9, 0.4, "unlikely"), (0.6, 1 - 1e-9, "likely"), (1.0, 1.0, "certain"))


def band_of(p: float) -> str | None:
    for low, high, name in BANDS:
        if low <= p <= high:
            return name
    return None  # between 0.4 and 0.6: left out, the hard label would be a coin flip


@functools.lru_cache(maxsize=1)
def hanabi_points() -> tuple[dict, ...]:
    """One decision point per human game, with the exact chance that the chosen card is playable."""
    load_env()
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    out = []
    for fname in AH2AC2_FILES:
        data = hb.read_safetensors(Path(hf_hub_download(AH2AC2_REPO, f"{fname}.safetensors", repo_type="dataset", revision=AH2AC2_REV)))
        players = int(data["num_players"])
        for g in range(len(data["scores"])):
            snaps, score = hb.replay(data["decks"][g], data["actions"][g], players, int(data["num_actions"][g]))
            if score != int(data["scores"][g]):
                raise ValueError(f"replay of {fname} game {g} does not reproduce its score")
            key = _order(f"{fname}:{g}")
            candidates = [s for s in snaps[2:] if s.deck_left > 0]
            if not candidates:
                continue
            snap = candidates[int(key[:8], 16) % len(candidates)]
            hand = snap.hands[snap.player]
            clued = [i for i, c in enumerate(hand) if len(c.colors) < 5 or len(c.ranks) < 5]
            pick = int(key[8:16], 16)
            if snap.action and snap.action[0] == "play":
                slot = snap.action[1]  # the card the human chose to play
            elif clued and pick % 4:
                slot = clued[pick % len(clued)]
            else:
                slot = pick % len(hand)
            belief = hb.slot_belief(snap, slot)
            p = sum(w for t, w in belief.items() if snap.fireworks[t // 5] == t % 5)
            band = band_of(p)
            if band is None:
                continue
            out.append({"file": fname, "game": g, "snap": snap, "slot": slot, "p": round(p, 6), "band": band,
                        "human_action": snap.action[0]})
    return tuple(out)


def hanabi_state(s: hb.Snapshot) -> dict:
    fire = ", ".join(f"{hb.COLOR_NAMES[hb.COLORS[c]]} {s.fireworks[c]}" for c in range(5))
    discards = ", ".join(hb.card_name(c, r) for c, r in sorted(s.discards)) or "none"
    mates = []
    for k in range(1, s.players):
        q = (s.player + k) % s.players
        cards = ", ".join(f"slot {i + 1}: {hb.card_name(c.color, c.rank)}" for i, c in enumerate(s.hands[q]))
        mates.append(f"Player {q + 1}: {cards}")
    own = "\n".join(f"slot {i + 1}: {hb.knowledge_text(c)}" for i, c in enumerate(s.hands[s.player]))
    return {
        "game": f"Hanabi with {s.players} players; you are player {s.player + 1} and it is your turn. Slot 1 is your oldest card.",
        "fireworks": f"{fire} (the highest card played in each color; 0 = none yet)",
        "clue_tokens": f"{s.info} of 8", "strikes": f"{3 - s.lives} of 3", "cards_left_in_deck": str(s.deck_left),
        "discard_pile": discards,
        "teammates_hands": "\n".join(mates),
        "your_hand_as_hinted": own,
    }


@register_task
class HanabiPlayability(TaskAdapter):
    name = f"{LEAF}/hanabi_human_games"
    leaf = LEAF
    source = "AH2AC2 human Hanabi games from hanab.live, 2- and 3-player (Dizdarević et al., 2025), with exact card-counting beliefs"
    url = "https://huggingface.co/datasets/ah2ac2/datasets"
    license = "Apache-2.0"
    tier = "C"
    criteria_source = "Written; the soft label is the exact probability that the card is playable given the hints and the visible cards"
    notes = ("One decision point per game, from expert games (mean score about 23 of 25). The card asked about is the one the "
             "human played at that turn when they played, otherwise a card that has received a hint (or, one time in four, any "
             "card). The probability counts every unseen copy equally and uses only the literal content of the hints, not "
             "conventions. Points with a probability between 0.4 and 0.6 are left out; the hard label is the likelier answer. "
             "Replaying the recorded actions reproduces every game's recorded score.")

    def load(self):
        return [{k: v for k, v in r.items() if k != "snap"} | {"key": f"{r['file']}:{r['game']}"} for r in hanabi_points()]

    def label_of(self, record):
        return f"{record['band']}|{record['file'][0]}p"

    def to_item(self, record, rng):
        point = next(r for r in hanabi_points() if r["file"] == record["file"] and r["game"] == record["game"])
        s, slot, p = point["snap"], point["slot"], point["p"]
        question = Question(type="boolean", instructions=(
            f"If you play the card in your slot {slot + 1} now, will it succeed, that is, is it the next card needed on its color's "
            "firework? Judge only from what the hints literally told you about your cards and the cards you can see; any card "
            "you cannot see is equally likely to be any of the copies not accounted for."))
        gold = Gold(label="true" if p > 0.5 else "false", dist={"true": p, "false": round(1 - p, 6)})
        return self.item(hanabi_state(s), {"play_succeeds": question}, {"play_succeeds": gold},
                         game=record["key"], p_playable=p, human_action=point["human_action"])
