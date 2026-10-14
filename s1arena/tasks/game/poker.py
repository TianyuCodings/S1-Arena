"""Leaf: poker, imperfect information (扑克（不完全信息）).

PokerBench spots of 6-handed No Limit Texas Hold'em, 100 big blinds deep. The options are the
actions of the solver's game tree at that spot (so bet sizes come already discretized, e.g. one
or two bet sizes), and the gold is the solver's dominant action: GTO Wizard preflop charts and
the WASM-Postflop solver, keeping only spots where one action has more than 50% frequency.
Action frequencies are not released, so there are no soft labels.
"""

from __future__ import annotations

import ast
import collections
import csv
import functools
import json
import re

from ...env import load_env
from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter

LEAF = "game/poker"
REPO, REV = "RZ412/PokerBench", "7ac61f961c81a50fc0f667820b2fb0e432dfec0d"
RANKS = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J", "Ten": "T", "Nine": "9", "Eight": "8", "Seven": "7",
         "Six": "6", "Five": "5", "Four": "4", "Three": "3", "Two": "2", "Deuce": "2"}
SUITS = {"Heart": "h", "Spade": "s", "Diamond": "d", "Club": "c"}
INSTRUCTIONS = ("You are a specialist in playing 6-handed No Limit Texas Holdem. The following will be a game scenario and you "
                "need to make the optimal decision. Now it is your turn to make a move. Decide on an action based on the strength "
                "of your hand on this board, your position, and actions before you.")


def _card(words: str) -> str:
    rank, _, suit = words.strip().split()
    return RANKS[rank.capitalize()] + SUITS[suit.capitalize()]


def _board(text: str) -> str:
    cards = []
    m = re.search(r"The flop comes (.+?), (.+?), and (.+?),", text)
    if m:
        cards += [_card(m.group(i)) for i in (1, 2, 3)]
    for street in ("turn", "river"):
        m = re.search(rf"The {street} comes (.+?),", text)
        if m:
            cards.append(_card(m.group(1)))
    return "".join(cards)


def action_id(move: str) -> str:
    """'Bet 24' -> 'bet 24', '2.5bb' -> 'raise 2.5', 'raise 14.0' -> 'raise 14', 'allin' -> 'all-in'."""
    move = move.lower().strip()
    m = re.fullmatch(r"(raise|bet) ([0-9.]+)", move)
    if m:
        return f"{m.group(1)} {float(m.group(2)):g}"
    m = re.fullmatch(r"([0-9.]+)bb", move)
    if m:
        return f"raise {float(m.group(1)):g}"
    return "all-in" if move == "allin" else move


ORDER = {"fold": 0, "check": 1, "call": 2, "bet": 3, "raise": 4, "all-in": 5}


def natural_order(moves: list[str]) -> list[str]:
    """Fold, check, call, bets and raises by size, all-in."""
    return sorted(moves, key=lambda m: (ORDER[m.split()[0]], float(m.split()[1]) if " " in m else 0.0))


def describe(action: str, pot: float) -> str:
    kind, _, amount = action.partition(" ")
    if kind == "bet":
        return f"bet {amount} chips ({float(amount) / pot:.0%} of the pot)"
    if kind == "raise":
        return f"raise to {amount} chips"
    return {"all-in": "go all-in", "fold": "fold", "call": "call", "check": "check"}[kind]


@functools.lru_cache(maxsize=2)
def pokerbench(stage: str) -> tuple[dict, ...]:
    """Test spots joined with their structured rows (for the list of available actions)."""
    load_env()
    from huggingface_hub import hf_hub_download

    size = "1k" if stage == "preflop" else "10k"
    prompts = json.load(open(hf_hub_download(REPO, f"{stage}_{size}_test_set_prompt_and_label.json", repo_type="dataset", revision=REV), encoding="utf-8"))
    with open(hf_hub_download(REPO, f"{stage}_{size}_test_set_game_scenario_information.csv", repo_type="dataset", revision=REV), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    index = collections.defaultdict(list)
    for r in rows:
        board = ""
        if stage == "postflop":
            board = r["board_flop"] + (r["board_turn"] if r["evaluation_at"] in ("Turn", "River") else "") + (r["board_river"] if r["evaluation_at"] == "River" else "")
        holding = r.get("holding") or r.get("hero_holding")
        index[(holding, round(float(r["pot_size"]), 2), action_id(r["correct_decision"]), board)].append(r)
    out = []
    for i, p in enumerate(prompts):
        text = p["instruction"]
        hand = re.search(r"your holding is \[(.+?) and (.+?)\]", text)
        pot = float(re.search(r"current pot size is ([0-9.]+) chips", text).group(1))
        key = (_card(hand.group(1)) + _card(hand.group(2)), round(pot, 2), action_id(p["output"]), _board(text) if stage == "postflop" else "")
        matches = index.get(key, [])
        if len({m["available_moves"] for m in matches}) != 1:
            continue  # no structured row, or rows with different action lists
        row = matches[0]
        moves = [action_id(m) for m in ast.literal_eval(row["available_moves"])]
        summary = text[text.index("Here is a game summary:") + len("Here is a game summary:"): text.index("Now it is your turn")].strip()
        out.append({"id": i, "summary": summary, "pot": pot, "holding": f"{hand.group(1)} and {hand.group(2)}", "moves": moves,
                    "answer": action_id(p["output"]), "street": row.get("evaluation_at", "Preflop"),
                    "spot": row.get("prev_line", ""), "hero": row.get("hero_pos") or row.get("hero_position")})
    return tuple(out)


class PokerBenchTask(TaskAdapter):
    leaf = LEAF
    url = "https://huggingface.co/datasets/RZ412/PokerBench"
    license = "Apache-2.0"
    tier = "A"
    criteria_source = "PokerBench prompt sentences verbatim; options are the solver tree's available actions from the dataset's structured files"
    STAGE = ""

    def load(self):
        return [r for r in pokerbench(self.STAGE) if self.keep(r)]

    def keep(self, record) -> bool:
        return len(record["moves"]) >= 2 and record["answer"] in record["moves"]

    def label_of(self, record):
        return f"{record['answer'].split()[0]}|{record['street']}"

    def to_item(self, record, rng):
        options = {m: describe(m, record["pot"]) for m in natural_order(record["moves"])}
        question = Question(type="choice", instructions=INSTRUCTIONS, criteria=options)
        state = {"game_summary": record["summary"], "pot_size": f"{record['pot']:g} chips", "your_holding": record["holding"]}
        return self.item(state, {"action": question}, {"action": Gold(label=record["answer"])},
                         original_index=record["id"], street=record["street"])


@register_task
class PokerBenchPreflop(PokerBenchTask):
    name = f"{LEAF}/pokerbench_preflop"
    source = "PokerBench pre-flop test set, solutions from GTO Wizard charts (Zhuang et al., AAAI 2025)"
    STAGE = "preflop"
    notes = ("The 250 'check' spots are all one spot (the small blind limps, the big blind acts) with the pot shown as 1 instead "
             "of 2 chips, so they are left out, as are 47 spots whose action list holds only 'fold'. Raise sizes are raise-to "
             "amounts in chips (1 chip = 1 big blind). Options are listed from passive to aggressive.")

    def keep(self, record):
        return super().keep(record) and record["spot"] != "SB/call"


@register_task
class PokerBenchPostflop(PokerBenchTask):
    name = f"{LEAF}/pokerbench_postflop"
    source = "PokerBench post-flop test set, solutions from WASM-Postflop (Zhuang et al., AAAI 2025)"
    STAGE = "postflop"
    notes = ("Flop, turn and river decisions in single-raised and 3-bet pots between two players. The solver tree offers one or "
             "two bet or raise sizes per spot; bets are also shown as a share of the pot. Raise amounts are raise-to totals. "
             "Two spots whose structured row could not be matched uniquely are left out.")
