"""Hanabi: replaying human games recorded in the JaxMARL format, and exact card-counting beliefs.

The AH2AC2 human dataset stores each game as a deck of (color, rank) pairs and one action id per
player per step (JaxMARL Hanabi encoding: discard slot 0-4, play slot 0-4, color hints then rank
hints per other player in seat order, noop). Hands are dealt from the top of the deck, player 0
first; a played or discarded card leaves its slot and the new card is added at the end, so slot 1
is the oldest card. Replaying reproduces every recorded score (checked when loading).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

COLORS = "RYGWB"  # JaxMARL's color order
COLOR_NAMES = {"R": "red", "Y": "yellow", "G": "green", "W": "white", "B": "blue"}
COPIES = (3, 2, 2, 2, 1)  # copies of ranks 1-5 in each color


def read_safetensors(path) -> dict:
    """int32 arrays of a .safetensors file as nested lists (header: 8-byte length + JSON)."""
    import numpy as np

    data = path.read_bytes()
    n = struct.unpack("<Q", data[:8])[0]
    header, base = json.loads(data[8: 8 + n]), 8 + n
    out = {}
    for key, spec in header.items():
        if key == "__metadata__":
            continue
        start, end = spec["data_offsets"]
        out[key] = np.frombuffer(data[base + start: base + end], dtype=np.int32).reshape(spec["shape"])
    return out


@dataclass
class Card:
    color: int
    rank: int  # 0-4
    colors: set = field(default_factory=lambda: set(range(5)))  # still possible given the hints
    ranks: set = field(default_factory=lambda: set(range(5)))


@dataclass
class Snapshot:
    """What the acting player knows just before acting at one step."""
    step: int
    player: int
    players: int
    fireworks: list
    info: int
    lives: int
    deck_left: int
    discards: list
    hands: list  # hands[p] = list of Card, oldest first
    action: tuple  # what the human did: ('play'|'discard', slot) or ('hint', target, 'color'|'rank', value)


def replay(deck, actions, players: int, steps: int) -> tuple[list[Snapshot], int]:
    """Snapshots before every step, and the final score (0 when all lives are lost)."""
    hand_size = 5 if players < 4 else 4
    noop = 2 * hand_size + (players - 1) * 10
    hands = [[Card(*map(int, deck[p * hand_size + i])) for i in range(hand_size)] for p in range(players)]
    dealt, fire, info, lives, last_round, discards = players * hand_size, [0] * 5, 8, 3, 0, []
    snaps = []
    for t in range(steps):
        p = t % players
        a = int(actions[t][p])
        if any(int(actions[t][q]) != noop for q in range(players) if q != p):
            raise ValueError("a player acted out of turn")
        snap = Snapshot(t, p, players, list(fire), info, lives, 50 - dealt, list(discards),
                        [[Card(c.color, c.rank, set(c.colors), set(c.ranks)) for c in h] for h in hands], ())
        if a < 2 * hand_size:
            slot = a % hand_size
            card = hands[p].pop(slot)
            if a < hand_size:
                snap.action = ("discard", slot)
                info = min(8, info + 1)
                discards.append((card.color, card.rank))
            else:
                snap.action = ("play", slot)
                if fire[card.color] == card.rank:
                    fire[card.color] += 1
                    if card.rank == 4:
                        info = min(8, info + 1)
                else:
                    lives -= 1
                    discards.append((card.color, card.rank))
            if dealt < 50:
                hands[p].append(Card(*map(int, deck[dealt])))
                dealt += 1
        elif a < noop:
            k = a - 2 * hand_size
            kind = "color" if k < (players - 1) * 5 else "rank"
            if kind == "rank":
                k -= (players - 1) * 5
            target, value = (p + 1 + k // 5) % players, k % 5
            touched = 0
            for c in hands[target]:
                hit = (c.color if kind == "color" else c.rank) == value
                touched += hit
                attr = c.colors if kind == "color" else c.ranks
                if hit:
                    attr.intersection_update({value})
                else:
                    attr.discard(value)
            if not touched or info == 0:
                raise ValueError("illegal hint")
            info -= 1
            snap.action = ("hint", target, kind, value)
        else:
            raise ValueError("the acting player passed")
        snaps.append(snap)
        if dealt >= 50:
            last_round += 1
        if lives == 0 or sum(fire) == 25 or last_round == players + 1:
            return snaps, (sum(fire) if lives > 0 else 0)
    return snaps, (sum(fire) if lives > 0 else 0)


def unseen_counts(s: Snapshot) -> list[int]:
    """Copies of each card type (index color * 5 + rank) that the acting player cannot see."""
    pool = [COPIES[r] for c in range(5) for r in range(5)]
    for c in range(5):
        for r in range(s.fireworks[c]):
            pool[c * 5 + r] -= 1
    for c, r in s.discards:
        pool[c * 5 + r] -= 1
    for q, hand in enumerate(s.hands):
        if q != s.player:
            for card in hand:
                pool[card.color * 5 + card.rank] -= 1
    if min(pool) < 0:
        raise ValueError("negative card count")
    return pool


def slot_belief(s: Snapshot, slot: int) -> dict[int, float]:
    """Exact posterior over the type of one own card: the own hand is a uniformly random draw of
    physical cards from the unseen ones, restricted to what the hints allow for each slot."""
    pool = tuple(unseen_counts(s))
    hand = s.hands[s.player]
    allowed = [[c * 5 + r for c in card.colors for r in card.ranks] for card in hand]
    others = [allowed[i] for i in range(len(hand)) if i != slot]
    memo: dict = {}

    def ways(i: int, counts: tuple) -> int:
        if i == len(others):
            return 1
        key = (i, counts)
        if key not in memo:
            total = 0
            for t in others[i]:
                if counts[t]:
                    total += counts[t] * ways(i + 1, counts[:t] + (counts[t] - 1,) + counts[t + 1:])
            memo[key] = total
        return memo[key]

    weights = {t: pool[t] * ways(0, pool[:t] + (pool[t] - 1,) + pool[t + 1:]) for t in allowed[slot] if pool[t]}
    total = sum(weights.values())
    return {t: w / total for t, w in weights.items() if w}


def card_name(color: int, rank: int) -> str:
    return f"{COLOR_NAMES[COLORS[color]]} {rank + 1}"


def knowledge_text(card: Card) -> str:
    colors = "any color" if len(card.colors) == 5 else " or ".join(COLOR_NAMES[COLORS[c]] for c in sorted(card.colors))
    ranks = "any rank" if len(card.ranks) == 5 else " or ".join(str(r + 1) for r in sorted(card.ranks))
    return f"{colors}; {ranks}"
