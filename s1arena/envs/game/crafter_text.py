"""Crafter (Hafner, ICLR 2022) played through BALROG's text wrapper.

BALROG (Paglieri et al., ICLR 2025) evaluates agents on Crafter through the language wrapper of
SmartPlay (Wu et al., 2023) as ported in balrog/environments/crafter/env.py: for each kind of thing
in the agent's 9 x 7 local view, the nearest one with its Manhattan distance and compass direction
("tree 3 steps to your north-east"), the thing in the cell the agent faces, and a status block
(health, food, drink and energy out of 9, then the inventory). describe_view() and describe_status()
port that wrapper with BALROG's default settings (nearest thing of each kind, coarse locations, no
skipped items) and one fix: BALROG slices the view out of the world map without padding, so within
4 cells of the west or 3 cells of the north edge of the 64 x 64 world the slice comes out empty and
the agent "sees nothing"; here cells outside the world are simply left out.

The task text reuses BALROG's instruction (the 17 actions with BALROG's descriptions and the 22
achievements) without "In a moment I will present..." and "PLAY!", and adds how the observation is
laid out and when the episode ends. Like BALROG's agent, which keeps the last 16 observations, the
state lists the 15 previous steps (observation and action) before the current one. Crafter ignores
every action while the player sleeps and its energy is below the maximum, so those steps are played
automatically (Noop) and are not decisions; they count toward BALROG's limit of 2000 steps after the
no-op BALROG's wrapper takes on reset. The world is crafter.Env(seed=seed) with BALROG's area and
view (64 x 64, 9 x 9); BALROG leaves the world seed random.

Crafter keeps the creatures of each world chunk in a Python set, whose iteration order follows memory
addresses, and despawns the creature at a random position in that order (crafter/env.py,
_balance_object), so a seed alone does not fix the game: two runs of the same seed and actions drift
apart once a creature is despawned. The harness keeps those per-chunk sets in insertion order
(ordered_chunks()), which fixes the game for a seed without changing its rules. Crafter also draws the
night-time noise of its image from the world's generator, so the frames rendered change later
events: the harness renders only the 64 x 64 frame Env.step() draws, BALROG a second 256 x 256 frame
per step, so the same world seed plays out differently at night than under BALROG's harness (same
rules and distribution of games).

Score = BALROG's progression: the percentage of the 22 achievements unlocked at least once.
"""

from __future__ import annotations

import collections
import threading
from typing import Any

from ...registry import register_env
from ...schema import DecisionRequest, Question
from ..base import Environment

ACTIONS = {  # option id -> (Crafter action index, BALROG's action name and description, verbatim)
    "noop": (0, "Noop: do nothing"),
    "move_west": (1, "Move West: move west on flat ground"),
    "move_east": (2, "Move East: move east on flat ground"),
    "move_north": (3, "Move North: move north on flat ground"),
    "move_south": (4, "Move South: move south on flat ground"),
    "do": (5, "Do: Multiuse action to collect material, drink from lake and hit creature in front"),
    "sleep": (6, "Sleep: sleep when energy level is below maximum"),
    "place_stone": (7, "Place Stone: place a stone in front"),
    "place_table": (8, "Place Table: place a table"),
    "place_furnace": (9, "Place Furnace: place a furnace"),
    "place_plant": (10, "Place Plant: place a plant"),
    "make_wood_pickaxe": (11, "Make Wood Pickaxe: craft a wood pickaxe with a nearby table and wood in inventory"),
    "make_stone_pickaxe": (12, "Make Stone Pickaxe: craft a stone pickaxe with a nearby table, wood, and stone in inventory"),
    "make_iron_pickaxe": (13, "Make Iron Pickaxe: craft an iron pickaxe with a nearby table and furnace, wood, coal, and iron in "
                              "inventory"),
    "make_wood_sword": (14, "Make Wood Sword: craft a wood sword with a nearby table and wood in inventory"),
    "make_stone_sword": (15, "Make Stone Sword: craft a stone sword with a nearby table, wood, and stone in inventory"),
    "make_iron_sword": (16, "Make Iron Sword: craft an iron sword with a nearby table and furnace, wood, coal, and iron in inventory"),
}
ACHIEVEMENTS = ["Collect Wood", "Place Table", "Eat Cow", "Collect Sapling", "Collect Drink", "Make Wood Pickaxe", "Make Wood Sword",
                "Place Plant", "Defeat Zombie", "Collect Stone", "Place Stone", "Eat Plant", "Defeat Skeleton", "Make Stone Pickaxe",
                "Make Stone Sword", "Wake Up", "Place Furnace", "Collect Coal", "Collect Iron", "Make Iron Pickaxe", "Make Iron Sword",
                "Collect Diamond"]  # as numbered in BALROG's instruction
VITALS = ["health", "food", "drink", "energy"]
AREA, VIEW = (64, 64), (9, 9)  # BALROG's crafter_kwargs
STEP_LIMIT = 2000  # BALROG's crafter max_episode_steps
HISTORY = 15  # previous steps shown; with the current observation, BALROG's max_text_history of 16

_names: list[str] = []
_names_lock = threading.Lock()


def item_names() -> list[str]:
    """Crafter's semantic ids -> names, built as BALROG builds id_to_item (materials, then object classes lowercased)."""
    with _names_lock:
        if not _names:
            import crafter

            env = crafter.Env(area=AREA, view=VIEW)
            names = ["None"] * 19
            for name, index in [*env._world._mat_ids.items(), *env._sem_view._obj_ids.items()]:
                text = str(name)
                names[index] = text[text.find("objects.") + len("objects."):-2].lower() if "objects." in text else text
            _names.extend(names)
        return _names


class _OrderedSet(dict):
    """The operations Crafter's World uses on a chunk's object set (add, remove, iterate), in insertion order."""

    def add(self, item) -> None:
        self[item] = None

    def remove(self, item) -> None:
        del self[item]


def ordered_chunks(world) -> None:
    """Replace the world's per-chunk object sets by insertion-ordered ones (rebuilt in object creation order, which is the
    order they were filled in by world generation, so the chunk order is kept too)."""
    chunks = collections.defaultdict(_OrderedSet)
    for obj in world._objects:
        if obj is not None:
            chunks[world.chunk_key(obj.pos)].add(obj)
    world._chunks = chunks


def describe_location(dx: int, dy: int) -> str:
    """BALROG's describe_loc_old: Manhattan distance and compass direction of an offset (x east, y south)."""
    desc = []
    if dy < 0:
        desc.append("north")
    elif dy > 0:
        desc.append("south")
    if dx < 0:
        desc.append("west")
    elif dx > 0:
        desc.append("east")
    distance = abs(dx) + abs(dy)
    return f"{distance} step{'s' if distance > 1 else ''} to your {'-'.join(desc)}"


def view_parts(info: dict) -> tuple[str, list[str], str]:
    """BALROG's describe_frame, in parts: the sleeping/dead status, the nearest thing of each kind in view, the faced thing."""
    names = item_names()
    semantic = info["semantic"]
    px, py = (int(v) for v in info["player_pos"])
    half_w, half_h = VIEW[0] // 2, VIEW[1] // 2 - 1  # the 9 x 7 local view drawn above Crafter's inventory rows
    width, height = semantic.shape
    nearest: dict[str, tuple[int, str]] = {}  # kept in BALROG's scan order: west to east, then north to south
    for dx in range(-half_w, half_w + 1):
        for dy in range(-half_h, half_h + 1):
            x, y = px + dx, py + dy
            if not (0 <= x < width and 0 <= y < height):  # the fix: cells outside the world are not described
                continue
            name = names[int(semantic[x, y])]
            if name == "player":
                continue
            if name not in nearest or abs(dx) + abs(dy) < nearest[name][0]:
                nearest[name] = (abs(dx) + abs(dy), describe_location(dx, dy))
    fx, fy = px + int(info["player_facing"][0]), py + int(info["player_facing"][1])
    front = names[int(semantic[fx, fy])] if 0 <= fx < width and 0 <= fy < height else "nothing"
    status = ("You are sleeping, and will not be able take actions until energy is full." if info["sleeping"]
              else "You died." if info["dead"] else "")
    return status, [f"{name} {loc}" for name, (_, loc) in nearest.items()], f"You face {front} at your front."


def describe_view(info: dict) -> str:
    status, seen, front = view_parts(info)
    listing = ("You see:\n" + "\n".join(f"- {s}" for s in seen)) if seen else "You see nothing away from you."
    return "\n\n".join(part for part in (status, listing, front) if part)


def describe_view_line(info: dict) -> str:
    """The same observation on one line, for the list of previous steps."""
    status, seen, front = view_parts(info)
    listing = ("You see: " + "; ".join(seen) + ".") if seen else "You see nothing away from you."
    return " ".join(part for part in (status, listing, front) if part)


def describe_status(inventory: dict) -> str:
    """BALROG's describe_inventory: the four vitals out of 9, then the non-empty inventory items."""
    vitals = "Your status:\n" + "\n".join(f"- {v}: {inventory[v]}/9" for v in VITALS)
    items = "\n".join(f"- {k}: {n}" for k, n in inventory.items() if k not in VITALS and n != 0)
    return vitals + "\n\n" + (f"Your inventory:\n{items}" if items else "You have nothing in your inventory.")


TASK = " ".join([
    "You are an agent playing Crafter, a two-dimensional open-world survival game on a grid of cells.",
    "These are the game achievements you can get: " + "; ".join(f"{i}. {a}" for i, a in enumerate(ACHIEVEMENTS, 1)) + ".",
    "Your goal is to get as far as possible by completing all the achievements.",
    "Each observation lists what you see in a window 9 cells wide and 7 cells tall centred on you: for each kind of thing, the "
    "nearest one with its distance in steps along the grid and its compass direction; then the thing in the cell you face. "
    "Your status (health, food, drink and energy, each from 0 to 9) and inventory are given for the current step.",
    f"The episode ends when your health reaches 0 or after {STEP_LIMIT} steps; your score is the percentage of the 22 "
    "achievements unlocked at least once. While you sleep, steps pass without a decision until you wake up.",
])


@register_env
class Crafter(Environment):
    name = "game/open_world/crafter"
    leaf = "game/open_world"
    source = "Crafter (Hafner, 2022; crafter 1.8.3), default world, through BALROG's text wrapper"
    url = "https://github.com/danijar/crafter"
    license = "MIT (Crafter); MIT (BALROG and its text wrapper)"
    tier = "B"
    criteria_source = ("BALROG's Crafter instruction (balrog/environments/crafter): the 17 actions with BALROG's descriptions "
                       "(verbatim) and the 22 achievements, 'PLAY!' removed; observation text from BALROG's language wrapper "
                       "(fixed at the world's west and north edges); one paragraph on the observation layout and the episode end added")
    notes = ("Seeds 0-9: crafter.Env(seed=seed), 64 x 64 world, BALROG's settings; the per-chunk creature sets are kept in "
             "insertion order so that the seed fixes the game (Crafter's own sets follow memory addresses). 2000 steps after "
             "BALROG's reset no-op, as in BALROG; steps taken while the player sleeps (all actions ignored by the game) are played "
             "automatically and are not decisions. "
             "The state shows the 15 previous steps and the current observation, as BALROG's 16-observation history. Score = BALROG's "
             "progression, % of the 22 achievements unlocked; no success flag (BALROG reports progression only).")
    mode = "paused"
    episodes = 10  # BALROG plays 10 Crafter episodes
    max_decisions = STEP_LIMIT
    score_name = "progression (% of 22 achievements)"

    def reset(self, seed: int) -> None:
        import crafter

        self.close()
        self.env = crafter.Env(area=AREA, view=VIEW, size=(64, 64), reward=True, seed=seed)  # size only sets the unused image
        self.env.reset()
        ordered_chunks(self.env._world)  # fixes the game for the seed (see the module notes)
        self.ended = False
        self.steps, self.total = 0, 0.0
        self._advance(0)  # BALROG's wrapper takes one Noop on reset to read the state; not counted, as in BALROG
        self.steps = self.decisions = self.asleep_steps = 0
        self.total = 0.0
        self.history: list[str] = []

    def _advance(self, action: int) -> None:
        _, reward, done, info = self.env.step(action)
        player = self.env._player
        info["sleeping"], info["player_facing"], info["dead"] = player.sleeping, player.facing, player.health <= 0
        self.info = info
        self.steps += 1
        self.total += float(reward)
        self.ended = self.ended or bool(done)

    def _asleep(self) -> bool:
        """The game replaces any action with 'sleep' until energy is full (crafter/objects.py, Player.update)."""
        player = self.env._player
        return player.sleeping and player.inventory["energy"] < 9

    def done(self) -> bool:
        return self.ended or self.steps >= STEP_LIMIT

    def observe(self) -> DecisionRequest:
        state = {"task": TASK,
                 "steps_taken": f"{self.steps} of at most {STEP_LIMIT}",
                 "previous_steps": "\n".join(self.history[-HISTORY:]) or "none: this is the first step",
                 "status_and_inventory": describe_status(self.info["inventory"]),
                 "observation": describe_view(self.info)}
        question = Question(type="choice", instructions="Choose your next action.",
                            criteria={name: text for name, (_, text) in ACTIONS.items()})
        return DecisionRequest(state=state, questions={"action": question})

    def act(self, choice: dict[str, str]) -> None:
        action, text = ACTIONS[choice["action"]]
        self.history.append(f"step {self.steps + 1}: observation: {describe_view_line(self.info)} Action: {text.split(':')[0]}.")
        self.decisions += 1
        self._advance(action)
        first = self.steps + 1
        while not self.done() and self._asleep():
            self._advance(0)
            self.asleep_steps += 1
        if self.steps >= first:
            self.history.append(f"steps {first}-{self.steps}: asleep (no decisions: the game ignores actions while you sleep).")

    def outcome(self) -> dict[str, Any]:
        player = self.env._player
        unlocked = sorted(name for name, count in player.achievements.items() if count > 0)
        return {"success": None, "score": round(100 * len(unlocked) / len(ACHIEVEMENTS), 2), "achievements": unlocked,
                "steps": self.steps, "decisions": self.decisions, "asleep_steps": self.asleep_steps,
                "died": player.health <= 0, "return": round(self.total, 2)}

    def close(self) -> None:
        self.env = None
