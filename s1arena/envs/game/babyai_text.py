"""BabyAI levels (Minigrid) played through text observations, as in BALROG's BabyAI-Text setting.

BALROG (Paglieri et al., ICLR 2025) evaluates agents on BabyAI through the BabyAI-Text descriptions
of Carta et al. (ICML 2023): one line per thing in the agent's 7x7 egocentric view ("a red ball
1 step left and 2 steps forward"), the nearest wall straight ahead, left and right, and what the
agent carries. describe_view() ports that generator (gen_graph in BALROG's Minigrid fork, English)
to the official Minigrid package, and BALROG's prompt drops the leading "You see ". The task text
reuses BALROG's instruction ("You are an agent playing a simple navigation game. Your goal is to
<mission>.") and its six actions with their descriptions, without BALROG's tips (strategy hints)
and with two corrections: pick up takes the object in front of the agent (BALROG says "below
you"), and "an open door" instead of the generator's "a open door".

Like BALROG's agent, which keeps the last 16 observations in its context, the state lists the 15
previous steps (observation and action) before the current observation. Mission terms whose
meaning is fixed by the level's verifier ('go to', 'next to', locations such as 'on your left')
are defined in the task text, from the verifier's own docstrings. Score = the environment's
reward, 1 - 0.9 * steps / step limit on success and 0 otherwise; success = mission accomplished.

Version 2 (see budget.py): BabyAI's step limits let random walks accomplish the
one-room missions (50% on GoToLocal, 30% on GoToObj and PickupLoc). The level's step limit is now
set per seed to max(2 L*, 6) and at most its own limit, where L* is the number of steps BabyAI's
bot (Minigrid's BabyAIBot, which plans from what the agent has seen and explores the rest, as the
agent must) takes to accomplish the seed's mission. The level enforces the new limit and uses it in
its reward, and the task text and the step counter state it.
"""

from __future__ import annotations

import contextlib
import functools
import sys
from typing import Any

from ...registry import register_env
from ...schema import DecisionRequest, Question
from ..base import Environment
from .budget import decision_budget

ACTIONS = {  # option id -> (Minigrid action, BALROG's name and description, corrected and completed with the rules)
    "left": (0, "turn left: turn to the left (you stay on the same cell)"),
    "right": (1, "turn right: turn to the right (you stay on the same cell)"),
    "forward": (2, "go forward: take one step forward (walls, closed doors and objects block the way)"),
    "pickup": (3, "pick up: pick up the object in front of you (you can carry one object at a time)"),
    "drop": (4, "drop: drop the object that you are holding onto the empty cell in front of you"),
    "toggle": (5, "toggle: manipulate the object in front of you: open or close a door (a locked door opens only while you "
                  "carry a key of the same colour); a box opens and disappears"),
}
HISTORY = 15  # previous steps shown; with the current observation, BALROG's max_text_history of 16
STATES = {0: "open", 1: "closed", 2: "locked"}
NOTHING = "no objects in view and no wall straight ahead, left or right"


def describe_view(env) -> list[str]:
    """BabyAI-Text description of what the agent sees (Carta et al., 2023; BALROG's Minigrid fork), 'You see ' dropped."""
    from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX

    colors = {v: k for k, v in COLOR_TO_IDX.items()}
    objects = {v: k for k, v in OBJECT_TO_IDX.items()}
    grid, vis_mask = env.gen_obs_grid()
    image = grid.encode(vis_mask)  # [column][row] -> (object, colour, state); 0 unseen, 1 empty, 2 wall
    ax, ay = env.get_view_coords(*env.agent_pos)  # the agent's cell in the view: bottom row, middle column
    lines = []
    if env.carrying is not None:
        lines.append(f"You carry a {env.carrying.color} {env.carrying.type}")

    def steps(n: int, direction: str) -> str:
        return f"{n} step{'s' if n > 1 else ''} {direction}"

    # a wall is described only when it is the first thing in view straight ahead, left or right
    rays = (("forward", [(ax, j) for j in range(ay - 1, -1, -1)]),
            ("left", [(i, ay) for i in range(ax - 1, -1, -1)]),
            ("right", [(i, ay) for i in range(ax + 1, image.shape[0])]))
    for direction, cells in rays:
        for n, (i, j) in enumerate(cells, 1):
            if image[i][j][0] not in (0, 1):
                if image[i][j][0] == 2:
                    lines.append(f"a wall {steps(n, direction)}")
                break
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            kind, color, state = (int(v) for v in image[i][j])
            if kind in (0, 1, 2) or (i, j) == (ax, ay):
                continue
            dx, forward = i - ax, ay - j
            side = steps(abs(dx), "right" if dx > 0 else "left")
            where = steps(forward, "forward") if dx == 0 else side if forward == 0 else f"{side} and {steps(forward, 'forward')}"
            if objects[kind] == "door":
                article = "an" if STATES[state] == "open" else "a"
                lines.append(f"{article} {STATES[state]} {colors[color]} door {where}")
            else:
                lines.append(f"a {colors[color]} {objects[kind]} {where}")
    return lines


def mission_terms(mission: str) -> list[str]:
    """Definitions from the BabyAI verifier (minigrid/envs/babyai/core/verifier.py) for the terms this mission uses."""
    out = []
    if "go to" in mission:
        out.append("'Go to' an object means being next to it and facing it (the object is directly in front of you).")
    if "next to" in mission:
        out.append("'Put X next to Y' means X ends up on a cell directly beside Y, in line horizontally or vertically "
                   "(diagonal cells do not count); you move X by picking it up and dropping it.")
    if any(loc in mission for loc in ("in front of you", "behind you", "on your left", "on your right")):
        out.append("'In front of you', 'behind you', 'on your left' and 'on your right' describe where an object was relative "
                   "to your position and facing direction at the start of the episode, among the objects in the room you "
                   "started in.")
    return out


@functools.cache
def bot_steps(level_id: str, seed: int) -> int | None:
    """Steps BabyAI's bot takes to accomplish the seed's mission, or None when it does not within the level's limit."""
    import gymnasium as gym
    from minigrid.utils.baby_ai_bot import BabyAIBot

    env = gym.make(level_id, disable_env_checker=True)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            env.reset(seed=seed)
        bot, steps = BabyAIBot(env), 0
        while True:
            _, reward, terminated, truncated, _ = env.step(bot.replan())
            steps += 1
            if terminated or truncated:
                return steps if terminated and reward > 0 else None
    except Exception:  # the bot gives up on some layouts (e.g. after opening a box); the level then keeps its own limit
        return None
    finally:
        env.close()


class BabyAILevel(Environment):
    leaf = "game/roguelike"
    url = "https://minigrid.farama.org/environments/babyai/"
    license = "Apache-2.0 (Minigrid, BabyAI levels); MIT (BALROG and BabyAI-Text text wrapper)"
    tier = "B"
    criteria_source = ("Mission strings generated by the level (verbatim); instruction sentence and action list from BALROG's "
                       "BabyAI prompt (balrog/environments/babyai_text), tips removed and the pick-up description corrected; "
                       "observation text from BabyAI-Text; mission terms from the BabyAI verifier docstrings")
    mode = "paused"
    episodes = 10  # BALROG plays 10 episodes per BabyAI task
    score_name = "reward"
    version = 2  # per-seed step limit
    level_id = ""  # Gymnasium id of the Minigrid BabyAI level

    def reset(self, seed: int) -> None:
        import gymnasium as gym
        import minigrid  # noqa: F401  (registers the BabyAI levels)

        self.close()
        self.env = gym.make(self.level_id, disable_env_checker=True)
        with contextlib.redirect_stdout(sys.stderr):  # the level generator prints its rejection-sampling retries
            obs, _ = self.env.reset(seed=seed)
        self.game = self.env.unwrapped
        self.shortest = bot_steps(self.level_id, seed)
        self.level_max_steps = self.game.max_steps
        self.game.max_steps = decision_budget(self.shortest, min(self.level_max_steps, self.max_decisions))
        self.mission = obs["mission"]
        self.view = describe_view(self.game)
        self.history: list[tuple[int, list[str], str]] = []
        self.total, self.ended, self.succeeded = 0.0, False, False

    def task_text(self) -> str:
        max_steps = self.game.max_steps
        parts = [f"You are an agent playing a simple navigation game. Your goal is to {self.mission}.",
                 "The world is a grid of cells with walls, doors, keys, balls and boxes. You see the cells ahead of you: your "
                 "own row and 6 rows forward, 3 columns to each side; walls and closed doors block the view. Each observation "
                 "lists the objects you see with their positions in steps relative to you (left or right, and forward), the "
                 "nearest wall straight ahead, to your left and to your right when it is in view, and what you carry.",
                 *mission_terms(self.mission),
                 f"The episode ends when the goal is achieved, with reward 1 - 0.9 x (steps taken / {max_steps}), or after "
                 f"{max_steps} steps with reward 0."]
        return " ".join(parts)

    def observe(self) -> DecisionRequest:
        earlier = [f"step {t}: observation: {'; '.join(view) or NOTHING}. Action: {action}."
                   for t, view, action in self.history[-HISTORY:]]
        state = {"task": self.task_text(),
                 "steps_taken": f"{self.game.step_count} of at most {self.game.max_steps}",
                 "previous_steps": "\n".join(earlier) or "none: this is the first step",
                 "observation": "\n".join(self.view) or NOTHING}
        question = Question(type="choice", instructions="Choose your next action.",
                            criteria={name: text for name, (_, text) in ACTIONS.items()})
        return DecisionRequest(state=state, questions={"action": question})

    def act(self, choice: dict[str, str]) -> None:
        action, text = ACTIONS[choice["action"]]
        self.history.append((self.game.step_count + 1, self.view, text.split(":")[0]))
        _, reward, terminated, truncated, _ = self.env.step(action)
        self.total += float(reward)
        self.succeeded = self.succeeded or (bool(terminated) and reward > 0)
        self.ended = bool(terminated) or bool(truncated)
        self.view = describe_view(self.game)

    def done(self) -> bool:
        return self.ended

    def outcome(self) -> dict[str, Any]:
        return {"success": self.succeeded, "score": round(self.total, 4), "steps": self.game.step_count,
                "max_steps": self.game.max_steps, "shortest": self.shortest, "level_max_steps": self.level_max_steps,
                "mission": self.mission}

    def close(self) -> None:
        env = getattr(self, "env", None)
        if env is not None:
            env.close()
            self.env = None


@register_env
class GoToObj(BabyAILevel):
    name = "game/roguelike/babyai_goto_obj"
    source = "BabyAI GoToObj (BabyAI-GoToObj-v0, Minigrid)"
    level_id = "BabyAI-GoToObj-v0"
    max_decisions = 64
    notes = ("Easiest level: one room (6x6 free cells), one object and no distractors; go to it. Step limit: twice the "
             "bot's steps for the seed, 6 to 64.")


@register_env
class GoToLocal(BabyAILevel):
    name = "game/roguelike/babyai_goto_local"
    source = "BabyAI GoToLocal (BabyAI-GoToLocal-v0, Minigrid)"
    level_id = "BabyAI-GoToLocal-v0"
    max_decisions = 64
    notes = ("One room with 8 objects; go to the one the mission names ('a' when several match). Step limit: twice the "
             "bot's steps for the seed, 6 to 64.")


@register_env
class PickupLoc(BabyAILevel):
    name = "game/roguelike/babyai_pickup_loc"
    source = "BabyAI PickupLoc (BabyAI-PickupLoc-v0, Minigrid)"
    level_id = "BabyAI-PickupLoc-v0"
    max_decisions = 64
    notes = ("One room with 8 objects; pick up the named one, which may be given by its location at the start. Step limit: "
             "twice the bot's steps for the seed, 6 to 64.")


@register_env
class PutNextLocal(BabyAILevel):
    name = "game/roguelike/babyai_put_next_local"
    source = "BabyAI PutNextLocal (BabyAI-PutNextLocal-v0, Minigrid)"
    level_id = "BabyAI-PutNextLocal-v0"
    max_decisions = 128
    notes = ("One room with 8 objects; move one object next to another (pick up, carry, drop). Step limit: twice the bot's "
             "steps for the seed, 6 to 128.")


@register_env
class UnlockPickup(BabyAILevel):
    name = "game/roguelike/babyai_unlock_pickup"
    source = "BabyAI UnlockPickup (BabyAI-UnlockPickup-v0, Minigrid)"
    level_id = "BabyAI-UnlockPickup-v0"
    max_decisions = 72
    notes = ("Two rooms; the box to pick up is behind a locked door whose key lies in the starting room, so the key must be "
             "fetched, the door unlocked and the key dropped before the box can be picked up. Step limit: twice the bot's steps "
             "for the seed, 6 to 72.")


@register_env
class MiniBossLevel(BabyAILevel):
    name = "game/roguelike/babyai_mini_boss_level"
    source = "BabyAI MiniBossLevel (BabyAI-MiniBossLevel-v0, Minigrid)"
    level_id = "BabyAI-MiniBossLevel-v0"
    max_decisions = 800
    notes = ("Hardest level: the full BabyAI instruction language (go to, pick up, open, put next to, locations, sequences "
             "with 'and', 'then', 'after you') in a 2x2 maze of small rooms with closed and sometimes locked doors. Step limit: "
             "twice the bot's steps for the seed, at most the level's own limit of 100 per navigation the mission needs (100 to "
             "600 over seeds 0-9).")
