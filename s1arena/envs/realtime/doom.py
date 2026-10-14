"""First-person shooting in ViZDoom's standard scenarios, on a real-time clock.

Each scenario runs with its own configuration file (map, buttons, game variables, rewards, timeout,
difficulty). One env step is 4 game tics at Doom's 35 tics per second, so control_dt = 4/35 s; while
the model is deciding, the previous buttons stay pressed.

The state is the scenario's game variables (health, ammunition) plus the objects visible on screen,
read from ViZDoom's labels buffer: each object's name and category, its distance and bearing from the
player (computed from the object's and the player's map positions and the player's facing angle), and
its horizontal extent on the 320-pixel-wide screen. The player itself and the bullet-impact and blood
effect sprites are left out. The action is one choice over the button combinations of the scenario's
buttons, leaving out pairs of opposite buttons (which cancel out). Scores are the raw episode
returns defined by the scenario.

A scenario can have fewer bullets than its WAD gives (basic, version 2): the harness counts the shots
from the scenario's ammunition variable (each pistol shot uses one bullet), shows the bullets left
under the limit, and once they are spent it offers no combination with the shoot button and keeps
that button released, also while the model is deciding.
"""

from __future__ import annotations

import itertools
import math
import os
from typing import Any

from ...registry import register_env
from ...schema import DecisionRequest, Question
from ..base import Environment

URL = "https://vizdoom.farama.org/environments/default/"
LICENSE = ("MIT (ViZDoom; its ZDoom engine parts under their own licences); scenario WADs from ViZDoom; "
           "Freedoom assets BSD-3-Clause")
TICS = 4  # game tics per env step
TICRATE = 35  # Doom's fixed tics per second
SCREEN_WIDTH = 320
EFFECTS = {"BulletPuff", "Blood"}  # effect sprites, not game objects

BUTTONS = {"MOVE_FORWARD": "move forward", "MOVE_BACKWARD": "move backward", "MOVE_LEFT": "strafe left",
           "MOVE_RIGHT": "strafe right", "TURN_LEFT": "turn left", "TURN_RIGHT": "turn right", "ATTACK": "shoot"}
GROUPS = [("MOVE_FORWARD", "MOVE_BACKWARD"), ("MOVE_LEFT", "MOVE_RIGHT"), ("TURN_LEFT", "TURN_RIGHT"), ("ATTACK",)]
VARIABLES = {"HEALTH": "health", "AMMO2": "ammunition (pistol bullets)"}

INSTRUCTIONS = ("Choose the buttons to hold now. The game does not pause while you decide: until your answer takes effect, "
                "the previously chosen buttons stay pressed.")

CONTROLS = {"TURN_LEFT": "A held turn button rotates your view by about 120° per second.",
            "MOVE_LEFT": "Strafing moves you sideways without turning."}

LEGEND = ("Objects visible on screen, nearest first: name (category), distance in map units (a player is 56 units tall), "
          "bearing relative to the direction you are facing, and the columns it covers on the 320-pixel-wide screen "
          "(your shots go to the centre, column 160).")


def combinations(buttons: list[str]) -> dict[str, list[str]]:
    """Option id -> pressed buttons: every combination of the scenario's buttons except opposite pairs, fewest first."""
    groups = [[b for b in group if b in buttons] for group in GROUPS]
    choices = [[None] + group for group in groups if group]
    combos = sorted(([b for b in pick if b] for pick in itertools.product(*choices)), key=len)
    return {"+".join(c) if c else "NOOP": c for c in combos}


def describe(pressed: list[str]) -> str:
    if not pressed:
        return "No buttons pressed."
    words = [BUTTONS[b] for b in pressed]
    text = words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]
    return text[0].upper() + text[1:] + "."


class DoomScenario(Environment):
    leaf = "realtime/doom"
    url = URL
    license = LICENSE
    tier = "B"
    criteria_source = ("Scenario descriptions and rewards from the ViZDoom documentation (training-purpose remarks and advice "
                       "removed); rewards, endings, object names and turn rate checked in the simulator")
    mode = "realtime"
    control_dt = TICS / TICRATE
    episodes = 10
    score_name = "return"
    scenario = ""  # ViZDoom configuration name, e.g. "basic"
    goal = ""
    bullets: int | None = None  # pistol bullets allowed by the harness (None: the scenario's own ammunition)

    def reset(self, seed: int) -> None:
        import vizdoom as vzd

        game = vzd.DoomGame()
        game.load_config(os.path.join(vzd.scenarios_path, f"{self.scenario}.cfg"))
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_labels_buffer_enabled(True)
        game.set_seed(seed)
        game.init()  # starts the first episode with this seed
        self.game, self.vzd = game, vzd
        self.buttons = [b.name for b in game.get_available_buttons()]
        self.variables = [v.name for v in game.get_available_game_variables()]
        self.options = combinations(self.buttons)
        self.pressed = [0] * len(self.buttons)
        self.start = game.get_episode_start_time()
        self.limit_s = game.get_episode_timeout() / TICRATE  # counted from the episode start
        self.ammo_start = game.get_game_variable(vzd.GameVariable.AMMO2)
        self.total, self.steps = 0.0, 0

    def shots(self) -> int:
        """Pistol shots fired so far (bullets used)."""
        return int(self.ammo_start - self.game.get_game_variable(self.vzd.GameVariable.AMMO2))

    def spent(self) -> bool:
        return self.bullets is not None and self.shots() >= self.bullets

    def offered(self) -> dict[str, list[str]]:
        """The combinations the model may choose now: none with the shoot button once the bullets are spent."""
        if not self.spent():
            return self.options
        return {name: pressed for name, pressed in self.options.items() if "ATTACK" not in pressed}

    def _step(self) -> None:
        if self.spent():
            self.pressed = [0 if b == "ATTACK" else p for b, p in zip(self.buttons, self.pressed)]
        self.total += float(self.game.make_action(self.pressed, TICS))
        self.steps += 1

    def act(self, choice: dict[str, str]) -> None:
        held = set(self.options[choice["buttons"]])  # shoot is released in _step if the bullets ran out meanwhile
        self.pressed = [int(b in held) for b in self.buttons]
        self._step()

    def idle(self) -> None:
        self._step()

    def done(self) -> bool:
        return self.game.is_episode_finished()

    def default_choice(self, request: DecisionRequest) -> dict[str, str]:
        return {"buttons": "NOOP"}

    def outcome(self) -> dict[str, Any]:
        game, gv = self.game, self.vzd.GameVariable
        finished = bool(game.is_episode_finished())  # False only when the runner's decision cap stopped the episode
        timed_out = bool(game.is_episode_timeout_reached())
        died = bool(game.is_player_dead())
        out = {"success": None, "score": round(self.total, 3), "finished": finished, "died": died, "timed_out": timed_out,
               "kills": int(game.get_game_variable(gv.KILLCOUNT)), "health": int(game.get_game_variable(gv.HEALTH)),
               "sim_time_s": round(self.steps * self.control_dt, 2), "env_steps": self.steps}
        return {**out, **self.judge(out)}

    def judge(self, out: dict[str, Any]) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        game = getattr(self, "game", None)
        if game is not None:
            game.close()

    # --- request -------------------------------------------------------------------------------

    def task_parts(self) -> list[str]:
        return [self.goal, "Held buttons act continuously."] + [text for b, text in CONTROLS.items() if b in self.buttons]

    def observe(self) -> DecisionRequest:
        game, gv = self.game, self.vzd.GameVariable
        state = game.get_state()
        px, py = game.get_game_variable(gv.POSITION_X), game.get_game_variable(gv.POSITION_Y)
        facing = game.get_game_variable(gv.ANGLE)
        seen = []
        for label in state.labels:
            if label.object_category == "Self" or label.object_name in EFFECTS:
                continue
            dx, dy = label.object_position_x - px, label.object_position_y - py
            distance = math.hypot(dx, dy)
            bearing = (math.degrees(math.atan2(dy, dx)) - facing + 180.0) % 360.0 - 180.0  # positive = to the left
            side = "straight ahead" if abs(bearing) < 0.5 else f"{abs(bearing):.0f}° to your {'left' if bearing > 0 else 'right'}"
            span = f"{max(0, label.x)}–{min(SCREEN_WIDTH - 1, label.x + label.width - 1)}"
            seen.append((distance, f"- {label.object_name} ({label.object_category}): {distance:.0f} units away, {side}; "
                                   f"screen columns {span}"))
        rows = [text for _, text in sorted(seen, key=lambda item: item[0])]
        values = [(name, max(0, self.bullets - self.shots()) if name == "AMMO2" and self.bullets is not None else value)
                  for name, value in zip(self.variables, state.game_variables)]
        you = ", ".join(f"{VARIABLES.get(name, name.lower())} {value:.0f}" for name, value in values)
        elapsed = (state.tic - self.start) / TICRATE
        request_state = {
            "task": " ".join(self.task_parts()),
            "time": f"{elapsed:.1f} s of {self.limit_s:.1f} s",
            "you": you or "(no game variables in this scenario)",
            "visible": LEGEND + "\n" + ("\n".join(rows) if rows else "(nothing)"),
        }
        options = {name: describe(pressed) for name, pressed in self.offered().items()}
        question = Question(type="choice", instructions=INSTRUCTIONS, criteria=options)
        return DecisionRequest(state=request_state, questions={"buttons": question})


@register_env
class Basic(DoomScenario):
    name = "realtime/doom/basic"
    source = "ViZDoom basic scenario (Kempka et al., 2016)"
    scenario = "basic"
    max_decisions = 75
    bullets = 1
    version = 2
    notes = ("One monster at a random spot on the opposite wall; buttons: strafe left/right, shoot (6 combinations); the "
             "scenario's timeout of 300 tics (8.6 s, 75 env steps). Success = the monster is killed. Version 2 (decision "
             "#91: a random policy killed the monster in 7 of 10 version-1 episodes by strafing while firing the "
             "scenario's 50 bullets): 1 bullet, so the monster has to be lined up with the screen centre before the one "
             "shot. The harness counts shots from the scenario's ammunition variable; once the bullet is spent it offers "
             "only the 3 combinations without the shoot button and keeps that button released, and the episode runs on "
             "to the timeout. Map, spawn, timeout and rewards are the scenario's own. With 3 bullets a random policy "
             "still won 32% of 1,000 seeds (a shoot button held through a 320 ms decision fires twice, and in 16.5% of "
             "seeds the monster spawns within its radius of the line of fire); with 1 bullet it wins about 17%, close to "
             "that floor, which no bullet limit can lower.")
    goal = ("Basic. The map is a rectangle with gray walls, ceiling and floor. You are spawned along the longer wall in the "
            "center, facing the opposite wall. A red, circular monster (Cacodemon) is spawned randomly somewhere along the "
            "opposite wall. You can only move left/right and shoot. 1 hit is enough to kill the monster. You have 1 "
            "bullet; once it is fired you can no longer shoot. The episode finishes when the monster is killed or on "
            "timeout (300 tics, 8.6 s). Rewards: +106 for killing the monster, -5 for every shot, -1 for every tic (1/35 s) "
            "you are alive.")

    def judge(self, out):
        return {"success": out["kills"] > 0, "shots": self.shots()}


@register_env
class DefendTheCenter(DoomScenario):
    name = "realtime/doom/defend_the_center"
    source = "ViZDoom defend_the_center scenario (Kempka et al., 2016)"
    scenario = "defend_the_center"
    max_decisions = 525
    notes = ("Melee monsters approach from all around a circular room; buttons: turn left/right, shoot (6 combinations); "
             "ends at death or on the scenario's timeout of 2,100 tics (60 s, 525 env steps). Score = kills - 1 if killed; "
             "no success criterion (the scenario defines none).")
    goal = ("Defend the center. The map is a large circle. You are spawned in the exact center. 5 melee-only monsters "
            "(Demon, MarineChainsawVzd) are spawned along the wall. Monsters are killed after a single shot. After dying, "
            "each monster is respawned after some time. Your ammunition is limited. You can only turn left/right and shoot. "
            "The episode ends when you die or on timeout (2,100 tics, 60 s). Rewards: +1 for killing a monster, -1 for death.")


@register_env
class HealthGathering(DoomScenario):
    name = "realtime/doom/health_gathering"
    source = "ViZDoom health_gathering scenario (Kempka et al., 2016)"
    scenario = "health_gathering"
    max_decisions = 525
    notes = ("Acid floor drains health; medkits heal; buttons: turn left/right, move forward (6 combinations); ends at "
             "death or on the scenario's timeout of 2,100 tics (60 s, 525 env steps). Success = alive at the timeout.")
    goal = ("Health gathering. The map is a rectangle with a green, acidic floor, which hurts you periodically. Initially, "
            "there are some medkits (Medikit) spread uniformly over the map. A new medkit falls from the skies every now and "
            "then. Medkits heal some portion of your health; you pick one up by walking over it. You can only turn "
            "left/right and move forward. The episode finishes after your death or on timeout (2,100 tics, 60 s). Rewards: "
            "+1 for every tic (1/35 s) you are alive, -100 for death.")

    def judge(self, out):
        return {"success": out["timed_out"] and not out["died"]}


@register_env
class DeadlyCorridor(DoomScenario):
    name = "realtime/doom/deadly_corridor"
    source = "ViZDoom deadly_corridor scenario (Kempka et al., 2016)"
    scenario = "deadly_corridor"
    max_decisions = 525
    notes = ("A corridor lined with 6 shooting monsters at the hardest skill level, a vest at the far end; 7 buttons (54 "
             "combinations); ends on reaching the vest, at death or on the scenario's timeout of 2,100 tics (60 s, 525 env "
             "steps). Success = reached the vest.")
    goal = ("Deadly corridor. The goal is to reach the vest at the end of the corridor and survive. The map is a corridor "
            "with shooting monsters on both sides (6 monsters in total: Zombieman, ShotgunGuy, ChaingunGuy). A green vest "
            "(GreenArmor) is placed at the opposite end of the corridor; you start at the other end, facing it. The reward "
            "is proportional (negative or positive) to the change in the distance between you and the vest: +dX for getting "
            "closer to the vest, -dX for getting further from it, +1000 for reaching the vest (which ends the episode), -100 "
            "for death. The episode also ends on timeout (2,100 tics, 60 s). Difficulty: skill level 5 of 5.")

    def judge(self, out):
        return {"success": out["finished"] and not out["died"] and not out["timed_out"]}
