"""Atari 2600 games through OCAtari, with the object list read from the console RAM, on a real-time clock.

Each game runs in the Arcade Learning Environment (ALE v5 settings: minimal action set, 4 frames per
env step, sticky actions with probability 0.25 seeded by the episode seed). One env step is 4 frames
at 60 frames per second, so control_dt = 1/15 s; while the model is deciding, the previous action
stays pressed, as a held joystick would.

The state is OCAtari's RAM-extracted object list (class, bounding box in screen pixels and, for
moving objects, the velocity over the last env step) plus the game variables shown on screen (score,
lives), read from the RAM addresses the ALE uses for its rewards. OCAtari's HUD objects are not used:
Boxing's live in module-level globals shared by parallel episodes and hold the BCD-coded score
undecoded. The actions are the game's minimal action set, with each action's effect (checked in the
emulator). Scores are the raw game scores (the sum of ALE rewards) within a bounded time: the
game's own length where it has one (Freeway, Boxing), otherwise two minutes of game time.
"""

from __future__ import annotations

from typing import Any

from ...registry import register_env
from ...schema import DecisionRequest, Question
from ..base import Environment

URL = "https://github.com/k4ntz/OC_Atari"
LICENSE = "MIT (OCAtari); GPL-2.0 (Arcade Learning Environment); game ROMs as distributed with ale-py"
FRAMESKIP = 4
FPS = 60
MAX_STEP_PX = 40  # a larger jump between two steps is a wrap-around or respawn, not a velocity

INSTRUCTIONS = ("Choose the input for the next moments of play. The game does not pause while you decide: until your "
                "answer takes effect, the previous input stays pressed.")

LEGEND = ("Screen coordinates in pixels (160 wide, 210 high): x from the left edge, y from the top edge, for the top-left "
          "corner of each object's bounding box, then its width × height. Velocity: movement over the last 1/15 s, in "
          "pixels per second (x positive = rightwards, y positive = downwards); omitted for objects that do not move and "
          "when unknown (an object that just appeared, wrapped around the screen or was reset).")


class OCAtariGame(Environment):
    leaf = "realtime/atari"
    url = URL
    license = LICENSE
    tier = "B"
    criteria_source = ("Game descriptions adapted from the ALE documentation and the Atari manuals (strategy advice removed); "
                       "action effects, object meanings and rewards checked in the emulator")
    mode = "realtime"
    control_dt = FRAMESKIP / FPS  # one env step = 4 frames at 60 fps
    episodes = 10
    score_name = "game score"
    game = ""  # ALE game name, e.g. "Pong"
    game_mode: int | None = None  # ALE game mode (a cartridge's game variation); None keeps the default game
    time_limit = 120.0  # seconds of game time
    goal = ""
    actions: dict[str, str] = {}  # ALE action name -> effect, in the game's minimal action set order
    movers: tuple[str, ...] = ()  # OCAtari classes whose velocity is shown

    def reset(self, seed: int) -> None:
        import ale_py
        from ocatari.core import OCAtari

        ale_py.ALEInterface.setLoggerMode(ale_py.LoggerMode.Error)  # no ROM banner per episode
        self.env = OCAtari(f"ALE/{self.game}-v5", mode="ram", hud=False, obs_mode="obj", create_buffer_stacks=[],
                           frameskip=FRAMESKIP, repeat_action_probability=0.25, full_action_space=False)
        if self.game_mode is not None:  # OCAtari's own `mode` argument (ram/vision) hides the ALE's; the seeded reset applies it
            self.env._env.unwrapped._game_mode = self.game_mode
        self.env.reset(seed=seed)
        self.ale = self.env._env.unwrapped.ale
        self.names = list(self.env.get_action_meanings())
        if self.names != list(self.actions):
            raise RuntimeError(f"{self.game}: unexpected action set {self.names}")
        self.index = {name: i for i, name in enumerate(self.names)}
        self.action = self.index["NOOP"]
        self.total, self.steps, self.ended, self.game_over = 0.0, 0, False, False
        self.lives0 = self.ale.lives()

    def _step(self) -> None:
        _, reward, first, second, _ = self.env.step(self.action)  # OCAtari returns (truncated, terminated) swapped
        self.total += float(reward)
        self.steps += 1
        self.game_over = self.game_over or bool(first) or bool(second)
        self.ended = self.game_over or self.steps * self.control_dt >= self.time_limit - 1e-9

    def act(self, choice: dict[str, str]) -> None:
        self.action = self.index[choice["action"]]
        self._step()

    def idle(self) -> None:
        self._step()

    def done(self) -> bool:
        return self.ended

    def default_choice(self, request: DecisionRequest) -> dict[str, str]:
        return {"action": "NOOP"}

    def outcome(self) -> dict[str, Any]:
        return {"success": None, "score": self.total, "game_over": self.game_over,
                "sim_time_s": round(self.steps * self.control_dt, 2), "env_steps": self.steps, **self.extra()}

    def extra(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        env = getattr(self, "env", None)
        if env is not None:
            env.close()

    # --- request -------------------------------------------------------------------------------

    def observe(self) -> DecisionRequest:
        rows = [self.row(i, o) for i, o in enumerate(self.env.objects) if o]
        state = {
            "task": self.goal,
            "time": f"{self.steps * self.control_dt:.1f} s of {self.time_limit:g} s",
            "game": self.variables(),
            "objects": LEGEND + "\n" + "\n".join(r for r in rows if r),
        }
        question = Question(type="choice", instructions=INSTRUCTIONS, criteria=dict(self.actions))
        return DecisionRequest(state=state, questions={"action": question})

    def row(self, slot: int, o) -> str:
        text = f"- {self.label(slot, o)}: x={o.x}, y={o.y}, {o.w}×{o.h}"
        if o.category in self.movers:
            known = self.steps > 0 and getattr(o, "_prev_xy", None) is not None
            if known and abs(o.dx) <= MAX_STEP_PX and abs(o.dy) <= MAX_STEP_PX:
                text += f", velocity ({o.dx / self.control_dt:+.0f}, {o.dy / self.control_dt:+.0f}) px/s"
        return text

    def label(self, slot: int, o) -> str:
        return o.category

    def ram(self, address: int, bcd: bool = False) -> int:
        value = int(self.ale.getRAM()[address])
        return (value >> 4) * 10 + (value & 0xF) if bcd else value

    def variables(self) -> str:
        raise NotImplementedError


@register_env
class Pong(OCAtariGame):
    name = "realtime/atari/pong"
    source = "Pong (Atari 2600) in the Arcade Learning Environment, objects from OCAtari"
    game = "Pong"
    time_limit = 120.0
    max_decisions = 1800
    notes = ("Two minutes of game time (1,800 env steps of 4 frames) or until one side reaches 21 points. Score = points won "
             "minus points lost (the ALE reward); both are reported. No success criterion.")
    goal = ("Pong. You control the right paddle, you compete against the left paddle controlled by the computer. You each "
            "try to keep deflecting the ball away from your goal and into your opponent's goal. You get score points for "
            "getting the ball to pass the opponent's paddle. You lose points if the ball passes your paddle. The game ends "
            "when either side reaches 21 points. The ball bounces off the top and bottom walls; your paddle's y ranges "
            "from 34 (top) to 190 (bottom). Score for this episode: points won minus points lost within the time limit. "
            "Objects: Player = your paddle (right, x = 140), Enemy = the computer's paddle (left, x = 16), Ball = the ball "
            "(absent between points).")
    actions = {"NOOP": "No input.", "FIRE": "Press the button (it does not move the paddle).",
               "RIGHT": "Move your paddle up.", "LEFT": "Move your paddle down.",
               "RIGHTFIRE": "Move your paddle up while pressing the button.",
               "LEFTFIRE": "Move your paddle down while pressing the button."}
    movers = ("Player", "Enemy", "Ball")

    def points(self) -> tuple[int, int]:
        return self.ram(14), self.ram(13)  # as in the ALE's Pong reward

    def variables(self) -> str:
        won, lost = self.points()
        return f"score: you {won}, computer {lost}"

    def extra(self) -> dict[str, Any]:
        won, lost = self.points()
        return {"points_won": won, "points_lost": lost}


@register_env
class Breakout(OCAtariGame):
    name = "realtime/atari/breakout"
    source = "Breakout (Atari 2600) in the Arcade Learning Environment, objects from OCAtari"
    game = "Breakout"
    time_limit = 120.0
    max_decisions = 1800
    notes = ("Two minutes of game time (1,800 env steps of 4 frames) or until the fifth ball is lost. Score = points from "
             "destroyed bricks. No success criterion.")
    goal = ("Breakout. You move a paddle and hit the ball in a brick wall at the top of the screen. Your goal is to "
            "destroy the brick wall. You have five lives (balls). You score points by destroying bricks in the wall; the "
            "reward for destroying a brick depends on its row: 1 point in the two bottom rows, 4 in the two middle rows, 7 "
            "in the two top rows. A ball that passes your paddle is lost; a new ball is launched with FIRE. The ball "
            "bounces off the side walls and the top wall; the paddle's x ranges from 8 to 144. Objects: Player = your "
            "paddle, Ball = the ball (absent until launched), Block = a run of adjacent bricks in one row (each brick is 8 "
            "pixels wide; rows from y = 57 at the top to y = 87 at the bottom).")
    actions = {"NOOP": "No input.", "FIRE": "Launch a new ball (when none is in play).",
               "RIGHT": "Move the paddle right.", "LEFT": "Move the paddle left."}
    movers = ("Player", "Ball")

    def variables(self) -> str:
        return f"score {self.total:.0f}; balls left {self.ale.lives()}"

    def extra(self) -> dict[str, Any]:
        return {"balls_lost": self.lives0 - self.ale.lives()}


@register_env
class Freeway(OCAtariGame):
    name = "realtime/atari/freeway"
    source = "Freeway (Atari 2600) in the Arcade Learning Environment, objects from OCAtari"
    game = "Freeway"
    version = 2  # version 1: the default game, in which pressing up all the time already scores 21
    game_mode = 3  # the cartridge's game 4: the cars move 2-8 pixels per env step instead of 0-4 in the default game
    time_limit = 136.6  # the game's own end (2 min 16 s on its clock) comes after 2,048 env steps (136.5 s)
    max_decisions = 2049
    notes = ("The game's own length (2 min 16 s on the game clock: 2,048 env steps of 4 frames, 136.5 s). Score = number of "
             "crossings. No success criterion. ALE game mode 3 (the cartridge's game 4), whose traffic is two to four times "
             "faster than the default game's, so the chicken has to wait for gaps: pressing up all the time scores 5 instead "
             "of 21, while a scripted policy that waits for gaps scores 23.8 on average over the 10 seeds at a 320 ms "
             "decision latency and 18.5 at 1,270 ms (scripts/realtime_oracle.py).")
    goal = ("Freeway. Your objective is to guide your chicken across lane after lane of busy rush hour traffic. You "
            "receive a point for every chicken that makes it to the top of the screen after crossing all the lanes of "
            "traffic (ten lanes); the chicken then starts again at the bottom. A car that hits the chicken pushes it back "
            "down. The game lasts 2 minutes 16 seconds. Objects: the Chicken at x = 44 is yours; the Chicken at x = 108 "
            "belongs to the second player and does not move in this one-player game; each Car drives along its own lane.")
    actions = {"NOOP": "Stand still.", "UP": "Move the chicken up (forward).", "DOWN": "Move the chicken down (back)."}
    movers = ("Chicken", "Car")

    def label(self, slot: int, o) -> str:
        if o.category == "Chicken":
            return "Chicken (yours)" if slot == 0 else "Chicken (second player's)"
        return o.category

    def variables(self) -> str:
        return f"score {self.total:.0f}"

    def extra(self) -> dict[str, Any]:
        return {"crossings": int(self.total)}


@register_env
class Boxing(OCAtariGame):
    name = "realtime/atari/boxing"
    source = "Boxing (Atari 2600) in the Arcade Learning Environment, objects from OCAtari"
    game = "Boxing"
    time_limit = 119.1  # the bout's 2-minute clock runs out after 1,786 env steps (119.07 s) unless a knockout ends it
    max_decisions = 1787
    notes = ("The game's own length (a 2-minute bout on the game clock: 1,786 env steps of 4 frames, 119.1 s; a knockout at "
             "100 points ends it early). Score = your points minus the computer's points (the ALE reward); both are "
             "reported. No success criterion.")
    goal = ("Boxing. You fight an opponent in a boxing ring, seen from above. You score points for hitting the opponent (1 "
            "point for a long jab, 2 for a close power punch), and the opponent, played by the computer, scores on you the "
            "same way. If you score 100 points, your opponent is knocked out, which ends the bout; otherwise the bout "
            "lasts 2 minutes. Score for this episode: your points minus the computer's points. Objects: Player = your "
            "boxer (white), Enemy = the computer's boxer (black).")
    actions = {
        "NOOP": "No input.", "FIRE": "Punch.", "UP": "Move up.", "RIGHT": "Move right.", "LEFT": "Move left.",
        "DOWN": "Move down.", "UPRIGHT": "Move up and right.", "UPLEFT": "Move up and left.",
        "DOWNRIGHT": "Move down and right.", "DOWNLEFT": "Move down and left.", "UPFIRE": "Punch while moving up.",
        "RIGHTFIRE": "Punch while moving right.", "LEFTFIRE": "Punch while moving left.",
        "DOWNFIRE": "Punch while moving down.", "UPRIGHTFIRE": "Punch while moving up and right.",
        "UPLEFTFIRE": "Punch while moving up and left.", "DOWNRIGHTFIRE": "Punch while moving down and right.",
        "DOWNLEFTFIRE": "Punch while moving down and left.",
    }
    movers = ("Player", "Enemy")

    def points(self) -> tuple[int, int]:
        return self.ram(18, bcd=True), self.ram(19, bcd=True)  # as in the ALE's Boxing reward

    def variables(self) -> str:
        won, lost = self.points()
        return f"score: you {won}, computer {lost}"

    def extra(self) -> dict[str, Any]:
        won, lost = self.points()
        return {"points_won": won, "points_lost": lost}
