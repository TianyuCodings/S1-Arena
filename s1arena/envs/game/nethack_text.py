"""MiniHack tasks and NetHack (NLE) played through BALROG's NetHack language wrapper.

BALROG (Paglieri et al., ICLR 2025) plays the NetHack Learning Environment (Küttler et al., 2020)
and eight MiniHack tasks (Samvelyan et al., 2021) through its own NLE fork, balrog-nle 0.9.0, which
bundles the NetHack Language Wrapper (Goodger et al., 2023). For text-only agents BALROG's "hybrid"
observation (balrog/environments/nle/base.py) is: the game's message, the wrapper's language
observation (what is in view and in which direction), what is under the cursor, and the terminal
screen (map and status lines) minus its top line; the inventory is shown for the current step.
Messages ending in --More-- are continued automatically (BALROG's AutoMore wrapper), and those key
presses are not decisions. observe() reproduces that text, with trailing spaces of screen rows removed.

Actions are BALROG's lists with BALROG's descriptions, one choice question per key press:
- MiniHack: the actions of BALROG's MiniHack prompt that the task allows (4 to 32).
- NetHack: the 80 commands BALROG's NetHack prompt describes, less 'overview' (not in NLE's action
  set), plus the 52 letter keys BALROG's action space adds for answering prompts ("What do you want
  to eat? [fg or ?*]"): 131 options. BALROG's space also lists the 100 strings 00-99, which its
  language wrapper cannot map to a key press (they raise an error), and 22 undescribed commands
  that the prompt never shows; both are left out. A count such as 20 is typed as 2, then 0.
  Two strategy tips inside descriptions are dropped (eat: "replenish food when hungry", engrave:
  "Elbereth"), and the instruction's tips are reduced to how prompts are answered. The failed-answer
  default is BALROG's default action for NetHack, esc (for MiniHack, north: the first option).

Episodes, limits and seeds follow BALROG's config: MiniHack tasks with character '@' (random role
unless the task fixes one), autopickup off and a 100-step limit; NetHackChallenge-v0 with a random
character and BALROG's no-progress timeout (the episode is aborted after 150 key presses without the
game clock advancing). BALROG caps NetHack at 100,000 steps; here the cap is 2000 decisions (see
the environment notes). Seeds set NetHack's core and display generators (reseeding off), so the level
and the game are fixed by the seed; Boxoban levels, which MiniHack draws with Python's global random
generator, are chosen from the seed instead. NetHack also reads the machine's clock (full moon: +1
Luck and a message at the start; new moon; Friday the 13th: -1 Luck; night-time monster behaviour),
which NLE does not fix, so a seed replays identically only under the same clock flags; they are
reported with every episode. With balrog-nle built with the clock patch (version 0.9.0+clockpin: NetHack
reads the time from NLE_FIXED_TIME when it is set), the harness pins the clock to PINNED_TIME, an
ordinary day (no full or new moon, not Friday the 13th, daytime), which makes seeds fully replayable
and plays as NetHack does on most days.

MiniHack score = BALROG's progression (100 if the task is solved, i.e. the goal state is reached;
else 0); success = solved. NetHack score = BALROG's progression: the highest of the values BALROG
assigns (from human games in the NLD-NAO data) to the deepest dungeon level and the highest
experience level reached, in percent; no success flag.

MazeWalk version 2 (see budget.py): with 100 steps, random walks reached the stairs
of the 9 x 9 mazes in 20% of the episodes. Each episode now has a budget of max(2 L*, 6) decisions,
at most 100, where L* is the fewest moves from the start to the stairs down: a breadth-first search
over the seed's maze, read from the task's premapped variant (maze_solution). The task text and the
step counter state the budget. The other MiniHack tasks and NetHack are unchanged.
"""

from __future__ import annotations

import datetime
import functools
import io
import os
import random
import shutil
import threading
import urllib.request
import zipfile
from collections import deque
from pathlib import Path
from typing import Any

from ...env import ROOT
from ...registry import register_env
from ...schema import DecisionRequest, Question
from ..base import Environment
from .budget import decision_budget

HISTORY = 15  # previous steps shown (message and action); BALROG keeps 16 observations

MINIHACK_ACTIONS = {  # BALROG's MiniHack prompt (balrog/environments/minihack/__init__.py), verbatim
    "north": "move north", "east": "move east", "south": "move south", "west": "move west",
    "northeast": "move northeast", "southeast": "move southeast", "southwest": "move southwest", "northwest": "move northwest",
    "far north": "move far north", "far east": "move far east", "far south": "move far south", "far west": "move far west",
    "far northeast": "move far northeast", "far southeast": "move far southeast", "far southwest": "move far southwest",
    "far northwest": "move far northwest", "up": "go up the stairs", "down": "go down the stairs",
    "wait": "rest one move while doing nothing", "more": "display more of the message", "apply": "apply (use) a tool",
    "close": "close an adjacent door", "open": "open an adjacent door", "eat": "eat something", "force": "force a lock",
    "kick": "kick an enemy or a locked door or chest", "loot": "loot a box on the floor",
    "pickup": "pick up things at the current location if there are any", "pray": "pray to the gods for help",
    "puton": "put on an accessory", "quaff": "quaff (drink) something", "search": "search for hidden doors and passages",
    "zap": "zap a wand",
}

NETHACK_ACTIONS = {  # BALROG's NetHack prompt (balrog/environments/nle/__init__.py); changes marked
    "north": "move north", "east": "move east", "south": "move south", "west": "move west",
    "northeast": "move northeast", "southeast": "move southeast", "southwest": "move southwest", "northwest": "move northwest",
    "far north": "move far north", "far east": "move far east", "far south": "move far south", "far west": "move far west",
    "far northeast": "move far northeast", "far southeast": "move far southeast", "far southwest": "move far southwest",
    "far northwest": "move far northwest", "up": "go up a staircase",
    "down": "go down a staircase (tip: you can only go down if you are standing on the stairs)",
    "wait": "rest one move while doing nothing",
    "more": "display more of the message (tip: ONLY ever use when current message ends with --More--)",
    "annotate": "leave a note about the level", "apply": "apply (use) a tool", "call": "name a monster or object, or add an annotation",
    "cast": "cast a spell", "close": "close an adjacent door", "open": "open an adjacent door", "dip": "dip an object into something",
    "drop": "drop an item", "droptype": "drop specific item types (specify in the next prompt)",
    "eat": "eat something",  # BALROG adds "(tip: replenish food when hungry)"
    "esc": "exit menu or message",
    "engrave": "engrave writing on the floor",  # BALROG adds "(tip: Elbereth)"
    "enhance": "advance or check weapons skills", "fire": "fire ammunition from quiver",
    "fight": "fight a monster (even if you only guess one is there)", "force": "force a lock", "inventory": "show your inventory",
    "invoke": "invoke an object's special power",  # BALROG: "invoke "
    "jump": "jump to a location", "kick": "kick an enemy or a locked door or chest", "look": "look at what is under you",
    "loot": "loot a box on the floor", "monster": "use a monster's special ability (when polymorphed)",
    "offer": "offer a sacrifice to the gods (tip: on an aligned altar)",
    # "overview" (display an overview of the dungeon) is not in NetHackChallenge's action set
    "pay": "pay your shopping bill", "pickup": "pick up things at the current location", "pray": "pray to the gods for help",
    "puton": "put on an accessory", "quaff": "quaff (drink) something", "quiver": "select ammunition for quiver",
    "read": "read a scroll or spellbook", "remove": "remove an accessory", "rub": "rub a lamp or a stone",
    "search": "search for hidden doors and passages", "swap": "swap wielded and secondary weapons",
    "takeoff": "take off one piece of armor", "takeoffall": "take off all armor",
    "teleport": "teleport to another level (if you have the ability)", "throw": "throw something (e.g. a dagger or dart)",
    "travel": ("travel to a specific location on the map (tip: in the next action, specify > or < for stairs, { for fountain, "
               "and _ for altar)"),
    "twoweapon": "toggle two-weapon combat", "untrap": "untrap something", "wear": "wear a piece of armor", "wield": "wield a weapon",
    "wipe": "wipe off your face", "zap": "zap a wand",
    "minus": "the key -", "space": "the space key", "apos": "the key '",  # BALROG: the bare characters
    **{str(d): f"the key {d}" for d in range(10)},
}
LETTERS = [chr(c) for c in range(ord("a"), ord("z") + 1)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]

# BALROG's progression values (balrog/environments/nle/achievements.json, commit b7afe79): for each dungeon level and experience
# level, a data-informed estimate of how far towards winning a game that reached it has come; the other keys of that file
# ('Home 1'-'Home 5', 'Astral Plane', 'You ascend t') are never matched by BALROG's code and are left out.
DLVL_PROGRESS = [0.0, 0.015392269075361172, 0.017539535798967013, 0.021221378364235505, 0.026482449457437766, 0.035428686118173014,
                 0.04846098359243788, 0.0695812141543123, 0.09770935198701912, 0.12558597019922987, 0.1612677822768481,
                 0.20612854836308672, 0.2566039887664476, 0.29247245425293333, 0.30881240110986263, 0.3249071349317381,
                 0.34007883702829594, 0.3526455577544006, 0.3654881095022441, 0.37895667923256743, 0.39293747679675045,
                 0.40832255575976223, 0.4258360748858015, 0.445181408701678, 0.46637631565303056, 0.5067605633802816,
                 0.5543572044866264, 0.6015648510382184, 0.6465049415992812, 0.6913110342176086, 0.7101094299371864,
                 0.7176456494828894, 0.7232124894378948, 0.7284405900470092, 0.7327761924174481, 0.7368528917489855,
                 0.7421458943978863, 0.7471321695760599, 0.753731343283582, 0.7610398408061306, 0.76740810314648, 0.7725338491295938,
                 0.7771626297577855, 0.7812567949554251, 0.7858986134802957, 0.7890256738822022, 0.7929718875502008,
                 0.7967408445768355, 0.8018467495718222, 0.8067964442444019]  # Dlvl 1..50
XL_PROGRESS = [0.0, 0.01847840456172601, 0.02081163581974355, 0.024160136550546978, 0.029108986017138745, 0.036887590648350246,
               0.0507583712345433, 0.07453595273884014, 0.11704996473565571, 0.17909953770432294, 0.25480449090205187,
               0.33264942385807844, 0.41442590519821954, 0.49403949403949404, 0.5778823703813513, 0.6261221999477034,
               0.656281764586298, 0.6810218550418937, 0.7015328516442704, 0.7174494081710576, 0.7286977843944303,
               0.7371004909327723, 0.7443446629532795, 0.7513449678519879, 0.7585537128343045, 0.7642151855635002,
               0.7703376822716808, 0.7759810263044415, 0.7783345263660224, 0.7804561949196475]  # experience level 1..30

BL_DEPTH, BL_XL, BL_SCORE, BL_TIME = 12, 18, 9, 20  # blstats indices (NLE)

_lock = threading.Lock()


# ---------------------------------------------------------------------------------------------------
# NetHack's clock (src/hacklib.c), for the report: NLE runs the game on the machine's local time.


PINNED_TIME = 1768478400  # 2026-01-15 12:00 UTC: NetHack moon phase 7 (neither full nor new), a Thursday, noon


def pinned_clock() -> int | None:
    """The time NetHack's clock is pinned to, or None when it runs on the machine's clock (stock balrog-nle)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        patched = "clockpin" in version("balrog-nle")
    except PackageNotFoundError:
        patched = False
    if not patched:
        return None
    os.environ.setdefault("NLE_FIXED_TIME", str(PINNED_TIME))  # read by the patched NetHack at every clock lookup
    value = os.environ.get("NLE_FIXED_TIME", "")
    return int(value) if value.strip() else None


def clock_flags(now: datetime.datetime | None = None) -> dict[str, Any]:
    lt = now or datetime.datetime.now()
    diy, goldn = lt.timetuple().tm_yday - 1, ((lt.year - 1900) % 19) + 1
    epact = (11 * goldn + 18) % 30
    if (epact == 25 and goldn > 11) or epact == 24:
        epact += 1
    phase = ((((diy + epact) * 6) + 11) % 177) // 22 & 7
    return {"moon": "full" if phase == 4 else "new" if phase == 0 else "other", "moon_phase": phase,
            "friday13": lt.weekday() == 4 and lt.day == 13, "night": lt.hour < 6 or lt.hour > 21, "midnight": lt.hour == 0}


# ---------------------------------------------------------------------------------------------------
# Boxoban levels: the DeepMind level sets MiniHack reads, pinned and fetched on first use.

BOXOBAN_COMMIT = "78011d8c86ad423bfb2e1005012393b4c3c24409"
BOXOBAN_URL = f"https://github.com/google-deepmind/boxoban-levels/archive/{BOXOBAN_COMMIT}.zip"
BOXOBAN_SHA256 = "e2ca9eb749442c62c1f3b598df43b03b1849b299914ec9f6f6d6bf8a8ed693ed"
BOXOBAN_DIR = ROOT / ".cache" / "minihack" / f"boxoban-levels-{BOXOBAN_COMMIT[:7]}"


def boxoban_levels_dir() -> Path:
    """The hard and medium level sets (all MiniHack uses), extracted once from the pinned archive."""
    import hashlib

    marker = BOXOBAN_DIR / "complete"
    with _lock:
        if marker.exists():
            return BOXOBAN_DIR
        request = urllib.request.Request(BOXOBAN_URL, headers={"User-Agent": "s1arena"})
        with urllib.request.urlopen(request, timeout=600) as response:
            data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != BOXOBAN_SHA256:
            raise RuntimeError(f"Boxoban archive SHA-256 {digest} does not match {BOXOBAN_SHA256}")
        tmp = BOXOBAN_DIR.with_name(BOXOBAN_DIR.name + ".part")
        shutil.rmtree(tmp, ignore_errors=True)
        prefix = f"boxoban-levels-{BOXOBAN_COMMIT}/"
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.infolist():
                rel = member.filename[len(prefix):]
                if member.is_dir() or not rel.startswith(("hard/", "medium/", "LICENSE")):
                    continue
                target = tmp / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
        shutil.rmtree(BOXOBAN_DIR, ignore_errors=True)
        tmp.rename(BOXOBAN_DIR)
        marker.write_text(f"hard and medium level sets from {BOXOBAN_URL}\n", encoding="utf-8")
        return BOXOBAN_DIR


def _levels_in(path: Path) -> list[str]:
    """One level file, parsed as MiniHack's load_boxoban_levels does (each level: the lines after its number)."""
    levels, block = [], []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if line != "\n":
            block.append(line)
        else:
            levels.append("".join(block[1:]))
            block = []
    return levels


_boxoban_index: dict[str, list[tuple[Path, int]]] = {}


def boxoban_level(folder: Path, seed: int, level_set: str) -> str:
    """The level for a seed: files in name order, levels in file order, one index drawn from a generator seeded by the seed."""
    with _lock:
        index = _boxoban_index.get(str(folder))
        if index is None:
            index = _boxoban_index[str(folder)] = [(path, len(_levels_in(path))) for path in sorted(folder.glob("*.txt"))]
    i = random.Random(f"minihack-boxoban-{level_set}-{seed}").randrange(sum(n for _, n in index))
    for path, n in index:
        if i < n:
            return _levels_in(path)[i]
        i -= n
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------------------------------
# The shared harness.


class _NLEGame(Environment):
    leaf = "game/roguelike"
    mode = "paused"
    url = "https://github.com/balrog-ai/BALROG"
    tier = "B"

    def _make(self, seed: int):
        raise NotImplementedError

    def _options(self) -> dict[str, str]:
        raise NotImplementedError

    def reset(self, seed: int) -> None:
        from nle import nle_language_obsv
        from nle.language_wrapper.wrappers.nle_language_wrapper import NLELanguageWrapper
        from nle.nethack import actions as A

        self.close()
        pinned = pinned_clock()
        when = datetime.datetime.fromtimestamp(pinned, tz=datetime.timezone.utc) if pinned is not None else None
        self.clock = {"clock": "pinned" if pinned is not None else "machine", **clock_flags(when)}
        self.env = self._make(seed)
        self.env.unwrapped.seed(seed, seed, False)  # NetHack's core and display generators; no reseeding during the game
        self.obs = self.env.reset()
        self.language = nle_language_obsv.NLELanguageObsv()
        actions = list(self.env.unwrapped.actions)
        self.keys: dict[str, int] = {}  # key string -> index in the env's action set (NLELanguageWrapper's two maps)
        for enum, strings in NLELanguageWrapper.all_nle_action_map.items():
            if enum in actions:
                for text in strings:
                    self.keys[text] = actions.index(enum)
        self.more = actions.index(A.MiscAction.MORE) if A.MiscAction.MORE in actions else None
        self.message = self._text_message()
        self.options = self._options()
        self.budget = self.max_decisions  # decisions this episode may take; MazeWalk sets one per seed
        self.decisions = self.env_steps = 0
        self.total = 0.0
        self.ended, self.status, self.last_reward = False, 0, 0.0
        self.history: list[str] = []
        self._track()

    def _text_message(self) -> str:
        return self.language.text_message(self.obs["tty_chars"]).decode("latin-1")

    def _track(self) -> None:
        pass

    def _step(self, index: int) -> None:
        """One key press, then --More-- continued as BALROG's AutoMore wrapper does (those presses are not decisions)."""
        self.obs, reward, done, info = self.env.step(index)
        self.env_steps += 1
        message = self._text_message()
        while "--More--" in message and not done and self.more is not None:
            message = message.replace("--More--", "\n")
            self.obs, extra, done, info = self.env.step(self.more)
            self.env_steps += 1
            message += self._text_message()
            reward += extra
        self.message = message
        self.total += float(reward)
        self.last_reward = float(reward)
        self.ended = bool(done)
        self.status = int(info.get("end_status", 0))
        self._track()

    def screen(self) -> dict[str, str]:
        """BALROG's hybrid observation: message, language observation, cursor, and the screen below its top line."""
        obs, lang = self.obs, self.language
        cursor = obs["tty_cursor"]
        rows = [bytes(row).decode("latin-1").rstrip() for row in obs["tty_chars"][1:]]
        return {"message": self.message,
                "language observation": lang.text_glyphs(obs["glyphs"], obs["blstats"]).decode("latin-1"),
                "cursor": lang.text_cursor(obs["glyphs"], obs["blstats"], cursor).decode("latin-1") + f"\n(x={cursor[1]}, y={cursor[0]})",
                "map": "\n".join(rows).rstrip("\n"),
                "inventory": lang.text_inventory(obs["inv_strs"], obs["inv_letters"]).decode("latin-1")}

    def task_text(self) -> str:
        raise NotImplementedError

    def observe(self) -> DecisionRequest:
        view = self.screen()
        state = {"task": self.task_text(), "steps_taken": self.progress_text(),
                 "previous_steps": "\n".join(self.history[-HISTORY:]) or "none: this is the first step", **view}
        question = Question(type="choice", instructions="Choose your next action.", criteria=self.options)
        return DecisionRequest(state=state, questions={"action": question})

    def progress_text(self) -> str:
        return f"{self.decisions} of at most {self.budget}"

    def act(self, choice: dict[str, str]) -> None:
        option = choice["action"]
        message = " ".join(self.message.split()) or "(no message)"
        self.history.append(f"step {self.decisions + 1}: message: {message} Action: {option}.")
        self.decisions += 1
        self._step(self.keys[option])

    def done(self) -> bool:
        return self.ended

    def close(self) -> None:
        env = getattr(self, "env", None)
        if env is not None:
            env.close()
            self.env = None


# ---------------------------------------------------------------------------------------------------
# MiniHack: BALROG's eight tasks.

MINIHACK_KWARGS = dict(  # BALROG's minihack_kwargs (balrog/config/config.yaml), with its observation keys
    observation_keys=["glyphs", "blstats", "tty_chars", "inv_letters", "inv_strs", "tty_cursor", "tty_colors"],
    character="@", max_episode_steps=100, penalty_step=-0.01, penalty_time=0.0, penalty_mode="constant",
    savedir=None, save_ttyrec_every=0, autopickup=False)
GOAL_STAIRS = "Your goal is to explore the level and reach the stairs down"  # BALROG's corridor goal
GOALS = {  # BALROG's goal sentence per task family (balrog/environments/minihack/__init__.py)
    "corridor": GOAL_STAIRS,
    "quest": ("Your goal is to explore the level, fight monsters, and navigate rooms and mazes to ultimately reach the stairs "
              "down."),
    "boxoban": ("You are playing Boxoban, a box-pushing game inspired by Sokoban. Your goal is to push the boulders onto the "
                "fountains on the map. You can push the boulders by walking into them, as long as there are no obstacles "
                "behind them."),
    "mazewalk": GOAL_STAIRS,  # BALROG: "Your goal is to get as far as possible in the game." (the task is reaching the stairs)
}


class MiniHackTask(_NLEGame):
    license = "Apache-2.0 (MiniHack; Boxoban levels); NetHack General Public License (NetHack, NLE / balrog-nle); MIT (BALROG)"
    criteria_source = ("BALROG's MiniHack instruction: goal sentence per task (MazeWalk given BALROG's reach-the-stairs sentence) "
                       "and the task's actions with BALROG's descriptions; the tip and 'PLAY!' removed; observation text from "
                       "BALROG's NetHack language wrapper (hybrid mode); one paragraph on the observation and episode end added")
    episodes = 10
    max_decisions = 100  # BALROG's MiniHack max_episode_steps
    score_name = "progression (%: 100 if solved)"
    env_id = ""
    family = ""

    def _make(self, seed: int):
        import gym
        import minihack  # noqa: F401  (registers the tasks)

        if self.family == "boxoban":
            import minihack.envs.boxohack as boxohack

            level_set = "hard" if "Hard" in self.env_id else "medium"
            root = boxoban_levels_dir()
            with _lock:
                boxohack.LEVELS_PATH = str(root)  # MiniHack reads its level sets from here
                env = gym.make(self.env_id, **MINIHACK_KWARGS)
            folder = root / level_set / ("" if level_set == "hard" else "train")
            level = boxoban_level(folder, seed, level_set)
            game = env.unwrapped

            def get_lvl_gen(self=game, level=level):  # MiniHack's BoxoHack.get_lvl_gen with the level fixed by the seed
                from minihack import LevelGenerator

                grid, info = self.get_env_map(level.split("\n"))
                flags = [*self._flags, "noteleport", "premapped"]
                gen = LevelGenerator(map=grid, lit=True, flags=flags, solidfill=" ")
                for b in info["boulders"]:
                    gen.add_boulder(b)
                for f in info["fountains"]:
                    gen.add_fountain(f)
                gen.set_start_pos(info["player"])
                return gen

            game.get_lvl_gen = get_lvl_gen
            return env
        return gym.make(self.env_id, **MINIHACK_KWARGS)

    def _options(self) -> dict[str, str]:
        from nle.language_wrapper.wrappers.nle_language_wrapper import NLELanguageWrapper

        names = [NLELanguageWrapper.all_nle_action_map[a][0] for a in self.env.unwrapped.actions]  # BALROG's order
        return {name: MINIHACK_ACTIONS[name] for name in names if name in MINIHACK_ACTIONS}

    def task_text(self) -> str:
        end = ("every boulder is on a fountain" if self.family == "boxoban" else "you stand on the stairs down (>)")
        return " ".join([
            "You are an agent playing MiniHack, a set of small tasks built on the game NetHack.", GOALS[self.family].rstrip(".") + ".",
            f"The task is solved as soon as {end}.",
            "Each observation gives the game's latest message, a description of what you see and in which direction (language "
            "observation), what is under the cursor, the screen as NetHack draws it (the map, you are @, then two status lines) "
            "and your inventory. You attack a monster by moving into it.",
            f"The episode ends when the task is solved, when you die, or after {self.budget} steps.",
        ])

    def outcome(self) -> dict[str, Any]:
        solved = self.status == 2  # StepStatus.TASK_SUCCESSFUL
        end = {2: "solved", 1: "died", -1: "step limit", 0: "running"}.get(self.status, str(self.status))
        return {"success": solved, "score": 100.0 if solved else 0.0, "balrog_progression": 100.0 if self.last_reward >= 1.0 else 0.0,
                "end": end, "return": round(self.total, 3), "decisions": self.decisions, "env_steps": self.env_steps,
                **self.clock}


MAZE_FLOOR = {12, 19, 20, 21, 22, 23, 24}  # NetHack's map symbols: doorway, room floor (lit, dark), corridor (dark, lit), stairs
STAIRS_DOWN = 24
COMPASS = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0),
           "northeast": (1, -1), "southeast": (1, 1), "southwest": (-1, 1), "northwest": (-1, -1)}


@functools.cache
def maze_solution(env_id: str, seed: int) -> tuple[str, ...] | None:
    """The fewest moves (option ids) from the start to the stairs down of the seed's maze, or None when there is no way.

    The maze is read from the task's premapped variant (MiniHack-MazeWalk-Mapped-*: the same level description with the
    whole map shown), made with the same settings, clock and seed, so the maze and the start are the episode's.
    Breadth-first search then runs over its floor cells. A diagonal move needs a floor cell beside it on at least one
    side: NetHack lets a hero squeeze diagonally between two cells of rock only when carrying at most 600 weight units,
    and these mazes have no such gap.
    """
    import gym
    import minihack  # noqa: F401  (registers the tasks)
    from nle import nethack

    pinned_clock()
    env = gym.make(env_id.replace("MazeWalk-", "MazeWalk-Mapped-"), **MINIHACK_KWARGS)
    try:
        env.unwrapped.seed(seed, seed, False)
        obs = env.reset()
        glyphs = obs["glyphs"].tolist()
        start = (int(obs["blstats"][nethack.NLE_BL_X]), int(obs["blstats"][nethack.NLE_BL_Y]))
    finally:
        env.close()
    floor, goal = {start}, None
    for y, row in enumerate(glyphs):
        for x, glyph in enumerate(row):
            symbol = nethack.glyph_to_cmap(glyph) if nethack.glyph_is_cmap(glyph) else None
            if symbol in MAZE_FLOOR:
                floor.add((x, y))
            if symbol == STAIRS_DOWN:
                goal = (x, y)
    previous = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell == goal:
            moves = []
            while previous[cell] is not None:
                cell, move = previous[cell]
                moves.append(move)
            return tuple(reversed(moves))
        x, y = cell
        for move, (dx, dy) in COMPASS.items():
            nxt = (x + dx, y + dy)
            if nxt in floor and nxt not in previous and (not (dx and dy) or (x + dx, y) in floor or (x, y + dy) in floor):
                previous[nxt] = (cell, move)
                queue.append(nxt)
    return None


class MazeWalkTask(MiniHackTask):
    version = 2  # per-seed decision budget

    def reset(self, seed: int) -> None:
        super().reset(seed)
        solution = maze_solution(self.env_id, seed)
        self.shortest = None if solution is None else len(solution)
        self.budget = decision_budget(self.shortest, self.max_decisions)

    def done(self) -> bool:
        return self.ended or self.decisions >= self.budget

    def outcome(self) -> dict[str, Any]:
        out = super().outcome()
        if out["end"] == "running" and self.decisions >= self.budget:
            out["end"] = "decision budget"
        return {**out, "shortest": self.shortest, "budget": self.budget}


MINIHACK_TASKS = [  # BALROG's minihack_tasks, in its order: (env id, family, name, notes)
    ("MiniHack-Boxoban-Hard-v0", "boxoban", "boxoban_hard",
     "Boxoban 'hard' level set (DeepMind, 3332 levels): push four boulders onto four fountains; 4 moves. Level chosen by the seed."),
    ("MiniHack-Boxoban-Medium-v0", "boxoban", "boxoban_medium",
     "Boxoban 'medium' training set (DeepMind, 450,000 levels): push four boulders onto four fountains; 4 moves. Level chosen "
     "by the seed."),
    ("MiniHack-MazeWalk-9x9-v0", "mazewalk", "mazewalk_9x9",
     "Random 9 x 9 maze, lit (corridors are seen as far as the line of sight goes); reach the stairs down; 8 moves."),
    ("MiniHack-MazeWalk-15x15-v0", "mazewalk", "mazewalk_15x15",
     "Random 15 x 15 maze, lit (corridors are seen as far as the line of sight goes); reach the stairs down; 8 moves."),
    ("MiniHack-Corridor-R3-v0", "corridor", "corridor_r3",
     "Three random rooms joined by corridors (NetHack's room generator); reach the stairs down; 8 moves, open, kick, search."),
    ("MiniHack-CorridorBattle-Dark-v0", "corridor", "corridorbattle_dark",
     "Unlit room and corridor full of monsters between you (a knight) and the stairs down; 8 moves."),
    ("MiniHack-Quest-Easy-v0", "quest", "quest_easy",
     "Cross a lava river and fight monsters to reach the stairs down; 32 actions. The wand of cold and frost horn that freeze the "
     "lava lie on the start square, but with BALROG's settings they cannot be picked up (autopickup off, and NLE closes the "
     "pick-up menu for two objects before the agent sees it), so only a character that starts with a way across can win."),
    ("MiniHack-Quest-Medium-v0", "quest", "quest_medium",
     "Quest-Easy with a corridor and six giant rats at the lava (a knight); 32 actions. As in Quest-Easy, the wand and horn on the "
     "start square cannot be picked up with BALROG's settings."),
]


def _minihack(env_id: str, family: str, short: str, what: str) -> type:
    budget = "; version 2 cuts the steps to twice the shortest path to the stairs (6 to 100)" if family == "mazewalk" else ""
    attrs = {"name": f"game/roguelike/minihack_{short}", "env_id": env_id, "family": family,
             "source": f"MiniHack {env_id} (Samvelyan et al., 2021; BALROG's fork 3ecb6da on balrog-nle 0.9.0), a BALROG task",
             "url": "https://github.com/balrog-ai/minihack",
             "notes": (f"{what} Seeds 0-9 (BALROG plays 5); BALROG's settings: random character unless the task fixes one, "
                       f"autopickup off, 100 steps{budget}. Score = BALROG's progression, 100 if solved else 0; success = solved.")}
    return register_env(type("MiniHack_" + short, (MazeWalkTask if family == "mazewalk" else MiniHackTask,), attrs))


for _spec in MINIHACK_TASKS:
    _minihack(*_spec)


# ---------------------------------------------------------------------------------------------------
# NetHack: BALROG's NetHackChallenge-v0.


def end_reason(tty_chars, status: int) -> str:
    """BALROG's Progress._get_end_reason: the cause of death from the final screen."""
    words = bytes(tty_chars.reshape(-1)).decode(errors="ignore").replace("You made the top ten list!", "").split()
    text = " ".join(words[8:-2]) if len(words) > 7 and words[7].startswith("Agent") else " ".join(words[7:-2])
    sentences = text.split(".")
    first = sentences[0].split()
    head = " ".join(first[:first.index("in")]) if "in" in first else " ".join(first)
    rest = ".".join(sentences[1:]).strip()
    name = {1: "DEATH", -1: "ABORTED"}.get(status, str(status))
    return f"{name}: {head}. {rest}".strip()


@register_env
class NetHack(_NLEGame):
    name = "game/roguelike/nethack"
    url = "https://github.com/balrog-ai/nle"
    source = ("NetHack 3.6.6 in the NetHack Learning Environment, NetHackChallenge-v0 (Küttler et al., 2020; balrog-nle 0.9.0), "
              "a BALROG task")
    license = "NetHack General Public License (NetHack, NLE / balrog-nle); MIT (BALROG and its progression table)"
    criteria_source = ("BALROG's NetHack instruction (balrog/environments/nle): its 80 described commands less 'overview', with "
                       "BALROG's descriptions (two strategy tips dropped), plus the 52 letter keys of BALROG's action space; the "
                       "instruction's tips reduced to how prompts are answered; observation text from BALROG's NetHack language "
                       "wrapper (hybrid mode)")
    notes = ("NetHackChallenge-v0 with BALROG's settings: random role, race, gender and alignment ('@'), no-progress timeout 150. "
             "Seeds 0-4 (BALROG plays 5). Cap: 2000 decisions (BALROG allows 100,000 steps; LLM calls make that unaffordable, and "
             "2000 matches Crafter's budget); --More-- is continued automatically without a decision. 131 options. Score = BALROG's "
             "progression (%, from the deepest dungeon level and highest experience level reached); no success flag. NetHack reads "
             "the machine's clock (moon phase, Friday the 13th, night); the clock flags are reported with each episode, and a "
             "clock-patched balrog-nle (0.9.0+clockpin) lets the harness pin them to an ordinary day.")
    episodes = 5
    max_decisions = 2000
    score_name = "progression (%, BALROG's NetHack progression)"

    def _make(self, seed: int):
        import gym
        import nle  # noqa: F401  (registers NetHackChallenge-v0)

        return gym.make("NetHackChallenge-v0", character="@", max_episode_steps=100_000, no_progress_timeout=150,
                        savedir=None, save_ttyrec_every=0)

    def _options(self) -> dict[str, str]:
        options = {name: text for name, text in NETHACK_ACTIONS.items() if name in self.keys}
        from nle.language_wrapper.wrappers.nle_language_wrapper import NLELanguageWrapper

        commands = {}  # key -> the command it is outside prompts
        for strings in NLELanguageWrapper.all_nle_action_map.values():
            for text in strings[1:]:
                commands.setdefault(text, strings[0])
        for letter in LETTERS:
            options[letter] = (f"the key {letter}: answers a prompt or picks item or menu entry {letter}; "
                               f"otherwise the command '{commands[letter]}'")
        return options

    def default_choice(self, request: DecisionRequest) -> dict[str, str]:
        return {"action": "esc"}  # BALROG's default action for NetHack

    def _track(self) -> None:
        """BALROG's Progress.update: the best progression value of any dungeon level and experience level reached."""
        blstats = self.obs["blstats"]
        depth, xl = int(blstats[BL_DEPTH]), int(blstats[BL_XL])
        self.max_depth, self.max_xl = max(self.max_depth, depth), max(self.max_xl, xl)
        if depth:  # the final screen after death has empty status fields; keep the last values
            self.turns, self.points = int(blstats[BL_TIME]), int(blstats[BL_SCORE])
        if 1 <= depth <= len(DLVL_PROGRESS):
            self.progression = max(self.progression, DLVL_PROGRESS[depth - 1])
        if 1 <= xl <= len(XL_PROGRESS):
            self.progression = max(self.progression, XL_PROGRESS[xl - 1])

    def reset(self, seed: int) -> None:
        self.max_depth, self.max_xl, self.progression, self.turns, self.points = 1, 1, 0.0, 0, 0
        super().reset(seed)
        self.role = " ".join(self.message.split())

    def task_text(self) -> str:
        return " ".join([
            "You are an agent playing NetHack, with a randomly chosen role, race, gender and alignment.",
            "Your goal is to get as far as possible in the game; progress is measured by the deepest dungeon level and the "
            "highest experience level you reach.",
            "Each observation gives the game's latest message, a description of what you see and in which direction (language "
            "observation), what is under the cursor, the screen as NetHack draws it (the map, you are @, then two status lines) "
            "and your inventory.",
            "Each decision is one key press. When the game asks you to choose an item, e.g. 'What do you want to eat? [dgh or "
            "?*]', answer with the item's letter; when it asks 'In what direction?', answer with a direction. You attack a monster "
            "by moving into it. Messages that continue (--More--) are shown in full without a decision.",
            f"The episode ends when you die, after {self.max_decisions} decisions, or after 150 key presses in a row during which "
            "the game's clock does not advance.",
        ])

    def outcome(self) -> dict[str, Any]:
        end = {1: "died", -1: "aborted (no progress)", 0: "decision cap"}.get(self.status, str(self.status))
        out = {"success": None, "score": round(100 * self.progression, 3), "max_depth": self.max_depth, "max_xl": self.max_xl,
               "end": end, "in_game_score": self.points, "turns": self.turns, "decisions": self.decisions,
               "env_steps": self.env_steps, "character": self.role[:160], **self.clock}
        if self.ended and self.status == 1:
            out["end_reason"] = end_reason(self.obs["tty_chars"], self.status)
        return out
