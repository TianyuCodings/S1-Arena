"""Text games with a finite set of legal commands per turn: TextWorld cooking games and ALFWorld household tasks.

Both run on Microsoft TextWorld. Every turn the model picks one of the commands the game accepts right
now (TextWorld's admissible commands, in the game's own sorted order) as a choice question. The state
is the game's own text: the objective, the current room ("look") and inventory for TextWorld, the
room overview for ALFWorld, and the recent turns as a transcript ("> command", then the game's reply,
verbatim). A classifier keeps no memory between decisions, so the harness shows every earlier command
and as many of the most recent replies as fit in REPLY_CHARS characters (at least the last one); nothing
else is added and no strategy is written anywhere. All state values are plain strings.

TextWorld cooking: the test games of GATA (Adhikari et al., 2020; the same set TALES uses), generated
with TextWorld's cooking challenge (First TextWorld Problems) and compiled to Z-machine files. They are
played through Jericho, so the text is exactly what the game prints. Generating new games needs
TextWorld's Inform 7 compiler, which ships only as x86-64 binaries (it fails on Apple Silicon without
Rosetta), so the published games are used instead; only the needed members of the release archive are
read (HTTP range requests) into .cache/text_games/.

ALFWorld: 24 of the 134 out-of-distribution evaluation games (valid_unseen) of ALFWorld (Shridhar et
al., 2021) in TextWorld's PDDL format, played with TextWorld's PDDL engine. That engine needs the optional
package fast-downward-textworld (not part of the "envs" extra); the environment is registered only when
it is installed. Object names are made readable exactly as ALFWorld's AlfredDemangler does ("apple 1").
"""

from __future__ import annotations

import collections
import importlib.util
import io
import random
import re
import shutil
import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from ...env import ROOT
from ...registry import register_env
from ...schema import DecisionRequest, MAX_OPTIONS, Question
from ..base import Environment

CACHE = ROOT / ".cache" / "text_games"
GATA_URL = "https://github.com/xingdi-eric-yuan/GATA-public/releases/download/data/rl.0.2.zip"
ALFWORLD_URL = "https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip"
REPLY_CHARS = 3000  # recent turns are shown with the game's reply up to this many characters; older turns as commands only
MORE, PREVIOUS = "more", "previous"

_LOCK = threading.RLock()  # TextWorld's loaders and PDDL translator keep module-level state: one game call at a time


# ---------------------------------------------------------------------------------------------------
# Downloads: read only the needed members of a large remote zip.


class _RemoteFile(io.RawIOBase):
    """A read-only, seekable view of a remote file through HTTP range requests."""

    def __init__(self, url: str) -> None:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "s1arena"})
        with urllib.request.urlopen(request, timeout=60) as response:
            self.url = response.geturl()  # after redirects (GitHub serves release assets from a signed URL)
            self.size = int(response.headers["Content-Length"])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def readinto(self, buffer) -> int:
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        request = urllib.request.Request(self.url, headers={"User-Agent": "s1arena", "Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 206:
                raise OSError("the server ignored the range request")
            data = response.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def _extract(url: str, prefix: str, dest: Path) -> None:
    """Extract the members of the remote zip under `prefix` into dest (once; a marker file records completion)."""
    marker = dest / (prefix.strip("/").replace("/", "__") + ".complete")
    if marker.exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    try:
        archive = zipfile.ZipFile(io.BufferedReader(_RemoteFile(url), buffer_size=1 << 20))
    except OSError:  # no range support: download the whole archive once
        from ...tasks.common import fetch

        archive = zipfile.ZipFile(fetch(url))
    tmp = dest / (marker.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    members = [m for m in archive.infolist() if m.filename.startswith(prefix) and not m.is_dir()]
    if not members:
        raise FileNotFoundError(f"no members under {prefix} in {url}")
    for member in members:
        archive.extract(member, tmp)
    for path in sorted(tmp.rglob("*")):
        if path.is_file():
            target = dest / path.relative_to(tmp)
            target.parent.mkdir(parents=True, exist_ok=True)
            path.replace(target)
    shutil.rmtree(tmp, ignore_errors=True)
    marker.write_text(f"{len(members)} files from {url}\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------------
# Shared harness: history, options, state.


def _clean(text: str) -> str:
    """The game's text with trailing spaces and runs of blank lines removed (layout only)."""
    lines = [line.rstrip() for line in (text or "").strip("\n").splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


class _TextGame(Environment):
    leaf = "loop/text_games"
    mode = "paused"
    rules = ""  # what the game is and how a turn works (no strategy)

    def _start(self) -> None:
        self.turns: list[tuple[str, str]] = []  # (command, reply) for every command entered so far
        self.page = 0

    def _commands(self) -> list[str]:
        raise NotImplementedError

    def _state(self) -> dict[str, Any]:
        raise NotImplementedError

    def _options(self) -> dict[str, str]:
        """Commands numbered 1..n in the game's order; if more than fit in one question, pages of 253 with more/previous."""
        commands = self._commands()
        if len(commands) <= MAX_OPTIONS:
            self.page = 0
            return {str(i + 1): c for i, c in enumerate(commands)}
        size = MAX_OPTIONS - 2
        pages = (len(commands) + size - 1) // size
        self.page = min(self.page, pages - 1)
        start = self.page * size
        options = {str(start + i + 1): c for i, c in enumerate(commands[start : start + size])}
        if self.page < pages - 1:
            options[MORE] = f"Show more commands (the game accepts {len(commands)}; these are {start + 1}-{start + len(options)})"
        if self.page > 0:
            options[PREVIOUS] = "Show the previous commands"
        return options

    def _history(self) -> dict[str, str]:
        """Older commands as one line; the most recent turns as a transcript ("> command", then the game's reply)."""
        blocks = [f"> {command}\n{reply}" for command, reply in self.turns]
        keep, size = 0, 0
        while keep < len(blocks) and (keep == 0 or size + len(blocks[-1 - keep]) <= REPLY_CHARS):
            size += len(blocks[-1 - keep]) + 2
            keep += 1
        out = {}
        if len(blocks) > keep:
            out["earlier commands"] = "; ".join(command for command, _ in self.turns[: len(blocks) - keep])
        if keep:
            out["last turns"] = "\n\n".join(blocks[len(blocks) - keep :])
        return out

    def observe(self) -> DecisionRequest:
        self.current = self._options()
        state = {"game": self.rules, **self._state(), "commands entered": str(len(self.turns)), **self._history()}
        question = Question(type="choice", instructions="Which command do you enter next?", criteria=self.current)
        return DecisionRequest(state=state, questions={"command": question})

    def act(self, choice: dict[str, str]) -> None:
        option = choice["command"]
        if option == MORE:
            self.page += 1
            return
        if option == PREVIOUS:
            self.page = max(0, self.page - 1)
            return
        command = self.current[option]
        self.page = 0
        with _LOCK:
            reply = self._step(command)
        self.turns.append((command, _clean(reply)))

    def _step(self, command: str) -> str:
        raise NotImplementedError

    def default_choice(self, request: DecisionRequest) -> dict[str, str]:
        return {"command": next(iter(request.questions["command"].criteria))}  # the first listed command


# ---------------------------------------------------------------------------------------------------
# TextWorld cooking games (GATA test sets), played from the compiled Z-machine files.


def gata_games(level: int) -> list[Path]:
    folder = CACHE / "gata_rl.0.2"
    with _LOCK:
        _extract(GATA_URL, f"test/difficulty_level_{level}/", folder)
    games = sorted((folder / "test" / f"difficulty_level_{level}").glob("*.z8"))
    if not games:
        raise FileNotFoundError(f"no games for difficulty level {level} in {folder}")
    return games


class TextWorldCooking(_TextGame):
    url = "https://github.com/xingdi-eric-yuan/GATA-public"
    license = "MIT (TextWorld; GATA game files)"
    tier = "A"
    criteria_source = "Objective, room descriptions, inventory, replies and the list of accepted commands are the game's own text; one sentence of rules added"
    episodes = 20  # the 20 test games of the level, in file-name order
    score_name = "points"
    level = 0
    settings = ""

    def __init__(self) -> None:
        self.rules = ("A text adventure game made with TextWorld. Each turn you enter one of the commands the game accepts at that "
                      "moment; the game replies in text. The game ends when you win or lose; the episode also stops after "
                      f"{self.max_decisions} commands.")

    def reset(self, seed: int) -> None:
        import textworld

        games = gata_games(self.level)
        if not 0 <= seed < len(games):
            raise ValueError(f"seed {seed} out of range: level {self.level} has {len(games)} test games")
        self.game_file = games[seed]
        infos = textworld.EnvInfos(feedback=True, description=True, inventory=True, admissible_commands=True, objective=True,
                                   score=True, max_score=True, won=True, lost=True)
        with _LOCK:
            self.env = textworld.start(str(self.game_file), infos)
            self.env.seed(1234)  # the Z-machine's random generator (the games also seed Inform's own at start)
            self.gs = self.env.reset()
        self._start()

    def _commands(self) -> list[str]:
        return list(self.gs["admissible_commands"])

    def _state(self) -> dict[str, Any]:
        return {"objective": _clean(self.gs["objective"]),
                "score": f"{self.gs['score']} of {self.gs['max_score']} points",
                "location": _clean(self.gs["description"]),
                "inventory": _clean(self.gs["inventory"])}

    def _step(self, command: str) -> str:
        self.gs, _, _ = self.env.step(command)
        return self.gs.feedback

    def done(self) -> bool:
        return bool(self.gs["won"] or self.gs["lost"])

    def outcome(self) -> dict[str, Any]:
        return {"success": bool(self.gs["won"]), "score": int(self.gs["score"]), "max_score": int(self.gs["max_score"]),
                "lost": bool(self.gs["lost"]), "commands": len(self.turns), "game": self.game_file.name}

    def close(self) -> None:
        env = getattr(self, "env", None)
        if env is not None:
            with _LOCK:
                env.close()
            self.env = None


def _cooking(level: int, settings: str, max_decisions: int, walkthrough: str) -> type:
    source = f"TextWorld cooking game (First TextWorld Problems generator), GATA test set, difficulty level {level}: {settings}"
    notes = (f"GATA/TALES difficulty level {level} ({settings}); 20 test games from GATA's rl.0.2 release, seed i = the i-th file. "
             f"The game's walkthrough takes {walkthrough} commands; the episode stops after {max_decisions}. Score = the game's points "
             "(one per recipe step done), success = the game is won (meal eaten); eating an ingredient or burning food loses the game.")
    attrs = {"name": f"loop/text_games/tw_cooking_level{level}", "source": source, "level": level, "settings": settings,
             "max_decisions": max_decisions, "notes": notes}
    return register_env(type(f"TextWorldCookingLevel{level}", (TextWorldCooking,), attrs))


# The four levels of the GATA paper (its levels 1-4 are the release's difficulty levels 3, 7, 5 and 9).
_cooking(3, "1 ingredient to find and cut; 1 room", 50, "7-8")
_cooking(7, "1 ingredient to find, cut and cook; 1 room", 50, "8-9")
_cooking(5, "1 ingredient to find; 9 rooms", 100, "4-19")
_cooking(9, "3 ingredients to find, cut and cook; 6 rooms", 100, "18-23")


# ---------------------------------------------------------------------------------------------------
# ALFWorld household tasks (TextWorld PDDL engine).

ALFWORLD_TASK_TYPES = ["pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
                       "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place"]
ALFWORLD_PER_TYPE = 4


def alfworld_games() -> list[Path]:
    """24 games of valid_unseen: per task type, 4 different tasks (goal and room) and one trial of each, chosen by a fixed
    shuffle; interleaved by type so that any prefix of the seeds mixes the types."""
    folder = CACHE / "alfworld_json_2.1.3_tw-pddl"
    with _LOCK:
        _extract(ALFWORLD_URL, "json_2.1.1/valid_unseen/", folder)
    tasks: dict[str, dict[str, list[Path]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in sorted((folder / "json_2.1.1" / "valid_unseen").glob("*/*/game.tw-pddl")):
        task = path.parent.parent.name  # e.g. pick_and_place_simple-Vase-None-Safe-219; its trials differ in object placement
        tasks[task.split("-", 1)[0]][task].append(path)
    picked = []
    for task_type in ALFWORLD_TASK_TYPES:
        rng = random.Random(f"alfworld/{task_type}")
        names = sorted(tasks[task_type])
        rng.shuffle(names)
        picked.append([rng.choice(tasks[task_type][name]) for name in names[:ALFWORLD_PER_TYPE]])
    return [games[i] for i in range(ALFWORLD_PER_TYPE) for games in picked]


def _demangler():
    """ALFWorld's AlfredDemangler (alfworld/agents/environment/alfred_tw_env.py, MIT): 'Apple_bar_...' -> 'apple 1'."""
    import textworld

    class AlfredDemangler(textworld.core.Wrapper):
        def load(self, *args, **kwargs):
            super().load(*args, **kwargs)
            infos = self._entity_infos
            ids = sorted(info.id for info in infos.values())
            counts: collections.Counter = collections.Counter()
            for id_ in ids:
                parts = id_.split("_bar_", 1)
                if len(parts) > 1:
                    counts[parts[0] + ("basin" if "basin" in id_ else "")] += 1
            numbers = {name: list(range(1, n + 1)) for name, n in counts.most_common()}
            names = {}
            for id_ in ids:
                text = id_.replace("_bar_", "|").replace("_minus_", "-").replace("_plus_", "+").replace("_dot_", ".").replace("_comma_", ",")
                parts = text.split("|", 1)
                if len(parts) == 1:
                    names[id_] = text + " "  # as in ALFWorld: unnumbered names keep a trailing space
                else:
                    base = parts[0] + ("basin" if "basin" in id_ else "")
                    names[id_] = f"{base} {numbers[base].pop()}"
            for info in infos.values():
                info.name = names[info.id]

    return AlfredDemangler


class ALFWorld(_TextGame):
    # household tasks in ALFRED scenes: listed under the embodied household leaf, sharing the text-game harness
    name = "embodied/household_loop/alfworld"
    leaf = "embodied/household_loop"
    source = "ALFWorld (Shridhar et al., 2021), TextWorld version, out-of-distribution evaluation games (valid_unseen)"
    url = "https://github.com/alfworld/alfworld"
    license = "MIT (ALFWorld, ALFRED)"
    tier = "A"
    criteria_source = "Task sentence, room overview, replies and the list of accepted commands are the game's own text; one sentence of rules added"
    episodes = 24
    max_decisions = 50
    score_name = "won (1) or not (0)"
    notes = ("24 of the 134 valid_unseen games: 4 different tasks per task type (pick & place, examine in light, clean, heat, cool, "
             "pick two), one trial each, chosen by a fixed shuffle. 50 commands per episode, the usual ALFWorld limit. Success = the "
             "task's goal condition holds; the score repeats it as 1/0. Needs TextWorld's PDDL engine, the optional package "
             "fast-downward-textworld (GPL-3.0, built from source); the alfworld package itself is not needed.")

    def __init__(self) -> None:
        self.rules = ("A household task in ALFWorld, a text version of a simulated home. Each turn you enter one of the commands the game "
                      "accepts at that moment; the game replies in text. The episode ends as soon as the task is done, or after "
                      f"{self.max_decisions} commands.")

    def reset(self, seed: int) -> None:
        import textworld

        games = alfworld_games()
        if not 0 <= seed < len(games):
            raise ValueError(f"seed {seed} out of range: {len(games)} ALFWorld games")
        self.game_file = games[seed]
        infos = textworld.EnvInfos(feedback=True, admissible_commands=True, won=True)
        with _LOCK:
            try:
                self.env = textworld.start(str(self.game_file), infos, wrappers=[_demangler()])
            except ImportError as error:
                raise RuntimeError("ALFWorld needs TextWorld's PDDL engine: pip install fast-downward-textworld (see the notes)") from error
            self.gs = self.env.reset()
        intro = _clean(self.gs.feedback)
        body = intro.split("\n\n")
        task = [p for p in body if p.startswith("Your task is to:")]
        self.task = task[-1] if task else ""
        self.room = "\n\n".join(p for p in body if p not in task and not p.startswith("-= Welcome"))
        self._start()

    def _commands(self) -> list[str]:
        return list(self.gs["admissible_commands"])

    def _state(self) -> dict[str, Any]:
        return {"task": self.task, "room": self.room}

    def _step(self, command: str) -> str:
        self.gs, _, _ = self.env.step(command)
        return self.gs.feedback

    def done(self) -> bool:
        return bool(self.gs["won"])

    def outcome(self) -> dict[str, Any]:
        won = bool(self.gs["won"])
        game = self.game_file.parent.parent.name + "/" + self.game_file.parent.name
        return {"success": won, "score": 1.0 if won else 0.0, "commands": len(self.turns), "task_type": game.split("-", 1)[0], "game": game}

    def close(self) -> None:
        env = getattr(self, "env", None)
        if env is not None:
            with _LOCK:
                env.close()
            self.env = None


# Registered only where TextWorld's PDDL engine is installed, so that listing or exporting the loop environments
# does not fail on a machine without it.
if importlib.util.find_spec("fast_downward") is not None:
    register_env(ALFWorld)
