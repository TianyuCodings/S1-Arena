"""Classic interactive fiction through Jericho (Hausknecht et al., 2020): Zork I, Detective and eight more.

Jericho (microsoft/jericho, GPL-2.0) runs Z-machine story files in an embedded Frotz and lists "valid actions": it
fills the game's own command templates with the objects around the player, runs every candidate from a saved state
and keeps those that change the game's state, one command per distinct change. Each turn the model picks one command,
as a choice question, in the harness of the TextWorld cooking games (text_games.py). The state is the game's own
text: its opening text, the score, the location ("look") and the inventory ("inventory"), both run on a copy of the
state as DRRN and XTX do, and the turns so far ("> command", then the reply, verbatim; older turns as commands only).
One sentence of rules is added; no strategy is written anywhere. No language model is involved.

Options: Jericho's valid actions (valid_actions below: FrotzEnv.get_valid_actions with the object tree, filtered in
Python as FrotzEnv._filter_candidate_actions does) with three corrections: ties are broken by the game's template
order and sorted object names, not by Python's set order, which changes between processes (the same state got
different options); commands whose effect on the score or the end of the game differs are kept apart (Jericho groups
by object changes only, so a non-scoring variant such as "take all from chest" could stand for "take emerald"); a
command that wins the game and stops the interpreter is kept (Jericho drops every command that halts it, e.g. the
last move of OMNIQuest). "look" and "wait" are always offered too: Jericho removes "look" from its templates and never
keeps "wait" (it changes no object), yet timed events need turns to pass and some games react to a look. Options are
sorted alphabetically (at most 63 in the walkthrough states of these games; the text games' paging applies beyond
255). What Jericho still cannot list: commands whose effect lies only in the game's variables (reading a note that
teaches a tune, asking about a topic, learning a spell) and commands on objects not named around the player.

Games: the 12 games of Yao et al. (2021) and Tuyls et al. (2022, XTX), the usual set for DRRN, INV-DY, RC-DQN and XTX
comparisons and a subset of the Jericho paper's 32, minus three whose walkthroughs need such commands within their
first steps, so that no model could score beyond a few points (balances and enchanter: learning spells; inhumane:
walking through a desert without rooms), plus adventureland (Jericho paper, 'possible' tier): zork1, zork3,
deephome, ludicorp, pentari, detective, library, dragon, omniquest, adventureland.
Seeds: the emulator's seed is the game's walkthrough seed + the episode seed. The six games whose solution depends on
the seed (the Jericho paper's 'stochastic' column) play 3 seeds, the others 1. An episode stops after 100 commands,
the episode length of the Jericho paper (100 valid steps) and of TALES. Score = the game's points (the outcome also
gives them divided by the game's maximum); success = the game is won.

Game files: the Jericho game suite (BYU-PCCL/z-machine-games, jericho-game-suite/, pinned commit, sha256 checked),
downloaded on first use into .cache/jericho/. That repository has no licence (its README asks copyright holders to
write if they object); Zork I and III are Infocom's commercial releases (the Zork source code is MIT-licensed since
November 2025, the story files are the original builds), the other eight are hobbyist freeware. The files are not
redistributed. Installing Jericho builds Frotz (a C compiler and make are needed); its valid actions need spaCy's
en_core_web_sm model.
"""

from __future__ import annotations

import hashlib
import importlib.util
import threading
import urllib.request
from pathlib import Path
from typing import Any

from ...env import ROOT
from ...registry import register_env
from .text_games import MORE, PREVIOUS, _TextGame, _clean

GAMES_COMMIT = "bfd7544aa1f50549fc327bd4eb3bc8144bfd80ea"  # BYU-PCCL/z-machine-games (the Jericho game suite)
GAMES_URL = "https://raw.githubusercontent.com/BYU-PCCL/z-machine-games/{commit}/jericho-game-suite/{file}"
CACHE = ROOT / ".cache" / "jericho"
# Always offered besides the valid actions: Jericho drops "look" from its command templates (a no-effect command) and
# never keeps "wait" (it changes no object), yet timed events need a way to let a turn pass and some games react to a look.
ALWAYS = ("look", "wait")
STOCHASTIC_SEEDS = 3

_LOCK = threading.RLock()  # spaCy and Jericho's module-level caches: one valid-action computation at a time


def game_file(file: str, sha256: str) -> Path:
    """The story file from the pinned Jericho game suite (downloaded once, checksum verified)."""
    path = CACHE / GAMES_COMMIT[:7] / file
    with _LOCK:
        if not path.exists():
            request = urllib.request.Request(GAMES_URL.format(commit=GAMES_COMMIT, file=file), headers={"User-Agent": "s1arena"})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != sha256:
                raise RuntimeError(f"{file}: checksum mismatch with the pinned Jericho game suite")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(path)
    return path


def _name_score(obj: tuple) -> float:
    """jericho.FrotzEnv._score_object_names: how good a name (word, part of speech, source) is for an object."""
    score = -0.01 * len(obj[0])
    score += {"NOUN": 1.0, "PROPN": 0.5}.get(obj[1], 0.0)
    if obj[2] == "OBJTREE":
        score += 0.1
    return score


def valid_actions(env) -> list[str]:
    """The commands that change the game's state now: Jericho's valid-action procedure, deterministic.

    As FrotzEnv.get_valid_actions (object tree on, run in Python as FrotzEnv._filter_candidate_actions does): name the
    interactive objects, fill the game's templates with them, run every candidate from the saved state, keep those that
    change the world-object tree, the score or end the game (skipping Jericho's '(Taken)' side effects), and keep one
    command per distinct change, the one whose verb is most used in ClubFloyd transcripts. Differences: ties are broken
    by the candidates' order (the game's template order, object names sorted) instead of Python's set order, which
    changes between processes; a change is the world diff together with the score and the end of the game (Jericho
    groups by the world diff alone, so 'take all from chest' could stand for 'take emerald' although only the latter
    scores); a command that wins the game and stops the interpreter is kept (Jericho drops every command that halts it).
    """
    from jericho import util

    if env.game_over() or env.victory() or env._emulator_halted():
        return []
    state = env.get_state()
    try:
        groups = env._identify_interactive_objects(use_object_tree=True)
        names = sorted({sorted(objs, key=lambda o: (-_name_score(o), o))[0][0] for objs in groups.values()})
        score = env.get_score()
        changes: dict[tuple, list[str]] = {}
        for command in env.act_gen.generate_actions(names):
            env.set_state(state)
            obs, _, done, info = env.step(command)
            won = env.victory()
            if env._emulator_halted():
                if won:
                    changes.setdefault(("won", info["score"]), []).append(command)
                env.reset()  # as Jericho does after a halt; the next set_state restores the game
                continue
            if info["score"] != score or done or env._world_changed():
                if "(Taken)" in obs:
                    continue
                changes.setdefault((env._get_world_diff(), info["score"], bool(done)), []).append(command)
        return sorted(max(commands, key=util.verb_usage_count) for commands in changes.values())
    finally:
        env.set_state(state)


def options(valid: list[str]) -> list[str]:
    """The commands offered: the valid actions plus "look" and "wait", sorted."""
    return sorted(set(valid) | set(ALWAYS))


class JerichoGame(_TextGame):
    url = "https://github.com/microsoft/jericho"
    license = ("GPL-2.0 (Jericho); game files from the Jericho game suite (no licence; Infocom titles commercial, others "
               "freeware), downloaded at run time, not redistributed")
    tier = "A"
    criteria_source = ("Opening text, location ('look'), inventory, replies and the commands (Jericho's valid actions, plus "
                       "'look' and 'wait') are the game's own text; one sentence of rules added")
    score_name = "points"
    max_decisions = 100
    game = ""
    file = ""
    sha256 = ""
    walkthrough_seed = 0

    def __init__(self) -> None:
        self.rules = ("A classic interactive fiction game (a text adventure), played through Jericho. Each turn you enter one of the "
                      "commands that change the game's state at that moment, as found by Jericho, which tries the game's own "
                      "command patterns on the objects around you, or \"look\" or \"wait\"; the game replies in text. The game "
                      f"ends when you win or lose; the episode also stops after {self.max_decisions} commands.")
        self.env = None

    def reset(self, seed: int) -> None:
        import jericho

        if seed < 0:
            raise ValueError(f"seed {seed} out of range (seeds are 0, 1, ...; the benchmark plays {self.episodes})")
        if importlib.util.find_spec("en_core_web_sm") is None:
            raise RuntimeError("Jericho's valid actions need spaCy's en_core_web_sm model (pip install en_core_web_sm from "
                               "https://github.com/explosion/spacy-models/releases)")
        path = game_file(self.file, self.sha256)
        self.close()
        self.emulator_seed = self.walkthrough_seed + seed
        with _LOCK:
            self.env = jericho.FrotzEnv(str(path), seed=self.emulator_seed)
            obs, _ = self.env.reset()
            self.max_score = self.env.get_max_score()
        self.opening = _clean(obs)
        self._start()
        self._refresh()

    def _refresh(self) -> None:
        """Location, inventory and valid commands of the current state (each probe runs on a copy of the state)."""
        env = self.env
        with _LOCK:
            state = env.get_state()
            look = env.step("look")[0]
            env.set_state(state)
            inventory = env.step("inventory")[0]
            env.set_state(state)
            self.valid = valid_actions(env)
        self.look, self.inventory = _clean(look), _clean(inventory)

    def _commands(self) -> list[str]:
        return options(self.valid)

    def _state(self) -> dict[str, Any]:
        return {"opening": self.opening, "score": f"{self.env.get_score()} of {self.max_score} points",
                "location": self.look, "inventory": self.inventory}

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
            reply = self.env.step(command)[0]
        self.turns.append((command, _clean(reply)))
        if not self.done():
            self._refresh()

    def done(self) -> bool:
        env = self.env
        return env is None or bool(env.game_over() or env.victory() or env._emulator_halted())

    def outcome(self) -> dict[str, Any]:
        score = int(self.env.get_score())
        return {"success": bool(self.env.victory()), "score": score, "max_score": self.max_score,
                "normalized_score": round(score / self.max_score, 4) if self.max_score else None,
                "lost": bool(self.env.game_over()), "commands": len(self.turns), "moves": int(self.env.get_moves()),
                "game": self.game, "emulator_seed": self.emulator_seed}

    def close(self) -> None:
        env = getattr(self, "env", None)
        if env is not None:
            with _LOCK:
                env.close()
            self.env = None

    def card(self) -> dict[str, Any]:
        return {**super().card(), "games_commit": GAMES_COMMIT, "game_file": self.file, "walkthrough_seed": self.walkthrough_seed}


def _jericho(game: str, file: str, sha256: str, walkthrough_seed: int, stochastic: bool, title: str, tier: str,
             max_score: int, walkthrough: int, reach: str) -> type:
    episodes = STOCHASTIC_SEEDS if stochastic else 1
    seeds = (f"{episodes} seeds: the emulator seed is the walkthrough seed ({walkthrough_seed}) + 0, 1, 2; the solution depends "
             "on the seed" if stochastic else f"1 episode at the walkthrough seed ({walkthrough_seed}); the game is deterministic")
    notes = (f"{title}. Jericho paper difficulty: {tier}. Max score {max_score}; Jericho's walkthrough takes {walkthrough} commands. "
             f"{seeds}. Each turn one of Jericho's valid actions (deterministic, score-aware) or look/wait; 100 commands per "
             "episode. "
             f"Score = the game's points (normalized score also reported); success = the game is won. Walkthrough through these "
             f"options: {reach}.")
    attrs = {"name": f"loop/text_games/jericho_{game}", "game": game, "file": file, "sha256": sha256,
             "walkthrough_seed": walkthrough_seed, "episodes": episodes, "notes": notes,
             "source": f"Jericho (Hausknecht et al., 2020) game suite: {title}"}
    return register_env(type(f"Jericho{game.capitalize()}", (JerichoGame,), attrs))


# name, story file, sha256, walkthrough seed, stochastic (Jericho paper, Table 2), title, difficulty tier, max score, walkthrough
_jericho("zork1", "zork1.z5", "0ae5ac229e79094ff368b6669356444af0f35e21d862a1baaa546989085c15fd", 12, True,
         "Zork I: The Great Underground Empire (Infocom, 1980)", "difficult", 350, 396,
         "every step's effect is an option (390 of 390), but options word some steps differently ('push all' for 'move "
         "rug'), which changes random draws, so the replayed fight with the troll goes differently and the replay dies "
         "with 57 points (67 after 100 commands)")
_jericho("zork3", "zork3.z5", "dce7e6f757fb8379dea9da9c13cdda5412ba03fa9b70d79fb6b8c7faf5970692", 0, True,
         "Zork III: The Dungeon Master (Infocom, 1982)", "difficult", 7, 273,
         "154 of the 167 steps that change the game are options (not e.g. 'get chest' or setting the dial), the replay "
         "dies with 2 points")
_jericho("deephome", "deephome.z5", "fdae5bd090fda6fc1b0390051531190968e04355060153ad5b2ab6b75a387b31", 0, True,
         "Deephome: A Telleen Adventure (Joshua Wise, 1999)", "difficult", 300, 327,
         "every step's effect is an option (292 of 292); 84 points after 100 commands as in the walkthrough, then the "
         "replay diverges on random draws and dies with 129")
_jericho("ludicorp", "ludicorp.z5", "b903da4b2f97117bea7814259a9df56d1e9b0f95d548c3a211dde2455e1fe2c8", 0, False,
         "The Ludicorp Mystery (Cal Henderson, 2003)", "difficult", 150, 364,
         "272 of the 276 steps that change the game are options, but pushing the button never is (Jericho does not find "
         "a button object around the player), so the replay stops at 23 points")
_jericho("pentari", "pentari.z5", "bbc8b30fa17089166698ed22740ab15be179babebec5252d07eb5260f1aa1316", 0, True,
         "Pentari (Howard A. Sherman, 1998)", "possible", 70, 49,
         "70 of 70 points in 49 commands, won")
_jericho("detective", "detective.z5", "f6b0b9b5f4ff8890ab481b11e63ea18350b43640c88a9b8c23d6a2b269f9a040", 0, False,
         "Detective (Matt Barringer, ported by Stuart Moore, 2000)", "possible", 360, 51,
         "360 of 360 points in 51 commands, won")
_jericho("library", "library.z5", "b3665eaf5e30520355a1f07657ffbd69e967ecb53942f86913d044e65fa30e11", 0, False,
         "All Quiet on the Library Front (Michael S. Phillips, 1995)", "possible", 30, 52,
         "18 of 30 points: 'ask technician about security gates' (a topic) is never an option")
_jericho("dragon", "dragon.z5", "9f7d54f7c18ce07ec1c6eab7d5c2176f3562d46ffd2357f6d6cfc48a6fa683a8", 0, True,
         "Dragon Adventure (William Stott)", "possible", 25, 101,
         "every step's effect is an option, but reading the booklet only sets a variable and is never offered, so "
         "playing the flute later scores nothing: 13 of 25 points")
_jericho("omniquest", "omniquest.z5", "1c905ffce496afe702c04453c4abd0ab01e272afd138b87c7e0ebaed4d8accd0", 0, False,
         "OMNIQuest (Chris Barden and Chris Ethridge, c. 1988; Inform port)", "possible", 50, 78,
         "50 of 50 points in 78 commands, won")
_jericho("adventureland", "adventureland.z5", "ca19b493a8190382a95b15c7645480ef467207ecaa0b446525f0328b89996777", 4, True,
         "Adventureland (Scott Adams, 1978; Inform port, 1996)", "possible", 100, 170,
         "100 of 100 points in 170 commands (42 after 100), won")
