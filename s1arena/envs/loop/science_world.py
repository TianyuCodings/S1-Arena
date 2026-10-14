"""Elementary science experiments in ScienceWorld (Wang et al., 2022).

ScienceWorld (allenai/ScienceWorld, Apache-2.0) simulates a house with rooms, objects, substances, devices, plants
and animals, and scores 30 task types from the elementary science curriculum (boil, melt or freeze a substance,
measure a melting point, build a circuit, test conductivity, find living and non-living things, grow a plant or a
fruit, mix chemicals or paints, compare life spans, order life stages, inclined planes and friction, Mendelian
genetics). The simulator runs in a JVM (Java 8 or later) behind py4j. No language model is involved.

Episodes: ScienceWorld's test variations (the paper's split: a quarter of each task's variations, with substances,
animals and plants unseen in training). Per task type, two test variations are fixed by a seeded shuffle of its test
variations (VARIATIONS; pick_variations() recomputes them). Seed i < 30 plays the first pick of task type i, in
ScienceWorld's task-id order (1-1 boil ... 10-2); seed 30 + i plays the second pick. The benchmark plays seeds 0-29,
one episode per task type. Simplifications: ScienceWorld's "easy" preset (teleport action, open doors, self-watering
flower pots, no electrical actions outside the electricity tasks), as in the paper's baseline commands.

State: ScienceWorld's own text: the task description, the score, the room and the inventory (ScienceWorld's free
"look around" and "inventory") and the actions so far, the latest replies verbatim (as in the text games). One
sentence says how the interface works; no strategy is written anywhere.

Actions: ScienceWorld's valid-action filter (get_valid_action_object_combinations_with_templates) lists every
action-object combination the simulator accepts now, typically 100 to 3,000, each with its template and object ids.
An action takes up to three decisions, each a choice question built from that list alone:
1. "action": the kinds of action with at least one valid combination, as ScienceWorld's action templates ("look
   around", "wait1", "focus on OBJ", "move OBJ to OBJ", ...), alphabetical;
2. "object": for a one-object kind, its valid actions in full ("focus on red wire"); for a two-object kind, the
   objects that can fill the first slot (named as in ScienceWorld's action strings);
3. "second object": the valid actions of that kind with the chosen first object, in full.
A question with a single option is not asked (the option is taken, as forced moves are in the chess games). The
chosen string is sent verbatim. When ScienceWorld's parser finds it ambiguous (several objects answer to one name,
e.g. identical seeds in a jar) it asks for a number; the valid list gives the objects of each number, and the
harness answers the number of the chosen objects (the exchange stays in the history). If no number stands for them
(rarely, ScienceWorld's parser reads a name its own filter generated as other objects), the model answers
ScienceWorld's question itself in the next request.
Why this design: a flat list would need thousands of options, or pages that favour whatever comes first. Asking the
kind of action first and the objects next, like the tool and then its arguments in τ², keeps every question small
(at most 78 options in the gold-path replays, usually 20-30) and ordered the same way for every model; conditioning
each question on the earlier choices means every completed path is a valid action. Nothing is pruned beyond
ScienceWorld's own filter: replaying ScienceWorld's gold action sequences through these questions reaches 100 on all
60 episodes (with the action limit lifted; gold actions that use other names are matched through ScienceWorld's
aliases, the objects its numbered answers stand for, or their effect on a replay; 26 of the 30 benchmark episodes'
gold paths fit in 100 actions).

Score: ScienceWorld's task score (0-100, partial credit for sub-goals); a failed task (e.g. focusing on the wrong
object) ends the episode with ScienceWorld's -100, recorded as 0 with failed=True. Success = 100.
Limit: 100 actions per episode (TALES' limit; the paper's baselines stop at 100 simulator moves, where "wait" costs
11), so "wait" (10 ticks) counts once. A decision without a usable answer enters nothing (in the second or third
decision it drops the chosen kind); max_decisions is 300, the most 100 actions can take.

Determinism: ScienceWorld 1.3.0 at commit e8216d6 (not on PyPI yet). The last release, 1.2.3, enumerates objects in an
order that changes between JVM launches, so the same seed and actions gave different gold paths and outcomes (seen
in 1 of 4 launches); 1.3.0 fixes the enumeration order. Listing the valid actions draws from the simulator's random
generator (e.g. where bees fly), so the harness lists them exactly once per step, as ScienceWorldEnv.step does, and
the simulation runs as through ScienceWorld's own API (look and inventory draw nothing). Install:
pip install "scienceworld @ https://github.com/allenai/ScienceWorld/archive/e8216d6044e8e39be9fcb185e3b2dfb602584b52.tar.gz"
"""

from __future__ import annotations

import collections
import os
import random
import shutil
import threading
from pathlib import Path
from typing import Any

from ...registry import register_env
from ...schema import DecisionRequest, MAX_OPTIONS, Question
from ..base import Environment
from .text_games import REPLY_CHARS, _clean

SCIENCEWORLD_COMMIT = "e8216d6044e8e39be9fcb185e3b2dfb602584b52"  # allenai/ScienceWorld main, version 1.3.0
SIMPLIFICATIONS = "easy"
ACTION_LIMIT = 100  # actions per episode, as in TALES (the paper's baselines: 100 simulator moves)
NO_OP = "none"
VARIATION_SEED = "s1arena/loop/science_experiments/{task}"

# ScienceWorld's 30 task types in task-id order, with two test variations each (pick_variations() recomputes them).
VARIATIONS: list[tuple[str, str, tuple[int, int]]] = [
    ("1-1", "boil", (26, 27)),
    ("1-2", "melt", (21, 24)),
    ("1-3", "freeze", (23, 27)),
    ("1-4", "change-the-state-of-matter-of", (22, 24)),
    ("2-1", "use-thermometer", (469, 429)),
    ("2-2", "measure-melting-point-known-substance", (371, 355)),
    ("2-3", "measure-melting-point-unknown-substance", (256, 239)),
    ("3-1", "power-component", (18, 15)),
    ("3-2", "power-component-renewable-vs-nonrenewable-energy", (19, 15)),
    ("3-3", "test-conductivity", (796, 801)),
    ("3-4", "test-conductivity-of-unknown-substances", (510, 507)),
    ("4-1", "find-living-thing", (262, 237)),
    ("4-2", "find-non-living-thing", (280, 268)),
    ("4-3", "find-plant", (225, 293)),
    ("4-4", "find-animal", (278, 261)),
    ("5-1", "grow-plant", (95, 121)),
    ("5-2", "grow-fruit", (107, 114)),
    ("6-1", "chemistry-mix", (29, 31)),
    ("6-2", "chemistry-mix-paint-secondary-color", (32, 35)),
    ("6-3", "chemistry-mix-paint-tertiary-color", (34, 32)),
    ("7-1", "lifespan-longest-lived", (98, 116)),
    ("7-2", "lifespan-shortest-lived", (102, 122)),
    ("7-3", "lifespan-longest-lived-then-shortest-lived", (93, 119)),
    ("8-1", "identify-life-stages-1", (12, 9)),
    ("8-2", "identify-life-stages-2", (9, 6)),
    ("9-1", "inclined-plane-determine-angle", (149, 138)),
    ("9-2", "inclined-plane-friction-named-surfaces", (1366, 1360)),
    ("9-3", "inclined-plane-friction-unnamed-surfaces", (138, 146)),
    ("10-1", "mendelian-genetics-known-plant", (107, 103)),
    ("10-2", "mendelian-genetics-unknown-plant", (387, 396)),
]
EPISODES = [(task_id, task, picks[0]) for task_id, task, picks in VARIATIONS] + [(task_id, task, picks[1]) for task_id, task, picks in VARIATIONS]

# Verb phrases before the object in one-object actions, to read an object's name off a valid action string.
_NAME_PREFIXES = ("look at ", "focus on ", "pick up ", "put down ", "look in ", "open ", "close ", "activate ", "deactivate ",
                  "eat ", "flush ", "mix ", "read ", "disconnect ")
_START_LOCK = threading.Lock()  # py4j's gateway launch is not guaranteed thread-safe


def _java() -> None:
    """ScienceWorld starts `java` from PATH; fall back to $JAVA_HOME/bin."""
    if shutil.which("java"):
        return
    home = os.environ.get("JAVA_HOME")
    if home and (Path(home) / "bin" / "java").exists():
        os.environ["PATH"] = str(Path(home) / "bin") + os.pathsep + os.environ.get("PATH", "")
        return
    raise RuntimeError("ScienceWorld needs a Java runtime (8 or later): put java on PATH or set JAVA_HOME")


def start_scienceworld():
    """A new simulator (one JVM) without ScienceWorld's own step limit: the harness counts actions itself."""
    _java()
    from scienceworld import ScienceWorldEnv

    with _START_LOCK:
        return ScienceWorldEnv("", envStepLimit=10**9)


def pick_variations(env=None) -> list[tuple[str, str, tuple[int, int]]]:
    """Recompute VARIATIONS: per task, a seeded shuffle of its test variations, first two."""
    import json

    import scienceworld

    own = env is None
    env = env or start_scienceworld()
    try:
        tasks = json.loads((Path(scienceworld.__file__).parent / "tasks.json").read_text())
        out = []
        for task in tasks:
            env.load(task["task_name"], 0, SIMPLIFICATIONS)
            order = list(env.get_variations_test())
            random.Random(VARIATION_SEED.format(task=task["task_name"])).shuffle(order)
            out.append((task["task_id"], task["task_name"], (order[0], order[1])))
        return out
    finally:
        if own:
            env.close()


class ScienceWorldTask(Environment):
    name = "loop/science_experiments/scienceworld"
    leaf = "loop/science_experiments"
    source = ("ScienceWorld (Wang et al., 2022), 30 task types, test variations, 'easy' simplifications; "
              "ScienceWorld 1.3.0 at commit e8216d6")
    url = "https://github.com/allenai/ScienceWorld"
    license = "Apache-2.0"
    tier = "A"
    criteria_source = ("Task description, room, inventory, replies and the action options are ScienceWorld's own text (action "
                       "templates and the valid action strings verbatim); one sentence on the interface added")
    notes = ("30 task types x 1 test variation (seeds 0-29; seeds 30-59 hold a second variation per task), fixed by a seeded "
             "shuffle of each task's test split; 'easy' simplifications. Each action takes up to three choices over "
             "ScienceWorld's valid actions: the kind of action (its template), then the object(s), each list conditioned on the "
             "earlier choices (single options are taken without asking). Score = ScienceWorld's task score (0-100, a failed "
             "task counts 0), success = 100. 100 actions per episode (as in TALES); ScienceWorld 1.3.0 (git commit, "
             "deterministic) and a Java runtime are needed.")
    mode = "paused"
    episodes = 30
    max_decisions = 3 * ACTION_LIMIT
    score_name = "task score (0-100)"

    def __init__(self) -> None:
        self.rules = ("ScienceWorld: a text simulation of a house with rooms, objects, substances, devices, plants and animals, in "
                      "which you carry out an elementary science task. Each action is entered in up to three choices: first the "
                      "kind of action, then the object or objects it applies to, among those the simulator accepts right now. "
                      "The simulator replies in text. The episode ends when the task is completed or failed, or after "
                      f"{ACTION_LIMIT} actions.")
        self.env = None

    # --- episode ---

    def reset(self, seed: int) -> None:
        if not 0 <= seed < len(EPISODES):
            raise ValueError(f"seed {seed} out of range: {len(EPISODES)} episodes (2 test variations x 30 tasks)")
        self.seed = seed
        self.task_id, self.task, self.variation = EPISODES[seed]
        self.close()
        self.env = start_scienceworld()
        self.env.load(self.task, self.variation, SIMPLIFICATIONS, generateGoldPath=False)
        if self.variation not in set(self.env.get_variations_test()):
            raise RuntimeError(f"{self.task} variation {self.variation} is not in ScienceWorld's test split")
        self.examples: dict[int, list[str]] = collections.defaultdict(list)
        for entry in self.env.get_possible_actions_with_IDs():
            self.examples[int(entry["template_id"])].append(entry["action_example"])
        self.turns: list[tuple[str, str]] = []
        self.pending: tuple | None = None  # the chosen kind of action waiting for its object(s)
        self.pending_first: int | None = None
        self.unresolved_ambiguities = 0
        self.env.server.reset()  # as ScienceWorldEnv.reset(), whose first move is "look around"
        self._sim_step("look around")
        self.task_description = _clean(self.env.taskdescription())
        self._refresh()

    def _sim_step(self, command: str) -> str:
        """ScienceWorldEnv.step(command), reading the valid combinations with their templates and object ids.

        ScienceWorldEnv.step lists the valid actions once per step, and listing them advances the simulator's random
        draws (e.g. where bees fly); so this makes exactly one such call per step, the templated one the harness needs,
        and the simulation runs as it does through ScienceWorld's own API (the free look and inventory calls draw
        nothing)."""
        server = self.env.server
        obs = server.step(command)
        self.score = int(round(100 * server.getScore()))
        self.completed = bool(server.getCompleted()) or self.score < 0  # ScienceWorldEnv.step ends a failed task too
        self.moves = int(self.env.get_num_moves())
        self.combos = [c for c in self.env.get_valid_action_object_combinations_with_templates() if c["action"].strip()]
        return obs

    def _refresh(self) -> None:
        """Group the current valid combinations by kind of action."""
        kinds: dict[tuple, dict[str, Any]] = {}
        for c in self.combos:
            text = c["action"].strip()
            if text.isdigit():  # ScienceWorld is waiting for the number of an ambiguous action
                key, arity, label = ("number", text), 0, self._numbered_line(text)
            else:
                tid, first = int(c["template_id"]), text.split()[0]
                key, arity, label = (tid, first), len(c["obj_ids"]), self._label(tid, first, c)
            if key not in kinds:
                kinds[key] = {"label": label, "arity": arity, "combos": []}
            kinds[key]["combos"].append(c)
        self.kinds = kinds
        self.names = self._names()

    def _numbered_line(self, number: str) -> str:
        """The line of ScienceWorld's 'which one?' question for this number."""
        reply = self.turns[-1][1] if self.turns else ""
        for line in reply.splitlines():
            if line.startswith(number + ":"):
                return line.replace("\t", " ").strip()
        return number

    def _label(self, tid: int, first: str, combo: dict) -> str:
        """The kind's ScienceWorld template ('go OBJ' and 'teleport OBJ' share a template id: told apart by the verb)."""
        examples = self.examples.get(tid, [])
        for example in examples:
            if example.split()[0] == first:
                return example
        return examples[0] if examples else combo["action"]

    def _names(self) -> dict[int, str]:
        """Each object's name as ScienceWorld writes it in its actions, read off one-object actions."""
        names: dict[int, str] = {}
        for c in self.combos:
            if len(c["obj_ids"]) == 1:
                text = c["action"]
                for prefix in _NAME_PREFIXES:
                    if text.startswith(prefix):
                        names.setdefault(int(c["obj_ids"][0]), text[len(prefix):])
                        break
        return names

    # --- requests ---

    def _history(self) -> dict[str, str]:
        blocks = [f"> {command}\n{reply}" for command, reply in self.turns]
        keep, size = 0, 0
        while keep < len(blocks) and (keep == 0 or size + len(blocks[-1 - keep]) <= REPLY_CHARS):
            size += len(blocks[-1 - keep]) + 2
            keep += 1
        out = {}
        if len(blocks) > keep:
            out["earlier actions"] = "; ".join(command for command, _ in self.turns[: len(blocks) - keep])
        if keep:
            out["last turns"] = "\n\n".join(blocks[len(blocks) - keep:])
        return out

    def _state(self) -> dict[str, Any]:
        return {"game": self.rules, "task": self.task_description, "score": f"{max(self.score, 0)} of 100 points",
                "room": _clean(self.env.look()), "inventory": _clean(self.env.inventory()),
                "actions entered": f"{len(self.turns)} of {ACTION_LIMIT}", **self._history()}

    def _levels(self) -> tuple[str, dict[str, Any]]:
        """The current question: its name and a mapping option id -> (text, payload)."""
        if self.pending is None:
            kinds = sorted(self.kinds.items(), key=lambda kv: (kv[1]["label"], kv[0]))
            return "action", {str(i + 1): (kind["label"], key) for i, (key, kind) in enumerate(kinds)}
        combos = self.kinds[self.pending]["combos"]
        if self.pending_first is None and self.kinds[self.pending]["arity"] >= 2:
            firsts = sorted({int(c["obj_ids"][0]) for c in combos}, key=lambda o: (self.names.get(o) or self._fallback(o, combos), o))
            return "object", {str(i + 1): (self.names.get(o) or self._fallback(o, combos), o) for i, o in enumerate(firsts)}
        chosen = [c for c in combos if self.pending_first is None or int(c["obj_ids"][0]) == self.pending_first]
        chosen = sorted(chosen, key=lambda c: (c["action"], c["obj_ids"]))
        return ("object" if self.pending_first is None else "second object"), {str(i + 1): (c["action"], c) for i, c in enumerate(chosen)}

    def _fallback(self, obj: int, combos: list[dict]) -> str:
        """An object without a one-object action: its first valid two-object action with the second object elided."""
        for c in combos:
            if int(c["obj_ids"][0]) == obj:
                second = self.names.get(int(c["obj_ids"][1]))
                if second and c["action"].endswith(second):
                    head = c["action"][: -len(second)].rstrip()
                    return head.split(" ", 1)[1].rsplit(" ", 1)[0] if " " in head else head
                return c["action"]
        return str(obj)

    def observe(self) -> DecisionRequest:
        level, options = self._levels()
        self.current = (level, options)
        state = self._state()
        if level == "action" and all(key[0] == "number" for key in self.kinds):
            instructions = "ScienceWorld asks which one you meant (see its last reply). Which do you choose?"
        elif level == "action":
            instructions = ("Which kind of action do you take next? Each option is one of ScienceWorld's action templates; OBJ is an "
                            "object you choose next, among those the action can apply to now. The chosen kind is carried out.")
        else:
            label = self.kinds[self.pending]["label"]
            state["action being entered"] = label if self.pending_first is None else \
                f"{label} (first object: {self.names.get(self.pending_first) or self.pending_first})"
            instructions = ("Which object fills the first OBJ of the action?" if level == "object" and self._two_objects()
                            else "Which of these actions do you enter?")
        texts = {option: text for option, (text, _) in options.items()}
        if len(texts) > MAX_OPTIONS:
            raise RuntimeError(f"{len(texts)} options in one question (more than {MAX_OPTIONS})")
        question = Question(type="choice", instructions=instructions, criteria=texts)
        return DecisionRequest(state=state, questions={level: question})

    def _two_objects(self) -> bool:
        return self.pending is not None and self.kinds[self.pending]["arity"] >= 2

    def default_choice(self, request: DecisionRequest) -> dict[str, str]:
        """No usable answer: enter nothing (a pending kind of action is dropped)."""
        return {name: NO_OP for name in request.questions}

    # --- actions ---

    def act(self, choice: dict[str, str]) -> None:
        level, options = self.current
        picked = options.get(choice.get(level, NO_OP))
        if picked is None:  # no usable answer
            self.pending, self.pending_first = None, None
            return
        payload = picked[1]
        if level == "action":
            self.pending, self.pending_first = payload, None
        elif level == "object" and self._two_objects():
            self.pending_first = payload
        else:
            self._enter(payload)
            return
        self._advance()

    def _advance(self) -> None:
        """Take single options without asking; enter the action once it is complete."""
        while self.pending is not None:
            kind = self.kinds[self.pending]
            if kind["arity"] == 0:
                self._enter(kind["combos"][0])
                return
            level, options = self._levels()
            if len(options) != 1:
                return
            payload = next(iter(options.values()))[1]
            if level == "object" and self._two_objects():
                self.pending_first = payload
            else:
                self._enter(payload)
                return

    def _enter(self, combo: dict) -> None:
        """Send the action string; answer ScienceWorld's 'which one?' question with the chosen objects."""
        self.pending, self.pending_first = None, None
        command = combo["action"]
        obs = self._sim_step(command)
        reply = obs
        for _ in range(3):
            if not obs.startswith("Ambiguous request"):
                break
            numbers = [c for c in self.combos if c["action"].strip().isdigit()]
            wanted = [int(o) for o in combo["obj_ids"]]
            match = [c for c in numbers if [int(o) for o in c["obj_ids"]] == wanted]
            if not match:  # ScienceWorld reads the name as other objects: the model answers its question (next request)
                self.unresolved_ambiguities += 1
                break
            answer = match[0]["action"].strip()
            obs = self._sim_step(answer)
            reply += f"\n> {answer}\n{obs}"
        self.turns.append((command, _clean(reply)))
        self._refresh()

    def done(self) -> bool:
        return self.completed or len(self.turns) >= ACTION_LIMIT

    def outcome(self) -> dict[str, Any]:
        return {"success": self.score >= 100, "score": float(max(self.score, 0)), "task_score": self.score,
                "failed": self.score < 0, "actions": len(self.turns), "moves": self.moves,
                "task": self.task, "task_id": self.task_id, "variation": self.variation,
                "unresolved_ambiguities": self.unresolved_ambiguities}

    def close(self) -> None:
        env = getattr(self, "env", None)
        if env is not None:
            try:
                env.close()
            except Exception:  # the JVM may already be gone
                pass
            self.env = None

    def card(self) -> dict[str, Any]:
        return {**super().card(), "scienceworld_commit": SCIENCEWORLD_COMMIT, "simplifications": SIMPLIFICATIONS,
                "action_limit": ACTION_LIMIT}


register_env(ScienceWorldTask)
