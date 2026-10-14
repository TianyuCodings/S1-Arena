"""ALFWorld household tasks (Shridhar et al., ICLR 2021): the next command on a shortest plan.

Each ALFWorld game ships its PDDL problem (every receptacle, object and fact of the ALFRED
scene) and the task sentence. The action rules follow the game's own PDDL domain: go to a
receptacle; open or close it; take an object from a receptacle you are at (open, if it can be
opened) with empty hands; put the held object in or on a receptacle that can contain it; clean,
heat or cool the held object at a sink basin, microwave or fridge; switch on a lamp you are at.
A breadth-first search over these rules gives the length of a shortest plan from any state;
items show the scene, the task and a prefix of a shortest plan, and ask for the next command.
Only states where exactly one command keeps the plan shortest are used.

Plan lengths come from a search restricted to what can matter (the task's objects and lamps,
the receptacles that hold them or can take them, the appliances), which gives the same lengths
as the full game. Every admissible command is then scored by the shortest plan left after it
(taking an object the task does not need is never on a shortest plan), so the gold is the only
admissible command that keeps the plan shortest. An independent, less pruned search confirms it.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
import zipfile
from dataclasses import dataclass

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch

ZIP = "https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip"
SPLITS = ("valid_seen", "valid_train", "train")  # valid_unseen is played in the closed-loop track
POOL = 600  # games examined, in a fixed hash order within the split order above
APPLIANCE = {"clean": ("Sink", "SinkBasin"), "heat": ("Microwave",), "cool": ("Fridge",)}
RULES = ("Going to a receptacle takes you to its spot, where every receptacle marked as sharing that spot can be used too. "
         "You can: go to a receptacle; open or close a receptacle you are at; take an object from a receptacle you are at, "
         "if your hands are empty and the receptacle is open (receptacles marked 'closed' must be opened first); put the "
         "object you hold in or on a receptacle you are at (open, if it can be opened); clean the object you hold with a sink "
         "basin, heat it with a microwave or cool it with a fridge that you are at; and use (switch on) a lamp at the receptacle "
         "you are at. You can hold only one object at a time. A task to put objects somewhere is done when they are there (both "
         "in the same receptacle for two objects), cleaned, heated or cooled first if the task says so; a task to look at or "
         "examine an object with a lamp is done when you hold the object at the lamp's spot with the lamp switched on.")


def _order(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@dataclass
class Scene:
    kind: str  # pick_and_place_simple, pick_two_obj_and_place, pick_clean/heat/cool_then_place_in_recep, look_at_obj_in_light
    target: str  # object type, e.g. 'Ladle'
    goal: str  # receptacle type, or lamp type for look_at
    task_text: str
    receptacles: list  # names in order
    rtype: dict
    rloc: dict
    openable: set
    objects: list
    otype: dict
    contains: dict  # object -> frozenset of receptacles holding it at the start
    props: dict  # property -> set of objects
    can_contain: set  # (receptacle type, object type)
    start: str
    names: dict  # PDDL name -> display name ('countertop 1', 'ladle 2')


def parse_game(path: str, data: dict) -> Scene | None:
    kind, target, _, goal = path.split("/")[2].split("-")[:4]
    problem = data["pddl_problem"]
    init = problem[problem.find("(:init"): problem.find("(:goal")]
    facts = [(f, tuple(a.split())) for f, a in re.findall(r"\((\w+)((?:\s+[^\s()]+)*)\s*\)", init)]
    rtype, otype, rloc, contains, props, can_contain, order = {}, {}, {}, {}, {}, set(), []
    start = None
    for f, a in facts:
        if f == "receptacleType":
            rtype[a[0]] = a[1][:-4]
            order.append(a[0])
        elif f == "objectType":
            otype[a[0]] = a[1][:-4]
            order.append(a[0])
        elif f == "receptacleAtLocation":
            rloc[a[0]] = a[1]
        elif f == "inReceptacle":
            contains.setdefault(a[0], set()).add(a[1])
        elif f == "canContain":
            can_contain.add((a[0][:-4], a[1][:-4]))
        elif f == "atLocation":
            start = a[1]
        elif f in ("openable", "pickupable", "cleanable", "heatable", "coolable", "toggleable"):
            props.setdefault(f, set()).add(a[0])
        elif f == "opened":
            return None  # every game starts with all receptacles closed; skip anything else
    task = re.search(r'"task":\s*\[\s*\{\s*"rhs":\s*"(Your task is to: [^"]+)"', data["grammar"])
    if not task or start is None:
        return None
    receptacles = [n for n in dict.fromkeys(order) if n in rtype and n in rloc]
    objects = [n for n in dict.fromkeys(order) if n in otype]
    names, count = {}, {}
    for n in receptacles + objects:
        t = (rtype.get(n) or otype[n]).lower()
        count[t] = count.get(t, 0) + 1
        names[n] = f"{t} {count[t]}"
    return Scene(kind, target, goal, task.group(1), receptacles, rtype, rloc, props.get("openable", set()), objects, otype,
                 {o: frozenset(r for r in rs if r in rloc) for o, rs in contains.items()}, props, can_contain, start, names)


# ---------- search ----------

@dataclass(frozen=True)
class State:
    loc: str
    held: str | None
    where: tuple  # per task object: frozenset of receptacles holding it (empty while held)
    opened: frozenset
    flags: tuple  # per task object: (clean, hot, cool)
    lamps_on: frozenset


TREAT_FLAG = {"clean": 0, "heat": 1, "cool": 2}
TREAT_PROP = {"clean": "cleanable", "heat": "heatable", "cool": "coolable"}
MAX_STATES = 5_000  # per search; games that need more are skipped


class TooLarge(Exception):
    pass


class Planner:
    def __init__(self, scene: Scene):
        s = self.scene = scene
        self.look = s.kind.startswith("look_at")
        self.rel = [o for o in s.objects if s.otype[o] == s.target]
        self.lamps = [o for o in s.objects if self.look and s.otype[o] == s.goal and o in s.props.get("toggleable", set())]
        self.goal_recs = [] if self.look else [r for r in s.receptacles if s.rtype[r] == s.goal]
        self.treat = s.kind.split("_")[1] if "_then_" in s.kind else None  # 'clean', 'heat', 'cool' or None
        self.need = 2 if s.kind.startswith("pick_two") else 1
        holders = {r for o in self.rel for r in s.contains.get(o, ())}
        lamp_recs = {r for o in self.lamps for r in s.contains.get(o, ())}
        appliances = {r for r in s.receptacles if self.treat and s.rtype[r] in APPLIANCE[self.treat]}
        self.useful = holders | set(self.goal_recs) | appliances | lamp_recs
        self.useful_locs = {s.rloc[r] for r in self.useful}
        self.memo: dict = {}

    def initial(self) -> State:
        s = self.scene
        return State(s.start, None, tuple(s.contains.get(o, frozenset()) for o in self.rel), frozenset(),
                     tuple((False, False, False) for _ in self.rel), frozenset())

    def ready(self, i: int, st: State) -> bool:
        return self.treat is None or st.flags[i][TREAT_FLAG[self.treat]]

    def is_goal(self, st: State) -> bool:
        s = self.scene
        if self.look:
            here = {r for r in s.receptacles if s.rloc[r] == st.loc}
            return st.held in self.rel and any(l in st.lamps_on and s.contains.get(l, frozenset()) & here for l in self.lamps)
        return any(sum(1 for i, rs in enumerate(st.where) if r in rs and self.ready(i, st)) >= self.need for r in self.goal_recs)

    def accessible(self, r: str, st: State) -> bool:
        return r not in self.scene.openable or r in st.opened

    def here(self, st: State) -> list[str]:
        return [r for r in self.scene.receptacles if self.scene.rloc[r] == st.loc]

    def _put(self, st: State, r: str) -> State:
        i = self.rel.index(st.held)
        return State(st.loc, None, st.where[:i] + (frozenset({r}),) + st.where[i + 1:], st.opened, st.flags, st.lamps_on)

    def moves(self, st: State) -> list[tuple[str, State]]:
        """Commands that can bring the task closer (enough to find shortest plans from any state)."""
        s, out, here = self.scene, [], self.here(st)
        for loc in sorted(self.useful_locs - {st.loc}):
            r = next(r for r in s.receptacles if s.rloc[r] == loc)
            out.append((f"go to {s.names[r]}", State(loc, st.held, st.where, st.opened, st.flags, st.lamps_on)))
        for r in here:
            if r in s.openable and r in self.useful and r not in st.opened:
                out.append((f"open {s.names[r]}", State(st.loc, st.held, st.where, st.opened | {r}, st.flags, st.lamps_on)))
        if st.held is None:
            for i, o in enumerate(self.rel):
                for r in sorted(st.where[i]):
                    if r in here and self.accessible(r, st) and o in s.props.get("pickupable", set()):
                        where = st.where[:i] + (frozenset(),) + st.where[i + 1:]
                        out.append((f"take {s.names[o]} from {s.names[r]}", State(st.loc, o, where, st.opened, st.flags, st.lamps_on)))
        else:
            i = self.rel.index(st.held)
            for r in here:
                if r in self.goal_recs and self.accessible(r, st) and (s.rtype[r], s.otype[st.held]) in s.can_contain and self.ready(i, st):
                    placed = [j for j, rs in enumerate(st.where) if r in rs and self.ready(j, st)]
                    others = [j for j, rs in enumerate(st.where) if j != i and set(rs) & set(self.goal_recs) - {r} and self.ready(j, st)]
                    if self.need == 1 or placed or not others:  # the second object goes where the first one is
                        out.append((f"move {s.names[st.held]} to {s.names[r]}", self._put(st, r)))
            if self.treat:
                for r in here:
                    if s.rtype[r] in APPLIANCE[self.treat] and st.held in s.props.get(TREAT_PROP[self.treat], set()) and not self.ready(i, st):
                        c, h, k = st.flags[i]
                        new = {"clean": (True, h, k), "heat": (c, True, False), "cool": (c, False, True)}[self.treat]
                        out.append((f"{self.treat} {s.names[st.held]} with {s.names[r]}",
                                    State(st.loc, st.held, st.where, st.opened, st.flags[:i] + (new,) + st.flags[i + 1:], st.lamps_on)))
        for lamp in self.lamps:
            if s.contains.get(lamp, frozenset()) & set(here) and lamp not in st.lamps_on:
                out.append((f"use {s.names[lamp]}", State(st.loc, st.held, st.where, st.opened, st.flags, st.lamps_on | {lamp})))
        return out

    def canonical(self, st: State) -> State:
        loc = st.loc if st.loc in self.useful_locs else "elsewhere"
        return State(loc, st.held, st.where, st.opened & self.useful, st.flags, st.lamps_on)

    def dist(self, st: State) -> int | None:
        """Shortest remaining plan length (breadth-first search, memoized)."""
        key = self.canonical(st)
        if key in self.memo:
            return self.memo[key]
        start = st if st.loc in self.useful_locs else State(self.scene.start, st.held, st.where, st.opened, st.flags, st.lamps_on)
        frontier, seen, depth, found = [start], {self.canonical(start)}, 0, None
        while frontier and found is None:
            if any(self.is_goal(x) for x in frontier):
                found = depth
                break
            nxt = []
            for x in frontier:
                for _, y in self.moves(x):
                    k = self.canonical(y)
                    if k not in seen:
                        seen.add(k)
                        nxt.append(y)
            if len(seen) > MAX_STATES:
                raise TooLarge
            frontier, depth = nxt, depth + 1
        self.memo[key] = found
        return found

    def successor(self, st: State, command: str) -> State | None:
        """The state after any admissible command; None for taking an object the task does not need (never optimal)."""
        s = self.scene
        names = {v: k for k, v in s.names.items()}
        m = re.fullmatch(r"go to (.+)", command)
        if m:
            return State(s.rloc[names[m.group(1)]], st.held, st.where, st.opened, st.flags, st.lamps_on)
        m = re.fullmatch(r"(open|close) (.+)", command)
        if m:
            r = names[m.group(2)]
            opened = st.opened | {r} if m.group(1) == "open" else st.opened - {r}
            return State(st.loc, st.held, st.where, opened, st.flags, st.lamps_on)
        m = re.fullmatch(r"take (.+) from (.+)", command)
        if m:
            o = names[m.group(1)]
            if o not in self.rel:
                return None
            i = self.rel.index(o)
            return State(st.loc, o, st.where[:i] + (frozenset(),) + st.where[i + 1:], st.opened, st.flags, st.lamps_on)
        m = re.fullmatch(r"move (.+) to (.+)", command)
        if m:
            return self._put(st, names[m.group(2)])
        m = re.fullmatch(r"(clean|heat|cool) (.+) with (.+)", command)
        if m:
            i = self.rel.index(st.held)
            c, h, k = st.flags[i]
            new = {"clean": (True, h, k), "heat": (c, True, False), "cool": (c, False, True)}[m.group(1)]
            return State(st.loc, st.held, st.where, st.opened, st.flags[:i] + (new,) + st.flags[i + 1:], st.lamps_on)
        m = re.fullmatch(r"use (.+)", command)
        if m:
            lamp = names[m.group(1)]
            return State(st.loc, st.held, st.where, st.opened, st.flags, st.lamps_on | ({lamp} if lamp in self.lamps else set()))
        raise ValueError(command)

    def admissible(self, st: State) -> list[str]:
        """Every command a player could give in this state."""
        s, here = self.scene, self.here(st)
        cmds = [f"go to {s.names[r]}" for r in s.receptacles if s.rloc[r] != st.loc]
        for r in here:
            if r in s.openable:
                cmds.append(f"close {s.names[r]}" if r in st.opened else f"open {s.names[r]}")
        if st.held is None:
            for o in s.objects:
                if o in s.props.get("pickupable", set()):
                    for r in sorted(self.position(o, st)):
                        if r in here and self.accessible(r, st):
                            cmds.append(f"take {s.names[o]} from {s.names[r]}")
        else:
            for r in here:
                if self.accessible(r, st) and (s.rtype[r], s.otype[st.held]) in s.can_contain:
                    cmds.append(f"move {s.names[st.held]} to {s.names[r]}")
            for kind, types in APPLIANCE.items():
                for r in here:
                    if s.rtype[r] in types and st.held in s.props.get(TREAT_PROP[kind], set()):
                        cmds.append(f"{kind} {s.names[st.held]} with {s.names[r]}")
        for o in s.objects:
            if o in s.props.get("toggleable", set()) and set(self.position(o, st)) & set(here):
                cmds.append(f"use {s.names[o]}")
        return sorted(set(cmds))

    def position(self, o: str, st: State) -> frozenset:
        return st.where[self.rel.index(o)] if o in self.rel else self.scene.contains.get(o, frozenset())

    def optimal_commands(self, st: State) -> list[str]:
        d = self.dist(st)
        best = []
        for c in self.admissible(st):
            t = self.successor(st, c)
            if t is not None and self.dist(t) == d - 1:
                best.append(c)
        return best


def layout_text(scene: Scene) -> str:
    lines = []
    for r in scene.receptacles:
        inside = [scene.names[o] for o in scene.objects if r in scene.contains.get(o, ())]
        closed = " (closed)" if r in scene.openable else ""
        mates = [scene.names[q] for q in scene.receptacles if q != r and scene.rloc[q] == scene.rloc[r]]
        spot = f" [same spot as {', '.join(mates)}]" if mates else ""
        items = ", ".join(("an " if n[0] in "aeiou" else "a ") + n for n in inside) if inside else "nothing"
        lines.append(f"{scene.names[r]}{closed}{spot}: {items}")
    return "\n".join(lines)


@functools.lru_cache(maxsize=1)
def alfworld_points() -> tuple[dict, ...]:
    out = []
    with zipfile.ZipFile(fetch(ZIP, "alfworld_json_2.1.3_tw-pddl.zip")) as z:
        games = [n for n in z.namelist() if n.endswith("game.tw-pddl") and n.split("/")[1] in SPLITS]
        ordered = sorted(games, key=lambda n: (SPLITS.index(n.split("/")[1]), _order(n)))
        for path in ordered[:POOL]:
            data = json.loads(z.read(path))
            scene = parse_game(path, data) if data.get("solvable", True) else None
            if scene is None:
                continue
            planner = Planner(scene)
            try:
                state = planner.initial()
                total = planner.dist(state)
                if not total:
                    continue
                prefix, candidates = [], []
                while planner.dist(state):
                    best = planner.optimal_commands(state)
                    if len(best) == 1:
                        candidates.append((list(prefix), best[0], planner.dist(state)))
                    command = best[0]
                    prefix.append(command)
                    state = planner.successor(state, command)
            except TooLarge:
                continue
            if not candidates:
                continue
            done, answer, remaining = candidates[int(_order(path)[:8], 16) % len(candidates)]
            out.append({"game": path, "split": path.split("/")[1], "kind": scene.kind, "done": done, "answer": answer,
                        "remaining": remaining, "plan_length": total})
    return tuple(out)


@functools.lru_cache(maxsize=None)
def scene_of(path: str) -> Scene:
    with zipfile.ZipFile(fetch(ZIP, "alfworld_json_2.1.3_tw-pddl.zip")) as z:
        return parse_game(path, json.loads(z.read(path)))


@register_task
class ALFWorldNextCommand(TaskAdapter):
    name = "embodied/next_skill/alfworld"
    leaf = "embodied/next_skill"
    source = "ALFWorld text games built on ALFRED scenes, PDDL problems of the 0.4.2 release (Shridhar et al., ICLR 2021); answers by exhaustive search"
    url = "https://github.com/alfworld/alfworld"
    license = "MIT"
    tier = "C"
    criteria_source = "ALFWorld task sentence verbatim; rules written after the game's PDDL domain; gold from breadth-first search"
    notes = ("Six ALFWorld task types (put, put two, clean, heat or cool then put, examine under a lamp), from the valid_seen, "
             "valid_train and train games (valid_unseen is left out: the closed-loop track plays those games). The state lists every receptacle with its contents (the simulator's full state, not what a player has "
             "seen yet), the task and the commands done so far, a prefix of a shortest plan. The options are all admissible "
             "commands; the gold is the only one that keeps the plan shortest. Names are numbered by order of appearance in "
             "the PDDL problem, so they can differ from the ALFWorld game's own numbering; an object standing on a receptacle "
             "that is part of another (a pan on a stove burner on a counter) is listed under both, as in the game's facts. "
             "Checked by an independent search.")

    def load(self):
        return list(alfworld_points())

    def label_of(self, record):
        return record["kind"]

    def to_item(self, record, rng):
        scene = scene_of(record["game"])
        planner = Planner(scene)
        state = planner.initial()
        for cmd in record["done"]:
            state = planner.successor(state, cmd)
        commands = planner.admissible(state)
        if record["answer"] not in commands:
            return None
        options = {f"command_{i + 1}": c for i, c in enumerate(commands)}
        gold = next(k for k, c in options.items() if c == record["answer"])
        done = "\n".join(f"{i}. {c}" for i, c in enumerate(record["done"], 1)) or "(nothing yet)"
        state_text = {"task": scene.task_text, "rules": RULES,
                      "starting_room": "You start in the middle of the room. The receptacles and what they hold:\n" + layout_text(scene),
                      "commands_done_so_far": done}
        question = Question(type="choice", criteria=options, instructions=(
            "Which command should come next to complete the task with as few commands as possible?"))
        return self.item(state_text, {"next_command": question}, {"next_command": Gold(label=gold)},
                         task_type=record["kind"], split=record["split"], remaining_commands=record["remaining"], game=record["game"])
