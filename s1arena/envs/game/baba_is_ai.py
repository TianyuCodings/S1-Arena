"""Baba Is AI (Cloos et al., 2024) played through BALROG's text wrapper: the 40 puzzles BALROG uses.

Baba Is AI is a gridworld version of the puzzle game Baba Is You: text blocks lined up as
"[object] IS [property]" are the rules of the level, and pushing the blocks makes or breaks rules.
BALROG (Paglieri et al., ICLR 2025) describes a level with its language wrapper
(balrog/environments/babaisai/base.py): the active rules, then every object and text block with its
offset from the object you control ("rule `win` 2 steps to the right and 1 step up"); the level's
outer border is not described. describe_level() ports it, with two fixes: BALROG writes "step" for
every offset to the left or up ("3 step to the left"), here the plural is used as for right and down;
and a listed object in the controlled object's own cell (a YOU object of a listed kind) gets "at your
position" instead of BALROG's empty line.

The five actions and their descriptions are BALROG's. BALROG's instruction also carries eight
strategy tips; they are replaced, as in game/roguelike/babyai_*, by the rules of the game as the
environment implements them (baba/grid.py, baba/rule.py, baba/world_object.py). As in BALROG, when a
move leaves no object that is YOU, the level is reset to a new layout of the same puzzle (drawn from
the episode's generator, so the replay is deterministic) and the episode goes on; BALROG's own reset
notice prints the observation object instead of the level, here the notice precedes the new level.
Steps are counted over the whole episode, as BALROG does. The state lists the 15 previous steps
(observation and action), as BALROG's 16-observation history.

Levels come from baba-is-ai at commit 40ae1b2 (2026-09-11), which draws them from the env's own
seeded generator; earlier commits, which BALROG pins by branch, drew them from NumPy's global
generator (BALROG seeds it per episode), so a level is fixed by reset(seed=...) here.

Score = BALROG's progression: 100 if the puzzle is solved, else 0; success = solved.

Version 2 (see budget.py): with the game's 100 steps, random play solved 10 of the
40 puzzles in 30% or more of the episodes. Each episode now has a budget of max(2 L*, 6) steps, at
most 100, where L* is the fewest moves that solve the seed's level, found by an exact search over
game states (shortest_solution). For the benchmark seeds L* comes from a table made with that
search (SHORTEST), as the hardest levels take it minutes; other seeds are searched at reset. The
task text and the step counter state the budget.
"""

from __future__ import annotations

import functools
from collections import deque
from typing import Any

from ...registry import register_env
from ...schema import DecisionRequest, Question
from ..base import Environment
from .budget import decision_budget

ACTIONS = {  # option id -> (Baba Is AI action index, BALROG's description)
    "idle": (0, "wait for one step"),
    "up": (1, "take one step up"),
    "right": (2, "take one step to the right"),
    "down": (3, "take one step down"),
    "left": (4, "take one step to the left"),
}
HISTORY = 15  # previous steps shown; with the current observation, BALROG's max_text_history of 16
LISTED = ["fball", "fwall", "fdoor", "fkey", "rule_object", "rule_is", "rule_property"]  # what BALROG's wrapper lists

RULES = " ".join([
    "Baba Is You is a puzzle game where you can manipulate the rules of each level.",
    "A rule is three text blocks in a row, read from left to right or from top to bottom: an object, 'is', and a property "
    "or another object, e.g. 'baba is you'. A rule holds only while its blocks are lined up.",
    "You control every object that is YOU: each move takes it one cell up, down, left or right. You win the level when an "
    "object that is YOU moves onto an object that is WIN. Objects that are STOP cannot be entered; other objects can be "
    "walked over. Text blocks cannot be walked over but are pushed by moving into them (a line of blocks moves together, "
    "unless something at its end blocks it), which is how rules are made and broken.",
    "The level is enclosed by a border that is not listed in the observation. If no object is YOU any more, the level is "
    "reset to a new layout of the same puzzle and the episode continues.",
    "Each observation lists the active rules and every object and text block with its position relative to the object you "
    "control.",
])
MOVES = {"up": (0, -1), "right": (1, 0), "down": (0, 1), "left": (-1, 0)}  # option id -> cell offset
SEARCH_STATES = 50_000  # states a search at reset may expand (about a minute and a gigabyte); beyond, 100 steps stay


def steps_text(n: int, direction: str) -> str:
    return f"{n} {'steps' if n > 1 else 'step'} {direction}"


def controlled_kind(ruleset) -> str | None:
    """The object kind BALROG's wrapper treats as controlled: the one the last active YOU rule names."""
    from baba.world_object import name_mapping

    kind = None
    for rule in ruleset["_rule_"]:
        if "property" in rule and name_mapping[rule["property"]] == "you":
            kind = rule["object"]
    return kind


def describe_level(env) -> tuple[str, str] | None:
    """BALROG's get_ruleset and get_text_observation: (active rules, objects with offsets), or None when nothing is YOU."""
    from baba.world_object import name_mapping

    rules = []
    for rule in env.grid._ruleset["_rule_"]:
        if "object" in rule:  # 'object is object' rules have object1/object2 and are not listed, as in BALROG
            rules.append(f"{rule['object'].removeprefix('f')} is {name_mapping[rule['property']]}")
    you = controlled_kind(env.grid._ruleset)

    def find(kinds: list[str]) -> list[tuple[tuple[int, int], str]]:
        found = []
        for j in range(env.height):
            for i in range(env.width):
                cell = env.grid.get(i, j)
                if cell is None or cell.type not in kinds:
                    continue
                if cell.type == "rule_object":
                    label = f"rule `{cell.name}`"
                elif cell.type == "rule_is":
                    label = f"rule `{name_mapping[cell.name]}`"
                elif cell.type == "rule_property":
                    label = f"rule `{name_mapping[cell.property]}`"
                else:
                    label = cell.type
                found.append(((i, j), label))
        return found

    mine = find([you]) if you else []
    if not mine:
        return None
    (x0, y0), _ = mine[0]
    lines = []
    for (i, j), label in find(LISTED):
        dx, dy = i - x0, j - y0
        across = steps_text(dx, "to the right") if dx > 0 else steps_text(-dx, "to the left") if dx < 0 else ""
        vertical = steps_text(dy, "down") if dy > 0 else steps_text(-dy, "up") if dy < 0 else ""
        where = " and ".join(part for part in (across, vertical) if part) or "at your position"  # BALROG: an empty line
        lines.append(f"{label.removeprefix('f')} {where}")
    return "\n".join(rules), "\n".join(lines)


def _top(cell):
    return cell[-1] if len(cell) > 1 else None  # as BabaIsYouGrid.get: a cell's list of objects starts with None


def _read_rules(cells, links: list[int], width: int, height: int, default):
    """baba's extract_ruleset (rule.py) on a grid given as its cells, looking only at the cells of the 'is' blocks (links,
    in the grid's order); an 'is' block is always its cell's top object, as nothing can overlap a text block."""
    from collections import defaultdict

    from baba.rule import maybe_add_rule

    ruleset = defaultdict(dict)
    ruleset.update(default)
    for k in links:
        block, i, j = cells[k][-1], k % width, k // width
        for stride, first, last in ((1, i, width), (width, j, height)):  # across, then down
            if 0 < first < last - 1:
                before, after = _top(cells[k - stride]), _top(cells[k + stride])
                if not (first >= 2 and maybe_add_rule([_top(cells[k - 2 * stride]), before, block, after], ruleset)):  # colour
                    maybe_add_rule([before, block, after], ruleset)
    return ruleset


@functools.cache
def shortest_solution(task_id: str, seed: int, limit: int, states: int = SEARCH_STATES) -> tuple[str, ...] | None:
    """The fewest moves (option ids) that solve the seed's level, or None when none is found within `limit` moves and
    `states` expanded states.

    An exact search over game states under the game's own rules, by cost in moves (Dijkstra's algorithm). Only pushes
    change a level; walking moves nothing but the object that is YOU. So the search expands the states that follow a
    push: from each, the object walks the shortest way (breadth-first) onto a WIN object or to each cell from which it
    pushes a line of text blocks, and the push is played as BabaIsYouEnv.move plays it, with the rules then read as
    extract_ruleset reads them. States where walking could change more than the walker (several objects that are YOU,
    or an 'X is Y' rule with an X left to replace), pushes that leave an 'X is Y' rule, and levels with properties other
    than YOU, WIN and STOP go one move at a time through BabaIsYouEnv.step instead. A move that leaves nothing YOU ends
    that line of search, as the harness would reset the level to a new layout.
    """
    import baba
    from baba.world_object import FlexibleWorldObj, RuleProperty

    env = baba.make(f"env/{task_id}")
    env.reset(seed=seed)
    env.gen_obs = lambda: None  # observations are never read here, and encoding them is most of a step's time
    width, height = env.width, env.height
    properties = {o.property for cell in env.grid.grid for o in cell if isinstance(o, RuleProperty)}
    plain = properties <= {"is_agent", "is_goal", "is_stop"}
    start = tuple(map(tuple, env.grid.grid))
    best, rules_of, parent = {start: 0}, {start: env._ruleset.ruleset_dict}, {start: None}
    rule_sets = {}  # one ruleset object per set of rules, shared by the states that have it
    frontier = [[start]] + [[] for _ in range(limit)]  # states by the moves that reach them
    wins: list[tuple | None] = [None] * (limit + 1)  # moves -> (state, moves from it) of the first win found

    def won(cost, state, moves):
        if cost <= limit and wins[cost] is None:
            wins[cost] = (state, moves)

    def reach(cost, cells, rules, state, moves, you=None):
        """A state after the moves from state; `you` is the object that is YOU, on top of its cell, when known."""
        if cost > limit or cost >= best.get(cells, cost + 1):
            return
        kind = controlled_kind(rules)
        if kind is None or not ((you and you.type == kind) or any((t := _top(c)) and t.type == kind for c in cells)):
            return
        rules = rule_sets.setdefault(tuple(tuple(rule.items()) for rule in rules["_rule_"]), rules)
        best[cells], rules_of[cells], parent[cells] = cost, rules, (state, moves)
        frontier[cost].append(cells)

    def step(state, rules, cost, moves, walk=None):
        """The last of the moves through BabaIsYouEnv.step, after the object that is YOU walked from walk[0] to walk[1]."""
        env.grid.grid = [list(cell) for cell in state]
        if walk:
            env.grid.grid[walk[1]].append(env.grid.grid[walk[0]].pop())
        env._ruleset.set(rules)
        env.step_count = 0
        _, reward, done, _ = env.step(ACTIONS[moves[-1]][0])
        after = env._ruleset.ruleset_dict
        env._ruleset.set(rules)
        if done:
            if reward > 0:
                won(cost, state, moves)
        else:
            reach(cost, tuple(map(tuple, env.grid.grid)), after, state, moves)

    def walker(state, rules):
        """(cell, object, top objects once it walks away) of the one object that is YOU when walking changes nothing else."""
        yous = [(i, o) for i, cell in enumerate(state) for o in cell if isinstance(o, FlexibleWorldObj) and o.is_agent()]
        if not plain or len(yous) != 1 or _top(state[yous[0][0]]) is not yous[0][1]:
            return None
        (frm, you), tops = yous[0], [_top(cell) for cell in state]
        tops[frm] = _top(state[frm][:-1])  # what YOU leaves behind when it walks away
        replaced = {kind for kind, _ in rules.get("replace", ())}
        return None if you.type in replaced or any(t and t.type in replaced for t in tops) else (frm, you, tops)

    def push(state, rules, links, cost, moves, you, frm, at):
        """The last of the moves, into a text block, after YOU walked from frm to at: the line of blocks ahead moves from its
        far end while each block's next cell is empty or can be overlapped, then YOU follows if its next cell is free."""
        dx, dy = MOVES[moves[-1]]
        ahead = dy * width + dx
        cells = list(state)
        if at != frm:
            cells[frm], cells[at] = cells[frm][:-1], cells[at] + (you,)
        line, cell = [], at + ahead
        while (t := _top(cells[cell])) is not None and t.is_push():
            line.append(cell)
            cell += ahead
        free, pushed = t is None or t.can_overlap(), set()
        for cell in reversed(line):
            if free:
                cells[cell], cells[cell + ahead] = cells[cell][:-1], cells[cell + ahead] + (cells[cell][-1],)
                pushed.add(cell)
            t = _top(cells[cell])
            free = t is None or t.can_overlap()
        if free:
            if t is not None and t.is_goal():
                return won(cost, state, moves)
            cells[at], cells[at + ahead] = cells[at][:-1], cells[at + ahead] + (you,)
        elif you.is_goal():  # a blocked move of an object that is WIN wins: the game checks the mover's own cell
            return won(cost, state, moves)
        if not pushed:
            return
        links = sorted(cell + ahead if cell in pushed else cell for cell in links)
        after = _read_rules(cells, links, width, height, env.default_ruleset)
        if after.get("replace"):
            return step(state, rules, cost, moves, (frm, at))
        reach(cost, tuple(cells), after, state, moves, you)

    expanded = 0
    for cost in range(limit + 1):
        if wins[cost] is not None:
            state, moves = wins[cost]
            solution = list(moves)
            while parent[state] is not None:
                state, moves = parent[state]
                solution[:0] = moves
            return tuple(solution)
        for state in frontier[cost]:
            if best[state] < cost:
                continue
            expanded += 1
            if expanded > states:
                return None
            rules = rules_of[state]
            env._ruleset.set(rules)
            walking = walker(state, rules)
            if walking is None:
                for move in MOVES:
                    step(state, rules, cost + 1, (move,))
                continue
            frm, you, tops = walking
            you_wins = you.is_goal()
            links = [k for k, t in enumerate(tops) if t is not None and t.type == "rule_is"]
            paths, queue, pushes = {frm: ()}, deque([frm]), []
            while queue:
                at = queue.popleft()
                for move, (dx, dy) in MOVES.items():
                    nxt, moves = at + dy * width + dx, paths[at] + (move,)
                    t = tops[nxt]
                    if t is not None and t.is_push():
                        pushes.append((moves, at))
                    elif t is None or t.can_overlap():
                        if t is not None and t.is_goal():
                            won(cost + len(moves), state, moves)
                        elif nxt not in paths:
                            paths[nxt] = moves
                            queue.append(nxt)
                    elif you_wins:
                        won(cost + len(moves), state, moves)
            for moves, at in pushes:
                push(state, rules, links, cost + len(moves), moves, you, frm, at)
    return None


class BabaIsAIPuzzle(Environment):
    leaf = "game/rule_puzzles"
    url = "https://github.com/nacloos/baba-is-ai"
    license = "MIT (Baba Is AI); MIT (BALROG and its text wrapper)"
    tier = "B"
    criteria_source = ("Actions and descriptions from BALROG's Baba Is AI instruction (balrog/environments/babaisai), its strategy "
                       "tips replaced by the rules as implemented in baba-is-ai; observation text from BALROG's language wrapper "
                       "(plural fixed)")
    mode = "paused"
    episodes = 3  # BALROG's default: 3 episodes per Baba Is AI task
    max_decisions = 100  # the game's step limit (BabaIsYouEnv max_steps), which BALROG uses
    score_name = "progression (%: 100 if solved)"
    version = 2  # per-seed step budget
    task_id = ""  # Baba Is AI environment id without the 'env/' prefix

    def reset(self, seed: int) -> None:
        import baba

        self.close()
        self.env = baba.make(f"env/{self.task_id}")
        self.env.reset(seed=seed)
        known = SHORTEST.get(self.task_id, ())
        if seed < len(known):
            self.shortest = known[seed]
        else:  # a solution longer than half the cap would not lower the budget
            solution = shortest_solution(self.task_id, seed, self.max_decisions // 2)
            self.shortest = None if solution is None else len(solution)
        self.budget = decision_budget(self.shortest, self.max_decisions, floor=1)  # the level is fully shown
        self.plan = getattr(self.env, "target_plan", None)
        self.steps, self.resets, self.reward = 0, 0, 0.0
        self.won = self.lost = False
        self.level = describe_level(self.env)
        self.notice = ""
        self.history: list[str] = []

    def _observation(self) -> str:
        rules, objects = self.level
        return (self.notice + "\n\n" if self.notice else "") + f"Active rules:\n{rules}\n\nObjects on the map:\n{objects}"

    def observe(self) -> DecisionRequest:
        state = {"task": f"{RULES} The episode ends when you win or after {self.budget} steps.",
                 "steps_taken": f"{self.steps} of at most {self.budget}",
                 "previous_steps": "\n".join(self.history[-HISTORY:]) or "none: this is the first step",
                 "observation": self._observation()}
        question = Question(type="choice", instructions="Choose your next action.",
                            criteria={name: text for name, (_, text) in ACTIONS.items()})
        return DecisionRequest(state=state, questions={"action": question})

    def act(self, choice: dict[str, str]) -> None:
        action = ACTIONS[choice["action"]][0]
        rules, objects = self.level
        seen = f"rules: {'; '.join(rules.splitlines()) or 'none'}; objects: {'; '.join(objects.splitlines()) or 'none'}"
        self.history.append(f"step {self.steps + 1}: {self.notice + ' ' if self.notice else ''}{seen}. Action: {choice['action']}.")
        _, reward, done, _ = self.env.step(action)
        self.steps += 1
        self.notice = ""
        if done:
            self.reward = float(reward)
            self.won, self.lost = reward > 0, reward < 0
        level = describe_level(self.env)
        if level is None and not done:  # nothing is YOU: BALROG resets to a new layout of the puzzle
            self.env.reset()
            self.resets += 1
            level = describe_level(self.env)
            self.notice = "No object was YOU any more, so the level was reset to a new layout of the same puzzle."
        self.level = level or self.level

    def done(self) -> bool:
        return self.won or self.lost or self.steps >= self.budget

    def outcome(self) -> dict[str, Any]:
        return {"success": self.won, "score": 100.0 if self.won else 0.0, "reward": round(self.reward, 4), "steps": self.steps,
                "budget": self.budget, "shortest": self.shortest, "resets": self.resets, "lost": self.lost,
                "target_plan": self.plan, "puzzle": self.task_id}

    def close(self) -> None:
        env = getattr(self, "env", None)
        if env is not None:
            env.close()
            self.env = None


# BALROG's 40 Baba Is AI tasks (balrog/config/config.yaml, babaisai_tasks), in its order.
TASKS = [
    "make_win-distr_obj_rule", "goto_win-distr_obj_rule", "goto_win", "goto_win-distr_obj", "goto_win-distr_rule",
    "goto_win-distr_obj-irrelevant_rule", "make_win-distr_obj", "make_win-distr_rule", "make_win", "make_win-distr_obj-irrelevant_rule",
    "two_room-goto_win", "two_room-goto_win-distr_obj_rule", "two_room-goto_win-distr_rule", "two_room-goto_win-distr_obj",
    "two_room-goto_win-distr_obj-irrelevant_rule", "two_room-goto_win-distr_win_rule", "two_room-break_stop-goto_win-distr_obj_rule",
    "two_room-break_stop-goto_win-distr_obj", "two_room-break_stop-goto_win-distr_rule",
    "two_room-break_stop-goto_win-distr_obj-irrelevant_rule", "two_room-break_stop-goto_win",
    "two_room-maybe_break_stop-goto_win-distr_obj_rule", "two_room-maybe_break_stop-goto_win",
    "two_room-maybe_break_stop-goto_win-distr_obj", "two_room-maybe_break_stop-goto_win-distr_rule",
    "two_room-maybe_break_stop-goto_win-distr_obj-irrelevant_rule", "two_room-make_win-distr_obj_rule", "two_room-make_win-distr_rule",
    "two_room-make_win", "two_room-make_win-distr_obj-irrelevant_rule", "two_room-make_win-distr_obj", "two_room-make_win-distr_win_rule",
    "two_room-break_stop-make_win-distr_obj_rule", "two_room-break_stop-make_win-distr_rule", "two_room-break_stop-make_win",
    "two_room-break_stop-make_win-distr_obj-irrelevant_rule", "two_room-break_stop-make_win-distr_obj", "two_room-make_you",
    "two_room-make_you-make_win", "two_room-make_wall_win",
]
# L* of each puzzle's benchmark seeds 0-2, from shortest_solution. On the hardest levels the search takes minutes and
# gigabytes, too much for a reset; scripts/balrog_expert.py searches every level again and checks the table.
SHORTEST = {
    "make_win-distr_obj_rule": (15, 20, 15), "goto_win-distr_obj_rule": (1, 1, 4), "goto_win": (7, 3, 6),
    "goto_win-distr_obj": (7, 3, 6), "goto_win-distr_rule": (1, 1, 4), "goto_win-distr_obj-irrelevant_rule": (1, 1, 4),
    "make_win-distr_obj": (16, 17, 13), "make_win-distr_rule": (15, 20, 15), "make_win": (16, 17, 13),
    "make_win-distr_obj-irrelevant_rule": (15, 20, 15), "two_room-goto_win": (1, 5, 4), "two_room-goto_win-distr_obj_rule": (3, 2, 4),
    "two_room-goto_win-distr_rule": (2, 5, 3), "two_room-goto_win-distr_obj": (1, 5, 4),
    "two_room-goto_win-distr_obj-irrelevant_rule": (2, 5, 3), "two_room-goto_win-distr_win_rule": (3, 2, 4),
    "two_room-break_stop-goto_win-distr_obj_rule": (14, 11, 14), "two_room-break_stop-goto_win-distr_obj": (13, 9, 14),
    "two_room-break_stop-goto_win-distr_rule": (14, 9, 11), "two_room-break_stop-goto_win-distr_obj-irrelevant_rule": (14, 9, 11),
    "two_room-break_stop-goto_win": (13, 9, 14), "two_room-maybe_break_stop-goto_win-distr_obj_rule": (17, 11, 13),
    "two_room-maybe_break_stop-goto_win": (12, 11, 12), "two_room-maybe_break_stop-goto_win-distr_obj": (12, 11, 12),
    "two_room-maybe_break_stop-goto_win-distr_rule": (15, 4, 3), "two_room-maybe_break_stop-goto_win-distr_obj-irrelevant_rule": (15, 4, 3),
    "two_room-make_win-distr_obj_rule": (23, 17, 22), "two_room-make_win-distr_rule": (15, 21, 19), "two_room-make_win": (20, 23, 21),
    "two_room-make_win-distr_obj-irrelevant_rule": (15, 21, 19), "two_room-make_win-distr_obj": (20, 23, 21),
    "two_room-make_win-distr_win_rule": (19, 15, 18), "two_room-break_stop-make_win-distr_obj_rule": (22, 17, 19),
    "two_room-break_stop-make_win-distr_rule": (24, 20, 23), "two_room-break_stop-make_win": (18, 19, 21),
    "two_room-break_stop-make_win-distr_obj-irrelevant_rule": (24, 20, 23), "two_room-break_stop-make_win-distr_obj": (18, 19, 21),
    "two_room-make_you": (10, 15, 9), "two_room-make_you-make_win": (21, 23, 20), "two_room-make_wall_win": (15, 15, 16),
}
_PARTS = {  # task id components -> plain words for the environment card
    "two_room": "two rooms split by a wall", "goto_win": "reach the WIN object", "make_win": "form the broken WIN rule, then reach it",
    "break_stop": "break 'wall is stop' to cross the wall", "maybe_break_stop": "'wall is stop' may need breaking",
    "make_you": "make another object YOU", "make_wall_win": "make the wall WIN", "distr_obj": "a distractor object",
    "distr_rule": "a distractor rule block", "distr_obj_rule": "a distractor object and rule block",
    "irrelevant_rule": "an irrelevant rule block", "distr_win_rule": "a distractor WIN rule",
}


def _puzzle(task_id: str) -> type:
    words = []
    for part in task_id.split("-"):
        words.append(_PARTS.get(part, part.replace("_", " ")))
    attrs = {"name": f"game/rule_puzzles/baba_is_ai/{task_id}", "task_id": task_id,
             "source": f"Baba Is AI env/{task_id} (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task",
             "notes": ("; ".join(words) + ". Seeds 0-2 (BALROG plays 3 episodes per task); a budget of twice the fewest moves "
                       "that solve the level (6 to 100 steps; BALROG allows 100). Score = BALROG's progression, 100 if solved "
                       "within the budget else 0; success = solved. The environment's own reward (1 - 0.9 x steps / 100 on a "
                       "win) is reported as well.")}
    return register_env(type("BabaIsAI_" + task_id.replace("-", "_"), (BabaIsAIPuzzle,), attrs))


for _task in TASKS:
    _puzzle(_task)
