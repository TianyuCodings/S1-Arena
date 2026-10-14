"""Template: a closed-loop environment.

Copy this file to s1arena/envs/<domain>/<your_module>.py (domains: embodied, game, loop, realtime), rename the class
and replace the toy world with your simulator. Import heavy simulator packages inside the methods, so discovery works
without them installed. Then play it:

    uv run s1arena envs --env loop/example
    uv run s1arena play --model random --env loop/example/number_line --episodes 2

The environment is the fixed harness: observe() turns the state into a request, act() applies the chosen option of
each question, and outcome() reports the raw result. Every model sees the same requests and gets the same score.
"""

from __future__ import annotations

import random

from s1arena.envs.base import Environment
from s1arena.registry import register_env
from s1arena.schema import DecisionRequest, Question

MOVES = {"left": -1, "stay": 0, "right": 1}


@register_env
class NumberLine(Environment):
    name = "loop/example/number_line"  # <domain>/<leaf>/<environment>
    leaf = "loop/example"
    source = "Number line walk (toy example)"
    url = ""
    license = "Apache-2.0"
    tier = "C"
    criteria_source = "Written"
    mode = "paused"  # "realtime" needs control_dt and idle()
    episodes = 5  # the benchmark plays seeds 0 .. episodes-1
    max_decisions = 20
    score_name = "distance_left"

    def reset(self, seed: int) -> None:
        rng = random.Random(seed)
        self.position, self.goal, self.steps = rng.randint(0, 10), rng.randint(0, 10), 0

    def observe(self) -> DecisionRequest:
        question = Question(type="choice", instructions="Which move brings you closer to the goal?",
                            criteria={"left": "Move one step left (-1).", "stay": "Stay where you are.",
                                      "right": "Move one step right (+1)."})
        return DecisionRequest(state={"position": self.position, "goal": self.goal, "line": "positions 0 to 10"},
                               questions={"move": question})

    def act(self, choice: dict[str, str]) -> None:
        self.position = min(10, max(0, self.position + MOVES[choice["move"]]))
        self.steps += 1

    def done(self) -> bool:
        return self.position == self.goal

    def outcome(self) -> dict:
        return {"success": self.position == self.goal, "score": -abs(self.goal - self.position)}
