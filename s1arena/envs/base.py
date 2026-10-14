"""Base class for closed-loop environments.

An environment is the fixed, minimal harness around a simulator or game: it turns the current
state into a DecisionRequest, applies the option the model chose for each question, and reports
the episode's raw outcome. Every model sees the same requests and is scored by the same outcome.

To add an environment, subclass Environment in a module under s1arena/envs/<domain>/, fill in the
class attributes, implement the methods below and decorate the class with @register_env. Import
heavy simulator packages inside the methods, so discovery works without them installed.

Two timing modes:
- "paused": the world waits for every decision (turn-based games, most agents).
- "realtime": the world keeps moving while the model decides. The runner charges each decision a
  fixed latency per model; during that time the environment keeps executing the previous action
  (see idle()), so slow models act on stale observations.
"""

from __future__ import annotations

from typing import Any

from ..schema import DecisionRequest, Question


class Environment:
    name: str = ""  # "embodied/mujoco_control/inverted_pendulum"
    leaf: str = ""  # "embodied/mujoco_control"
    source: str = ""  # simulator or benchmark as cited
    url: str = ""
    license: str = ""
    tier: str = "C"  # A: original task wording verbatim; B: adapted; C: newly written
    criteria_source: str = ""
    notes: str = ""
    mode: str = "paused"  # or "realtime"
    control_dt: float = 0.0  # simulated seconds per env step; required in realtime mode
    episodes: int = 10  # the benchmark plays seeds 0 .. episodes-1
    max_decisions: int = 200
    score_name: str = "return"  # what outcome()["score"] measures
    agents: list[str] | None = None  # several independently deciding agents (see observe_agent / act_all)
    version: int = 1  # harness version: raise it when the requests or the dynamics change, so cached episodes stop counting

    def reset(self, seed: int) -> None:
        raise NotImplementedError

    def observe(self) -> DecisionRequest:
        """The request the model answers now: the state as text plus the action questions."""
        raise NotImplementedError

    def act(self, choice: dict[str, str]) -> None:
        """Apply one decision (an option id per question) and advance the world."""
        raise NotImplementedError

    def idle(self) -> None:
        """Realtime only: advance one env step while the model is still deciding (previous action held)."""
        raise NotImplementedError

    def done(self) -> bool:
        raise NotImplementedError

    def outcome(self) -> dict[str, Any]:
        """Raw episode result: {"success": bool or None, "score": float, ...}."""
        raise NotImplementedError

    # Multi-agent environments (agents is a list): every active agent gets its own request each step, answered by an
    # independent call of the same model, and all decisions are applied together.
    def active_agents(self) -> list[str]:
        return list(self.agents or [])

    def observe_agent(self, agent: str) -> DecisionRequest:
        raise NotImplementedError

    def act_all(self, choices: dict[str, dict[str, str]]) -> None:
        raise NotImplementedError

    def default_choice(self, request: DecisionRequest) -> dict[str, str]:
        """Used when a model returns no usable answer: the middle level of a score, "false", or the first option."""
        out = {}
        for name, question in request.questions.items():
            ids = question.option_ids()
            out[name] = ids[len(ids) // 2] if question.type == "score" else "false" if question.type == "boolean" else ids[0]
        return out

    def close(self) -> None:
        pass

    def card(self) -> dict[str, Any]:
        keys = ("name", "leaf", "source", "url", "license", "tier", "criteria_source", "notes", "mode", "control_dt", "episodes", "max_decisions", "score_name",
                "version")
        return {k: getattr(self, k) for k in keys}


def levels(low: float, high: float, n: int = 5) -> list[float]:
    """n evenly spaced control values from low to high (n odd includes the midpoint)."""
    return [low + (high - low) * i / (n - 1) for i in range(n)]


def level_question(instructions: str, values: list[float], unit: str = "") -> Question:
    """An ordered action (e.g. one actuator's torque) as a score question with one level per value."""
    names = {0: "strongest negative", len(values) - 1: "strongest positive"}
    criteria = []
    for i, v in enumerate(values):
        label = names.get(i, "zero" if abs(v) < 1e-9 else ("negative" if v < 0 else "positive"))
        criteria.append(f"{v:+.2f}{unit} ({label})")
    return Question(type="score", instructions=instructions, criteria=criteria)
