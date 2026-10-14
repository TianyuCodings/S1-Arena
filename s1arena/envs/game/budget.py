"""Per-seed decision budgets for the game environments that random play solved too often.

The benchmark counts an environment as too easy when a uniform-random policy succeeds in 20% or more of its episodes.
BALROG's step limits are generous for small maps, so random walks reached the goal in the BabyAI rooms (50% on
GoToLocal, 30% on GoToObj and PickupLoc), the 9 x 9 mazes of MiniHack MazeWalk (20%) and 10 of the 40 Baba Is AI
puzzles (30% or more). Version 2 of these suites (game/roguelike/babyai_*, game/roguelike/minihack_mazewalk_*,
game/rule_puzzles/baba_is_ai/*) gives each episode a budget of about twice the length L* of the seed's shortest
solution: an agent that knows where to go needs about L* decisions, a random walk on the order of L*^2. Each suite's
module says how it finds L*.
"""

from __future__ import annotations

FLOOR = 6  # when L* is 1 to 3, still room for a look around or a wrong move and its undo


def decision_budget(shortest: int | None, cap: int, floor: int = FLOOR) -> int:
    """max(2 L*, floor), never above the environment's own cap; the cap itself when the seed's L* is unknown.

    The floor serves agents that see only part of the map (BabyAI, MiniHack). Baba Is AI shows the whole level, so it
    passes floor=1: with a floor of 6, random play still won three puzzles whose win object can start next to the
    player (L* = 1) in about 28% of the episodes; with 2 L* alone, about 18%.
    """
    return cap if shortest is None else min(max(2 * shortest, floor), cap)
