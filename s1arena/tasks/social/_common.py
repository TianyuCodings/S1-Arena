"""Helpers shared by the social adapters: turning recorded answers into gold labels."""

from __future__ import annotations

import string
from collections.abc import Mapping, Sequence

from ...schema import Gold

LETTERS = string.ascii_uppercase


def shares(counts: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(counts.values()))
    return {key: value / total for key, value in counts.items()}


def plurality_gold(counts: Mapping[str, float], tie_label: str | None = None) -> Gold:
    """Soft label = answer shares; hard label = the most common answer.

    Ties go to tie_label when it is among the leaders (e.g. the dataset's own majority
    label), otherwise to the first leader in the order the options are listed.
    """
    dist = shares(counts)
    top = max(dist.values())
    leaders = [key for key, value in dist.items() if abs(value - top) < 1e-12]
    label = tie_label if tie_label in leaders else leaders[0]
    return Gold(label=label, dist=dist)


def median_gold(levels: Sequence[int], n_levels: int) -> Gold:
    """Ordinal scale from individual ratings: soft label = share of raters per level, hard label = the (lower) median."""
    return ordinal_gold([sum(1 for x in levels if int(x) == level) for level in range(n_levels)])


def lettered(options: Sequence[str]) -> dict[str, str]:
    """Survey answer options as A, B, C, ... in the order the survey presented them."""
    return {LETTERS[i]: str(option) for i, option in enumerate(options)}


def median_level(counts: Sequence[int]) -> int:
    """Lower median of an ordinal scale given the number of raters per level."""
    total = sum(counts)
    needed, running = (total - 1) // 2 + 1, 0
    for level, count in enumerate(counts):
        running += count
        if running >= needed:
            return level
    raise ValueError("no ratings")


def ordinal_gold(counts: Sequence[int]) -> Gold:
    """Ordinal scale from counts per level: soft label = shares, hard label = the (lower) median level."""
    total = float(sum(counts))
    return Gold(label=str(median_level(counts)), dist={str(level): count / total for level, count in enumerate(counts)})
