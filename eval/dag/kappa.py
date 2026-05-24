"""Cohen's κ for inter-judge agreement on binary verdicts.

κ = (p_o - p_e) / (1 - p_e), where p_o is observed agreement and p_e is the
agreement expected by chance. Returns 1.0 on perfect agreement, 0.0 on
chance-level agreement, negative on systematic disagreement. NaN when either
rater is constant (no variance) — caller should treat NaN as "undefined".
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def cohen_kappa(a: Sequence[bool | int], b: Sequence[bool | int]) -> float:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return math.nan

    a_bools = [bool(x) for x in a]
    b_bools = [bool(x) for x in b]

    agree = sum(1 for x, y in zip(a_bools, b_bools) if x == y)
    p_o = agree / n

    a_pos = sum(a_bools) / n
    b_pos = sum(b_bools) / n
    p_e = a_pos * b_pos + (1 - a_pos) * (1 - b_pos)

    if p_e == 1.0:
        return math.nan
    return (p_o - p_e) / (1 - p_e)


def mean_pairwise_kappa(raters: list[list[bool | int]]) -> float:
    """Mean κ across every unordered pair of raters (judges)."""
    n = len(raters)
    if n < 2:
        return math.nan
    pairs: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            k = cohen_kappa(raters[i], raters[j])
            if not math.isnan(k):
                pairs.append(k)
    if not pairs:
        return math.nan
    return sum(pairs) / len(pairs)
