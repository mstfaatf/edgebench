"""Confidence intervals and paired tests for accuracy results (standard library only).

- ``wilson_interval``: the Wilson score interval for a binomial proportion
  (95% by default). Used for top-1 and top-5 accuracy: it stays inside [0, 1]
  and behaves well near 0 and 1, unlike the normal approximation.
- ``mcnemar_exact``: McNemar's test for two classifiers evaluated on the same
  images, exact form: under the null hypothesis the discordant images (one
  right, the other wrong) split 50/50, so the two-sided p-value is a binomial
  tail. Exact for any number of discordant pairs (no chi-square
  approximation, no continuity correction).
"""

from __future__ import annotations

import math
from fractions import Fraction

Z_95 = 1.959963984540054  # standard normal quantile for a two-sided 95% interval
CI_METHOD = "wilson"
TEST_METHOD = "mcnemar_exact"


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError(f"need 0 <= successes <= total and total > 0, got {successes}/{total}")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    # The interval ends exactly at 0 (no successes) or 1 (all successes); computed,
    # those ends would carry a rounding error.
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == total else min(1.0, centre + half)
    return low, high


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts.

    ``only_a``: images A gets right and B gets wrong; ``only_b``: the reverse.
    Computed with exact integers, so it is reproducible to the last digit.
    """
    if only_a < 0 or only_b < 0:
        raise ValueError("discordant counts must be non-negative")
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = Fraction(sum(math.comb(n, i) for i in range(k + 1)), 2 ** n)
    return float(min(Fraction(1), 2 * tail))


def paired_counts(correct_a: list[bool], correct_b: list[bool]) -> dict[str, int]:
    """Counts of the 2x2 table of two runs on the same images."""
    if len(correct_a) != len(correct_b):
        raise ValueError("runs cover different numbers of images")
    both = sum(a and b for a, b in zip(correct_a, correct_b))
    only_a = sum(a and not b for a, b in zip(correct_a, correct_b))
    only_b = sum(b and not a for a, b in zip(correct_a, correct_b))
    return {"both_correct": both, "only_a_correct": only_a, "only_b_correct": only_b,
            "both_wrong": len(correct_a) - both - only_a - only_b}
