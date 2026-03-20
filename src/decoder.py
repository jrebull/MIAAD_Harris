"""SPV decoding and greedy decoder for the visa allocation problem.

Implements the three-layer representation:
    Layer 1: Continuous vector H in R^G (hawk position)
    Layer 2: Permutation pi = argsort(H) via SPV (Bean, 1994)
    Layer 3: Greedy decoder -> feasible allocation x in Z^G+

References:
    - HarrisFase02.tex, Section 6 (Equations 6-8)
    - Bean, J.C. (1994). Genetic algorithms and random keys for sequencing and optimization.
"""

import numpy as np
from numpy.typing import NDArray


def spv(hawk: NDArray[np.float64]) -> list[int]:
    """Convert a continuous vector to a permutation via Smallest Position Value.

    Args:
        hawk: Continuous vector H in R^G.

    Returns:
        Permutation as list of group indices sorted by ascending hawk value.
    """
    return list(np.argsort(hawk, kind="stable"))


def decode(
    permutation: list[int],
    groups: list[dict],
    total_visas: int,
    country_caps: dict[str, int],
    category_caps: dict[str, int],
) -> dict[int, int]:
    """Greedy decoder: every permutation produces a feasible allocation.

    Processes groups in permutation order, assigning the maximum feasible
    number of visas to each group without violating any constraint (R1-R5).

    Args:
        permutation: List of group indices in priority order.
        groups: List of group dicts with keys: index, country, category, n.
        total_visas: V, total visas available.
        country_caps: Dict {country: cap}.
        category_caps: Dict {category: effective_cap}.

    Returns:
        Dict mapping group index to visas assigned.
    """
    x: dict[int, int] = {g["index"]: 0 for g in groups}
    v_remaining = total_visas
    country_usage: dict[str, int] = {}
    category_usage: dict[str, int] = {}

    group_lookup = {g["index"]: g for g in groups}

    for g_idx in permutation:
        g = group_lookup[g_idx]
        country = g["country"]
        category = g["category"]

        cap_country = country_caps[country] - country_usage.get(country, 0)
        cap_category = category_caps[category] - category_usage.get(category, 0)
        x_g = min(g["n"], v_remaining, cap_country, cap_category)

        x[g_idx] = x_g
        v_remaining -= x_g
        country_usage[country] = country_usage.get(country, 0) + x_g
        category_usage[category] = category_usage.get(category, 0) + x_g

    return x
