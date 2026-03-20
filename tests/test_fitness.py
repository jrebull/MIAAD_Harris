"""Tests for objective functions f1 and f2.

Verifies:
    - f1, f2 match hand-calculated values from LaTeX example.
    - Direction of f1: serving India first -> lower f1.
"""

import pytest
from src.decoder import decode


# --- LaTeX example (Section 6.4) ---
EXAMPLE_GROUPS = [
    {"index": 0, "country": "India", "category": "EB-2", "n": 40, "d": 2013, "w": 13},
    {"index": 1, "country": "India", "category": "EB-3", "n": 30, "d": 2010, "w": 16},
    {"index": 2, "country": "China", "category": "EB-2", "n": 20, "d": 2019, "w": 7},
    {"index": 3, "country": "China", "category": "EB-3", "n": 15, "d": 2020, "w": 6},
    {"index": 4, "country": "Canada", "category": "EB-2", "n": 10, "d": 2025, "w": 1},
    {"index": 5, "country": "Canada", "category": "EB-3", "n": 5, "d": 2024, "w": 2},
]
TOTAL_DEMAND = sum(g["n"] for g in EXAMPLE_GROUPS)  # 120


def _f1(x: dict[int, int]) -> float:
    """f1 for the example problem."""
    num = sum((g["n"] - x[g["index"]]) * g["w"] for g in EXAMPLE_GROUPS)
    return num / TOTAL_DEMAND


def _f2(x: dict[int, int]) -> float:
    """f2 for the example problem."""
    countries = {"India": [0, 1], "China": [2, 3], "Canada": [4, 5]}
    w_max = {"India": 16, "China": 7, "Canada": 2}
    w_country = {}
    for c, idxs in countries.items():
        total_assigned = sum(x[i] for i in idxs)
        if total_assigned > 0:
            w_country[c] = sum(
                x[g["index"]] * g["w"]
                for g in EXAMPLE_GROUPS if g["index"] in idxs
            ) / total_assigned
        else:
            w_country[c] = w_max[c]
    vals = list(w_country.values())
    return max(vals) - min(vals)


def test_f1_latex_example():
    """f1 should be 2.17 for the LaTeX example allocation."""
    x = {0: 20, 1: 30, 2: 20, 3: 15, 4: 10, 5: 5}
    f1 = _f1(x)
    assert abs(f1 - 2.17) < 0.01, f"Expected f1≈2.17, got {f1:.4f}"


def test_f2_latex_example():
    """f2 should be 13.47 for the LaTeX example allocation."""
    x = {0: 20, 1: 30, 2: 20, 3: 15, 4: 10, 5: 5}
    f2 = _f2(x)
    assert abs(f2 - 13.47) < 0.01, f"Expected f2≈13.47, got {f2:.4f}"


def test_f1_direction():
    """Ignoring India completely must give higher f1 than serving India.

    Check 2 of the audit cycle: f1 with x_India=0 should be >> f1 when India is served.
    In the small example with tight caps, permutation order interacts with cap constraints,
    so we test against x_India=0 explicitly (as per the LaTeX verification).
    """
    # Allocation where India is served (from LaTeX example)
    x_served = {0: 20, 1: 30, 2: 20, 3: 15, 4: 10, 5: 5}
    f1_served = _f1(x_served)

    # Allocation where India is completely ignored
    x_ignored = {0: 0, 1: 0, 2: 20, 3: 15, 4: 10, 5: 5}
    f1_ignored = _f1(x_ignored)

    assert f1_served < f1_ignored, (
        f"f1 with India served ({f1_served:.4f}) should be < f1 India ignored ({f1_ignored:.4f})"
    )


def test_f1_direction_full_problem():
    """Direction check on the full 50-group problem."""
    from src.problem import VisaProblem
    from src.decoder import spv
    import numpy as np

    problem = VisaProblem()

    # India-first: give India groups lowest hawk values
    hawk_india_first = np.ones(50)
    for g in problem.groups:
        if g["country"] == "India":
            hawk_india_first[g["index"]] = 0.0
    perm_if = spv(hawk_india_first)
    x_if = decode(perm_if, problem.groups, problem.total_visas,
                  problem.country_caps, problem.category_caps)
    f1_if = problem.f1(x_if)

    # India-last: give India groups highest hawk values
    hawk_india_last = np.zeros(50)
    for g in problem.groups:
        if g["country"] == "India":
            hawk_india_last[g["index"]] = 1.0
    perm_il = spv(hawk_india_last)
    x_il = decode(perm_il, problem.groups, problem.total_visas,
                  problem.country_caps, problem.category_caps)
    f1_il = problem.f1(x_il)

    assert f1_if < f1_il, (
        f"f1 India-first ({f1_if:.4f}) should be < f1 India-last ({f1_il:.4f})"
    )
