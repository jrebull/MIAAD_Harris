"""Tests for objective functions f1 and f2.

Verifies:
    - f1, f2 match hand-calculated values from LaTeX example.
    - Direction of f1: serving India first -> lower f1.
    - Production code (VisaProblem.f1, .f2) matches local calculations.
"""

import numpy as np
import pytest
from src.decoder import decode
from src.problem import VisaProblem


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
    w_max = {c: max(EXAMPLE_GROUPS[i]["w"] for i in idxs)
             for c, idxs in countries.items()}
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

    Check 2 of the audit cycle: f1 with x_India=0 should be >> f1 when
    India is served. Matches the LaTeX verification in Section 6.4.
    """
    x_served = {0: 20, 1: 30, 2: 20, 3: 15, 4: 10, 5: 5}
    f1_served = _f1(x_served)

    x_ignored = {0: 0, 1: 0, 2: 20, 3: 15, 4: 10, 5: 5}
    f1_ignored = _f1(x_ignored)

    assert f1_served < f1_ignored, (
        f"f1 with India served ({f1_served:.4f}) should be < "
        f"f1 India ignored ({f1_ignored:.4f})"
    )


def test_f1_direction_full_problem():
    """Direction check on the full 50-group problem using production code."""
    from src.decoder import spv

    problem = VisaProblem()

    hawk_india_first = np.ones(50)
    for g in problem.groups:
        if g["country"] == "India":
            hawk_india_first[g["index"]] = 0.0
    perm_if = spv(hawk_india_first)
    x_if = decode(perm_if, problem.groups, problem.total_visas,
                  problem.country_caps, problem.category_caps)
    f1_if = problem.f1(x_if)

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


def test_production_f1_matches_manual():
    """Verify VisaProblem.f1 against manual calculation on full problem."""
    problem = VisaProblem()
    from src.decoder import spv

    hawk = np.linspace(0, 1, 50)
    perm = spv(hawk)
    x = decode(perm, problem.groups, problem.total_visas,
               problem.country_caps, problem.category_caps)

    f1_prod = problem.f1(x)
    f1_manual = sum(
        (g["n"] - x[g["index"]]) * g["w"] for g in problem.groups
    ) / problem.total_demand
    assert f1_prod == pytest.approx(f1_manual)


def test_production_f2_matches_manual():
    """Verify VisaProblem.f2 against manual calculation on full problem."""
    problem = VisaProblem()
    from src.decoder import spv

    hawk = np.linspace(0, 1, 50)
    perm = spv(hawk)
    x = decode(perm, problem.groups, problem.total_visas,
               problem.country_caps, problem.category_caps)

    f2_prod = problem.f2(x)

    # Manual f2
    from src.config import COUNTRIES
    w_vals = []
    for country in COUNTRIES:
        gs = [g for g in problem.groups if g["country"] == country]
        total_assigned = sum(x[g["index"]] for g in gs)
        if total_assigned > 0:
            w_vals.append(sum(x[g["index"]] * g["w"] for g in gs) / total_assigned)
        else:
            w_vals.append(max(g["w"] for g in gs))

    f2_manual = max(w_vals) - min(w_vals)
    assert f2_prod == pytest.approx(f2_manual)


def test_f2_zero_visa_country():
    """f2 should use w_c_max when a country gets 0 visas."""
    problem = VisaProblem()

    x = {g["index"]: 0 for g in problem.groups}
    # Give all visas to India only
    for g in problem.groups:
        if g["country"] == "India" and g["category"] == "EB-1":
            x[g["index"]] = 25_620
            break

    f2 = problem.f2(x)
    assert f2 > 0, "f2 should be > 0 when most countries get 0 visas"
