"""Tests for the SPV decoder.

Verifies:
    - LaTeX example (G=6, V=100) produces expected allocation.
    - Every permutation generates a feasible allocation (R1-R5).
"""

import numpy as np
import pytest
from src.decoder import spv, decode


# --- LaTeX example data (Section 6.4 of HarrisFase02.tex) ---
EXAMPLE_GROUPS = [
    {"index": 0, "country": "India", "category": "EB-2", "n": 40, "d": 2013, "w": 13},
    {"index": 1, "country": "India", "category": "EB-3", "n": 30, "d": 2010, "w": 16},
    {"index": 2, "country": "China", "category": "EB-2", "n": 20, "d": 2019, "w": 7},
    {"index": 3, "country": "China", "category": "EB-3", "n": 15, "d": 2020, "w": 6},
    {"index": 4, "country": "Canada", "category": "EB-2", "n": 10, "d": 2025, "w": 1},
    {"index": 5, "country": "Canada", "category": "EB-3", "n": 5, "d": 2024, "w": 2},
]
EXAMPLE_V = 100
EXAMPLE_COUNTRY_CAPS = {"India": 50, "China": 50, "Canada": 50}
EXAMPLE_CAT_CAPS = {"EB-2": 50, "EB-3": 50}


def test_spv_latex_example():
    """Test SPV with the LaTeX example hawk vector."""
    hawk = np.array([0.72, 0.15, 0.91, 0.33, 0.88, 0.47])
    perm = spv(hawk)
    assert perm == [1, 3, 5, 0, 4, 2]


def test_decode_latex_example():
    """Test decoder with LaTeX example: expected x = {20, 30, 20, 15, 10, 5}."""
    perm = [1, 3, 5, 0, 4, 2]
    x = decode(perm, EXAMPLE_GROUPS, EXAMPLE_V,
               EXAMPLE_COUNTRY_CAPS, EXAMPLE_CAT_CAPS)
    assert x[0] == 20, f"India-EB2: expected 20, got {x[0]}"
    assert x[1] == 30, f"India-EB3: expected 30, got {x[1]}"
    assert x[2] == 20, f"China-EB2: expected 20, got {x[2]}"
    assert x[3] == 15, f"China-EB3: expected 15, got {x[3]}"
    assert x[4] == 10, f"Canada-EB2: expected 10, got {x[4]}"
    assert x[5] == 5, f"Canada-EB3: expected 5, got {x[5]}"
    assert sum(x.values()) == 100


def test_feasibility_random_permutations():
    """Check 10,000 random permutations all produce feasible allocations."""
    rng = np.random.default_rng(42)

    for _ in range(10_000):
        hawk = rng.uniform(0, 1, size=6)
        perm = spv(hawk)
        x = decode(perm, EXAMPLE_GROUPS, EXAMPLE_V,
                   EXAMPLE_COUNTRY_CAPS, EXAMPLE_CAT_CAPS)

        # R1: total <= V
        assert sum(x.values()) <= EXAMPLE_V

        # R2: per-country cap
        for country in ["India", "China", "Canada"]:
            country_total = sum(
                x[g["index"]] for g in EXAMPLE_GROUPS if g["country"] == country
            )
            assert country_total <= EXAMPLE_COUNTRY_CAPS[country]

        # R3: per-category cap
        for cat in ["EB-2", "EB-3"]:
            cat_total = sum(
                x[g["index"]] for g in EXAMPLE_GROUPS if g["category"] == cat
            )
            assert cat_total <= EXAMPLE_CAT_CAPS[cat]

        # R4: 0 <= x_g <= n_g
        for g in EXAMPLE_GROUPS:
            assert 0 <= x[g["index"]] <= g["n"]

        # R5: integers
        for v in x.values():
            assert isinstance(v, (int, np.integer))


def test_feasibility_full_problem():
    """Check 10,000 random permutations on the full problem."""
    from src.problem import VisaProblem
    from src.config import NUM_GROUPS
    problem = VisaProblem()
    rng = np.random.default_rng(123)

    for _ in range(10_000):
        hawk = rng.uniform(0, 1, size=NUM_GROUPS)
        perm = spv(hawk)
        x = decode(perm, problem.groups, problem.total_visas,
                   problem.country_caps, problem.category_caps)

        # R1
        assert sum(x.values()) <= problem.total_visas

        # R2
        for country, cap in problem.country_caps.items():
            total = sum(x[g["index"]] for g in problem.groups if g["country"] == country)
            assert total <= cap, f"{country}: {total} > {cap}"

        # R3
        for cat, cap in problem.category_caps.items():
            total = sum(x[g["index"]] for g in problem.groups if g["category"] == cat)
            assert total <= cap, f"{cat}: {total} > {cap}"

        # R4
        for g in problem.groups:
            assert 0 <= x[g["index"]] <= g["n"]
