"""Integration test: run MOHHO for a short period and verify results.

Verifies:
    - Pareto front is non-empty after 10 iterations.
    - All solutions in the front are feasible.
    - Reproducibility with same seed.
"""

import numpy as np
import pytest
from src.problem import VisaProblem
from src.mohho import run_mohho, dominates
from src.decoder import spv, decode


def test_mohho_short_run():
    """Run MOHHO for 10 iterations and verify basic properties."""
    problem = VisaProblem()
    positions, fitnesses, hv_history = run_mohho(
        problem, seed=42, pop_size=20, max_iter=10, archive_size=50
    )

    assert len(positions) > 0, "Pareto front should be non-empty"
    assert len(fitnesses) == len(positions)
    assert len(hv_history) == 10

    for pos in positions:
        perm = spv(pos)
        alloc = decode(perm, problem.groups, problem.total_visas,
                       problem.country_caps, problem.category_caps)

        assert sum(alloc.values()) <= problem.total_visas
        for country, cap in problem.country_caps.items():
            total = sum(alloc[g["index"]] for g in problem.groups
                        if g["country"] == country)
            assert total <= cap
        for cat, cap in problem.category_caps.items():
            total = sum(alloc[g["index"]] for g in problem.groups
                        if g["category"] == cat)
            assert total <= cap
        for g in problem.groups:
            assert 0 <= alloc[g["index"]] <= g["n"]


def test_pareto_front_no_dominated():
    """No solution in the Pareto front should dominate another."""
    problem = VisaProblem()
    _, fitnesses, _ = run_mohho(
        problem, seed=42, pop_size=20, max_iter=10, archive_size=50
    )

    for i in range(len(fitnesses)):
        for j in range(len(fitnesses)):
            if i != j:
                assert not dominates(fitnesses[j], fitnesses[i])


def test_reproducibility():
    """Same seed must produce identical results."""
    problem = VisaProblem()

    _, fit1, hv1 = run_mohho(problem, seed=42, pop_size=10, max_iter=5, archive_size=20)
    _, fit2, hv2 = run_mohho(problem, seed=42, pop_size=10, max_iter=5, archive_size=20)

    assert len(fit1) == len(fit2)
    for a, b in zip(fit1, fit2):
        assert a[0] == pytest.approx(b[0])
        assert a[1] == pytest.approx(b[1])
    for a, b in zip(hv1, hv2):
        assert a == pytest.approx(b)
