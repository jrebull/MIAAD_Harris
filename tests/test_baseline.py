"""Tests for the FIFO baseline simulation."""

import pytest
from src.baseline import fifo_permutation, run_baseline
from src.problem import VisaProblem


def test_fifo_order_oldest_first():
    """FIFO permutation should process oldest priority dates first."""
    problem = VisaProblem()
    perm = fifo_permutation(problem.groups)

    dates = [problem.groups[g_idx]["d"] for g_idx in perm]
    assert dates == sorted(dates), "FIFO should process oldest dates first"


def test_baseline_feasibility():
    """Baseline allocation should satisfy all constraints."""
    problem = VisaProblem()
    alloc, (f1, f2, f3) = run_baseline(problem)

    assert sum(alloc.values()) <= problem.total_visas
    for country, cap in problem.country_caps.items():
        total = sum(alloc[g["index"]] for g in problem.groups
                    if g["country"] == country)
        assert total <= cap
    for cat, cap in problem.category_caps.items():
        total = sum(alloc[g["index"]] for g in problem.groups
                    if g["category"] == cat)
        assert total <= cap


def test_baseline_f1_f2_values():
    """Baseline should produce f1≈7.21, f2≈12.64 (21-country data)."""
    problem = VisaProblem()
    _, (f1, f2, f3) = run_baseline(problem)

    assert abs(f1 - 7.2138) < 0.01, f"Expected f1≈7.2138, got {f1:.4f}"
    assert abs(f2 - 12.6377) < 0.01, f"Expected f2≈12.6377, got {f2:.4f}"
    assert f3 > 0, f"FIFO should waste visas, got f3={f3}"


def test_baseline_reproducible():
    """Baseline should always produce the same result (deterministic)."""
    problem = VisaProblem()
    _, fit1 = run_baseline(problem)
    _, fit2 = run_baseline(problem)
    assert fit1 == fit2
