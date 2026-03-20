"""Tests for Pareto dominance and archive operations.

Verifies:
    - Dominance relation is correct with known cases.
    - Archive never contains dominated solutions.
"""

import numpy as np
import pytest
from src.mohho import dominates, update_archive, crowding_distance


def test_dominance_basic():
    """Test dominance with obvious cases."""
    assert dominates((1.0, 1.0), (2.0, 2.0)) is True
    assert dominates((1.0, 1.0), (1.0, 2.0)) is True
    assert dominates((1.0, 1.0), (2.0, 1.0)) is True
    assert dominates((1.0, 2.0), (2.0, 1.0)) is False
    assert dominates((2.0, 1.0), (1.0, 2.0)) is False
    assert dominates((1.0, 1.0), (1.0, 1.0)) is False
    assert dominates((2.0, 2.0), (1.0, 1.0)) is False


def test_archive_no_dominated_solutions():
    """Archive should never contain dominated solutions."""
    positions = []
    fitnesses = []

    rng = np.random.default_rng(42)
    for _ in range(200):
        pos = rng.uniform(0, 1, size=50)
        fit = (rng.random() * 10, rng.random() * 15)
        update_archive(positions, fitnesses, pos, fit, max_size=100)

    for i in range(len(fitnesses)):
        for j in range(len(fitnesses)):
            if i != j:
                assert not dominates(fitnesses[j], fitnesses[i]), (
                    f"Solution {j} ({fitnesses[j]}) dominates {i} ({fitnesses[i]})"
                )


def test_archive_size_limit():
    """Archive should not exceed max_size."""
    positions = []
    fitnesses = []
    rng = np.random.default_rng(7)

    for _ in range(500):
        pos = rng.uniform(0, 1, size=50)
        fit = (rng.random() * 10, rng.random() * 15)
        update_archive(positions, fitnesses, pos, fit, max_size=50)

    assert len(positions) <= 50
    assert len(fitnesses) <= 50


def test_crowding_distance_extremes():
    """Extreme points should have infinite crowding distance."""
    fits = [(0.0, 10.0), (3.0, 5.0), (5.0, 3.0), (10.0, 0.0)]
    cd = crowding_distance(fits)
    assert cd[0] == float("inf")
    assert cd[-1] == float("inf")
    assert cd[1] > 0
    assert cd[2] > 0


def test_crowding_distance_two_points():
    """With 2 or fewer points, all should have infinite distance."""
    assert crowding_distance([(1.0, 2.0), (3.0, 1.0)]) == [float("inf"), float("inf")]
    assert crowding_distance([(1.0, 2.0)]) == [float("inf")]
