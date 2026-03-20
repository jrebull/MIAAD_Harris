"""Tests for HHO operators.

Verifies:
    - Each operator produces a vector in R^G.
    - Energy transitions are correct.
"""

import numpy as np
import pytest
from src.hho import (
    escape_energy, levy_flight,
    op1_exploration_random, op2_exploration_mean,
    op3_soft_siege, op4_hard_siege,
    op5_soft_siege_levy, op6_hard_siege_levy,
)
from src.config import LB, UB, NUM_GROUPS


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_vectors(rng):
    xi = rng.uniform(LB, UB, size=NUM_GROUPS)
    x_rabbit = rng.uniform(LB, UB, size=NUM_GROUPS)
    x_mean = rng.uniform(LB, UB, size=NUM_GROUPS)
    x_rand = rng.uniform(LB, UB, size=NUM_GROUPS)
    return xi, x_rabbit, x_mean, x_rand


def test_op1_shape_and_bounds(rng, sample_vectors):
    xi, _, _, x_rand = sample_vectors
    result = op1_exploration_random(xi, x_rand, rng)
    assert result.shape == (NUM_GROUPS,)
    assert np.all(result >= LB) and np.all(result <= UB)


def test_op2_shape_and_bounds(rng, sample_vectors):
    xi, x_rabbit, x_mean, _ = sample_vectors
    result = op2_exploration_mean(xi, x_rabbit, x_mean, rng)
    assert result.shape == (NUM_GROUPS,)
    assert np.all(result >= LB) and np.all(result <= UB)


def test_op3_shape_and_bounds(rng, sample_vectors):
    xi, x_rabbit, _, _ = sample_vectors
    result = op3_soft_siege(xi, x_rabbit, 0.7, rng)
    assert result.shape == (NUM_GROUPS,)
    assert np.all(result >= LB) and np.all(result <= UB)


def test_op4_shape_and_bounds(rng, sample_vectors):
    xi, x_rabbit, _, _ = sample_vectors
    result = op4_hard_siege(xi, x_rabbit, 0.3, rng)
    assert result.shape == (NUM_GROUPS,)
    assert np.all(result >= LB) and np.all(result <= UB)


def test_op5_shape_and_bounds(rng, sample_vectors):
    xi, x_rabbit, _, _ = sample_vectors
    y, z = op5_soft_siege_levy(xi, x_rabbit, 0.7, rng)
    assert y.shape == (NUM_GROUPS,)
    assert z.shape == (NUM_GROUPS,)
    assert np.all(y >= LB) and np.all(y <= UB)
    assert np.all(z >= LB) and np.all(z <= UB)


def test_op6_shape_and_bounds(rng, sample_vectors):
    xi, x_rabbit, x_mean, _ = sample_vectors
    y, z = op6_hard_siege_levy(xi, x_rabbit, 0.3, x_mean, rng)
    assert y.shape == (NUM_GROUPS,)
    assert z.shape == (NUM_GROUPS,)
    assert np.all(y >= LB) and np.all(y <= UB)
    assert np.all(z >= LB) and np.all(z <= UB)


def test_escape_energy_range(rng):
    """E should be in [-2, 2] at t=0 and approach 0 at t=T."""
    energies_start = [escape_energy(0, 500, rng) for _ in range(1000)]
    energies_end = [escape_energy(499, 500, rng) for _ in range(1000)]

    assert all(-2.1 <= e <= 2.1 for e in energies_start)
    assert all(abs(e) < 0.02 for e in energies_end)


def test_levy_flight_shape(rng):
    lf = levy_flight(NUM_GROUPS, rng)
    assert lf.shape == (NUM_GROUPS,)


def test_energy_transition():
    """Energy magnitude should decrease from exploration to exploitation."""
    rng = np.random.default_rng(0)
    early = [abs(escape_energy(10, 500, rng)) for _ in range(500)]
    late = [abs(escape_energy(490, 500, rng)) for _ in range(500)]
    assert np.mean(early) > np.mean(late)
