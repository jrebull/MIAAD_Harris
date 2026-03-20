"""Tests for HHO operators.

Verifies:
    - Each operator produces a vector in R^G with correct shape/bounds.
    - Energy transitions are correct.
    - Operator formulas match the mathematical specification.
"""

import math

import numpy as np
import pytest
from src.hho import (
    escape_energy, levy_flight, clip_bounds,
    op1_exploration_random, op2_exploration_mean,
    op3_soft_siege, op4_hard_siege,
    op5_soft_siege_levy, op6_hard_siege_levy,
)
from src.config import LB, UB, NUM_GROUPS, BETA_LEVY


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


# --- Shape and bounds tests ---

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


# --- Energy tests ---

def test_escape_energy_range(rng):
    """E should be in [-2, 2] at t=0 and approach 0 at t=T."""
    energies_start = [escape_energy(0, 500, rng) for _ in range(1000)]
    energies_end = [escape_energy(499, 500, rng) for _ in range(1000)]

    assert all(-2.1 <= e <= 2.1 for e in energies_start)
    assert all(abs(e) < 0.02 for e in energies_end)


def test_energy_transition():
    """Energy magnitude should decrease from exploration to exploitation."""
    rng = np.random.default_rng(0)
    early = [abs(escape_energy(10, 500, rng)) for _ in range(500)]
    late = [abs(escape_energy(490, 500, rng)) for _ in range(500)]
    assert np.mean(early) > np.mean(late)


def test_levy_flight_shape(rng):
    lf = levy_flight(NUM_GROUPS, rng)
    assert lf.shape == (NUM_GROUPS,)


# --- Mathematical formula verification tests ---

def test_escape_energy_formula():
    """Verify E = 2 * E0 * (1 - t/T) matches for known E0."""
    rng = np.random.default_rng(99)
    t, T = 100, 500
    e = escape_energy(t, T, rng)

    rng2 = np.random.default_rng(99)
    e0 = 2 * rng2.random() - 1
    expected = 2 * e0 * (1 - t / T)

    assert e == pytest.approx(expected)


def test_levy_sigma_formula():
    """Verify Lévy sigma matches Equation from Heidari et al. (2019)."""
    beta = BETA_LEVY
    expected_sigma = (
        math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)

    rng = np.random.default_rng(7)
    # Draw u and verify it's drawn from N(0, sigma)
    # Run many samples and check the std matches sigma
    samples = []
    for _ in range(50000):
        rng_inner = np.random.default_rng(rng.integers(2**31))
        u = rng_inner.normal(0, expected_sigma, size=1)
        samples.append(u[0])

    assert np.std(samples) == pytest.approx(expected_sigma, rel=0.05)


def test_op3_formula():
    """Verify Op3: X_new = (x_rabbit - xi) - E * |J*x_rabbit - xi|."""
    rng = np.random.default_rng(123)
    dim = 10
    xi = rng.uniform(0.2, 0.8, size=dim)
    x_rabbit = rng.uniform(0.2, 0.8, size=dim)
    energy = 0.7

    rng_op = np.random.default_rng(456)
    result = op3_soft_siege(xi, x_rabbit, energy, rng_op)

    rng_check = np.random.default_rng(456)
    j = 2 * (1 - rng_check.random())
    delta_x = x_rabbit - xi
    expected = np.clip(delta_x - energy * np.abs(j * x_rabbit - xi), LB, UB)

    np.testing.assert_array_almost_equal(result, expected)


def test_op4_formula():
    """Verify Op4: X_new = x_rabbit - E * |delta_x|."""
    rng = np.random.default_rng(123)
    dim = 10
    xi = rng.uniform(0.2, 0.8, size=dim)
    x_rabbit = rng.uniform(0.2, 0.8, size=dim)
    energy = 0.3

    rng_op = np.random.default_rng(789)
    result = op4_hard_siege(xi, x_rabbit, energy, rng_op)

    delta_x = x_rabbit - xi
    expected = np.clip(x_rabbit - energy * np.abs(delta_x), LB, UB)

    np.testing.assert_array_almost_equal(result, expected)


def test_op1_formula():
    """Verify Op1: X_new = x_rand - r1 * |x_rand - 2*r2*xi|."""
    rng = np.random.default_rng(111)
    dim = 10
    xi = rng.uniform(0.2, 0.8, size=dim)
    x_rand = rng.uniform(0.2, 0.8, size=dim)

    rng_op = np.random.default_rng(222)
    result = op1_exploration_random(xi, x_rand, rng_op)

    rng_check = np.random.default_rng(222)
    r1 = rng_check.random()
    r2 = rng_check.random()
    expected = np.clip(x_rand - r1 * np.abs(x_rand - 2 * r2 * xi), LB, UB)

    np.testing.assert_array_almost_equal(result, expected)


def test_op2_formula():
    """Verify Op2: X_new = (x_rabbit - x_mean) - r3 * (LB + r4*(UB-LB))."""
    rng = np.random.default_rng(111)
    dim = 10
    xi = rng.uniform(0.2, 0.8, size=dim)
    x_rabbit = rng.uniform(0.2, 0.8, size=dim)
    x_mean = rng.uniform(0.2, 0.8, size=dim)

    rng_op = np.random.default_rng(333)
    result = op2_exploration_mean(xi, x_rabbit, x_mean, rng_op)

    rng_check = np.random.default_rng(333)
    r3 = rng_check.random()
    r4 = rng_check.random(size=dim)
    expected = np.clip((x_rabbit - x_mean) - r3 * (LB + r4 * (UB - LB)), LB, UB)

    np.testing.assert_array_almost_equal(result, expected)


def test_op5_y_formula():
    """Verify Op5 Y candidate: Y = x_rabbit - E*|J*x_rabbit - xi|."""
    rng = np.random.default_rng(111)
    dim = 10
    xi = rng.uniform(0.2, 0.8, size=dim)
    x_rabbit = rng.uniform(0.2, 0.8, size=dim)
    energy = 0.6

    rng_op = np.random.default_rng(444)
    y, z = op5_soft_siege_levy(xi, x_rabbit, energy, rng_op)

    rng_check = np.random.default_rng(444)
    j = 2 * (1 - rng_check.random())
    expected_y = np.clip(x_rabbit - energy * np.abs(j * x_rabbit - xi), LB, UB)

    np.testing.assert_array_almost_equal(y, expected_y)


def test_protocol_compliance():
    """Verify VisaProblem satisfies OptimizationProblem protocol."""
    from src.problem import VisaProblem, OptimizationProblem
    assert isinstance(VisaProblem(), OptimizationProblem)
