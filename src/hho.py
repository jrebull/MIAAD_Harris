"""Harris Hawks Optimization — 6 operators (Heidari et al., 2019).

The operators move vectors in R^G. Conversion to allocation is ALWAYS
done via SPV+decoder AFTER the movement. Operators NEVER touch x_g directly.

References:
    - Heidari et al. (2019), Future Generation Computer Systems, 97, 849-872.
    - HarrisFase01 LaTeX (Fase 1) for detailed operator equations.
"""

import math
import numpy as np
from numpy.typing import NDArray
from src.config import BETA_LEVY, LB, UB


def levy_flight(dim: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Generate a Lévy flight step vector.

    LF(x) = 0.01 * u * sigma / |v|^(1/beta), beta = 1.5

    Args:
        dim: Dimensionality of the vector.
        rng: NumPy random generator.

    Returns:
        Lévy flight vector of shape (dim,).
    """
    beta = BETA_LEVY
    sigma = (
        math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)

    u = rng.normal(0, sigma, size=dim)
    v = rng.normal(0, 1, size=dim)
    step = 0.01 * u / (np.abs(v) ** (1 / beta))
    return step


def clip_bounds(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Clip vector to [LB, UB] bounds.

    Args:
        x: Input vector.

    Returns:
        Clipped vector.
    """
    return np.clip(x, LB, UB)


def escape_energy(t: int, max_t: int, rng: np.random.Generator) -> float:
    """Compute escape energy E.

    E = 2 * E0 * (1 - t/T), where E0 ~ Uniform(-1, 1).

    Args:
        t: Current iteration.
        max_t: Maximum iterations.
        rng: NumPy random generator.

    Returns:
        Escape energy value.
    """
    e0 = 2 * rng.random() - 1
    return 2 * e0 * (1 - t / max_t)


def op1_exploration_random(
    xi: NDArray[np.float64],
    x_rand: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Operator 1: Exploration (q >= 0.5).

    X_new = X_rand - r1 * |X_rand - 2*r2*X_i|

    Args:
        xi: Current hawk position.
        x_rand: Random hawk position.
        rng: NumPy random generator.

    Returns:
        New position vector.
    """
    r1 = rng.random()
    r2 = rng.random()
    return clip_bounds(x_rand - r1 * np.abs(x_rand - 2 * r2 * xi))


def op2_exploration_mean(
    xi: NDArray[np.float64],
    x_rabbit: NDArray[np.float64],
    x_mean: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Operator 2: Exploration (q < 0.5).

    X_new = (X_rabbit - X_mean) - r3 * (LB + r4 * (UB - LB))

    Args:
        xi: Current hawk position.
        x_rabbit: Best solution (leader) position.
        x_mean: Mean position of population.
        rng: NumPy random generator.

    Returns:
        New position vector.
    """
    dim = xi.shape[0]
    r3 = rng.random()
    r4 = rng.random(size=dim)
    return clip_bounds((x_rabbit - x_mean) - r3 * (LB + r4 * (UB - LB)))


def op3_soft_siege(
    xi: NDArray[np.float64],
    x_rabbit: NDArray[np.float64],
    energy: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Operator 3: Soft siege (r >= 0.5, |E| >= 0.5).

    delta_x = X_rabbit - X_i
    X_new = delta_x - E * |J * X_rabbit - X_i|

    Args:
        xi: Current hawk position.
        x_rabbit: Leader position.
        energy: Escape energy E.
        rng: NumPy random generator.

    Returns:
        New position vector.
    """
    j = 2 * (1 - rng.random())
    delta_x = x_rabbit - xi
    return clip_bounds(delta_x - energy * np.abs(j * x_rabbit - xi))


def op4_hard_siege(
    xi: NDArray[np.float64],
    x_rabbit: NDArray[np.float64],
    energy: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Operator 4: Hard siege (r >= 0.5, |E| < 0.5).

    X_new = X_rabbit - E * |delta_x|

    Args:
        xi: Current hawk position.
        x_rabbit: Leader position.
        energy: Escape energy E.
        rng: NumPy random generator.

    Returns:
        New position vector.
    """
    delta_x = x_rabbit - xi
    return clip_bounds(x_rabbit - energy * np.abs(delta_x))


def op5_soft_siege_levy(
    xi: NDArray[np.float64],
    x_rabbit: NDArray[np.float64],
    energy: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Operator 5: Soft siege with Lévy (r < 0.5, |E| >= 0.5).

    Y = X_rabbit - E * |J * X_rabbit - X_i|
    Z = Y + S * LF(dim), where S is random vector in [0,1]^dim

    Try Y first; if not better, try Z; if not better, keep X_i.
    Selection is handled by the caller (mohho.py).

    Args:
        xi: Current hawk position.
        x_rabbit: Leader position.
        energy: Escape energy E.
        rng: NumPy random generator.

    Returns:
        Tuple of (Y, Z) candidates.
    """
    dim = xi.shape[0]
    j = 2 * (1 - rng.random())
    y = clip_bounds(x_rabbit - energy * np.abs(j * x_rabbit - xi))
    s = rng.random(size=dim)
    z = clip_bounds(y + s * levy_flight(dim, rng))
    return y, z


def op6_hard_siege_levy(
    xi: NDArray[np.float64],
    x_rabbit: NDArray[np.float64],
    energy: float,
    x_mean: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Operator 6: Hard siege with Lévy (r < 0.5, |E| < 0.5).

    Y = X_rabbit - E * |J * X_rabbit - X_mean|
    Z = Y + S * LF(dim)

    Args:
        xi: Current hawk position.
        x_rabbit: Leader position.
        energy: Escape energy E.
        x_mean: Mean position of population.
        rng: NumPy random generator.

    Returns:
        Tuple of (Y, Z) candidates.
    """
    dim = xi.shape[0]
    j = 2 * (1 - rng.random())
    y = clip_bounds(x_rabbit - energy * np.abs(j * x_rabbit - x_mean))
    s = rng.random(size=dim)
    z = clip_bounds(y + s * levy_flight(dim, rng))
    return y, z
