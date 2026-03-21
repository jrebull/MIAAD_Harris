"""Multi-Objective Harris Hawks Optimization (MOHHO).

Combines HHO operators with:
    1. External Pareto archive (max size configurable)
    2. Pareto dominance comparison (3 objectives)
    3. Leader selection via crowding distance (roulette weighted)
    4. Archive update after each evaluation

References:
    - Heidari et al. (2019) — base HHO
    - Kusoglu & Yüzgeç (2020) — multi-objective HHO
    - HarrisFase02.tex, Section 8
"""

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from src.config import (
    POPULATION_SIZE, MAX_ITERATIONS, ARCHIVE_SIZE,
    LB, UB, NUM_GROUPS,
)
from src.decoder import spv, decode
from src.hho import (
    escape_energy,
    op1_exploration_random, op2_exploration_mean,
    op3_soft_siege, op4_hard_siege,
    op5_soft_siege_levy, op6_hard_siege_levy,
)
from src.problem import VisaProblem

# Fixed reference point for 3D hypervolume (worst plausible f1, f2, f3 + margin).
# f1 theoretical max ~ 8.5, f2 theoretical max ~ 14, f3 theoretical max ~ 42000.
HV_REF_POINT: tuple[float, float, float] = (10.0, 16.0, 50_000.0)

# Cap for replacing inf in crowding distance roulette selection
CD_INF_REPLACEMENT: float = 1e6

# 3-tuple type alias
Fitness3 = tuple[float, float, float]


def dominates(a: Fitness3, b: Fitness3) -> bool:
    """Check if solution a dominates solution b (3 objectives).

    a dominates b iff: for all m, f_m(a) <= f_m(b) AND exists m: f_m(a) < f_m(b).

    Args:
        a: Fitness tuple (f1, f2, f3) of solution a.
        b: Fitness tuple (f1, f2, f3) of solution b.

    Returns:
        True if a dominates b.
    """
    return (a[0] <= b[0] and a[1] <= b[1] and a[2] <= b[2]) and \
           (a[0] < b[0] or a[1] < b[1] or a[2] < b[2])


def crowding_distance(fitnesses: list[Fitness3]) -> list[float]:
    """Compute crowding distance for a set of fitness values (3 objectives).

    Args:
        fitnesses: List of (f1, f2, f3) tuples.

    Returns:
        List of crowding distances, same order as input.
    """
    n = len(fitnesses)
    if n <= 2:
        return [float("inf")] * n

    distances = [0.0] * n

    for m in range(3):
        indices = sorted(range(n), key=lambda i: fitnesses[i][m])
        distances[indices[0]] = float("inf")
        distances[indices[-1]] = float("inf")

        f_min = fitnesses[indices[0]][m]
        f_max = fitnesses[indices[-1]][m]
        span = f_max - f_min
        if span == 0:
            continue

        for k in range(1, n - 1):
            diff = fitnesses[indices[k + 1]][m] - fitnesses[indices[k - 1]][m]
            distances[indices[k]] += diff / span

    return distances


def select_leader(
    archive_positions: list[NDArray[np.float64]],
    archive_fitnesses: list[Fitness3],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Select a leader from the Pareto archive using crowding distance roulette.

    Solutions in less crowded regions have higher selection probability.

    Args:
        archive_positions: List of hawk position vectors in the archive.
        archive_fitnesses: List of fitness tuples in the archive.
        rng: NumPy random generator.

    Returns:
        Position vector of the selected leader.
    """
    cd = crowding_distance(archive_fitnesses)
    cd_finite = [d if d != float("inf") else CD_INF_REPLACEMENT for d in cd]
    total = sum(cd_finite)
    if total == 0:
        return archive_positions[rng.integers(len(archive_positions))]
    probs = np.array(cd_finite) / total
    idx = rng.choice(len(archive_positions), p=probs)
    return archive_positions[idx]


def _is_duplicate(
    new_fit: Fitness3,
    archive_fitnesses: list[Fitness3],
) -> bool:
    """Check if a fitness tuple already exists in the archive.

    Args:
        new_fit: Candidate fitness.
        archive_fitnesses: Existing archive fitnesses.

    Returns:
        True if a near-duplicate exists.
    """
    for af in archive_fitnesses:
        if (abs(af[0] - new_fit[0]) < 1e-12 and
            abs(af[1] - new_fit[1]) < 1e-12 and
            abs(af[2] - new_fit[2]) < 1e-12):
            return True
    return False


def update_archive(
    archive_positions: list[NDArray[np.float64]],
    archive_fitnesses: list[Fitness3],
    new_pos: NDArray[np.float64],
    new_fit: Fitness3,
    max_size: int,
) -> None:
    """Update the Pareto archive with a new solution (in-place).

    Inserts new solution if non-dominated and not a duplicate.
    Removes solutions it dominates. If archive exceeds max_size,
    removes the most crowded solution.

    Args:
        archive_positions: List of position vectors (modified in-place).
        archive_fitnesses: List of fitness tuples (modified in-place).
        new_pos: New candidate position.
        new_fit: New candidate fitness.
        max_size: Maximum archive size.
    """
    if _is_duplicate(new_fit, archive_fitnesses):
        return

    dominated_by_new = []
    for i, af in enumerate(archive_fitnesses):
        if dominates(af, new_fit):
            return
        if dominates(new_fit, af):
            dominated_by_new.append(i)

    for i in sorted(dominated_by_new, reverse=True):
        archive_positions.pop(i)
        archive_fitnesses.pop(i)

    archive_positions.append(new_pos.copy())
    archive_fitnesses.append(new_fit)

    if len(archive_positions) > max_size:
        cd = crowding_distance(archive_fitnesses)
        # Find the most crowded non-boundary solution to remove
        finite_indices = [i for i in range(len(cd)) if cd[i] != float("inf")]
        if finite_indices:
            min_cd_idx = min(finite_indices, key=lambda i: cd[i])
        else:
            # All inf: remove random non-extreme
            min_cd_idx = rng_fallback_idx(len(archive_positions))
        archive_positions.pop(min_cd_idx)
        archive_fitnesses.pop(min_cd_idx)


def rng_fallback_idx(n: int) -> int:
    """Return a random non-extreme index when all CDs are infinite.

    Avoids removing extreme solutions (first/last) to preserve
    Pareto front boundaries.

    Args:
        n: Number of archive members.

    Returns:
        Random index in [1, n-2], or 0 if n <= 2.
    """
    import random
    if n <= 2:
        return 0
    return random.randint(1, n - 2)


def evaluate_hawk(
    hawk: NDArray[np.float64], problem: VisaProblem
) -> tuple[dict[int, int], Fitness3]:
    """Decode a hawk position and evaluate its fitness.

    Args:
        hawk: Continuous vector in R^G.
        problem: VisaProblem instance.

    Returns:
        Tuple of (allocation dict, fitness tuple).
    """
    perm = spv(hawk)
    alloc = decode(perm, problem.groups, problem.total_visas,
                   problem.country_caps, problem.category_caps)
    fitness = problem.evaluate(alloc)
    return alloc, fitness


def _greedy_select_levy(
    xi: NDArray[np.float64],
    fit_i: Fitness3,
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    problem: VisaProblem,
) -> tuple[
    tuple[NDArray[np.float64], Fitness3] | None,
    list[tuple[NDArray[np.float64], Fitness3]],
]:
    """Greedy selection for Lévy operators (Ops 5-6).

    Try Y first: adopt if Y dominates X_i.
    Otherwise try Z: adopt if Z dominates X_i.
    Otherwise keep X_i (return None).

    Also returns all evaluated candidates so the caller can offer them
    to the archive even when they don't dominate the current hawk.

    Args:
        xi: Current hawk position.
        fit_i: Current hawk fitness.
        y: First candidate (Y).
        z: Second candidate (Z).
        problem: VisaProblem instance.

    Returns:
        Tuple of (winner_or_None, list_of_all_candidates).
    """
    candidates: list[tuple[NDArray[np.float64], Fitness3]] = []

    _, fit_y = evaluate_hawk(y, problem)
    candidates.append((y, fit_y))
    if dominates(fit_y, fit_i):
        return (y, fit_y), candidates

    _, fit_z = evaluate_hawk(z, problem)
    candidates.append((z, fit_z))
    if dominates(fit_z, fit_i):
        return (z, fit_z), candidates

    return None, candidates


def _step_hawk(
    i: int,
    population: NDArray[np.float64],
    fitnesses: list[Fitness3],
    archive_positions: list[NDArray[np.float64]],
    archive_fitnesses: list[Fitness3],
    x_mean: NDArray[np.float64],
    t: int,
    max_iter: int,
    pop_size: int,
    archive_size: int,
    problem: VisaProblem,
    rng: np.random.Generator,
) -> None:
    """Process one hawk in the MOHHO iteration (in-place update).

    Args:
        i: Hawk index.
        population: Population matrix (modified in-place).
        fitnesses: Population fitnesses (modified in-place).
        archive_positions: Pareto archive positions (modified in-place).
        archive_fitnesses: Pareto archive fitnesses (modified in-place).
        x_mean: Mean population position.
        t: Current iteration.
        max_iter: Max iterations.
        pop_size: Population size.
        archive_size: Max archive size.
        problem: VisaProblem instance.
        rng: Random generator.
    """
    e = escape_energy(t, max_iter, rng)
    abs_e = abs(e)
    x_rabbit = select_leader(archive_positions, archive_fitnesses, rng)

    if abs_e >= 1:
        new_pos = _exploration_step(population, i, x_rabbit, x_mean, pop_size, rng)
        _accept_and_archive(
            i, new_pos, population, fitnesses,
            archive_positions, archive_fitnesses, archive_size, problem)
    elif rng.random() >= 0.5:
        new_pos = _siege_step(population[i], x_rabbit, e, abs_e, rng)
        _accept_and_archive(
            i, new_pos, population, fitnesses,
            archive_positions, archive_fitnesses, archive_size, problem)
    else:
        _levy_step(
            i, population, fitnesses, x_rabbit, x_mean, e, abs_e,
            archive_positions, archive_fitnesses, archive_size, problem, rng)


def _exploration_step(
    population: NDArray[np.float64],
    i: int,
    x_rabbit: NDArray[np.float64],
    x_mean: NDArray[np.float64],
    pop_size: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Execute exploration operator (Op1 or Op2).

    Args:
        population: Population matrix.
        i: Current hawk index.
        x_rabbit: Leader position.
        x_mean: Mean position.
        pop_size: Population size.
        rng: Random generator.

    Returns:
        New position vector.
    """
    if rng.random() >= 0.5:
        rand_idx = rng.integers(pop_size)
        return op1_exploration_random(population[i], population[rand_idx], rng)
    return op2_exploration_mean(population[i], x_rabbit, x_mean, rng)


def _siege_step(
    xi: NDArray[np.float64],
    x_rabbit: NDArray[np.float64],
    e: float,
    abs_e: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Execute siege operator (Op3 or Op4).

    Args:
        xi: Current hawk position.
        x_rabbit: Leader position.
        e: Escape energy.
        abs_e: Absolute escape energy.
        rng: Random generator.

    Returns:
        New position vector.
    """
    if abs_e >= 0.5:
        return op3_soft_siege(xi, x_rabbit, e, rng)
    return op4_hard_siege(xi, x_rabbit, e, rng)


def _levy_step(
    i: int,
    population: NDArray[np.float64],
    fitnesses: list[Fitness3],
    x_rabbit: NDArray[np.float64],
    x_mean: NDArray[np.float64],
    e: float,
    abs_e: float,
    archive_positions: list[NDArray[np.float64]],
    archive_fitnesses: list[Fitness3],
    archive_size: int,
    problem: VisaProblem,
    rng: np.random.Generator,
) -> None:
    """Execute Lévy operator (Op5 or Op6) with greedy selection (in-place).

    Args:
        i: Hawk index.
        population: Population matrix (modified in-place).
        fitnesses: Fitnesses list (modified in-place).
        x_rabbit: Leader position.
        x_mean: Mean position.
        e: Escape energy.
        abs_e: Absolute escape energy.
        archive_positions: Archive positions (modified in-place).
        archive_fitnesses: Archive fitnesses (modified in-place).
        archive_size: Max archive size.
        problem: VisaProblem instance.
        rng: Random generator.
    """
    if abs_e >= 0.5:
        y, z = op5_soft_siege_levy(population[i], x_rabbit, e, rng)
    else:
        y, z = op6_hard_siege_levy(population[i], x_rabbit, e, x_mean, rng)

    winner, candidates = _greedy_select_levy(
        population[i], fitnesses[i], y, z, problem)
    if winner is not None:
        new_pos, fit_new = winner
        population[i] = new_pos
        fitnesses[i] = fit_new
    # Offer ALL evaluated candidates to the archive (not just the winner).
    for cand_pos, cand_fit in candidates:
        update_archive(archive_positions, archive_fitnesses,
                       cand_pos, cand_fit, archive_size)


def _accept_and_archive(
    i: int,
    new_pos: NDArray[np.float64],
    population: NDArray[np.float64],
    fitnesses: list[Fitness3],
    archive_positions: list[NDArray[np.float64]],
    archive_fitnesses: list[Fitness3],
    archive_size: int,
    problem: VisaProblem,
) -> None:
    """Evaluate, accept if dominating, and update archive (in-place).

    Args:
        i: Hawk index.
        new_pos: Candidate position.
        population: Population matrix (modified in-place).
        fitnesses: Fitnesses list (modified in-place).
        archive_positions: Archive positions (modified in-place).
        archive_fitnesses: Archive fitnesses (modified in-place).
        archive_size: Max archive size.
        problem: VisaProblem instance.
    """
    _, fit_new = evaluate_hawk(new_pos, problem)
    if dominates(fit_new, fitnesses[i]):
        population[i] = new_pos
        fitnesses[i] = fit_new
    update_archive(archive_positions, archive_fitnesses,
                   new_pos, fit_new, archive_size)


def run_mohho(
    problem: VisaProblem,
    seed: int,
    pop_size: int = POPULATION_SIZE,
    max_iter: int = MAX_ITERATIONS,
    archive_size: int = ARCHIVE_SIZE,
    callback: Callable[[int, list[Fitness3]], None] | None = None,
) -> tuple[list[NDArray[np.float64]], list[Fitness3], list[float]]:
    """Run the MOHHO algorithm.

    Args:
        problem: VisaProblem instance.
        seed: Random seed for reproducibility.
        pop_size: Population size (number of hawks).
        max_iter: Maximum iterations.
        archive_size: Maximum Pareto archive size.
        callback: Optional function called each iteration with (t, archive_fitnesses).

    Returns:
        Tuple of (archive_positions, archive_fitnesses, hv_history).
    """
    rng = np.random.default_rng(seed)
    dim = NUM_GROUPS

    population = rng.uniform(LB, UB, size=(pop_size, dim))
    fitnesses: list[Fitness3] = []
    for i in range(pop_size):
        _, fit = evaluate_hawk(population[i], problem)
        fitnesses.append(fit)

    archive_positions: list[NDArray[np.float64]] = []
    archive_fitnesses: list[Fitness3] = []
    for i in range(pop_size):
        update_archive(archive_positions, archive_fitnesses,
                       population[i], fitnesses[i], archive_size)

    hv_history: list[float] = []

    for t in range(max_iter):
        x_mean = np.mean(population, axis=0)
        for i in range(pop_size):
            _step_hawk(
                i, population, fitnesses,
                archive_positions, archive_fitnesses,
                x_mean, t, max_iter, pop_size, archive_size,
                problem, rng)

        hv = compute_hypervolume(archive_fitnesses)
        hv_history.append(hv)

        if callback:
            callback(t, archive_fitnesses)

    return archive_positions, archive_fitnesses, hv_history


def _hv_2d(
    pts: list[tuple[float, float]], ref: tuple[float, float]
) -> float:
    """2D hypervolume via sweep-line.

    Args:
        pts: List of (f_a, f_b) points.
        ref: 2D reference point.

    Returns:
        2D hypervolume value.
    """
    sorted_pts = sorted(pts, key=lambda p: p[0])
    hv = 0.0
    prev = ref[1]
    for fa, fb in sorted_pts:
        if fa < ref[0] and fb < ref[1]:
            h = prev - fb
            if h > 0:
                hv += (ref[0] - fa) * h
                prev = fb
    return hv


def compute_hypervolume(
    fitnesses: list[Fitness3],
    ref_point: tuple[float, float, float] | None = None,
) -> float:
    """Compute 3D hypervolume indicator via slicing.

    Slices the point set along the first objective (f1), and for each slice
    computes the 2D hypervolume of (f2, f3) contributions.

    Args:
        fitnesses: List of (f1, f2, f3) non-dominated points.
        ref_point: Reference point. Defaults to HV_REF_POINT (fixed).

    Returns:
        Hypervolume value.
    """
    if not fitnesses:
        return 0.0

    if ref_point is None:
        ref_point = HV_REF_POINT

    # Filter points that are within the reference point bounds
    pts = [(f1, f2, f3) for f1, f2, f3 in fitnesses
           if f1 < ref_point[0] and f2 < ref_point[1] and f3 < ref_point[2]]
    if not pts:
        return 0.0

    # Sort by f1
    pts.sort(key=lambda p: p[0])

    hv = 0.0
    for i, (f1, f2, f3) in enumerate(pts):
        # Width in f1 dimension
        f1_next = pts[i + 1][0] if i + 1 < len(pts) else ref_point[0]
        width_f1 = f1_next - f1
        # For this f1-slice, compute 2D HV of (f2, f3) of all points with f1' <= current
        slice_pts = [(p[1], p[2]) for p in pts[:i + 1]]
        hv_slice = _hv_2d(slice_pts, (ref_point[1], ref_point[2]))
        hv += width_f1 * hv_slice

    return hv
