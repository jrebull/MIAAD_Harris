"""Multi-Objective Harris Hawks Optimization (MOHHO).

Combines HHO operators with:
    1. External Pareto archive (max size configurable)
    2. Pareto dominance comparison
    3. Leader selection via crowding distance (roulette weighted)
    4. Archive update after each evaluation

References:
    - Heidari et al. (2019) — base HHO
    - Kusoglu & Yüzgeç (2020) — multi-objective HHO
    - HarrisFase02.tex, Section 8
"""

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
    clip_bounds,
)
from src.problem import VisaProblem


def dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Check if solution a dominates solution b.

    a dominates b iff: for all m, f_m(a) <= f_m(b) AND exists m: f_m(a) < f_m(b).

    Args:
        a: Fitness tuple (f1, f2) of solution a.
        b: Fitness tuple (f1, f2) of solution b.

    Returns:
        True if a dominates b.
    """
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


def crowding_distance(fitnesses: list[tuple[float, float]]) -> list[float]:
    """Compute crowding distance for a set of fitness values.

    Args:
        fitnesses: List of (f1, f2) tuples.

    Returns:
        List of crowding distances, same order as input.
    """
    n = len(fitnesses)
    if n <= 2:
        return [float("inf")] * n

    distances = [0.0] * n

    for m in range(2):
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
    archive_fitnesses: list[tuple[float, float]],
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
    cd_finite = [d if d != float("inf") else 1e12 for d in cd]
    total = sum(cd_finite)
    if total == 0:
        return archive_positions[rng.integers(len(archive_positions))]
    probs = np.array(cd_finite) / total
    idx = rng.choice(len(archive_positions), p=probs)
    return archive_positions[idx]


def update_archive(
    archive_positions: list[NDArray[np.float64]],
    archive_fitnesses: list[tuple[float, float]],
    new_pos: NDArray[np.float64],
    new_fit: tuple[float, float],
    max_size: int,
) -> None:
    """Update the Pareto archive with a new solution (in-place).

    Inserts new solution if non-dominated. Removes solutions it dominates.
    If archive exceeds max_size, removes the most crowded solution.

    Args:
        archive_positions: List of position vectors (modified in-place).
        archive_fitnesses: List of fitness tuples (modified in-place).
        new_pos: New candidate position.
        new_fit: New candidate fitness.
        max_size: Maximum archive size.
    """
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
        min_cd_idx = min(
            (i for i in range(len(cd)) if cd[i] != float("inf")),
            key=lambda i: cd[i],
            default=0,
        )
        archive_positions.pop(min_cd_idx)
        archive_fitnesses.pop(min_cd_idx)


def evaluate_hawk(
    hawk: NDArray[np.float64], problem: VisaProblem
) -> tuple[dict[int, int], tuple[float, float]]:
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


def run_mohho(
    problem: VisaProblem,
    seed: int,
    pop_size: int = POPULATION_SIZE,
    max_iter: int = MAX_ITERATIONS,
    archive_size: int = ARCHIVE_SIZE,
    callback=None,
) -> tuple[list[NDArray[np.float64]], list[tuple[float, float]], list[float]]:
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
    fitnesses = []
    for i in range(pop_size):
        _, fit = evaluate_hawk(population[i], problem)
        fitnesses.append(fit)

    archive_positions: list[NDArray[np.float64]] = []
    archive_fitnesses: list[tuple[float, float]] = []
    for i in range(pop_size):
        update_archive(archive_positions, archive_fitnesses,
                       population[i], fitnesses[i], archive_size)

    hv_history: list[float] = []

    for t in range(max_iter):
        x_mean = np.mean(population, axis=0)

        for i in range(pop_size):
            e = escape_energy(t, max_iter, rng)
            abs_e = abs(e)

            x_rabbit = select_leader(archive_positions, archive_fitnesses, rng)

            if abs_e >= 1:
                q = rng.random()
                if q >= 0.5:
                    rand_idx = rng.integers(pop_size)
                    new_pos = op1_exploration_random(population[i], population[rand_idx], rng)
                else:
                    new_pos = op2_exploration_mean(population[i], x_rabbit, x_mean, rng)
            else:
                r = rng.random()
                if r >= 0.5:
                    if abs_e >= 0.5:
                        new_pos = op3_soft_siege(population[i], x_rabbit, e, rng)
                    else:
                        new_pos = op4_hard_siege(population[i], x_rabbit, e, rng)
                else:
                    if abs_e >= 0.5:
                        y, z = op5_soft_siege_levy(population[i], x_rabbit, e, rng)
                        _, fit_y = evaluate_hawk(y, problem)
                        if dominates(fit_y, fitnesses[i]) or (not dominates(fitnesses[i], fit_y)):
                            new_pos = y
                            _, new_fit = fit_y, fit_y
                        else:
                            _, fit_z = evaluate_hawk(z, problem)
                            if dominates(fit_z, fitnesses[i]) or (not dominates(fitnesses[i], fit_z)):
                                new_pos = z
                            else:
                                continue
                        _, fit_new = evaluate_hawk(new_pos, problem)
                        fitnesses[i] = fit_new
                        population[i] = new_pos
                        update_archive(archive_positions, archive_fitnesses,
                                       new_pos, fit_new, archive_size)
                        continue
                    else:
                        y, z = op6_hard_siege_levy(population[i], x_rabbit, e, x_mean, rng)
                        _, fit_y = evaluate_hawk(y, problem)
                        if dominates(fit_y, fitnesses[i]) or (not dominates(fitnesses[i], fit_y)):
                            new_pos = y
                        else:
                            _, fit_z = evaluate_hawk(z, problem)
                            if dominates(fit_z, fitnesses[i]) or (not dominates(fitnesses[i], fit_z)):
                                new_pos = z
                            else:
                                continue
                        _, fit_new = evaluate_hawk(new_pos, problem)
                        fitnesses[i] = fit_new
                        population[i] = new_pos
                        update_archive(archive_positions, archive_fitnesses,
                                       new_pos, fit_new, archive_size)
                        continue

            _, fit_new = evaluate_hawk(new_pos, problem)
            if dominates(fit_new, fitnesses[i]) or (not dominates(fitnesses[i], fit_new)):
                population[i] = new_pos
                fitnesses[i] = fit_new

            update_archive(archive_positions, archive_fitnesses,
                           new_pos, fit_new, archive_size)

        hv = compute_hypervolume(archive_fitnesses)
        hv_history.append(hv)

        if callback:
            callback(t, archive_fitnesses)

    return archive_positions, archive_fitnesses, hv_history


def compute_hypervolume(
    fitnesses: list[tuple[float, float]],
    ref_point: tuple[float, float] | None = None,
) -> float:
    """Compute 2D hypervolume indicator.

    Uses a simple sweep-line algorithm for 2 objectives.

    Args:
        fitnesses: List of (f1, f2) non-dominated points.
        ref_point: Reference point. If None, uses (max_f1*1.1, max_f2*1.1).

    Returns:
        Hypervolume value.
    """
    if not fitnesses:
        return 0.0

    if ref_point is None:
        max_f1 = max(f[0] for f in fitnesses)
        max_f2 = max(f[1] for f in fitnesses)
        ref_point = (max_f1 * 1.1 + 0.1, max_f2 * 1.1 + 0.1)

    sorted_pts = sorted(fitnesses, key=lambda p: p[0])

    hv = 0.0
    prev_f2 = ref_point[1]

    for f1, f2 in sorted_pts:
        if f1 < ref_point[0] and f2 < ref_point[1]:
            width = ref_point[0] - f1
            height = prev_f2 - f2
            if height > 0:
                hv += width * height
                prev_f2 = f2

    return hv
