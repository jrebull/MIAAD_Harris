"""Baseline FIFO simulation of the current visa allocation system.

Simulates strict FIFO processing: groups with oldest priority dates
are processed first within each category, respecting all caps.
This gives ONE reference point (f1_fifo, f2_fifo) for comparison.

References:
    - HarrisFase02.tex, Section 7 (comparison with current system)
    - Prompt Section 4.3
"""

from src.decoder import decode
from src.problem import VisaProblem


def fifo_permutation(groups: list[dict]) -> list[int]:
    """Generate FIFO permutation: oldest priority date first.

    Ties broken by group index (stable sort).

    Args:
        groups: List of group dicts.

    Returns:
        Permutation as list of group indices.
    """
    sorted_groups = sorted(groups, key=lambda g: (g["d"], g["index"]))
    return [g["index"] for g in sorted_groups]


def run_baseline(problem: VisaProblem) -> tuple[dict[int, int], tuple[float, float]]:
    """Run the FIFO baseline and return allocation + fitness.

    Args:
        problem: VisaProblem instance.

    Returns:
        Tuple of (allocation dict, fitness tuple (f1, f2)).
    """
    perm = fifo_permutation(problem.groups)
    alloc = decode(perm, problem.groups, problem.total_visas,
                   problem.country_caps, problem.category_caps)
    fitness = problem.evaluate(alloc)
    return alloc, fitness
