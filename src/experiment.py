"""Experiment runner: executes NUM_RUNS independent MOHHO runs and saves results.

Outputs:
    results/pareto_front.csv: Combined non-dominated front.
    results/convergence.csv: HV per iteration, averaged over runs.
    results/run_XX.json: Per-run details.
    results/figures/*.png: Publication-quality plots.
"""

import json
import logging
import os
import csv

import numpy as np

from src.config import (
    NUM_RUNS, SEED_BASE, POPULATION_SIZE, MAX_ITERATIONS,
    ARCHIVE_SIZE, COUNTRIES, CATEGORIES,
)
from src.problem import VisaProblem
from src.mohho import run_mohho, dominates, compute_hypervolume
from src.baseline import run_baseline
from src.decoder import decode

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
LATEX_FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "references", "Fase03_Latex", "Figures"
)


def _setup_logging() -> None:
    """Configure JSON-structured logging to stderr."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '{"time":"%(asctime)s","level":"%(levelname)s",'
        '"module":"%(name)s","message":"%(message)s"}'
    ))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(LATEX_FIGURES_DIR, exist_ok=True)


def run_all_experiments() -> None:
    """Execute all experiment runs and save results."""
    _setup_logging()
    ensure_dirs()
    problem = VisaProblem()

    baseline_alloc, baseline_fit = run_baseline(problem)
    logger.info("Baseline FIFO: f1=%.4f, f2=%.4f", baseline_fit[0], baseline_fit[1])

    all_fronts: list[list[tuple[float, float]]] = []
    all_hv_histories: list[list[float]] = []

    for run_idx in range(NUM_RUNS):
        seed = SEED_BASE + run_idx
        logger.info("Run %d/%d (seed=%d) starting", run_idx + 1, NUM_RUNS, seed)

        positions, fitnesses, hv_history = run_mohho(
            problem, seed=seed,
            pop_size=POPULATION_SIZE,
            max_iter=MAX_ITERATIONS,
            archive_size=ARCHIVE_SIZE,
        )

        all_fronts.append(fitnesses)
        all_hv_histories.append(hv_history)

        run_data = {
            "run": run_idx,
            "seed": seed,
            "num_pareto": len(fitnesses),
            "hv_final": hv_history[-1] if hv_history else 0.0,
            "pareto_front": [{"f1": f[0], "f2": f[1]} for f in fitnesses],
        }
        run_path = os.path.join(RESULTS_DIR, f"run_{run_idx:02d}.json")
        with open(run_path, "w") as f:
            json.dump(run_data, f, indent=2)

        logger.info(
            "Run %d/%d done: pareto_size=%d, hv=%.4f",
            run_idx + 1, NUM_RUNS, len(fitnesses),
            hv_history[-1] if hv_history else 0.0,
        )

    combined_front = _merge_fronts(all_fronts)
    _save_pareto_csv(combined_front, baseline_fit)
    _save_convergence_csv(all_hv_histories)
    _generate_figures(combined_front, all_hv_histories, baseline_fit, problem)

    summary = {
        "num_runs": NUM_RUNS,
        "pop_size": POPULATION_SIZE,
        "max_iter": MAX_ITERATIONS,
        "archive_size": ARCHIVE_SIZE,
        "baseline": {"f1": baseline_fit[0], "f2": baseline_fit[1]},
        "combined_pareto_size": len(combined_front),
        "hv_stats": {
            "mean": float(np.mean([h[-1] for h in all_hv_histories])),
            "std": float(np.std([h[-1] for h in all_hv_histories])),
            "min": float(np.min([h[-1] for h in all_hv_histories])),
            "max": float(np.max([h[-1] for h in all_hv_histories])),
        },
        "best_f1": min(combined_front, key=lambda x: x[0]),
        "best_f2": min(combined_front, key=lambda x: x[1]),
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Done. Combined Pareto: %d solutions, HV=%.4f ± %.4f",
        len(combined_front),
        summary["hv_stats"]["mean"],
        summary["hv_stats"]["std"],
    )


def _merge_fronts(
    all_fronts: list[list[tuple[float, float]]],
) -> list[tuple[float, float]]:
    """Merge all run fronts into a single non-dominated front.

    Args:
        all_fronts: List of per-run Pareto fronts.

    Returns:
        Sorted list of non-dominated (f1, f2) tuples.
    """
    combined: list[tuple[float, float]] = []
    for front in all_fronts:
        combined.extend(front)

    non_dominated: list[tuple[float, float]] = []
    for i, fi in enumerate(combined):
        if not any(dominates(fj, fi) for j, fj in enumerate(combined) if i != j):
            non_dominated.append(fi)

    unique = sorted(set(non_dominated), key=lambda x: x[0])
    return unique


def _save_pareto_csv(
    front: list[tuple[float, float]], baseline: tuple[float, float]
) -> None:
    """Save Pareto front and baseline to CSV.

    Args:
        front: Non-dominated front.
        baseline: FIFO baseline fitness.
    """
    path = os.path.join(RESULTS_DIR, "pareto_front.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["f1", "f2", "type"])
        for f1, f2 in front:
            writer.writerow([f"{f1:.6f}", f"{f2:.6f}", "pareto"])
        writer.writerow([f"{baseline[0]:.6f}", f"{baseline[1]:.6f}", "baseline"])


def _save_convergence_csv(all_hv: list[list[float]]) -> None:
    """Save mean/std hypervolume per iteration to CSV.

    Args:
        all_hv: List of per-run HV histories.
    """
    path = os.path.join(RESULTS_DIR, "convergence.csv")
    max_len = max(len(h) for h in all_hv)
    padded = np.array([h + [h[-1]] * (max_len - len(h)) for h in all_hv])
    mean_hv = padded.mean(axis=0)
    std_hv = padded.std(axis=0)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "hv_mean", "hv_std"])
        for i in range(max_len):
            writer.writerow([i, f"{mean_hv[i]:.6f}", f"{std_hv[i]:.6f}"])


# ---------------------------------------------------------------------------
# Figure generation — one function per figure
# ---------------------------------------------------------------------------

def _generate_figures(
    combined_front: list[tuple[float, float]],
    all_hv: list[list[float]],
    baseline: tuple[float, float],
    problem: VisaProblem,
) -> None:
    """Orchestrate generation of all 5 publication figures.

    Args:
        combined_front: Merged non-dominated front.
        all_hv: Per-run HV histories.
        baseline: FIFO baseline fitness.
        problem: VisaProblem instance.
    """
    import matplotlib
    matplotlib.use("Agg")

    f1s = [p[0] for p in combined_front]
    f2s = [p[1] for p in combined_front]

    _plot_pareto_front(combined_front, baseline, f1s, f2s)
    _plot_convergence(all_hv)
    _plot_heatmap(combined_front, problem, f1s, f2s)
    _plot_hv_boxplot(all_hv)
    _plot_comparison_baseline(combined_front, baseline, f1s, f2s)

    logger.info("Figures saved to results/figures/ and references/Fase03_Latex/Figures/")


def _save_fig(fig: "matplotlib.figure.Figure", name: str) -> None:
    """Save a figure to both output directories and close it.

    Args:
        fig: Matplotlib figure.
        name: Filename (e.g. 'pareto_front.png').
    """
    import matplotlib.pyplot as plt
    for directory in (FIGURES_DIR, LATEX_FIGURES_DIR):
        fig.savefig(os.path.join(directory, name), dpi=300)
    plt.close(fig)


def _plot_pareto_front(
    front: list[tuple[float, float]],
    baseline: tuple[float, float],
    f1s: list[float],
    f2s: list[float],
) -> None:
    """Generate Pareto front scatter plot.

    Args:
        front: Non-dominated front.
        baseline: FIFO baseline point.
        f1s: Pre-extracted f1 values.
        f2s: Pre-extracted f2 values.
    """
    import matplotlib.pyplot as plt
    uacj_blue = "#003CA6"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(f1s, f2s, c=uacj_blue, s=30, alpha=0.7, label="Frente de Pareto (MOHHO)")
    ax.scatter(baseline[0], baseline[1], c="red", s=150, marker="*",
               zorder=5, label=f"FIFO baseline ({baseline[0]:.2f}, {baseline[1]:.2f})")
    ax.set_xlabel("f₁ — Carga de espera no atendida (años)", fontsize=11)
    ax.set_ylabel("f₂ — Máxima disparidad entre países (años)", fontsize=11)
    ax.set_title("Frente de Pareto: MOHHO vs Sistema FIFO", fontsize=13, color=uacj_blue)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "pareto_front.png")


def _plot_convergence(all_hv: list[list[float]]) -> None:
    """Generate hypervolume convergence plot.

    Args:
        all_hv: Per-run HV histories.
    """
    import matplotlib.pyplot as plt
    uacj_blue = "#003CA6"

    max_len = max(len(h) for h in all_hv)
    padded = np.array([h + [h[-1]] * (max_len - len(h)) for h in all_hv])
    mean_hv = padded.mean(axis=0)
    std_hv = padded.std(axis=0)
    iters = np.arange(max_len)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, mean_hv, color=uacj_blue, linewidth=1.5, label="HV medio")
    ax.fill_between(iters, mean_hv - std_hv, mean_hv + std_hv,
                     color=uacj_blue, alpha=0.15, label="± 1 σ")
    ax.set_xlabel("Iteración", fontsize=11)
    ax.set_ylabel("Hipervolumen", fontsize=11)
    ax.set_title("Convergencia del hipervolumen (30 corridas)", fontsize=13, color=uacj_blue)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "convergence.png")


def _plot_heatmap(
    combined_front: list[tuple[float, float]],
    problem: VisaProblem,
    f1s: list[float],
    f2s: list[float],
) -> None:
    """Generate assignment heatmap for the equilibrium solution.

    Args:
        combined_front: Non-dominated front.
        problem: VisaProblem instance.
        f1s: Pre-extracted f1 values.
        f2s: Pre-extracted f2 values.
    """
    import matplotlib.pyplot as plt
    uacj_blue = "#003CA6"

    f1_range = max(f1s) - min(f1s) if len(f1s) > 1 else 1.0
    f2_range = max(f2s) - min(f2s) if len(f2s) > 1 else 1.0
    distances = [((p[0] - min(f1s)) / max(f1_range, 1e-9)) ** 2 +
                 ((p[1] - min(f2s)) / max(f2_range, 1e-9)) ** 2
                 for p in combined_front]
    target = combined_front[int(np.argmin(distances))]

    rng = np.random.default_rng(42)
    best_pos: dict[int, int] | None = None
    best_dist = float("inf")
    for _ in range(10000):
        hawk = rng.uniform(0, 1, size=50)
        perm = list(np.argsort(hawk, kind="stable"))
        alloc = decode(perm, problem.groups, problem.total_visas,
                       problem.country_caps, problem.category_caps)
        fit = problem.evaluate(alloc)
        d = (fit[0] - target[0]) ** 2 + (fit[1] - target[1]) ** 2
        if d < best_dist:
            best_dist = d
            best_pos = alloc

    matrix = np.zeros((len(COUNTRIES), len(CATEGORIES)))
    for g in problem.groups:
        ci = COUNTRIES.index(g["country"])
        ji = CATEGORIES.index(g["category"])
        matrix[ci][ji] = best_pos[g["index"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels(CATEGORIES, fontsize=10)
    ax.set_yticks(range(len(COUNTRIES)))
    ax.set_yticklabels(COUNTRIES, fontsize=10)
    for i in range(len(COUNTRIES)):
        for j in range(len(CATEGORIES)):
            val = int(matrix[i][j])
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", fontsize=8, color=color)
    ax.set_title("Asignación de visas — Solución equilibrio", fontsize=13, color=uacj_blue)
    ax.set_xlabel("Categoría", fontsize=11)
    ax.set_ylabel("País", fontsize=11)
    plt.colorbar(im, ax=ax, label="Visas asignadas")
    fig.tight_layout()
    _save_fig(fig, "assignment_heatmap.png")


def _plot_hv_boxplot(all_hv: list[list[float]]) -> None:
    """Generate boxplot of final HV values across runs.

    Args:
        all_hv: Per-run HV histories.
    """
    import matplotlib.pyplot as plt
    uacj_blue, uacj_yellow = "#003CA6", "#FFD600"

    final_hvs = [h[-1] for h in all_hv]
    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(final_hvs, patch_artist=True)
    bp["boxes"][0].set_facecolor(uacj_blue)
    bp["boxes"][0].set_alpha(0.6)
    bp["medians"][0].set_color(uacj_yellow)
    bp["medians"][0].set_linewidth(2)
    ax.set_ylabel("Hipervolumen final", fontsize=11)
    ax.set_title("Distribución de HV (30 corridas)", fontsize=13, color=uacj_blue)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save_fig(fig, "hv_boxplot.png")


def _plot_comparison_baseline(
    combined_front: list[tuple[float, float]],
    baseline: tuple[float, float],
    f1s: list[float],
    f2s: list[float],
) -> None:
    """Generate bar chart comparing MOHHO solutions vs FIFO baseline.

    Args:
        combined_front: Non-dominated front.
        baseline: FIFO baseline fitness.
        f1s: Pre-extracted f1 values.
        f2s: Pre-extracted f2 values.
    """
    import matplotlib.pyplot as plt
    uacj_blue, uacj_yellow = "#003CA6", "#FFD600"

    best_f1_sol = min(combined_front, key=lambda x: x[0])
    best_f2_sol = min(combined_front, key=lambda x: x[1])

    f1_range = max(f1s) - min(f1s) if len(f1s) > 1 else 1.0
    f2_range = max(f2s) - min(f2s) if len(f2s) > 1 else 1.0
    distances = [((p[0] - min(f1s)) / max(f1_range, 1e-9)) ** 2 +
                 ((p[1] - min(f2s)) / max(f2_range, 1e-9)) ** 2
                 for p in combined_front]
    eq_sol = combined_front[int(np.argmin(distances))]

    labels = ["Mejor f₁", "Mejor f₂", "Equilibrio", "FIFO"]
    f1_vals = [best_f1_sol[0], best_f2_sol[0], eq_sol[0], baseline[0]]
    f2_vals = [best_f1_sol[1], best_f2_sol[1], eq_sol[1], baseline[1]]

    x_pos = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_pos - width / 2, f1_vals, width, label="f₁", color=uacj_blue, alpha=0.8)
    ax.bar(x_pos + width / 2, f2_vals, width, label="f₂", color=uacj_yellow, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Valor objetivo (años)", fontsize=11)
    ax.set_title("Comparación: MOHHO vs FIFO", fontsize=13, color=uacj_blue)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save_fig(fig, "comparison_baseline.png")


if __name__ == "__main__":
    run_all_experiments()
