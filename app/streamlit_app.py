"""Streamlit dashboard for MOHHO Green Card Optimization results.

Features:
    1. Interactive Pareto front scatter (with FIFO baseline point)
    2. Assignment heatmap for selected solution
    3. Comparison with baseline (bar chart)
    4. Convergence plot (hypervolume per iteration)
    5. Run selector dropdown
    6. UACJ color palette
"""

import json
import os
import csv

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# --- UACJ Colors ---
UACJ_BLUE = "#003CA6"
UACJ_YELLOW = "#FFD600"
UACJ_GRAY = "#555559"


@st.cache_data
def load_summary():
    """Load experiment summary."""
    path = os.path.join(RESULTS_DIR, "summary.json")
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_pareto_csv():
    """Load combined Pareto front and baseline from CSV."""
    path = os.path.join(RESULTS_DIR, "pareto_front.csv")
    pareto = []
    baseline = None
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            point = (float(row["f1"]), float(row["f2"]))
            if row["type"] == "baseline":
                baseline = point
            else:
                pareto.append(point)
    return pareto, baseline


@st.cache_data
def load_convergence():
    """Load convergence data."""
    path = os.path.join(RESULTS_DIR, "convergence.csv")
    iters, means, stds = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            iters.append(int(row["iteration"]))
            means.append(float(row["hv_mean"]))
            stds.append(float(row["hv_std"]))
    return iters, means, stds


@st.cache_data
def load_run(run_idx):
    """Load a specific run's results."""
    path = os.path.join(RESULTS_DIR, f"run_{run_idx:02d}.json")
    with open(path) as f:
        return json.load(f)


def get_allocation_for_point(target_f1, target_f2):
    """Find an allocation close to the target point via random search."""
    from src.problem import VisaProblem
    from src.decoder import spv, decode
    from src.config import COUNTRIES, CATEGORIES

    problem = VisaProblem()
    rng = np.random.default_rng(42)
    best_alloc = None
    best_dist = float("inf")

    for _ in range(20000):
        hawk = rng.uniform(0, 1, size=50)
        perm = spv(hawk)
        alloc = decode(perm, problem.groups, problem.total_visas,
                       problem.country_caps, problem.category_caps)
        fit = problem.evaluate(alloc)
        d = (fit[0] - target_f1) ** 2 + (fit[1] - target_f2) ** 2
        if d < best_dist:
            best_dist = d
            best_alloc = alloc
            best_fit = fit

    matrix = np.zeros((len(COUNTRIES), len(CATEGORIES)))
    for g in problem.groups:
        ci = COUNTRIES.index(g["country"])
        ji = CATEGORIES.index(g["category"])
        matrix[ci][ji] = best_alloc[g["index"]]

    return matrix, best_fit


def main():
    st.set_page_config(
        page_title="Visa Predict AI — MOHHO",
        page_icon="🦅",
        layout="wide",
    )

    st.markdown(
        f"""<h1 style='color: {UACJ_BLUE};'>Visa Predict AI — MOHHO Dashboard</h1>
        <p style='color: {UACJ_GRAY};'>Optimización Multiobjetivo de Green Cards con Harris Hawks Optimization</p>
        <hr style='border-top: 3px solid {UACJ_YELLOW};'>""",
        unsafe_allow_html=True,
    )

    summary = load_summary()
    pareto, baseline = load_pareto_csv()
    iters, hv_means, hv_stds = load_convergence()

    # --- Sidebar ---
    st.sidebar.markdown(f"### Configuración")
    st.sidebar.markdown(f"**Corridas:** {summary['num_runs']}")
    st.sidebar.markdown(f"**Población:** {summary['pop_size']}")
    st.sidebar.markdown(f"**Iteraciones:** {summary['max_iter']}")
    st.sidebar.markdown(f"**Archivo Pareto:** {summary['archive_size']}")
    st.sidebar.markdown(f"---")
    st.sidebar.markdown(f"**HV medio:** {summary['hv_stats']['mean']:.4f} ± {summary['hv_stats']['std']:.4f}")
    st.sidebar.markdown(f"**Baseline FIFO:** f₁={baseline[0]:.4f}, f₂={baseline[1]:.4f}")

    run_idx = st.sidebar.selectbox(
        "Seleccionar corrida",
        range(summary["num_runs"]),
        format_func=lambda x: f"Corrida {x + 1} (seed={42 + x})",
    )

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Frente de Pareto", "Heatmap", "Comparación", "Convergencia", "Corrida individual"
    ])

    with tab1:
        st.subheader("Frente de Pareto combinado (30 corridas)")
        fig, ax = plt.subplots(figsize=(9, 6))
        f1s = [p[0] for p in pareto]
        f2s = [p[1] for p in pareto]
        ax.scatter(f1s, f2s, c=UACJ_BLUE, s=35, alpha=0.7, label="MOHHO Pareto")
        ax.scatter(baseline[0], baseline[1], c="red", s=200, marker="*",
                   zorder=5, label=f"FIFO ({baseline[0]:.2f}, {baseline[1]:.2f})")
        ax.set_xlabel("f₁ — Carga de espera no atendida (años)", fontsize=11)
        ax.set_ylabel("f₂ — Máxima disparidad (años)", fontsize=11)
        ax.set_title("Frente de Pareto: MOHHO vs Sistema FIFO", fontsize=13, color=UACJ_BLUE)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**Seleccionar solución para ver heatmap:**")
        sol_idx = st.slider("Índice de solución en el frente", 0, len(pareto) - 1, len(pareto) // 2)
        selected = pareto[sol_idx]
        st.markdown(f"Solución seleccionada: **f₁={selected[0]:.4f}, f₂={selected[1]:.4f}**")

    with tab2:
        st.subheader("Heatmap de asignación de visas")
        from src.config import COUNTRIES, CATEGORIES
        matrix, actual_fit = get_allocation_for_point(selected[0], selected[1])

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
                ax.text(j, i, f"{val:,}", ha="center", va="center", fontsize=9, color=color)
        ax.set_title(f"Asignación — f₁={actual_fit[0]:.2f}, f₂={actual_fit[1]:.2f}",
                     fontsize=13, color=UACJ_BLUE)
        plt.colorbar(im, ax=ax, label="Visas asignadas")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(f"**Total visas asignadas:** {int(matrix.sum()):,}")

    with tab3:
        st.subheader("Comparación: MOHHO vs Sistema FIFO")
        best_f1 = min(pareto, key=lambda x: x[0])
        best_f2 = min(pareto, key=lambda x: x[1])
        f1_range = max(f1s) - min(f1s) if len(f1s) > 1 else 1
        f2_range = max(f2s) - min(f2s) if len(f2s) > 1 else 1
        dists = [((p[0] - min(f1s)) / max(f1_range, 1e-9)) ** 2 +
                 ((p[1] - min(f2s)) / max(f2_range, 1e-9)) ** 2 for p in pareto]
        eq_sol = pareto[int(np.argmin(dists))]

        labels = ["Mejor f₁", "Mejor f₂", "Equilibrio", "FIFO"]
        f1_vals = [best_f1[0], best_f2[0], eq_sol[0], baseline[0]]
        f2_vals = [best_f1[1], best_f2[1], eq_sol[1], baseline[1]]

        x_pos = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x_pos - width / 2, f1_vals, width, label="f₁", color=UACJ_BLUE, alpha=0.8)
        ax.bar(x_pos + width / 2, f2_vals, width, label="f₂", color=UACJ_YELLOW, alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Valor objetivo (años)", fontsize=11)
        ax.set_title("Comparación MOHHO vs FIFO", fontsize=13, color=UACJ_BLUE)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mejor f₁", f"{best_f1[0]:.4f}",
                       delta=f"{best_f1[0] - baseline[0]:.4f} vs FIFO")
        with col2:
            st.metric("Mejor f₂", f"{best_f2[1]:.4f}",
                       delta=f"{best_f2[1] - baseline[1]:.4f} vs FIFO")
        with col3:
            st.metric("Equilibrio f₁", f"{eq_sol[0]:.4f}",
                       delta=f"{eq_sol[0] - baseline[0]:.4f} vs FIFO")

    with tab4:
        st.subheader("Convergencia del hipervolumen")
        fig, ax = plt.subplots(figsize=(9, 5))
        means_arr = np.array(hv_means)
        stds_arr = np.array(hv_stds)
        iters_arr = np.array(iters)
        ax.plot(iters_arr, means_arr, color=UACJ_BLUE, linewidth=1.5, label="HV medio")
        ax.fill_between(iters_arr, means_arr - stds_arr, means_arr + stds_arr,
                         color=UACJ_BLUE, alpha=0.15, label="± 1 σ")
        ax.set_xlabel("Iteración", fontsize=11)
        ax.set_ylabel("Hipervolumen", fontsize=11)
        ax.set_title("Convergencia (30 corridas)", fontsize=13, color=UACJ_BLUE)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab5:
        st.subheader(f"Corrida {run_idx + 1}")
        run_data = load_run(run_idx)
        st.markdown(f"**Seed:** {run_data['seed']} | **Soluciones Pareto:** {run_data['num_pareto']} | **HV final:** {run_data['hv_final']:.4f}")

        fig, ax = plt.subplots(figsize=(8, 5))
        rf1 = [p["f1"] for p in run_data["pareto_front"]]
        rf2 = [p["f2"] for p in run_data["pareto_front"]]
        ax.scatter(rf1, rf2, c=UACJ_BLUE, s=25, alpha=0.7)
        ax.scatter(baseline[0], baseline[1], c="red", s=150, marker="*", zorder=5)
        ax.set_xlabel("f₁", fontsize=11)
        ax.set_ylabel("f₂", fontsize=11)
        ax.set_title(f"Frente de Pareto — Corrida {run_idx + 1}", fontsize=13, color=UACJ_BLUE)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
