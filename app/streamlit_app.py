"""MOHHO Green Card Optimization — Dashboard Profesional v2.

21 country blocs x 5 EB categories = 105 groups.
7 tabs: Resumen, Datos, Asignacion, Objetivos, Paises, Convergencia, Glosario.

Sources:
    USCIS FY2025 Q2 Approved Petitions
    Visa Bulletin April 2026 (Final Action Dates)
"""

import json
import os
import csv
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# High-resolution charts + dot thousands separator on axes
matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["savefig.dpi"] = 200
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["axes.titlesize"] = 13
matplotlib.rcParams["axes.labelsize"] = 11


def _dot_fmt(x: float, _pos: object = None) -> str:
    """Format axis tick with dot thousands separator."""
    if x == int(x):
        return f"{int(x):,}".replace(",", ".")
    return f"{x:,.1f}".replace(",", ".")


_DOT_FORMATTER = mticker.FuncFormatter(_dot_fmt)

# --- Paths ---
BASE_DIR: str = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR: str = os.path.join(BASE_DIR, "results")

# --- Color palette ---
BLUE: str = "#003CA6"
YELLOW: str = "#FFD600"
GRAY: str = "#555559"
DARK: str = "#1a1a2e"
LIGHT_BLUE: str = "#e8f0fe"
GREEN: str = "#00873c"
RED: str = "#c62828"
ORANGE: str = "#e65100"
TEAL: str = "#00796b"
PURPLE: str = "#6a1b9a"

# Category labels with short description for charts
CAT_LABELS: dict[str, str] = {
    "EB-1": "EB-1\nExtraordinarios",
    "EB-2": "EB-2\nProfesionales",
    "EB-3": "EB-3\nCalificados",
    "EB-4": "EB-4\nEspeciales/SIV",
    "EB-5": "EB-5\nInversores",
}

CAT_LABELS_SHORT: dict[str, str] = {
    "EB-1": "EB-1 Extrao.",
    "EB-2": "EB-2 Profes.",
    "EB-3": "EB-3 Calific.",
    "EB-4": "EB-4 Espec.",
    "EB-5": "EB-5 Invers.",
}

CAT_LABELS_HTML: dict[str, str] = {
    "EB-1": "EB-1<br><span style='font-size:0.6rem;font-weight:400;opacity:0.8;'>Extraordinarios</span>",
    "EB-2": "EB-2<br><span style='font-size:0.6rem;font-weight:400;opacity:0.8;'>Profesionales</span>",
    "EB-3": "EB-3<br><span style='font-size:0.6rem;font-weight:400;opacity:0.8;'>Calificados</span>",
    "EB-4": "EB-4<br><span style='font-size:0.6rem;font-weight:400;opacity:0.8;'>Especiales/SIV</span>",
    "EB-5": "EB-5<br><span style='font-size:0.6rem;font-weight:400;opacity:0.8;'>Inversores</span>",
}

COUNTRY_COLORS: list[str] = [
    "#003CA6", "#d32f2f", "#2e7d32", "#e65100", "#6a1b9a",
    "#00838f", "#4e342e", "#37474f", "#ad1457", "#33691e",
    "#e64a19", "#283593", "#00695c", "#f9a825", "#4527a0",
    "#558b2f", "#b71c1c", "#0277bd", "#9e9d24", "#ef6c00",
    "#455a64",
]

CAT_COLORS: list[str] = ["#003CA6", "#1565c0", "#1976d2", "#42a5f5", "#90caf9"]

# Source type for each (country, category) pair
_OFICIAL_SET: set[tuple[str, str]] = {
    ("India", "EB-1"), ("India", "EB-2"), ("India", "EB-3"),
    ("India", "EB-4"), ("India", "EB-5"),
    ("China", "EB-1"), ("China", "EB-2"), ("China", "EB-3"),
    ("China", "EB-4"), ("China", "EB-5"),
    ("Filipinas", "EB-2"), ("Filipinas", "EB-3"), ("Filipinas", "EB-4"),
    ("Mexico", "EB-2"), ("Mexico", "EB-3"), ("Mexico", "EB-4"),
}


def _source_type(country: str, category: str) -> str:
    """Return 'OFICIAL', 'EST', or 'EST-DEM' for a (country, category) pair."""
    if (country, category) in _OFICIAL_SET:
        return "OFICIAL"
    if country not in ("India", "China") and category in ("EB-1", "EB-5"):
        return "EST-DEM"
    return "EST"


def _source_badge(src: str) -> str:
    """HTML badge for data source."""
    if src == "OFICIAL":
        return '<span style="background:#d1fae5;color:#065f46;font-size:0.65rem;padding:2px 6px;border-radius:4px;font-weight:700;">OFICIAL</span>'
    if src == "EST-DEM":
        return '<span style="background:#dbeafe;color:#1e40af;font-size:0.65rem;padding:2px 6px;border-radius:4px;font-weight:700;">EST-DEM</span>'
    return '<span style="background:#fef3c7;color:#92400e;font-size:0.65rem;padding:2px 6px;border-radius:4px;font-weight:700;">EST</span>'


# ──────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────

def _inject_css() -> None:
    """Inject custom CSS for a professional look."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* -- Force light mode -- */
    html, body, .main, [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"], [data-testid="stHeader"],
    .stApp, section[data-testid="stSidebar"] > div {
        background-color: #f8f9fb !important;
        color: #1a1a2e !important;
    }
    [data-testid="stSidebar"] { background-color: #eef1f6 !important; }
    [data-testid="stSidebar"] * { color: #1a1a2e !important; }
    .main .block-container { max-width: 1200px; padding-top: 1rem; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #1a1a2e; }
    p, li, span, div, label { color: #1a1a2e !important; }
    h1, h2, h3, h4, h5, h6 { color: #1a1a2e !important; }

    /* -- Hero -- */
    .hero {
        background: linear-gradient(135deg, #003CA6 0%, #0052d4 50%, #003080 100%);
        padding: 2.5rem 2rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(0,60,166,0.3);
    }
    .hero::before {
        content: '';
        position: absolute; top: 0; right: 0;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(255,214,0,0.18) 0%, transparent 70%);
        border-radius: 50%;
        transform: translate(30%, -30%);
    }
    .hero h1 { color: white !important; font-size: 2.2rem; font-weight: 800;
        margin: 0 0 0.3rem; letter-spacing: -0.5px; }
    .hero .accent { color: #FFD600 !important; }
    .hero p { color: rgba(255,255,255,0.8) !important; font-size: 1rem; margin: 0; }
    .hero .subtitle { color: rgba(255,255,255,0.6) !important; font-size: 0.85rem; margin-top: 4px; }

    /* -- KPI cards -- */
    .kpi-row { display: flex; gap: 12px; margin-bottom: 1.5rem; flex-wrap: wrap; }
    .kpi-card {
        flex: 1; min-width: 140px;
        background: #ffffff;
        border: 1px solid #dfe3ea;
        border-radius: 12px;
        padding: 1.1rem 0.9rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 4px 14px rgba(0,0,0,0.08); }
    .kpi-label { font-size: 0.68rem; color: #5a6170 !important; text-transform: uppercase;
        letter-spacing: 0.5px; font-weight: 600; }
    .kpi-value { font-size: 1.5rem; font-weight: 800; color: #1a1a2e !important; margin: 4px 0 2px; }
    .kpi-delta { font-size: 0.75rem; font-weight: 600; }
    .kpi-delta.good { color: #00873c !important; }
    .kpi-delta.bad { color: #c62828 !important; }
    .kpi-delta.neutral { color: #5a6170 !important; }

    /* -- Section headers -- */
    .section-head {
        font-size: 1.1rem; font-weight: 700; color: #003CA6 !important;
        border-left: 4px solid #FFD600; padding-left: 12px;
        margin: 1.5rem 0 1rem;
    }

    /* -- Info callout -- */
    .info-box {
        background: #eef1f6; border-left: 4px solid #003CA6;
        padding: 12px 16px; border-radius: 0 8px 8px 0;
        margin: 0.8rem 0; font-size: 0.84rem; color: #2d3142 !important;
    }
    .info-box strong { color: #003CA6 !important; }

    /* -- Warning box -- */
    .warn-box {
        background: #fff7ed; border-left: 4px solid #f97316;
        padding: 12px 16px; border-radius: 0 8px 8px 0;
        margin: 0.8rem 0; font-size: 0.84rem; color: #2d3142 !important;
    }

    /* -- Badges -- */
    .badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.3px;
    }
    .badge-fifo { background: #fde8e8; color: #991b1b !important; }
    .badge-mohho { background: #d1fae5; color: #065f46 !important; }

    /* -- Tables -- */
    .styled-table {
        width: 100%; border-collapse: collapse; font-size: 0.78rem;
        border-radius: 8px; overflow: hidden;
        background: #ffffff;
    }
    .styled-table thead th {
        background: #2d3a52; color: white !important; padding: 9px 6px;
        font-weight: 600; text-align: center; font-size: 0.72rem;
    }
    .styled-table tbody td {
        padding: 7px 5px; text-align: center; border-bottom: 1px solid #eef1f6;
        color: #1a1a2e !important; font-size: 0.76rem;
    }
    .styled-table tbody tr:nth-child(even) { background: #f4f6f9; }
    .styled-table tbody tr:hover { background: #e3e9f5; }
    .cell-high { background: #fef3c7 !important; font-weight: 700; color: #92400e !important; }
    .cell-zero { color: #b0b8c4 !important; }
    .cell-improve { color: #065f46 !important; font-weight: 600; }
    .cell-worse { color: #991b1b !important; font-weight: 600; }

    /* -- Tabs -- */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0; font-weight: 600; font-size: 0.82rem;
        padding: 8px 14px; background: #eef1f6; color: #2d3a52 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #003CA6 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stTabs [data-baseweb="tab"][aria-selected="true"] *,
    .stTabs [data-baseweb="tab"][aria-selected="true"] p,
    .stTabs [data-baseweb="tab"][aria-selected="true"] span,
    .stTabs [data-baseweb="tab"][aria-selected="true"] div {
        color: #ffffff !important;
    }

    /* -- Streamlit overrides -- */
    .stSlider label, .stSelectbox label { color: #2d3a52 !important; font-weight: 500; }
    .stDataFrame { border: 1px solid #dfe3ea; border-radius: 8px; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* -- Glossary -- */
    .glossary-term {
        background: white; border: 1px solid #e5e7eb; border-radius: 10px;
        padding: 1rem 1.2rem; margin-bottom: 0.8rem;
    }
    .glossary-term h4 { color: #003CA6 !important; margin: 0 0 4px; font-size: 0.95rem; }
    .glossary-term p { color: #4b5563 !important; margin: 0; font-size: 0.85rem; line-height: 1.5; }
    .glossary-term .tag {
        display: inline-block; background: #eef1f6; color: #003CA6 !important;
        font-size: 0.65rem; padding: 2px 8px; border-radius: 10px;
        font-weight: 600; margin-bottom: 4px;
    }

    /* -- Country cards -- */
    .country-card {
        background: white; border: 1px solid #e5e7eb; border-radius: 10px;
        padding: 1rem; margin-bottom: 0.6rem;
    }
    .country-card .name { font-weight: 700; font-size: 0.95rem; color: #1a1a2e !important; }
    .country-card .row { display: flex; gap: 1.5rem; margin-top: 6px; flex-wrap: wrap; }
    .country-card .metric {
        font-size: 0.72rem; color: #6b7280 !important; text-transform: uppercase;
        font-weight: 600; letter-spacing: 0.3px;
    }
    .country-card .val { font-size: 1.1rem; font-weight: 800; color: #1a1a2e !important; }
    .country-card .val.red { color: #991b1b !important; }
    .country-card .val.green { color: #065f46 !important; }

    /* -- Progress bar -- */
    .prog-bar {
        background: #e5e7eb; border-radius: 8px; height: 22px; overflow: hidden;
        position: relative; margin: 4px 0;
    }
    .prog-fill {
        height: 100%; border-radius: 8px; display: flex; align-items: center;
        padding-left: 8px; font-size: 0.7rem; font-weight: 700; color: white !important;
        transition: width 0.5s ease;
    }
    </style>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

@st.cache_data
def load_summary() -> dict[str, Any]:
    with open(os.path.join(RESULTS_DIR, "summary.json")) as f:
        return json.load(f)


@st.cache_data
def load_pareto_csv() -> tuple[list[tuple[float, float]], tuple[float, float]]:
    path = os.path.join(RESULTS_DIR, "pareto_front.csv")
    pareto: list[tuple[float, float]] = []
    baseline: tuple[float, float] | None = None
    with open(path) as f:
        for row in csv.DictReader(f):
            pt = (float(row["f1"]), float(row["f2"]))
            if row["type"] == "baseline":
                baseline = pt
            else:
                pareto.append(pt)
    return pareto, baseline


@st.cache_data
def load_convergence() -> tuple[list[int], list[float], list[float]]:
    iters, means, stds = [], [], []
    with open(os.path.join(RESULTS_DIR, "convergence.csv")) as f:
        for row in csv.DictReader(f):
            iters.append(int(row["iteration"]))
            means.append(float(row["hv_mean"]))
            stds.append(float(row["hv_std"]))
    return iters, means, stds


@st.cache_data
def load_run(run_idx: int) -> dict[str, Any]:
    with open(os.path.join(RESULTS_DIR, f"run_{run_idx:02d}.json")) as f:
        return json.load(f)


@st.cache_data
def compute_allocations() -> dict[str, Any]:
    """Compute FIFO and MOHHO allocations for display."""
    from src.problem import VisaProblem
    from src.baseline import run_baseline
    from src.config import COUNTRIES, CATEGORIES
    from src.data import VISA_DATA

    problem = VisaProblem()
    fifo_alloc, fifo_fit = run_baseline(problem)

    nc, nj = len(COUNTRIES), len(CATEGORIES)
    fifo_matrix = np.zeros((nc, nj), dtype=int)
    demand_matrix = np.zeros((nc, nj), dtype=int)
    wait_matrix = np.zeros((nc, nj), dtype=int)
    date_matrix = np.zeros((nc, nj), dtype=int)

    for g in problem.groups:
        ci = COUNTRIES.index(g["country"])
        ji = CATEGORIES.index(g["category"])
        fifo_matrix[ci][ji] = fifo_alloc[g["index"]]
        demand_matrix[ci][ji] = g["n"]
        wait_matrix[ci][ji] = g["w"]
        date_matrix[ci][ji] = g["d"]

    country_demand = {}
    country_wait = {}
    for c in COUNTRIES:
        gs = [g for g in problem.groups if g["country"] == c]
        country_demand[c] = sum(g["n"] for g in gs)
        country_wait[c] = max(g["w"] for g in gs)

    # Count source types
    n_oficial = sum(1 for c in COUNTRIES for j in CATEGORIES if _source_type(c, j) == "OFICIAL")
    n_est = sum(1 for c in COUNTRIES for j in CATEGORIES if _source_type(c, j) == "EST")
    n_estdem = sum(1 for c in COUNTRIES for j in CATEGORIES if _source_type(c, j) == "EST-DEM")

    fifo_used = sum(sum(row) for row in fifo_matrix.tolist())

    return {
        "fifo_matrix": fifo_matrix.tolist(),
        "fifo_fit": fifo_fit,
        "fifo_used": fifo_used,
        "demand_matrix": demand_matrix.tolist(),
        "wait_matrix": wait_matrix.tolist(),
        "date_matrix": date_matrix.tolist(),
        "countries": COUNTRIES,
        "categories": CATEGORIES,
        "country_demand": country_demand,
        "country_wait": country_wait,
        "total_visas": problem.total_visas,
        "total_demand": problem.total_demand,
        "country_caps": dict(problem.country_caps),
        "category_caps": dict(problem.category_caps),
        "n_oficial": n_oficial,
        "n_est": n_est,
        "n_estdem": n_estdem,
    }


@st.cache_data
def compute_mohho_allocation(
    target_f1: float, target_f2: float
) -> tuple[list[list[int]], tuple[float, float], int]:
    """Find MOHHO allocation closest to target fitness via Monte Carlo sampling.

    Returns (matrix, (f1, f2), total_visas_used).
    """
    from src.problem import VisaProblem
    from src.decoder import spv, decode
    from src.config import COUNTRIES, CATEGORIES, NUM_GROUPS

    problem = VisaProblem()
    rng = np.random.default_rng(42)
    best_alloc = None
    best_dist = float("inf")
    best_fit = (0.0, 0.0)

    for _ in range(50_000):
        hawk = rng.uniform(0, 1, size=NUM_GROUPS)
        perm = spv(hawk)
        alloc = decode(perm, problem.groups, problem.total_visas,
                       problem.country_caps, problem.category_caps)
        fit = problem.evaluate(alloc)
        d = (fit[0] - target_f1) ** 2 + (fit[1] - target_f2) ** 2
        if d < best_dist:
            best_dist = d
            best_alloc = alloc
            best_fit = fit

    matrix = np.zeros((len(COUNTRIES), len(CATEGORIES)), dtype=int)
    for g in problem.groups:
        ci = COUNTRIES.index(g["country"])
        ji = CATEGORIES.index(g["category"])
        matrix[ci][ji] = best_alloc[g["index"]]

    total_used = sum(best_alloc.values())
    return matrix.tolist(), best_fit, total_used


# ──────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────

def _fmt(n: int) -> str:
    """Format integer with dot thousands separator."""
    if n == 0:
        return "-"
    return f"{n:,}".replace(",", ".")


def _pct(num: float, den: float) -> str:
    """Format percentage."""
    if den == 0:
        return "-"
    return f"{num / den * 100:.1f}%"


# ──────────────────────────────────────────────────────────────
# HTML builders
# ──────────────────────────────────────────────────────────────

def _hero() -> None:
    st.markdown("""
    <div class="hero">
        <h1>MOHHO <span class="accent">Visa Optimizer</span></h1>
        <p>Multi-Objective Harris Hawks Optimization para la Asignacion de Green Cards</p>
        <p class="subtitle">21 bloques de paises  --  5 categorias EB  --  105 grupos  --  140.000 visas/año</p>
    </div>
    """, unsafe_allow_html=True)


def _kpi_cards(summary: dict, baseline: tuple, pareto: list, data: dict) -> None:
    best_f2 = min(pareto, key=lambda x: x[1])
    f2_imp = ((baseline[1] - best_f2[1]) / baseline[1]) * 100
    fifo_used = data["fifo_used"]
    fifo_waste = data["total_visas"] - fifo_used
    gap = data["total_demand"] - data["total_visas"]

    html = f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-label">Visas Disponibles</div>
            <div class="kpi-value">{_fmt(data['total_visas'])}</div>
            <div class="kpi-delta neutral">Año fiscal 2026</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Demanda Total</div>
            <div class="kpi-value">{_fmt(data['total_demand'])}</div>
            <div class="kpi-delta bad">{_fmt(gap)} sin atender</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">FIFO Desperdicia</div>
            <div class="kpi-value">{_fmt(fifo_waste)}</div>
            <div class="kpi-delta bad">{fifo_used/data['total_visas']*100:.1f}% utilizado</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Soluciones Pareto</div>
            <div class="kpi-value">{summary['combined_pareto_size']}</div>
            <div class="kpi-delta neutral">{summary['num_runs']} corridas</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Disparidad Sist. Actual</div>
            <div class="kpi-value">{baseline[1]:.2f} a</div>
            <div class="kpi-delta bad">{baseline[1]:.0f} años entre paises</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Mejor Equidad MOHHO</div>
            <div class="kpi-value">{best_f2[1]:.2f} a</div>
            <div class="kpi-delta good">-{f2_imp:.1f}% disparidad</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _allocation_table_html(
    matrix: list[list[int]],
    countries: list[str],
    categories: list[str],
    label: str,
    badge_class: str,
) -> str:
    max_val = max(max(row) for row in matrix) if matrix else 1
    rows_html = ""
    for i, country in enumerate(countries):
        row_total = sum(matrix[i])
        cells = ""
        for j in range(len(categories)):
            val = matrix[i][j]
            cls = ""
            if val == 0:
                cls = 'class="cell-zero"'
            elif val > max_val * 0.5:
                cls = 'class="cell-high"'
            cells += f"<td {cls}>{_fmt(val)}</td>"
        name = country if len(country) <= 14 else country[:12] + ".."
        rows_html += f"<tr><td style='text-align:left;font-weight:600;white-space:nowrap;'>{name}</td>{cells}<td style='font-weight:700;'>{_fmt(row_total)}</td></tr>"

    col_totals = ""
    for j in range(len(categories)):
        t = sum(matrix[i][j] for i in range(len(countries)))
        col_totals += f"<td style='font-weight:700;'>{_fmt(t)}</td>"
    grand = sum(sum(row) for row in matrix)
    rows_html += f"<tr style='background:#e8f0fe;font-weight:700;'><td style='text-align:left;'>TOTAL</td>{col_totals}<td>{_fmt(grand)}</td></tr>"

    cat_h = "".join(f"<th>{CAT_LABELS_HTML.get(c, c)}</th>" for c in categories)
    return f"""
    <div style="margin-bottom:0.5rem;">
        <span class="badge {badge_class}">{label}</span>
    </div>
    <div style="overflow-x:auto;">
    <table class="styled-table">
        <thead><tr><th style="text-align:left;">Pais</th>{cat_h}<th>Total</th></tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
    """


def _diff_table_html(
    fifo: list[list[int]], mohho: list[list[int]],
    countries: list[str], categories: list[str],
) -> str:
    rows_html = ""
    for i, country in enumerate(countries):
        cells = ""
        row_diff = 0
        for j in range(len(categories)):
            d = mohho[i][j] - fifo[i][j]
            row_diff += d
            if d > 0:
                cells += f'<td class="cell-improve">+{_fmt(d)}</td>'
            elif d < 0:
                cells += f'<td class="cell-worse">{_fmt(d)}</td>'
            else:
                cells += '<td class="cell-zero">=</td>'
        sign = "+" if row_diff > 0 else ""
        cls = "cell-improve" if row_diff > 0 else ("cell-worse" if row_diff < 0 else "cell-zero")
        name = country if len(country) <= 14 else country[:12] + ".."
        rows_html += f'<tr><td style="text-align:left;font-weight:600;white-space:nowrap;">{name}</td>{cells}<td class="{cls}" style="font-weight:700;">{sign}{_fmt(row_diff)}</td></tr>'

    cat_h = "".join(f"<th>{CAT_LABELS_HTML.get(c, c)}</th>" for c in categories)
    return f"""
    <div style="margin-bottom:0.5rem;">
        <span class="badge" style="background:#fef3c7;color:#92400e;">DIFERENCIA (MOHHO - Sist. Actual)</span>
    </div>
    <div style="overflow-x:auto;">
    <table class="styled-table">
        <thead><tr><th style="text-align:left;">Pais</th>{cat_h}<th>Total</th></tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
    """


def _country_cards_html(
    countries: list[str],
    fifo_matrix: list[list[int]],
    mohho_matrix: list[list[int]],
    demand_matrix: list[list[int]],
    wait_data: dict[str, int],
) -> str:
    cards = ""
    for i, country in enumerate(countries):
        demand = sum(demand_matrix[i])
        fifo_t = sum(fifo_matrix[i])
        mohho_t = sum(mohho_matrix[i])
        diff = mohho_t - fifo_t
        wait = wait_data.get(country, 0)
        cov_f = (fifo_t / demand * 100) if demand > 0 else 0
        cov_m = (mohho_t / demand * 100) if demand > 0 else 0

        diff_sign = "+" if diff > 0 else ""
        diff_cls = "green" if diff > 0 else ("red" if diff < 0 else "")
        diff_text = f'{diff_sign}{_fmt(diff)}' if diff != 0 else "="

        cards += f"""
        <div class="country-card">
            <div class="name">{country}</div>
            <div class="row">
                <div>
                    <div class="metric">Demanda</div>
                    <div class="val">{_fmt(demand)}</div>
                </div>
                <div>
                    <div class="metric">Sist. Actual</div>
                    <div class="val red">{_fmt(fifo_t)}</div>
                    <div style="font-size:0.7rem;color:#6b7280 !important;">{cov_f:.1f}% cobertura</div>
                </div>
                <div>
                    <div class="metric">MOHHO</div>
                    <div class="val green">{_fmt(mohho_t)}</div>
                    <div style="font-size:0.7rem;color:#6b7280 !important;">{cov_m:.1f}% cobertura</div>
                </div>
                <div>
                    <div class="metric">Diferencia</div>
                    <div class="val {diff_cls}">{diff_text}</div>
                </div>
                <div>
                    <div class="metric">Espera max.</div>
                    <div class="val">{wait} años</div>
                </div>
            </div>
        </div>
        """
    return cards


# ──────────────────────────────────────────────────────────────
# Chart helper
# ──────────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes, title: str = "", fmt_x: bool = False) -> None:
    """Apply consistent styling to an axes."""
    ax.set_facecolor("#fafafa")
    ax.grid(True, alpha=0.15, axis="y", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.3)
    ax.spines["bottom"].set_alpha(0.3)
    # Dot thousands separator on y-axis (always numeric)
    ax.yaxis.set_major_formatter(_DOT_FORMATTER)
    if fmt_x:
        ax.xaxis.set_major_formatter(_DOT_FORMATTER)
    if title:
        ax.set_title(title, fontsize=12, fontweight="700", color=DARK, pad=12)


def _short_name(name: str, max_len: int = 10) -> str:
    """Shorten country name for axis labels."""
    if name == "Resto del Mundo":
        return "RdM"
    if name == "Corea del Sur":
        return "Corea S."
    if name == "Reino Unido":
        return "R. Unido"
    return name[:max_len]


# ──────────────────────────────────────────────────────────────
# TAB 1: Resumen Ejecutivo
# ──────────────────────────────────────────────────────────────

def _tab_resumen(data: dict, pareto: list, baseline: tuple, summary: dict) -> None:
    # Key comparison panel
    best_f2 = min(pareto, key=lambda x: x[1])
    best_f1 = min(pareto, key=lambda x: x[0])
    f2_imp = ((baseline[1] - best_f2[1]) / baseline[1]) * 100

    fifo_used = data["fifo_used"]
    fifo_waste = data["total_visas"] - fifo_used

    st.markdown(f"""
    <div style="background:white;border:1px solid #e5e7eb;border-radius:14px;padding:2rem;margin-bottom:1.5rem;">
    <div style="font-size:1.2rem;font-weight:800;color:#003CA6 !important;margin-bottom:1rem;">
    De 140.000 visas disponibles, el sistema actual solo asigna {_fmt(fifo_used)}. MOHHO mejora en los 3 objetivos.
    </div>
    <div style="display:flex;gap:2rem;flex-wrap:wrap;">
        <div style="flex:1;min-width:250px;background:#fef2f2;border-radius:12px;padding:1.2rem;">
            <div style="font-size:0.75rem;color:#991b1b !important;font-weight:700;text-transform:uppercase;">Sistema Actual (FIFO)</div>
            <div style="font-size:2rem;font-weight:800;color:#991b1b !important;">{_fmt(fifo_used)} visas</div>
            <div style="font-size:0.85rem;color:#7f1d1d !important;">Desperdicia <strong>{_fmt(fifo_waste)} visas</strong> ({fifo_used/data['total_visas']*100:.1f}% utilizado)</div>
            <div style="font-size:0.85rem;color:#7f1d1d !important;">Disparidad: <strong>{baseline[1]:.2f} años</strong> entre paises</div>
            <div style="font-size:0.85rem;color:#7f1d1d !important;">Espera no atendida: <strong>{baseline[0]:.2f} años</strong></div>
        </div>
        <div style="flex:0;display:flex;align-items:center;font-size:2rem;color:#d1d5db;">vs</div>
        <div style="flex:1;min-width:250px;background:#f0fdf4;border-radius:12px;padding:1.2rem;">
            <div style="font-size:0.75rem;color:#065f46 !important;font-weight:700;text-transform:uppercase;">MOHHO (Optimizado)</div>
            <div style="font-size:2rem;font-weight:800;color:#065f46 !important;">Hasta {_fmt(data['total_visas'])} visas</div>
            <div style="font-size:0.85rem;color:#064e3b !important;">Puede asignar <strong>100%</strong> de las visas disponibles</div>
            <div style="font-size:0.85rem;color:#064e3b !important;">Disparidad: <strong>{best_f2[1]:.2f} años</strong> (-{f2_imp:.1f}%)</div>
            <div style="font-size:0.85rem;color:#064e3b !important;">Espera no atendida: <strong>{best_f1[0]:.2f} años</strong></div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Solution selector with nicknames ---
    st.markdown('<div class="section-head">Explorador Rapido de Soluciones</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:0.85rem;color:#6b7280 !important;margin-bottom:0.5rem;">
    Cada solucion del frente de Pareto representa un enfoque diferente. Selecciona una para ver que prioriza.
    </p>
    """, unsafe_allow_html=True)

    sorted_pareto = sorted(pareto, key=lambda x: x[1])
    n_sol = len(sorted_pareto)

    # Assign nicknames based on position in Pareto front
    def _sol_nickname(idx: int, total: int, sol: tuple) -> str:
        pct = idx / max(total - 1, 1)
        if idx == 0:
            return "Maxima Equidad"
        if idx == total - 1:
            return "Minima Espera"
        if abs(pct - 0.5) < 0.06:
            return "Equilibrada"
        if pct < 0.2:
            return "Pro-Equidad"
        if pct < 0.4:
            return "Equidad-Moderada"
        if pct > 0.8:
            return "Pro-Espera"
        if pct > 0.6:
            return "Espera-Moderada"
        return "Balanceada"

    sol_options = []
    for i, s in enumerate(sorted_pareto):
        nick = _sol_nickname(i, n_sol, s)
        sol_options.append(f"{nick} -- f1={s[0]:.2f}a, f2={s[1]:.2f}a")

    default_idx = n_sol // 2
    chosen_idx = st.selectbox(
        "Selecciona una solucion:",
        range(n_sol),
        index=default_idx,
        format_func=lambda i: sol_options[i],
        key="resumen_sol_select",
    )
    chosen_sol = sorted_pareto[chosen_idx]
    chosen_nick = _sol_nickname(chosen_idx, n_sol, chosen_sol)

    imp_f1 = ((baseline[0] - chosen_sol[0]) / baseline[0]) * 100
    imp_f2 = ((baseline[1] - chosen_sol[1]) / baseline[1]) * 100

    nick_color = "#065f46" if "Equidad" in chosen_nick else (
        "#0277bd" if "Espera" in chosen_nick else "#6a1b9a")

    st.markdown(f"""
    <div style="background:white;border:2px solid {nick_color};border-radius:14px;padding:1.5rem;margin-bottom:1rem;">
    <div style="font-size:1.1rem;font-weight:800;color:{nick_color} !important;margin-bottom:8px;">
    Solucion: "{chosen_nick}"
    </div>
    <div style="display:flex;gap:2rem;flex-wrap:wrap;">
        <div>
            <div style="font-size:0.7rem;color:#6b7280 !important;text-transform:uppercase;font-weight:600;">Espera no atendida</div>
            <div style="font-size:1.8rem;font-weight:800;color:#1a1a2e !important;">{chosen_sol[0]:.2f} <span style="font-size:0.9rem;">años</span></div>
            <div style="font-size:0.78rem;color:{'#065f46' if imp_f1 > 0 else '#991b1b'} !important;font-weight:600;">{'Mejora' if imp_f1 > 0 else 'Sacrifica'} {abs(imp_f1):.1f}% vs Sist. Actual</div>
        </div>
        <div>
            <div style="font-size:0.7rem;color:#6b7280 !important;text-transform:uppercase;font-weight:600;">Disparidad entre paises</div>
            <div style="font-size:1.8rem;font-weight:800;color:#1a1a2e !important;">{chosen_sol[1]:.2f} <span style="font-size:0.9rem;">años</span></div>
            <div style="font-size:0.78rem;color:#065f46 !important;font-weight:600;">Mejora {imp_f2:.1f}% vs Sist. Actual</div>
        </div>
        <div>
            <div style="font-size:0.7rem;color:#6b7280 !important;text-transform:uppercase;font-weight:600;">Que significa?</div>
            <div style="font-size:0.85rem;color:#4b5563 !important;max-width:300px;">
            {"Prioriza que todos los paises esperen tiempos similares, aunque algunos solicitantes antiguos esperen un poco mas." if "Equidad" in chosen_nick or chosen_nick == "Maxima Equidad" else
             "Prioriza atender primero a quienes mas tiempo llevan esperando, aunque algunos paises queden con mas disparidad." if "Espera" in chosen_nick or chosen_nick == "Minima Espera" else
             "Busca un compromiso intermedio entre reducir la espera y mantener la equidad entre paises."}
            </div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # Utilization visualization
    st.markdown('<div class="section-head">Utilizacion de Visas (Objetivo 3: No Desperdiciar)</div>',
                unsafe_allow_html=True)

    fifo_used = data["fifo_used"]
    fifo_waste = data["total_visas"] - fifo_used
    fifo_pct = fifo_used / data["total_visas"] * 100

    # Compute MOHHO utilization for a balanced solution
    countries = data["countries"]
    sorted_pareto = sorted(pareto, key=lambda x: x[1])
    eq_sol = sorted_pareto[len(sorted_pareto) // 2]
    mohho_matrix, _, mohho_used = compute_mohho_allocation(eq_sol[0], eq_sol[1])
    mohho_waste = data["total_visas"] - mohho_used
    mohho_pct = mohho_used / data["total_visas"] * 100

    st.markdown(f"""
    <div style="background:white;border:1px solid #e5e7eb;border-radius:12px;padding:1.5rem;margin-bottom:1rem;">
    <div style="display:flex;gap:2rem;flex-wrap:wrap;">
        <div style="flex:1;min-width:250px;">
            <div style="font-size:0.75rem;color:#6b7280 !important;font-weight:600;text-transform:uppercase;margin-bottom:6px;">Sist. Actual (FIFO): {_fmt(fifo_used)} / {_fmt(data['total_visas'])}</div>
            <div class="prog-bar">
                <div class="prog-fill" style="width:{fifo_pct:.1f}%;background:linear-gradient(90deg,#ef4444,#dc2626);">{fifo_pct:.1f}%</div>
            </div>
            <div style="font-size:0.78rem;color:#991b1b !important;font-weight:600;margin-top:4px;">{_fmt(fifo_waste)} visas desperdiciadas</div>
        </div>
        <div style="flex:1;min-width:250px;">
            <div style="font-size:0.75rem;color:#6b7280 !important;font-weight:600;text-transform:uppercase;margin-bottom:6px;">MOHHO (solucion equilibrada): {_fmt(mohho_used)} / {_fmt(data['total_visas'])}</div>
            <div class="prog-bar">
                <div class="prog-fill" style="width:{mohho_pct:.1f}%;background:linear-gradient(90deg,#22c55e,#16a34a);">{mohho_pct:.1f}%</div>
            </div>
            <div style="font-size:0.78rem;color:#065f46 !important;font-weight:600;margin-top:4px;">{_fmt(mohho_waste)} visas desperdiciadas{' (mejor!)' if mohho_waste < fifo_waste else ''}</div>
        </div>
    </div>
    <p style="font-size:0.82rem;color:#6b7280 !important;margin-top:12px;">
    <strong>¿Por que se desperdician visas?</strong> El sistema FIFO procesa primero las fechas mas antiguas (India EB-2, EB-3),
    agotando los limites por pais de India y China. Esto deja visas disponibles en EB-1 que no pueden ser asignadas
    porque los paises con mayor demanda ya alcanzaron su tope. MOHHO, al reordenar el procesamiento,
    puede aprovechar mejor las 140.000 visas.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Distribution comparison: 100% stacked bar
    st.markdown('<div class="section-head">Como se Redistribuyen las Visas</div>',
                unsafe_allow_html=True)

    fifo_by_c = [sum(data["fifo_matrix"][i]) for i in range(len(countries))]
    mohho_by_c = [sum(mohho_matrix[i]) for i in range(len(countries))]

    fig, axes = plt.subplots(2, 1, figsize=(12, 4), gridspec_kw={"hspace": 0.5})

    for ax, values, label, cmap_name in [
        (axes[0], fifo_by_c, "Sist. Actual (FIFO)", "Reds"),
        (axes[1], mohho_by_c, "MOHHO", "Greens"),
    ]:
        total = sum(values)
        left = 0
        for i, (v, c) in enumerate(zip(values, countries)):
            pct = v / total if total > 0 else 0
            color = COUNTRY_COLORS[i]
            ax.barh(0, v, left=left, height=0.6, color=color, edgecolor="white", linewidth=0.3)
            if pct > 0.04:
                ax.text(left + v / 2, 0, f"{_short_name(c)}\n{pct*100:.1f}%",
                        ha="center", va="center", fontsize=6.5, fontweight="600", color="white",
                        path_effects=[pe.withStroke(linewidth=2, foreground="black")])
            left += v
        ax.set_xlim(0, total)
        ax.set_yticks([])
        ax.set_title(f"{label} ({_fmt(total)} visas)", fontsize=10, fontweight="700",
                     color=DARK, loc="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000):,}k".replace(",", ".")))

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Data quality summary
    st.markdown('<div class="section-head">Calidad de los Datos</div>',
                unsafe_allow_html=True)

    total_entries = data["n_oficial"] + data["n_est"] + data["n_estdem"]
    st.markdown(f"""
    <div style="display:flex;gap:1rem;flex-wrap:wrap;">
        <div style="flex:1;min-width:180px;background:#d1fae5;border-radius:12px;padding:1rem;text-align:center;">
            <div style="font-size:2rem;font-weight:800;color:#065f46 !important;">{data['n_oficial']}</div>
            <div style="font-size:0.8rem;color:#065f46 !important;font-weight:600;">OFICIAL ({data['n_oficial']/total_entries*100:.0f}%)</div>
            <div style="font-size:0.72rem;color:#6b7280 !important;">USCIS FY2025 Q2</div>
        </div>
        <div style="flex:1;min-width:180px;background:#fef3c7;border-radius:12px;padding:1rem;text-align:center;">
            <div style="font-size:2rem;font-weight:800;color:#92400e !important;">{data['n_est']}</div>
            <div style="font-size:0.8rem;color:#92400e !important;font-weight:600;">EST ({data['n_est']/total_entries*100:.0f}%)</div>
            <div style="font-size:0.72rem;color:#6b7280 !important;">DHS Yearbook FY2023</div>
        </div>
        <div style="flex:1;min-width:180px;background:#dbeafe;border-radius:12px;padding:1rem;text-align:center;">
            <div style="font-size:2rem;font-weight:800;color:#1e40af !important;">{data['n_estdem']}</div>
            <div style="font-size:0.8rem;color:#1e40af !important;font-weight:600;">EST-DEM ({data['n_estdem']/total_entries*100:.0f}%)</div>
            <div style="font-size:0.72rem;color:#6b7280 !important;">Demanda anual estimada</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# TAB 2: Datos de Entrada
# ──────────────────────────────────────────────────────────────

def _tab_raw_data(data: dict) -> None:
    countries = data["countries"]
    categories = data["categories"]

    st.markdown("""
    <div class="info-box">
    <strong>Fuentes de datos:</strong><br>
    <strong>OFICIAL</strong> = USCIS "Approved I-140/I-360/I-526 Petitions", FY2025 Q2.<br>
    <strong>EST</strong> = Estimaciones basadas en porcentajes del DHS Yearbook FY2023 sobre bloque "All Other" de USCIS.<br>
    <strong>EST-DEM</strong> = Demanda anual estimada para categorias con fecha "Current" (sin backlog).<br>
    <strong>Fechas de prioridad:</strong> Visa Bulletin, Abril 2026 (Final Action Dates).
    </div>
    """, unsafe_allow_html=True)

    # --- Demand table ---
    st.markdown('<div class="section-head">Peticiones Aprobadas Pendientes (n) por Pais y Categoria</div>',
                unsafe_allow_html=True)

    cat_h = "".join(f"<th>{CAT_LABELS_HTML.get(c, c)}</th>" for c in categories)
    rows = ""
    for i, country in enumerate(countries):
        cells = ""
        for j, cat in enumerate(categories):
            val = data["demand_matrix"][i][j]
            src = _source_type(country, cat)
            badge = _source_badge(src)
            cls = 'class="cell-high"' if val > 20000 else ('class="cell-zero"' if val == 0 else "")
            cells += f"<td {cls}>{_fmt(val)} {badge}</td>"
        row_total = sum(data["demand_matrix"][i])
        rows += f"<tr><td style='text-align:left;font-weight:600;white-space:nowrap;'>{country}</td>{cells}<td style='font-weight:700;'>{_fmt(row_total)}</td></tr>"

    # Totals row
    col_tots = ""
    for j in range(len(categories)):
        t = sum(data["demand_matrix"][i][j] for i in range(len(countries)))
        col_tots += f"<td style='font-weight:700;'>{_fmt(t)}</td>"
    grand = sum(sum(data["demand_matrix"][i]) for i in range(len(countries)))
    rows += f"<tr style='background:#e8f0fe;font-weight:700;'><td style='text-align:left;'>TOTAL</td>{col_tots}<td>{_fmt(grand)}</td></tr>"

    st.markdown(f"""
    <div style="overflow-x:auto;">
    <table class="styled-table">
        <thead><tr><th style="text-align:left;">Pais</th>{cat_h}<th>Total</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    # --- Priority dates table ---
    st.markdown("---")
    st.markdown('<div class="section-head">Fechas de Prioridad (d) -- Visa Bulletin Abril 2026</div>',
                unsafe_allow_html=True)

    rows_d = ""
    for i, country in enumerate(countries):
        cells = ""
        for j, cat in enumerate(categories):
            d = data["date_matrix"][i][j]
            w = data["wait_matrix"][i][j]
            if d >= 2026:
                cells += '<td style="color:#00873c !important;font-weight:600;">Current</td>'
            elif w >= 10:
                cells += f'<td style="color:#c62828 !important;font-weight:700;">{d}</td>'
            elif w >= 3:
                cells += f'<td style="color:#e65100 !important;font-weight:600;">{d}</td>'
            else:
                cells += f"<td>{d}</td>"
        max_w = max(data["wait_matrix"][i])
        rows_d += f"<tr><td style='text-align:left;font-weight:600;white-space:nowrap;'>{country}</td>{cells}<td style='font-weight:700;'>{max_w} a</td></tr>"

    st.markdown(f"""
    <div style="overflow-x:auto;">
    <table class="styled-table">
        <thead><tr><th style="text-align:left;">Pais</th>{cat_h}<th>Espera max.</th></tr></thead>
        <tbody>{rows_d}</tbody>
    </table>
    </div>
    <p style="font-size:0.78rem;color:#6b7280 !important;margin-top:8px;">
    <strong>Rojo</strong> = espera >= 10 años. <strong>Naranja</strong> = 3-9 años.
    <strong>Verde "Current"</strong> = sin backlog (fecha de prioridad = 2026).
    </p>
    """, unsafe_allow_html=True)

    # --- Caps table ---
    st.markdown("---")
    st.markdown('<div class="section-head">Restricciones del Sistema</div>',
                unsafe_allow_html=True)

    cap_rows = ""
    for country in countries:
        cap = data["country_caps"].get(country, 0)
        dem = data["country_demand"].get(country, 0)
        uso = min(dem, cap)
        sat = "SI" if dem > cap else "NO"
        sat_cls = "cell-worse" if dem > cap else "cell-improve"
        cap_rows += f"<tr><td style='text-align:left;font-weight:600;'>{country}</td><td>{_fmt(cap)}</td><td>{_fmt(dem)}</td><td class='{sat_cls}'>{sat}</td></tr>"

    st.markdown(f"""
    <div style="display:flex;gap:2rem;flex-wrap:wrap;">
    <div style="flex:1;min-width:300px;">
    <div style="font-weight:700;color:#003CA6 !important;margin-bottom:8px;">Limites por Pais (7% rule = 25.620)</div>
    <div style="overflow-x:auto;">
    <table class="styled-table">
        <thead><tr><th style="text-align:left;">Pais</th><th>Limite</th><th>Demanda</th><th>Saturado?</th></tr></thead>
        <tbody>{cap_rows}</tbody>
    </table>
    </div>
    </div>
    <div style="flex:1;min-width:300px;">
    <div style="font-weight:700;color:#003CA6 !important;margin-bottom:8px;">Limites por Categoria (tras spillover)</div>
    <table class="styled-table">
        <thead><tr><th>Categoria</th><th>Limite efectivo</th><th>Demanda total</th><th>Saturada?</th></tr></thead>
        <tbody>
    """, unsafe_allow_html=True)

    cat_rows_html = ""
    for cat in categories:
        cap = data["category_caps"].get(cat, 0)
        dem = sum(data["demand_matrix"][i][categories.index(cat)] for i in range(len(countries)))
        sat = "SI" if dem > cap else "NO"
        sat_cls = "cell-worse" if dem > cap else "cell-improve"
        cat_rows_html += f"<tr><td style='font-weight:600;'>{cat}</td><td>{_fmt(cap)}</td><td>{_fmt(dem)}</td><td class='{sat_cls}'>{sat}</td></tr>"

    st.markdown(f"""
        {cat_rows_html}
        </tbody>
    </table>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # Demand heatmap
    st.markdown("---")
    st.markdown('<div class="section-head">Mapa de Calor: Demanda por Pais y Categoria</div>',
                unsafe_allow_html=True)

    mat = np.array(data["demand_matrix"])
    fig, ax = plt.subplots(figsize=(9, 10))
    im = ax.imshow(np.log1p(mat), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([CAT_LABELS.get(c, c) for c in categories], fontsize=8, fontweight="600")
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries, fontsize=9)
    for i in range(len(countries)):
        for j in range(len(categories)):
            v = int(mat[i][j])
            src = _source_type(countries[i], categories[j])
            mark = "" if src == "OFICIAL" else "*"
            clr = "white" if v > 5000 else DARK
            ax.text(j, i, f"{_fmt(v)}{mark}", ha="center", va="center",
                    fontsize=8, color=clr, fontweight="600")
    _style_ax(ax, "Demanda pendiente (escala log) -- * = estimacion")
    plt.colorbar(im, ax=ax, shrink=0.7, label="log(n+1)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# TAB 3: Asignacion Completa
# ──────────────────────────────────────────────────────────────

def _tab_allocation(data: dict, pareto: list, baseline: tuple) -> None:
    countries = data["countries"]
    categories = data["categories"]

    st.markdown('<div class="section-head">Demanda Total por Pais y Categoria</div>',
                unsafe_allow_html=True)
    st.markdown(_allocation_table_html(data["demand_matrix"], countries, categories,
                                        "DEMANDA PENDIENTE", "badge-fifo"),
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="warn-box">
    <strong>Problema:</strong> {_fmt(data['total_demand'])} solicitantes compiten
    por solo {_fmt(data['total_visas'])} visas
    ({data['total_visas']/data['total_demand']*100:.1f}% de cobertura).<br>
    <strong>Modelo:</strong> 21 bloques de paises x 5 categorias EB = <strong>105 grupos</strong>.
    "Resto del Mundo" agrupa a los ~190 paises con menor demanda individual.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-head">Sistema Actual (FIFO) vs MOHHO (Optimizado)</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <p style="font-size:0.85rem;color:#6b7280 !important;margin-bottom:1rem;">
    Selecciona una solucion del frente de Pareto para comparar.
    <strong>Izquierda</strong> = mas equidad. <strong>Derecha</strong> = menor espera.
    </p>
    """, unsafe_allow_html=True)

    sorted_pareto = sorted(pareto, key=lambda x: x[1])
    sol_idx = st.slider(
        "Navegar soluciones Pareto (izq=equidad, der=espera)",
        0, len(sorted_pareto) - 1, len(sorted_pareto) // 2,
        key="alloc_slider",
    )
    sel = sorted_pareto[sol_idx]

    mohho_matrix, mohho_fit, mohho_used = compute_mohho_allocation(sel[0], sel[1])

    st.markdown(
        f"**Solucion seleccionada:** "
        f"Espera = **{sel[0]:.2f}** años -- "
        f"Disparidad = **{sel[1]:.2f}** años -- "
        f"Visas asignadas: **{_fmt(mohho_used)}** / {_fmt(data['total_visas'])} "
        f"({mohho_used/data['total_visas']*100:.1f}%)"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(_allocation_table_html(
            data["fifo_matrix"], countries, categories,
            f"SIST. ACTUAL -- f1={data['fifo_fit'][0]:.2f}  f2={data['fifo_fit'][1]:.2f}",
            "badge-fifo"),
            unsafe_allow_html=True)
    with col2:
        st.markdown(_allocation_table_html(
            mohho_matrix, countries, categories,
            f"MOHHO -- f1={mohho_fit[0]:.2f}  f2={mohho_fit[1]:.2f}",
            "badge-mohho"),
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(_diff_table_html(data["fifo_matrix"], mohho_matrix, countries, categories),
                unsafe_allow_html=True)

    # Per-country cards
    st.markdown("---")
    st.markdown('<div class="section-head">Resumen por Pais (21 bloques)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:0.85rem;color:#6b7280 !important;margin-bottom:0.5rem;">
    Vista individual de cada bloque: demanda, asignacion actual, asignacion MOHHO.
    "Resto del Mundo" agrupa ~190 paises con menor demanda individual.
    </p>
    """, unsafe_allow_html=True)
    st.markdown(
        _country_cards_html(countries, data["fifo_matrix"], mohho_matrix,
                            data["demand_matrix"], data["country_wait"]),
        unsafe_allow_html=True,
    )

    # Coverage chart
    st.markdown("---")
    st.markdown('<div class="section-head">Tasa de Cobertura por Pais</div>',
                unsafe_allow_html=True)

    fifo_by_c = [sum(data["fifo_matrix"][i]) for i in range(len(countries))]
    mohho_by_c = [sum(mohho_matrix[i]) for i in range(len(countries))]
    demand_by_c = [sum(data["demand_matrix"][i]) for i in range(len(countries))]

    cov_fifo = [(f / d * 100) if d > 0 else 0 for f, d in zip(fifo_by_c, demand_by_c)]
    cov_mohho = [(m / d * 100) if d > 0 else 0 for m, d in zip(mohho_by_c, demand_by_c)]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(countries))
    w = 0.35
    ax.bar(x - w / 2, cov_fifo, w, label="Sist. Actual (FIFO)", color="#fca5a5", edgecolor="white", linewidth=0.8)
    ax.bar(x + w / 2, cov_mohho, w, label="MOHHO", color="#6ee7b7", edgecolor="white", linewidth=0.8)
    for xi, cf, cm in zip(x, cov_fifo, cov_mohho):
        if cf > 0:
            ax.text(xi - w / 2, cf + 0.8, f"{cf:.1f}%", ha="center", fontsize=7, fontweight="600")
        if cm > 0:
            ax.text(xi + w / 2, cm + 0.8, f"{cm:.1f}%", ha="center", fontsize=7, fontweight="600")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_name(c) for c in countries], fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Cobertura (%)", fontsize=11, fontweight="600")
    _style_ax(ax, "% de demanda cubierta por pais")
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# TAB 4: Analisis de Objetivos
# ──────────────────────────────────────────────────────────────

def _tab_objectives(data: dict, pareto: list, baseline: tuple) -> None:
    st.markdown('<div class="section-head">Explorador Interactivo de Soluciones</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:0.85rem;color:#6b7280 !important;margin-bottom:0.5rem;">
    Cada punto del frente de Pareto es una forma distinta de repartir las 140.000 visas.
    <strong>Izquierda</strong> = equidad. <strong>Derecha</strong> = menor espera.
    </p>
    """, unsafe_allow_html=True)

    sorted_pareto = sorted(pareto, key=lambda x: x[1])
    obj_idx = st.slider(
        "Navegar combinaciones (izq=equidad, der=espera)",
        0, len(sorted_pareto) - 1, len(sorted_pareto) // 2,
        key="obj_slider",
    )
    sel = sorted_pareto[obj_idx]
    best_f1 = min(pareto, key=lambda x: x[0])
    best_f2 = min(pareto, key=lambda x: x[1])

    # Metrics panel
    imp_f1 = ((baseline[0] - sel[0]) / baseline[0]) * 100
    imp_f2 = ((baseline[1] - sel[1]) / baseline[1]) * 100
    st.markdown(f"""
    <div style="background:white;border:1px solid #e5e7eb;border-radius:12px;padding:1.5rem;margin-bottom:1rem;">
    <div style="display:flex;gap:2rem;align-items:center;flex-wrap:wrap;">
        <div>
            <div style="font-size:0.7rem;color:#6b7280 !important;text-transform:uppercase;font-weight:600;">Espera no atendida</div>
            <div style="font-size:2rem;font-weight:800;color:#003CA6 !important;">{sel[0]:.2f} <span style="font-size:1rem;">años</span></div>
        </div>
        <div>
            <div style="font-size:0.7rem;color:#6b7280 !important;text-transform:uppercase;font-weight:600;">Disparidad entre paises</div>
            <div style="font-size:2rem;font-weight:800;color:#003CA6 !important;">{sel[1]:.2f} <span style="font-size:1rem;">años</span></div>
        </div>
        <div>
            <div style="font-size:0.7rem;color:#6b7280 !important;text-transform:uppercase;font-weight:600;">Mejora espera</div>
            <div style="font-size:1.3rem;font-weight:700;color:{'#065f46' if imp_f1 > 0 else '#991b1b'} !important;">
            {'+' if imp_f1 > 0 else ''}{imp_f1:.2f}%</div>
        </div>
        <div>
            <div style="font-size:0.7rem;color:#6b7280 !important;text-transform:uppercase;font-weight:600;">Mejora equidad</div>
            <div style="font-size:1.3rem;font-weight:700;color:#065f46 !important;">
            +{imp_f2:.1f}%</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # Pareto scatter
    f1s = [p[0] for p in pareto]
    f2s = [p[1] for p in pareto]

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(f1s, f2s, c=f2s, cmap="RdYlGn_r", s=50, alpha=0.85,
                         edgecolors="white", linewidths=0.5, zorder=3)
    ax.scatter(baseline[0], baseline[1], c="#c62828", s=280, marker="*",
               zorder=5, label=f"Sist. Actual ({baseline[0]:.2f}, {baseline[1]:.2f})",
               edgecolors="white", linewidths=1)
    ax.scatter(best_f1[0], best_f1[1], c=BLUE, s=120, marker="D", zorder=5,
               label=f"Menor espera ({best_f1[0]:.2f}, {best_f1[1]:.2f})",
               edgecolors="white", linewidths=1)
    ax.scatter(best_f2[0], best_f2[1], c="#00873c", s=120, marker="s", zorder=5,
               label=f"Mayor equidad ({best_f2[0]:.2f}, {best_f2[1]:.2f})",
               edgecolors="white", linewidths=1)
    ax.scatter(sel[0], sel[1], c=YELLOW, s=220, marker="o", zorder=6,
               label=f"Seleccionada ({sel[0]:.2f}, {sel[1]:.2f})",
               edgecolors=DARK, linewidths=2)

    ax.annotate("Sist. Actual\n(FIFO)", xy=(baseline[0], baseline[1]),
                xytext=(baseline[0] - 0.08, baseline[1] - 1.5),
                fontsize=8, color="#c62828", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.2))

    # Improvement arrow
    ax.annotate("", xy=(best_f2[0], best_f2[1]),
                xytext=(baseline[0], baseline[1]),
                arrowprops=dict(arrowstyle="-|>", color="#00873c", lw=1.5,
                                linestyle="--", alpha=0.5))

    ax.set_xlabel("Espera no atendida (años)", fontsize=10, fontweight="600")
    ax.set_ylabel("Disparidad entre paises (años)", fontsize=10, fontweight="600")
    _style_ax(ax, "Frente de Pareto: cada punto es una forma distinta de repartir las visas", fmt_x=True)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    plt.colorbar(scatter, ax=ax, shrink=0.7, label="Disparidad (años)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # Objective explanations
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-head">Obj. 1 -- Espera No Atendida</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:white;border:1px solid #e5e7eb;border-radius:12px;padding:1.5rem;">
        <p style="font-size:0.85rem;color:#6b7280 !important;margin:0 0 8px;">
        Promedio ponderado de años de espera de quienes <em>no</em> reciben visa. Cuanto menor, mejor.
        </p>
        <div style="display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap;">
            <div>
                <div style="font-size:0.7rem;color:#6b7280 !important;text-transform:uppercase;font-weight:600;">Sist. Actual</div>
                <div style="font-size:1.5rem;font-weight:800;color:#991b1b !important;">{baseline[0]:.2f} años</div>
            </div>
            <div style="font-size:1.5rem;color:#d1d5db;">&rarr;</div>
            <div>
                <div style="font-size:0.7rem;color:#6b7280 !important;text-transform:uppercase;font-weight:600;">Mejor MOHHO</div>
                <div style="font-size:1.5rem;font-weight:800;color:#065f46 !important;">{best_f1[0]:.2f} años</div>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        f2_imp = ((baseline[1] - best_f2[1]) / baseline[1]) * 100
        st.markdown('<div class="section-head">Obj. 2 -- Equidad entre Paises</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:white;border:1px solid #e5e7eb;border-radius:12px;padding:1.5rem;">
        <p style="font-size:0.85rem;color:#6b7280 !important;margin:0 0 8px;">
        Mayor diferencia de espera promedio entre paises. Cuanto menor, mas justo.
        </p>
        <div style="display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap;">
            <div>
                <div style="font-size:0.7rem;color:#6b7280 !important;text-transform:uppercase;font-weight:600;">Sist. Actual</div>
                <div style="font-size:1.5rem;font-weight:800;color:#991b1b !important;">{baseline[1]:.2f} años</div>
            </div>
            <div style="font-size:1.5rem;color:#d1d5db;">&rarr;</div>
            <div>
                <div style="font-size:0.7rem;color:#6b7280 !important;text-transform:uppercase;font-weight:600;">Mejor MOHHO</div>
                <div style="font-size:1.5rem;font-weight:800;color:#065f46 !important;">{best_f2[1]:.2f} años</div>
            </div>
        </div>
        <div style="background:#d1fae5;border-radius:8px;padding:10px 14px;margin-top:12px;">
        <strong style="color:#065f46 !important;">Impacto:</strong>
        <span style="color:#065f46 !important;font-size:0.85rem;">
        La disparidad baja de {baseline[1]:.0f} a {best_f2[1]:.1f} años ({f2_imp:.1f}% de mejora).
        </span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # f3 explanation
    st.markdown("---")
    fifo_used = data.get("fifo_used", 140000)
    fifo_waste = data["total_visas"] - fifo_used
    st.markdown(f"""
    <div style="background:#f0f4ff;border:1px solid #c7d2fe;border-radius:12px;padding:1.5rem;margin:1rem 0;">
    <div style="font-size:0.95rem;font-weight:700;color:#003CA6 !important;margin-bottom:8px;">
    Objetivo 3: Visas Desperdiciadas (f3 = V - Sum x_g)
    </div>
    <p style="font-size:0.85rem;color:#4b5563 !important;margin:0 0 8px;">
    El sistema actual FIFO solo asigna <strong>{_fmt(fifo_used)} de {_fmt(data['total_visas'])}</strong> visas
    disponibles, desperdiciando <strong>{_fmt(fifo_waste)}</strong> ({fifo_waste/data['total_visas']*100:.1f}%).
    </p>
    <p style="font-size:0.85rem;color:#4b5563 !important;margin:0 0 8px;">
    <strong>¿Por que?</strong> FIFO procesa las solicitudes mas antiguas primero. India EB-2 (2014) y EB-3 (2013)
    tienen fechas muy antiguas, asi que se procesan primero y agotan el limite por pais de India (25.620).
    Cuando FIFO llega a India EB-1, India ya esta llena. La categoria EB-1 tiene {_fmt(40040)} plazas disponibles
    pero solo usa {_fmt(40040 - fifo_waste)}, porque los paises con mayor demanda de EB-1 ya estan topes.
    </p>
    <p style="font-size:0.85rem;color:#4b5563 !important;margin:0;">
    <strong>MOHHO puede resolver esto:</strong> al reordenar el procesamiento, puede lograr que se asignen
    las {_fmt(data['total_visas'])} visas completas (100% utilizacion) mientras tambien mejora la equidad.
    El objetivo bi-objetivo actual captura esto implicitamente: menos desperdicio = menor f1.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Histograms
    st.markdown('<div class="section-head">Distribucion de Soluciones Pareto</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(f1s, bins=20, color=BLUE, alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.axvline(baseline[0], color="#c62828", linestyle="--", linewidth=2, label="Sist. Actual")
        ax.set_xlabel("Espera no atendida (años)", fontsize=9)
        ax.set_ylabel("Frecuencia", fontsize=9)
        _style_ax(ax, "Distribucion Obj. 1", fmt_x=True)
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(f2s, bins=20, color="#00873c", alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.axvline(baseline[1], color="#c62828", linestyle="--", linewidth=2, label="Sist. Actual")
        ax.set_xlabel("Disparidad entre paises (años)", fontsize=9)
        ax.set_ylabel("Frecuencia", fontsize=9)
        _style_ax(ax, "Distribucion Obj. 2", fmt_x=True)
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
# TAB 5: Analisis por Pais
# ──────────────────────────────────────────────────────────────

def _tab_country_analysis(data: dict, pareto: list) -> None:
    countries = data["countries"]
    categories = data["categories"]

    sorted_pareto = sorted(pareto, key=lambda x: x[1])
    eq_sol = sorted_pareto[len(sorted_pareto) // 2]
    mohho_matrix, mohho_fit, _ = compute_mohho_allocation(eq_sol[0], eq_sol[1])
    fifo_matrix = data["fifo_matrix"]

    fifo_by_c = [sum(fifo_matrix[i]) for i in range(len(countries))]
    mohho_by_c = [sum(mohho_matrix[i]) for i in range(len(countries))]
    demand_by_c = [sum(data["demand_matrix"][i]) for i in range(len(countries))]

    # --- Grouped bar: Demand vs FIFO vs MOHHO ---
    st.markdown('<div class="section-head">Demanda vs Asignacion por Pais</div>',
                unsafe_allow_html=True)

    # Use horizontal grouped bar for better label readability
    fig, ax = plt.subplots(figsize=(12, 10))
    y = np.arange(len(countries))
    h = 0.25

    ax.barh(y + h, demand_by_c, h, label="Demanda", color="#cbd5e1", edgecolor="white", linewidth=0.8)
    ax.barh(y, fifo_by_c, h, label="Sist. Actual (FIFO)", color="#fca5a5", edgecolor="white", linewidth=0.8)
    ax.barh(y - h, mohho_by_c, h, label="MOHHO", color="#6ee7b7", edgecolor="white", linewidth=0.8)

    for bar_vals, offset in [(fifo_by_c, 0), (mohho_by_c, -h)]:
        for yi, val in zip(y, bar_vals):
            if val > 500:
                ax.text(val + 200, yi + offset, _fmt(int(val)),
                        ha="left", va="center", fontsize=7, fontweight="600", color=DARK)

    ax.set_yticks(y)
    ax.set_yticklabels(countries, fontsize=9, fontweight="500")
    ax.set_xlabel("Visas", fontsize=11, fontweight="600")
    ax.invert_yaxis()
    _style_ax(ax, "Demanda vs Asignacion por Pais")
    ax.grid(True, alpha=0.15, axis="x")
    ax.grid(False, axis="y")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000):,}k".replace(",", ".")))
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Donut charts: Demand / FIFO / MOHHO distribution ---
    st.markdown('<div class="section-head">Distribucion Porcentual: Demanda vs Asignacion</div>',
                unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for ax, values, title in [
        (axes[0], demand_by_c, "Demanda"),
        (axes[1], fifo_by_c, "Sist. Actual (FIFO)"),
        (axes[2], mohho_by_c, "MOHHO"),
    ]:
        total = sum(values)
        # Group slices < 3% into "Otros"
        threshold = total * 0.03
        main_labels, main_vals, main_colors = [], [], []
        otros_val = 0
        for ci, (c, v) in enumerate(zip(countries, values)):
            if v >= threshold:
                main_labels.append(_short_name(c))
                main_vals.append(v)
                main_colors.append(COUNTRY_COLORS[ci])
            else:
                otros_val += v
        if otros_val > 0:
            main_labels.append("Otros")
            main_vals.append(otros_val)
            main_colors.append("#b0bec5")

        wedges, texts, autotexts = ax.pie(
            main_vals, labels=None, autopct="",
            colors=main_colors, startangle=90,
            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
            pctdistance=0.75)

        # External labels with lines
        for i, (wedge, lbl, val) in enumerate(zip(wedges, main_labels, main_vals)):
            ang = (wedge.theta2 + wedge.theta1) / 2
            x_pt = np.cos(np.radians(ang))
            y_pt = np.sin(np.radians(ang))
            pct = val / total * 100
            ha = "left" if x_pt >= 0 else "right"
            ax.annotate(
                f"{lbl} ({pct:.1f}%)",
                xy=(0.85 * x_pt, 0.85 * y_pt),
                xytext=(1.25 * x_pt, 1.15 * y_pt),
                fontsize=8, fontweight="600", color=DARK,
                ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", color="#9ca3af", lw=0.8),
            )

        # Center text
        ax.text(0, 0, f"{_fmt(total)}\nvisas", ha="center", va="center",
                fontsize=11, fontweight="800", color=DARK)
        ax.set_title(title, fontsize=12, fontweight="700", color=DARK, pad=16)

    fig.tight_layout(pad=2.0)
    st.pyplot(fig)
    plt.close(fig)

    # --- Stacked bar: MOHHO allocation by category ---
    st.markdown('<div class="section-head">Asignacion MOHHO por Categoria (Desglose)</div>',
                unsafe_allow_html=True)

    # Horizontal stacked bar for readability
    fig, ax = plt.subplots(figsize=(12, 10))
    y = np.arange(len(countries))
    left = np.zeros(len(countries))
    for j, cat in enumerate(categories):
        vals = np.array([mohho_matrix[i][j] for i in range(len(countries))])
        ax.barh(y, vals, 0.65, left=left, label=cat, color=CAT_COLORS[j],
                edgecolor="white", linewidth=0.8)
        # Label segments > 1000
        for i, v in enumerate(vals):
            if v > 1000:
                ax.text(left[i] + v / 2, i, _fmt(int(v)),
                        ha="center", va="center", fontsize=6.5, fontweight="600",
                        color="white")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(countries, fontsize=9, fontweight="500")
    ax.set_xlabel("Visas asignadas", fontsize=11, fontweight="600")
    ax.invert_yaxis()
    _style_ax(ax, "Visas MOHHO desglosadas por categoria EB")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000):,}k".replace(",", ".")))
    ax.grid(True, alpha=0.15, axis="x")
    ax.grid(False, axis="y")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Side-by-side heatmaps ---
    st.markdown('<div class="section-head">Heatmaps de Asignacion</div>',
                unsafe_allow_html=True)

    # One heatmap at a time for readability
    for matrix, title, cmap in [
        (fifo_matrix, "Sist. Actual (FIFO)", "Reds"),
        (mohho_matrix, f"MOHHO (disp.={mohho_fit[1]:.1f}a)", "Greens"),
    ]:
        mat = np.array(matrix)
        fig, ax = plt.subplots(figsize=(9, 10))
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels([CAT_LABELS.get(c, c) for c in categories], fontsize=8, fontweight="600")
        ax.set_yticks(range(len(countries)))
        ax.set_yticklabels(countries, fontsize=9)
        vmax = mat.max() if mat.max() > 0 else 1
        for i in range(len(countries)):
            for j in range(len(categories)):
                v = int(mat[i][j])
                clr = "white" if v > vmax * 0.45 else DARK
                ax.text(j, i, _fmt(v), ha="center", va="center",
                        fontsize=8, color=clr, fontweight="600")
        ax.set_title(title, fontsize=13, fontweight="700", color=DARK, pad=12)
        plt.colorbar(im, ax=ax, shrink=0.65, pad=0.02)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --- Difference heatmap ---
    st.markdown('<div class="section-head">Heatmap de Diferencias (MOHHO - Sist. Actual)</div>',
                unsafe_allow_html=True)

    diff_mat = np.array(mohho_matrix) - np.array(fifo_matrix)
    fig, ax = plt.subplots(figsize=(9, 10))
    vabs = max(abs(diff_mat.min()), abs(diff_mat.max())) or 1
    im = ax.imshow(diff_mat, cmap="RdYlGn", aspect="auto", vmin=-vabs, vmax=vabs)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([CAT_LABELS.get(c, c) for c in categories], fontsize=8, fontweight="600")
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries, fontsize=9)
    for i in range(len(countries)):
        for j in range(len(categories)):
            v = int(diff_mat[i][j])
            sign = "+" if v > 0 else ""
            ax.text(j, i, f"{sign}{_fmt(v)}" if v != 0 else "=",
                    ha="center", va="center", fontsize=8, fontweight="600")
    _style_ax(ax, "Verde = MOHHO asigna mas, Rojo = MOHHO asigna menos")
    plt.colorbar(im, ax=ax, shrink=0.65, pad=0.02)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Lollipop chart: Wait times ---
    st.markdown('<div class="section-head">Tiempo de Espera Maximo por Pais</div>',
                unsafe_allow_html=True)

    waits = data["country_wait"]
    sorted_waits = sorted(waits.items(), key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 9))
    y_pos = range(len(sorted_waits))
    for i, (c, w) in enumerate(sorted_waits):
        color = "#c62828" if w >= 10 else ("#e65100" if w >= 3 else "#00873c")
        ax.plot([0, w], [i, i], color=color, linewidth=2.5, zorder=2)
        ax.scatter(w, i, color=color, s=120, zorder=3, edgecolors="white", linewidths=2)
        ax.text(w + 0.3, i, f"{w} años", va="center", fontsize=10, fontweight="700", color=color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([c for c, _ in sorted_waits], fontsize=10, fontweight="500")
    ax.set_xlabel("Años de espera", fontsize=11, fontweight="600")
    _style_ax(ax, "Espera maxima por pais (fecha mas antigua)", fmt_x=True)
    ax.set_xlim(0, max(w for _, w in sorted_waits) + 3)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.15, axis="x")
    ax.grid(False, axis="y")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Radar chart: Top 6 countries ---
    st.markdown('<div class="section-head">Perfil por Categoria -- Top 6 Paises (Radar)</div>',
                unsafe_allow_html=True)

    # Pick top 6 by demand
    demand_sorted = sorted(range(len(countries)), key=lambda i: demand_by_c[i], reverse=True)[:6]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw=dict(polar=True))

    for ax, matrix, title in [
        (axes[0], fifo_matrix, "Sist. Actual (FIFO)"),
        (axes[1], mohho_matrix, "MOHHO"),
    ]:
        for ci in demand_sorted:
            vals = [matrix[ci][j] for j in range(len(categories))]
            mx = max(vals) if max(vals) > 0 else 1
            vals_norm = [v / mx for v in vals]
            vals_norm += vals_norm[:1]
            ax.plot(angles, vals_norm, linewidth=2.5, label=_short_name(countries[ci]),
                    color=COUNTRY_COLORS[ci])
            ax.fill(angles, vals_norm, alpha=0.1, color=COUNTRY_COLORS[ci])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([CAT_LABELS_SHORT.get(c, c) for c in categories], fontsize=9, fontweight="700")
        ax.set_title(title, fontsize=13, fontweight="700", color=DARK, pad=24)
        ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.35, 1.1),
                  framealpha=0.95)
        ax.set_yticklabels([])

    fig.tight_layout(pad=3.0)
    st.pyplot(fig)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# TAB 6: Convergencia
# ──────────────────────────────────────────────────────────────

def _tab_convergence(
    iters: list[int], means: list[float], stds: list[float],
    summary: dict, pareto: list, baseline: tuple,
) -> None:
    st.markdown('<div class="section-head">Convergencia del Hipervolumen</div>',
                unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    m = np.array(means)
    s = np.array(stds)
    it = np.array(iters)
    ax.plot(it, m, color=BLUE, linewidth=2.5, label="HV medio (30 corridas)")
    ax.fill_between(it, m - s, m + s, color=BLUE, alpha=0.12, label="  1 sigma")
    ax.axhline(y=m[-1], color=YELLOW, linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"HV final = {m[-1]:.2f}")
    # Mark 90% convergence point
    target_90 = m[-1] * 0.9
    for i, val in enumerate(m):
        if val >= target_90:
            ax.axvline(it[i], color="#e65100", linestyle=":", linewidth=1, alpha=0.5)
            ax.text(it[i] + 5, m.min(), f"90% en iter {it[i]}", fontsize=7,
                    color="#e65100", fontweight="600")
            break

    ax.set_xlabel("Iteracion", fontsize=10, fontweight="600")
    ax.set_ylabel("Hipervolumen", fontsize=10, fontweight="600")
    _style_ax(ax, "Convergencia del Indicador de Hipervolumen", fmt_x=True)
    ax.legend(fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # HV boxplot + run stats table
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="section-head">Distribucion HV Final</div>',
                    unsafe_allow_html=True)
        final_hvs = []
        for i in range(summary["num_runs"]):
            rd = load_run(i)
            final_hvs.append(rd["hv_final"])

        fig, ax = plt.subplots(figsize=(5, 4))
        bp = ax.boxplot(final_hvs, patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor(BLUE)
        bp["boxes"][0].set_alpha(0.5)
        bp["medians"][0].set_color(YELLOW)
        bp["medians"][0].set_linewidth(2.5)
        for flier in bp["fliers"]:
            flier.set(marker="o", markersize=4, alpha=0.5)
        ax.set_ylabel("Hipervolumen", fontsize=10, fontweight="600")
        ax.set_xticklabels(["30 corridas"], fontsize=9)
        _style_ax(ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown('<div class="section-head">Estadisticas por Corrida</div>',
                    unsafe_allow_html=True)
        run_stats = []
        for i in range(summary["num_runs"]):
            rd = load_run(i)
            run_stats.append({
                "Corrida": i + 1,
                "Seed": rd["seed"],
                "Pareto": rd["num_pareto"],
                "HV": round(rd["hv_final"], 2),
            })
        df = pd.DataFrame(run_stats)
        st.dataframe(df, use_container_width=True, height=350, hide_index=True)

    # Per-run HV bar chart
    st.markdown('<div class="section-head">Hipervolumen por Corrida</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    run_labels = [f"R{i+1}" for i in range(len(final_hvs))]
    mean_hv = np.mean(final_hvs)
    colors_bar = [BLUE if hv >= mean_hv else "#94a3b8" for hv in final_hvs]
    ax.bar(run_labels, final_hvs, color=colors_bar, edgecolor="white", linewidth=0.3)
    ax.axhline(mean_hv, color=YELLOW, linestyle="--", linewidth=1.5,
               label=f"Media = {mean_hv:.2f}")
    ax.set_ylabel("Hipervolumen", fontsize=10, fontweight="600")
    _style_ax(ax, "HV final de cada corrida")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", labelsize=6.5)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Individual run explorer
    st.markdown('<div class="section-head">Explorador de Corrida Individual</div>',
                unsafe_allow_html=True)
    run_idx = st.selectbox(
        "Seleccionar corrida",
        range(summary["num_runs"]),
        format_func=lambda x: f"Corrida {x + 1} (seed={42 + x})",
    )
    run_data = load_run(run_idx)

    fig, ax = plt.subplots(figsize=(9, 5))
    rf1 = [p["f1"] for p in run_data["pareto_front"]]
    rf2 = [p["f2"] for p in run_data["pareto_front"]]
    ax.scatter(rf1, rf2, c=BLUE, s=40, alpha=0.7, edgecolors="white", linewidths=0.5,
               label=f"Corrida {run_idx + 1} ({run_data['num_pareto']} sol.)")
    ax.scatter(baseline[0], baseline[1], c="#c62828", s=200, marker="*",
               zorder=5, label="Sist. Actual (FIFO)", edgecolors="white")
    ax.set_xlabel("Espera no atendida (años)", fontsize=10, fontweight="600")
    ax.set_ylabel("Disparidad entre paises (años)", fontsize=10, fontweight="600")
    _style_ax(ax, f"Corrida {run_idx + 1} (HV={run_data['hv_final']:.2f})", fmt_x=True)
    ax.legend(fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# TAB 7: Glosario
# ──────────────────────────────────────────────────────────────

def _tab_glossary() -> None:
    st.markdown('<div class="section-head">Glosario Completo</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:0.85rem;color:#6b7280 !important;margin-bottom:1rem;">
    Referencia de todos los terminos tecnicos del dashboard y la investigacion.
    </p>
    """, unsafe_allow_html=True)

    glossary = [
        ("FIFO (First In, First Out)", "Sistema / Politica",
         "El sistema actual del gobierno de EE.UU. para procesar solicitudes de Green Card. "
         "Las solicitudes se atienden estrictamente por orden de llegada: quien registro su fecha de prioridad "
         "primero, se procesa primero. Es simple y transparente, pero ignora el tiempo de espera acumulado "
         "y genera grandes disparidades entre paises con alta y baja demanda."),

        ("MOHHO", "Algoritmo",
         "Multi-Objective Harris Hawks Optimization. Algoritmo metaheuristico bio-inspirado basado en "
         "el comportamiento de caza cooperativa de los halcones de Harris. Busca soluciones que optimicen "
         "simultaneamente dos objetivos: reducir la espera y mejorar la equidad."),

        ("Green Card", "Inmigracion",
         "Tarjeta de Residencia Permanente de EE.UU. Permite vivir y trabajar permanentemente. "
         "Cada año fiscal se autorizan aprox. 140.000 visas de empleo (categorias EB-1 a EB-5)."),

        ("Frente de Pareto", "Optimizacion",
         "Conjunto de soluciones que no pueden mejorar un objetivo sin empeorar otro. "
         "Cada punto del frente representa un balance diferente entre espera y equidad. "
         "No existe una solucion unica 'mejor': el tomador de decisiones elige segun sus prioridades."),

        ("Dominancia de Pareto", "Optimizacion",
         "Una solucion A domina a B si A es al menos tan buena en todos los objetivos y estrictamente "
         "mejor en al menos uno. Las soluciones no dominadas forman el frente de Pareto."),

        ("Espera no atendida (Objetivo 1)", "Objetivo",
         "Mide el promedio ponderado de años de espera de los solicitantes que NO reciben visa. "
         "f1(x) = Sum[(n_g - x_g) * w_g] / Sum[n_g]. Minimizar prioriza a quienes mas tiempo esperan."),

        ("Disparidad entre paises (Objetivo 2)", "Objetivo",
         "Mide la mayor diferencia de tiempo de espera promedio entre cualquier par de paises. "
         "f2 = max|W_c1 - W_c2|. Minimizarlo busca que ningun pais espere desproporcionadamente."),

        ("Visas desperdiciadas (Objetivo 3)", "Objetivo",
         "f3 = V - Sum(x_g). Minimiza visas no asignadas. "
         "El sistema actual FIFO desperdicia ~17.500 visas por interacciones entre limites por pais "
         "y por categoria. MOHHO puede lograr 100% de utilizacion al reordenar el procesamiento."),

        ("Hipervolumen (HV)", "Metrica",
         "Indicador de calidad del frente de Pareto. Mide el volumen del espacio objetivo dominado. "
         "Un HV mayor indica un frente mas cercano al optimo y con mejor dispersion."),

        ("SPV (Smallest Position Value)", "Codificacion",
         "Metodo de Bean (1994) para convertir vectores continuos en permutaciones. "
         "Cada halcon tiene un vector en [0,1]^G; el SPV ordena los indices generando una permutacion."),

        ("Decodificador Greedy", "Algoritmo",
         "Convierte una permutacion en una asignacion factible de visas. "
         "Procesa los grupos en orden y asigna la mayor cantidad posible respetando 5 restricciones."),

        ("Categorias EB (EB-1 a EB-5)", "Inmigracion",
         "EB-1: habilidades extraordinarias. EB-2: profesionales avanzados. "
         "EB-3: trabajadores calificados. EB-4: inmigrantes especiales (incl. SIV). "
         "EB-5: inversores. Cada categoria tiene un tope legal."),

        ("Spillover (Desbordamiento)", "Inmigracion",
         "Cuando una categoria no usa todas sus visas, el excedente pasa a la siguiente segun reglas del INA."),

        ("Limite por pais (7%)", "Inmigracion",
         "Ningun pais puede recibir mas del 7% del total de visas (25.620/año). "
         "Afecta especialmente a India y China."),

        ("Fecha de Prioridad", "Inmigracion",
         "La fecha en que se presento la solicitud. En FIFO, determina el orden de procesamiento."),

        ("Resto del Mundo", "Modelo",
         "Bloque agregado que representa a los ~190 paises no listados individualmente. "
         "Su limite efectivo se calcula como V menos lo consumido por los 20 paises explicitos."),

        ("Corrida (Run)", "Experimento",
         "Ejecucion independiente del algoritmo MOHHO con semilla aleatoria distinta. "
         "Se realizan 30 corridas para evaluar la robustez estadistica."),

        ("Convergencia", "Experimento",
         "Como evoluciona el hipervolumen a lo largo de las iteraciones. "
         "Una curva que se estabiliza indica buena calidad de soluciones."),

        ("Energia de Escape (E)", "HHO",
         "Controla la transicion entre exploracion (E >= 1) y explotacion (E < 1). "
         "Decrece de 2 a 0 durante la ejecucion."),

        ("Vuelo de Levy", "HHO",
         "Patron de movimiento con pasos largos ocasionales. Permite escapar de optimos locales."),

        ("Solucion Pareto", "Optimizacion",
         "Asignacion de visas perteneciente al frente de Pareto. Cada solucion es un compromiso "
         "diferente entre espera y equidad que el decisor puede elegir."),
    ]

    # Search filter
    search = st.text_input("Buscar termino...", key="glossary_search")

    for term, tag, desc in glossary:
        if search and search.lower() not in term.lower() and search.lower() not in desc.lower():
            continue
        st.markdown(f"""
        <div class="glossary-term">
            <div class="tag">{tag}</div>
            <h4>{term}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="MOHHO Visa Optimizer",
        page_icon="H",
        layout="wide",
    )
    _inject_css()
    _hero()

    st.markdown("""
    <div class="info-box">
    <strong>Sistema Actual (FIFO)</strong>: las solicitudes de Green Card se procesan por orden de llegada.
    <strong>MOHHO</strong>: nuestra optimizacion multi-objetivo redistribuye las mismas 140.000 visas
    para reducir la espera acumulada y la disparidad entre paises.
    </div>
    """, unsafe_allow_html=True)

    summary = load_summary()
    pareto, baseline = load_pareto_csv()
    iters, hv_means, hv_stds = load_convergence()
    data = compute_allocations()

    _kpi_cards(summary, baseline, pareto, data)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Resumen Ejecutivo",
        "Datos de Entrada",
        "Asignacion Completa",
        "Analisis de Objetivos",
        "Analisis por Pais",
        "Convergencia",
        "Glosario",
    ])

    with tab1:
        _tab_resumen(data, pareto, baseline, summary)
    with tab2:
        _tab_raw_data(data)
    with tab3:
        _tab_allocation(data, pareto, baseline)
    with tab4:
        _tab_objectives(data, pareto, baseline)
    with tab5:
        _tab_country_analysis(data, pareto)
    with tab6:
        _tab_convergence(iters, hv_means, hv_stds, summary, pareto, baseline)
    with tab7:
        _tab_glossary()


if __name__ == "__main__":
    main()
