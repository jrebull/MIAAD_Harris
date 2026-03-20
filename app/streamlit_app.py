"""MOHHO Visa Predict AI — Dashboard Dali Edition.

21 country blocs x 5 EB categories = 105 groups.
7 tabs + sidebar. Full Plotly interactivity, 3 selectable themes.

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
import streamlit.components.v1 as components
import plotly.graph_objects as go

# ── Paths ────────────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR: str = os.path.join(BASE_DIR, "results")

# ── Category descriptions & country flags ────────────────────
CATS_DESC: dict[str, str] = {
    "EB-1": "Extraordinarios", "EB-2": "Profesionales",
    "EB-3": "Calificados", "EB-4": "Especiales/SIV", "EB-5": "Inversores",
}

FLAGS: dict[str, str] = {
    "India": "\U0001f1ee\U0001f1f3", "China": "\U0001f1e8\U0001f1f3",
    "Filipinas": "\U0001f1f5\U0001f1ed", "Mexico": "\U0001f1f2\U0001f1fd",
    "Afganistan": "\U0001f1e6\U0001f1eb", "Irak": "\U0001f1ee\U0001f1f6",
    "Corea del Sur": "\U0001f1f0\U0001f1f7", "Pakistan": "\U0001f1f5\U0001f1f0",
    "Iran": "\U0001f1ee\U0001f1f7", "Taiwan": "\U0001f1f9\U0001f1fc",
    "Brasil": "\U0001f1e7\U0001f1f7", "Canada": "\U0001f1e8\U0001f1e6",
    "Reino Unido": "\U0001f1ec\U0001f1e7", "Nigeria": "\U0001f1f3\U0001f1ec",
    "Japon": "\U0001f1ef\U0001f1f5", "Bangladesh": "\U0001f1e7\U0001f1e9",
    "Colombia": "\U0001f1e8\U0001f1f4", "Alemania": "\U0001f1e9\U0001f1ea",
    "Vietnam": "\U0001f1fb\U0001f1f3", "Etiopia": "\U0001f1ea\U0001f1f9",
    "Resto del Mundo": "\U0001f30d",
}

# ── Theme definitions ─────────────────────────────────────────
THEMES: dict[str, dict[str, str]] = {
    "Oscuro": {
        "bg1": "#0a0a1a", "bg2": "#0d1b2a", "bg3": "#1b2838",
        "sidebar_bg1": "#0d1b2a", "sidebar_bg2": "#0a0a1a",
        "text": "#E8E8E8", "text_muted": "rgba(255,255,255,0.5)",
        "text_subtle": "rgba(255,255,255,0.4)",
        "card_bg": "rgba(255,255,255,0.04)", "card_border": "rgba(255,255,255,0.08)",
        "card_hover_shadow": "rgba(0,60,166,0.25)",
        "accent1": "#003CA6", "accent2": "#FFD600", "accent3": "#00E5A0",
        "danger": "#FF3366", "info_bg": "rgba(0,60,166,0.1)",
        "tab_bg": "rgba(255,255,255,0.04)", "tab_text": "#B0B8C8",
        "tab_active": "#003CA6", "tab_active_text": "#ffffff",
        "grid": "rgba(255,255,255,0.06)", "zeroline": "rgba(255,255,255,0.1)",
        "table_head": "rgba(0,60,166,0.3)", "table_border": "rgba(255,255,255,0.05)",
        "table_hover": "rgba(0,60,166,0.1)",
        "gloss_bg": "rgba(255,255,255,0.03)", "gloss_border": "rgba(255,255,255,0.06)",
        "tag_bg": "rgba(0,60,166,0.2)", "tag_text": "#70A0FF",
        "scroll_track": "#0a0a1a", "scroll_thumb": "#003CA6",
        "callout_bg": "rgba(255,51,102,0.06)",
        "plotly_font": "#E8E8E8", "plotly_bg": "rgba(0,0,0,0)",
        "hero_grad_start": "#FFD600", "hero_grad_end": "#00E5A0",
        "sidebar_metric_bg": "rgba(255,255,255,0.04)",
        "real_tag": "#00E5A0", "est_tag": "#FFD600", "calc_tag": "#70A0FF",
    },
    "Azul UACJ": {
        "bg1": "#071428", "bg2": "#0c2244", "bg3": "#153366",
        "sidebar_bg1": "#0a1c3d", "sidebar_bg2": "#071428",
        "text": "#E8F0FF", "text_muted": "rgba(200,220,255,0.55)",
        "text_subtle": "rgba(200,220,255,0.4)",
        "card_bg": "rgba(80,140,255,0.07)", "card_border": "rgba(80,140,255,0.15)",
        "card_hover_shadow": "rgba(0,100,255,0.3)",
        "accent1": "#1565C0", "accent2": "#FFD600", "accent3": "#4FC3F7",
        "danger": "#FF5252", "info_bg": "rgba(21,101,192,0.15)",
        "tab_bg": "rgba(80,140,255,0.06)", "tab_text": "#90B4E0",
        "tab_active": "#1565C0", "tab_active_text": "#ffffff",
        "grid": "rgba(100,160,255,0.08)", "zeroline": "rgba(100,160,255,0.12)",
        "table_head": "rgba(21,101,192,0.35)", "table_border": "rgba(100,160,255,0.08)",
        "table_hover": "rgba(21,101,192,0.12)",
        "gloss_bg": "rgba(80,140,255,0.04)", "gloss_border": "rgba(80,140,255,0.1)",
        "tag_bg": "rgba(21,101,192,0.25)", "tag_text": "#64B5F6",
        "scroll_track": "#071428", "scroll_thumb": "#1565C0",
        "callout_bg": "rgba(255,82,82,0.08)",
        "plotly_font": "#E8F0FF", "plotly_bg": "rgba(0,0,0,0)",
        "hero_grad_start": "#FFD600", "hero_grad_end": "#4FC3F7",
        "sidebar_metric_bg": "rgba(80,140,255,0.06)",
        "real_tag": "#4FC3F7", "est_tag": "#FFD600", "calc_tag": "#90B4E0",
    },
    "Claro": {
        "bg1": "#F5F7FA", "bg2": "#E8ECF2", "bg3": "#DCE2EC",
        "sidebar_bg1": "#E0E5ED", "sidebar_bg2": "#D4DAE5",
        "text": "#1a1a2e", "text_muted": "rgba(26,26,46,0.6)",
        "text_subtle": "rgba(26,26,46,0.45)",
        "card_bg": "rgba(0,0,0,0.03)", "card_border": "rgba(0,0,0,0.08)",
        "card_hover_shadow": "rgba(0,60,166,0.12)",
        "accent1": "#003CA6", "accent2": "#B8860B", "accent3": "#00895E",
        "danger": "#D32F2F", "info_bg": "rgba(0,60,166,0.06)",
        "tab_bg": "rgba(0,0,0,0.03)", "tab_text": "#5A6577",
        "tab_active": "#003CA6", "tab_active_text": "#ffffff",
        "grid": "rgba(0,0,0,0.06)", "zeroline": "rgba(0,0,0,0.1)",
        "table_head": "rgba(0,60,166,0.1)", "table_border": "rgba(0,0,0,0.06)",
        "table_hover": "rgba(0,60,166,0.05)",
        "gloss_bg": "rgba(0,0,0,0.02)", "gloss_border": "rgba(0,0,0,0.06)",
        "tag_bg": "rgba(0,60,166,0.1)", "tag_text": "#003CA6",
        "scroll_track": "#F5F7FA", "scroll_thumb": "#003CA6",
        "callout_bg": "rgba(211,47,47,0.06)",
        "plotly_font": "#1a1a2e", "plotly_bg": "rgba(0,0,0,0)",
        "hero_grad_start": "#B8860B", "hero_grad_end": "#00895E",
        "sidebar_metric_bg": "rgba(0,0,0,0.03)",
        "real_tag": "#00895E", "est_tag": "#B8860B", "calc_tag": "#003CA6",
    },
}


def _get_theme() -> dict[str, str]:
    return THEMES[st.session_state.get("theme_name", "Oscuro")]


# ── CSS injection ────────────────────────────────────────────
def _inject_css() -> None:
    t = _get_theme()
    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(135deg, {t['bg1']} 0%, {t['bg2']} 50%, {t['bg3']} 100%) !important;
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {t['sidebar_bg1']} 0%, {t['sidebar_bg2']} 100%) !important;
    }}
    [data-testid="stSidebar"] * {{ color: {t['text']} !important; }}

    /* Selectbox & radio widgets — sidebar */
    [data-testid="stSidebar"] div[data-baseweb="select"] {{
        background: {t['sidebar_bg1']} !important;
        border: 1px solid {t['card_border']} !important;
        border-radius: 8px;
    }}
    [data-testid="stSidebar"] div[data-baseweb="select"] span,
    [data-testid="stSidebar"] div[data-baseweb="select"] svg {{
        color: {t['text']} !important;
        fill: {t['text']} !important;
    }}
    /* Selectbox dropdown menu (global — renders outside sidebar) */
    div[data-baseweb="popover"] {{
        background: {t['sidebar_bg1']} !important;
        border: 1px solid {t['card_border']} !important;
    }}
    div[data-baseweb="popover"] li {{
        color: {t['text']} !important;
        background: transparent !important;
    }}
    div[data-baseweb="popover"] li:hover {{
        background: {t['card_bg']} !important;
    }}
    div[data-baseweb="popover"] li[aria-selected="true"] {{
        background: {t['accent1']}22 !important;
    }}
    /* Slider widget */
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stRadio label span {{
        color: {t['text']} !important;
    }}
    .main .block-container {{ max-width: 1300px; padding: 1rem 2rem; }}
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; color: {t['text']} !important; }}

    .metric-card {{
        background: {t['card_bg']};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {t['card_border']};
        border-radius: 16px;
        padding: 24px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        text-align: center;
    }}
    .metric-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 40px {t['card_hover_shadow']};
    }}
    .hero-number {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.8rem; font-weight: 700; line-height: 1;
        background: linear-gradient(135deg, {t['hero_grad_start']}, {t['hero_grad_end']});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .section-title {{
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem; font-weight: 700;
        color: {t['accent2']} !important; letter-spacing: -0.02em;
        border-left: 4px solid {t['accent1']};
        padding-left: 16px; margin: 2rem 0 1rem;
    }}
    .callout {{
        border-left: 4px solid {t['danger']};
        padding: 16px 20px; margin: 1.5rem 0;
        background: {t['callout_bg']};
        border-radius: 0 12px 12px 0;
        font-size: 1.05rem; color: {t['text']} !important;
        font-style: italic; line-height: 1.6;
    }}
    .info-box {{
        background: {t['info_bg']};
        border-left: 4px solid {t['accent1']};
        padding: 12px 16px; border-radius: 0 8px 8px 0;
        margin: 0.8rem 0; font-size: 0.84rem; color: {t['tab_text']} !important;
    }}
    .info-box strong {{ color: {t['accent2']} !important; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; background: transparent; }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0; font-weight: 600; font-size: 0.82rem;
        padding: 8px 14px; background: {t['tab_bg']}; color: {t['tab_text']} !important;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{ background: {t['tab_active']} !important; }}
    .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stTabs [data-baseweb="tab"][aria-selected="true"] * {{ color: {t['tab_active_text']} !important; }}

    .gloss-card {{
        background: {t['gloss_bg']};
        border: 1px solid {t['gloss_border']};
        border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
    }}
    .gloss-card h4 {{ color: {t['accent2']} !important; margin: 0 0 4px; font-size: 0.95rem; }}
    .gloss-card p {{ color: {t['tab_text']} !important; margin: 0; font-size: 0.85rem; line-height: 1.5; }}
    .gloss-tag {{
        display: inline-block; background: {t['tag_bg']}; color: {t['tag_text']} !important;
        font-size: 0.65rem; padding: 2px 8px; border-radius: 10px; font-weight: 600; margin-bottom: 4px;
    }}
    .dark-table {{
        width: 100%; border-collapse: collapse; font-size: 0.78rem;
        border-radius: 8px; overflow: hidden;
    }}
    .dark-table thead th {{
        background: {t['table_head']}; color: {t['accent2']} !important;
        padding: 9px 6px; font-weight: 600; text-align: center; font-size: 0.72rem;
    }}
    .dark-table tbody td {{
        padding: 7px 5px; text-align: center;
        border-bottom: 1px solid {t['table_border']};
        color: {t['text']} !important; font-size: 0.76rem;
    }}
    .dark-table tbody tr:hover {{ background: {t['table_hover']}; }}
    .sidebar-metric {{
        background: {t['sidebar_metric_bg']}; border-radius: 10px;
        padding: 10px 14px; margin: 6px 0; text-align: center;
    }}
    .sidebar-metric .label {{
        font-size: 0.65rem; color: {t['text_muted']} !important;
        text-transform: uppercase; letter-spacing: 0.3px; font-weight: 600;
    }}
    .sidebar-metric .value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem; font-weight: 700; color: {t['accent2']} !important;
    }}

    .source-badge {{
        display: inline-block; padding: 2px 8px; border-radius: 8px;
        font-size: 0.68rem; font-weight: 700; letter-spacing: 0.3px;
    }}
    .source-real {{ background: {t['real_tag']}22; color: {t['real_tag']}; border: 1px solid {t['real_tag']}44; }}
    .source-est {{ background: {t['est_tag']}22; color: {t['est_tag']}; border: 1px solid {t['est_tag']}44; }}
    .source-calc {{ background: {t['calc_tag']}22; color: {t['calc_tag']}; border: 1px solid {t['calc_tag']}44; }}

    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    [data-testid="stHeader"] {{ background: transparent !important; }}
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: {t['scroll_track']}; }}
    ::-webkit-scrollbar-thumb {{ background: {t['scroll_thumb']}; border-radius: 3px; }}
    </style>""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────
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
    from src.problem import VisaProblem
    from src.baseline import run_baseline
    from src.config import COUNTRIES, CATEGORIES

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

    return {
        "fifo_matrix": fifo_matrix.tolist(),
        "fifo_fit": fifo_fit,
        "fifo_used": int(fifo_matrix.sum()),
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
    }


@st.cache_data
def compute_mohho_allocation(
    target_f1: float, target_f2: float,
) -> tuple[list[list[int]], tuple[float, float], int]:
    """Find MOHHO allocation closest to target fitness via Monte Carlo sampling."""
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

    return matrix.tolist(), best_fit, int(matrix.sum())


@st.cache_data
def compute_pareto_visas(
    pareto_keys: tuple[tuple[float, float], ...],
) -> dict[tuple[float, float], int]:
    """Estimate visas used for each Pareto point via one shared Monte Carlo pool."""
    from src.problem import VisaProblem
    from src.decoder import spv, decode
    from src.config import NUM_GROUPS

    problem = VisaProblem()
    rng = np.random.default_rng(42)

    # Build pool of (fit, used) samples
    pool: list[tuple[float, float, int]] = []
    for _ in range(80_000):
        hawk = rng.uniform(0, 1, size=NUM_GROUPS)
        perm = spv(hawk)
        alloc = decode(perm, problem.groups, problem.total_visas,
                       problem.country_caps, problem.category_caps)
        fit = problem.evaluate(alloc)
        used = sum(alloc.values())
        pool.append((fit[0], fit[1], used))

    # For each Pareto point find closest sample
    result: dict[tuple[float, float], int] = {}
    for pf1, pf2 in pareto_keys:
        best_d = float("inf")
        best_u = 0
        for sf1, sf2, su in pool:
            d = (sf1 - pf1) ** 2 + (sf2 - pf2) ** 2
            if d < best_d:
                best_d = d
                best_u = su
        result[(pf1, pf2)] = best_u
    return result


# ── Helpers ──────────────────────────────────────────────────
def _fmt(n: int) -> str:
    """Format integer with comma thousands separator."""
    if n == 0:
        return "0"
    return f"{int(n):,}"


def _fmtd(x: float, d: int = 2) -> str:
    """Format float with comma thousands and dot decimal."""
    return f"{x:,.{d}f}"


def _find_knee(pareto: list[tuple[float, float]]) -> tuple[float, float]:
    """Find knee point using max perpendicular distance from extreme-to-extreme line.

    Normalizes both axes to [0,1] to avoid scale bias (Bug 2 fix).
    """
    pts = sorted(pareto, key=lambda p: p[0])
    if len(pts) <= 2:
        return pts[len(pts) // 2]

    f1_vals = [p[0] for p in pts]
    f2_vals = [p[1] for p in pts]
    f1_min, f1_max = min(f1_vals), max(f1_vals)
    f2_min, f2_max = min(f2_vals), max(f2_vals)
    f1_range = f1_max - f1_min
    f2_range = f2_max - f2_min

    if f1_range < 1e-12 or f2_range < 1e-12:
        return pts[len(pts) // 2]

    norm = [((p[0] - f1_min) / f1_range, (p[1] - f2_min) / f2_range) for p in pts]
    p1n = np.array(norm[0])
    p2n = np.array(norm[-1])
    line_vec = p2n - p1n
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-12:
        return pts[len(pts) // 2]
    line_unit = line_vec / line_len

    max_dist = -1.0
    knee_idx = 0
    for i, pn in enumerate(norm):
        v = np.array(pn) - p1n
        proj = np.dot(v, line_unit)
        perp = v - proj * line_unit
        dist = np.linalg.norm(perp)
        if dist > max_dist:
            max_dist = dist
            knee_idx = i

    return pts[knee_idx]


def _short(name: str) -> str:
    """Shorten country name for chart labels."""
    shorts = {"Resto del Mundo": "RdM", "Corea del Sur": "Corea S.",
              "Reino Unido": "R. Unido", "Bangladesh": "Bangla."}
    return shorts.get(name, name[:10])


def _flag(name: str) -> str:
    return FLAGS.get(name, "")


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB + alpha (0-1) to rgba() string for Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Plotly theme ─────────────────────────────────────────────
def _plotly_layout(**kw: Any) -> dict[str, Any]:
    t = _get_theme()
    base = dict(
        plot_bgcolor=t["plotly_bg"],
        paper_bgcolor=t["plotly_bg"],
        font=dict(family="Inter, sans-serif", color=t["plotly_font"], size=12),
        margin=dict(l=60, r=30, t=50, b=50),
    )
    base.update(kw)
    return base


def _grid() -> dict[str, str]:
    t = _get_theme()
    return dict(gridcolor=t["grid"], zerolinecolor=t["zeroline"])


# ── Data provenance map ──────────────────────────────────────
# For each country, which EB categories have REAL demand data vs estimated
_REAL_DEMAND = {
    "India": {"EB-1", "EB-2", "EB-3", "EB-4", "EB-5"},
    "China": {"EB-1", "EB-2", "EB-3", "EB-4", "EB-5"},
    "Filipinas": {"EB-2", "EB-3"},
    "Mexico": {"EB-2", "EB-3"},
}
_EST_DEM_CATS = {"EB-1", "EB-5"}  # Synthetic demand for "Current" categories
_DHS_COUNTRIES = {
    "Afganistan", "Irak", "Corea del Sur", "Pakistan", "Iran", "Taiwan",
    "Brasil", "Canada", "Reino Unido", "Nigeria", "Japon", "Bangladesh",
    "Colombia", "Alemania", "Vietnam", "Etiopia",
}


def _demand_source(country: str, category: str) -> str:
    """Return 'REAL', 'EST' or 'EST-DEM' for a country-category demand cell."""
    if country in _REAL_DEMAND:
        if category in _REAL_DEMAND[country]:
            return "REAL"
        return "EST-DEM"
    if country in _DHS_COUNTRIES:
        return "EST"
    # Resto del Mundo
    if category in _EST_DEM_CATS:
        return "EST-DEM"
    return "EST"


# ── Tab 1: El Problema ──────────────────────────────────────
def _tab_problem(data: dict, summary: dict, pareto: list, baseline: tuple) -> None:
    t = _get_theme()

    # Hero title
    st.markdown(f"""
    <div style="text-align:center;padding:2rem 0 1rem;">
        <div style="font-size:3rem;font-weight:900;letter-spacing:-0.03em;">
            <span style="color:{t['text']};">\U0001f985 VISA </span>
            <span style="background:linear-gradient(135deg,{t['hero_grad_start']},{t['hero_grad_end']});
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                PREDICT AI</span>
        </div>
        <div style="color:{t['text_muted']};font-size:0.95rem;margin-top:4px;">
            Multi-Objective Harris Hawks Optimization para la Asignacion de Green Cards
        </div>
    </div>""", unsafe_allow_html=True)

    # Animated hero numbers
    max_wait = max(data["country_wait"].values())
    components.html(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@700;900&family=JetBrains+Mono:wght@700&display=swap');
    body {{ background: transparent; margin: 0; padding: 0; }}
    .cards {{ display:flex; gap:20px; justify-content:center; flex-wrap:wrap; padding:10px 0; }}
    .card {{
        flex:1; min-width:160px; max-width:240px; text-align:center;
        background: {t['card_bg']};
        border: 1px solid {t['card_border']};
        border-radius: 16px; padding: 24px 16px;
    }}
    .num {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.6rem; font-weight: 700; line-height: 1;
        background: linear-gradient(135deg, {t['hero_grad_start']}, {t['hero_grad_end']});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .lbl {{
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem; color: {t['text_muted']};
        text-transform: uppercase; letter-spacing: 0.5px;
        font-weight: 600; margin-top: 8px;
    }}
    </style>
    <div class="cards">
        <div class="card"><div class="num" id="n1">0</div><div class="lbl">Visas / a\u00f1o</div></div>
        <div class="card"><div class="num" id="n2">0</div><div class="lbl">En cola de espera</div></div>
        <div class="card"><div class="num" id="n3">0</div><div class="lbl">Pa\u00edses analizados</div></div>
        <div class="card"><div class="num" id="n4">0</div><div class="lbl">A\u00f1os m\u00e1x. espera</div></div>
    </div>
    <script>
    function anim(id, end, dur) {{
        const el = document.getElementById(id);
        let start = null;
        function step(ts) {{
            if (!start) start = ts;
            const p = Math.min((ts - start) / dur, 1);
            const eased = 1 - Math.pow(1 - p, 3);
            el.textContent = Math.floor(eased * end).toLocaleString('en-US');
            if (p < 1) requestAnimationFrame(step);
        }}
        requestAnimationFrame(step);
    }}
    anim('n1', {data['total_visas']}, 1500);
    anim('n2', {data['total_demand']}, 1500);
    anim('n3', {len(data['countries'])}, 800);
    anim('n4', {max_wait}, 1200);
    </script>
    """, height=160)

    # Emotional callout
    max_wait_country = max(data["country_wait"], key=data["country_wait"].get)
    st.markdown(f"""
    <div class="callout">
        Un solicitante de {_flag(max_wait_country)} {max_wait_country} que registro su peticion
        hace <strong>{max_wait} a\u00f1os</strong> aun espera su Green Card.
        El sistema actual no distingue urgencia ni equidad \u2014 solo orden de llegada.
    </div>""", unsafe_allow_html=True)

    # FIFO vs MOHHO comparison
    fifo_used = data["fifo_used"]
    fifo_waste = data["total_visas"] - fifo_used
    best_f2 = min(pareto, key=lambda x: x[1])
    f2_imp = ((baseline[1] - best_f2[1]) / baseline[1]) * 100
    fifo_pct = fifo_used * 100 // data["total_visas"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-color:{t['danger']}44;">
            <div style="font-size:0.75rem;color:{t['danger']} !important;font-weight:700;text-transform:uppercase;">
                Sistema Actual (FIFO)</div>
            <div class="hero-number" style="font-size:2rem;
                background:linear-gradient(135deg,{t['danger']},#ff6b6b);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                {_fmt(fifo_used)} visas</div>
            <div style="color:{t['text_muted']};font-size:0.85rem;margin-top:8px;">
                Desperdicia <strong style="color:{t['danger']};">{_fmt(fifo_waste)}</strong> visas
                ({fifo_pct}% utilizacion)<br>
                Disparidad: <strong style="color:{t['danger']};">{_fmtd(baseline[1])} a\u00f1os</strong>
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-color:{t['accent3']}44;">
            <div style="font-size:0.75rem;color:{t['accent3']} !important;font-weight:700;text-transform:uppercase;">
                MOHHO (Optimizado)</div>
            <div class="hero-number" style="font-size:2rem;">
                Hasta {_fmt(data['total_visas'])} visas</div>
            <div style="color:{t['text_muted']};font-size:0.85rem;margin-top:8px;">
                Puede asignar <strong style="color:{t['accent3']};">100%</strong> de las visas<br>
                Disparidad: <strong style="color:{t['accent3']};">{_fmtd(best_f2[1])} a\u00f1os</strong>
                (-{_fmtd(f2_imp, 1)}%)
            </div>
        </div>""", unsafe_allow_html=True)

    # f3 explanation
    st.markdown(f"""
    <div style="background:{t['info_bg']};border:1px solid {t['accent1']}33;
        border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
        <div style="font-size:0.95rem;font-weight:700;color:{t['accent2']} !important;margin-bottom:8px;">
            \u00bfPor que se desperdician visas?</div>
        <p style="font-size:0.85rem;color:{t['tab_text']} !important;margin:0 0 8px;">
            FIFO procesa las solicitudes mas antiguas primero. India EB-2 y EB-3 tienen fechas muy
            antiguas, asi que se procesan primero y agotan el limite por pais de India (25,620).
            Cuando FIFO llega a India EB-1, India ya esta llena. La categoria EB-1 tiene plazas
            disponibles pero los paises con mayor demanda ya estan al tope.</p>
        <p style="font-size:0.85rem;color:{t['tab_text']} !important;margin:0;">
            <strong style="color:{t['accent3']};">MOHHO resuelve esto:</strong> al reordenar el procesamiento,
            puede asignar las {_fmt(data['total_visas'])} visas completas (100% utilizacion)
            mientras mejora la equidad.</p>
    </div>""", unsafe_allow_html=True)


# ── Tab 2: Frente de Pareto ──────────────────────────────────
def _tab_pareto(pareto: list, baseline: tuple, knee: tuple,
                sel_point: tuple, sel_label: str,
                sel_used: int, total_visas: int,
                best_f1_point: tuple, best_f2_point: tuple) -> None:
    t = _get_theme()

    st.markdown('<div class="section-title">Frente de Pareto Interactivo</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Cada punto representa una forma distinta de repartir las 140,000 visas.
    <strong>No existe una solucion "mejor"</strong> \u2014 el decisor elige el balance entre espera y equidad.
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:flex;gap:2rem;padding:0.6rem 1rem;margin-bottom:0.5rem;
        background:{t['card_bg']};border-radius:10px;font-size:0.85rem;">
        <span><strong style="color:{t['accent1']};">f\u2081</strong>
        <span style="color:{t['text_muted']};"> = Carga de espera no atendida (a\u00f1os) \u2014 menor es mejor</span></span>
        <span><strong style="color:{t['accent3']};">f\u2082</strong>
        <span style="color:{t['text_muted']};"> = Disparidad entre pa\u00edses (a\u00f1os) \u2014 menor es mejor</span></span>
    </div>""", unsafe_allow_html=True)

    sorted_p = sorted(pareto, key=lambda x: x[0])
    f1s = [p[0] for p in sorted_p]
    f2s = [p[1] for p in sorted_p]

    # Compute visas for all Pareto solutions (cached)
    pareto_visas = compute_pareto_visas(tuple(sorted_p))
    visas_list = [pareto_visas.get(p, total_visas) for p in sorted_p]

    # Selected solution card
    sel_pct = sel_used * 100 // total_visas
    st.markdown(f"""
    <div class="metric-card" style="border-color:{t['accent2']}4D;margin-bottom:1rem;">
        <div style="display:flex;justify-content:center;align-items:center;gap:2.5rem;flex-wrap:wrap;">
            <div>
                <div style="font-size:0.65rem;color:{t['text_muted']};text-transform:uppercase;font-weight:600;">
                    Escenario activo: {sel_label}</div>
                <div style="font-family:'JetBrains Mono';font-size:2rem;color:{t['accent2']};font-weight:700;">
                    {_fmt(sel_used)} visas</div>
                <div style="font-size:0.8rem;color:{t['text_subtle']};">
                    de {_fmt(total_visas)} disponibles ({sel_pct}%)</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:0.65rem;color:{t['text_muted']};text-transform:uppercase;">f\u2081 espera</div>
                <div style="font-family:'JetBrains Mono';font-size:1.4rem;color:{t['accent1']};font-weight:700;">
                    {_fmtd(sel_point[0])}</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:0.65rem;color:{t['text_muted']};text-transform:uppercase;">f\u2082 disparidad</div>
                <div style="font-family:'JetBrains Mono';font-size:1.4rem;color:{t['accent3']};font-weight:700;">
                    {_fmtd(sel_point[1])}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    fig = go.Figure()
    # Custom hover with visas
    hover_texts = [
        f"<b>Solucion MOHHO</b><br>"
        f"f\u2081 (espera): {_fmtd(f1, 3)} a\u00f1os<br>"
        f"f\u2082 (disparidad): {_fmtd(f2, 2)} a\u00f1os<br>"
        f"Visas otorgadas: {_fmt(v)} ({v * 100 // total_visas}%)"
        for f1, f2, v in zip(f1s, f2s, visas_list)
    ]
    fig.add_trace(go.Scatter(
        x=f1s, y=f2s, mode='markers+lines',
        marker=dict(size=9, color=visas_list,
                    colorscale=[[0, t['danger']], [0.5, t['accent1']], [1, t['accent3']]],
                    colorbar=dict(title="Visas<br>otorgadas", thickness=15, len=0.6),
                    line=dict(width=1, color=t['text_subtle'])),
        line=dict(width=1, color=_rgba(t['accent1'], 0.3)),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_texts,
        name='Frente de Pareto',
    ))

    # FIFO baseline
    fifo_label = f"FIFO ({_fmtd(baseline[0], 2)}, {_fmtd(baseline[1], 2)})"
    fig.add_trace(go.Scatter(
        x=[baseline[0]], y=[baseline[1]], mode='markers+text',
        marker=dict(size=20, color=t['danger'], symbol='star',
                    line=dict(width=2, color='white')),
        text=[fifo_label], textposition='middle right',
        textfont=dict(color=t['danger'], size=11),
        hovertemplate='<b>Sistema Actual (FIFO)</b><br>'
                      'f\u2081 espera: %{x:.3f} a\u00f1os<br>'
                      'f\u2082 disparidad: %{y:.2f} a\u00f1os<extra></extra>',
        name='FIFO (actual)',
    ))

    # Knee point
    fig.add_trace(go.Scatter(
        x=[knee[0]], y=[knee[1]], mode='markers+text',
        marker=dict(size=16, color=t['accent2'], symbol='diamond',
                    line=dict(width=2, color='white')),
        text=['Equilibrio'], textposition='bottom center',
        textfont=dict(color=t['accent2'], size=10),
        hovertemplate='<b>Knee Point (Equilibrio)</b><br>'
                      'f\u2081 espera: %{x:.3f} a\u00f1os<br>'
                      'f\u2082 disparidad: %{y:.2f} a\u00f1os<extra></extra>',
        name='Equilibrio (Knee)',
    ))

    # Selected point highlight (driven by sidebar scenario)
    if sel_point != baseline and sel_point != knee:
        fig.add_trace(go.Scatter(
            x=[sel_point[0]], y=[sel_point[1]], mode='markers',
            marker=dict(size=22, color='rgba(0,0,0,0)', symbol='circle',
                        line=dict(width=3, color=t['accent2'])),
            hovertemplate='<b>Seleccion activa</b><br>'
                          'f\u2081 espera: %{x:.3f} a\u00f1os<br>'
                          'f\u2082 disparidad: %{y:.2f} a\u00f1os<extra></extra>',
            name=f'Seleccion ({sel_label})',
        ))

    # Annotations
    fig.add_annotation(x=best_f1[0], y=best_f1[1] + 0.5,
                       text="Zona humanitaria", showarrow=False,
                       font=dict(size=10, color=_rgba(t['accent1'], 0.6)))
    fig.add_annotation(x=best_f2[0], y=best_f2[1] - 0.5,
                       text="Zona de equidad", showarrow=False,
                       font=dict(size=10, color=_rgba(t['accent3'], 0.6)))

    y_max = baseline[1] + 1.5
    y_min = min(f2s) - 0.5
    fig.update_layout(**_plotly_layout(
        title=dict(text="Cada punto es un balance diferente entre espera y equidad",
                   font=dict(size=14)),
        xaxis=dict(title="f\u2081 \u2014 Carga de espera no atendida (a\u00f1os)", **_grid()),
        yaxis=dict(title="f\u2082 \u2014 Disparidad entre paises (a\u00f1os)",
                   range=[y_min, y_max], **_grid()),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0.3)", font=dict(size=10)),
        height=550,
    ))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Stats cards with visas
    v_f1 = pareto_visas.get(best_f1_point, total_visas)
    v_knee = pareto_visas.get(knee, total_visas)
    v_f2 = pareto_visas.get(best_f2_point, total_visas)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.7rem;color:{t['text_muted']};text-transform:uppercase;font-weight:600;">
                Mejor Humanitario</div>
            <div style="font-family:'JetBrains Mono';font-size:1.3rem;color:{t['accent1']};font-weight:700;">
                {_fmtd(best_f1_point[0])} / {_fmtd(best_f1_point[1])} a\u00f1os</div>
            <div style="font-size:0.8rem;color:{t['text']};margin-top:4px;">
                <strong>{_fmt(v_f1)}</strong> visas ({v_f1 * 100 // total_visas}%)</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-color:{t['accent2']}4D;">
            <div style="font-size:0.7rem;color:{t['text_muted']};text-transform:uppercase;font-weight:600;">
                Equilibrio (Knee Point)</div>
            <div style="font-family:'JetBrains Mono';font-size:1.3rem;color:{t['accent2']};font-weight:700;">
                {_fmtd(knee[0])} / {_fmtd(knee[1])} a\u00f1os</div>
            <div style="font-size:0.8rem;color:{t['text']};margin-top:4px;">
                <strong>{_fmt(v_knee)}</strong> visas ({v_knee * 100 // total_visas}%)</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.7rem;color:{t['text_muted']};text-transform:uppercase;font-weight:600;">
                Mejor Equidad</div>
            <div style="font-family:'JetBrains Mono';font-size:1.3rem;color:{t['accent3']};font-weight:700;">
                {_fmtd(best_f2_point[0])} / {_fmtd(best_f2_point[1])} a\u00f1os</div>
            <div style="font-size:0.8rem;color:{t['text']};margin-top:4px;">
                <strong>{_fmt(v_f2)}</strong> visas ({v_f2 * 100 // total_visas}%)</div>
        </div>""", unsafe_allow_html=True)

    # Visas por solucion chart
    st.markdown('<div class="section-title">Visas Otorgadas por Solucion del Frente</div>',
                unsafe_allow_html=True)

    fig_v = go.Figure()
    bar_colors = [t['accent3'] if v >= total_visas * 0.95
                  else (t['accent2'] if v >= total_visas * 0.85 else t['danger'])
                  for v in visas_list]
    fig_v.add_trace(go.Bar(
        x=list(range(1, len(sorted_p) + 1)),
        y=visas_list,
        marker_color=bar_colors,
        hovertemplate=(
            'Solucion #%{x}<br>'
            'Visas: %{y:,.0f}<br>'
            f'f\u2081: %{{customdata[0]:.3f}} | f\u2082: %{{customdata[1]:.2f}}'
            '<extra></extra>'),
        customdata=list(zip(f1s, f2s)),
    ))
    fig_v.add_hline(y=total_visas, line_dash="dash", line_color=t['accent3'],
                    annotation_text=f"Maximo: {_fmt(total_visas)}")
    fifo_used_val = total_visas - (total_visas - pareto_visas.get(baseline, total_visas))
    fig_v.add_hline(y=pareto_visas.get(baseline, 0), line_dash="dot",
                    line_color=t['danger'],
                    annotation_text="FIFO")
    fig_v.update_layout(**_plotly_layout(
        xaxis=dict(title="Solucion Pareto (ordenada por f\u2081)", **_grid()),
        yaxis=dict(title="Visas otorgadas", **_grid()),
        height=350,
        margin=dict(l=60, r=30, t=40, b=50),
    ))
    st.plotly_chart(fig_v, use_container_width=True, config={"displayModeBar": False})


# ── Tab 3: Asignacion ────────────────────────────────────────
def _tab_allocation(data: dict, sel_matrix: list, sel_fit: tuple,
                    sel_used: int, sel_label: str, baseline: tuple) -> None:
    t = _get_theme()

    st.markdown('<div class="section-title">Asignacion de Visas \u2014 Escenario Activo</div>',
                unsafe_allow_html=True)

    countries = data["countries"]
    categories = data["categories"]
    fifo_matrix = data["fifo_matrix"]
    mohho_matrix = sel_matrix
    mohho_fit = sel_fit
    mohho_used = sel_used

    sel_pct = mohho_used * 100 // data["total_visas"]
    st.markdown(f"""
    <div class="info-box">
        <strong>Escenario: {sel_label}</strong> \u2014
        f\u2081 espera = {_fmtd(sel_fit[0])} a\u00f1os \u2014
        f\u2082 disparidad = {_fmtd(sel_fit[1])} a\u00f1os \u2014
        Visas otorgadas: <strong>{_fmt(mohho_used)} / {_fmt(data['total_visas'])}
        ({sel_pct}%)</strong>
    </div>""", unsafe_allow_html=True)

    cat_labels = [f"{c} {CATS_DESC.get(c, '')}" for c in categories]
    mohho_arr = np.array(mohho_matrix)
    fifo_arr = np.array(fifo_matrix)
    demand_arr = np.array(data["demand_matrix"])
    wait_arr = np.array(data["wait_matrix"])

    def _build_hover(mat: np.ndarray) -> list[list[str]]:
        hover = []
        for i in range(len(countries)):
            row = []
            for j in range(len(categories)):
                pct = mat[i][j] / demand_arr[i][j] * 100 if demand_arr[i][j] > 0 else 0
                row.append(
                    f"Pais: {countries[i]}<br>"
                    f"Categoria: {categories[j]} ({CATS_DESC.get(categories[j], '')})<br>"
                    f"Visas: {_fmt(int(mat[i][j]))}<br>"
                    f"Demanda: {_fmt(int(demand_arr[i][j]))}<br>"
                    f"Cobertura: {_fmtd(pct, 1)}%<br>"
                    f"Espera: {int(wait_arr[i][j])} a\u00f1os"
                )
            hover.append(row)
        return hover

    # Side-by-side heatmaps
    col1, col2 = st.columns(2)
    for col, mat, title, hover in [
        (col1, mohho_arr,
         f"{sel_label} \u2014 {_fmt(mohho_used)} visas (f\u2082={_fmtd(mohho_fit[1])}a)", _build_hover(mohho_arr)),
        (col2, fifo_arr,
         f"FIFO \u2014 {_fmt(data['fifo_used'])} visas (f\u2082={_fmtd(baseline[1])}a)", _build_hover(fifo_arr)),
    ]:
        with col:
            fig = go.Figure(go.Heatmap(
                z=mat, x=cat_labels, y=countries,
                colorscale='Viridis', showscale=True,
                text=[[_fmt(int(v)) for v in row] for row in mat],
                texttemplate="%{text}", textfont=dict(size=8, color='white'),
                hovertext=hover, hoverinfo='text',
            ))
            fig.update_layout(**_plotly_layout(
                title=dict(text=title, font=dict(size=13, color=t['accent2'])),
                height=650, margin=dict(l=110, r=20, t=50, b=80),
            ))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Difference heatmap
    st.markdown(f'<div class="section-title">Diferencias ({sel_label} \u2212 FIFO)</div>',
                unsafe_allow_html=True)
    diff_arr = mohho_arr - fifo_arr
    vabs = max(abs(int(diff_arr.min())), abs(int(diff_arr.max()))) or 1

    fig_diff = go.Figure(go.Heatmap(
        z=diff_arr, x=cat_labels, y=countries,
        colorscale='RdYlGn', zmid=0, zmin=-vabs, zmax=vabs,
        text=[[(f"+{_fmt(int(v))}" if int(v) > 0 else ("=" if int(v) == 0 else _fmt(int(v))))
               for v in row] for row in diff_arr],
        texttemplate="%{text}", textfont=dict(size=8),
        colorbar=dict(title="\u0394 Visas"),
    ))
    fig_diff.update_layout(**_plotly_layout(
        height=600, margin=dict(l=110, r=60, t=30, b=80),
    ))
    st.plotly_chart(fig_diff, use_container_width=True, config={"displayModeBar": False})

    # Data tables in expander
    with st.expander("Ver tablas numericas completas"):
        st.markdown("**Demanda pendiente (peticiones aprobadas)**")
        df_demand = pd.DataFrame(data["demand_matrix"],
                                 index=countries, columns=categories)
        df_demand["Total"] = df_demand.sum(axis=1)
        st.dataframe(df_demand, use_container_width=True)

        st.markdown("**Fechas de prioridad (Visa Bulletin)**")
        date_strs = []
        for i in range(len(countries)):
            row = []
            for j in range(len(categories)):
                d = data["date_matrix"][i][j]
                row.append("Current" if d >= 2026 else str(d))
            date_strs.append(row)
        df_dates = pd.DataFrame(date_strs, index=countries, columns=categories)
        st.dataframe(df_dates, use_container_width=True)


# ── Tab 4: Impacto por Pais ─────────────────────────────────
def _tab_country(data: dict, sel_matrix: list, sel_fit: tuple,
                 sel_used: int, sel_label: str,
                 baseline: tuple, knee: tuple,
                 best_f1: tuple, best_f2: tuple) -> None:
    t = _get_theme()

    st.markdown('<div class="section-title">Impacto por Pais</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        <strong>Escenario: {sel_label}</strong> \u2014
        Visas otorgadas: <strong>{_fmt(sel_used)} / {_fmt(data['total_visas'])}</strong> \u2014
        f\u2081 espera = {_fmtd(sel_fit[0])} \u2014
        f\u2082 disparidad = {_fmtd(sel_fit[1])}
    </div>""", unsafe_allow_html=True)

    countries = data["countries"]

    mohho_matrix = sel_matrix
    fifo_matrix = data["fifo_matrix"]

    fifo_by_c = [sum(fifo_matrix[i]) for i in range(len(countries))]
    mohho_by_c = [sum(mohho_matrix[i]) for i in range(len(countries))]
    impact = [mohho_by_c[i] - fifo_by_c[i] for i in range(len(countries))]

    # Sort by impact
    sorted_idx = sorted(range(len(countries)), key=lambda i: impact[i], reverse=True)
    sorted_countries = [f"{_flag(countries[i])} {countries[i]}" for i in sorted_idx]
    sorted_fifo = [fifo_by_c[i] for i in sorted_idx]
    sorted_mohho = [mohho_by_c[i] for i in sorted_idx]

    # Grouped bar
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sorted_countries, x=sorted_fifo, orientation='h',
        name='FIFO (actual)', marker_color=t['danger'], opacity=0.7,
        hovertemplate='<b>%{y}</b><br>FIFO: %{x:,.0f} visas<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        y=sorted_countries, x=sorted_mohho, orientation='h',
        name=f'{sel_label} (escenario)', marker_color=t['accent3'], opacity=0.7,
        hovertemplate=f'<b>%{{y}}</b><br>{sel_label}: %{{x:,.0f}} visas<extra></extra>',
    ))
    fig.update_layout(**_plotly_layout(
        title=dict(text=f"{sel_label} vs FIFO \u2014 visas por pais", font=dict(size=14)),
        barmode='group',
        xaxis=dict(title="Visas asignadas", **_grid()),
        yaxis=dict(autorange='reversed'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0.3)"),
        height=max(500, len(countries) * 35),
    ))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Delta bars
    st.markdown(f'<div class="section-title">Cambio {sel_label} \u2212 FIFO por Pais</div>',
                unsafe_allow_html=True)

    sorted_impact = [impact[i] for i in sorted_idx]
    colors = [t['accent3'] if v >= 0 else t['danger'] for v in sorted_impact]

    fig2 = go.Figure(go.Bar(
        y=sorted_countries, x=sorted_impact, orientation='h',
        marker_color=colors,
        text=[f"+{_fmt(v)}" if v > 0 else _fmt(v) for v in sorted_impact],
        textposition='outside', textfont=dict(size=10),
        hovertemplate='<b>%{y}</b><br>Cambio: %{x:+,.0f} visas<extra></extra>',
    ))
    fig2.update_layout(**_plotly_layout(
        xaxis=dict(title=f"\u0394 Visas ({sel_label} \u2212 FIFO)", zeroline=True,
                   zerolinecolor=t['zeroline'],
                   gridcolor=t['grid']),
        yaxis=dict(autorange='reversed'),
        height=max(450, len(countries) * 30),
        margin=dict(l=140, r=80, t=30, b=50),
    ))
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # Wait times lollipop
    st.markdown('<div class="section-title">Tiempo de Espera Maximo por Pais</div>',
                unsafe_allow_html=True)

    wait_sorted = sorted(data["country_wait"].items(), key=lambda x: x[1], reverse=True)
    w_countries = [f"{_flag(c)} {c}" for c, _ in wait_sorted]
    w_values = [w for _, w in wait_sorted]
    w_colors = [t['danger'] if w >= 10 else (t['accent2'] if w >= 3 else t['accent3']) for w in w_values]

    fig3 = go.Figure()
    for i, (c, w) in enumerate(zip(w_countries, w_values)):
        fig3.add_trace(go.Scatter(
            x=[0, w], y=[c, c], mode='lines',
            line=dict(color=w_colors[i], width=3),
            showlegend=False, hoverinfo='skip',
        ))
    fig3.add_trace(go.Scatter(
        x=w_values, y=w_countries, mode='markers+text',
        marker=dict(size=14, color=w_colors, line=dict(width=2, color='white')),
        text=[f"{w} a\u00f1os" for w in w_values],
        textposition='middle right', textfont=dict(size=10, color=t['text']),
        showlegend=False,
        hovertemplate='<b>%{y}</b><br>Espera max: %{x} a\u00f1os<extra></extra>',
    ))
    fig3.update_layout(**_plotly_layout(
        xaxis=dict(title="A\u00f1os de espera", range=[0, max(w_values) + 4], **_grid()),
        yaxis=dict(autorange='reversed'),
        height=max(500, len(w_countries) * 30),
        margin=dict(l=140, r=80, t=30, b=50),
    ))
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # ── Comparativa de los 4 escenarios ──
    show_all = st.checkbox("Ver todos los escenarios", key="show_all_scenarios")
    if show_all:
        _all_scenarios_chart(data, baseline, knee, best_f1, best_f2, countries)


def _all_scenarios_chart(data: dict, baseline: tuple, knee: tuple,
                         best_f1: tuple, best_f2: tuple,
                         countries: list[str]) -> None:
    """Grouped bar chart comparing all 4 scenarios side by side."""
    t = _get_theme()

    st.markdown('<div class="section-title">Comparativa: 4 Escenarios por Pa\u00eds</div>',
                unsafe_allow_html=True)

    scenarios = [
        ("FIFO", baseline, data["fifo_matrix"], data["fifo_used"]),
        ("Humanitario (f\u2081)", best_f1, None, 0),
        ("Equilibrio (knee)", knee, None, 0),
        ("Equidad (f\u2082)", best_f2, None, 0),
    ]
    # Compute allocations for MOHHO scenarios
    resolved = []
    for name, point, matrix, used in scenarios:
        if matrix is not None:
            by_c = [sum(matrix[i]) for i in range(len(countries))]
            resolved.append((name, point, by_c, used))
        else:
            mat, fit, u = compute_mohho_allocation(point[0], point[1])
            by_c = [sum(mat[i]) for i in range(len(countries))]
            resolved.append((name, fit, by_c, u))

    # Summary cards
    cols = st.columns(4)
    for col, (name, fit, by_c, used) in zip(cols, resolved):
        pct = used * 100 // data["total_visas"] if used > 0 else 0
        with col:
            st.markdown(f"""
            <div class="metric-card" style="padding:14px;">
                <div style="font-size:0.7rem;color:{t['text_muted']};text-transform:uppercase;
                    font-weight:600;">{name}</div>
                <div style="font-family:'JetBrains Mono';font-size:1.1rem;color:{t['accent2']};
                    font-weight:700;">{_fmt(used)} visas</div>
                <div style="font-size:0.72rem;color:{t['text_subtle']};">
                    f\u2081={_fmtd(fit[0])} | f\u2082={_fmtd(fit[1])}</div>
            </div>""", unsafe_allow_html=True)

    # Sorted by FIFO allocation
    fifo_by_c = resolved[0][2]
    sorted_idx = sorted(range(len(countries)), key=lambda i: fifo_by_c[i], reverse=True)
    sorted_countries = [f"{_flag(countries[i])} {countries[i]}" for i in sorted_idx]

    scenario_colors = [t['danger'], t['accent1'], t['accent2'], t['accent3']]

    fig = go.Figure()
    for idx, (name, _, by_c, _) in enumerate(resolved):
        sorted_vals = [by_c[i] for i in sorted_idx]
        fig.add_trace(go.Bar(
            y=sorted_countries, x=sorted_vals, orientation='h',
            name=name, marker_color=scenario_colors[idx], opacity=0.8,
            hovertemplate=f'<b>%{{y}}</b><br>{name}: %{{x:,.0f}} visas<extra></extra>',
        ))

    fig.update_layout(**_plotly_layout(
        title=dict(text="Visas otorgadas por pa\u00eds \u2014 4 escenarios",
                   font=dict(size=14)),
        barmode='group',
        xaxis=dict(title="Visas asignadas", **_grid()),
        yaxis=dict(autorange='reversed'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0.3)", font=dict(size=10)),
        height=max(600, len(countries) * 45),
        margin=dict(l=140, r=30, t=60, b=50),
    ))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Delta table: each scenario vs FIFO
    st.markdown('<div class="section-title">Diferencia vs FIFO por Pa\u00eds</div>',
                unsafe_allow_html=True)

    fig_delta = go.Figure()
    for idx in range(1, len(resolved)):  # skip FIFO itself
        name = resolved[idx][0]
        by_c = resolved[idx][2]
        deltas = [by_c[i] - fifo_by_c[i] for i in sorted_idx]
        fig_delta.add_trace(go.Bar(
            y=sorted_countries, x=deltas, orientation='h',
            name=name, marker_color=scenario_colors[idx], opacity=0.8,
            hovertemplate=f'<b>%{{y}}</b><br>{name} vs FIFO: %{{x:+,.0f}}<extra></extra>',
        ))

    fig_delta.update_layout(**_plotly_layout(
        barmode='group',
        xaxis=dict(title="\u0394 Visas vs FIFO", zeroline=True,
                   zerolinecolor=t['zeroline'], gridcolor=t['grid']),
        yaxis=dict(autorange='reversed'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0.3)", font=dict(size=10)),
        height=max(600, len(countries) * 45),
        margin=dict(l=140, r=30, t=60, b=50),
    ))
    st.plotly_chart(fig_delta, use_container_width=True, config={"displayModeBar": False})


# ── Tab 5: Convergencia ─────────────────────────────────────
def _tab_convergence(
    iters: list, means: list, stds: list,
    summary: dict, baseline: tuple, run_idx: int,
) -> None:
    t = _get_theme()

    st.markdown('<div class="section-title">Convergencia del Hipervolumen</div>',
                unsafe_allow_html=True)

    m = np.array(means)
    s = np.array(stds)
    it = np.array(iters)

    fig = go.Figure()
    # band
    fig.add_trace(go.Scatter(
        x=np.concatenate([it, it[::-1]]).tolist(),
        y=np.concatenate([m + s, (m - s)[::-1]]).tolist(),
        fill='toself', fillcolor=_rgba(t['accent1'], 0.15),
        line=dict(width=0), name='\u00b11\u03c3', hoverinfo='skip',
    ))
    # mean line
    fig.add_trace(go.Scatter(
        x=it.tolist(), y=m.tolist(), mode='lines',
        line=dict(color=t['accent1'], width=3), name='HV medio (30 corridas)',
        hovertemplate='Iteracion %{x}<br>HV: %{y:.2f}<extra></extra>',
    ))
    fig.add_hline(y=float(m[-1]), line_dash="dash", line_color=t['accent2'],
                  annotation_text=f"HV final = {_fmtd(float(m[-1]))}")

    # 95% convergence marker
    target_95 = float(m[-1]) * 0.95
    for i, val in enumerate(m):
        if val >= target_95:
            fig.add_vline(x=int(it[i]), line_dash="dot", line_color=t['danger'],
                          annotation_text=f"95% en iter {int(it[i])}")
            break

    fig.update_layout(**_plotly_layout(
        title=dict(text="Evolucion del Indicador de Hipervolumen", font=dict(size=14)),
        xaxis=dict(title="Iteracion", **_grid()),
        yaxis=dict(title="Hipervolumen", **_grid()),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0.3)"),
        height=450,
    ))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # HV distribution & per-run bars
    final_hvs = [load_run(i)["hv_final"] for i in range(summary["num_runs"])]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title" style="font-size:1.1rem;">Distribucion HV Final</div>',
                    unsafe_allow_html=True)
        fig_box = go.Figure(go.Box(
            y=final_hvs, boxpoints='all', jitter=0.3,
            marker=dict(color=t['accent1'], size=5),
            line=dict(color=t['accent2']),
            fillcolor=_rgba(t['accent1'], 0.3), name='30 corridas',
        ))
        fig_box.update_layout(**_plotly_layout(
            yaxis=dict(title="Hipervolumen", **_grid()), height=350,
        ))
        st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown('<div class="section-title" style="font-size:1.1rem;">HV por Corrida</div>',
                    unsafe_allow_html=True)
        mean_hv = float(np.mean(final_hvs))
        colors = [t['accent1'] if hv >= mean_hv else _rgba(t['text'], 0.15) for hv in final_hvs]
        fig_bar = go.Figure(go.Bar(
            x=[f"R{i+1}" for i in range(len(final_hvs))],
            y=final_hvs, marker_color=colors,
            hovertemplate='Corrida %{x}<br>HV: %{y:.2f}<extra></extra>',
        ))
        fig_bar.add_hline(y=mean_hv, line_dash="dash", line_color=t['accent2'],
                          annotation_text=f"Media = {_fmtd(mean_hv)}")
        fig_bar.update_layout(**_plotly_layout(
            yaxis=dict(title="Hipervolumen", **_grid()), height=350,
        ))
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # Individual run explorer
    st.markdown('<div class="section-title">Explorador de Corrida Individual</div>',
                unsafe_allow_html=True)

    run_data = load_run(run_idx)
    rf1 = [p["f1"] for p in run_data["pareto_front"]]
    rf2 = [p["f2"] for p in run_data["pareto_front"]]

    fig_run = go.Figure()
    fig_run.add_trace(go.Scatter(
        x=rf1, y=rf2, mode='markers',
        marker=dict(size=8, color=t['accent1'], line=dict(width=1, color='white')),
        name=f"Corrida {run_idx + 1} ({run_data['num_pareto']} sol.)",
        hovertemplate='f\u2081 espera: %{x:.3f} a\u00f1os<br>f\u2082 disparidad: %{y:.2f} a\u00f1os<extra></extra>',
    ))
    fig_run.add_trace(go.Scatter(
        x=[baseline[0]], y=[baseline[1]], mode='markers',
        marker=dict(size=18, color=t['danger'], symbol='star',
                    line=dict(width=2, color='white')),
        name='FIFO (actual)',
    ))
    fig_run.update_layout(**_plotly_layout(
        title=dict(text=f"Corrida {run_idx + 1} \u2014 HV = {_fmtd(run_data['hv_final'])}",
                   font=dict(size=14)),
        xaxis=dict(title="f\u2081 \u2014 Carga de espera no atendida (a\u00f1os)", **_grid()),
        yaxis=dict(title="f\u2082 \u2014 Disparidad entre paises (a\u00f1os)", **_grid()),
        legend=dict(bgcolor="rgba(0,0,0,0.3)"), height=400,
    ))
    st.plotly_chart(fig_run, use_container_width=True, config={"displayModeBar": False})


# ── Tab 6: Datos y Fuentes ───────────────────────────────────
def _tab_data_sources(data: dict) -> None:
    t = _get_theme()

    st.markdown('<div class="section-title">Transparencia de Datos</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        Este modelo combina datos de multiples fuentes oficiales. Algunos paises tienen datos completos
        publicados por USCIS; para otros, se estimaron las cifras a partir del DHS Yearbook.
        <strong>La transparencia sobre que es real y que es estimado es fundamental para la credibilidad del modelo.</strong>
    </div>""", unsafe_allow_html=True)

    # Legend
    st.markdown(f"""
    <div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin:1rem 0;padding:0.8rem 1rem;
        background:{t['card_bg']};border-radius:12px;border:1px solid {t['card_border']};">
        <div>
            <span class="source-badge source-real">REAL</span>
            <span style="font-size:0.8rem;color:{t['text_muted']};margin-left:6px;">
                Dato oficial de USCIS o Visa Bulletin</span>
        </div>
        <div>
            <span class="source-badge source-est">EST</span>
            <span style="font-size:0.8rem;color:{t['text_muted']};margin-left:6px;">
                Estimado (DHS Yearbook FY2023 / demanda sintetica)</span>
        </div>
        <div>
            <span class="source-badge source-calc">CALC</span>
            <span style="font-size:0.8rem;color:{t['text_muted']};margin-left:6px;">
                Calculado a partir de otros datos</span>
        </div>
    </div>""", unsafe_allow_html=True)

    countries = data["countries"]
    categories = data["categories"]

    # ── Demand provenance heatmap ──
    st.markdown(f'<div class="section-title" style="font-size:1.1rem;">Demanda por Pa\u00eds-Categor\u00eda: Origen del Dato</div>',
                unsafe_allow_html=True)

    # Build provenance matrix (0=REAL, 1=EST, 2=EST-DEM)
    prov_codes = {"REAL": 0, "EST": 1, "EST-DEM": 2}
    prov_labels_map = {0: "REAL", 1: "EST", 2: "EST-DEM"}
    prov_matrix = []
    prov_text = []
    for i, c in enumerate(countries):
        row_codes = []
        row_text = []
        for j, cat in enumerate(categories):
            src = _demand_source(c, cat)
            row_codes.append(prov_codes[src])
            demand_val = data["demand_matrix"][i][j]
            row_text.append(f"{src}<br>{_fmt(demand_val)}")
        prov_matrix.append(row_codes)
        prov_text.append(row_text)

    cat_labels = [f"{c} {CATS_DESC.get(c, '')}" for c in categories]

    # Discrete colorscale: REAL=teal, EST=coral, EST-DEM=slate blue
    c_real = "#0d9488"
    c_est = "#e07850"
    c_calc = "#6477b0"
    colorscale = [
        [0.0, c_real], [0.33, c_real],
        [0.34, c_est], [0.66, c_est],
        [0.67, c_calc], [1.0, c_calc],
    ]

    fig_prov = go.Figure(go.Heatmap(
        z=prov_matrix, x=cat_labels, y=countries,
        colorscale=colorscale, zmin=0, zmax=2,
        showscale=False,
        text=prov_text,
        texttemplate="%{text}",
        textfont=dict(size=9, color='white', family='Inter'),
        hovertemplate='<b>%{y}</b> | %{x}<br>%{text}<extra></extra>',
    ))

    # Manual legend for the 3 source types
    for label, color in [("REAL", c_real), ("EST", c_est), ("EST-DEM", c_calc)]:
        fig_prov.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=12, color=color, symbol='square'),
            name=label, showlegend=True,
        ))

    fig_prov.update_layout(**_plotly_layout(
        height=650, margin=dict(l=110, r=20, t=40, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0.3)", font=dict(size=11)),
    ))
    st.plotly_chart(fig_prov, use_container_width=True, config={"displayModeBar": False})

    # ── Full data table with source tags ──
    st.markdown(f'<div class="section-title" style="font-size:1.1rem;">Tabla Completa de Datos de Entrada</div>',
                unsafe_allow_html=True)

    rows_html = ""
    for i, c in enumerate(countries):
        flag = _flag(c)
        for j, cat in enumerate(categories):
            src = _demand_source(c, cat)
            demand = data["demand_matrix"][i][j]
            wait = data["wait_matrix"][i][j]
            date = data["date_matrix"][i][j]
            date_str = "Current" if date >= 2026 else str(date)

            src_class = "source-real" if src == "REAL" else ("source-est" if src == "EST" else "source-calc")

            rows_html += (
                f"<tr>"
                f"<td style='text-align:left;font-weight:500;'>{flag} {c}</td>"
                f"<td>{cat}</td>"
                f"<td style=\"font-family:'JetBrains Mono',monospace;\">{_fmt(demand)}</td>"
                f"<td><span class='source-badge {src_class}'>{src}</span></td>"
                f"<td>{date_str}</td>"
                f"<td><span class='source-badge source-real'>REAL</span></td>"
                f"<td style=\"font-family:'JetBrains Mono',monospace;\">{wait}</td>"
                f"<td><span class='source-badge source-calc'>CALC</span></td>"
                f"</tr>"
            )

    st.markdown(f"""
    <div style="max-height:500px;overflow-y:auto;border-radius:12px;border:1px solid {t['card_border']};">
    <table class="dark-table">
        <thead><tr>
            <th style="text-align:left;">Pa\u00eds</th>
            <th>Cat.</th>
            <th>Demanda</th>
            <th>Fuente</th>
            <th>Fecha Prioridad</th>
            <th>Fuente</th>
            <th>Espera (a\u00f1os)</th>
            <th>Fuente</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>""", unsafe_allow_html=True)

    # ── Sources detail ──
    st.markdown(f'<div class="section-title" style="font-size:1.1rem;">Fuentes Detalladas</div>',
                unsafe_allow_html=True)

    sources = [
        ("Demanda \u2014 India, China (todas las EB)",
         "REAL",
         "USCIS, \u201cApproved I-140/I-360/I-526 Petitions Awaiting Visa Availability,\u201d FY2025 Q2 (datos a marzo 2025)."),
        ("Demanda \u2014 Filipinas, Mexico (EB-2, EB-3)",
         "REAL",
         "Misma fuente USCIS FY2025 Q2. Solo EB-2 y EB-3 tienen datos directos."),
        ("Demanda \u2014 Filipinas, Mexico (EB-1, EB-5)",
         "EST-DEM",
         "Demanda sintetica estimada para categorias \u201cCurrent\u201d sin backlog visible."),
        ("Demanda \u2014 16 pa\u00edses derivados de \u201cAll Other\u201d",
         "EST",
         "DHS Yearbook FY2023, tablas de inmigracion calificada. Se uso la proporcion historica de cada pais."),
        ("Demanda \u2014 Resto del Mundo",
         "EST",
         "Residuo: total USCIS \u201cAll Other\u201d menos los 16 pa\u00edses estimados."),
        ("Fechas de prioridad (todas)",
         "REAL",
         "U.S. Department of State, Visa Bulletin abril 2026, \u201cFinal Action Dates for Employment-Based Preference Cases.\u201d"),
        ("Tiempo de espera (w)",
         "CALC",
         "w = T_actual (2026) \u2212 a\u00f1o de fecha de prioridad. Derivado automaticamente."),
        ("L\u00edmites por categor\u00eda (K_base)",
         "REAL",
         "INA Section 203(b). EB-1/2/3: 28.6% cada una; EB-4/5: 7.1% cada una de 140,000."),
        ("L\u00edmites efectivos (K_eff, spillover)",
         "CALC",
         "Cascada de spillover seg\u00fan INA: EB-4/5 \u2192 EB-1 \u2192 EB-2 \u2192 EB-3."),
        ("L\u00edmite por pa\u00eds (P_C = 25,620)",
         "REAL",
         "7% del total de visas (V_TOTAL = 366,000), INA Section 202(a)(2)."),
        ("Total visas EB (V = 140,000)",
         "REAL",
         "Asignacion estatutaria anual para visas de empleo."),
    ]

    for title, src_type, desc in sources:
        src_class = "source-real" if src_type == "REAL" else ("source-est" if src_type == "EST" else "source-calc")
        st.markdown(f"""
        <div class="gloss-card">
            <span class="source-badge {src_class}" style="margin-bottom:6px;">{src_type}</span>
            <h4 style="margin-top:4px;">{title}</h4>
            <p>{desc}</p>
        </div>""", unsafe_allow_html=True)


# ── Tab 7: Acerca del Modelo ────────────────────────────────
def _tab_about(summary: dict, data: dict) -> None:
    t = _get_theme()

    st.markdown('<div class="section-title">Flujo del Algoritmo MOHHO</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:center;gap:12px;flex-wrap:wrap;
        padding:1.5rem;background:{t['card_bg']};border-radius:16px;
        border:1px solid {t['card_border']};">
        <div style="text-align:center;padding:12px 16px;background:{t['accent1']}26;border-radius:12px;">
            <div style="font-size:1.5rem;">\U0001f985</div>
            <div style="font-size:0.75rem;color:{t['accent2']};font-weight:700;">Halcon</div>
            <div style="font-size:0.65rem;color:{t['text_subtle']};">Vector continuo<br>[0,1]^G</div>
        </div>
        <div style="color:{t['accent2']};font-size:1.5rem;">\u2192</div>
        <div style="text-align:center;padding:12px 16px;background:{t['accent1']}26;border-radius:12px;">
            <div style="font-size:1.5rem;">\U0001f522</div>
            <div style="font-size:0.75rem;color:{t['accent2']};font-weight:700;">SPV</div>
            <div style="font-size:0.65rem;color:{t['text_subtle']};">Smallest Position<br>Value \u2192 permutacion</div>
        </div>
        <div style="color:{t['accent2']};font-size:1.5rem;">\u2192</div>
        <div style="text-align:center;padding:12px 16px;background:{t['accent1']}26;border-radius:12px;">
            <div style="font-size:1.5rem;">\u2699\ufe0f</div>
            <div style="font-size:0.75rem;color:{t['accent2']};font-weight:700;">Decoder</div>
            <div style="font-size:0.65rem;color:{t['text_subtle']};">Greedy + 5<br>restricciones</div>
        </div>
        <div style="color:{t['accent2']};font-size:1.5rem;">\u2192</div>
        <div style="text-align:center;padding:12px 16px;background:{t['accent1']}26;border-radius:12px;">
            <div style="font-size:1.5rem;">\U0001f4ca</div>
            <div style="font-size:0.75rem;color:{t['accent2']};font-weight:700;">f\u2081, f\u2082</div>
            <div style="font-size:0.65rem;color:{t['text_subtle']};">f\u2081 espera<br>f\u2082 disparidad</div>
        </div>
        <div style="color:{t['accent2']};font-size:1.5rem;">\u2192</div>
        <div style="text-align:center;padding:12px 16px;background:{t['accent3']}26;border-radius:12px;">
            <div style="font-size:1.5rem;">\u2728</div>
            <div style="font-size:0.75rem;color:{t['accent3']};font-weight:700;">Pareto</div>
            <div style="font-size:0.65rem;color:{t['text_subtle']};">Archivo<br>externo</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Parameters table
    st.markdown('<div class="section-title">Parametros del Experimento</div>',
                unsafe_allow_html=True)

    params = [
        ("Poblacion", str(summary["pop_size"]), "Halcones por generacion"),
        ("Iteraciones", str(summary["max_iter"]), "Generaciones por corrida"),
        ("Archivo Pareto", str(summary["archive_size"]), "Capacidad del archivo externo"),
        ("Corridas", str(summary["num_runs"]), "Repeticiones independientes"),
        ("V (visas)", _fmt(data["total_visas"]), "Visas anuales disponibles"),
        ("G (grupos)", str(len(data["countries"]) * len(data["categories"])),
         "21 paises \u00d7 5 categorias"),
        ("Paises", str(len(data["countries"])), "20 explicitos + Resto del Mundo"),
        ("Limite pais", _fmt(25620), "7% del total (P_C)"),
    ]

    rows_html = ""
    for name, value, desc in params:
        rows_html += (
            f"<tr><td style='text-align:left;font-weight:600;'>{name}</td>"
            f"<td style=\"font-family:'JetBrains Mono',monospace;color:{t['accent2']} !important;\">{value}</td>"
            f"<td style='color:{t['text_muted']} !important;font-size:0.72rem;'>{desc}</td></tr>"
        )

    st.markdown(f"""
    <table class="dark-table">
        <thead><tr>
            <th style="text-align:left;">Parametro</th><th>Valor</th><th>Descripcion</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)

    # Glossary
    st.markdown('<div class="section-title">Glosario Interactivo</div>',
                unsafe_allow_html=True)

    search = st.text_input("Buscar termino...", key="glossary_search")

    glossary = [
        ("FIFO (First In, First Out)", "Sistema",
         "Sistema actual de EE.UU. para procesar Green Cards. Atiende por orden de llegada, "
         "ignorando el tiempo acumulado y generando disparidades entre paises."),
        ("MOHHO", "Algoritmo",
         "Multi-Objective Harris Hawks Optimization. Metaheuristica bio-inspirada basada en "
         "la caza cooperativa de halcones. Optimiza simultaneamente espera y equidad."),
        ("Green Card", "Inmigracion",
         "Tarjeta de Residencia Permanente de EE.UU. 140,000 visas de empleo anuales (EB-1 a EB-5)."),
        ("Frente de Pareto", "Optimizacion",
         "Conjunto de soluciones no dominadas. Cada punto es un balance diferente "
         "entre espera y equidad."),
        ("Dominancia de Pareto", "Optimizacion",
         "Una solucion A domina a B si A es al menos tan buena en todos los objetivos y "
         "estrictamente mejor en al menos uno."),
        ("f\u2081 \u2014 Espera no atendida", "Objetivo",
         "Promedio ponderado de anios de espera de quienes NO reciben visa. "
         "f\u2081(x) = \u03a3[(n_g - x_g) \u00b7 w_g] / \u03a3[n_g]. Minimizar prioriza a los mas antiguos."),
        ("f\u2082 \u2014 Disparidad entre paises", "Objetivo",
         "Maxima diferencia de espera promedio entre cualquier par de paises. "
         "f\u2082 = max|W_c1 - W_c2|. Minimizar busca equidad."),
        ("f\u2083 \u2014 Visas desperdiciadas", "Objetivo",
         f"f\u2083 = V - \u03a3x_g. FIFO desperdicia ~{_fmt(data['total_visas'] - data['fifo_used'])} "
         "visas por interacciones entre limites. MOHHO puede lograr 100% utilizacion."),
        ("Hipervolumen (HV)", "Metrica",
         "Indicador de calidad del frente de Pareto. Mide el volumen del espacio objetivo dominado. "
         "Mayor = mejor."),
        ("SPV (Smallest Position Value)", "Codificacion",
         "Metodo de Bean (1994) para convertir vectores continuos en permutaciones."),
        ("Decodificador Greedy", "Algoritmo",
         "Convierte una permutacion en asignacion factible respetando 5 restricciones."),
        ("Categorias EB (EB-1 a EB-5)", "Inmigracion",
         "EB-1: extraordinarios. EB-2: profesionales. EB-3: calificados. "
         "EB-4: especiales/SIV. EB-5: inversores."),
        ("Spillover (Desbordamiento)", "Inmigracion",
         "Cuando una categoria no usa todas sus visas, el excedente pasa a la siguiente segun reglas del INA."),
        ("Limite por pais (7%)", "Inmigracion",
         "Ningun pais puede recibir mas del 7% del total (25,620/anio). "
         "Afecta especialmente a India y China."),
        ("Knee Point (Punto de rodilla)", "Optimizacion",
         "Punto del frente de Pareto con maxima curvatura. Representa el mejor compromiso "
         "entre objetivos, calculado como el punto mas lejano de la linea entre los extremos."),
        ("Fecha de Prioridad", "Inmigracion",
         "La fecha en que se presento la solicitud. En FIFO, determina el orden de procesamiento."),
        ("Resto del Mundo", "Modelo",
         "Bloque agregado que representa a los ~190 paises no listados individualmente."),
        ("Convergencia", "Experimento",
         "Evolucion del hipervolumen a lo largo de las iteraciones. "
         "Estabilizacion indica buena calidad de soluciones."),
        ("Vuelo de Levy", "HHO",
         "Patron de movimiento con pasos largos ocasionales. Permite escapar de optimos locales."),
        ("Energia de Escape (E)", "HHO",
         "Controla la transicion entre exploracion (E >= 1) y explotacion (E < 1). "
         "Decrece de 2 a 0 durante la ejecucion."),
    ]

    for term, tag, desc in glossary:
        if search and search.lower() not in term.lower() and search.lower() not in desc.lower():
            continue
        st.markdown(f"""
        <div class="gloss-card">
            <div class="gloss-tag">{tag}</div>
            <h4>{term}</h4>
            <p>{desc}</p>
        </div>""", unsafe_allow_html=True)


# ── Main ─────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Visa Predict AI \u2014 MOHHO",
        page_icon="\U0001f985",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize theme
    if "theme_name" not in st.session_state:
        st.session_state["theme_name"] = "Oscuro"

    _inject_css()
    t = _get_theme()

    # Load data
    summary = load_summary()
    pareto, baseline = load_pareto_csv()
    iters, hv_means, hv_stds = load_convergence()
    data = compute_allocations()

    # Knee point (Bug 2 fix: replaces biased euclidean distance)
    knee = _find_knee(pareto)
    best_f1 = min(pareto, key=lambda x: x[0])
    best_f2 = min(pareto, key=lambda x: x[1])
    f2_imp = ((baseline[1] - best_f2[1]) / baseline[1]) * 100

    # ── Sidebar (single block to preserve tab state on rerun) ──
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:1rem 0;">
            <div style="font-size:2rem;">\U0001f985</div>
            <div style="font-family:'JetBrains Mono';font-size:0.9rem;font-weight:700;
                color:{t['accent2']};">VISA PREDICT AI</div>
            <div style="font-size:0.65rem;color:{t['text_subtle']};margin-top:2px;">
                MOHHO Optimizer v2.0</div>
        </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("### Configuracion")

        run_idx = st.selectbox(
            "Corrida",
            range(summary["num_runs"]),
            format_func=lambda x: f"#{x+1} (seed={42+x})",
        )

        scenario_labels = [
            "Mejor f\u2081 espera (humanitario)",
            "Equilibrio (knee point)",
            "Mejor f\u2082 disparidad (equidad)",
            "FIFO (sistema actual)",
        ]
        scenario = st.radio(
            "Escenario",
            scenario_labels,
            index=1,
            key="scenario",
        )

        # Resolve selected solution from scenario
        scenario_map = {
            scenario_labels[0]: best_f1,
            scenario_labels[1]: knee,
            scenario_labels[2]: best_f2,
            scenario_labels[3]: baseline,
        }
        sel_point = scenario_map[scenario]
        is_fifo = (scenario == scenario_labels[3])

        if is_fifo:
            sel_matrix = data["fifo_matrix"]
            sel_fit = data["fifo_fit"]
            sel_used = data["fifo_used"]
            sel_label = "FIFO"
        else:
            sel_matrix, sel_fit, sel_used = compute_mohho_allocation(
                sel_point[0], sel_point[1])
            sel_label = "MOHHO"

        st.divider()
        st.markdown("### Escenario activo")

        sel_pct = sel_used * 100 // data["total_visas"]
        st.markdown(f"""
        <div class="sidebar-metric">
            <div class="label">Visas otorgadas ({sel_label})</div>
            <div class="value" style="color:{t['accent3']} !important;">
                {_fmt(sel_used)} / {_fmt(data['total_visas'])}</div>
            <div class="label" style="margin-top:2px;">{sel_pct}% utilizacion</div>
        </div>
        <div class="sidebar-metric">
            <div class="label">f\u2081 espera (a\u00f1os)</div>
            <div class="value">{_fmtd(sel_fit[0])}</div>
        </div>
        <div class="sidebar-metric">
            <div class="label">f\u2082 disparidad (a\u00f1os)</div>
            <div class="value">{_fmtd(sel_fit[1])}</div>
        </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("### Metricas globales")

        st.markdown(f"""
        <div class="sidebar-metric">
            <div class="label">Soluciones Pareto</div>
            <div class="value">{summary['combined_pareto_size']}</div>
        </div>
        <div class="sidebar-metric">
            <div class="label">HV medio</div>
            <div class="value">{_fmtd(summary['hv_stats']['mean'])} \u00b1 {_fmtd(summary['hv_stats']['std'])}</div>
        </div>
        <div class="sidebar-metric">
            <div class="label">Mejora vs FIFO (f\u2082 disparidad)</div>
            <div class="value" style="color:{t['accent3']} !important;">\u2212{_fmtd(f2_imp, 1)}%</div>
        </div>""", unsafe_allow_html=True)

        # Progress ring
        pct = int(f2_imp)
        dash = pct * 2.51
        st.markdown(f"""
        <div style="text-align:center;margin-top:1rem;">
            <svg width="100" height="100" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="40" fill="none"
                    stroke="{t['card_border']}" stroke-width="8"/>
                <circle cx="50" cy="50" r="40" fill="none"
                    stroke="{t['accent3']}" stroke-width="8"
                    stroke-dasharray="{dash:.1f} 251.3" stroke-linecap="round"
                    transform="rotate(-90 50 50)"/>
                <text x="50" y="45" text-anchor="middle" dominant-baseline="central"
                    font-family="JetBrains Mono" font-size="18" font-weight="700"
                    fill="{t['accent2']}">{pct}%</text>
                <text x="50" y="62" text-anchor="middle" font-family="Inter"
                    font-size="8" fill="{t['text_muted']}">mejora</text>
            </svg>
        </div>""", unsafe_allow_html=True)

        # Theme selector — discreto al final
        st.divider()
        theme_icons = {"Oscuro": "\U0001f311", "Azul UACJ": "\U0001f535", "Claro": "\u2600\ufe0f"}
        theme_choice = st.selectbox(
            "Tema",
            list(THEMES.keys()),
            index=list(THEMES.keys()).index(st.session_state["theme_name"]),
            format_func=lambda x: f"{theme_icons[x]} {x}",
            key="theme_select",
        )
        if theme_choice != st.session_state["theme_name"]:
            st.session_state["theme_name"] = theme_choice
            st.rerun()

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "El Problema",
        "Frente de Pareto",
        "Asignacion",
        "Impacto por Pais",
        "Convergencia",
        "Datos y Fuentes",
        "Acerca del Modelo",
    ])

    with tab1:
        _tab_problem(data, summary, pareto, baseline)
    with tab2:
        _tab_pareto(pareto, baseline, knee, sel_point, sel_label,
                    sel_used, data["total_visas"], best_f1, best_f2)
    with tab3:
        _tab_allocation(data, sel_matrix, sel_fit, sel_used, sel_label, baseline)
    with tab4:
        _tab_country(data, sel_matrix, sel_fit, sel_used, sel_label,
                     baseline, knee, best_f1, best_f2)
    with tab5:
        _tab_convergence(iters, hv_means, hv_stds, summary, baseline, run_idx)
    with tab6:
        _tab_data_sources(data)
    with tab7:
        _tab_about(summary, data)

    # Footer
    st.markdown(f"""
    <div style="text-align:center;padding:3rem 0 1rem;opacity:0.4;font-size:0.75rem;color:{t['text']};">
        <div style="font-size:1rem;font-weight:700;margin-bottom:4px;">
            Visa Predict AI</div>
        MIAAD \u00d7 UACJ 2026<br>
        Javier Rebull & Yazmin Flores<br>
        Optimizacion Inteligente \u2014 Mtro. Raul Gibran Porras Alaniz
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
