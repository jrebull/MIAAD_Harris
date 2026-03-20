"""MOHHO Visa Predict AI — Dashboard Dali Edition.

21 country blocs x 5 EB categories = 105 groups.
6 tabs + sidebar. Full Plotly interactivity, dark glassmorphism theme.

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

# ── Dali palette ─────────────────────────────────────────────
UACJ_BLUE = "#003CA6"
UACJ_GOLD = "#FFD600"
HOPE = "#00E5A0"
SMOKE = "#E8E8E8"
DANGER = "#FF3366"
GRID_COLOR = "rgba(255,255,255,0.06)"

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


# ── CSS injection ────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #1b2838 100%) !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #0a0a1a 100%) !important;
    }
    [data-testid="stSidebar"] * { color: #E8E8E8 !important; }
    .main .block-container { max-width: 1300px; padding: 1rem 2rem; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

    .metric-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 24px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        text-align: center;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,60,166,0.2);
    }
    .hero-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.8rem; font-weight: 700; line-height: 1;
        background: linear-gradient(135deg, #FFD600, #00E5A0);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem; font-weight: 700;
        color: #FFD600 !important; letter-spacing: -0.02em;
        border-left: 4px solid #003CA6;
        padding-left: 16px; margin: 2rem 0 1rem;
    }
    .callout {
        border-left: 4px solid #FF3366;
        padding: 16px 20px; margin: 1.5rem 0;
        background: rgba(255,51,102,0.06);
        border-radius: 0 12px 12px 0;
        font-size: 1.05rem; color: #E8E8E8 !important;
        font-style: italic; line-height: 1.6;
    }
    .info-box {
        background: rgba(0,60,166,0.1);
        border-left: 4px solid #003CA6;
        padding: 12px 16px; border-radius: 0 8px 8px 0;
        margin: 0.8rem 0; font-size: 0.84rem; color: #B0B8C8 !important;
    }
    .info-box strong { color: #FFD600 !important; }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0; font-weight: 600; font-size: 0.82rem;
        padding: 8px 14px; background: rgba(255,255,255,0.04); color: #B0B8C8 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background: #003CA6 !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stTabs [data-baseweb="tab"][aria-selected="true"] * { color: #ffffff !important; }

    .gloss-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
    }
    .gloss-card h4 { color: #FFD600 !important; margin: 0 0 4px; font-size: 0.95rem; }
    .gloss-card p { color: #B0B8C8 !important; margin: 0; font-size: 0.85rem; line-height: 1.5; }
    .gloss-tag {
        display: inline-block; background: rgba(0,60,166,0.2); color: #70A0FF !important;
        font-size: 0.65rem; padding: 2px 8px; border-radius: 10px; font-weight: 600; margin-bottom: 4px;
    }
    .dark-table {
        width: 100%; border-collapse: collapse; font-size: 0.78rem;
        border-radius: 8px; overflow: hidden;
    }
    .dark-table thead th {
        background: rgba(0,60,166,0.3); color: #FFD600 !important;
        padding: 9px 6px; font-weight: 600; text-align: center; font-size: 0.72rem;
    }
    .dark-table tbody td {
        padding: 7px 5px; text-align: center;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        color: #E8E8E8 !important; font-size: 0.76rem;
    }
    .dark-table tbody tr:hover { background: rgba(0,60,166,0.1); }
    .sidebar-metric {
        background: rgba(255,255,255,0.04); border-radius: 10px;
        padding: 10px 14px; margin: 6px 0; text-align: center;
    }
    .sidebar-metric .label {
        font-size: 0.65rem; color: rgba(255,255,255,0.5) !important;
        text-transform: uppercase; letter-spacing: 0.3px; font-weight: 600;
    }
    .sidebar-metric .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem; font-weight: 700; color: #FFD600 !important;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0a1a; }
    ::-webkit-scrollbar-thumb { background: #003CA6; border-radius: 3px; }
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


# ── Plotly theme ─────────────────────────────────────────────
def _plotly_layout(**kw: Any) -> dict[str, Any]:
    base = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=SMOKE, size=12),
        margin=dict(l=60, r=30, t=50, b=50),
    )
    base.update(kw)
    return base


_GRID = dict(gridcolor=GRID_COLOR, zerolinecolor="rgba(255,255,255,0.1)")


# ── Tab 1: El Problema ──────────────────────────────────────
def _tab_problem(data: dict, summary: dict, pareto: list, baseline: tuple) -> None:
    # Hero title
    st.markdown("""
    <div style="text-align:center;padding:2rem 0 1rem;">
        <div style="font-size:3rem;font-weight:900;letter-spacing:-0.03em;">
            <span style="color:#E8E8E8;">\U0001f985 VISA </span>
            <span style="background:linear-gradient(135deg,#FFD600,#00E5A0);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                PREDICT AI</span>
        </div>
        <div style="color:rgba(255,255,255,0.5);font-size:0.95rem;margin-top:4px;">
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
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 24px 16px;
    }}
    .num {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.6rem; font-weight: 700; line-height: 1;
        background: linear-gradient(135deg, #FFD600, #00E5A0);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .lbl {{
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem; color: rgba(255,255,255,0.5);
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
        El sistema actual no distingue urgencia ni equidad — solo orden de llegada.
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
        <div class="metric-card" style="border-color:rgba(255,51,102,0.3);">
            <div style="font-size:0.75rem;color:#FF3366 !important;font-weight:700;text-transform:uppercase;">
                Sistema Actual (FIFO)</div>
            <div class="hero-number" style="font-size:2rem;
                background:linear-gradient(135deg,#FF3366,#ff6b6b);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                {_fmt(fifo_used)} visas</div>
            <div style="color:rgba(255,255,255,0.5);font-size:0.85rem;margin-top:8px;">
                Desperdicia <strong style="color:#FF3366;">{_fmt(fifo_waste)}</strong> visas
                ({fifo_pct}% utilizacion)<br>
                Disparidad: <strong style="color:#FF3366;">{_fmtd(baseline[1])} a\u00f1os</strong>
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-color:rgba(0,229,160,0.3);">
            <div style="font-size:0.75rem;color:#00E5A0 !important;font-weight:700;text-transform:uppercase;">
                MOHHO (Optimizado)</div>
            <div class="hero-number" style="font-size:2rem;">
                Hasta {_fmt(data['total_visas'])} visas</div>
            <div style="color:rgba(255,255,255,0.5);font-size:0.85rem;margin-top:8px;">
                Puede asignar <strong style="color:#00E5A0;">100%</strong> de las visas<br>
                Disparidad: <strong style="color:#00E5A0;">{_fmtd(best_f2[1])} a\u00f1os</strong>
                (-{_fmtd(f2_imp, 1)}%)
            </div>
        </div>""", unsafe_allow_html=True)

    # f3 explanation
    st.markdown(f"""
    <div style="background:rgba(0,60,166,0.08);border:1px solid rgba(0,60,166,0.2);
        border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
        <div style="font-size:0.95rem;font-weight:700;color:#FFD600 !important;margin-bottom:8px;">
            \u00bfPor que se desperdician visas?</div>
        <p style="font-size:0.85rem;color:#B0B8C8 !important;margin:0 0 8px;">
            FIFO procesa las solicitudes mas antiguas primero. India EB-2 y EB-3 tienen fechas muy
            antiguas, asi que se procesan primero y agotan el limite por pais de India (25.620).
            Cuando FIFO llega a India EB-1, India ya esta llena. La categoria EB-1 tiene plazas
            disponibles pero los paises con mayor demanda ya estan al tope.</p>
        <p style="font-size:0.85rem;color:#B0B8C8 !important;margin:0;">
            <strong style="color:#00E5A0;">MOHHO resuelve esto:</strong> al reordenar el procesamiento,
            puede asignar las {_fmt(data['total_visas'])} visas completas (100% utilizacion)
            mientras mejora la equidad.</p>
    </div>""", unsafe_allow_html=True)


# ── Tab 2: Frente de Pareto ──────────────────────────────────
def _tab_pareto(pareto: list, baseline: tuple, knee: tuple) -> None:
    st.markdown('<div class="section-title">Frente de Pareto Interactivo</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Cada punto representa una forma distinta de repartir las 140.000 visas.
    <strong>No existe una solucion "mejor"</strong> — el decisor elige el balance entre espera y equidad.
    </div>""", unsafe_allow_html=True)

    sorted_p = sorted(pareto, key=lambda x: x[0])
    f1s = [p[0] for p in sorted_p]
    f2s = [p[1] for p in sorted_p]
    best_f1 = min(pareto, key=lambda x: x[0])
    best_f2 = min(pareto, key=lambda x: x[1])

    fig = go.Figure()

    # Pareto front
    fig.add_trace(go.Scatter(
        x=f1s, y=f2s, mode='markers+lines',
        marker=dict(size=9, color=f2s,
                    colorscale=[[0, '#00E5A0'], [0.5, '#003CA6'], [1, '#FF3366']],
                    colorbar=dict(title="Disparidad<br>(a\u00f1os)", thickness=15, len=0.6),
                    line=dict(width=1, color='rgba(255,255,255,0.3)')),
        line=dict(width=1, color='rgba(0,60,166,0.3)'),
        hovertemplate=(
            '<b>Solucion MOHHO</b><br>'
            'f\u2081 (espera): %{x:.3f} a\u00f1os<br>'
            'f\u2082 (disparidad): %{y:.2f} a\u00f1os<extra></extra>'),
        name='Frente de Pareto',
    ))

    # FIFO baseline
    fig.add_trace(go.Scatter(
        x=[baseline[0]], y=[baseline[1]], mode='markers+text',
        marker=dict(size=20, color=DANGER, symbol='star',
                    line=dict(width=2, color='white')),
        text=['FIFO'], textposition='top center',
        textfont=dict(color=DANGER, size=11),
        hovertemplate='<b>Sistema Actual (FIFO)</b><br>'
                      'f\u2081: %{x:.3f}<br>f\u2082: %{y:.2f}<extra></extra>',
        name='FIFO (actual)',
    ))

    # Knee point
    fig.add_trace(go.Scatter(
        x=[knee[0]], y=[knee[1]], mode='markers',
        marker=dict(size=16, color=UACJ_GOLD, symbol='diamond',
                    line=dict(width=2, color='white')),
        hovertemplate='<b>Equilibrio (Knee Point)</b><br>'
                      'f\u2081: %{x:.3f}<br>f\u2082: %{y:.2f}<extra></extra>',
        name='Equilibrio',
    ))

    # Best f1 & best f2
    fig.add_trace(go.Scatter(
        x=[best_f1[0]], y=[best_f1[1]], mode='markers',
        marker=dict(size=14, color=UACJ_BLUE, symbol='triangle-up',
                    line=dict(width=2, color='white')),
        name=f'Mejor f\u2081 ({_fmtd(best_f1[0])})',
    ))
    fig.add_trace(go.Scatter(
        x=[best_f2[0]], y=[best_f2[1]], mode='markers',
        marker=dict(size=14, color=HOPE, symbol='square',
                    line=dict(width=2, color='white')),
        name=f'Mejor f\u2082 ({_fmtd(best_f2[1])})',
    ))

    # Zone annotations
    fig.add_annotation(x=best_f1[0], y=best_f1[1] + 0.5,
                       text="Zona humanitaria", showarrow=False,
                       font=dict(size=10, color='rgba(0,60,166,0.6)'))
    fig.add_annotation(x=best_f2[0], y=best_f2[1] - 0.5,
                       text="Zona de equidad", showarrow=False,
                       font=dict(size=10, color='rgba(0,229,160,0.6)'))

    fig.update_layout(**_plotly_layout(
        title=dict(text="Cada punto es un balance diferente entre espera y equidad",
                   font=dict(size=14)),
        xaxis=dict(title="f\u2081 — Carga de espera no atendida (a\u00f1os)", **_GRID),
        yaxis=dict(title="f\u2082 — Disparidad entre paises (a\u00f1os)", **_GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0.3)", font=dict(size=10)),
        height=550,
    ))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Stats cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.7rem;color:rgba(255,255,255,0.5);text-transform:uppercase;font-weight:600;">
                Mejor Humanitario (f\u2081)</div>
            <div style="font-family:'JetBrains Mono';font-size:1.5rem;color:#003CA6;font-weight:700;">
                {_fmtd(best_f1[0])} a\u00f1os</div>
            <div style="font-size:0.75rem;color:rgba(255,255,255,0.4);">
                f\u2082 = {_fmtd(best_f1[1])} a\u00f1os</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-color:rgba(255,214,0,0.3);">
            <div style="font-size:0.7rem;color:rgba(255,255,255,0.5);text-transform:uppercase;font-weight:600;">
                Equilibrio (Knee Point)</div>
            <div style="font-family:'JetBrains Mono';font-size:1.5rem;color:#FFD600;font-weight:700;">
                {_fmtd(knee[0])} / {_fmtd(knee[1])}</div>
            <div style="font-size:0.75rem;color:rgba(255,255,255,0.4);">
                f\u2081 / f\u2082 (a\u00f1os)</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.7rem;color:rgba(255,255,255,0.5);text-transform:uppercase;font-weight:600;">
                Mejor Equidad (f\u2082)</div>
            <div style="font-family:'JetBrains Mono';font-size:1.5rem;color:#00E5A0;font-weight:700;">
                {_fmtd(best_f2[1])} a\u00f1os</div>
            <div style="font-size:0.75rem;color:rgba(255,255,255,0.4);">
                f\u2081 = {_fmtd(best_f2[0])} a\u00f1os</div>
        </div>""", unsafe_allow_html=True)


# ── Tab 3: Asignacion ────────────────────────────────────────
def _tab_allocation(data: dict, pareto: list, baseline: tuple) -> None:
    st.markdown('<div class="section-title">Asignacion de Visas — MOHHO vs FIFO</div>',
                unsafe_allow_html=True)

    countries = data["countries"]
    categories = data["categories"]

    sorted_pareto = sorted(pareto, key=lambda x: x[1])
    sol_idx = st.slider(
        "Navegar soluciones Pareto (izq = equidad, der = espera)",
        0, len(sorted_pareto) - 1, len(sorted_pareto) // 2,
        key="alloc_slider",
    )
    sel = sorted_pareto[sol_idx]
    mohho_matrix, mohho_fit, mohho_used = compute_mohho_allocation(sel[0], sel[1])
    fifo_matrix = data["fifo_matrix"]

    st.markdown(f"""
    <div class="info-box">
        <strong>Solucion seleccionada:</strong>
        Espera = {_fmtd(sel[0])} a\u00f1os —
        Disparidad = {_fmtd(sel[1])} a\u00f1os —
        Visas MOHHO: {_fmt(mohho_used)} / {_fmt(data['total_visas'])}
        ({mohho_used * 100 // data['total_visas']}%)
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
        (col1, mohho_arr, f"MOHHO (f\u2082={_fmtd(mohho_fit[1])}a)", _build_hover(mohho_arr)),
        (col2, fifo_arr, f"FIFO (f\u2082={_fmtd(baseline[1])}a)", _build_hover(fifo_arr)),
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
                title=dict(text=title, font=dict(size=13, color=UACJ_GOLD)),
                height=650, margin=dict(l=110, r=20, t=50, b=80),
            ))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Difference heatmap
    st.markdown('<div class="section-title">Diferencias (MOHHO \u2212 FIFO)</div>',
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
def _tab_country(data: dict, pareto: list, knee: tuple) -> None:
    st.markdown('<div class="section-title">Impacto por Pais</div>',
                unsafe_allow_html=True)

    countries = data["countries"]

    mohho_matrix, mohho_fit, _ = compute_mohho_allocation(knee[0], knee[1])
    fifo_matrix = data["fifo_matrix"]

    fifo_by_c = [sum(fifo_matrix[i]) for i in range(len(countries))]
    mohho_by_c = [sum(mohho_matrix[i]) for i in range(len(countries))]
    impact = [mohho_by_c[i] - fifo_by_c[i] for i in range(len(countries))]

    # Sort by impact
    sorted_idx = sorted(range(len(countries)), key=lambda i: impact[i], reverse=True)
    sorted_countries = [f"{_flag(countries[i])} {countries[i]}" for i in sorted_idx]
    sorted_fifo = [fifo_by_c[i] for i in sorted_idx]
    sorted_mohho = [mohho_by_c[i] for i in sorted_idx]
    sorted_wait = [data["country_wait"].get(countries[i], 0) for i in sorted_idx]

    # Grouped bar
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sorted_countries, x=sorted_fifo, orientation='h',
        name='FIFO (actual)', marker_color=DANGER, opacity=0.7,
        hovertemplate='<b>%{y}</b><br>FIFO: %{x:,.0f} visas<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        y=sorted_countries, x=sorted_mohho, orientation='h',
        name='MOHHO (equilibrio)', marker_color=HOPE, opacity=0.7,
        hovertemplate='<b>%{y}</b><br>MOHHO: %{x:,.0f} visas<extra></extra>',
    ))
    fig.update_layout(**_plotly_layout(
        title=dict(text="MOHHO (equilibrio) vs FIFO — visas por pais", font=dict(size=14)),
        barmode='group',
        xaxis=dict(title="Visas asignadas", **_GRID),
        yaxis=dict(autorange='reversed'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0.3)"),
        height=max(500, len(countries) * 35),
    ))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Delta bars
    st.markdown('<div class="section-title">Cambio MOHHO \u2212 FIFO por Pais</div>',
                unsafe_allow_html=True)

    sorted_impact = [impact[i] for i in sorted_idx]
    colors = [HOPE if v >= 0 else DANGER for v in sorted_impact]

    fig2 = go.Figure(go.Bar(
        y=sorted_countries, x=sorted_impact, orientation='h',
        marker_color=colors,
        text=[f"+{_fmt(v)}" if v > 0 else _fmt(v) for v in sorted_impact],
        textposition='outside', textfont=dict(size=10),
        hovertemplate='<b>%{y}</b><br>Cambio: %{x:+,.0f} visas<extra></extra>',
    ))
    fig2.update_layout(**_plotly_layout(
        xaxis=dict(title="\u0394 Visas (MOHHO \u2212 FIFO)", zeroline=True,
                   zerolinecolor='rgba(255,255,255,0.3)',
                   gridcolor=GRID_COLOR),
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
    w_colors = [DANGER if w >= 10 else (UACJ_GOLD if w >= 3 else HOPE) for w in w_values]

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
        textposition='middle right', textfont=dict(size=10, color=SMOKE),
        showlegend=False,
        hovertemplate='<b>%{y}</b><br>Espera max: %{x} a\u00f1os<extra></extra>',
    ))
    fig3.update_layout(**_plotly_layout(
        xaxis=dict(title="A\u00f1os de espera", range=[0, max(w_values) + 4], **_GRID),
        yaxis=dict(autorange='reversed'),
        height=max(500, len(w_countries) * 30),
        margin=dict(l=140, r=80, t=30, b=50),
    ))
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})


# ── Tab 5: Convergencia ─────────────────────────────────────
def _tab_convergence(
    iters: list, means: list, stds: list,
    summary: dict, baseline: tuple, run_idx: int,
) -> None:
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
        fill='toself', fillcolor='rgba(0,60,166,0.15)',
        line=dict(width=0), name='\u00b11\u03c3', hoverinfo='skip',
    ))
    # mean line
    fig.add_trace(go.Scatter(
        x=it.tolist(), y=m.tolist(), mode='lines',
        line=dict(color=UACJ_BLUE, width=3), name='HV medio (30 corridas)',
        hovertemplate='Iteracion %{x}<br>HV: %{y:.2f}<extra></extra>',
    ))
    fig.add_hline(y=float(m[-1]), line_dash="dash", line_color=UACJ_GOLD,
                  annotation_text=f"HV final = {_fmtd(float(m[-1]))}")

    # 95% convergence marker
    target_95 = float(m[-1]) * 0.95
    for i, val in enumerate(m):
        if val >= target_95:
            fig.add_vline(x=int(it[i]), line_dash="dot", line_color=DANGER,
                          annotation_text=f"95% en iter {int(it[i])}")
            break

    fig.update_layout(**_plotly_layout(
        title=dict(text="Evolucion del Indicador de Hipervolumen", font=dict(size=14)),
        xaxis=dict(title="Iteracion", **_GRID),
        yaxis=dict(title="Hipervolumen", **_GRID),
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
            marker=dict(color=UACJ_BLUE, size=5),
            line=dict(color=UACJ_GOLD),
            fillcolor='rgba(0,60,166,0.3)', name='30 corridas',
        ))
        fig_box.update_layout(**_plotly_layout(
            yaxis=dict(title="Hipervolumen", **_GRID), height=350,
        ))
        st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown('<div class="section-title" style="font-size:1.1rem;">HV por Corrida</div>',
                    unsafe_allow_html=True)
        mean_hv = float(np.mean(final_hvs))
        colors = [UACJ_BLUE if hv >= mean_hv else 'rgba(255,255,255,0.15)' for hv in final_hvs]
        fig_bar = go.Figure(go.Bar(
            x=[f"R{i+1}" for i in range(len(final_hvs))],
            y=final_hvs, marker_color=colors,
            hovertemplate='Corrida %{x}<br>HV: %{y:.2f}<extra></extra>',
        ))
        fig_bar.add_hline(y=mean_hv, line_dash="dash", line_color=UACJ_GOLD,
                          annotation_text=f"Media = {_fmtd(mean_hv)}")
        fig_bar.update_layout(**_plotly_layout(
            yaxis=dict(title="Hipervolumen", **_GRID), height=350,
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
        marker=dict(size=8, color=UACJ_BLUE, line=dict(width=1, color='white')),
        name=f"Corrida {run_idx + 1} ({run_data['num_pareto']} sol.)",
        hovertemplate='f\u2081: %{x:.3f}<br>f\u2082: %{y:.2f}<extra></extra>',
    ))
    fig_run.add_trace(go.Scatter(
        x=[baseline[0]], y=[baseline[1]], mode='markers',
        marker=dict(size=18, color=DANGER, symbol='star',
                    line=dict(width=2, color='white')),
        name='FIFO (actual)',
    ))
    fig_run.update_layout(**_plotly_layout(
        title=dict(text=f"Corrida {run_idx + 1} — HV = {_fmtd(run_data['hv_final'])}",
                   font=dict(size=14)),
        xaxis=dict(title="f\u2081 (espera)", **_GRID),
        yaxis=dict(title="f\u2082 (disparidad)", **_GRID),
        legend=dict(bgcolor="rgba(0,0,0,0.3)"), height=400,
    ))
    st.plotly_chart(fig_run, use_container_width=True, config={"displayModeBar": False})


# ── Tab 6: Acerca del Modelo ────────────────────────────────
def _tab_about(summary: dict, data: dict) -> None:
    st.markdown('<div class="section-title">Flujo del Algoritmo MOHHO</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:center;gap:12px;flex-wrap:wrap;
        padding:1.5rem;background:rgba(255,255,255,0.02);border-radius:16px;
        border:1px solid rgba(255,255,255,0.06);">
        <div style="text-align:center;padding:12px 16px;background:rgba(0,60,166,0.15);border-radius:12px;">
            <div style="font-size:1.5rem;">\U0001f985</div>
            <div style="font-size:0.75rem;color:#FFD600;font-weight:700;">Halcon</div>
            <div style="font-size:0.65rem;color:rgba(255,255,255,0.4);">Vector continuo<br>[0,1]^G</div>
        </div>
        <div style="color:#FFD600;font-size:1.5rem;">\u2192</div>
        <div style="text-align:center;padding:12px 16px;background:rgba(0,60,166,0.15);border-radius:12px;">
            <div style="font-size:1.5rem;">\U0001f522</div>
            <div style="font-size:0.75rem;color:#FFD600;font-weight:700;">SPV</div>
            <div style="font-size:0.65rem;color:rgba(255,255,255,0.4);">Smallest Position<br>Value \u2192 permutacion</div>
        </div>
        <div style="color:#FFD600;font-size:1.5rem;">\u2192</div>
        <div style="text-align:center;padding:12px 16px;background:rgba(0,60,166,0.15);border-radius:12px;">
            <div style="font-size:1.5rem;">\u2699\ufe0f</div>
            <div style="font-size:0.75rem;color:#FFD600;font-weight:700;">Decoder</div>
            <div style="font-size:0.65rem;color:rgba(255,255,255,0.4);">Greedy + 5<br>restricciones</div>
        </div>
        <div style="color:#FFD600;font-size:1.5rem;">\u2192</div>
        <div style="text-align:center;padding:12px 16px;background:rgba(0,60,166,0.15);border-radius:12px;">
            <div style="font-size:1.5rem;">\U0001f4ca</div>
            <div style="font-size:0.75rem;color:#FFD600;font-weight:700;">f\u2081, f\u2082</div>
            <div style="font-size:0.65rem;color:rgba(255,255,255,0.4);">Evaluar<br>objetivos</div>
        </div>
        <div style="color:#FFD600;font-size:1.5rem;">\u2192</div>
        <div style="text-align:center;padding:12px 16px;background:rgba(0,229,160,0.15);border-radius:12px;">
            <div style="font-size:1.5rem;">\u2728</div>
            <div style="font-size:0.75rem;color:#00E5A0;font-weight:700;">Pareto</div>
            <div style="font-size:0.65rem;color:rgba(255,255,255,0.4);">Archivo<br>externo</div>
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
            f"<td style=\"font-family:'JetBrains Mono',monospace;color:#FFD600 !important;\">{value}</td>"
            f"<td style='color:rgba(255,255,255,0.5) !important;font-size:0.72rem;'>{desc}</td></tr>"
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
         "Tarjeta de Residencia Permanente de EE.UU. 140.000 visas de empleo anuales (EB-1 a EB-5)."),
        ("Frente de Pareto", "Optimizacion",
         "Conjunto de soluciones no dominadas. Cada punto es un balance diferente "
         "entre espera y equidad."),
        ("Dominancia de Pareto", "Optimizacion",
         "Una solucion A domina a B si A es al menos tan buena en todos los objetivos y "
         "estrictamente mejor en al menos uno."),
        ("f\u2081 — Espera no atendida", "Objetivo",
         "Promedio ponderado de anios de espera de quienes NO reciben visa. "
         "f\u2081(x) = \u03a3[(n_g - x_g) \u00b7 w_g] / \u03a3[n_g]. Minimizar prioriza a los mas antiguos."),
        ("f\u2082 — Disparidad entre paises", "Objetivo",
         "Maxima diferencia de espera promedio entre cualquier par de paises. "
         "f\u2082 = max|W_c1 - W_c2|. Minimizar busca equidad."),
        ("f\u2083 — Visas desperdiciadas", "Objetivo",
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
         "Ningun pais puede recibir mas del 7% del total (25.620/anio). "
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
    _inject_css()

    # Load data
    summary = load_summary()
    pareto, baseline = load_pareto_csv()
    iters, hv_means, hv_stds = load_convergence()
    data = compute_allocations()

    # Knee point (Bug 2 fix: replaces biased euclidean distance)
    knee = _find_knee(pareto)
    best_f2 = min(pareto, key=lambda x: x[1])
    f2_imp = ((baseline[1] - best_f2[1]) / baseline[1]) * 100

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0;">
            <div style="font-size:2rem;">\U0001f985</div>
            <div style="font-family:'JetBrains Mono';font-size:0.9rem;font-weight:700;
                color:#FFD600;">VISA PREDICT AI</div>
            <div style="font-size:0.65rem;color:rgba(255,255,255,0.3);margin-top:2px;">
                MOHHO Optimizer v2.0</div>
        </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("### Configuracion")

        run_idx = st.selectbox(
            "Corrida",
            range(summary["num_runs"]),
            format_func=lambda x: f"#{x+1} (seed={42+x})",
        )

        st.radio(
            "Escenario del frente",
            ["Mejor f\u2081 (humanitario)",
             "Equilibrio (knee point)",
             "Mejor f\u2082 (equidad)",
             "FIFO (sistema actual)"],
            index=1,
            key="scenario",
        )

        st.divider()
        st.markdown("### Metricas rapidas")

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
            <div class="label">Mejora vs FIFO (f\u2082)</div>
            <div class="value" style="color:#00E5A0 !important;">\u2212{_fmtd(f2_imp, 1)}%</div>
        </div>""", unsafe_allow_html=True)

        # Progress ring
        pct = int(f2_imp)
        dash = pct * 2.51
        st.markdown(f"""
        <div style="text-align:center;margin-top:1rem;">
            <svg width="100" height="100" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="40" fill="none"
                    stroke="rgba(255,255,255,0.06)" stroke-width="8"/>
                <circle cx="50" cy="50" r="40" fill="none"
                    stroke="#00E5A0" stroke-width="8"
                    stroke-dasharray="{dash:.1f} 251.3" stroke-linecap="round"
                    transform="rotate(-90 50 50)"/>
                <text x="50" y="45" text-anchor="middle" dominant-baseline="central"
                    font-family="JetBrains Mono" font-size="18" font-weight="700"
                    fill="#FFD600">{pct}%</text>
                <text x="50" y="62" text-anchor="middle" font-family="Inter"
                    font-size="8" fill="rgba(255,255,255,0.4)">mejora</text>
            </svg>
        </div>""", unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "\U0001f985 El Problema",
        "\U0001f4c8 Frente de Pareto",
        "\U0001f5fa\ufe0f Asignacion",
        "\U0001f30d Impacto por Pais",
        "\U0001f4c9 Convergencia",
        "\u2139\ufe0f Acerca del Modelo",
    ])

    with tab1:
        _tab_problem(data, summary, pareto, baseline)
    with tab2:
        _tab_pareto(pareto, baseline, knee)
    with tab3:
        _tab_allocation(data, pareto, baseline)
    with tab4:
        _tab_country(data, pareto, knee)
    with tab5:
        _tab_convergence(iters, hv_means, hv_stds, summary, baseline, run_idx)
    with tab6:
        _tab_about(summary, data)

    # Footer
    st.markdown("""
    <div style="text-align:center;padding:3rem 0 1rem;opacity:0.4;font-size:0.75rem;color:#E8E8E8;">
        <div style="font-size:1rem;font-weight:700;margin-bottom:4px;">
            Visa Predict AI</div>
        MIAAD \u00d7 UACJ 2026<br>
        Javier Rebull & Yazmin Flores<br>
        Optimizacion Inteligente — Mtro. Raul Gibran Porras Alaniz
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
