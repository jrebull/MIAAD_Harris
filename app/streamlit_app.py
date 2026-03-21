"""MOHHO Visa Predict AI — Dashboard Dali Edition.

21 country blocs x 5 EB categories = 105 groups.
9 tabs + sidebar. Full Plotly interactivity, 3 selectable themes.

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
        "legend_bg": "rgba(0,0,0,0.3)", "marker_line": "#ffffff", "mid_blue": "#60a5fa",
    },
    "Azul UACJ": {
        "bg1": "#071428", "bg2": "#0c2244", "bg3": "#153366",
        "sidebar_bg1": "#0a1c3d", "sidebar_bg2": "#071428",
        "text": "#E8F0FF", "text_muted": "#8DA4C4",
        "text_subtle": "#6B89AD",
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
        "legend_bg": "rgba(7,20,40,0.7)", "marker_line": "#E8F0FF", "mid_blue": "#64B5F6",
    },
    "Claro": {
        "bg1": "#F0F2F6", "bg2": "#E8EBF0", "bg3": "#DFE3EA",
        "sidebar_bg1": "#DDE2EA", "sidebar_bg2": "#D2D8E2",
        "text": "#111827", "text_muted": "#4B5563",
        "text_subtle": "#6B7280",
        "card_bg": "rgba(255,255,255,0.85)", "card_border": "rgba(0,60,166,0.20)",
        "card_hover_shadow": "rgba(0,60,166,0.22)",
        "accent1": "#1E40AF", "accent2": "#B45309", "accent3": "#047857",
        "danger": "#B91C1C", "info_bg": "rgba(30,64,175,0.10)",
        "tab_bg": "rgba(0,60,166,0.08)", "tab_text": "#1F2937",
        "tab_active": "#1E40AF", "tab_active_text": "#ffffff",
        "grid": "rgba(0,0,0,0.10)", "zeroline": "rgba(0,0,0,0.15)",
        "table_head": "rgba(30,64,175,0.15)", "table_border": "rgba(0,0,0,0.10)",
        "table_hover": "rgba(30,64,175,0.08)",
        "gloss_bg": "rgba(255,255,255,0.9)", "gloss_border": "rgba(0,60,166,0.18)",
        "tag_bg": "rgba(30,64,175,0.12)", "tag_text": "#1E40AF",
        "scroll_track": "#F0F2F6", "scroll_thumb": "#1E40AF",
        "callout_bg": "rgba(185,28,28,0.08)",
        "plotly_font": "#111827", "plotly_bg": "rgba(255,255,255,0.6)",
        "hero_grad_start": "#92400E", "hero_grad_end": "#065F46",
        "sidebar_metric_bg": "rgba(30,64,175,0.08)",
        "real_tag": "#047857", "est_tag": "#B45309", "calc_tag": "#1E40AF",
        "legend_bg": "rgba(255,255,255,0.7)", "marker_line": "#1F2937", "mid_blue": "#3B82F6",
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
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
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
        box-shadow: 0 1px 6px rgba(0,0,0,0.04);
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
        border: 1px solid {t['card_border']};
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
    /* Ensure Streamlit native elements respect our theme */
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] .stMarkdown p,
    [data-testid="stAppViewContainer"] .stMarkdown li {{ color: {t['text']} !important; }}
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3 {{ color: {t['text']} !important; }}
    /* Tab content panel */
    [data-testid="stAppViewContainer"] [role="tabpanel"] {{ background: transparent !important; }}
    /* Plotly chart container bg */
    .stPlotlyChart {{ background: transparent !important; }}
    /* Dividers */
    [data-testid="stAppViewContainer"] hr {{ border-color: {t['card_border']} !important; }}
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: {t['scroll_track']}; }}
    ::-webkit-scrollbar-thumb {{ background: {t['scroll_thumb']}; border-radius: 3px; }}

    /* ── Native widget overrides for all themes ── */
    /* Expanders */
    [data-testid="stExpander"] {{
        background: {t['card_bg']} !important;
        border: 1px solid {t['card_border']} !important;
        border-radius: 12px !important;
    }}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary svg {{
        color: {t['text']} !important;
        fill: {t['text']} !important;
    }}
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {{
        color: {t['tab_text']} !important;
    }}
    /* Checkboxes */
    [data-testid="stCheckbox"] label span {{
        color: {t['text']} !important;
    }}
    /* Text inputs */
    [data-testid="stTextInput"] input {{
        background: {t['card_bg']} !important;
        color: {t['text']} !important;
        border-color: {t['card_border']} !important;
    }}
    /* Dataframes */
    [data-testid="stDataFrame"] {{
        background: {t['card_bg']} !important;
        border-radius: 8px;
    }}
    /* Radio buttons (main area) */
    [data-testid="stAppViewContainer"] .stRadio label span {{
        color: {t['text']} !important;
    }}
    /* Selectbox (main area) */
    [data-testid="stAppViewContainer"] div[data-baseweb="select"] {{
        background: {t['card_bg']} !important;
        border: 1px solid {t['card_border']} !important;
        border-radius: 8px;
    }}
    [data-testid="stAppViewContainer"] div[data-baseweb="select"] span,
    [data-testid="stAppViewContainer"] div[data-baseweb="select"] svg {{
        color: {t['text']} !important;
        fill: {t['text']} !important;
    }}
    /* Force theme on root .stApp — overrides config.toml defaults */
    .stApp {{
        background: linear-gradient(135deg, {t['bg1']} 0%, {t['bg2']} 50%, {t['bg3']} 100%) !important;
    }}
    .stApp > header {{
        background: transparent !important;
    }}
    [data-testid="stAppViewContainer"] > section,
    .stMainBlockContainer,
    section[data-testid="stMain"] {{
        background: transparent !important;
    }}
    /* Streamlit metric components */
    [data-testid="stMetric"] {{
        background: transparent !important;
    }}
    [data-testid="stMetric"] label {{
        color: {t['text_muted']} !important;
    }}
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {t['text']} !important;
    }}
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {{
        color: {t['accent3']} !important;
    }}
    /* Buttons */
    .stButton button {{
        color: {t['text']} !important;
        border-color: {t['card_border']} !important;
        background: {t['card_bg']} !important;
    }}
    .stButton button[kind="primary"],
    .stButton button[data-testid="stBaseButton-primary"] {{
        background-color: {t['accent1']} !important;
        color: white !important;
        border-color: {t['accent1']} !important;
    }}
    .stButton button:hover {{
        border-color: {t['accent1']} !important;
    }}
    /* Number input & text input */
    [data-testid="stNumberInput"] input,
    .stNumberInput input {{
        background: {t['card_bg']} !important;
        color: {t['text']} !important;
        border-color: {t['card_border']} !important;
    }}
    /* Slider labels & values */
    .stSlider label, .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"],
    .stSlider [data-testid="stThumbValue"] {{
        color: {t['text']} !important;
    }}
    /* Progress bar */
    .stProgress > div > div {{
        background-color: {t['accent1']} !important;
    }}
    .stProgress [data-testid="stProgressBarText"] {{
        color: {t['text']} !important;
    }}
    /* Toast / alert overrides */
    [data-testid="stNotification"] {{
        background: {t['card_bg']} !important;
        color: {t['text']} !important;
        border: 1px solid {t['card_border']} !important;
    }}
    /* --- Mobile responsive --- */
    @media (max-width: 768px) {{
        .metric-card {{ padding: 16px; }}
        .hero-number {{ font-size: 2rem !important; }}
        .section-title {{ font-size: 1.2rem; }}
        .dark-table {{ font-size: 0.7rem; }}
        .sidebar-metric .value {{ font-size: 1.1rem; }}
        .info-box {{ font-size: 0.85rem; padding: 12px; }}
        .gloss-card {{ padding: 12px; }}
    }}
    </style>""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────
@st.cache_data
def load_summary() -> dict[str, Any]:
    try:
        with open(os.path.join(RESULTS_DIR, "summary.json")) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"No se encontraron resultados. Ejecuta primero: python -m src.experiment\n\nDetalle: {e}")
        st.stop()


@st.cache_data
def load_pareto_csv() -> tuple[list[tuple[float, float, float]], tuple[float, float, float]]:
    path = os.path.join(RESULTS_DIR, "pareto_front.csv")
    pareto: list[tuple[float, float, float]] = []
    baseline: tuple[float, float, float] | None = None
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                pt = (float(row["f1"]), float(row["f2"]), float(row["f3"]))
                if row["type"] == "baseline":
                    baseline = pt
                else:
                    pareto.append(pt)
    except (FileNotFoundError, KeyError) as e:
        st.error(f"No se encontró pareto_front.csv. Ejecuta primero: python -m src.experiment\n\nDetalle: {e}")
        st.stop()
    return pareto, baseline


@st.cache_data
def load_convergence() -> tuple[list[int], list[float], list[float]]:
    iters, means, stds = [], [], []
    try:
        with open(os.path.join(RESULTS_DIR, "convergence.csv")) as f:
            for row in csv.DictReader(f):
                iters.append(int(row["iteration"]))
                means.append(float(row["hv_mean"]))
                stds.append(float(row["hv_std"]))
    except (FileNotFoundError, KeyError) as e:
        st.error(f"No se encontró convergence.csv. Ejecuta: python -m src.experiment\n\nDetalle: {e}")
        st.stop()
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
    target_f1: float, target_f2: float, target_f3: float = -1.0,
) -> tuple[list[list[int]], tuple[float, float, float], int]:
    """Find MOHHO allocation closest to target fitness via Monte Carlo sampling.

    Uses normalized distance across all 3 objectives so f₃ is properly matched.
    """
    from src.problem import VisaProblem
    from src.decoder import spv, decode
    from src.config import COUNTRIES, CATEGORIES, NUM_GROUPS

    problem = VisaProblem()
    rng = np.random.default_rng(42)
    best_alloc = None
    best_dist = float("inf")
    best_fit = (0.0, 0.0, 0.0)

    # Normalization ranges (from empirical Pareto front)
    r1, r2, r3 = 0.35, 10.0, 18000.0

    for _ in range(50_000):
        hawk = rng.uniform(0, 1, size=NUM_GROUPS)
        perm = spv(hawk)
        alloc = decode(perm, problem.groups, problem.total_visas,
                       problem.country_caps, problem.category_caps)
        fit = problem.evaluate(alloc)
        d = ((fit[0] - target_f1) / r1) ** 2 + ((fit[1] - target_f2) / r2) ** 2
        if target_f3 >= 0:
            d += ((fit[2] - target_f3) / r3) ** 2
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
    pareto_keys: tuple[tuple[float, float, float], ...],
    total_visas: int = 140_000,
) -> dict[tuple[float, float, float], int]:
    """Compute visas used for each Pareto point from f3 (visa waste = V - sum(x)).

    Since f3 is now stored directly in the Pareto CSV, visas_used = V - f3.
    """
    result: dict[tuple[float, float, float], int] = {}
    for pt in pareto_keys:
        f3 = pt[2]
        result[pt] = total_visas - int(f3)
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


def _find_knee(pareto: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Find knee point using max perpendicular distance from extreme-to-extreme line.

    Projects onto f1-f2 plane for knee detection, carries f3 along.
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
    "Filipinas": {"EB-2", "EB-3", "EB-4"},
    "Mexico": {"EB-2", "EB-3", "EB-4"},
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
            Multi-Objective Harris Hawks Optimization para la Asignación de Green Cards
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
        <div class="card"><div class="num" id="n1" aria-label="140,000 visas por a\u00f1o" role="text">0</div><div class="lbl">Visas / a\u00f1o</div></div>
        <div class="card"><div class="num" id="n2" aria-label="Peticiones en cola de espera" role="text">0</div><div class="lbl">En cola de espera</div></div>
        <div class="card"><div class="num" id="n3" aria-label="Pa\u00edses analizados" role="text">0</div><div class="lbl">Pa\u00edses analizados</div></div>
        <div class="card"><div class="num" id="n4" aria-label="A\u00f1os m\u00e1ximos de espera" role="text">0</div><div class="lbl">A\u00f1os m\u00e1x. espera</div></div>
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
        Un solicitante de {_flag(max_wait_country)} {max_wait_country} que registró su petición
        hace <strong>{max_wait} a\u00f1os</strong> aún espera su Green Card.
        El sistema actual no distingue urgencia ni equidad \u2014 solo orden de llegada.
    </div>""", unsafe_allow_html=True)

    # FIFO vs MOHHO comparison
    fifo_used = data["fifo_used"]
    fifo_waste = data["total_visas"] - fifo_used
    best_f2 = min(pareto, key=lambda x: x[1])
    best_f3 = min(pareto, key=lambda x: x[2])
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
                (f\u2083 = {_fmt(int(baseline[2]))}; {fifo_pct}% utilización)<br>
                Disparidad (brecha máx. de espera entre pa\u00edses):
                <strong style="color:{t['danger']};">{_fmtd(baseline[1])} a\u00f1os</strong>
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        best_f3_used = data['total_visas'] - int(best_f3[2])
        st.markdown(f"""
        <div class="metric-card" style="border-color:{t['accent3']}44;">
            <div style="font-size:0.75rem;color:{t['accent3']} !important;font-weight:700;text-transform:uppercase;">
                MOHHO (Optimizado)</div>
            <div class="hero-number" style="font-size:2rem;">
                Hasta {_fmt(best_f3_used)} visas</div>
            <div style="color:{t['text_muted']};font-size:0.85rem;margin-top:8px;">
                Mejor utilización: <strong style="color:{t['accent3']};">f\u2083 = {_fmt(int(best_f3[2]))}</strong>
                ({best_f3_used * 100 // data['total_visas']}% utilización)<br>
                Disparidad (brecha máx. de espera entre pa\u00edses):
                <strong style="color:{t['accent2']};">{_fmtd(best_f2[1])} a\u00f1os</strong>
                (-{_fmtd(f2_imp, 1)}%)
            </div>
        </div>""", unsafe_allow_html=True)

    # f3 explanation
    st.markdown(f"""
    <div style="background:{t['info_bg']};border:1px solid {t['accent1']}33;
        border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
        <div style="font-size:0.95rem;font-weight:700;color:{t['accent2']} !important;margin-bottom:8px;">
            \u00bfPor qué se desperdician visas?</div>
        <p style="font-size:0.85rem;color:{t['tab_text']} !important;margin:0 0 8px;">
            FIFO procesa las solicitudes más antiguas primero. India EB-2 y EB-3 tienen fechas muy
            antiguas, así que se procesan primero y agotan el límite por país de India (25,620).
            Cuando FIFO llega a India EB-1, India ya está llena. La categoría EB-1 tiene plazas
            disponibles pero los países con mayor demanda ya están al tope.</p>
        <p style="font-size:0.85rem;color:{t['tab_text']} !important;margin:0;">
            <strong style="color:{t['accent3']};">MOHHO resuelve esto:</strong> al reordenar el procesamiento,
            puede asignar las {_fmt(data['total_visas'])} visas completas (100% utilización)
            mientras mejora la equidad.</p>
    </div>""", unsafe_allow_html=True)

    # f3 hero card
    st.markdown(f"""
    <div class="metric-card" style="border-color:{t['accent3']}44;margin:1rem 0;">
        <div style="font-size:0.75rem;color:{t['accent3']} !important;font-weight:700;text-transform:uppercase;">
            Objetivo 3 (f\u2083): Desperdicio de Visas</div>
        <div style="display:flex;justify-content:center;gap:3rem;flex-wrap:wrap;margin:10px 0;">
            <div style="text-align:center;">
                <div style="font-size:0.65rem;color:{t['text_muted']};text-transform:uppercase;">FIFO desperdicia</div>
                <div style="font-family:'JetBrains Mono';font-size:1.5rem;color:{t['danger']};font-weight:700;">
                    {_fmt(int(baseline[2]))} visas</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:0.65rem;color:{t['text_muted']};text-transform:uppercase;">Mejor MOHHO (f\u2083)</div>
                <div style="font-family:'JetBrains Mono';font-size:1.5rem;color:{t['accent3']};font-weight:700;">
                    {_fmt(int(best_f3[2]))} visas</div>
            </div>
        </div>
        <div style="color:{t['text_muted']};font-size:0.82rem;text-align:center;">
            f\u2083 = V \u2212 \u03a3x<sub>g</sub> — visas disponibles que quedan sin asignar.
            Minimizar f\u2083 maximiza la utilización del presupuesto anual.</div>
    </div>""", unsafe_allow_html=True)

    # f1 and f2 explanations side by side
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown(f"""
        <div style="background:{t['info_bg']};border:1px solid {t['accent1']}33;
            border-radius:12px;padding:1.5rem;margin:1rem 0;height:100%;">
            <div style="font-size:0.95rem;font-weight:700;color:{t['accent1']} !important;margin-bottom:8px;">
                ¿Qué es la espera no atendida (f\u2081)?</div>
            <p style="font-size:0.85rem;color:{t['tab_text']} !important;margin:0 0 8px;">
                Para cada grupo de solicitantes, se mide cuántos <strong>no reciben visa</strong>
                y cu\u00e1ntos a\u00f1os llevan esperando. f\u2081 es el promedio ponderado:</p>
            <p style="font-family:'JetBrains Mono';font-size:0.85rem;color:{t['accent2']} !important;
                margin:0 0 8px;text-align:center;">
                f\u2081 = \u03a3 (n<sub>g</sub> \u2212 x<sub>g</sub>) \u00b7 w<sub>g</sub>
                &nbsp;/&nbsp; \u03a3 n<sub>g</sub></p>
            <p style="font-size:0.85rem;color:{t['tab_text']} !important;margin:0;">
                Ejemplo: si India EB-2 tiene 800,000 solicitantes esperando 13 a\u00f1os y solo
                recibe 25,000 visas, los 775,000 restantes contribuyen
                <strong>775,000 \u00d7 13 = 10M persona-a\u00f1os</strong> de carga sin atender.
                Minimizar f\u2081 prioriza dar visas a quienes más tiempo llevan esperando.</p>
        </div>""", unsafe_allow_html=True)
    with col_f2:
        st.markdown(f"""
        <div style="background:{t['info_bg']};border:1px solid {t['accent2']}33;
            border-radius:12px;padding:1.5rem;margin:1rem 0;height:100%;">
            <div style="font-size:0.95rem;font-weight:700;color:{t['accent2']} !important;margin-bottom:8px;">
                ¿Qué es la disparidad (f\u2082)?</div>
            <p style="font-size:0.85rem;color:{t['tab_text']} !important;margin:0 0 8px;">
                Para cada país se calcula <strong>W\u0304<sub>c</sub></strong>: el promedio ponderado
                de a\u00f1os de espera de las visas que se le asignan. La disparidad es la brecha
                máxima entre países:</p>
            <p style="font-family:'JetBrains Mono';font-size:0.85rem;color:{t['accent2']} !important;
                margin:0 0 8px;text-align:center;">
                f\u2082 = max |W\u0304<sub>c1</sub> \u2212 W\u0304<sub>c2</sub>|
                entre todos los pares de países</p>
            <p style="font-size:0.85rem;color:{t['tab_text']} !important;margin:0;">
                Ejemplo: si India recibe visas con espera promedio de 13 a\u00f1os y Canada de 0.5 a\u00f1os,
                la disparidad es <strong>12.5 a\u00f1os</strong>. Minimizar f\u2082 busca que la espera
                ponderada sea lo más parecida posible entre todos los países.</p>
        </div>""", unsafe_allow_html=True)

    # How to read this app
    st.markdown(f"""
    <div style="background:{t['card_bg']};border:1px solid {t['card_border']};
        border-radius:16px;padding:1.5rem 2rem;margin:1.5rem 0;">
        <div style="font-size:1.1rem;font-weight:700;color:{t['accent2']} !important;margin-bottom:12px;">
            Cómo navegar esta app</div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:16px;">
            <div style="padding:12px;background:{t['info_bg']};border-radius:10px;">
                <div style="font-weight:700;color:{t['accent1']} !important;margin-bottom:4px;">
                    1. Elige un escenario en el sidebar</div>
                <p style="font-size:0.8rem;color:{t['tab_text']} !important;margin:0;">
                    <strong>Humanitario</strong>: prioriza reducir espera global (f\u2081 m\u00ednima).<br>
                    <strong>Equilibrio</strong>: mejor compromiso entre los tres objetivos.<br>
                    <strong>Equidad</strong>: minimiza brecha entre pa\u00edses (f\u2082 m\u00ednima).<br>
                    <strong>M\u00e1x. Utilizaci\u00f3n</strong>: minimiza desperdicio de visas (f\u2083 m\u00ednima).<br>
                    <strong>FIFO</strong>: sistema actual sin optimizar.</p>
            </div>
            <div style="padding:12px;background:{t['info_bg']};border-radius:10px;">
                <div style="font-weight:700;color:{t['accent1']} !important;margin-bottom:4px;">
                    2. Explora las pesta\u00f1as</div>
                <p style="font-size:0.8rem;color:{t['tab_text']} !important;margin:0;">
                    <strong>Frente de Pareto</strong>: las {len(pareto)} soluciones óptimas y su trade-off.<br>
                    <strong>Asignación</strong>: heatmap de visas por país-categoría.<br>
                    <strong>Impacto por País</strong>: qué gana o pierde cada país.<br>
                    <strong>Convergencia</strong>: cómo mejora el algoritmo por iteración.</p>
            </div>
            <div style="padding:12px;background:{t['info_bg']};border-radius:10px;">
                <div style="font-weight:700;color:{t['accent1']} !important;margin-bottom:4px;">
                    3. Interpreta los números</div>
                <p style="font-size:0.8rem;color:{t['tab_text']} !important;margin:0;">
                    <strong>f\u2081</strong> (espera): carga de espera no atendida \u2014 menor es mejor.<br>
                    <strong>f\u2082</strong> (disparidad): brecha m\u00e1xima de espera entre pa\u00edses \u2014 menor es mejor.<br>
                    <strong>f\u2083</strong> (desperdicio): visas no asignadas \u2014 menor es mejor.<br>
                    Toda soluci\u00f3n Pareto es \u00f3ptima \u2014 ninguna mejora los tres objetivos a la vez.</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


# ── Tab 2: Frente de Pareto ──────────────────────────────────
def _tab_pareto(pareto: list, baseline: tuple, knee: tuple,
                sel_point: tuple, sel_label: str,
                sel_used: int, total_visas: int,
                best_f1_point: tuple, best_f2_point: tuple,
                data: dict = None) -> None:
    t = _get_theme()

    st.markdown('<div class="section-title">Frente de Pareto Interactivo</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Cada punto representa una forma distinta de repartir las 140,000 visas.
    <strong>No existe una soluci\u00f3n "mejor"</strong> \u2014 el decisor elige el balance entre espera, equidad y utilizaci\u00f3n.
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:flex;gap:2rem;padding:0.6rem 1rem;margin-bottom:0.5rem;
        background:{t['card_bg']};border-radius:10px;font-size:0.85rem;flex-wrap:wrap;">
        <span><strong style="color:{t['accent1']};">f\u2081</strong>
        <span style="color:{t['text_muted']};"> = Carga de espera no atendida (a\u00f1os) \u2014 menor es mejor</span></span>
        <span><strong style="color:{t['accent2']};">f\u2082</strong>
        <span style="color:{t['text_muted']};"> = Disparidad entre pa\u00edses (a\u00f1os) \u2014 menor es mejor</span></span>
        <span><strong style="color:{t['accent3']};">f\u2083</strong>
        <span style="color:{t['text_muted']};"> = Desperdicio de visas \u2014 menor es mejor</span></span>
    </div>""", unsafe_allow_html=True)

    sorted_p = sorted(pareto, key=lambda x: x[0])
    f1s = [p[0] for p in sorted_p]
    f2s = [p[1] for p in sorted_p]

    # Compute visas for all Pareto solutions (cached)
    pareto_visas = compute_pareto_visas(tuple(sorted_p), total_visas)
    visas_list = [pareto_visas.get(p, total_visas) for p in sorted_p]
    f3s = [p[2] for p in sorted_p]

    # Selected solution card
    sel_pct = sel_used * 100 // total_visas
    # Pareto solution index (1-based)
    sol_idx = None
    for i, p in enumerate(sorted_p):
        if abs(p[0] - sel_point[0]) < 1e-6 and abs(p[1] - sel_point[1]) < 1e-6:
            sol_idx = i + 1
            break
    sol_txt = f"Solución #{sol_idx} de {len(pareto)}" if sol_idx else "FIFO (baseline)"
    # Gap vs FIFO
    fifo_f1, fifo_f2, fifo_f3 = baseline
    if sel_point != baseline:
        gap_f1 = ((sel_point[0] - fifo_f1) / fifo_f1) * 100
        gap_f2 = ((sel_point[1] - fifo_f2) / fifo_f2) * 100
        gap_f3 = ((sel_point[2] - fifo_f3) / fifo_f3) * 100 if fifo_f3 > 0 else 0
        gap_txt = (f"vs FIFO: f\u2081 {gap_f1:+.1f}% | f\u2082 {gap_f2:+.1f}% | f\u2083 {gap_f3:+.1f}%")
    else:
        gap_txt = "Sistema actual sin optimizar"
    sel_f3 = sel_point[2] if len(sel_point) > 2 else 0
    st.markdown(f"""
    <div class="metric-card" style="border-color:{t['accent2']}4D;margin-bottom:1rem;">
        <div style="display:flex;justify-content:center;align-items:center;gap:2rem;flex-wrap:wrap;">
            <div style="text-align:center;">
                <div style="font-size:0.65rem;color:{t['text_muted']};text-transform:uppercase;font-weight:600;">
                    Escenario: {sel_label}</div>
                <div style="font-family:'JetBrains Mono';font-size:1.8rem;color:{t['accent2']};font-weight:700;">
                    {_fmt(sel_used)}</div>
                <div style="font-size:0.75rem;color:{t['text_subtle']};">
                    de {_fmt(total_visas)} visas EB asignadas ({sel_pct}%)</div>
                <div style="font-size:0.65rem;color:{t['text_muted']};margin-top:4px;">
                    {sol_txt}</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:0.65rem;color:{t['text_muted']};text-transform:uppercase;">f\u2081 espera</div>
                <div style="font-family:'JetBrains Mono';font-size:1.8rem;color:{t['accent1']};font-weight:700;">
                    {_fmtd(sel_point[0])} <span style="font-size:0.7rem;">a\u00f1os</span></div>
                <div style="font-size:0.7rem;color:{t['text_subtle']};max-width:160px;line-height:1.3;">
                    promedio ponderado de a\u00f1os que los solicitantes <strong>siguen esperando</strong> despu\u00e9s de este reparto</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:0.65rem;color:{t['text_muted']};text-transform:uppercase;">f\u2082 disparidad</div>
                <div style="font-family:'JetBrains Mono';font-size:1.8rem;color:{t['accent2']};font-weight:700;">
                    {_fmtd(sel_point[1])} <span style="font-size:0.7rem;">a\u00f1os</span></div>
                <div style="font-size:0.7rem;color:{t['text_subtle']};max-width:160px;line-height:1.3;">
                    diferencia de espera entre el pa\u00eds <strong>m\u00e1s favorecido y el menos</strong> favorecido</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:0.65rem;color:{t['text_muted']};text-transform:uppercase;">f\u2083 desperdicio</div>
                <div style="font-family:'JetBrains Mono';font-size:1.8rem;color:{t['accent3']};font-weight:700;">
                    {_fmt(int(sel_f3))} <span style="font-size:0.7rem;">visas</span></div>
                <div style="font-size:0.7rem;color:{t['text_subtle']};max-width:160px;line-height:1.3;">
                    visas disponibles que <strong>quedan sin asignar</strong> a nadie</div>
            </div>
        </div>
        <div style="text-align:center;margin-top:8px;font-size:0.75rem;color:{t['danger']};font-weight:600;">
            {gap_txt}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style="text-align:center;font-size:0.9rem;color:{t['text_muted']};
        font-style:italic;margin-bottom:-10px;">
        Cada punto es un balance diferente entre espera, equidad y utilizaci\u00f3n</div>""",
                unsafe_allow_html=True)

    fig = go.Figure()
    # Custom hover with visas and f3
    hover_texts = [
        f"<b>Solución MOHHO</b><br>"
        f"f\u2081 (espera): {_fmtd(f1, 3)} a\u00f1os<br>"
        f"f\u2082 (disparidad): {_fmtd(f2, 2)} a\u00f1os<br>"
        f"f\u2083 (desperdicio): {_fmt(int(f3))}<br>"
        f"Visas otorgadas: {_fmt(v)} ({v * 100 // total_visas}%)"
        for f1, f2, f3, v in zip(f1s, f2s, f3s, visas_list)
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
    fifo_label = f"FIFO ({_fmtd(baseline[0], 2)}, {_fmtd(baseline[1], 2)}, {_fmt(int(baseline[2]))})"
    fig.add_trace(go.Scatter(
        x=[baseline[0]], y=[baseline[1]], mode='markers+text',
        marker=dict(size=20, color=t['danger'], symbol='star',
                    line=dict(width=2, color=t['marker_line'])),
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
                    line=dict(width=2, color=t['marker_line'])),
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
            hovertemplate='<b>Selección activa</b><br>'
                          'f\u2081 espera: %{x:.3f} a\u00f1os<br>'
                          'f\u2082 disparidad: %{y:.2f} a\u00f1os<extra></extra>',
            name=f'Selección ({sel_label})',
        ))

    # Annotations
    fig.add_annotation(x=best_f1_point[0], y=best_f1_point[1] + 0.5,
                       text="Zona humanitaria", showarrow=False,
                       font=dict(size=10, color=_rgba(t['accent1'], 0.6)))
    fig.add_annotation(x=best_f2_point[0], y=best_f2_point[1] - 0.5,
                       text="Zona de equidad", showarrow=False,
                       font=dict(size=10, color=_rgba(t['accent3'], 0.6)))

    y_max = baseline[1] + 1.5
    y_min = min(f2s) - 0.5
    fig.update_layout(**_plotly_layout(
        xaxis=dict(title="f\u2081 \u2014 Carga de espera no atendida (a\u00f1os)", **_grid()),
        yaxis=dict(title="f\u2082 \u2014 Disparidad entre países (a\u00f1os)",
                   range=[y_min, y_max], **_grid()),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor=t['legend_bg'], font=dict(size=10)),
        height=550,
        margin=dict(t=40),
    ))
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

    # Projection selector for 2D views and 3D
    st.markdown('<div class="section-title">Proyecciones del Frente Triobjetivo</div>',
                unsafe_allow_html=True)
    proj_choice = st.radio(
        "Vista",
        ["f\u2081 vs f\u2082", "f\u2081 vs f\u2083", "f\u2082 vs f\u2083", "3D (f\u2081 \u00d7 f\u2082 \u00d7 f\u2083)"],
        horizontal=True, key="pareto_proj",
    )

    proj_axes = {
        "f\u2081 vs f\u2082": (f1s, f2s, f3s, "f\u2081 \u2014 Espera (a\u00f1os)", "f\u2082 \u2014 Disparidad (a\u00f1os)", "f\u2083"),
        "f\u2081 vs f\u2083": (f1s, f3s, f2s, "f\u2081 \u2014 Espera (a\u00f1os)", "f\u2083 \u2014 Desperdicio (visas)", "f\u2082"),
        "f\u2082 vs f\u2083": (f2s, f3s, f1s, "f\u2082 \u2014 Disparidad (a\u00f1os)", "f\u2083 \u2014 Desperdicio (visas)", "f\u2081"),
    }
    if proj_choice != "3D (f\u2081 \u00d7 f\u2082 \u00d7 f\u2083)":
        px, py, pc, xlabel, ylabel, clabel = proj_axes[proj_choice]
        bl_map = {
            "f\u2081 vs f\u2082": (baseline[0], baseline[1]),
            "f\u2081 vs f\u2083": (baseline[0], baseline[2]),
            "f\u2082 vs f\u2083": (baseline[1], baseline[2]),
        }
        kn_map = {
            "f\u2081 vs f\u2082": (knee[0], knee[1]),
            "f\u2081 vs f\u2083": (knee[0], knee[2]),
            "f\u2082 vs f\u2083": (knee[1], knee[2]),
        }
        bl_xy = bl_map[proj_choice]
        kn_xy = kn_map[proj_choice]

        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(
            x=px, y=py, mode='markers',
            marker=dict(size=8, color=pc,
                        colorscale=[[0, t['danger']], [0.5, t['accent1']], [1, t['accent3']]],
                        colorbar=dict(title=clabel, thickness=15, len=0.6),
                        line=dict(width=1, color=t['text_subtle'])),
            hovertemplate=f'{xlabel}: %{{x:.3f}}<br>{ylabel}: %{{y:.2f}}<br>{clabel}: %{{marker.color:.2f}}<extra></extra>',
            name='Frente de Pareto',
        ))
        fig_proj.add_trace(go.Scatter(
            x=[bl_xy[0]], y=[bl_xy[1]], mode='markers+text',
            marker=dict(size=18, color=t['danger'], symbol='star',
                        line=dict(width=2, color=t['marker_line'])),
            text=['FIFO'], textposition='middle right',
            textfont=dict(color=t['danger'], size=10),
            name='FIFO',
        ))
        fig_proj.add_trace(go.Scatter(
            x=[kn_xy[0]], y=[kn_xy[1]], mode='markers+text',
            marker=dict(size=14, color=t['accent2'], symbol='diamond',
                        line=dict(width=2, color=t['marker_line'])),
            text=['Equilibrio'], textposition='bottom center',
            textfont=dict(color=t['accent2'], size=10),
            name='Equilibrio (Knee)',
        ))
        fig_proj.update_layout(**_plotly_layout(
            xaxis=dict(title=xlabel, **_grid()),
            yaxis=dict(title=ylabel, **_grid()),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor=t['legend_bg'], font=dict(size=10)),
            height=500, margin=dict(t=40),
        ))
        st.plotly_chart(fig_proj, use_container_width=True, theme=None,
                        config={"displayModeBar": False}, key="pareto_proj_2d")
    else:
        # 3D Pareto surface f1 x f2 x f3
        fig3d = go.Figure()
        fig3d.add_trace(go.Scatter3d(
            x=f1s, y=f2s, z=f3s,
            mode='markers',
            marker=dict(
                size=5, color=visas_list,
                colorscale=[[0, t['danger']], [0.5, t['accent1']], [1, t['accent3']]],
                colorbar=dict(title="Visas", thickness=12, len=0.6, x=1.02),
                line=dict(width=0.5, color=t['text_subtle']),
            ),
            hovertemplate=(
                '<b>Soluci\u00f3n MOHHO</b><br>'
                'f\u2081 (espera): %{x:.3f} a\u00f1os<br>'
                'f\u2082 (disparidad): %{y:.2f} a\u00f1os<br>'
                'f\u2083 (desperdicio): %{z:,.0f}<extra></extra>'
            ),
            name='Frente de Pareto',
        ))
        # FIFO point
        fig3d.add_trace(go.Scatter3d(
            x=[baseline[0]], y=[baseline[1]], z=[baseline[2]],
            mode='markers', name='FIFO (actual)',
            marker=dict(size=10, color=t['danger'], symbol='diamond',
                        line=dict(width=2, color=t['marker_line'])),
            hovertemplate=(
                '<b>FIFO (sistema actual)</b><br>'
                'f\u2081: %{x:.3f}<br>f\u2082: %{y:.2f}<br>'
                'f\u2083: %{z:,.0f}<extra></extra>'
            ),
        ))
        # Knee point
        fig3d.add_trace(go.Scatter3d(
            x=[knee[0]], y=[knee[1]], z=[knee[2]],
            mode='markers', name='Equilibrio (Knee)',
            marker=dict(size=8, color=t['accent2'], symbol='diamond',
                        line=dict(width=2, color=t['marker_line'])),
            hovertemplate=(
                '<b>Equilibrio (Knee)</b><br>'
                'f\u2081: %{x:.3f}<br>f\u2082: %{y:.2f}<br>'
                'f\u2083: %{z:,.0f}<extra></extra>'
            ),
        ))
        fig3d.update_layout(**_plotly_layout(
            scene=dict(
                xaxis=dict(title='f\u2081 \u2014 Espera (a\u00f1os)', backgroundcolor='rgba(0,0,0,0)',
                           gridcolor=_rgba(t['text'], 0.1), showbackground=True),
                yaxis=dict(title='f\u2082 \u2014 Disparidad (a\u00f1os)', backgroundcolor='rgba(0,0,0,0)',
                           gridcolor=_rgba(t['text'], 0.1), showbackground=True),
                zaxis=dict(title='f\u2083 \u2014 Desperdicio (visas)', backgroundcolor='rgba(0,0,0,0)',
                           gridcolor=_rgba(t['text'], 0.1), showbackground=True),
                bgcolor='rgba(0,0,0,0)',
                camera=dict(eye=dict(x=1.8, y=-1.4, z=0.8)),
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                        bgcolor=t['legend_bg'], font=dict(size=10)),
            height=600,
            margin=dict(l=0, r=0, t=30, b=0),
        ))
        st.plotly_chart(fig3d, use_container_width=True, theme=None, key="pareto_3d")
        st.markdown(f"""<div class="info-box" style="font-size:0.8rem;">
        Frente de Pareto triobjetivo en 3D: f\u2081 (espera), f\u2082 (disparidad), f\u2083 (desperdicio de visas).
        Arrastra para rotar, scroll para zoom. El punto rojo es FIFO; el dorado es el equilibrio (knee).
        </div>""", unsafe_allow_html=True)

    # Stats cards with visas — all 5 scenarios
    best_f3_point = min(pareto, key=lambda x: x[2])
    v_f1 = pareto_visas.get(best_f1_point, total_visas)
    v_knee = pareto_visas.get(knee, total_visas)
    v_f2 = pareto_visas.get(best_f2_point, total_visas)
    v_f3 = pareto_visas.get(best_f3_point, total_visas)
    fifo_used = data.get("fifo_used", total_visas) if data else total_visas
    baseline_pt = data.get("fifo_fit", (0, 0, 0)) if data else (0, 0, 0)

    _card_style = (
        "font-family:'JetBrains Mono';font-weight:700;font-size:1rem;"
    )
    _lbl_style = (
        f"font-size:0.6rem;color:{t['text_muted']};text-transform:uppercase;"
    )

    scenarios_cards = [
        ("Humanitario", t['accent1'], best_f1_point, v_f1, "",
         "Minimiza la espera global (f\u2081): prioriza a quienes llevan m\u00e1s a\u00f1os en la fila."),
        ("Equilibrio", t['accent2'], knee, v_knee,
         f"border-color:{t['accent2']}4D;",
         "Mejor compromiso entre los tres objetivos. Punto de m\u00e1xima curvatura del frente."),
        ("Equidad", t['accent3'], best_f2_point, v_f2, "",
         "Minimiza la brecha entre pa\u00edses (f\u2082): todos esperan tiempos similares."),
        ("M\u00e1x. Utilizaci\u00f3n", t['mid_blue'], best_f3_point, v_f3, "",
         "Minimiza el desperdicio (f\u2083): usa todas las visas disponibles."),
        ("FIFO (actual)", t['danger'], baseline_pt, fifo_used, "",
         "Sistema vigente: orden de llegada. Sin optimizaci\u00f3n."),
    ]

    cols = st.columns(len(scenarios_cards))
    for col, (title, color, pt, v_val, border, hint) in zip(cols, scenarios_cards):
        with col:
            v_pct = v_val * 100 // total_visas if v_val > 0 else 0
            st.markdown(f"""
            <div class="metric-card" style="{border}text-align:center;padding:12px 8px;">
                <div style="font-size:0.65rem;color:{color};text-transform:uppercase;font-weight:700;
                    margin-bottom:6px;">{title}</div>
                <div style="display:flex;justify-content:center;gap:8px;flex-wrap:wrap;">
                    <div>
                        <div style="{_lbl_style}">f\u2081</div>
                        <div style="{_card_style}color:{color};">{_fmtd(pt[0])}</div>
                    </div>
                    <div>
                        <div style="{_lbl_style}">f\u2082</div>
                        <div style="{_card_style}color:{color};">{_fmtd(pt[1])}</div>
                    </div>
                    <div>
                        <div style="{_lbl_style}">f\u2083</div>
                        <div style="{_card_style}color:{color};">{_fmt(int(pt[2]))}</div>
                    </div>
                </div>
                <div style="font-size:0.75rem;color:{t['text']};margin-top:6px;">
                    <strong>{_fmt(v_val)}</strong> visas ({v_pct}%)</div>
                <div style="font-size:0.65rem;color:{t['text_subtle']};margin-top:6px;
                    font-style:italic;line-height:1.3;padding:0 2px;">
                    {hint}</div>
            </div>""", unsafe_allow_html=True)

    # Visas por solución chart
    st.markdown('<div class="section-title">Visas Otorgadas por Solución del Frente</div>',
                unsafe_allow_html=True)

    fig_v = go.Figure()
    fig_v.add_trace(go.Bar(
        x=list(range(1, len(sorted_p) + 1)),
        y=visas_list,
        marker=dict(
            color=visas_list,
            colorscale=[[0, t['danger']], [0.5, t['accent1']], [1, t['accent3']]],
            line=dict(width=0),
        ),
        hovertemplate=(
            'Soluci\u00f3n #%{x}<br>'
            'Visas: %{y:,.0f}<br>'
            f'f\u2081: %{{customdata[0]:.3f}} | f\u2082: %{{customdata[1]:.2f}} | f\u2083: %{{customdata[2]:,.0f}}'
            '<extra></extra>'),
        customdata=list(zip(f1s, f2s, f3s)),
    ))
    fig_v.add_hline(y=total_visas, line_dash="dash", line_color=t['accent3'],
                    annotation_text=f"Máximo: {_fmt(total_visas)}")
    fifo_used_val = data.get('fifo_used', total_visas)
    fig_v.add_hline(y=fifo_used_val, line_dash="dot",
                    line_color=t['danger'],
                    annotation_text=f"FIFO: {_fmt(fifo_used_val)}")
    fig_v.update_layout(**_plotly_layout(
        xaxis=dict(title="Solución Pareto (ordenada por f\u2081)", **_grid()),
        yaxis=dict(title="Visas otorgadas", **_grid()),
        height=350,
        margin=dict(l=60, r=30, t=40, b=50),
    ))
    st.plotly_chart(fig_v, use_container_width=True, theme=None, config={"displayModeBar": False})

    # ── Tabla de soluciones Pareto ──
    with st.expander(f"Ver las {len(pareto)} soluciones del frente de Pareto"):
        st.markdown(f"""
        <div class="info-box">
            <strong>\u00bfPor qu\u00e9 {len(pareto)} soluciones?</strong> El algoritmo MOHHO explora miles de asignaciones
            posibles. De todas ellas, solo {len(pareto)} son <em>no dominadas</em>: ninguna otra soluci\u00f3n
            es mejor en los tres objetivos a la vez. Forman el <em>frente de Pareto</em> \u2014 el conjunto
            de mejores compromisos posibles entre minimizar la espera (f\u2081), minimizar la
            disparidad entre pa\u00edses (f\u2082) y minimizar el desperdicio de visas (f\u2083). Moverse a lo largo del frente implica siempre un
            trade-off: mejorar un objetivo empeora al menos otro.
        </div>""", unsafe_allow_html=True)

        # Build table data
        rows = []
        for i, p in enumerate(sorted_p):
            v = pareto_visas.get(p, total_visas)
            gap_f1_pct = ((p[0] - fifo_f1) / fifo_f1) * 100
            gap_f2_pct = ((p[1] - fifo_f2) / fifo_f2) * 100
            # Tag special solutions
            _bf3p = min(pareto, key=lambda x: x[2])
            tag = ""
            if abs(p[0] - best_f1_point[0]) < 1e-6 and abs(p[1] - best_f1_point[1]) < 1e-6:
                tag = "Humanitario"
            elif abs(p[0] - knee[0]) < 1e-6 and abs(p[1] - knee[1]) < 1e-6:
                tag = "Equilibrio"
            elif abs(p[0] - best_f2_point[0]) < 1e-6 and abs(p[1] - best_f2_point[1]) < 1e-6:
                tag = "Equidad"
            elif abs(p[0] - _bf3p[0]) < 1e-6 and abs(p[2] - _bf3p[2]) < 1e-6:
                tag = "M\u00e1x. Utilizaci\u00f3n"
            gap_f3_pct = ((p[2] - fifo_f3) / fifo_f3) * 100 if fifo_f3 > 0 else 0
            rows.append({
                "#": i + 1,
                "f\u2081 espera": round(p[0], 4),
                "f\u2082 disparidad": round(p[1], 4),
                "f\u2083 desperdicio": int(p[2]),
                "Visas": v,
                "% visas": f"{v * 100 // total_visas}%",
                "\u0394f\u2081 vs FIFO": f"{gap_f1_pct:+.1f}%",
                "\u0394f\u2082 vs FIFO": f"{gap_f2_pct:+.1f}%",
                "\u0394f\u2083 vs FIFO": f"{gap_f3_pct:+.1f}%",
                "Escenario": tag,
            })
        df_pareto = pd.DataFrame(rows)
        st.dataframe(df_pareto, use_container_width=True, height=400)

    # ── Comparativa por pais: espera, visas, disparidad ──
    if data is not None:
        st.markdown('<div class="section-title">Comparativa por País</div>',
                    unsafe_allow_html=True)

        countries = data["countries"]
        wait_arr = np.array(data["wait_matrix"])

        def _wc_from_matrix(mat):
            mat_arr = np.array(mat)
            wc = {}
            for i, c in enumerate(countries):
                assigned = mat_arr[i].sum()
                if assigned > 0:
                    wc[c] = float((mat_arr[i] * wait_arr[i]).sum() / assigned)
                else:
                    wc[c] = float(wait_arr[i].max())
            return wc

        def _visas_from_matrix(mat):
            mat_arr = np.array(mat)
            return {c: int(mat_arr[i].sum()) for i, c in enumerate(countries)}

        # Compute all 5 scenarios
        best_f3_point = min(pareto, key=lambda x: x[2])
        all_scenarios = [
            ("FIFO", data["fifo_matrix"], data["fifo_fit"]),
        ]
        for label, point in [("Humanitario", best_f1_point),
                             ("Equilibrio", knee),
                             ("Equidad", best_f2_point),
                             ("M\u00e1x. Utilizaci\u00f3n", best_f3_point)]:
            mat, fit, _ = compute_mohho_allocation(point[0], point[1], point[2])
            all_scenarios.append((label, mat, fit))

        scenarios_wc = [(n, _wc_from_matrix(m)) for n, m, _ in all_scenarios]
        scenarios_visas = [(n, _visas_from_matrix(m)) for n, m, _ in all_scenarios]

        # Toggle: 2 scenarios (FIFO vs selected) or all 5
        show_all = st.checkbox("Ver los 5 escenarios", value=True, key="pareto_4scenarios")

        if show_all:
            show_scenarios = list(range(5))
        else:
            # FIFO + current selected
            sel_map = {"Humanitario": 1, "Equilibrio (Knee)": 2, "Equidad": 3, "M\u00e1x. Utilizaci\u00f3n": 4, "FIFO": 0}
            sel_idx = sel_map.get(sel_label, 2)
            show_scenarios = [0, sel_idx] if sel_idx != 0 else [0]

        scenario_colors = [t['danger'], t['accent1'], t['accent2'], t['accent3'], t['mid_blue']]

        # Sort countries by FIFO wait
        fifo_wc = scenarios_wc[0][1]
        sorted_countries = sorted(countries, key=lambda c: fifo_wc[c], reverse=True)
        display_countries = [f"{_flag(c)} {c}" for c in sorted_countries]

        # ── Chart 1: Espera ponderada W_c ──
        st.markdown(f"""<div style="text-align:center;font-size:0.9rem;color:{t['text_muted']};
            font-style:italic;margin-bottom:-10px;">
            W\u2099 — Espera ponderada por país (a\u00f1os)</div>""",
                    unsafe_allow_html=True)

        fig_wc = go.Figure()
        for idx in show_scenarios:
            name, wc = scenarios_wc[idx]
            vals = [round(wc[c], 2) for c in sorted_countries]
            fig_wc.add_trace(go.Bar(
                y=display_countries, x=vals, orientation='h',
                name=name, marker_color=scenario_colors[idx], opacity=0.85,
                hovertemplate=f'<b>%{{y}}</b><br>{name}: %{{x:.1f}} a\u00f1os<extra></extra>',
            ))
        fig_wc.update_layout(**_plotly_layout(
            barmode='group',
            xaxis=dict(title="A\u00f1os de espera ponderada", **_grid()),
            yaxis=dict(autorange='reversed'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor=t['legend_bg'], font=dict(size=10)),
            height=max(550, len(countries) * 40),
            margin=dict(l=160, r=30, t=40, b=50),
        ))
        st.plotly_chart(fig_wc, use_container_width=True, theme=None, config={"displayModeBar": False})

        # ── Chart 2: Visas por país ──
        st.markdown(f"""<div style="text-align:center;font-size:0.9rem;color:{t['text_muted']};
            font-style:italic;margin-bottom:-10px;">
            Visas asignadas por país</div>""",
                    unsafe_allow_html=True)

        fifo_visas = scenarios_visas[0][1]
        sorted_countries_v = sorted(countries, key=lambda c: fifo_visas[c], reverse=True)
        display_countries_v = [f"{_flag(c)} {c}" for c in sorted_countries_v]

        fig_vis = go.Figure()
        for idx in show_scenarios:
            name, vis = scenarios_visas[idx]
            vals = [vis[c] for c in sorted_countries_v]
            fig_vis.add_trace(go.Bar(
                y=display_countries_v, x=vals, orientation='h',
                name=name, marker_color=scenario_colors[idx], opacity=0.85,
                hovertemplate=f'<b>%{{y}}</b><br>{name}: %{{x:,.0f}} visas<extra></extra>',
            ))
        fig_vis.update_layout(**_plotly_layout(
            barmode='group',
            xaxis=dict(title="Visas asignadas", **_grid()),
            yaxis=dict(autorange='reversed'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor=t['legend_bg'], font=dict(size=10)),
            height=max(550, len(countries) * 40),
            margin=dict(l=160, r=30, t=40, b=50),
        ))
        st.plotly_chart(fig_vis, use_container_width=True, theme=None, config={"displayModeBar": False})

        # ── Chart 3: Disparidad (delta W_c vs promedio global) ──
        st.markdown(f"""<div style="text-align:center;font-size:0.9rem;color:{t['text_muted']};
            font-style:italic;margin-bottom:-10px;">
            Desviación de espera vs promedio global (disparidad por país)</div>""",
                    unsafe_allow_html=True)

        fig_disp = go.Figure()
        for idx in show_scenarios:
            name, wc = scenarios_wc[idx]
            wc_vals = [wc[c] for c in sorted_countries]
            wc_mean = sum(wc_vals) / len(wc_vals)
            deltas = [round(wc[c] - wc_mean, 2) for c in sorted_countries]
            fig_disp.add_trace(go.Bar(
                y=display_countries, x=deltas, orientation='h',
                name=name, marker_color=scenario_colors[idx], opacity=0.85,
                hovertemplate=f'<b>%{{y}}</b><br>{name}: %{{x:+.1f}} a\u00f1os vs promedio<extra></extra>',
            ))
        fig_disp.update_layout(**_plotly_layout(
            barmode='group',
            xaxis=dict(title="Desviación vs promedio (a\u00f1os)", zeroline=True, **_grid()),
            yaxis=dict(autorange='reversed'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor=t['legend_bg'], font=dict(size=10)),
            height=max(550, len(countries) * 40),
            margin=dict(l=160, r=30, t=40, b=50),
        ))
        st.plotly_chart(fig_disp, use_container_width=True, theme=None, config={"displayModeBar": False})


# ── Tab 3: Asignación ────────────────────────────────────────
def _tab_allocation(data: dict, sel_matrix: list, sel_fit: tuple,
                    sel_used: int, sel_label: str, baseline: tuple) -> None:
    t = _get_theme()

    st.markdown('<div class="section-title">Asignación de Visas \u2014 Escenario Activo</div>',
                unsafe_allow_html=True)

    countries = data["countries"]
    categories = data["categories"]
    fifo_matrix = data["fifo_matrix"]
    mohho_matrix = sel_matrix
    mohho_fit = sel_fit
    mohho_used = sel_used

    sel_pct = mohho_used * 100 // data["total_visas"]
    sel_f3_val = int(sel_fit[2]) if len(sel_fit) > 2 else data['total_visas'] - mohho_used
    st.markdown(f"""
    <div class="info-box">
        <strong>Escenario: {sel_label}</strong> \u2014
        Visas otorgadas: <strong>{_fmt(mohho_used)} / {_fmt(data['total_visas'])}
        ({sel_pct}%)</strong><br>
        <span style="color:{t['accent1']};">f\u2081 = {_fmtd(sel_fit[0])} a\u00f1os</span>
        (carga de espera no atendida ponderada \u2014 menor es mejor) \u2014
        <span style="color:{t['accent2']};">f\u2082 = {_fmtd(sel_fit[1])} a\u00f1os</span>
        (brecha m\u00e1xima de espera entre pa\u00edses \u2014 menor es mejor) \u2014
        <span style="color:{t['accent3']};">f\u2083 = {_fmt(sel_f3_val)}</span>
        (visas sin asignar \u2014 menor es mejor)
    </div>""", unsafe_allow_html=True)

    util_pct = mohho_used / data["total_visas"]
    st.markdown(f"""
<div style="background:{t['card_bg']};border-radius:12px;padding:16px;margin:1rem 0;
    border:1px solid {t['card_border']};">
    <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
        <span style="font-size:0.8rem;font-weight:600;color:{t['text']};">
            Utilizaci\u00f3n de visas</span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
            color:{t['accent3'] if util_pct > 0.95 else t['danger']};font-weight:700;">
            {_fmt(mohho_used)} / {_fmt(data['total_visas'])} ({util_pct*100:.1f}%)</span>
    </div>
    <div style="height:12px;background:{t['card_border']};border-radius:6px;overflow:hidden;">
        <div style="width:{util_pct*100:.1f}%;height:100%;border-radius:6px;
            background:linear-gradient(90deg,{t['accent1']},{t['accent3']});
            transition:width 0.5s ease;"></div>
    </div>
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
                    f"Categoría: {categories[j]} ({CATS_DESC.get(categories[j], '')})<br>"
                    f"Visas: {_fmt(int(mat[i][j]))}<br>"
                    f"Demanda: {_fmt(int(demand_arr[i][j]))}<br>"
                    f"Cobertura: {_fmtd(pct, 1)}%<br>"
                    f"Espera: {int(wait_arr[i][j])} a\u00f1os"
                )
            hover.append(row)
        return hover

    # Side-by-side heatmaps
    col1, col2 = st.columns(2)
    for col, mat, title, hover, hmap_key in [
        (col1, mohho_arr,
         f"{sel_label} \u2014 {_fmt(mohho_used)} visas (f\u2082={_fmtd(mohho_fit[1])}a)", _build_hover(mohho_arr), "heatmap_sel"),
        (col2, fifo_arr,
         f"FIFO \u2014 {_fmt(data['fifo_used'])} visas (f\u2082={_fmtd(baseline[1])}a)", _build_hover(fifo_arr), "heatmap_fifo"),
    ]:
        with col:
            # Custom colorscale: adapts to theme
            heatmap_cs = [
                [0.0, t['bg2']],
                [0.15, "#1b3a4b"],
                [0.35, "#0d6e6e"],
                [0.55, "#0d9488"],
                [0.75, "#2dd4bf"],
                [1.0, "#99f6e4"],
            ]
            mat_max = mat.max() if mat.max() > 0 else 1
            cell_colors = [
                [t['bg2'] if v / mat_max > 0.6 else t['text'] for v in row]
                for row in mat
            ]
            fig = go.Figure(go.Heatmap(
                z=mat, x=cat_labels, y=countries,
                colorscale=heatmap_cs, showscale=True,
                hovertext=hover, hoverinfo='text',
            ))
            # Per-cell annotations with adaptive text color
            for i, country in enumerate(countries):
                for j in range(len(categories)):
                    fig.add_annotation(
                        x=cat_labels[j], y=country,
                        text=_fmt(int(mat[i][j])),
                        showarrow=False,
                        font=dict(size=8, color=cell_colors[i][j]),
                    )
            fig.update_layout(**_plotly_layout(
                title=dict(text=title, font=dict(size=13, color=t['accent2'])),
                height=650, margin=dict(l=110, r=20, t=50, b=80),
            ))
            st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False}, key=hmap_key)

    # Difference heatmap
    st.markdown(f'<div class="section-title">Diferencias ({sel_label} \u2212 FIFO)</div>',
                unsafe_allow_html=True)
    diff_arr = mohho_arr - fifo_arr
    vabs = max(abs(int(diff_arr.min())), abs(int(diff_arr.max()))) or 1

    # Diverging colorscale: saturated red → dark neutral → saturated blue-green
    # Uses opaque colors that work on any background (dark or light themes)
    diff_colorscale = [
        [0.0, "#c0392b"],
        [0.2, "#d4644a"],
        [0.4, "#8b7355"],
        [0.5, "#555555"],
        [0.6, "#4a7a6a"],
        [0.8, "#2a9d8f"],
        [1.0, "#1a7a6d"],
    ]
    # Text color: always white on saturated ends, light gray on neutral middle
    diff_max = max(abs(diff_arr.min()), abs(diff_arr.max())) or 1
    diff_cell_colors = [
        ["#ffffff" if abs(v) / diff_max > 0.2 else "#cccccc"
         for v in row]
        for row in diff_arr
    ]
    fig_diff = go.Figure(go.Heatmap(
        z=diff_arr, x=cat_labels, y=countries,
        colorscale=diff_colorscale, zmid=0, zmin=-vabs, zmax=vabs,
        colorbar=dict(title="\u0394 Visas", tickfont=dict(color=t['text'])),
    ))
    # Per-cell annotations — bold white text, always legible
    for i, country in enumerate(countries):
        for j in range(len(categories)):
            v = int(diff_arr[i][j])
            txt = f"+{_fmt(v)}" if v > 0 else ("0" if v == 0 else _fmt(v))
            fig_diff.add_annotation(
                x=cat_labels[j], y=country, text=txt, showarrow=False,
                font=dict(size=9, color=diff_cell_colors[i][j],
                          family="JetBrains Mono"),
            )
    fig_diff.update_layout(**_plotly_layout(
        height=600, margin=dict(l=110, r=60, t=30, b=80),
    ))
    st.plotly_chart(fig_diff, use_container_width=True, theme=None, config={"displayModeBar": False}, key="heatmap_diff")

    # Data tables in expander
    with st.expander("Ver tablas numéricas completas"):
        st.markdown("**Demanda pendiente (peticiones aprobadas)**")
        df_demand = pd.DataFrame(data["demand_matrix"],
                                 index=countries, columns=categories)
        df_demand["Total"] = df_demand.sum(axis=1)
        st.dataframe(df_demand, use_container_width=True)

        st.markdown("**Fechas de prioridad (Visa Bulletin)**")
        st.caption("Estas fechas determinan el peso de espera w\u2082 = 2026 \u2212 a\u00f1o. "
                   "Un grupo con fecha 2013 lleva 13 a\u00f1os esperando (w=13), lo que "
                   "multiplica su impacto en f\u2081. 'Current' significa sin backlog (w=0).")
        date_strs = []
        for i in range(len(countries)):
            row = []
            for j in range(len(categories)):
                d = data["date_matrix"][i][j]
                row.append("Current" if d >= 2026 else str(d))
            date_strs.append(row)
        df_dates = pd.DataFrame(date_strs, index=countries, columns=categories)
        st.dataframe(df_dates, use_container_width=True)

    # Download CSV button
    df_mohho = pd.DataFrame(mohho_matrix, index=countries, columns=categories)
    df_mohho["Total"] = df_mohho.sum(axis=1)
    csv_data = df_mohho.to_csv()
    st.download_button(
        label="Descargar tabla de asignación como CSV",
        data=csv_data,
        file_name=f"asignacion_{sel_label.replace(' ', '_')}.csv",
        mime="text/csv",
    )


# ── Tab 4: Impacto por País ─────────────────────────────────
def _tab_country(data: dict, sel_matrix: list, sel_fit: tuple,
                 sel_used: int, sel_label: str,
                 baseline: tuple, knee: tuple,
                 best_f1: tuple, best_f2: tuple,
                 best_f3: tuple = None) -> None:
    t = _get_theme()

    st.markdown('<div class="section-title">Impacto por País</div>',
                unsafe_allow_html=True)

    sel_f3_country = int(sel_fit[2]) if len(sel_fit) > 2 else data['total_visas'] - sel_used
    st.markdown(f"""
    <div class="info-box">
        <strong>Escenario: {sel_label}</strong> \u2014
        Visas otorgadas: <strong>{_fmt(sel_used)} / {_fmt(data['total_visas'])}</strong> \u2014
        f\u2081 espera = {_fmtd(sel_fit[0])} \u2014
        f\u2082 disparidad = {_fmtd(sel_fit[1])} \u2014
        f\u2083 desperdicio = {_fmt(sel_f3_country)}
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
    st.markdown(f'<div class="section-title">{sel_label} vs FIFO — visas por país</div>',
                unsafe_allow_html=True)
    fig.update_layout(**_plotly_layout(
        barmode='group',
        xaxis=dict(title="Visas asignadas", **_grid()),
        yaxis=dict(autorange='reversed'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor=t['legend_bg']),
        height=max(500, len(countries) * 35),
        margin=dict(l=140, r=30, t=80, b=50),
    ))
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

    # Delta bars
    st.markdown(f'<div class="section-title">Cambio {sel_label} \u2212 FIFO por País</div>',
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
    st.plotly_chart(fig2, use_container_width=True, theme=None, config={"displayModeBar": False})

    # Wait times lollipop
    st.markdown('<div class="section-title">Tiempo de Espera Máximo por País</div>',
                unsafe_allow_html=True)

    wait_sorted = sorted(data["country_wait"].items(), key=lambda x: x[1], reverse=True)
    w_countries = [f"{_flag(c)} {c}" for c, _ in wait_sorted]
    w_values = [w for _, w in wait_sorted]
    w_colors = [t['danger'] if w >= 10 else (t['mid_blue'] if w >= 3 else t['accent3']) for w in w_values]

    fig3 = go.Figure()
    for i, (c, w) in enumerate(zip(w_countries, w_values)):
        fig3.add_trace(go.Scatter(
            x=[0, w], y=[c, c], mode='lines',
            line=dict(color=w_colors[i], width=3),
            showlegend=False, hoverinfo='skip',
        ))
    fig3.add_trace(go.Scatter(
        x=w_values, y=w_countries, mode='markers+text',
        marker=dict(size=14, color=w_colors, line=dict(width=2, color=t['marker_line'])),
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
    st.plotly_chart(fig3, use_container_width=True, theme=None, config={"displayModeBar": False})

    # ── Comparativa de los 4 escenarios ──
    show_all = st.checkbox("Ver todos los escenarios", key="show_all_scenarios")
    if show_all:
        _best_f3 = best_f3 if best_f3 is not None else min(
            [baseline], key=lambda x: x[2])
        _all_scenarios_chart(data, baseline, knee, best_f1, best_f2, _best_f3, countries)


def _all_scenarios_chart(data: dict, baseline: tuple, knee: tuple,
                         best_f1: tuple, best_f2: tuple,
                         best_f3: tuple,
                         countries: list[str]) -> None:
    """Grouped bar chart comparing all 5 scenarios side by side."""
    t = _get_theme()

    st.markdown('<div class="section-title">Comparativa: 5 Escenarios por Pa\u00eds</div>',
                unsafe_allow_html=True)

    scenarios = [
        ("FIFO", baseline, data["fifo_matrix"], data["fifo_used"]),
        ("Humanitario", best_f1, None, 0),
        ("Equilibrio", knee, None, 0),
        ("Equidad", best_f2, None, 0),
        ("M\u00e1x. Utilizaci\u00f3n", best_f3, None, 0),
    ]
    # Compute allocations for MOHHO scenarios
    resolved = []
    for name, point, matrix, used in scenarios:
        if matrix is not None:
            by_c = [sum(matrix[i]) for i in range(len(countries))]
            resolved.append((name, point, by_c, used))
        else:
            mat, fit, u = compute_mohho_allocation(point[0], point[1], point[2])
            by_c = [sum(mat[i]) for i in range(len(countries))]
            resolved.append((name, fit, by_c, u))

    # Summary cards
    cols = st.columns(5)
    for col, (name, fit, by_c, used) in zip(cols, resolved):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="padding:14px;">
                <div style="font-size:0.7rem;color:{t['text_muted']};text-transform:uppercase;
                    font-weight:600;">{name}</div>
                <div style="font-family:'JetBrains Mono';font-size:1.1rem;color:{t['accent2']};
                    font-weight:700;">{_fmt(used)} visas</div>
                <div style="font-size:0.72rem;color:{t['text_subtle']};">
                    f\u2081={_fmtd(fit[0])} | f\u2082={_fmtd(fit[1])} | f\u2083={_fmt(int(fit[2]) if len(fit) > 2 else 0)}</div>
            </div>""", unsafe_allow_html=True)

    # Sorted by FIFO allocation
    fifo_by_c = resolved[0][2]
    sorted_idx = sorted(range(len(countries)), key=lambda i: fifo_by_c[i], reverse=True)
    sorted_countries = [f"{_flag(countries[i])} {countries[i]}" for i in sorted_idx]

    scenario_colors = [t['danger'], t['accent1'], t['accent2'], t['accent3'], t['mid_blue']]

    fig = go.Figure()
    for idx, (name, _, by_c, _) in enumerate(resolved):
        sorted_vals = [by_c[i] for i in sorted_idx]
        fig.add_trace(go.Bar(
            y=sorted_countries, x=sorted_vals, orientation='h',
            name=name, marker_color=scenario_colors[idx], opacity=0.8,
            hovertemplate=f'<b>%{{y}}</b><br>{name}: %{{x:,.0f}} visas<extra></extra>',
        ))

    st.markdown(f'<div class="section-title">Visas otorgadas por pa\u00eds — 5 escenarios</div>',
                unsafe_allow_html=True)
    fig.update_layout(**_plotly_layout(
        barmode='group',
        xaxis=dict(title="Visas asignadas", **_grid()),
        yaxis=dict(autorange='reversed'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor=t['legend_bg'], font=dict(size=10)),
        height=max(600, len(countries) * 45),
        margin=dict(l=140, r=30, t=80, b=50),
    ))
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

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
                    bgcolor=t['legend_bg'], font=dict(size=10)),
        height=max(600, len(countries) * 45),
        margin=dict(l=140, r=30, t=80, b=50),
    ))
    st.plotly_chart(fig_delta, use_container_width=True, theme=None, config={"displayModeBar": False})


# ── Tab 5: Convergencia ─────────────────────────────────────
def _tab_convergence(
    iters: list, means: list, stds: list,
    summary: dict, baseline: tuple, run_idx: int,
) -> None:
    t = _get_theme()

    m = np.array(means)
    s = np.array(stds)
    it = np.array(iters)
    num_runs = summary["num_runs"]

    # ── Intro ──
    st.markdown(f"""
    <div class="info-box">
        El <strong>hipervolumen (HV)</strong> mide el volumen del espacio triobjetivo dominado por el frente de Pareto.
        Un HV mayor significa que el algoritmo encontr\u00f3 soluciones m\u00e1s cercanas al punto ideal.
        Aquí analizamos cómo evoluciona a lo largo de <strong>{int(it[-1])} iteraciones</strong>
        en <strong>{num_runs} corridas independientes</strong> con diferentes semillas aleatorias.
    </div>""", unsafe_allow_html=True)

    # ── Hero metric cards ──
    final_hvs = [load_run(i)["hv_final"] for i in range(num_runs)]
    mean_hv = float(np.mean(final_hvs))
    std_hv = float(np.std(final_hvs))
    best_hv = max(final_hvs)
    worst_hv = min(final_hvs)
    best_run = final_hvs.index(best_hv)
    hv_0 = float(m[0])
    hv_gain = float(m[-1]) - hv_0
    hv_gain_pct = hv_gain / hv_0 * 100 if hv_0 > 0 else 0

    # 95% convergence iteration
    target_95 = float(m[-1]) * 0.95
    conv_iter = int(it[-1])
    for i, val in enumerate(m):
        if val >= target_95:
            conv_iter = int(it[i])
            break
    conv_pct = conv_iter / int(it[-1]) * 100

    # Coefficient of variation
    cv = std_hv / mean_hv * 100 if mean_hv > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    _ms = "font-family:'JetBrains Mono';font-weight:700;"
    for col, label, value, sub, color in [
        (c1, "HV medio final", _fmtd(mean_hv), f"\u00b1{_fmtd(std_hv)} (CV={_fmtd(cv, 1)}%)", t['accent1']),
        (c2, "Mejor corrida", _fmtd(best_hv), f"Corrida #{best_run + 1} (seed={42 + best_run})", t['accent3']),
        (c3, "Ganancia HV", f"+{_fmtd(hv_gain_pct, 1)}%", f"De {_fmtd(hv_0)} a {_fmtd(float(m[-1]))}", t['accent2']),
        (c4, "Convergencia 95%", f"Iter {conv_iter}", f"En el {_fmtd(conv_pct, 0)}% del presupuesto", t['danger']),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;padding:16px;">
                <div style="font-size:0.65rem;color:{t['text_muted']};text-transform:uppercase;
                    font-weight:600;letter-spacing:0.05em;">{label}</div>
                <div style="{_ms}font-size:1.6rem;color:{color};margin:6px 0 2px;">{value}</div>
                <div style="font-size:0.72rem;color:{t['text_subtle']};">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # ── Main convergence chart ──
    st.markdown('<div class="section-title">Evolución del Hipervolumen</div>',
                unsafe_allow_html=True)

    fig = go.Figure()
    # ±1σ band
    fig.add_trace(go.Scatter(
        x=np.concatenate([it, it[::-1]]).tolist(),
        y=np.concatenate([m + s, (m - s)[::-1]]).tolist(),
        fill='toself', fillcolor=_rgba(t['accent1'], 0.12),
        line=dict(width=0), name='\u00b11\u03c3', hoverinfo='skip',
    ))
    # ±2σ band (lighter)
    fig.add_trace(go.Scatter(
        x=np.concatenate([it, it[::-1]]).tolist(),
        y=np.concatenate([m + 2 * s, (m - 2 * s)[::-1]]).tolist(),
        fill='toself', fillcolor=_rgba(t['accent1'], 0.05),
        line=dict(width=0), name='\u00b12\u03c3', hoverinfo='skip',
    ))
    # mean line
    fig.add_trace(go.Scatter(
        x=it.tolist(), y=m.tolist(), mode='lines',
        line=dict(color=t['accent1'], width=3), name='HV medio (30 corridas)',
        hovertemplate='Iter %{x}<br>HV: %{y:.2f}<extra></extra>',
    ))
    # HV final line
    fig.add_hline(y=float(m[-1]), line_dash="dash", line_color=t['accent2'],
                  annotation_text=f"HV final = {_fmtd(float(m[-1]))}",
                  annotation_font=dict(color=t['accent2'], size=11))
    # 95% convergence marker
    fig.add_vline(x=conv_iter, line_dash="dot", line_color=t['danger'],
                  annotation_text=f"95% en iter {conv_iter}",
                  annotation_font=dict(color=t['danger'], size=11))
    # Starting HV
    fig.add_annotation(x=0, y=hv_0, text=f"HV\u2080 = {_fmtd(hv_0)}",
                       showarrow=True, arrowhead=2, arrowcolor=t['text_muted'],
                       font=dict(size=10, color=t['text_muted']),
                       ax=50, ay=-30)

    fig.update_layout(**_plotly_layout(
        xaxis=dict(title="Iteracion", **_grid()),
        yaxis=dict(title="Hipervolumen (HV)", **_grid()),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor=t['legend_bg'], font=dict(size=10)),
        height=480,
        margin=dict(t=60),
    ))
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False},
                    key="conv_main")

    # ── HV explanation ──
    with st.expander("\u00bfC\u00f3mo se calcula el hipervolumen (HV)?"):
        st.markdown(f"""
<div style="font-size:0.85rem;color:{t['tab_text']} !important;line-height:1.8;">
<strong style="color:{t['accent2']};">Idea simple:</strong> imagina que cada soluci\u00f3n del frente de Pareto
es un punto en un espacio 3D (f\u2081, f\u2082, f\u2083). El hipervolumen es el <strong>volumen total</strong>
de la regi\u00f3n que queda entre esos puntos y un punto de referencia fijo.<br><br>

<strong style="color:{t['accent1']};">Paso a paso:</strong><br>
&nbsp;&nbsp;1. Se fija un <strong>punto de referencia</strong> peor que cualquier soluci\u00f3n: (10.0, 16.0, 50,000).<br>
&nbsp;&nbsp;2. Cada soluci\u00f3n \u201cdomina\u201d un rect\u00e1ngulo (en 2D) o un cubo (en 3D) entre ella y la referencia.<br>
&nbsp;&nbsp;3. El HV es la <strong>uni\u00f3n de todos esos cubos</strong>, sin contar doble las zonas que se solapan.<br><br>

<strong style="color:{t['accent3']};">\u00bfPor qu\u00e9 importa?</strong><br>
&nbsp;&nbsp;\u2022 <strong>M\u00e1s HV = mejor frente.</strong> Significa que las soluciones est\u00e1n m\u00e1s lejos de la referencia
(m\u00e1s cerca del \u00f3ptimo) y cubren m\u00e1s diversidad.<br>
&nbsp;&nbsp;\u2022 Si el HV deja de crecer entre iteraciones, el algoritmo ya convergi\u00f3.<br>
&nbsp;&nbsp;\u2022 Nuestro HV final es <strong>{_fmtd(float(m[-1]))}</strong> (en unidades de a\u00f1os\u00b2 \u00d7 visas,
porque es un volumen 3D).
</div>""", unsafe_allow_html=True)

    # ── Velocity of improvement (derivative) ──
    st.markdown('<div class="section-title">Velocidad de Mejora (\u0394HV por iteración)</div>',
                unsafe_allow_html=True)
    st.markdown(f"""<div style="font-size:0.8rem;color:{t['text_muted']};margin-bottom:8px;">
        Derivada discreta del HV medio: muestra en qué iteraciones el algoritmo aprende más rápido.
        Las barras altas indican saltos de calidad; la zona plana indica convergencia.</div>""",
        unsafe_allow_html=True)

    delta_hv = np.diff(m)
    # Smooth with moving average of 10 for readability
    window = min(10, len(delta_hv))
    if window > 1:
        kernel = np.ones(window) / window
        delta_smooth = np.convolve(delta_hv, kernel, mode='valid')
        delta_iters = it[window:len(delta_smooth) + window]
    else:
        delta_smooth = delta_hv
        delta_iters = it[1:]

    fig_vel = go.Figure()
    fig_vel.add_trace(go.Scatter(
        x=delta_iters.tolist(), y=delta_smooth.tolist(),
        mode='lines', fill='tozeroy',
        fillcolor=_rgba(t['accent3'], 0.15),
        line=dict(color=t['accent3'], width=2),
        hovertemplate='Iter %{x}<br>\u0394HV: %{y:.4f}<extra></extra>',
        name='\u0394HV (suavizado)',
    ))
    fig_vel.add_hline(y=0, line_color=t['text_subtle'], line_width=1)
    fig_vel.update_layout(**_plotly_layout(
        xaxis=dict(title="Iteracion", **_grid()),
        yaxis=dict(title="\u0394HV por iteración", **_grid()),
        height=280,
        margin=dict(t=20, b=40),
    ))
    st.plotly_chart(fig_vel, use_container_width=True, theme=None, config={"displayModeBar": False},
                    key="conv_velocity")

    # ── Row: Distribution + Per-run bars ──
    st.markdown('<div class="section-title">Análisis Estadístico de las 30 Corridas</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div style="font-size:0.9rem;color:{t["text"]};font-weight:600;'
                    f'margin-bottom:4px;">Distribución del HV Final</div>',
                    unsafe_allow_html=True)
        # Violin + strip instead of plain box
        fig_box = go.Figure()
        fig_box.add_trace(go.Violin(
            y=final_hvs, box_visible=True, meanline_visible=True,
            points='all', jitter=0.4, pointpos=-1.5,
            marker=dict(color=t['accent1'], size=6, opacity=0.7,
                        line=dict(width=1, color=t['marker_line'])),
            line=dict(color=t['accent3']),
            fillcolor=_rgba(t['accent1'], 0.2),
            name='30 corridas',
            hovertemplate='HV: %{y:.2f}<extra></extra>',
        ))
        fig_box.add_hline(y=mean_hv, line_dash="dash", line_color=t['accent2'],
                          annotation_text=f"\u03bc = {_fmtd(mean_hv)}",
                          annotation_font=dict(size=10, color=t['accent2']))
        fig_box.update_layout(**_plotly_layout(
            yaxis=dict(title="Hipervolumen", **_grid()), height=380,
            margin=dict(l=60, r=20, t=20, b=40),
        ))
        st.plotly_chart(fig_box, use_container_width=True, theme=None,
                        config={"displayModeBar": False}, key="conv_violin")

    with col2:
        st.markdown(f'<div style="font-size:0.9rem;color:{t["text"]};font-weight:600;'
                    f'margin-bottom:4px;">HV por Corrida (ordenado)</div>',
                    unsafe_allow_html=True)
        # Sort runs by HV for visual ranking
        sorted_runs = sorted(enumerate(final_hvs), key=lambda x: x[1], reverse=True)
        bar_x = [f"R{idx + 1}" for idx, _ in sorted_runs]
        bar_y = [hv for _, hv in sorted_runs]
        # Color gradient from best to worst
        n_bars = len(bar_y)
        bar_colors = [
            f'rgba({int(60 + (255 - 60) * i / max(n_bars - 1, 1))},'
            f'{int(165 - 100 * i / max(n_bars - 1, 1))},'
            f'{int(250 - 100 * i / max(n_bars - 1, 1))},0.85)'
            for i in range(n_bars)
        ]
        fig_bar = go.Figure(go.Bar(
            x=bar_x, y=bar_y, marker_color=bar_colors,
            hovertemplate='%{x}<br>HV: %{y:.2f}<extra></extra>',
        ))
        fig_bar.add_hline(y=mean_hv, line_dash="dash", line_color=t['accent2'],
                          annotation_text=f"\u03bc = {_fmtd(mean_hv)}",
                          annotation_font=dict(size=10, color=t['accent2']))
        fig_bar.update_layout(**_plotly_layout(
            yaxis=dict(title="Hipervolumen", **_grid()),
            xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
            height=380,
            margin=dict(l=60, r=20, t=20, b=60),
        ))
        st.plotly_chart(fig_bar, use_container_width=True, theme=None,
                        config={"displayModeBar": False}, key="conv_bars")

    # ── Pareto size per run ──
    st.markdown('<div class="section-title">Soluciones Pareto por Corrida</div>',
                unsafe_allow_html=True)

    pareto_sizes = [load_run(i)["num_pareto"] for i in range(num_runs)]
    mean_ps = float(np.mean(pareto_sizes))

    col_a, col_b = st.columns([2, 1])
    with col_a:
        ps_colors = [t['accent3'] if ps >= mean_ps else t['mid_blue'] for ps in pareto_sizes]
        fig_ps = go.Figure(go.Bar(
            x=[f"R{i + 1}" for i in range(num_runs)],
            y=pareto_sizes, marker_color=ps_colors,
            text=pareto_sizes, textposition='outside',
            textfont=dict(size=8, color=t['text_muted']),
            hovertemplate='Corrida %{x}<br>Soluciones Pareto: %{y}<extra></extra>',
        ))
        fig_ps.add_hline(y=mean_ps, line_dash="dash", line_color=t['accent2'],
                         annotation_text=f"\u03bc = {_fmtd(mean_ps, 1)}",
                         annotation_font=dict(size=10, color=t['accent2']))
        fig_ps.update_layout(**_plotly_layout(
            yaxis=dict(title="# Soluciones no dominadas", **_grid()),
            xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
            height=320,
            margin=dict(l=60, r=20, t=20, b=60),
        ))
        st.plotly_chart(fig_ps, use_container_width=True, theme=None,
                        config={"displayModeBar": False}, key="conv_pareto_sizes")

    with col_b:
        st.markdown(f"""
        <div class="metric-card" style="padding:18px;margin-top:10px;">
            <div style="font-size:0.65rem;color:{t['text_muted']};text-transform:uppercase;
                font-weight:600;">Soluciones Pareto</div>
            <div style="{_ms}font-size:1.4rem;color:{t['accent3']};margin:6px 0;">
                {_fmtd(mean_ps, 1)} \u00b1 {_fmtd(float(np.std(pareto_sizes)), 1)}</div>
            <div style="font-size:0.75rem;color:{t['text_subtle']};">promedio por corrida</div>
            <div style="margin-top:12px;font-size:0.8rem;">
                <span style="color:{t['text_muted']};">Min:</span>
                <span style="color:{t['text']};font-weight:600;"> {min(pareto_sizes)}</span>
                <span style="color:{t['text_muted']};margin-left:12px;">Max:</span>
                <span style="color:{t['text']};font-weight:600;"> {max(pareto_sizes)}</span>
            </div>
            <div style="margin-top:8px;font-size:0.8rem;">
                <span style="color:{t['text_muted']};">Combinadas:</span>
                <span style="color:{t['accent2']};font-weight:600;"> {summary['combined_pareto_size']}</span>
            </div>
            <div style="font-size:0.7rem;color:{t['text_subtle']};margin-top:10px;font-style:italic;
                line-height:1.3;">
                De {sum(pareto_sizes)} soluciones Pareto individuales, al combinar las 30 corridas
                y eliminar las dominadas, quedan {summary['combined_pareto_size']} soluciones únicas
                en el frente final.</div>
        </div>""", unsafe_allow_html=True)

    # ── Individual run explorer ──
    st.markdown('<div class="section-title">Explorador de Corrida Individual</div>',
                unsafe_allow_html=True)
    st.markdown(f"""<div style="font-size:0.8rem;color:{t['text_muted']};margin-bottom:8px;">
        Selecciona una corrida en el sidebar para ver su frente de Pareto individual
        comparado con el FIFO.</div>""", unsafe_allow_html=True)

    run_data = load_run(run_idx)
    rf1 = [p["f1"] for p in run_data["pareto_front"]]
    rf2 = [p["f2"] for p in run_data["pareto_front"]]
    rf3 = [p["f3"] for p in run_data["pareto_front"]]

    # Rank of this run
    rank = sorted(final_hvs, reverse=True).index(run_data["hv_final"]) + 1
    rank_color = t['accent3'] if rank <= 10 else (t['accent2'] if rank <= 20 else t['danger'])

    rc1, rc2, rc3 = st.columns(3)
    for col, lbl, val, clr in [
        (rc1, f"Corrida #{run_idx + 1}", f"HV = {_fmtd(run_data['hv_final'])}",
         t['accent1']),
        (rc2, "Ranking", f"#{rank} de {num_runs}", rank_color),
        (rc3, "Soluciones Pareto", f"{run_data['num_pareto']}", t['accent3']),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;padding:12px;">
                <div style="font-size:0.65rem;color:{t['text_muted']};text-transform:uppercase;
                    font-weight:600;">{lbl}</div>
                <div style="{_ms}font-size:1.2rem;color:{clr};margin-top:4px;">{val}</div>
            </div>""", unsafe_allow_html=True)

    fig_run = go.Figure()
    fig_run.add_trace(go.Scatter(
        x=rf1, y=rf2, mode='markers',
        marker=dict(size=9, color=t['accent1'],
                    line=dict(width=1, color=t['marker_line'])),
        name=f"Corrida {run_idx + 1} ({run_data['num_pareto']} sol.)",
        hovertemplate='f\u2081: %{x:.3f}<br>f\u2082: %{y:.2f}<extra></extra>',
    ))
    fig_run.add_trace(go.Scatter(
        x=[baseline[0]], y=[baseline[1]], mode='markers',
        marker=dict(size=18, color=t['danger'], symbol='star',
                    line=dict(width=2, color=t['marker_line'])),
        name='FIFO (actual)',
        hovertemplate='FIFO<br>f\u2081: %{x:.3f}<br>f\u2082: %{y:.2f}<extra></extra>',
    ))
    fig_run.update_layout(**_plotly_layout(
        xaxis=dict(title="f\u2081 \u2014 Carga de espera (a\u00f1os)", **_grid()),
        yaxis=dict(title="f\u2082 \u2014 Disparidad (a\u00f1os)", **_grid()),
        legend=dict(bgcolor=t['legend_bg']), height=420,
        margin=dict(t=20),
    ))
    st.plotly_chart(fig_run, use_container_width=True, theme=None,
                    config={"displayModeBar": False}, key="conv_run_pareto")

    # ── Summary table ──
    with st.expander("Ver tabla completa de las 30 corridas"):
        import pandas as pd
        df_runs = pd.DataFrame({
            "Corrida": [f"#{i + 1}" for i in range(num_runs)],
            "Seed": [42 + i for i in range(num_runs)],
            "HV Final": [f"{hv:.2f}" for hv in final_hvs],
            "# Pareto": pareto_sizes,
            "Ranking HV": [sorted(final_hvs, reverse=True).index(hv) + 1 for hv in final_hvs],
        })
        st.dataframe(df_runs.sort_values("Ranking HV"), use_container_width=True, hide_index=True)


# ── Tab 6: Datos y Fuentes ───────────────────────────────────
def _tab_data_sources(data: dict) -> None:
    t = _get_theme()

    st.markdown('<div class="section-title">Transparencia de Datos</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        Este modelo combina datos de múltiples fuentes oficiales. Algunos países tienen datos completos
        publicados por USCIS; para otros, se estimaron las cifras a partir del DHS Yearbook.
        <strong>La transparencia sobre qué es real y qué es estimado es fundamental para la credibilidad del modelo.</strong>
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
                Estimado (DHS Yearbook FY2023 / demanda sintética)</span>
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

    # Discrete colorscale: REAL=teal, EST=warm sand, EST-DEM=slate blue
    c_real = "#0d9488"
    c_est = "#a8956e"
    c_calc = "#7888b8"
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
        textfont=dict(size=9, color=t['text'], family='Inter'),
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
                    bgcolor=t['legend_bg'], font=dict(size=11)),
    ))
    st.plotly_chart(fig_prov, use_container_width=True, theme=None, config={"displayModeBar": False})

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
         "Demanda sintética estimada para categorías \u201cCurrent\u201d sin backlog visible."),
        ("Demanda \u2014 16 pa\u00edses derivados de \u201cAll Other\u201d",
         "EST",
         "DHS Yearbook FY2023, tablas de inmigración calificada. Se usó la proporción histórica de cada país."),
        ("Demanda \u2014 Resto del Mundo",
         "EST",
         "Residuo: total USCIS \u201cAll Other\u201d menos los 16 pa\u00edses estimados."),
        ("Fechas de prioridad (todas)",
         "REAL",
         "U.S. Department of State, Visa Bulletin abril 2026, \u201cFinal Action Dates for Employment-Based Preference Cases.\u201d"),
        ("Tiempo de espera (w)",
         "CALC",
         "w = T_actual (2026) \u2212 a\u00f1o de fecha de prioridad. Derivado automáticamente."),
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
         "Asignación estatutaria anual para visas de empleo."),
    ]

    for title, src_type, desc in sources:
        src_class = "source-real" if src_type == "REAL" else ("source-est" if src_type == "EST" else "source-calc")
        st.markdown(f"""
        <div class="gloss-card">
            <span class="source-badge {src_class}" style="margin-bottom:6px;">{src_type}</span>
            <h4 style="margin-top:4px;">{title}</h4>
            <p>{desc}</p>
        </div>""", unsafe_allow_html=True)

    # f₃ subsection
    st.markdown(f"""
<div style="background:{t['card_bg']};border-radius:12px;padding:20px;margin:1.5rem 0;
    border:1px solid {t['card_border']};">
<h4 style="color:{t['accent2']};margin-top:0;">Impacto en f\u2083 (desperdicio de visas)</h4>
<p style="color:{t['text']};line-height:1.6;">
El desperdicio de visas (f\u2083) surge de la interacci\u00f3n entre los topes por pa\u00eds (R2)
y por categor\u00eda (R3). Con los datos de 21 pa\u00edses:</p>
<ul style="color:{t['text']};line-height:1.8;">
<li>Categor\u00eda EB-4 tiene 217,438 peticiones pero solo 9,940 slots</li>
<li>Afganist\u00e1n (100K EB-4) e Irak (80K EB-4) saturan su cap de 25,620 r\u00e1pidamente</li>
<li>Resultado: EB-4 tiene slots disponibles que ning\u00fan pa\u00eds puede usar</li>
<li>El FIFO actual desperdicia <strong>17,540 visas</strong>; MOHHO encuentra soluciones con <strong>0 desperdicio</strong></li>
</ul>
</div>""", unsafe_allow_html=True)


# ── Tab 7: Acerca del Modelo ────────────────────────────────
def _tab_math(data: dict, summary: dict = None) -> None:
    """Tab: Modelo Matemático completo, explicado para cualquier audiencia."""
    t = _get_theme()
    mono = "font-family:'JetBrains Mono',monospace;"
    eq_bg = f"background:{t['info_bg']};border-radius:10px;padding:12px 16px;margin:8px 0;"
    eq_color = f"color:{t['accent2']} !important;"
    G = len(data['countries']) * len(data['categories'])

    # ── Intro: El problema en lenguaje humano ──
    st.markdown('<div class="section-title">El Problema en Palabras Simples</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box" style="font-size:0.9rem;line-height:1.8;">
        Imagina un hospital con <strong style="color:{t['accent2']};">{_fmt(data['total_visas'])} camas</strong>
        y <strong style="color:{t['accent2']};">{_fmt(data['total_demand'])} pacientes</strong> de
        <strong style="color:{t['accent2']};">{len(data['countries'])} ciudades</strong> esperando ingresar.
        Algunos llevan <strong style="color:{t['danger']};">13 a\u00f1os en lista de espera</strong>.
        <br><br>
        <strong>Regla actual (FIFO):</strong> se atiende por orden de llegada. Suena justo, pero hay una restricci\u00f3n:
        <em>ning\u00fan barrio puede ocupar m\u00e1s del 7% de las camas</em>.
        El resultado: los pacientes de las dos ciudades m\u00e1s grandes (India y China, que son el 80% de la demanda)
        llenan su cuota en minutos, y <strong>sus miles de pacientes restantes siguen esperando</strong>.
        Mientras tanto, camas asignadas a ciudades peque\u00f1as quedan vac\u00edas \u2014
        <strong style="color:{t['danger']};">{_fmt(int(summary['baseline']['f3'] if summary else 17540))} camas desperdiciadas</strong>.
        <br><br>
        <strong>El dilema tiene tres dimensiones:</strong>
        <br>
        &nbsp;&nbsp;\u2022 <strong style="color:{t['accent1']};">Urgencia (f\u2081):</strong>
        \u00bfDamos camas a los que m\u00e1s tiempo llevan esperando? Eso favorece a India y China, pero concentra todos los recursos en dos ciudades.
        <br>
        &nbsp;&nbsp;\u2022 <strong style="color:{t['accent2']};">Equidad (f\u2082):</strong>
        \u00bfRepartimos para que todas las ciudades esperen lo mismo? Eso reduce la brecha, pero deja sin atender a los m\u00e1s urgentes.
        <br>
        &nbsp;&nbsp;\u2022 <strong style="color:{t['accent3']};">Eficiencia (f\u2083):</strong>
        \u00bfLlenamos todas las camas? El orden de procesamiento importa: si llenas la cuota de India primero, quedan camas de EB-1 vac\u00edas que nadie puede usar.
        <br><br>
        <strong>No existe una soluci\u00f3n perfecta</strong> que gane en las tres dimensiones a la vez.
        Lo que s\u00ed existe es un <strong>mapa de {summary['combined_pareto_size'] if summary else 371} compromisos \u00f3ptimos</strong>:
        cada uno prioriza de forma diferente, y ninguno puede mejorar una dimensi\u00f3n sin empeorar otra.
        Eso es el frente de Pareto \u2014 y es lo que el algoritmo calcula.
    </div>""", unsafe_allow_html=True)

    # ── What are we deciding? ──
    st.markdown('<div class="section-title">¿Qué Decidimos? (Variable de Decisión)</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
        <div class="metric-card">
            <div style="font-size:0.7rem;color:{t['text_muted']};text-transform:uppercase;font-weight:600;">
                En lenguaje cotidiano</div>
            <p style="font-size:0.85rem;color:{t['tab_text']} !important;margin:8px 0;line-height:1.6;">
                Para cada combinación de país + categoría (ej. "India EB-2"),
                decidimos <strong>cuántas visas le damos</strong> este año.
                <br><br>
                Hay {len(data['countries'])} países y {len(data['categories'])} categorías = <strong>{G} grupos</strong>.
                Cada grupo tiene un número diferente: India EB-2 podría recibir 5,000 visas,
                Canada EB-1 podría recibir 200, etc.</p>
        </div>
        <div class="metric-card">
            <div style="font-size:0.7rem;color:{t['text_muted']};text-transform:uppercase;font-weight:600;">
                En lenguaje matematico</div>
            <div style="{eq_bg}">
                <div style="{mono}{eq_color}text-align:center;font-size:1rem;">
                    x<sub>g</sub> \u2208 \u2124\u207a\u2080 &nbsp;:&nbsp; visas asignadas al grupo g</div>
            </div>
            <p style="font-size:0.82rem;color:{t['tab_text']} !important;margin:6px 0;line-height:1.5;">
                El vector <strong>x</strong> = (x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>{G}</sub>)
                tiene {G} números enteros no negativos.<br>
                Cada x<sub>g</sub> dice cuántas visas recibe el grupo g.</p>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── The three objectives ──
    st.markdown('<div class="section-title">Los Tres Objetivos (\u00bfQu\u00e9 Queremos Lograr?)</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box" style="font-size:0.85rem;">
        Queremos tres cosas a la vez, pero est\u00e1n en conflicto: si priorizas a India (que espera m\u00e1s),
        otros pa\u00edses se quedan sin visas y la desigualdad crece. Si repartes equitativamente,
        India sigue esperando d\u00e9cadas y se desperdician visas al chocar con l\u00edmites por pa\u00eds.
        No hay soluci\u00f3n perfecta \u2014 hay <strong>{summary['combined_pareto_size'] if summary else 371} compromisos</strong>.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-color:{t['accent1']}44;">
            <div style="font-size:0.8rem;color:{t['accent1']} !important;font-weight:700;
                text-transform:uppercase;margin-bottom:6px;">
                Objetivo 1 (f\u2081): Reducir la Espera</div>
            <p style="font-size:0.85rem;color:{t['tab_text']} !important;line-height:1.6;margin:0 0 10px;">
                <strong>En simple:</strong> darle visas a la gente que lleva <em>más años esperando</em>.
                Si India lleva 13 años y Canada 4, priorizamos a India.</p>
            <div style="{eq_bg}">
                <div style="{mono}{eq_color}text-align:center;font-size:0.95rem;">
                    f\u2081(x) = \u03a3<sub>g</sub>
                    (n<sub>g</sub> \u2212 x<sub>g</sub>) \u00b7 w<sub>g</sub>
                    &nbsp;/&nbsp; \u03a3<sub>g</sub> n<sub>g</sub></div>
            </div>
            <div style="font-size:0.8rem;color:{t['tab_text']} !important;line-height:1.6;">
                <strong>n<sub>g</sub></strong> = cuántas personas esperan en el grupo g<br>
                <strong>x<sub>g</sub></strong> = cuántas visas le damos al grupo g<br>
                <strong>w<sub>g</sub></strong> = años que llevan esperando<br>
                <strong>(n<sub>g</sub> \u2212 x<sub>g</sub>)</strong> = personas que se <em>quedan sin visa</em><br><br>
                La fórmula dice: de los que no recibieron visa, ¿cuántos años de espera acumulan?
                Dividimos entre el total para normalizar. <strong>Queremos que sea lo más bajo posible.</strong></div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-color:{t['accent2']}44;">
            <div style="font-size:0.8rem;color:{t['accent2']} !important;font-weight:700;
                text-transform:uppercase;margin-bottom:6px;">
                Objetivo 2 (f\u2082): Reducir la Desigualdad</div>
            <p style="font-size:0.85rem;color:{t['tab_text']} !important;line-height:1.6;margin:0 0 10px;">
                <strong>En simple:</strong> que la espera sea <em>parecida</em> entre países.
                Si India espera 13 años y Canada 1, la brecha es 12 — queremos reducirla.</p>
            <div style="{eq_bg}">
                <div style="{mono}{eq_color}text-align:center;font-size:0.95rem;">
                    W\u0304<sub>c</sub> = \u03a3 x<sub>g</sub>\u00b7w<sub>g</sub>
                    &nbsp;/&nbsp; \u03a3 x<sub>g</sub>
                    &nbsp;&nbsp; (por país c)</div>
                <div style="{mono}{eq_color}text-align:center;font-size:0.95rem;margin-top:6px;">
                    f\u2082(x) = max |W\u0304<sub>c1</sub> \u2212 W\u0304<sub>c2</sub>|</div>
            </div>
            <div style="font-size:0.8rem;color:{t['tab_text']} !important;line-height:1.6;">
                <strong>W\u0304<sub>c</sub></strong> = espera promedio de las visas que le dimos al país c<br>
                Si un país <strong>no recibe visas</strong>: W\u0304<sub>c</sub> = su espera máxima (castigo)<br><br>
                La fórmula busca el par de países con la <strong>mayor diferencia</strong> de espera.
                Esa brecha máxima es f\u2082. <strong>Queremos que sea lo más baja posible.</strong></div>
        </div>""", unsafe_allow_html=True)

    # f3 objective card
    st.markdown(f"""
    <div class="metric-card" style="border-color:{t['accent3']}44;margin:1rem 0;">
        <div style="font-size:0.8rem;color:{t['accent3']} !important;font-weight:700;
            text-transform:uppercase;margin-bottom:6px;">
            Objetivo 3 (f\u2083): Minimizar el Desperdicio de Visas</div>
        <p style="font-size:0.85rem;color:{t['tab_text']} !important;line-height:1.6;margin:0 0 10px;">
            <strong>En simple:</strong> usar la mayor cantidad posible de las {_fmt(data['total_visas'])} visas disponibles.
            Cada visa no asignada es una oportunidad perdida.</p>
        <div style="{eq_bg}">
            <div style="{mono}{eq_color}text-align:center;font-size:0.95rem;">
                f\u2083(x) = V \u2212 \u03a3<sub>g</sub> x<sub>g</sub></div>
        </div>
        <div style="font-size:0.8rem;color:{t['tab_text']} !important;line-height:1.6;">
            <strong>V</strong> = {_fmt(data['total_visas'])} visas disponibles por a\u00f1o<br>
            <strong>\u03a3 x<sub>g</sub></strong> = total de visas efectivamente asignadas<br><br>
            FIFO desperdicia <strong style="color:{t['danger']};">{_fmt(int(summary['baseline']['f3'] if summary else 17540))}</strong> visas
            porque choca con l\u00edmites por pa\u00eds antes de agotar el presupuesto.
            <strong>Queremos que f\u2083 sea lo m\u00e1s bajo posible.</strong></div>
    </div>""", unsafe_allow_html=True)

    # ── Why conflict ──
    st.markdown('<div class="section-title">\u00bfPor qu\u00e9 No se Puede Ganar en los Tres?</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-color:{t['accent1']}44;">
            <div style="font-size:0.75rem;color:{t['accent1']} !important;font-weight:700;
                text-transform:uppercase;">Estrategia A: "Ayudar a los que más esperan"</div>
            <p style="font-size:0.82rem;color:{t['tab_text']} !important;margin:8px 0;line-height:1.5;">
                Damos todas las visas posibles a India y China primero.
                Se reduce mucho la espera global, pero Canada, Alemania y otros
                con poca espera se quedan sin nada.</p>
            <div style="display:flex;gap:1rem;margin-top:8px;">
                <div style="text-align:center;flex:1;padding:8px;background:{t['info_bg']};border-radius:8px;">
                    <div style="font-size:0.6rem;color:{t['text_muted']};text-transform:uppercase;">f\u2081 espera</div>
                    <div style="{mono}font-size:1rem;color:{t['accent3']} !important;">Baja</div>
                </div>
                <div style="text-align:center;flex:1;padding:8px;background:{t['info_bg']};border-radius:8px;">
                    <div style="font-size:0.6rem;color:{t['text_muted']};text-transform:uppercase;">f\u2082 brecha</div>
                    <div style="{mono}font-size:1rem;color:{t['danger']} !important;">Alta</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-color:{t['accent3']}44;">
            <div style="font-size:0.75rem;color:{t['accent3']} !important;font-weight:700;
                text-transform:uppercase;">Estrategia B: "Tratar a todos igual"</div>
            <p style="font-size:0.82rem;color:{t['tab_text']} !important;margin:8px 0;line-height:1.5;">
                Repartimos visas para que todos los países tengan esperas similares.
                Muy equitativo, pero India sigue con colas de décadas porque no recibe suficientes.</p>
            <div style="display:flex;gap:1rem;margin-top:8px;">
                <div style="text-align:center;flex:1;padding:8px;background:{t['info_bg']};border-radius:8px;">
                    <div style="font-size:0.6rem;color:{t['text_muted']};text-transform:uppercase;">f\u2081 espera</div>
                    <div style="{mono}font-size:1rem;color:{t['danger']} !important;">Alta</div>
                </div>
                <div style="text-align:center;flex:1;padding:8px;background:{t['info_bg']};border-radius:8px;">
                    <div style="font-size:0.6rem;color:{t['text_muted']};text-transform:uppercase;">f\u2082 brecha</div>
                    <div style="{mono}font-size:1rem;color:{t['accent3']} !important;">Baja</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box" style="border-left:4px solid {t['accent2']};font-size:0.85rem;">
        <strong>Ninguna estrategia gana en los tres:</strong> A es mejor en f\u2081 pero peor en f\u2082;
        B es mejor en f\u2082 pero peor en f\u2081; y las soluciones que maximizan utilizaci\u00f3n (f\u2083 bajo)
        no siempre minimizan espera o disparidad. Esto se llama <strong>conflicto estructural</strong>
        y es la raz\u00f3n por la que no hay UNA soluci\u00f3n \u00f3ptima, sino un <strong>frente de {summary['combined_pareto_size']} compromisos</strong>.
    </div>""", unsafe_allow_html=True)

    # ── Why triobjetivo ──
    st.markdown('<div class="section-title">\u00bfPor qu\u00e9 3 Objetivos?</div>',
                unsafe_allow_html=True)

    # Compute actual visa usage stats from Pareto solutions
    pareto_all = load_pareto_csv()[0]
    pareto_visa_stats = compute_pareto_visas(
        tuple(sorted(pareto_all, key=lambda x: x[0])), data['total_visas'])
    pv_vals = list(pareto_visa_stats.values())
    pv_min, pv_max, pv_mean = min(pv_vals), max(pv_vals), sum(pv_vals) / len(pv_vals)
    f3_vals = [p[2] for p in pareto_all]
    f3_min, f3_max = min(f3_vals), max(f3_vals)

    st.markdown(f"""
    <div class="info-box" style="font-size:0.85rem;line-height:1.7;">
        El tercer objetivo <strong>f\u2083 = V \u2212 \u03a3x<sub>g</sub></strong> mide las visas desperdiciadas.
        Los datos emp\u00edricos muestran que f\u2083 var\u00eda significativamente entre soluciones:
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    _mono = "font-family:'JetBrains Mono';font-weight:700;"
    with c1:
        st.markdown(f"""
        <div class="metric-card" style="border-color:{t['danger']}44;text-align:center;">
            <div style="font-size:0.65rem;color:{t['text_muted']};text-transform:uppercase;font-weight:600;">
                FIFO (sistema actual)</div>
            <div style="{_mono}font-size:1.3rem;color:{t['danger']};margin:6px 0;">
                f\u2083 = {_fmt(int(summary['baseline']['f3'] if summary else 17540))}</div>
            <div style="font-size:0.75rem;color:{t['tab_text']} !important;">
                Desperdicia <strong style="color:{t['danger']};">{_fmt(data['total_visas'] - data['fifo_used'])}</strong>
                visas por chocar con l\u00edmites por pa\u00eds</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card" style="border-color:{t['accent2']}44;text-align:center;">
            <div style="font-size:0.65rem;color:{t['text_muted']};text-transform:uppercase;font-weight:600;">
                MOHHO ({summary['combined_pareto_size']} soluciones Pareto)</div>
            <div style="{_mono}font-size:1.3rem;color:{t['accent2']};margin:6px 0;">
                f\u2083: {_fmt(int(f3_min))} a {_fmt(int(f3_max))}</div>
            <div style="font-size:0.75rem;color:{t['tab_text']} !important;">
                Visas usadas: {_fmt(pv_min)} a {_fmt(pv_max)}<br>
                Promedio: {_fmt(int(pv_mean))} visas ({int(pv_mean) * 100 // data['total_visas']}%)</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card" style="border-color:{t['accent3']}44;text-align:center;">
            <div style="font-size:0.65rem;color:{t['text_muted']};text-transform:uppercase;font-weight:600;">
                Mejor utilizaci\u00f3n (f\u2083 m\u00ednimo)</div>
            <div style="{_mono}font-size:1.3rem;color:{t['accent3']};margin:6px 0;">
                f\u2083 = {_fmt(int(f3_min))}</div>
            <div style="font-size:0.75rem;color:{t['tab_text']} !important;">
                {_fmt(data['total_visas'] - int(f3_min))} visas asignadas<br>
                ({(data['total_visas'] - int(f3_min)) * 100 // data['total_visas']}% utilizaci\u00f3n)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box" style="font-size:0.85rem;line-height:1.7;">
        <strong>\u00bfPor qu\u00e9 restaurar f\u2083 como objetivo?</strong><br><br>
        f\u2083 (visas no usadas) <em>var\u00eda significativamente</em> entre soluciones \u2014 de {_fmt(int(f3_min))} a {_fmt(int(f3_max))}.
        Las soluciones que priorizan equidad (f\u2082 bajo) tienden a usar menos visas porque
        reparten entre m\u00e1s pa\u00edses y chocan con m\u00e1s l\u00edmites. Esto convierte a f\u2083 en un
        objetivo genuinamente en conflicto con f\u2082.<br><br>
        <strong>Decisi\u00f3n:</strong> se modela como <strong>triobjetivo min\u007bf\u2081, f\u2082, f\u2083\u007d</strong>.
        El frente de Pareto pasa de ser una curva 2D a una <strong>superficie 3D</strong>
        con {summary['combined_pareto_size']} soluciones no dominadas. El presupuesto total de {_fmt(data['total_visas'])} visas
        se sigue respetando como <strong>restricci\u00f3n R1</strong> (\u03a3x<sub>g</sub> \u2264 V),
        mientras que f\u2083 mide cu\u00e1nto de ese presupuesto se logra asignar efectivamente.
    </div>""", unsafe_allow_html=True)

    # ── Constraints (with real-world meaning) ──
    st.markdown('<div class="section-title">Las 6 Reglas del Juego (Restricciones)</div>',
                unsafe_allow_html=True)
    st.markdown(f"""<p style="font-size:0.85rem;color:{t['tab_text']} !important;">
        No podemos repartir visas como se nos antoje. La ley de inmigración de EE.UU.
        impone reglas estrictas:</p>""", unsafe_allow_html=True)

    constraints = [
        ("R1", "Presupuesto anual",
         f"\u03a3 x<sub>g</sub> \u2264 {_fmt(data['total_visas'])}",
         "No puedes dar más visas de las que existen.",
         f"Solo hay {_fmt(data['total_visas'])} visas EB por año. Ni una más."),
        ("R2", "Limite por país (7%)",
         f"\u03a3 x<sub>g</sub> \u2264 {_fmt(25620)} por país",
         "Ningún país puede acaparar todo.",
         "Cada país recibe máximo 7% del total. Por eso India, con 70% de la demanda, tiene colas de décadas."),
        ("R3", "Límite por categoría",
         "\u03a3 x<sub>g</sub> \u2264 K<sub>j</sub><sup>eff</sup> por categoría",
         "Cada tipo de visa tiene su tope.",
         "EB-1 (extraordinarios), EB-2 (profesionales), etc. tienen cupos propios, ajustados por spillover."),
        ("R4", "No dar de mas",
         "x<sub>g</sub> \u2264 n<sub>g</sub>",
         "No puedes dar más visas de las que piden.",
         "Si Canada EB-5 pide 100, no puedes darle 200."),
        ("R5", "Enteros",
         "x<sub>g</sub> \u2208 enteros \u2265 0",
         "No hay media visa.",
         "Cada visa es un permiso real — no se puede fraccionar."),
        ("R6", "FIFO interno",
         "Orden por fecha de prioridad",
         "Dentro de cada grupo, los más antiguos primero.",
         "Aunque reordenamos ENTRE grupos, DENTRO de cada grupo se respeta el orden de llegada."),
    ]

    for rid, name, formula, simple, detail in constraints:
        st.markdown(f"""
        <div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:10px;
            padding:12px 16px;background:{t['card_bg']};border-radius:12px;
            border:1px solid {t['card_border']};border-left:4px solid {t['accent1']};">
            <div style="min-width:36px;text-align:center;">
                <div style="{mono}font-weight:700;font-size:0.9rem;color:{t['accent1']} !important;">{rid}</div>
            </div>
            <div style="flex:1;">
                <div style="font-weight:700;font-size:0.85rem;color:{t['text']} !important;">{name}:
                    <span style="font-weight:400;color:{t['tab_text']} !important;">{simple}</span></div>
                <div style="font-size:0.78rem;color:{t['text_subtle']} !important;margin-top:4px;">{detail}</div>
                <div style="{mono}font-size:0.75rem;color:{t['accent2']} !important;margin-top:4px;
                    opacity:0.8;">{formula}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Spillover ──
    st.markdown('<div class="section-title">Spillover: Las Visas que Sobran se Pasan</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box" style="font-size:0.85rem;line-height:1.7;">
        <strong>Ejemplo:</strong> EB-4 (visas especiales) y EB-5 (inversores) tienen poca demanda.
        Sus visas sobrantes <strong>suben</strong> a EB-1. Si a EB-1 le sobran, <strong>bajan</strong>
        a EB-2, y de EB-2 a EB-3. Así las visas no se desperdician.
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:center;gap:8px;flex-wrap:wrap;
        padding:1.2rem;background:{t['card_bg']};border-radius:16px;
        border:1px solid {t['card_border']};margin-bottom:12px;">
        <div style="text-align:center;padding:10px 14px;background:{t['info_bg']};border-radius:10px;">
            <div style="font-size:0.65rem;color:{t['text_muted']};text-transform:uppercase;">EB-4, EB-5</div>
            <div style="{mono}font-size:0.8rem;color:{t['accent3']} !important;">Sobran visas</div>
        </div>
        <div style="color:{t['accent2']};font-size:1.2rem;">\u2192</div>
        <div style="text-align:center;padding:10px 14px;background:{t['info_bg']};border-radius:10px;">
            <div style="font-size:0.65rem;color:{t['text_muted']};text-transform:uppercase;">EB-1</div>
            <div style="{mono}font-size:0.8rem;color:{t['accent2']} !important;">Recibe excedente</div>
        </div>
        <div style="color:{t['accent2']};font-size:1.2rem;">\u2192</div>
        <div style="text-align:center;padding:10px 14px;background:{t['info_bg']};border-radius:10px;">
            <div style="font-size:0.65rem;color:{t['text_muted']};text-transform:uppercase;">EB-2</div>
            <div style="{mono}font-size:0.8rem;color:{t['accent2']} !important;">Recibe sobrante de EB-1</div>
        </div>
        <div style="color:{t['accent2']};font-size:1.2rem;">\u2192</div>
        <div style="text-align:center;padding:10px 14px;background:{t['info_bg']};border-radius:10px;">
            <div style="font-size:0.65rem;color:{t['text_muted']};text-transform:uppercase;">EB-3</div>
            <div style="{mono}font-size:0.8rem;color:{t['accent2']} !important;">Recibe sobrante de EB-2</div>
        </div>
    </div>""", unsafe_allow_html=True)

    with st.expander("Ver formulas de spillover"):
        st.markdown(f"""
        <div style="{eq_bg}">
            <div style="{mono}{eq_color}font-size:0.85rem;line-height:2;">
                S\u2084 = max(0, K\u2084 \u2212 D\u2084), &nbsp;
                S\u2085 = max(0, K\u2085 \u2212 D\u2085)<br>
                K\u2081<sup>eff</sup> = K\u2081 + S\u2084 + S\u2085<br>
                S\u2081 = max(0, K\u2081<sup>eff</sup> \u2212 D\u2081)<br>
                K\u2082<sup>eff</sup> = K\u2082 + S\u2081, &nbsp;
                S\u2082 = max(0, K\u2082<sup>eff</sup> \u2212 D\u2082)<br>
                K\u2083<sup>eff</sup> = K\u2083 + S\u2082
            </div>
        </div>""", unsafe_allow_html=True)

    # ── How does the algorithm work (3 layers) ──
    st.markdown('<div class="section-title">¿Cómo Encuentra las Soluciones? (Codificacion en 3 Capas)</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box" style="font-size:0.85rem;line-height:1.7;">
        El algoritmo HHO está inspirado en <strong>halcones de Harris</strong> que cazan en grupo.
        Cada halcón representa una posible forma de repartir visas. Pero los halcones
        "piensan" en números decimales, y nosotros necesitamos decisiones enteras.
        Aquí está el truco en 3 pasos:
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card" style="text-align:center;">
            <div style="margin-bottom:6px;display:flex;justify-content:center;"><svg viewBox="0 0 24 24" width="32" height="32" fill="none" stroke="{t['accent1']}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10 10-4.5 10-10S17.5 2 12 2z"/><path d="M8 12l3 3 5-5"/></svg></div>
            <div style="font-size:0.7rem;color:{t['text_muted']};text-transform:uppercase;font-weight:600;">
                Paso 1: El Halcon Vuela</div>
            <div style="{mono}font-size:0.85rem;color:{t['accent1']} !important;margin:8px 0;">
                H = (0.72, 0.15, 0.91, ...)</div>
            <div style="font-size:0.78rem;color:{t['tab_text']} !important;line-height:1.5;">
                El halcón tiene {G} números entre 0 y 1.
                Los mueve usando 6 estrategias de caza
                (exploracion, asedio, picada, vuelo de Levy...).</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card" style="text-align:center;">
            <div style="margin-bottom:6px;display:flex;justify-content:center;"><svg viewBox="0 0 24 24" width="32" height="32" fill="none" stroke="{t['accent2']}" stroke-width="1.8" stroke-linecap="round"><path d="M3 6h7M3 12h5M3 18h3M16 6v12M13 15l3 3 3-3"/></svg></div>
            <div style="font-size:0.7rem;color:{t['text_muted']};text-transform:uppercase;font-weight:600;">
                Paso 2: Convertir a Orden (SPV)</div>
            <div style="{mono}font-size:0.85rem;color:{t['accent2']} !important;margin:8px 0;">
                \u03c0 = argsort(H) \u2192 (2, 1, 3, ...)</div>
            <div style="font-size:0.78rem;color:{t['tab_text']} !important;line-height:1.5;">
                Ordenamos los grupos de menor a mayor valor.
                El grupo con el número más chico va primero
                en la fila de las visas.</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card" style="text-align:center;">
            <div style="margin-bottom:6px;display:flex;justify-content:center;"><svg viewBox="0 0 24 24" width="32" height="32" fill="none" stroke="{t['accent3']}" stroke-width="1.8"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 01-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg></div>
            <div style="font-size:0.7rem;color:{t['text_muted']};text-transform:uppercase;font-weight:600;">
                Paso 3: Repartir (Decoder Greedy)</div>
            <div style="{mono}font-size:0.85rem;color:{t['accent3']} !important;margin:8px 0;">
                x<sub>g</sub> = min(demanda, disponible)</div>
            <div style="font-size:0.78rem;color:{t['tab_text']} !important;line-height:1.5;">
                Recorremos la fila y damos visas hasta agotar.
                Cada grupo recibe lo que puede
                sin violar ninguna regla.</div>
        </div>""", unsafe_allow_html=True)

    # ── HHO ↔ Math model mapping ──
    st.markdown('<div class="section-title">Conexi\u00f3n: Modelo Matem\u00e1tico \u2194 Metaheur\u00edstica HHO</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
<div class="info-box" style="font-size:0.85rem;line-height:1.7;">
En la naturaleza, los halcones de Harris cazan conejos en equipo usando distintas estrategias.
El algoritmo traduce cada elemento de la caza a un componente del modelo matem\u00e1tico:
</div>""", unsafe_allow_html=True)

    hho_mapping = [
        ("\U0001f985 Halc\u00f3n (posici\u00f3n H)", "Vector continuo",
         f"Cada halc\u00f3n es un vector H \u2208 \u211d\u00b9\u2070\u2075 (un n\u00famero real por cada uno de los {G} grupos pa\u00eds-categor\u00eda). No es una soluci\u00f3n directamente \u2014 es un c\u00f3digo que el decodificador SPV+Greedy traduce a una asignaci\u00f3n factible."),
        ("\U0001f407 Presa (l\u00edder)", "Soluci\u00f3n no dominada",
         "En HHO mono-objetivo, la presa es el mejor individuo. En MOHHO, la presa se elige <strong>aleatoriamente del archivo de Pareto</strong>, con preferencia por soluciones en zonas poco densas (crowding distance). As\u00ed los halcones exploran todo el frente, no solo un extremo."),
        ("\u26a1 Energ\u00eda de escape (E)", "Balance exploraci\u00f3n/explotaci\u00f3n",
         "E = 2\u00b7E\u2080\u00b7(1 \u2212 t/T). Al inicio |E| es grande: los halcones <strong>exploran</strong> posiciones lejanas (b\u00fasqueda global). Conforme avanzan las iteraciones, |E| decrece: los halcones <strong>explotan</strong> la vecindad de la presa (refinamiento local). Esto es lo que hace que HHO converja."),
        ("\U0001f4a8 Vuelo de L\u00e9vy", "Saltos largos ocasionales",
         f"Cuando |E| < 0.5, el halc\u00f3n puede hacer una \u201cpicada r\u00e1pida\u201d con salto de L\u00e9vy: pasos cortos frecuentes + saltos largos raros. Esto evita que el algoritmo se quede atrapado en \u00f3ptimos locales, crucial para encontrar las {summary['combined_pareto_size']} soluciones diversas del frente."),
        ("\U0001f4c2 Archivo externo", "Frente de Pareto",
         "Las mejores soluciones no dominadas se guardan en un archivo de tama\u00f1o 100. Cuando se llena, se eliminan las soluciones en zonas densas (menor crowding distance). Esto garantiza diversidad en el frente final."),
        ("\u2699\ufe0f Decodificador SPV+Greedy", "Factibilidad garantizada",
         "El halc\u00f3n vuela libremente en \u211d\u00b9\u2070\u2075 sin preocuparse por restricciones. La regla SPV (Smallest Position Value) convierte esos n\u00fameros en un orden de prioridad, y el decoder greedy reparte visas respetando R1\u2013R6 por construcci\u00f3n. <strong>No hay soluciones inv\u00e1lidas.</strong>"),
    ]

    for icon_title, math_concept, explanation in hho_mapping:
        st.markdown(f"""
<div style="background:{t['card_bg']};border:1px solid {t['card_border']};border-radius:10px;
    padding:14px 16px;margin:8px 0;display:flex;gap:14px;align-items:flex-start;">
<div style="min-width:140px;">
    <div style="font-size:0.82rem;font-weight:700;color:{t['accent1']} !important;">{icon_title}</div>
    <div style="font-size:0.7rem;color:{t['accent2']} !important;font-weight:600;margin-top:2px;">
        \u2192 {math_concept}</div>
</div>
<div style="font-size:0.82rem;color:{t['tab_text']} !important;line-height:1.6;">{explanation}</div>
</div>""", unsafe_allow_html=True)

    # ── Worked example ──
    st.markdown('<div class="section-title">Ejemplo Paso a Paso</div>',
                unsafe_allow_html=True)
    st.markdown(f"""<div class="metric-card" style="padding:1.2rem 1.5rem;">
<div style="font-size:0.85rem;color:{t['tab_text']} !important;line-height:1.8;">
<strong style="color:{t['accent1']} !important;">Paso 1:</strong>
El halcón genera: H = (<span style="{mono}color:{t['accent2']} !important;">0.72</span>,
<span style="{mono}color:{t['accent2']} !important;">0.15</span>,
<span style="{mono}color:{t['accent2']} !important;">0.91</span>,
<span style="{mono}color:{t['accent2']} !important;">0.03</span>,
<span style="{mono}color:{t['accent2']} !important;">0.58</span>)<br>
<strong style="color:{t['accent2']} !important;">Paso 2 (SPV):</strong>
Ordenamos de menor a mayor \u2192 \u03c0 = (4, 2, 5, 1, 3)<br>
<span style="color:{t['text_muted']};">El grupo 4 (valor 0.03) va primero,
luego el 2 (0.15), etc.</span><br>
<strong style="color:{t['accent3']} !important;">Paso 3 (Decoder):</strong>
Recorremos \u03c0 y repartimos:<br>
<span style="color:{t['text_muted']};">
&nbsp;&nbsp;\u2022 Grupo 4: pide 8,000 \u2192 le damos min(8000, disponible, cupo_pa\u00eds, cupo_cat)<br>
&nbsp;&nbsp;\u2022 Grupo 2: pide 15,000 \u2192 le damos lo que se pueda sin violar R1-R6<br>
&nbsp;&nbsp;\u2022 ... y as\u00ed hasta agotar las {_fmt(data['total_visas'])} visas.
</span><br><br>
<strong>Propiedad clave:</strong> <em>cualquier</em> orden que genere el halc\u00f3n produce una asignaci\u00f3n
v\u00e1lida. No se necesitan penalizaciones — la factibilidad est\u00e1 garantizada por dise\u00f1o.<br><br>
<strong style="color:{t['accent1']} !important;">Paso 4 (Evaluar):</strong>
Calculamos los tres objetivos sobre la asignaci\u00f3n resultante x:<br>
<span style="color:{t['text_muted']};">
&nbsp;&nbsp;\u2022 f\u2081 = \u03a3(n<sub>g</sub> \u2212 x<sub>g</sub>) \u00b7 w<sub>g</sub> / \u03a3 n<sub>g</sub><br>
&nbsp;&nbsp;\u2022 f\u2082 = max|W\u0304<sub>c1</sub> \u2212 W\u0304<sub>c2</sub>| entre todos los pares de pa\u00edses<br>
&nbsp;&nbsp;\u2022 f\u2083 = V \u2212 \u03a3 x<sub>g</sub> (visas sin asignar)
</span>
</div>
</div>""", unsafe_allow_html=True)

    # ── Numerical worked example ──
    _bf1 = summary['best_f1'] if summary else [7.186, 10.744, 16280]
    _bf2 = summary['best_f2'] if summary else [7.514, 2.064, 17105]
    _bf3 = summary['best_f3'] if summary else [7.258, 9.390, 0]
    _bl = summary['baseline'] if summary else {'f1': 7.214, 'f2': 12.638, 'f3': 17540}
    st.markdown(f"""<div class="metric-card" style="padding:1.2rem 1.5rem;">
<div style="font-size:0.95rem;font-weight:700;color:{t['accent2']} !important;margin-bottom:10px;">
Ejemplo con N\u00fameros Reales</div>
<div style="font-size:0.85rem;color:{t['tab_text']} !important;line-height:1.9;">
El algoritmo evalu\u00f3 las tres funciones objetivo para cada soluci\u00f3n. Estos son los resultados
de cuatro soluciones reales del experimento:<br><br>
<table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
<tr style="border-bottom:2px solid {t['card_border']};">
<th style="text-align:left;padding:6px;color:{t['text']};">Soluci\u00f3n</th>
<th style="text-align:right;padding:6px;color:{t['accent1']};">f\u2081 (espera)</th>
<th style="text-align:right;padding:6px;color:{t['accent2']};">f\u2082 (disparidad)</th>
<th style="text-align:right;padding:6px;color:{t['accent3']};">f\u2083 (desperdicio)</th>
</tr>
<tr style="border-bottom:1px solid {t['card_border']};">
<td style="padding:6px;color:{t['text']};">Mejor f\u2081</td>
<td style="text-align:right;padding:6px;font-family:'JetBrains Mono';color:{t['accent1']};font-weight:700;">{_bf1[0]:.3f} a\u00f1os</td>
<td style="text-align:right;padding:6px;font-family:'JetBrains Mono';">{_bf1[1]:.3f} a\u00f1os</td>
<td style="text-align:right;padding:6px;font-family:'JetBrains Mono';">{_fmt(int(_bf1[2]))}</td>
</tr>
<tr style="border-bottom:1px solid {t['card_border']};background:{t['info_bg']};">
<td style="padding:6px;color:{t['text']};">Mejor f\u2082</td>
<td style="text-align:right;padding:6px;font-family:'JetBrains Mono';">{_bf2[0]:.3f} a\u00f1os</td>
<td style="text-align:right;padding:6px;font-family:'JetBrains Mono';color:{t['accent2']};font-weight:700;">{_bf2[1]:.3f} a\u00f1os</td>
<td style="text-align:right;padding:6px;font-family:'JetBrains Mono';">{_fmt(int(_bf2[2]))}</td>
</tr>
<tr style="border-bottom:1px solid {t['card_border']};">
<td style="padding:6px;color:{t['text']};">Mejor f\u2083</td>
<td style="text-align:right;padding:6px;font-family:'JetBrains Mono';">{_bf3[0]:.3f} a\u00f1os</td>
<td style="text-align:right;padding:6px;font-family:'JetBrains Mono';">{_bf3[1]:.3f} a\u00f1os</td>
<td style="text-align:right;padding:6px;font-family:'JetBrains Mono';color:{t['accent3']};font-weight:700;">{_fmt(int(_bf3[2]))}</td>
</tr>
<tr style="background:{t['info_bg']};">
<td style="padding:6px;color:{t['danger']};font-weight:600;">FIFO (actual)</td>
<td style="text-align:right;padding:6px;font-family:'JetBrains Mono';color:{t['danger']};">{_bl['f1']:.3f} a\u00f1os</td>
<td style="text-align:right;padding:6px;font-family:'JetBrains Mono';color:{t['danger']};">{_bl['f2']:.3f} a\u00f1os</td>
<td style="text-align:right;padding:6px;font-family:'JetBrains Mono';color:{t['danger']};">{_fmt(int(_bl['f3']))}</td>
</tr>
</table>
<br>
<strong style="color:{t['accent1']};">\u00bfPor qu\u00e9 ninguna es la \u201cmejor\u201d?</strong>
Mejor-f\u2081 tiene la menor espera (7.186) pero desperdicia 16,280 visas.
Mejor-f\u2083 no desperdicia ninguna visa pero su espera es 7.258.
Mejor-f\u2082 tiene la menor brecha (2.064) pero la peor espera (7.514).
<strong>Las tres son \u00f3ptimas</strong> \u2014 cada una gana en un objetivo y pierde en otro.
Eso es el frente de Pareto.
</div>
</div>""", unsafe_allow_html=True)

    # ── Pareto dominance ──
    st.markdown('<div class="section-title">¿Qué es el Frente de Pareto?</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box" style="font-size:0.85rem;line-height:1.7;">
        Imagina que comparas dos repartos de visas, <strong>A</strong> y <strong>B</strong>:<br><br>
        \u2022 Si A tiene <strong>menor espera, menor brecha Y menos desperdicio</strong> que B \u2192 A es claramente mejor.
        Decimos que A <em>domina</em> a B.<br>
        \u2022 Si A tiene <strong>menor espera PERO mayor brecha o m\u00e1s desperdicio</strong> \u2192 ninguno domina al otro.
        Ambos son v\u00e1lidos, solo son compromisos diferentes.<br><br>
        El <strong>frente de Pareto</strong> es el grupo de soluciones que nadie puede superar en los tres
        objetivos a la vez. Son las "mejores opciones disponibles" \u2014 {summary['combined_pareto_size'] if summary else 371}
        en nuestro caso.
    </div>""", unsafe_allow_html=True)

    with st.expander("Ver definici\u00f3n formal de dominancia"):
        st.markdown(f"""
        <div style="{eq_bg}">
            <div style="{mono}{eq_color}text-align:center;font-size:0.9rem;">
                a \u227b b &nbsp;\u27fa&nbsp;
                \u2200m \u2208 \u007b1,2,3\u007d: f<sub>m</sub>(a) \u2264 f<sub>m</sub>(b)
                &nbsp;\u2227&nbsp; \u2203m: f<sub>m</sub>(a) &lt; f<sub>m</sub>(b)</div>
        </div>
        <p style="font-size:0.82rem;color:{t['tab_text']} !important;">
            a domina a b si es al menos igual en los tres objetivos y estrictamente mejor en al menos uno.</p>
        <div style="{eq_bg}">
            <div style="{mono}{eq_color}text-align:center;font-size:0.9rem;">
                Frente de Pareto = \u007b x | no existe x' que domine a x \u007d</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Hypervolume ──
    st.markdown('<div class="section-title">¿Cómo Sabemos si las Soluciones son Buenas? (Hipervolumen)</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box" style="font-size:0.85rem;line-height:1.7;">
        El <strong>hipervolumen (HV)</strong> es como medir el "volumen" que cubre el frente de Pareto
        en el espacio tridimensional de f\u2081 \u00d7 f\u2082 \u00d7 f\u2083. Si el frente se expande (mejores soluciones),
        el volumen crece.<br><br>
        \u2022 <strong>Mayor HV = mejor:</strong> las soluciones cubren más espacio de opciones.<br>
        \u2022 Lo calculamos para cada una de las 30 corridas del algoritmo para verificar que es
        <strong>consistente</strong> (no fue suerte de una sola vez).<br>
        \u2022 HV medio de nuestro modelo: <strong style="color:{t['accent2']};">{_fmtd(summary['hv_stats']['mean'] if summary else 35.92)}</strong>
        con desviación estándar de solo <strong>{_fmtd(summary['hv_stats']['std'] if summary else 0.98)}</strong> \u2014 muy estable.
    </div>""", unsafe_allow_html=True)


def _tab_about(summary: dict, data: dict) -> None:
    t = _get_theme()
    mono = "font-family:'JetBrains Mono',monospace;"

    # ── What is this project ──
    st.markdown('<div class="section-title">¿Qué es Visa Predict AI?</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box" style="font-size:0.9rem;line-height:1.8;">
        <strong>Visa Predict AI</strong> es un proyecto de investigación que aplica inteligencia artificial
        para repensar cómo se reparten las <strong>{_fmt(data['total_visas'])}</strong> Green Cards de empleo
        que Estados Unidos otorga cada año.
        <br><br>
        <strong style="color:{t['danger']};">El problema:</strong> El sistema actual (FIFO) atiende por orden
        de llegada, pero no considera la demanda real. Un ingeniero de India puede esperar
        <strong style="color:{t['danger']};">más de una década</strong>, mientras países con poca demanda
        reciben visas en meses. Además, el FIFO desperdicia
        <strong style="color:{t['danger']};">{_fmt(int(summary['baseline']['f3']))}</strong> visas que quedan
        sin asignar porque ningún país elegible tiene solicitantes suficientes en esa categoría.
        <br><br>
        <strong style="color:{t['accent1']};">Nuestra propuesta:</strong> Usamos un algoritmo bio-inspirado
        llamado <strong>MOHHO</strong> (Multi-Objective Harris Hawks Optimization) que busca optimizar
        <strong>tres objetivos simultáneamente</strong>:
        <br>
        <span style="color:{t['accent1']};">▸ f₁</span> Reducir la espera promedio ponderada
        &nbsp;&nbsp;
        <span style="color:{t['accent2']};">▸ f₂</span> Tratar a todos los países con equidad
        &nbsp;&nbsp;
        <span style="color:{t['accent3']};">▸ f₃</span> Maximizar la utilización (cero visas desperdiciadas)
        <br><br>
        El resultado: <strong>{summary['combined_pareto_size']} escenarios alternativos</strong> de reparto,
        cada uno con un balance diferente entre estos tres objetivos. Ninguno es "el mejor" en todo
        — el tomador de decisiones elige según sus prioridades.
    </div>""", unsafe_allow_html=True)

    # ── How does MOHHO work ──
    st.markdown('<div class="section-title">¿Cómo Funciona MOHHO?</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box" style="font-size:0.85rem;line-height:1.7;border-left:4px solid {t['accent1']};">
        Los <strong>halcones de Harris</strong> son las únicas aves rapaces que cazan en grupo de forma
        cooperativa. Rodean a su presa, la acorralan y atacan desde múltiples ángulos.
        <br><br>
        El algoritmo HHO imita esta estrategia: un grupo de "halcones virtuales" explora
        el espacio de posibles repartos de visas. Cada halcón representa una forma diferente
        de distribuir las {_fmt(data['total_visas'])} visas. Comparten información y convergen
        hacia las mejores soluciones.
    </div>""", unsafe_allow_html=True)

    # ── La Caza Multiobjetivo (visual) ──
    st.markdown(f'<div class="section-title">La Caza Multiobjetivo</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.82rem;color:{t['text_muted']};margin-bottom:12px;">
        En la naturaleza, los halcones de Harris tienen un solo objetivo: atrapar a la presa.
        En nuestro problema, la "presa" tiene <strong style="color:{t['accent1']};">tres dimensiones</strong>
        — hay que perseguirla en tres direcciones a la vez.
    </div>""", unsafe_allow_html=True)

    hunt_cols = st.columns(3)
    hunt_items = [
        (t['accent1'], "f\u2081 Reducir espera",
         "svg_wait",
         "Un grupo de halcones persigue soluciones que "
         "minimicen la espera promedio. Priorizan a India y "
         "China, que llevan más años en la fila."),
        (t['accent2'], "f\u2082 Igualar pa\u00edses",
         "svg_equity",
         "Otro flanco ataca la disparidad: buscan repartos "
         "donde ningún pa\u00eds espere mucho más que otro. "
         "Sacrifican velocidad por justicia."),
        (t['accent3'], "f\u2083 Cero desperdicio",
         "svg_util",
         "El tercer ángulo busca usar todas las visas. "
         "Reasignan visas que el FIFO dejar\u00eda vacantes por "
         "topes de pa\u00eds o categor\u00eda."),
    ]
    for col, (color, title, _, desc) in zip(hunt_cols, hunt_items):
        with col:
            st.markdown(f"""<div class="metric-card" style="text-align:center;
                border-top:3px solid {color};padding:16px 12px;">
                <div style="font-size:2rem;margin-bottom:6px;">
                    <svg viewBox="0 0 24 24" width="32" height="32" fill="none"
                         stroke="{color}" stroke-width="1.5" stroke-linecap="round">
                        <circle cx="12" cy="12" r="10"/>
                        <path d="M12 8v4l3 3"/>
                    </svg>
                </div>
                <div style="font-size:0.85rem;font-weight:700;color:{color};
                    margin-bottom:6px;">{title}</div>
                <div style="font-size:0.78rem;color:{t['tab_text']} !important;
                    line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box" style="font-size:0.82rem;margin-top:4px;">
        <strong>El dilema:</strong> Ningún halcón puede "atrapar" los tres objetivos a la vez.
        Reducir la espera (f\u2081) puede aumentar la disparidad (f\u2082) o desperdiciar visas (f\u2083).
        Por eso el algoritmo no busca <em>una</em> solución perfecta, sino un
        <strong>frente de Pareto</strong> — {summary['combined_pareto_size']} compromisos diferentes
        donde mejorar un objetivo inevitablemente empeora otro.
    </div>""", unsafe_allow_html=True)

    # ── Pipeline flow (using columns to avoid HTML rendering issues) ──
    G = len(data['countries']) * len(data['categories'])
    # SVG icons for pipeline steps (cleaner than emoji)
    _svg_hawk = f'<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="{t["accent1"]}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10 10-4.5 10-10S17.5 2 12 2z"/><path d="M8 12l3 3 5-5"/></svg>'
    _svg_sort = f'<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="{t["accent2"]}" stroke-width="1.8" stroke-linecap="round"><path d="M3 6h7M3 12h5M3 18h3M16 6v12M13 15l3 3 3-3"/></svg>'
    _svg_gear = f'<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="{t["accent3"]}" stroke-width="1.8"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 01-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>'
    _svg_chart = f'<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="{t["accent1"]}" stroke-width="1.8" stroke-linecap="round"><path d="M18 20V10M12 20V4M6 20v-6"/></svg>'
    _svg_star = f'<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="{t["accent3"]}" stroke-width="1.8" stroke-linejoin="round"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>'

    steps = [
        (_svg_hawk, "Halc\u00f3n", f"50 halcones con vectores de {G} n\u00fameros entre 0 y 1",
         t['accent1'], "Cada halc\u00f3n es una propuesta de reparto"),
        (_svg_sort, "SPV", "Convertir n\u00fameros decimales a un orden de prioridad",
         t['accent2'], "Qui\u00e9n va primero en la fila de visas"),
        (_svg_gear, "Decoder", "Repartir visas respetando las 6 reglas legales",
         t['accent3'], "Asignaci\u00f3n factible garantizada"),
        (_svg_chart, "Evaluar", "Calcular f\u2081 (espera), f\u2082 (brecha) y f\u2083 (desperdicio)",
         t['accent1'], "Qu\u00e9 tan buena es esta soluci\u00f3n"),
        (_svg_star, "Pareto", "Guardar las soluciones que nadie puede superar",
         t['accent3'], f"{summary['combined_pareto_size']} compromisos \u00f3ptimos"),
    ]

    cols = st.columns(len(steps))
    for col, (icon, name, desc, color, hint) in zip(cols, steps):
        with col:
            st.markdown(f"""<div class="metric-card" style="text-align:center;padding:14px 10px;
                border-color:{_rgba(color, 0.2)};">
                <div style="display:flex;justify-content:center;margin-bottom:4px;">{icon}</div>
                <div style="font-size:0.8rem;color:{color};font-weight:700;margin:4px 0;">{name}</div>
                <div style="font-size:0.68rem;color:{t['tab_text']} !important;line-height:1.4;">{desc}</div>
                <div style="font-size:0.62rem;color:{t['text_muted']};margin-top:4px;
                    font-style:italic;">{hint}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;font-size:0.8rem;color:{t['text_muted']};margin-bottom:1rem;">
        Este ciclo se repite <strong>{summary['max_iter']} veces</strong> por corrida,
        y se ejecuta <strong>{summary['num_runs']} corridas</strong> con semillas diferentes
        para verificar consistencia.
    </div>""", unsafe_allow_html=True)

    # ── The 6 hawk strategies ──
    st.markdown('<div class="section-title">Las 6 Estrategias de los Halcones</div>',
                unsafe_allow_html=True)

    strategies = [
        ("Exploración", "E \u2265 1",
         "Buscar en zonas nuevas del espacio. Los halcones vuelan lejos del grupo, "
         "como scouts que exploran territorio desconocido.",
         t['accent1']),
        ("Asedio suave", "0.5 \u2264 E < 1, r \u2265 0.5",
         "Rodear la presa desde lejos. Los halcones ajustan su posición gradualmente "
         "hacia la mejor solución conocida.",
         t['accent2']),
        ("Asedio duro", "0.5 \u2264 E < 1, r < 0.5",
         "Acercarse rápidamente a la presa. Movimientos más agresivos hacia la solución líder.",
         t['accent3']),
        ("Picada suave", "E < 0.5, r \u2265 0.5",
         "Simular que la presa escapa y recalcular. Incluye vuelos de Lévy "
         "para dar saltos largos y evitar quedarse atrapado.",
         t['danger']),
        ("Picada dura", "E < 0.5, r < 0.5",
         "Ataque final: movimiento preciso con vuelo de Lévy hacia la posición óptima.",
         t['mid_blue']),
        ("Vuelo de Lévy", "Componente especial",
         "Patrón de movimiento con pasos largos ocasionales — como un halcón que de repente "
         "vuela lejos del grupo para cubrir más terreno.",
         t['accent2']),
    ]

    cols = st.columns(3)
    for i, (name, trigger, desc, color) in enumerate(strategies):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="metric-card" style="border-left:3px solid {color};margin-bottom:10px;">
                <div style="font-size:0.75rem;color:{color};font-weight:700;text-transform:uppercase;">
                    {name}</div>
                <div style="{mono}font-size:0.65rem;color:{t['text_muted']};margin:2px 0 6px;">
                    {trigger}</div>
                <div style="font-size:0.78rem;color:{t['tab_text']} !important;line-height:1.5;">
                    {desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box" style="font-size:0.82rem;">
        La <strong>Energía de Escape (E)</strong> controla la transición: empieza en 2 (exploración pura)
        y decrece a 0 (explotación pura) a lo largo de las iteraciones. Esto permite al algoritmo
        primero explorar ampliamente y luego refinar las mejores soluciones.
    </div>""", unsafe_allow_html=True)

    # ── Parameters table ──
    st.markdown('<div class="section-title">Parámetros del Experimento</div>',
                unsafe_allow_html=True)

    params = [
        ("Población", str(summary["pop_size"]),
         "Cuántos halcones buscan simultáneamente.",
         "Heidari et al. (2019) recomiendan N=30-50 para problemas de dimensión media. "
         "Con dim=105 (nuestros 105 grupos), 50 halcones dan suficiente diversidad sin "
         "desperdicio computacional. Regla empírica: N \u2248 0.5\u00d7dim."),
        ("Iteraciones", str(summary["max_iter"]),
         "Rondas de caza que completa cada corrida.",
         "500 iteraciones garantizan convergencia: nuestras curvas de hipervolumen se estabilizan "
         "al 95% del valor final antes de la iteración 100. Las 400 restantes permiten "
         "explotación fina (Ops 3-6 de HHO). En metaheurísticas, T=500 es estándar "
         "para asegurar que el resultado no dependa de parar demasiado pronto."),
        ("Archivo Pareto", str(summary["archive_size"]),
         "Capacidad máxima de soluciones no dominadas.",
         "100 es el valor canónico en MOEA/D y NSGA-II (Deb et al., 2002). "
         "Suficiente para representar un frente triobjetivo con buena diversidad. "
         "Si se llena, se poda por crowding distance — se eliminan las soluciones "
         "más apretadas, conservando los extremos y el knee."),
        ("Corridas independientes", str(summary["num_runs"]),
         "Repeticiones con diferentes semillas aleatorias.",
         "30 corridas es el mínimo recomendado por Derrac et al. (2011) para aplicar "
         "pruebas estadísticas no paramétricas (Wilcoxon, Mann-Whitney). "
         "Con 30 muestras el Teorema Central del Límite justifica intervalos de confianza "
         "sobre la media del hipervolumen. Semillas consecutivas: 42, 43, ..., 71."),
        ("Visas disponibles (V)", _fmt(data["total_visas"]),
         "Total de visas EB por año fiscal.",
         "Establecido por la Ley de Inmigración y Nacionalidad (INA) §201(d). "
         "No es un parámetro libre — es un dato fijo del gobierno de EE.UU. "
         "Se mantiene en 140,000 desde 1990. Es la restricción R1 del modelo."),
        ("Grupos (G)", str(len(data["countries"]) * len(data["categories"])),
         f"{len(data['countries'])} países \u00d7 {len(data['categories'])} categorías EB.",
         "Cada combinación país-categoría es una variable de decisión (x\u2081 a x\u2081\u2080\u2085). "
         "El halcón genera un vector de 105 valores continuos en [0,1], que el SPV "
         "convierte en permutación y el decoder traduce a asignación factible. "
         "dim=105 define el tamaño del espacio de búsqueda."),
        ("Países analizados", str(len(data["countries"])),
         "20 países con mayor demanda + 'Resto del Mundo'.",
         "Se seleccionaron los 20 países con mayor backlog documentado por USCIS (Visa Bulletin). "
         "El bloque 'Resto del Mundo' agrupa ~170 países con demanda individual pequeña. "
         "Estos 21 bloques representan >99% de la demanda total de visas EB."),
        ("Límite por país", _fmt(25620),
         "7% del total de visas de inmigración (366,000).",
         "INA §202(a)(2): ningún país puede recibir más del 7% del total de visas de inmigración "
         "(familiares + empleo = 366,000). 366,000 \u00d7 0.07 = 25,620. "
         "Es la causa estructural del problema: India genera ~70% de la demanda EB "
         "pero recibe máximo 7%. Esto crea colas de 10-134 años."),
    ]

    for name, value, short_desc, justification in params:
        st.markdown(f"""<div style="margin-bottom:8px;padding:12px 16px;background:{t['card_bg']};
border-radius:12px;border:1px solid {t['card_border']};border-left:4px solid {t['accent1']};">
<div style="display:flex;gap:12px;align-items:center;">
<div style="min-width:160px;font-weight:600;font-size:0.85rem;color:{t['text']} !important;">
{name}</div>
<div style="{mono}min-width:70px;color:{t['accent2']} !important;font-weight:700;
font-size:1rem;">{value}</div>
<div style="font-size:0.82rem;color:{t['tab_text']} !important;flex:1;">{short_desc}</div>
</div>
<div style="font-size:0.78rem;color:{t['text_muted']} !important;margin-top:6px;
line-height:1.55;padding-left:4px;border-top:1px solid {t['card_border']};padding-top:6px;">
<strong style="color:{t['accent3']} !important;">¿Por qué este valor?</strong> {justification}</div>
</div>""", unsafe_allow_html=True)

    # ── Glossary ──
    st.markdown('<div class="section-title">Glosario</div>',
                unsafe_allow_html=True)

    search = st.text_input("Buscar término...", key="glossary_search")

    glossary = [
        ("Green Card (Tarjeta de Residencia)", "Inmigracion",
         "Permiso de residencia permanente en EE.UU. Se emiten {V} visas de empleo al año "
         "en 5 categorías (EB-1 a EB-5). Es lo que reparte nuestro modelo.".format(V=_fmt(data['total_visas']))),
        ("FIFO (First In, First Out)", "Sistema actual",
         "El sistema vigente: quien llegó primero se atiende primero. Simple pero injusto — "
         "ignora cuánto llevas esperando y genera colas de décadas para India y China."),
        ("MOHHO", "Nuestro algoritmo",
         "Multi-Objective Harris Hawks Optimization. Algoritmo de IA inspirado en la caza cooperativa "
         f"de halcones de Harris. Encuentra {summary['combined_pareto_size']} formas alternativas de repartir visas, cada una "
         "con un balance diferente entre espera (f\u2081), equidad (f\u2082) y utilizaci\u00f3n (f\u2083)."),
        ("f\u2081 \u2014 Carga de espera", "Objetivo 1",
         "Mide cuántos años de espera quedan sin resolver. Minimizarlo prioriza dar visas "
         "a la gente que lleva más tiempo en la fila (India, China)."),
        ("f\u2082 \u2014 Brecha entre pa\u00edses", "Objetivo 2",
         "Mide la mayor diferencia de espera entre cualquier par de pa\u00edses. "
         "Minimizarlo busca que todos los pa\u00edses esperen tiempos similares \u2014 equidad pura."),
        ("f\u2083 \u2014 Desperdicio de visas", "Objetivo 3",
         "Mide cu\u00e1ntas de las {V} visas quedan sin asignar (f\u2083 = V \u2212 \u03a3x). "
         "Minimizarlo maximiza la utilizaci\u00f3n del presupuesto anual. FIFO desperdicia "
         "{w} visas; MOHHO puede llegar hasta f\u2083 = 0.".format(
             V=_fmt(data['total_visas']),
             w=_fmt(data['total_visas'] - data['fifo_used']))),
        ("Frente de Pareto", "Concepto clave",
         "El grupo de soluciones que nadie puede superar en los tres objetivos a la vez. "
         "Si una soluci\u00f3n tiene menos espera, otra tendr\u00e1 menos brecha o menos desperdicio \u2014 "
         f"no se puede ganar en todo. Nuestro frente tiene {summary['combined_pareto_size']} puntos."),
        ("Knee Point (Equilibrio)", "Punto especial",
         "El punto del frente de Pareto donde un poquito m\u00e1s de espera produce mucha m\u00e1s equidad "
         "(o viceversa). Es el compromiso m\u00e1s eficiente entre los tres objetivos."),
        ("Categorías EB", "Inmigración",
         "EB-1: personas extraordinarias (premios Nobel, etc). "
         "EB-2: profesionales con maestría. EB-3: trabajadores calificados. "
         "EB-4: inmigrantes especiales (religiosos, traductores militares). "
         "EB-5: inversores ($800k+)."),
        ("Spillover", "Ley migratoria",
         "Si una categoría no usa todas sus visas, el sobrante pasa a otra. "
         "EB-4/5 tienen poca demanda, así que sus visas extras suben a EB-1, "
         "luego bajan a EB-2 y EB-3."),
        ("Límite del 7%", "Ley migratoria",
         "Ningún país puede recibir más del 7% del total ({c} visas/ano). "
         "India tiene ~70% de la demanda pero recibe máximo 7%. "
         "Esta es la causa principal del problema.".format(c=_fmt(25620))),
        ("Hipervolumen (HV)", "M\u00e9trica",
         "Mide qu\u00e9 tan buenas son las soluciones: el 'volumen' del espacio triobjetivo cubierto por el frente de Pareto. "
         "Mayor volumen = mejores soluciones. Lo usamos para comparar las 30 corridas del algoritmo."),
        ("SPV", "Técnica",
         "Smallest Position Value (Bean, 1994). Truco matemático para convertir números "
         "decimales (que maneja el halcon) en un orden de prioridad (que necesita el decoder)."),
        ("Decodificador Greedy", "Componente",
         "Recibe un orden de prioridad y reparte visas uno por uno, respetando todas las reglas legales. "
         "Cualquier orden produce un reparto válido — no puede equivocarse."),
        ("Vuelo de Lévy", "Técnica HHO",
         "Patrón de movimiento con pasos cortos frecuentes y saltos largos ocasionales. "
         "Evita que los halcones se queden atrapados en soluciones mediocres."),
        ("Convergencia", "Experimento",
         "Cómo mejora el algoritmo con el tiempo. Al inicio explora mucho (HV sube rápido), "
         "luego se estabiliza (encontró las mejores soluciones). "
         "Nuestro MOHHO converge al 95% en las primeras ~100 iteraciones."),
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

    # ── Team & Credits ──
    st.markdown('<div class="section-title">Equipo y Créditos</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:1rem;">
        <div class="metric-card" style="text-align:center;">
            <div style="font-size:0.7rem;color:{t['text_muted']};text-transform:uppercase;font-weight:600;">
                Investigadores</div>
            <div style="font-size:1rem;color:{t['text']} !important;font-weight:700;margin:8px 0;">
                Yazmín Flores</div>
            <div style="font-size:0.85rem;color:{t['text']} !important;">
                Javier Rebull</div>
            <div style="font-size:0.75rem;color:{t['text_muted']};margin-top:8px;">
                Maestría en Inteligencia Artificial<br>Aplicada a Datos (MIAAD)</div>
        </div>
        <div class="metric-card" style="text-align:center;">
            <div style="font-size:0.7rem;color:{t['text_muted']};text-transform:uppercase;font-weight:600;">
                Dirección académica</div>
            <div style="font-size:1rem;color:{t['text']} !important;font-weight:700;margin:8px 0;">
                Mtro. Raúl Gibrán Porras Alaniz</div>
            <div style="font-size:0.85rem;color:{t['text']} !important;">
                Optimización Inteligente</div>
            <div style="font-size:0.75rem;color:{t['text_muted']};margin-top:8px;">
                Universidad Autónoma de<br>Ciudad Juarez (UACJ)</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card" style="text-align:center;padding:1rem;">
        <div style="font-size:0.7rem;color:{t['text_muted']};text-transform:uppercase;font-weight:600;">
            Stack tecnológico</div>
        <div style="display:flex;justify-content:center;gap:20px;margin:10px 0;flex-wrap:wrap;">
            <div style="font-size:0.8rem;color:{t['tab_text']} !important;">
                <strong style="color:{t['accent1']};">Python</strong> + NumPy</div>
            <div style="font-size:0.8rem;color:{t['tab_text']} !important;">
                <strong style="color:{t['accent2']};">Streamlit</strong> + Plotly</div>
            <div style="font-size:0.8rem;color:{t['tab_text']} !important;">
                <strong style="color:{t['accent3']};">MOHHO</strong> (implementación propia)</div>
        </div>
        <div style="font-size:0.72rem;color:{t['text_muted']};margin-top:6px;">
            Código fuente: Python 3.12 | Visualización 100% Plotly (zero matplotlib) | Dark-first UI</div>
    </div>""", unsafe_allow_html=True)


# ── Tab 9: Simulación en Vivo ──────────────────────────────
def _tab_simulation(data: dict, baseline: tuple) -> None:
    """Ejecuta MOHHO en tiempo real con visualización de la bandada."""
    from src.problem import VisaProblem
    from src.mohho import (
        evaluate_hawk, update_archive, _step_hawk,
        compute_hypervolume,
    )
    from src.config import LB, UB, NUM_GROUPS, ARCHIVE_SIZE

    t = _get_theme()
    st.markdown(
        f'<div class="section-title">Simulación en Vivo: La Caza del Halcón</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"""<div style="color:{t['text']};line-height:1.7;margin-bottom:1rem;font-size:0.92rem;">
        Ejecuta el algoritmo MOHHO en tiempo real. Cada <strong style="color:{t['accent2']};">halcón</strong>
        representa una asignación candidata de visas; el <strong style="color:{t['danger']};">conejo</strong>
        (líder Pareto) es la mejor solución conocida. Observa cómo la bandada converge hacia el
        frente de Pareto óptimo.
    </div>""", unsafe_allow_html=True)

    # ── Controles ──
    c1, c2, c3 = st.columns(3)
    with c1:
        sim_pop = st.slider("Halcones (N)", 10, 80, 30, 5, key="sim_pop")
    with c2:
        sim_iter = st.slider("Iteraciones (T)", 20, 300, 100, 10, key="sim_iter")
    with c3:
        sim_seed = st.number_input("Semilla", 1, 9999, 42, key="sim_seed")

    run_btn = st.button(
        "\U0001f985 Iniciar Simulación", type="primary", key="run_sim_btn",
    )

    if run_btn:
        problem = VisaProblem()
        rng = np.random.default_rng(sim_seed)

        # Inicializar población
        population = rng.uniform(LB, UB, size=(sim_pop, NUM_GROUPS))
        fitnesses: list[tuple[float, float, float]] = []
        for i in range(sim_pop):
            _, fit = evaluate_hawk(population[i], problem)
            fitnesses.append(fit)

        archive_pos: list = []
        archive_fit: list[tuple[float, float, float]] = []
        for i in range(sim_pop):
            update_archive(archive_pos, archive_fit,
                           population[i], fitnesses[i], ARCHIVE_SIZE)

        # Snapshots para animación (~40 cuadros)
        snap_interval = max(1, sim_iter // 40)
        snapshots: list[dict] = []
        hv_curve: list[float] = []

        def _take_snapshot(it: int) -> None:
            snapshots.append({
                "iter": it,
                "pop": [tuple(f) for f in fitnesses],
                "archive": [tuple(f) for f in archive_fit],
                "hv": compute_hypervolume(archive_fit),
                "size": len(archive_fit),
            })

        _take_snapshot(-1)  # estado inicial

        progress = st.progress(0, text="Inicializando bandada de halcones...")

        for t_iter in range(sim_iter):
            x_mean = np.mean(population, axis=0)
            for i in range(sim_pop):
                _step_hawk(
                    i, population, fitnesses,
                    archive_pos, archive_fit,
                    x_mean, t_iter, sim_iter, sim_pop, ARCHIVE_SIZE,
                    problem, rng,
                )

            hv = compute_hypervolume(archive_fit)
            hv_curve.append(hv)

            if t_iter % snap_interval == 0 or t_iter == sim_iter - 1:
                _take_snapshot(t_iter)

            progress.progress(
                (t_iter + 1) / sim_iter,
                text=(f"Iteración {t_iter + 1}/{sim_iter} — "
                      f"HV: {hv:.2f} — Archivo: {len(archive_fit)} soluciones"),
            )

        progress.progress(
            1.0,
            text=(f"\u2705 Completado — HV final: {hv_curve[-1]:.2f} — "
                  f"{len(archive_fit)} soluciones Pareto"),
        )

        st.session_state["sim_snapshots"] = snapshots
        st.session_state["sim_hv_curve"] = hv_curve
        st.session_state["sim_params"] = {
            "pop": sim_pop, "iter": sim_iter, "seed": sim_seed,
        }

    # ── Mostrar resultados (si existen) ──
    if "sim_snapshots" not in st.session_state:
        st.markdown(f"""<div class="metric-card" style="text-align:center;padding:3rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">\U0001f985</div>
            <div style="font-size:1.1rem;font-weight:600;color:{t['text']};">
                Configura los parámetros y presiona Iniciar</div>
            <div style="font-size:0.85rem;color:{t['text_muted']};margin-top:8px;">
                N=30, T=100 toma aproximadamente 10 segundos</div>
        </div>""", unsafe_allow_html=True)
        return

    snapshots = st.session_state["sim_snapshots"]
    hv_curve = st.session_state["sim_hv_curve"]
    params = st.session_state["sim_params"]

    # ── Tarjetas resumen ──
    final_snap = snapshots[-1]
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Halcones", str(params["pop"]), t["accent1"]),
        ("Iteraciones", str(params["iter"]), t["accent2"]),
        ("HV Final", f"{final_snap['hv']:.2f}", t["accent3"]),
        ("Soluciones Pareto", str(final_snap["size"]), t["danger"]),
    ]
    for col, (label, val, color) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""<div class="metric-card" style="text-align:center;">
                <div style="font-size:0.7rem;color:{t['text_muted']};text-transform:uppercase;
                    font-weight:600;">{label}</div>
                <div style="font-family:'JetBrains Mono';font-size:1.8rem;font-weight:700;
                    color:{color} !important;margin:6px 0;">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Animación del frente de Pareto ──
    st.markdown(
        f'<div class="section-title">Evolución del Frente de Pareto</div>',
        unsafe_allow_html=True,
    )

    proj_options = {
        "f₁ vs f₂ (espera vs disparidad)": (0, 1),
        "f₁ vs f₃ (espera vs desperdicio)": (0, 2),
        "f₂ vs f₃ (disparidad vs desperdicio)": (1, 2),
    }
    proj_label = st.radio(
        "Proyección 2D", list(proj_options.keys()),
        horizontal=True, key="sim_proj",
    )
    ix, iy = proj_options[proj_label]
    obj_names = ["f₁ — Carga de espera (años)", "f₂ — Disparidad máxima (años)",
                 "f₃ — Desperdicio de visas"]

    frames = []
    for snap in snapshots:
        pop_x = [f[ix] for f in snap["pop"]]
        pop_y = [f[iy] for f in snap["pop"]]
        arch = sorted(snap["archive"], key=lambda p: p[ix])
        arch_x = [f[ix] for f in arch]
        arch_y = [f[iy] for f in arch]
        label = f"Iter {snap['iter']}" if snap["iter"] >= 0 else "Inicio"
        frames.append(go.Frame(
            data=[
                go.Scatter(
                    x=pop_x, y=pop_y, mode="markers",
                    marker=dict(size=5, color=_rgba(t["accent3"], 0.35),
                                line=dict(width=0)),
                    name="Halcones", showlegend=True,
                ),
                go.Scatter(
                    x=arch_x, y=arch_y, mode="markers+lines",
                    marker=dict(size=7, color=t["accent1"],
                                line=dict(width=1, color="white")),
                    line=dict(width=1.5, color=_rgba(t["accent1"], 0.5)),
                    name="Frente Pareto", showlegend=True,
                ),
                go.Scatter(
                    x=[baseline[ix]], y=[baseline[iy]], mode="markers",
                    marker=dict(size=14, symbol="star", color=t["danger"],
                                line=dict(width=1.5, color="white")),
                    name="FIFO actual", showlegend=True,
                ),
            ],
            name=label,
        ))

    slider_steps = [
        dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True),
                                  mode="immediate",
                                  transition=dict(duration=0))],
             label=f.name, method="animate")
        for f in frames
    ]

    fig_anim = go.Figure(
        data=frames[0].data,
        frames=frames,
    )
    fig_anim.update_layout(
        **_plotly_layout(height=520),
        xaxis=dict(title=obj_names[ix], **_grid()),
        yaxis=dict(title=obj_names[iy], **_grid()),
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=0.05, y=1.12,
            buttons=[
                dict(label="\u25b6 Reproducir", method="animate",
                     args=[None, dict(frame=dict(duration=120, redraw=True),
                                      fromcurrent=True,
                                      transition=dict(duration=40))]),
                dict(label="\u23f8 Pausar", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate",
                                        transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            active=0, steps=slider_steps,
            currentvalue=dict(prefix="", visible=True,
                              font=dict(color=t["text"], size=12)),
            pad=dict(b=10, t=30), len=0.9, x=0.05,
            bgcolor=_rgba(t["text"], 0.08),
            activebgcolor=t["accent1"],
            font=dict(color=t["text_muted"], size=9),
        )],
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11),
        ),
    )
    st.plotly_chart(fig_anim, use_container_width=True, theme=None, key="sim_pareto_anim")

    # ── Convergencia + Crecimiento del archivo ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(
            f'<div class="section-title">Convergencia (Hipervolumen)</div>',
            unsafe_allow_html=True,
        )
        fig_hv = go.Figure()
        fig_hv.add_trace(go.Scatter(
            x=list(range(1, len(hv_curve) + 1)), y=hv_curve,
            mode="lines", line=dict(width=2.5, color=t["accent1"]),
            fill="tozeroy", fillcolor=_rgba(t["accent1"], 0.1),
            name="HV",
        ))
        final_hv = hv_curve[-1]
        threshold_95 = 0.95 * final_hv
        conv_iter = next(
            (i for i, h in enumerate(hv_curve) if h >= threshold_95),
            len(hv_curve),
        )
        fig_hv.add_vline(
            x=conv_iter + 1, line_dash="dash",
            line_color=t["accent2"], line_width=1.5,
            annotation_text=f"95% en iter {conv_iter + 1}",
            annotation_font_color=t["accent2"],
            annotation_font_size=10,
        )
        fig_hv.update_layout(
            **_plotly_layout(height=350),
            xaxis=dict(title="Iteración", **_grid()),
            yaxis=dict(title="Hipervolumen", **_grid()),
            showlegend=False,
        )
        st.plotly_chart(fig_hv, use_container_width=True, theme=None, key="sim_hv_conv")

    with col_right:
        st.markdown(
            f'<div class="section-title">Crecimiento del Archivo Pareto</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "El **archivo** guarda las mejores soluciones no dominadas encontradas "
            "hasta el momento. Al inicio está vacío; conforme los halcones exploran, "
            "descubren soluciones que no son superadas por ninguna otra en los tres "
            "objetivos simultáneamente. Cuando la barra deja de crecer, el algoritmo "
            "ya encontró la mayoría de soluciones Pareto-óptimas."
        )
        sizes = [s["size"] for s in snapshots]
        iters_s = [max(0, s["iter"]) for s in snapshots]
        fig_sz = go.Figure()
        fig_sz.add_trace(go.Bar(
            x=iters_s, y=sizes,
            marker=dict(color=sizes, colorscale=[
                [0, _rgba(t["accent3"], 0.3)],
                [1, t["accent3"]],
            ]),
        ))
        fig_sz.update_layout(
            **_plotly_layout(height=350),
            xaxis=dict(title="Iteración", **_grid()),
            yaxis=dict(title="Soluciones en archivo", **_grid()),
            showlegend=False,
        )
        st.plotly_chart(fig_sz, use_container_width=True, theme=None, key="sim_arch_growth")

    # ── Evolución de la bandada (4 instantáneas) ──
    st.markdown(
        f'<div class="section-title">Evolución de la Bandada</div>',
        unsafe_allow_html=True,
    )

    n_show = min(4, len(snapshots))
    indices = [0, len(snapshots) // 3, 2 * len(snapshots) // 3, len(snapshots) - 1]
    indices = indices[:n_show]
    cols_snap = st.columns(n_show)
    for col_s, idx in zip(cols_snap, indices):
        snap = snapshots[idx]
        with col_s:
            iter_label = f"Iter {snap['iter']}" if snap["iter"] >= 0 else "Inicio"
            fig_s = go.Figure()
            pop_x = [f[ix] for f in snap["pop"]]
            pop_y = [f[iy] for f in snap["pop"]]
            arch = sorted(snap["archive"], key=lambda p: p[ix])
            fig_s.add_trace(go.Scatter(
                x=pop_x, y=pop_y, mode="markers",
                marker=dict(size=4, color=_rgba(t["accent3"], 0.4)),
                showlegend=False,
            ))
            fig_s.add_trace(go.Scatter(
                x=[f[ix] for f in arch], y=[f[iy] for f in arch],
                mode="markers+lines",
                marker=dict(size=5, color=t["accent1"]),
                line=dict(width=1, color=_rgba(t["accent1"], 0.4)),
                showlegend=False,
            ))
            fig_s.add_trace(go.Scatter(
                x=[baseline[ix]], y=[baseline[iy]], mode="markers",
                marker=dict(size=10, symbol="star", color=t["danger"]),
                showlegend=False,
            ))
            fig_s.update_layout(
                **_plotly_layout(
                    height=250,
                    margin=dict(l=30, r=10, t=35, b=30),
                ),
                xaxis=dict(title="", **_grid(), tickfont=dict(size=8)),
                yaxis=dict(title="", **_grid(), tickfont=dict(size=8)),
                title=dict(
                    text=f"<b>{iter_label}</b> \u2014 {len(snap['archive'])} sol.",
                    font=dict(size=11, color=t["text"]),
                    x=0.5, y=0.97,
                ),
            )
            st.plotly_chart(fig_s, use_container_width=True, theme=None,
                            key=f"sim_snap_{idx}")

    # ── Interpretación ──
    st.markdown(f"""<div class="info-box">
        <strong>Interpretación:</strong> Al inicio, los halcones están dispersos en el espacio de
        los tres objetivos (puntos verdes claros). Conforme avanzan las iteraciones, la bandada se concentra
        cerca del frente de Pareto (línea azul), alejándose del punto FIFO (estrella roja).
        El hipervolumen 3D crece rápidamente al principio y se estabiliza \u2014 señal de convergencia.
        <br><br>
        <strong>Nota:</strong> La simulación optimiza f\u2081, f\u2082 y f\u2083 simultáneamente.
        Usa el selector de proyección arriba para ver diferentes pares de objetivos.
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
    best_f3_global = min(pareto, key=lambda x: x[2])
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
        st.markdown("### Configuración")

        run_idx = st.selectbox(
            "Corrida",
            range(summary["num_runs"]),
            format_func=lambda x: f"#{x+1} (seed={42+x})",
        )

        best_f3 = min(pareto, key=lambda x: x[2])

        scenario_labels = [
            "Min espera (f\u2081) \u2014 prioriza urgencia",
            "Balance (knee) \u2014 compromiso f\u2081/f\u2082/f\u2083",
            "Min brecha entre pa\u00edses (f\u2082) \u2014 igualdad",
            "Mejor f\u2083 (m\u00e1x. utilizaci\u00f3n)",
            "FIFO (sistema actual sin optimizar)",
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
            scenario_labels[3]: best_f3,
            scenario_labels[4]: baseline,
        }
        sel_point = scenario_map[scenario]
        is_fifo = (scenario == scenario_labels[4])

        # Short labels for display
        _short_labels = {
            scenario_labels[0]: "Humanitario",
            scenario_labels[1]: "Equilibrio (Knee)",
            scenario_labels[2]: "Equidad",
            scenario_labels[3]: "M\u00e1x. Utilizaci\u00f3n",
            scenario_labels[4]: "FIFO",
        }
        if is_fifo:
            sel_matrix = data["fifo_matrix"]
            sel_fit = data["fifo_fit"]
            sel_used = data["fifo_used"]
            sel_label = "FIFO"
        else:
            sel_matrix, sel_fit, sel_used = compute_mohho_allocation(
                sel_point[0], sel_point[1], sel_point[2])
            sel_label = _short_labels[scenario]

        st.divider()
        st.markdown("### Escenario activo")

        sel_pct = sel_used * 100 // data["total_visas"]
        sel_f3_sidebar = int(sel_fit[2]) if len(sel_fit) > 2 else data['total_visas'] - sel_used
        st.markdown(f"""
        <div class="sidebar-metric">
            <div class="label">Visas otorgadas ({sel_label})</div>
            <div class="value" style="color:{t['accent3']} !important;">
                {_fmt(sel_used)} / {_fmt(data['total_visas'])}</div>
            <div class="label" style="margin-top:2px;">{sel_pct}% utilizaci\u00f3n</div>
        </div>
        <div class="sidebar-metric">
            <div class="label">f\u2081 espera (a\u00f1os)</div>
            <div class="value">{_fmtd(sel_fit[0])}</div>
        </div>
        <div class="sidebar-metric">
            <div class="label">f\u2082 disparidad (a\u00f1os)</div>
            <div class="value">{_fmtd(sel_fit[1])}</div>
        </div>
        <div class="sidebar-metric">
            <div class="label">f\u2083 desperdicio (visas)</div>
            <div class="value">{_fmt(sel_f3_sidebar)}</div>
        </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("### Métricas globales")

        fifo_f3_val = int(summary['baseline']['f3']) if 'baseline' in summary and 'f3' in summary['baseline'] else data['total_visas'] - data['fifo_used']
        best_f3_val = int(summary['best_f3'][2]) if 'best_f3' in summary else 0
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
        </div>
        <div class="sidebar-metric">
            <div class="label">FIFO desperdicio (f\u2083)</div>
            <div class="value" style="color:{t['danger']} !important;">{_fmt(fifo_f3_val)}</div>
        </div>
        <div class="sidebar-metric">
            <div class="label">Mejor f\u2083 (m\u00e1x. utilizaci\u00f3n)</div>
            <div class="value" style="color:{t['accent3']} !important;">{_fmt(best_f3_val)}</div>
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "El Problema",
        "Frente de Pareto",
        "Asignación",
        "Impacto por País",
        "Convergencia",
        "Simulación en Vivo",
        "Datos y Fuentes",
        "Modelo Matemático",
        "Acerca del Modelo",
    ])

    with tab1:
        _tab_problem(data, summary, pareto, baseline)
    with tab2:
        _tab_pareto(pareto, baseline, knee, sel_point, sel_label,
                    sel_used, data["total_visas"], best_f1, best_f2, data)
    with tab3:
        _tab_allocation(data, sel_matrix, sel_fit, sel_used, sel_label, baseline)
    with tab4:
        _tab_country(data, sel_matrix, sel_fit, sel_used, sel_label,
                     baseline, knee, best_f1, best_f2, best_f3_global)
    with tab5:
        _tab_convergence(iters, hv_means, hv_stds, summary, baseline, run_idx)
    with tab6:
        _tab_simulation(data, baseline)
    with tab7:
        _tab_data_sources(data)
    with tab8:
        _tab_math(data, summary)
    with tab9:
        _tab_about(summary, data)

    # Footer
    st.markdown(f"""
    <div style="text-align:center;padding:3rem 0 1rem;opacity:0.4;font-size:0.75rem;color:{t['text']};">
        <div style="font-size:1rem;font-weight:700;margin-bottom:4px;">
            Visa Predict AI</div>
        MIAAD \u00d7 UACJ 2026<br>
        Yazmín Flores & Javier Rebull<br>
        Optimización Inteligente \u2014 Mtro. Raúl Gibrán Porras Alaniz
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
