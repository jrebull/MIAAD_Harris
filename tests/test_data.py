"""Tests for data module: groups, spillover, and country caps."""

import pytest
from src.data import build_groups, compute_spillover, compute_country_caps
from src.config import COUNTRIES, CATEGORIES, K_BASE, V, P_C, NUM_GROUPS


def test_build_groups_count():
    """Should produce exactly NUM_GROUPS groups (21 countries x 5 categories)."""
    groups = build_groups()
    assert len(groups) == NUM_GROUPS


def test_build_groups_unique_indices():
    """All group indices should be unique and 0 to NUM_GROUPS-1."""
    groups = build_groups()
    indices = [g["index"] for g in groups]
    assert sorted(indices) == list(range(NUM_GROUPS))


def test_build_groups_waiting_time():
    """w_g = t_actual - d_g, all non-negative."""
    groups = build_groups()
    for g in groups:
        assert g["w"] == 2026 - g["d"]
        assert g["w"] >= 0


def test_build_groups_all_countries_categories():
    """Every country-category pair should appear exactly once."""
    groups = build_groups()
    pairs = {(g["country"], g["category"]) for g in groups}
    expected = {(c, j) for c in COUNTRIES for j in CATEGORIES}
    assert pairs == expected


def test_spillover_sum_at_least_v():
    """Sum of K_eff should be >= V; spillover redistributes but never loses."""
    groups = build_groups()
    k_eff = compute_spillover(groups)
    assert sum(k_eff.values()) >= V


def test_spillover_eb4_eb5_no_incoming():
    """EB-4 and EB-5 effective caps should equal base caps."""
    groups = build_groups()
    k_eff = compute_spillover(groups)
    assert k_eff["EB-4"] == K_BASE["EB-4"]
    assert k_eff["EB-5"] == K_BASE["EB-5"]


def test_spillover_cascade_direction():
    """Spillover flows: EB-4/5 -> EB-1 -> EB-2 -> EB-3.

    With 21-country data, all five categories are overdemanded
    (demand exceeds base cap), so no spillover occurs and every
    effective cap equals its base cap.
    """
    groups = build_groups()
    k_eff = compute_spillover(groups)
    # EB-4/5 stay at base (no incoming spillover)
    assert k_eff["EB-4"] == K_BASE["EB-4"]
    assert k_eff["EB-5"] == K_BASE["EB-5"]
    # EB-1 stays at base (EB-4/5 both overdemanded, no spillover to EB-1)
    assert k_eff["EB-1"] == K_BASE["EB-1"]
    # EB-2 stays at base (EB-1 overdemanded, no spillover to EB-2)
    assert k_eff["EB-2"] == K_BASE["EB-2"]
    # EB-3 stays at base (EB-2 overdemanded, no spillover to EB-3)
    assert k_eff["EB-3"] == K_BASE["EB-3"]


def test_country_caps_explicit_countries():
    """All 20 explicit countries should have cap = P_C (25,620)."""
    groups = build_groups()
    caps = compute_country_caps(groups)
    for country in COUNTRIES:
        if country != "Resto del Mundo":
            assert caps[country] == P_C


def test_country_caps_rdm_non_negative():
    """Resto del Mundo cap should be non-negative."""
    groups = build_groups()
    caps = compute_country_caps(groups)
    assert caps["Resto del Mundo"] >= 0


def test_country_caps_rdm_at_least_pc():
    """Resto del Mundo cap should be at least P_C (safeguard)."""
    groups = build_groups()
    caps = compute_country_caps(groups)
    assert caps["Resto del Mundo"] >= P_C
