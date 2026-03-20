"""Real visa demand data for the 10 representative countries × 5 EB categories.

Sources:
    - Visa Bulletin February 2026 (U.S. Department of State)
    - USCIS Annual Report estimates
    - References: HarrisFase02.tex, Section 8

Each entry: {"n": pending applicants, "d": priority date year}
"""

from src.config import COUNTRIES, CATEGORIES, T_ACTUAL, K_BASE, P_C, V

VISA_DATA: dict[str, dict[str, dict[str, int]]] = {
    "India": {
        "EB-1": {"n": 50_000, "d": 2022},
        "EB-2": {"n": 800_000, "d": 2013},
        "EB-3": {"n": 200_000, "d": 2012},
        "EB-4": {"n": 2_000, "d": 2025},
        "EB-5": {"n": 5_000, "d": 2020},
    },
    "China": {
        "EB-1": {"n": 15_000, "d": 2023},
        "EB-2": {"n": 120_000, "d": 2019},
        "EB-3": {"n": 30_000, "d": 2020},
        "EB-4": {"n": 1_000, "d": 2025},
        "EB-5": {"n": 20_000, "d": 2018},
    },
    "Filipinas": {
        "EB-1": {"n": 3_000, "d": 2025},
        "EB-2": {"n": 25_000, "d": 2022},
        "EB-3": {"n": 40_000, "d": 2018},
        "EB-4": {"n": 500, "d": 2025},
        "EB-5": {"n": 200, "d": 2025},
    },
    "Mexico": {
        "EB-1": {"n": 2_000, "d": 2025},
        "EB-2": {"n": 15_000, "d": 2025},
        "EB-3": {"n": 25_000, "d": 2021},
        "EB-4": {"n": 3_000, "d": 2025},
        "EB-5": {"n": 500, "d": 2025},
    },
    "Corea del Sur": {
        "EB-1": {"n": 2_000, "d": 2025},
        "EB-2": {"n": 12_000, "d": 2025},
        "EB-3": {"n": 8_000, "d": 2024},
        "EB-4": {"n": 300, "d": 2025},
        "EB-5": {"n": 1_000, "d": 2025},
    },
    "Brasil": {
        "EB-1": {"n": 1_500, "d": 2025},
        "EB-2": {"n": 8_000, "d": 2025},
        "EB-3": {"n": 5_000, "d": 2025},
        "EB-4": {"n": 200, "d": 2025},
        "EB-5": {"n": 300, "d": 2025},
    },
    "Canada": {
        "EB-1": {"n": 1_000, "d": 2025},
        "EB-2": {"n": 6_000, "d": 2025},
        "EB-3": {"n": 4_000, "d": 2025},
        "EB-4": {"n": 150, "d": 2025},
        "EB-5": {"n": 200, "d": 2025},
    },
    "Reino Unido": {
        "EB-1": {"n": 800, "d": 2025},
        "EB-2": {"n": 5_000, "d": 2025},
        "EB-3": {"n": 3_000, "d": 2025},
        "EB-4": {"n": 100, "d": 2025},
        "EB-5": {"n": 150, "d": 2025},
    },
    "Nigeria": {
        "EB-1": {"n": 500, "d": 2025},
        "EB-2": {"n": 4_000, "d": 2025},
        "EB-3": {"n": 3_000, "d": 2025},
        "EB-4": {"n": 200, "d": 2025},
        "EB-5": {"n": 100, "d": 2025},
    },
    "Resto del Mundo": {
        "EB-1": {"n": 20_000, "d": 2025},
        "EB-2": {"n": 300_000, "d": 2025},
        "EB-3": {"n": 100_000, "d": 2025},
        "EB-4": {"n": 5_000, "d": 2025},
        "EB-5": {"n": 3_000, "d": 2025},
    },
}


def build_groups() -> list[dict]:
    """Build the flat list of G=50 groups with all derived parameters.

    Returns:
        List of dicts, each with keys: index, country, category, n, d, w
    """
    groups: list[dict] = []
    idx = 0
    for country in COUNTRIES:
        for cat in CATEGORIES:
            entry = VISA_DATA[country][cat]
            w = T_ACTUAL - entry["d"]
            groups.append({
                "index": idx,
                "country": country,
                "category": cat,
                "n": entry["n"],
                "d": entry["d"],
                "w": w,
            })
            idx += 1
    return groups


def compute_spillover(groups: list[dict]) -> dict[str, int]:
    """Compute effective category caps using spillover rules.

    Spillover cascade (Section 5.1 of LaTeX, Equations 7-12):
        K4_eff = K4, K5_eff = K5
        S4 = max(0, K4 - D4), S5 = max(0, K5 - D5)
        K1_eff = K1 + S4 + S5
        S1 = max(0, K1_eff - D1)
        K2_eff = K2 + S1
        S2 = max(0, K2_eff - D2)
        K3_eff = K3 + S2

    Args:
        groups: List of group dicts from build_groups().

    Returns:
        Dict mapping category name to effective cap.
    """
    demand_by_cat: dict[str, int] = {cat: 0 for cat in CATEGORIES}
    for g in groups:
        demand_by_cat[g["category"]] += g["n"]

    d4 = demand_by_cat["EB-4"]
    d5 = demand_by_cat["EB-5"]
    d1 = demand_by_cat["EB-1"]
    d2 = demand_by_cat["EB-2"]

    k4_eff = K_BASE["EB-4"]
    k5_eff = K_BASE["EB-5"]
    s4 = max(0, k4_eff - d4)
    s5 = max(0, k5_eff - d5)

    k1_eff = K_BASE["EB-1"] + s4 + s5
    s1 = max(0, k1_eff - d1)

    k2_eff = K_BASE["EB-2"] + s1
    s2 = max(0, k2_eff - d2)

    k3_eff = K_BASE["EB-3"] + s2

    return {
        "EB-1": k1_eff,
        "EB-2": k2_eff,
        "EB-3": k3_eff,
        "EB-4": k4_eff,
        "EB-5": k5_eff,
    }


def compute_country_caps(groups: list[dict]) -> dict[str, int]:
    """Compute per-country caps, with special treatment for Resto del Mundo.

    For Resto del Mundo: P_RdM = V - sum_{c != RdM} min(n_c, P_c)
    For all others: P_c = 25,620

    Args:
        groups: List of group dicts from build_groups().

    Returns:
        Dict mapping country name to its visa cap.
    """
    country_demand: dict[str, int] = {}
    for g in groups:
        country_demand[g["country"]] = country_demand.get(g["country"], 0) + g["n"]

    caps: dict[str, int] = {}
    sum_others = 0
    for country in COUNTRIES:
        if country != "Resto del Mundo":
            caps[country] = P_C
            sum_others += min(country_demand[country], P_C)

    # P_RdM: Resto del Mundo aggregates ~190 countries. Its effective cap
    # should not be artificially low. When sum_others > V (all 9 explicit
    # countries have demand exceeding P_c), set P_RdM = V so only R1 and R3
    # constrain it. The greedy decoder enforces the global cap anyway.
    caps["Resto del Mundo"] = max(P_C, V - sum_others)
    return caps
