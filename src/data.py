"""Visa demand data for 21 country blocs x 5 EB categories.

Sources — backlog (n):
    USCIS, "Approved I-140/I-360/I-526 Petitions Awaiting Visa Availability",
    FY2025 Q2 (data as of March 2025).  Primary beneficiaries only.
    https://www.uscis.gov/sites/default/files/document/data/
        eb_i140_i360_i526_performancedata_fy2025_q2.xlsx

    Official USCIS data covers 5 groups: India, China, Mexico, Philippines,
    and All Other.  Only India, China, Philippines, Mexico, and "All Other"
    are disaggregated officially.

Sources — priority dates (d):
    U.S. Department of State, Visa Bulletin for April 2026,
    "Final Action Dates for Employment-Based Preference Cases".
    Published 4 Mar 2026.
    https://travel.state.gov/content/travel/en/legal/visa-law0/
        visa-bulletin/2026/visa-bulletin-for-april-2026.html

    "Current" means no backlog; mapped to d = 2026 (w = 0).

For the 16 countries split from "All Other": n values are ESTIMATES based on
DHS Yearbook FY2023 immigration shares.  Marked [EST].
For "Current" categories (EB-1/EB-5 for non-India/China): small annual demand
estimates added to avoid n=0 dead slots.  Marked [EST-DEM].

Each entry: {"n": approved petitions awaiting visa, "d": priority-date year}
EB-3 includes Other Workers; EB-4 includes Certain Religious Workers.
"""

from src.config import COUNTRIES, CATEGORIES, T_ACTUAL, K_BASE, P_C, V

# ──────────────────────────────────────────────────────────────
# VISA_DATA — official USCIS + Visa Bulletin April 2026
# ──────────────────────────────────────────────────────────────

VISA_DATA: dict[str, dict[str, dict[str, int]]] = {
    # ── OFFICIAL USCIS data (4 countries) ──────────────────────
    "India": {
        "EB-1": {"n": 18_462, "d": 2023},   # VB: 01 APR 2023
        "EB-2": {"n": 331_561, "d": 2014},  # VB: 15 JUL 2014
        "EB-3": {"n": 102_683, "d": 2013},  # VB: 15 NOV 2013 (incl 202 OW)
        "EB-4": {"n": 4_866, "d": 2022},    # VB: 15 JUL 2022 (incl 336 RW)
        "EB-5": {"n": 116, "d": 2022},      # VB: 01 MAY 2022
    },
    "China": {
        "EB-1": {"n": 11_858, "d": 2023},   # VB: 01 APR 2023
        "EB-2": {"n": 36_172, "d": 2021},   # VB: 01 SEP 2021
        "EB-3": {"n": 13_334, "d": 2021},   # VB: 15 JUN 2021 (incl 916 OW)
        "EB-4": {"n": 853, "d": 2022},      # VB: 15 JUL 2022 (incl 129 RW)
        "EB-5": {"n": 12_117, "d": 2016},   # VB: 01 SEP 2016
    },
    "Filipinas": {
        "EB-1": {"n": 300, "d": 2026},      # Current [EST-DEM]
        "EB-2": {"n": 187, "d": 2026},      # Current (USCIS official)
        "EB-3": {"n": 22_040, "d": 2023},   # VB: 01 AUG 2023 (incl 2943 OW)
        "EB-4": {"n": 314, "d": 2022},      # VB: 15 JUL 2022 (incl 86 RW)
        "EB-5": {"n": 50, "d": 2026},       # Current [EST-DEM]
    },
    "Mexico": {
        "EB-1": {"n": 200, "d": 2026},      # Current [EST-DEM]
        "EB-2": {"n": 727, "d": 2026},      # Current (USCIS official)
        "EB-3": {"n": 5_432, "d": 2024},    # VB: 01 JUN 2024 (incl 4392 OW)
        "EB-4": {"n": 12_963, "d": 2022},   # VB: 15 JUL 2022 (incl 198 RW)
        "EB-5": {"n": 50, "d": 2026},       # Current [EST-DEM]
    },
    # ── ESTIMATED from "All Other" USCIS bloc ─────────────────
    # USCIS "All Other" totals: EB-1=0, EB-2=28,456, EB-3=43,855, EB-4=198,442, EB-5=0
    # EB-4 dominated by Afghan/Iraqi SIV programs (~180k combined).
    # Sub-splits [EST] based on DHS Yearbook FY2023 skilled-immigration shares.
    # [EST-DEM] = estimated annual demand for Current categories.
    # All use "All Chargeability" VB dates: EB-1 Current, EB-2 Current,
    # EB-3 01 JUN 2024, EB-4 15 JUL 2022, EB-5 Current.
    "Afganistan": {  # [EST] — SIV program dominates EB-4
        "EB-1": {"n": 30, "d": 2026},       # [EST-DEM]
        "EB-2": {"n": 100, "d": 2025},      # [EST]
        "EB-3": {"n": 500, "d": 2023},      # [EST]
        "EB-4": {"n": 100_000, "d": 2022},  # [EST] Afghan SIV
        "EB-5": {"n": 20, "d": 2026},       # [EST-DEM]
    },
    "Irak": {  # [EST] — SIV program dominates EB-4
        "EB-1": {"n": 30, "d": 2026},       # [EST-DEM]
        "EB-2": {"n": 100, "d": 2025},      # [EST]
        "EB-3": {"n": 400, "d": 2023},      # [EST]
        "EB-4": {"n": 80_000, "d": 2022},   # [EST] Iraqi SIV
        "EB-5": {"n": 20, "d": 2026},       # [EST-DEM]
    },
    "Corea del Sur": {  # [EST] ~4.5% of RoW skilled immigration
        "EB-1": {"n": 1_200, "d": 2026},    # [EST-DEM]
        "EB-2": {"n": 3_000, "d": 2025},    # [EST]
        "EB-3": {"n": 3_500, "d": 2023},    # [EST]
        "EB-4": {"n": 200, "d": 2022},      # [EST]
        "EB-5": {"n": 150, "d": 2026},      # [EST-DEM]
    },
    "Pakistan": {  # [EST] ~3.5%
        "EB-1": {"n": 800, "d": 2026},      # [EST-DEM]
        "EB-2": {"n": 2_500, "d": 2025},    # [EST]
        "EB-3": {"n": 3_000, "d": 2023},    # [EST]
        "EB-4": {"n": 500, "d": 2022},      # [EST]
        "EB-5": {"n": 30, "d": 2026},       # [EST-DEM]
    },
    "Iran": {  # [EST] ~3%
        "EB-1": {"n": 700, "d": 2026},      # [EST-DEM]
        "EB-2": {"n": 2_000, "d": 2025},    # [EST]
        "EB-3": {"n": 1_500, "d": 2023},    # [EST]
        "EB-4": {"n": 300, "d": 2022},      # [EST]
        "EB-5": {"n": 30, "d": 2026},       # [EST-DEM]
    },
    "Taiwan": {  # [EST] ~3%
        "EB-1": {"n": 800, "d": 2026},      # [EST-DEM]
        "EB-2": {"n": 1_500, "d": 2025},    # [EST]
        "EB-3": {"n": 2_000, "d": 2023},    # [EST]
        "EB-4": {"n": 50, "d": 2022},       # [EST]
        "EB-5": {"n": 100, "d": 2026},      # [EST-DEM]
    },
    "Brasil": {  # [EST] ~2.5%
        "EB-1": {"n": 500, "d": 2026},      # [EST-DEM]
        "EB-2": {"n": 1_200, "d": 2025},    # [EST]
        "EB-3": {"n": 1_800, "d": 2023},    # [EST]
        "EB-4": {"n": 100, "d": 2022},      # [EST]
        "EB-5": {"n": 50, "d": 2026},       # [EST-DEM]
    },
    "Canada": {  # [EST] ~2%
        "EB-1": {"n": 600, "d": 2026},      # [EST-DEM]
        "EB-2": {"n": 1_000, "d": 2025},    # [EST]
        "EB-3": {"n": 1_200, "d": 2023},    # [EST]
        "EB-4": {"n": 50, "d": 2022},       # [EST]
        "EB-5": {"n": 50, "d": 2026},       # [EST-DEM]
    },
    "Reino Unido": {  # [EST] ~1.8%
        "EB-1": {"n": 500, "d": 2026},      # [EST-DEM]
        "EB-2": {"n": 900, "d": 2025},      # [EST]
        "EB-3": {"n": 1_000, "d": 2023},    # [EST]
        "EB-4": {"n": 50, "d": 2022},       # [EST]
        "EB-5": {"n": 50, "d": 2026},       # [EST-DEM]
    },
    "Nigeria": {  # [EST] ~2%
        "EB-1": {"n": 300, "d": 2026},      # [EST-DEM]
        "EB-2": {"n": 800, "d": 2025},      # [EST]
        "EB-3": {"n": 1_200, "d": 2023},    # [EST]
        "EB-4": {"n": 500, "d": 2022},      # [EST]
        "EB-5": {"n": 20, "d": 2026},       # [EST-DEM]
    },
    "Japon": {  # [EST] ~1.5%
        "EB-1": {"n": 600, "d": 2026},      # [EST-DEM]
        "EB-2": {"n": 700, "d": 2025},      # [EST]
        "EB-3": {"n": 800, "d": 2023},      # [EST]
        "EB-4": {"n": 50, "d": 2022},       # [EST]
        "EB-5": {"n": 50, "d": 2026},       # [EST-DEM]
    },
    "Bangladesh": {  # [EST] ~1.5%
        "EB-1": {"n": 200, "d": 2026},      # [EST-DEM]
        "EB-2": {"n": 600, "d": 2025},      # [EST]
        "EB-3": {"n": 1_500, "d": 2023},    # [EST]
        "EB-4": {"n": 300, "d": 2022},      # [EST]
        "EB-5": {"n": 20, "d": 2026},       # [EST-DEM]
    },
    "Colombia": {  # [EST] ~1%
        "EB-1": {"n": 200, "d": 2026},      # [EST-DEM]
        "EB-2": {"n": 400, "d": 2025},      # [EST]
        "EB-3": {"n": 800, "d": 2023},      # [EST]
        "EB-4": {"n": 200, "d": 2022},      # [EST]
        "EB-5": {"n": 20, "d": 2026},       # [EST-DEM]
    },
    "Alemania": {  # [EST] ~1%
        "EB-1": {"n": 400, "d": 2026},      # [EST-DEM]
        "EB-2": {"n": 500, "d": 2025},      # [EST]
        "EB-3": {"n": 800, "d": 2023},      # [EST]
        "EB-4": {"n": 50, "d": 2022},       # [EST]
        "EB-5": {"n": 30, "d": 2026},       # [EST-DEM]
    },
    "Vietnam": {  # [EST] ~1.5%
        "EB-1": {"n": 200, "d": 2026},      # [EST-DEM]
        "EB-2": {"n": 400, "d": 2025},      # [EST]
        "EB-3": {"n": 1_200, "d": 2023},    # [EST]
        "EB-4": {"n": 200, "d": 2022},      # [EST]
        "EB-5": {"n": 100, "d": 2026},      # [EST-DEM]
    },
    "Etiopia": {  # [EST] ~0.5%
        "EB-1": {"n": 50, "d": 2026},       # [EST-DEM]
        "EB-2": {"n": 200, "d": 2025},      # [EST]
        "EB-3": {"n": 500, "d": 2023},      # [EST]
        "EB-4": {"n": 1_000, "d": 2022},    # [EST]
        "EB-5": {"n": 20, "d": 2026},       # [EST-DEM]
    },
    "Resto del Mundo": {
        # Remainder after subtracting the 16 estimated blocs from "All Other"
        "EB-1": {"n": 14_890, "d": 2026},   # [EST-DEM] annual demand
        "EB-2": {"n": 12_556, "d": 2025},   # Remaining of 28,456
        "EB-3": {"n": 22_155, "d": 2023},   # Remaining of 43,855
        "EB-4": {"n": 14_892, "d": 2022},   # Remaining of 198,442 after Afg/Irak/others
        "EB-5": {"n": 300, "d": 2026},      # [EST-DEM]
    },
}


def build_groups() -> list[dict]:
    """Build the flat list of G groups with all derived parameters.

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
