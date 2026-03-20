"""Configuration parameters for the MOHHO Green Card optimization problem.

References:
    - HarrisFase02.tex, Section 3 (Parameters)
    - HarrisFase02.tex, Section 5 (Spillover)
"""

# --- Problem parameters (Section 2.2 of prompt / Section 3 of LaTeX) ---
V: int = 140_000                     # Total EB visas available per fiscal year
V_TOTAL: int = 366_000               # Total visas (family + employment), base for 7%
P_C: int = 25_620                    # Per-country cap = 0.07 * V_TOTAL
T_ACTUAL: int = 2026                 # Current fiscal year

# Category base caps (before spillover) — Section 3, Table 5
K_BASE: dict[str, int] = {
    "EB-1": 40_040,
    "EB-2": 40_040,
    "EB-3": 40_040,
    "EB-4": 9_940,
    "EB-5": 9_940,
}

# --- Algorithm parameters (Section 4.1 of prompt) ---
POPULATION_SIZE: int = 50            # N hawks
MAX_ITERATIONS: int = 500            # T iterations
NUM_RUNS: int = 30                   # Independent runs
ARCHIVE_SIZE: int = 100              # Max solutions in Pareto archive
SEED_BASE: int = 42                  # Base seed (run i uses SEED_BASE + i)
LB: float = 0.0                     # Lower bound of continuous space
UB: float = 1.0                     # Upper bound of continuous space
BETA_LEVY: float = 1.5              # Lévy flight parameter

# --- Derived constants ---
NUM_COUNTRIES: int = 10
NUM_CATEGORIES: int = 5
NUM_GROUPS: int = NUM_COUNTRIES * NUM_CATEGORIES  # G = 50

# --- Country list (ordered) ---
COUNTRIES: list[str] = [
    "India", "China", "Filipinas", "Mexico", "Corea del Sur",
    "Brasil", "Canada", "Reino Unido", "Nigeria", "Resto del Mundo",
]

CATEGORIES: list[str] = ["EB-1", "EB-2", "EB-3", "EB-4", "EB-5"]
