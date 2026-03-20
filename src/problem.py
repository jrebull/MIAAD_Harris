"""Problem definition: objective functions f1, f2 and problem setup.

References:
    - HarrisFase02.tex, Section 4 (Equations 3-5)
    - f1: Unserved waiting load (minimize)
    - f2: Maximum disparity between countries (minimize)
"""

from src.config import T_ACTUAL
from src.data import build_groups, compute_spillover, compute_country_caps


class VisaProblem:
    """Encapsulates the bi-objective visa allocation problem.

    Attributes:
        groups: List of G=50 group dicts.
        country_caps: Per-country visa caps.
        category_caps: Effective category caps after spillover.
        total_visas: V = 140,000.
        total_demand: Sum of all n_g.
    """

    def __init__(self) -> None:
        self.groups = build_groups()
        self.category_caps = compute_spillover(self.groups)
        self.country_caps = compute_country_caps(self.groups)
        self.total_visas = 140_000
        self.total_demand = sum(g["n"] for g in self.groups)

        self._groups_by_country: dict[str, list[dict]] = {}
        for g in self.groups:
            self._groups_by_country.setdefault(g["country"], []).append(g)

        self._country_w_max: dict[str, int] = {}
        for country, gs in self._groups_by_country.items():
            self._country_w_max[country] = max(g["w"] for g in gs)

    def evaluate(self, x: dict[int, int]) -> tuple[float, float]:
        """Evaluate both objectives for a given allocation.

        Args:
            x: Dict mapping group index to visas assigned.

        Returns:
            Tuple (f1, f2).
        """
        return self.f1(x), self.f2(x)

    def f1(self, x: dict[int, int]) -> float:
        """Unserved waiting load (Equation 3 of LaTeX).

        f1(x) = sum_g (n_g - x_g) * w_g / sum_g n_g

        Minimizing f1 prioritizes groups with high waiting time.
        """
        numerator = sum((g["n"] - x[g["index"]]) * g["w"] for g in self.groups)
        return numerator / self.total_demand

    def f2(self, x: dict[int, int]) -> float:
        """Maximum disparity between countries (Equations 4-5 of LaTeX).

        W_c = (sum x_g * w_g) / (sum x_g) for country c, or w_c_max if sum x_g == 0.
        f2 = max_{c1, c2} |W_c1 - W_c2|
        """
        w_country: dict[str, float] = {}

        for country, gs in self._groups_by_country.items():
            total_assigned = sum(x[g["index"]] for g in gs)
            if total_assigned > 0:
                weighted_wait = sum(x[g["index"]] * g["w"] for g in gs)
                w_country[country] = weighted_wait / total_assigned
            else:
                w_country[country] = float(self._country_w_max[country])

        w_values = list(w_country.values())
        return max(w_values) - min(w_values)
