"""
Pareto frontier computation for alignment method selection.

Given multiple alignment methods with different overhead/safety
tradeoffs, computes the Pareto frontier — the set of methods where
no other method is strictly better on all metrics.

A method dominates another if it has lower overhead AND higher safety.
Methods on the Pareto frontier represent the optimal tradeoffs available.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class ParetoPoint:
    """Point on the Pareto frontier."""
    method: str
    overhead: float   # Lower is better
    safety: float     # Higher is better
    dominated: bool


class ParetoFrontier:
    """Compute and analyze Pareto frontiers for alignment tradeoffs."""

    @staticmethod
    def compute(methods: List[str], overheads: np.ndarray,
                safeties: np.ndarray) -> List[ParetoPoint]:
        """Find the Pareto-optimal methods.

        Parameters
        ----------
        methods : method names
        overheads : overhead values (lower better)
        safeties : safety scores (higher better)

        Returns list of ParetoPoint, with dominated flag.
        """
        n = len(methods)
        dominated = [False] * n

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # j dominates i if j has lower overhead AND higher safety
                if overheads[j] <= overheads[i] and safeties[j] >= safeties[i]:
                    if overheads[j] < overheads[i] or safeties[j] > safeties[i]:
                        dominated[i] = True
                        break

        return [ParetoPoint(methods[i], float(overheads[i]),
                            float(safeties[i]), dominated[i])
                for i in range(n)]

    @staticmethod
    def frontier_only(points: List[ParetoPoint]) -> List[ParetoPoint]:
        """Return only non-dominated points, sorted by overhead."""
        return sorted([p for p in points if not p.dominated],
                      key=lambda p: p.overhead)
