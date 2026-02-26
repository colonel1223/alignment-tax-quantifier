"""
Scaling law models for alignment overhead.

Fits power-law relationships between model scale and alignment tax:

    tax(N) = a * N^b + c

where N is parameter count. Empirically, alignment overhead tends to
decrease with scale (larger models are easier to align per parameter),
but total cost increases due to the base compute growing faster.

References
----------
Hoffmann et al. (2022). "Training Compute-Optimal Large Language Models."
Kaplan et al. (2020). "Scaling Laws for Neural Language Models."
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class ScalingFit:
    """Result of scaling law fit."""
    a: float
    b: float
    c: float
    r_squared: float
    method: str


class ScalingLawModel:
    """Fit and extrapolate alignment tax scaling laws."""

    def __init__(self) -> None:
        self._fit: Optional[ScalingFit] = None

    def fit(self, param_counts: np.ndarray, overheads: np.ndarray,
            method: str = "least_squares") -> ScalingFit:
        """Fit tax(N) = a * N^b + c in log space.

        Parameters
        ----------
        param_counts : array of model sizes
        overheads : array of overhead percentages
        """
        log_n = np.log(param_counts)
        log_o = np.log(np.maximum(overheads, 1e-10))

        # Linear fit in log space: log(overhead) = log(a) + b*log(N)
        coeffs = np.polyfit(log_n, log_o, 1)
        b = coeffs[0]
        a = np.exp(coeffs[1])
        c = 0.0

        # R-squared
        predicted = a * param_counts ** b + c
        ss_res = np.sum((overheads - predicted) ** 2)
        ss_tot = np.sum((overheads - overheads.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        self._fit = ScalingFit(a=float(a), b=float(b), c=float(c),
                                r_squared=float(r2), method=method)
        return self._fit

    def predict(self, param_count: float) -> float:
        """Extrapolate overhead to new scale."""
        if self._fit is None:
            raise RuntimeError("Call fit() first")
        return self._fit.a * param_count ** self._fit.b + self._fit.c
