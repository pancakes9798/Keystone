"""Historical covariance estimators.

Both estimators expose the same interface:
    estimator.fit(returns: pd.DataFrame) -> self
    estimator.matrix() -> np.ndarray  # shape (n_assets, n_assets)

This makes them drop-in replaceable in HRPOptimizer and, later, the
LSTM covariance estimator (Fase 3) will expose the same interface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class EWMACovariance:
    """Exponentially Weighted Moving Average covariance matrix.

    Parameters
    ----------
    lambda_ : float
        Decay factor in (0, 1). RiskMetrics standard: 0.94.
        Higher = more historical memory. Lower = faster adaptation.
    """

    def __init__(self, lambda_: float = 0.94) -> None:
        if not 0 < lambda_ < 1:
            raise ValueError(f"lambda_ must be in (0, 1), got {lambda_}")
        self.lambda_ = lambda_
        self._cov: np.ndarray | None = None

    def fit(self, returns: pd.DataFrame) -> "EWMACovariance":
        """Compute EWMA covariance from return history.

        Iterates through each row, updating:
            Σ_t = λ * Σ_{t-1} + (1 - λ) * r_t r_t^T
        """
        r = returns.values.astype(float)
        T, n = r.shape

        # Initialize with sample covariance of first 2 obs
        cov = np.cov(r[:2].T) if T >= 2 else np.eye(n) * 1e-4

        for t in range(2, T):
            outer = np.outer(r[t], r[t])
            cov = self.lambda_ * cov + (1 - self.lambda_) * outer

        self._cov = cov
        return self

    def matrix(self) -> np.ndarray:
        """Return a copy of the fitted covariance matrix (n_assets × n_assets)."""
        if self._cov is None:
            raise RuntimeError("Call fit() before matrix()")
        return self._cov.copy()


class SampleCovariance:
    """Standard sample covariance matrix (equal-weight baseline).

    Uses numpy's unbiased estimator (ddof=1).
    """

    def __init__(self) -> None:
        self._cov: np.ndarray | None = None

    def fit(self, returns: pd.DataFrame) -> "SampleCovariance":
        self._cov = np.cov(returns.values.T, ddof=1)
        return self

    def matrix(self) -> np.ndarray:
        """Return a copy of the fitted covariance matrix (n_assets × n_assets)."""
        if self._cov is None:
            raise RuntimeError("Call fit() before matrix()")
        return self._cov.copy()
