from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf as _LedoitWolf


class LedoitWolfCovariance:
    """Analytical Ledoit-Wolf shrinkage covariance estimator (Ledoit & Wolf 2004).

    Drop-in replacement for EWMACovariance: same .fit()/.matrix() interface.
    """

    def __init__(self, assume_centered: bool = False) -> None:
        self.assume_centered = assume_centered
        self._cov: np.ndarray | None = None
        self._shrinkage: float | None = None

    def fit(self, returns: pd.DataFrame) -> LedoitWolfCovariance:
        """Fit the analytical Ledoit-Wolf shrinkage estimator to `returns` (T × N)."""
        if returns.isnull().values.any():
            raise ValueError("returns contains NaN values — forward-fill or drop before fitting")
        n_obs, n_assets = returns.shape
        if n_obs <= n_assets:
            raise ValueError(
                f"LedoitWolf requires n_obs > n_assets, got n_obs={n_obs}, n_assets={n_assets}"
            )
        lw = _LedoitWolf(assume_centered=self.assume_centered)
        lw.fit(returns.values.astype(float))
        cov = lw.covariance_
        shrinkage = float(lw.shrinkage_)
        self._cov = cov
        self._shrinkage = shrinkage
        return self

    @property
    def shrinkage(self) -> float:
        if self._shrinkage is None:
            raise RuntimeError("Call fit() before shrinkage")
        return self._shrinkage

    def matrix(self) -> np.ndarray:
        if self._cov is None:
            raise RuntimeError("Call fit() before matrix()")
        return self._cov.copy()
