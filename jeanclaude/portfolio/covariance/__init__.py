"""
Covariance Estimation Module
=============================

Historical and learned covariance matrix estimators.
All estimators share the same interface:
    estimator.fit(returns: pd.DataFrame) -> self
    estimator.matrix() -> np.ndarray

Usage::

    from jeanclaude.portfolio.covariance import EWMACovariance
    ewma = EWMACovariance(lambda_=0.94).fit(returns)
    cov_matrix = ewma.matrix()
"""

from .historical import EWMACovariance, SampleCovariance
from .ledoit_wolf import LedoitWolfCovariance

__all__ = [
    "EWMACovariance",
    "LedoitWolfCovariance",
    "SampleCovariance",
]
