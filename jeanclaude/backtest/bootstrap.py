"""Stationary bootstrap confidence intervals for portfolio metrics.

Implements Politis & Romano (1994) stationary bootstrap. Block lengths
are drawn from a geometric distribution with mean b = sqrt(T), which
handles autocorrelation in financial return series without requiring
a fixed block length hyperparameter.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

__all__ = ["BootstrapResult", "stationary_bootstrap_ci"]

from jeanclaude.backtest.metrics import (
    annualized_return,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


@dataclass(frozen=True)
class BootstrapResult:
    """Output of stationary_bootstrap_ci(). Immutable."""
    point: MappingProxyType
    lower: MappingProxyType
    upper: MappingProxyType
    n_samples: int
    confidence: float


def stationary_bootstrap_ci(
    returns: pd.Series,
    n_samples: int = 1000,
    confidence: float = 0.95,
    block_prob: float | None = None,
    seed: int = 42,
) -> BootstrapResult:
    """Compute confidence intervals via stationary bootstrap.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    n_samples : int
        Number of bootstrap replications.
    confidence : float
        CI coverage level, e.g. 0.95 for a 95% CI.
    block_prob : float | None
        Probability parameter p of the geometric block length distribution
        (mean block length = 1/p). None = auto-set to 1/sqrt(T).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    BootstrapResult
    """
    arr = returns.to_numpy()
    if len(arr) == 0:
        raise ValueError("returns is empty.")
    if np.isnan(arr).any():
        raise ValueError(
            f"returns contains {int(np.isnan(arr).sum())} NaN value(s). "
            "Drop or fill NaNs before calling stationary_bootstrap_ci()."
        )
    T = len(arr)

    p = block_prob if block_prob is not None else 1.0 / max(1.0, T ** 0.5)

    point = _compute_metrics(returns)

    rng = np.random.default_rng(seed)
    boot_metrics: dict[str, list[float]] = {k: [] for k in point}

    for _ in range(n_samples):
        sample = _draw_stationary_sample(arr, p, rng)
        s = pd.Series(sample)
        for key, val in _compute_metrics(s).items():
            boot_metrics[key].append(val)

    alpha = (1.0 - confidence) / 2.0
    lower = {k: float(np.quantile(v, alpha)) for k, v in boot_metrics.items()}
    upper = {k: float(np.quantile(v, 1.0 - alpha)) for k, v in boot_metrics.items()}

    return BootstrapResult(
        point=MappingProxyType(point),
        lower=MappingProxyType(lower),
        upper=MappingProxyType(upper),
        n_samples=n_samples,
        confidence=confidence,
    )


def _draw_stationary_sample(arr: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Draw one bootstrap sample of same length as arr using stationary bootstrap."""
    T = len(arr)
    # Pre-generate enough blocks to cover T observations (3x overshoot to avoid refilling)
    expected_blocks = max(1, int(T * p * 3))
    starts = rng.integers(0, T, size=expected_blocks)
    block_lens = rng.geometric(p, size=expected_blocks)

    indices: list[np.ndarray] = []
    total = 0
    for s, bl in zip(starts, block_lens):
        seg = np.arange(s, s + bl) % T
        indices.append(seg)
        total += bl
        if total >= T:
            break
    # If we ran out of pre-generated blocks, keep drawing until covered
    while total < T:
        s = int(rng.integers(0, T))
        bl = int(rng.geometric(p))
        seg = np.arange(s, s + bl) % T
        indices.append(seg)
        total += bl
    return arr[np.concatenate(indices)[:T]]


def _compute_metrics(returns: pd.Series) -> dict[str, float]:
    return {
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "cagr": annualized_return(returns),
        "max_dd": max_drawdown(returns),
        "calmar": calmar_ratio(returns),
    }
