"""Tests for stationary bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jeanclaude.backtest.bootstrap import BootstrapResult, stationary_bootstrap_ci
from jeanclaude.backtest.metrics import sharpe_ratio, annualized_return, max_drawdown


@pytest.fixture
def iid_returns():
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0005, 0.01, 504))  # 2 anni di ritorni


@pytest.fixture
def trending_returns():
    """Ritorni con autocorrelazione positiva (AR(1) con ρ=0.3)."""
    rng = np.random.default_rng(1)
    n = 504
    eps = rng.normal(0, 0.01, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = 0.3 * r[t - 1] + eps[t]
    return pd.Series(r)


def test_returns_bootstrap_result_type(iid_returns):
    result = stationary_bootstrap_ci(iid_returns, n_samples=100, seed=0)
    assert isinstance(result, BootstrapResult)
    assert set(result.point.keys()) == {"sharpe", "sortino", "cagr", "max_dd", "calmar"}
    assert set(result.lower.keys()) == set(result.point.keys())
    assert set(result.upper.keys()) == set(result.point.keys())


def test_point_estimates_match_direct_calculation(iid_returns):
    result = stationary_bootstrap_ci(iid_returns, n_samples=100, seed=0)
    assert abs(result.point["sharpe"] - sharpe_ratio(iid_returns)) < 1e-9
    assert abs(result.point["cagr"] - annualized_return(iid_returns)) < 1e-9
    assert abs(result.point["max_dd"] - max_drawdown(iid_returns)) < 1e-9


def test_ci_lower_le_point_le_upper(iid_returns):
    result = stationary_bootstrap_ci(iid_returns, n_samples=200, seed=0)
    for key in result.point:
        assert result.lower[key] <= result.point[key] <= result.upper[key], (
            f"CI ordering violated for {key}: "
            f"[{result.lower[key]:.4f}, {result.point[key]:.4f}, {result.upper[key]:.4f}]"
        )


def test_reproducible_with_seed(iid_returns):
    r1 = stationary_bootstrap_ci(iid_returns, n_samples=100, seed=42)
    r2 = stationary_bootstrap_ci(iid_returns, n_samples=100, seed=42)
    assert r1.lower == r2.lower
    assert r1.upper == r2.upper


def test_different_seeds_differ(iid_returns):
    r1 = stationary_bootstrap_ci(iid_returns, n_samples=100, seed=1)
    r2 = stationary_bootstrap_ci(iid_returns, n_samples=100, seed=2)
    assert r1.lower["sharpe"] != r2.lower["sharpe"]


def test_autocorrelated_returns_wider_ci(iid_returns, trending_returns):
    """Block bootstrap deve produrre CI più larghi per ritorni autocorrelati."""
    r_iid = stationary_bootstrap_ci(iid_returns, n_samples=500, seed=0)
    r_ar1 = stationary_bootstrap_ci(trending_returns, n_samples=500, seed=0)
    width_iid = r_iid.upper["sharpe"] - r_iid.lower["sharpe"]
    width_ar1 = r_ar1.upper["sharpe"] - r_ar1.lower["sharpe"]
    assert width_ar1 > width_iid  # AR(1) should produce strictly wider CI than iid


def test_custom_block_prob(iid_returns):
    result = stationary_bootstrap_ci(iid_returns, n_samples=50, block_prob=0.5, seed=0)
    assert isinstance(result, BootstrapResult)
    assert result.n_samples == 50


def test_confidence_stored_in_result(iid_returns):
    result = stationary_bootstrap_ci(iid_returns, n_samples=50, confidence=0.90, seed=0)
    assert result.confidence == 0.90


def test_raises_on_nan_returns(iid_returns):
    returns_with_nan = iid_returns.copy()
    returns_with_nan.iloc[10] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        stationary_bootstrap_ci(returns_with_nan, n_samples=10, seed=0)


def test_raises_on_empty_returns():
    with pytest.raises(ValueError, match="empty"):
        stationary_bootstrap_ci(pd.Series([], dtype=float), n_samples=10, seed=0)
