"""Tests for TrendMomentumOptimizer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jeanclaude.portfolio.optimizer.trend_momentum import TrendMomentumOptimizer, TrendMomentumParams

N = 300  # enough for MA200 + 12m momentum
ASSETS = ["AAPL", "MSFT", "AMZN", "TLT", "GLD"]


@pytest.fixture
def returns():
    np.random.seed(0)
    data = np.random.randn(N, len(ASSETS)) * 0.01
    return pd.DataFrame(
        data,
        columns=ASSETS,
        index=pd.date_range("2023-01-01", periods=N, freq="B"),
    )


@pytest.fixture
def params_no_filters():
    return TrendMomentumParams(name="test", damp_factor=0.0, momentum_strength=0.0)


@pytest.fixture
def params_with_filters():
    return TrendMomentumParams(name="agg", damp_factor=0.30, momentum_strength=2.0)


def test_weights_sum_to_one(returns, params_no_filters):
    opt = TrendMomentumOptimizer(params_no_filters)
    w = opt.optimize(returns)
    assert abs(w.sum() - 1.0) < 1e-6


def test_weights_non_negative(returns, params_no_filters):
    opt = TrendMomentumOptimizer(params_no_filters)
    w = opt.optimize(returns)
    assert (w >= -1e-9).all()


def test_weights_sum_to_one_with_filters(returns, params_with_filters):
    opt = TrendMomentumOptimizer(params_with_filters)
    w = opt.optimize(returns)
    assert abs(w.sum() - 1.0) < 1e-6


def test_max_weight_cap_respected(returns):
    params = TrendMomentumParams(name="capped", damp_factor=0.0, momentum_strength=0.0, max_weight=0.30)
    opt = TrendMomentumOptimizer(params)
    w = opt.optimize(returns)
    assert (w <= 0.30 + 1e-9).all()


def test_weights_indexed_by_asset_names(returns, params_no_filters):
    opt = TrendMomentumOptimizer(params_no_filters)
    w = opt.optimize(returns)
    assert set(w.index).issubset(set(ASSETS))


def test_too_few_rows_returns_equal_weight():
    short_returns = pd.DataFrame(
        np.random.randn(10, 3) * 0.01,
        columns=["A", "B", "C"],
        index=pd.date_range("2024-01-01", periods=10, freq="B"),
    )
    params = TrendMomentumParams(name="t", damp_factor=0.0, momentum_strength=0.0)
    opt = TrendMomentumOptimizer(params)
    w = opt.optimize(short_returns)
    assert abs(w.sum() - 1.0) < 1e-6
    assert len(w) >= 1


def test_trend_damp_reduces_weight_of_below_ma_asset():
    """Asset whose cumulative price is below MA200 should get dampened weight."""
    np.random.seed(42)
    n = 300
    asset0_rets = np.full(n, -0.003)  # strong downtrend — below MA200
    asset1_rets = np.zeros(n)
    data = np.column_stack([asset0_rets, asset1_rets])
    rets = pd.DataFrame(data, columns=["DOWN", "FLAT"],
                        index=pd.date_range("2023-01-01", periods=n, freq="B"))

    params_nodamp = TrendMomentumParams(name="nd", damp_factor=0.0, momentum_strength=0.0)
    params_damp   = TrendMomentumParams(name="d",  damp_factor=0.3,  momentum_strength=0.0)

    w_nodamp = TrendMomentumOptimizer(params_nodamp).optimize(rets)
    w_damp   = TrendMomentumOptimizer(params_damp).optimize(rets)

    assert w_damp["DOWN"] < w_nodamp["DOWN"]


def test_momentum_tilt_favors_high_return_asset():
    """Asset with higher 12m return should get higher weight when momentum is on."""
    np.random.seed(7)
    n = 300
    # Asset "UP" has strong positive momentum; "FLAT" has zero
    up_rets   = np.full(n, 0.003)
    flat_rets = np.zeros(n)
    data = np.column_stack([up_rets, flat_rets])
    rets = pd.DataFrame(data, columns=["UP", "FLAT"],
                        index=pd.date_range("2023-01-01", periods=n, freq="B"))

    params_nom = TrendMomentumParams(name="nm", damp_factor=0.0, momentum_strength=0.0)
    params_mom = TrendMomentumParams(name="m",  damp_factor=0.0, momentum_strength=2.0)

    w_nom = TrendMomentumOptimizer(params_nom).optimize(rets)
    w_mom = TrendMomentumOptimizer(params_mom).optimize(rets)

    assert w_mom["UP"] > w_nom["UP"]


def test_trend_damp_all_below_ma_returns_valid_weights():
    """When all assets are below MA200 with damp_factor=1.0, should still return valid weights."""
    n = 300
    # Both assets in strong downtrend
    data = np.full((n, 2), -0.003)
    rets = pd.DataFrame(data, columns=["A", "B"],
                        index=pd.date_range("2023-01-01", periods=n, freq="B"))
    params = TrendMomentumParams(name="t", damp_factor=1.0, momentum_strength=0.0)
    opt = TrendMomentumOptimizer(params)
    w = opt.optimize(rets)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= 0).all()
