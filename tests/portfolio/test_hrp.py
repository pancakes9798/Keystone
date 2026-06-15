"""Tests for HRP Optimizer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jeanclaude.portfolio.covariance.historical import EWMACovariance, SampleCovariance
from jeanclaude.portfolio.optimizer.hrp import HRPOptimizer


N_ASSETS = 6
ASSETS = ["SPY", "TLT", "GLD", "EFA", "IEF", "HYG"]


@pytest.fixture
def returns():
    np.random.seed(42)
    factor = np.random.randn(252, 1) * 0.005
    idio = np.random.randn(252, N_ASSETS) * 0.008
    data = factor + idio
    return pd.DataFrame(
        data,
        columns=ASSETS,
        index=pd.date_range("2020-01-01", periods=252, freq="B"),
    )


def test_weights_sum_to_one(returns):
    optimizer = HRPOptimizer(cov_estimator=EWMACovariance())
    weights = optimizer.optimize(returns)
    assert abs(weights.sum() - 1.0) < 1e-6


def test_weights_are_non_negative(returns):
    """HRP with inverse-variance allocation always produces non-negative weights."""
    optimizer = HRPOptimizer(cov_estimator=EWMACovariance())
    weights = optimizer.optimize(returns)
    assert (weights >= -1e-10).all()


def test_weights_indexed_by_asset_names(returns):
    optimizer = HRPOptimizer(cov_estimator=EWMACovariance())
    weights = optimizer.optimize(returns)
    assert list(weights.index) == ASSETS


def test_hrp_diversifies_across_assets(returns):
    """No single asset should dominate (max weight < 50% for 6 assets)."""
    optimizer = HRPOptimizer(cov_estimator=EWMACovariance())
    weights = optimizer.optimize(returns)
    assert weights.max() < 0.5


def test_hrp_works_with_sample_covariance(returns):
    optimizer = HRPOptimizer(cov_estimator=SampleCovariance())
    weights = optimizer.optimize(returns)
    assert abs(weights.sum() - 1.0) < 1e-6


def test_swappable_covariance_gives_different_weights(returns):
    """EWMA and sample covariance should produce different allocations."""
    w_ewma = HRPOptimizer(cov_estimator=EWMACovariance()).optimize(returns)
    w_sample = HRPOptimizer(cov_estimator=SampleCovariance()).optimize(returns)
    assert not np.allclose(w_ewma.values, w_sample.values)
