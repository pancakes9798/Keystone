"""Tests for portfolio performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jeanclaude.backtest.metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    deflated_sharpe_ratio,
    annualized_return,
)


@pytest.fixture
def positive_returns():
    """252 days of constant 10bps daily return."""
    return pd.Series([0.001] * 252)


@pytest.fixture
def mixed_returns():
    np.random.seed(42)
    return pd.Series(np.random.randn(500) * 0.01)


def test_sharpe_positive_for_positive_returns(positive_returns):
    assert sharpe_ratio(positive_returns) > 0


def test_sharpe_zero_std_returns_zero():
    returns = pd.Series([0.0] * 100)
    assert sharpe_ratio(returns) == 0.0


def test_sharpe_annualized():
    returns = pd.Series([0.001] * 252 + [-0.001] * 252)
    sr = sharpe_ratio(returns)
    assert isinstance(sr, float)


def test_sortino_higher_than_sharpe_for_positive_skew():
    returns = pd.Series([0.005] * 200 + [-0.001] * 52)
    assert sortino_ratio(returns) > sharpe_ratio(returns)


def test_max_drawdown_known_case():
    # Goes up 10%, then drops 50% from peak
    returns = pd.Series([0.1, -0.5, 0.1])
    mdd = max_drawdown(returns)
    assert abs(mdd - 0.5) < 0.01


def test_max_drawdown_always_positive(mixed_returns):
    assert max_drawdown(mixed_returns) >= 0.0


def test_calmar_positive_for_net_positive_returns():
    returns = pd.Series([0.001] * 200 + [-0.0005] * 52)
    assert calmar_ratio(returns) > 0


def test_deflated_sr_between_0_and_1(mixed_returns):
    sr = sharpe_ratio(mixed_returns)
    dsr = deflated_sharpe_ratio(
        sharpe_obs=sr,
        n_trials=10,
        obs=len(mixed_returns),
    )
    assert 0.0 <= dsr <= 1.0


def test_deflated_sr_penalizes_many_trials():
    """More trials → lower DSR for same observed SR."""
    dsr_few = deflated_sharpe_ratio(sharpe_obs=2.0, n_trials=5, obs=252)
    dsr_many = deflated_sharpe_ratio(sharpe_obs=2.0, n_trials=100, obs=252)
    assert dsr_few > dsr_many


def test_annualized_return_known_cagr():
    # 252 giorni di +0.1% al giorno → CAGR ≈ 28.4%
    returns = pd.Series([0.001] * 252)
    cagr = annualized_return(returns)
    assert abs(cagr - ((1.001 ** 252) - 1)) < 1e-6


def test_annualized_return_empty_returns_zero():
    assert annualized_return(pd.Series([], dtype=float)) == 0.0


def test_annualized_return_negative_for_losing_series():
    returns = pd.Series([-0.001] * 252)  # 0.1% daily loss
    cagr = annualized_return(returns)
    assert cagr < 0
    assert abs(cagr - ((0.999 ** 252) - 1)) < 1e-6


def test_annualized_return_matches_calmar_numerator():
    # calmar = annualized_return / max_drawdown → cross-check
    returns = pd.Series([0.001] * 200 + [-0.0005] * 52)
    cagr = annualized_return(returns)
    mdd = max_drawdown(returns)
    assert abs(calmar_ratio(returns) - cagr / mdd) < 1e-9
