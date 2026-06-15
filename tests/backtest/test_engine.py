"""Tests for the walk-forward backtesting engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jeanclaude.backtest.engine import BacktestEngine, BacktestConfig, BacktestResult
from jeanclaude.backtest.report import summary
from jeanclaude.portfolio.covariance.historical import EWMACovariance
from jeanclaude.portfolio.optimizer.hrp import HRPOptimizer


N_ASSETS = 4
ASSETS = ["SPY", "TLT", "GLD", "EFA"]
N_DAYS = 300
MIN_HISTORY = 60


@pytest.fixture
def returns():
    np.random.seed(0)
    data = np.random.randn(N_DAYS, N_ASSETS) * 0.01
    return pd.DataFrame(
        data,
        columns=ASSETS,
        index=pd.date_range("2020-01-01", periods=N_DAYS, freq="B"),
    )


@pytest.fixture
def optimizer():
    return HRPOptimizer(cov_estimator=EWMACovariance())


def test_backtest_result_is_correct_type(returns, optimizer):
    engine = BacktestEngine(optimizer=optimizer)
    result = engine.run(returns, min_history=MIN_HISTORY)
    assert isinstance(result, BacktestResult)


def test_portfolio_returns_have_correct_length(returns, optimizer):
    engine = BacktestEngine(optimizer=optimizer)
    result = engine.run(returns, min_history=MIN_HISTORY)
    assert len(result.portfolio_returns) == N_DAYS - MIN_HISTORY


def test_weights_history_columns_match_assets(returns, optimizer):
    engine = BacktestEngine(optimizer=optimizer)
    result = engine.run(returns, min_history=MIN_HISTORY)
    assert list(result.weights_history.columns) == ASSETS


def test_weights_sum_to_one_at_each_rebalance(returns, optimizer):
    engine = BacktestEngine(optimizer=optimizer)
    result = engine.run(returns, min_history=MIN_HISTORY)
    sums = result.weights_history.sum(axis=1)
    assert (abs(sums - 1.0) < 1e-5).all()


def test_no_lookahead_bias(returns):
    """Optimizer must only see data strictly before the rebalance date."""
    call_log = []

    class LoggingOptimizer:
        def optimize(self, hist: pd.DataFrame) -> pd.Series:
            call_log.append(hist.index[-1])
            return pd.Series(0.25, index=ASSETS)

    engine = BacktestEngine(optimizer=LoggingOptimizer())
    result = engine.run(returns, min_history=MIN_HISTORY)

    rebal_dates = result.weights_history.index
    for call_date, rebal_date in zip(call_log, rebal_dates):
        assert call_date < rebal_date, (
            f"Lookahead: optimizer saw data up to {call_date} "
            f"but rebalance was on {rebal_date}"
        )


def test_monthly_rebalance_produces_fewer_trades(returns, optimizer):
    weekly = BacktestEngine(optimizer=optimizer, config=BacktestConfig(rebalance_freq="W", execution_lag=0))  # P1 2026-06-10: legacy same-day
    monthly = BacktestEngine(optimizer=optimizer, config=BacktestConfig(rebalance_freq="ME", execution_lag=0))  # P1 2026-06-10: legacy same-day
    res_w = weekly.run(returns, min_history=MIN_HISTORY)
    res_m = monthly.run(returns, min_history=MIN_HISTORY)
    assert len(res_w.weights_history) > len(res_m.weights_history)


def test_summary_returns_expected_keys(returns, optimizer):
    engine = BacktestEngine(optimizer=optimizer)
    result = engine.run(returns, min_history=MIN_HISTORY)
    s = summary(result)
    for key in ["sharpe_ratio", "max_drawdown", "calmar_ratio", "ann_return"]:
        assert key in s.index
