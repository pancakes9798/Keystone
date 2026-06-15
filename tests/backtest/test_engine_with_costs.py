# tests/backtest/test_engine_with_costs.py
import numpy as np
import pandas as pd
import pytest

from jeanclaude.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from jeanclaude.costs import AlmgrenChrissModel, CostConfig
from jeanclaude.portfolio.optimizer.hrp import HRPOptimizer
from jeanclaude.portfolio.covariance.historical import EWMACovariance


def _make_returns(n=300, k=4, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.normal(0.0005, 0.01, size=(n, k))
    return pd.DataFrame(data, index=idx, columns=["A", "B", "C", "D"])


def _make_adv_spread(returns: pd.DataFrame):
    adv = pd.DataFrame(
        {col: [1_000_000.0] * len(returns) for col in returns.columns},
        index=returns.index,
    )
    spread = pd.DataFrame(
        {col: [0.001] * len(returns) for col in returns.columns},
        index=returns.index,
    )
    return adv, spread


def test_engine_accepts_cost_model():
    """BacktestEngine should accept cost_model without error."""
    returns = _make_returns()
    optimizer = HRPOptimizer(EWMACovariance())
    cost_model = AlmgrenChrissModel(CostConfig(fixed_aum=100_000.0))
    engine = BacktestEngine(optimizer=optimizer, cost_model=cost_model)
    adv, spread = _make_adv_spread(returns)
    result = engine.run(returns, adv=adv, spread=spread)
    assert isinstance(result, BacktestResult)


def test_cost_model_reduces_nav():
    """NAV with Almgren-Chriss costs must be lower than without."""
    returns = _make_returns()
    optimizer = HRPOptimizer(EWMACovariance())
    adv, spread = _make_adv_spread(returns)

    # Without cost model
    engine_plain = BacktestEngine(
        optimizer=optimizer,
        config=BacktestConfig(transaction_cost_bps=0.0, execution_lag=0),  # P1 2026-06-10: legacy same-day
    )
    result_plain = engine_plain.run(returns)
    nav_plain = (1 + result_plain.portfolio_returns).prod()

    # With Almgren-Chriss cost model
    cost_model = AlmgrenChrissModel(CostConfig(fixed_aum=100_000.0))
    engine_ac = BacktestEngine(
        optimizer=optimizer,
        config=BacktestConfig(transaction_cost_bps=0.0, execution_lag=0),  # P1 2026-06-10: legacy same-day
        cost_model=cost_model,
    )
    result_ac = engine_ac.run(returns, adv=adv, spread=spread)
    nav_ac = (1 + result_ac.portfolio_returns).prod()

    assert nav_ac < nav_plain


def test_cost_history_populated():
    """BacktestResult.cost_history should be non-empty Series when cost_model used."""
    returns = _make_returns()
    optimizer = HRPOptimizer(EWMACovariance())
    adv, spread = _make_adv_spread(returns)
    cost_model = AlmgrenChrissModel(CostConfig(fixed_aum=100_000.0))
    engine = BacktestEngine(optimizer=optimizer, cost_model=cost_model)
    result = engine.run(returns, adv=adv, spread=spread)

    assert isinstance(result.cost_history, pd.Series)
    assert len(result.cost_history) > 0
    assert (result.cost_history >= 0).all()


def test_cost_history_empty_without_cost_model():
    """BacktestResult.cost_history should be empty when no cost_model."""
    returns = _make_returns()
    optimizer = HRPOptimizer(EWMACovariance())
    engine = BacktestEngine(optimizer=optimizer)
    result = engine.run(returns)

    assert isinstance(result.cost_history, pd.Series)
    assert len(result.cost_history) == 0


def test_backward_compat_no_adv_spread():
    """Engine with cost_model but no adv/spread falls back to transaction_cost_bps."""
    returns = _make_returns()
    optimizer = HRPOptimizer(EWMACovariance())
    cost_model = AlmgrenChrissModel()
    engine = BacktestEngine(
        optimizer=optimizer,
        config=BacktestConfig(transaction_cost_bps=10.0, execution_lag=0),  # P1 2026-06-10: legacy same-day
        cost_model=cost_model,
    )
    # Should not raise even without adv/spread
    result = engine.run(returns)
    assert isinstance(result, BacktestResult)
