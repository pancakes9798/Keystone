"""Leva nel BacktestEngine: pesi Σ>1 ammessi fino a max_leverage, funding su (Σw−1)."""
import numpy as np
import pandas as pd
import pytest

from jeanclaude.backtest.engine import BacktestConfig, BacktestEngine


class LeveredOptimizer:
    def __init__(self, gross):
        self._g = gross

    def optimize(self, returns):
        n = len(returns.columns)
        return pd.Series(self._g / n, index=returns.columns)


@pytest.fixture
def returns():
    idx = pd.bdate_range("2020-01-01", "2021-12-31")
    rng = np.random.default_rng(9)
    return pd.DataFrame(
        {"A": rng.normal(0.0005, 0.01, len(idx)), "B": rng.normal(0.0003, 0.008, len(idx))},
        index=idx,
    )


def _cfg(**kw):
    base = dict(rebalance_freq="ME", transaction_cost_bps=0.0,
                min_history=10, execution_lag=0)
    base.update(kw)
    return BacktestConfig(**base)


def test_default_max_leverage_is_one_and_rejects_levered_weights(returns):
    """Comportamento legacy: senza opt-in, pesi Σ>1 sollevano ValueError."""
    engine = BacktestEngine(LeveredOptimizer(1.5), config=_cfg())
    with pytest.raises(ValueError, match="leva"):
        engine.run(returns)


def test_levered_weights_accepted_within_cap(returns):
    cfg = _cfg(max_leverage=2.0, funding_rate_annual=0.0)
    result = BacktestEngine(LeveredOptimizer(1.5), config=cfg).run(returns)
    assert result.weights_history.iloc[-1].sum() == pytest.approx(1.5)


def test_levered_weights_above_cap_rejected(returns):
    cfg = _cfg(max_leverage=2.0)
    engine = BacktestEngine(LeveredOptimizer(2.5), config=cfg)
    with pytest.raises(ValueError, match="leva"):
        engine.run(returns)


def test_funding_cost_charged_on_borrowed_fraction(returns):
    """Con leva 1.5 e funding 2% annuo: drag ≈ 0.5 × 2% sul periodo (verifica vs run a costo zero)."""
    cfg0 = _cfg(max_leverage=2.0, funding_rate_annual=0.0)
    cfg2 = _cfg(max_leverage=2.0, funding_rate_annual=0.02)
    nav0 = (1 + BacktestEngine(LeveredOptimizer(1.5), config=cfg0).run(returns).portfolio_returns).prod()
    nav2 = (1 + BacktestEngine(LeveredOptimizer(1.5), config=cfg2).run(returns).portfolio_returns).prod()
    # Count actual levered days from the result (engine starts levering after first rebalance)
    result2 = BacktestEngine(LeveredOptimizer(1.5), config=cfg2).run(returns)
    # First rebalance happens at end of first full month after min_history.
    # Days where weights are active (weights_history is populated from first execution date).
    # Use the portfolio_returns to count days where leverage is active:
    # leverage is active from the first execution date onward.
    first_exec = result2.weights_history.index[0]
    n_days = len(returns.loc[first_exec:]) - 1  # days AFTER first execution (funding applies from next day)
    # Actually funding applies on each day we hold levered weights;
    # count days from first_exec through end (inclusive of first_exec return)
    n_days = len(result2.portfolio_returns.loc[first_exec:])
    expected_drag = (1 + 0.5 * 0.02 / 252) ** (-n_days)
    assert nav2 / nav0 == pytest.approx(expected_drag, rel=5e-3)


def test_no_funding_cost_when_unlevered(returns):
    cfg0 = _cfg(max_leverage=2.0, funding_rate_annual=0.0)
    cfg2 = _cfg(max_leverage=2.0, funding_rate_annual=0.05)
    nav0 = (1 + BacktestEngine(LeveredOptimizer(1.0), config=cfg0).run(returns).portfolio_returns).prod()
    nav2 = (1 + BacktestEngine(LeveredOptimizer(1.0), config=cfg2).run(returns).portfolio_returns).prod()
    assert nav0 == pytest.approx(nav2)


def test_funding_rate_series_used_pointwise(returns):
    """funding_rate come Serie: il tasso del giorno t viene applicato a t."""
    rate = pd.Series(0.0, index=returns.index)
    rate.loc["2021-06":] = 0.04   # tasso sale a metà 2021
    cfg = _cfg(max_leverage=2.0, funding_rate_annual=rate)
    result = BacktestEngine(LeveredOptimizer(1.5), config=cfg).run(returns)
    cfg0 = _cfg(max_leverage=2.0, funding_rate_annual=0.0)
    base = BacktestEngine(LeveredOptimizer(1.5), config=cfg0).run(returns)
    # prima di giugno 2021 le serie coincidono, dopo no
    pre = result.portfolio_returns.loc[:"2021-05"]
    pre0 = base.portfolio_returns.loc[:"2021-05"]
    pd.testing.assert_series_equal(pre, pre0, check_exact=False, atol=1e-12)
    assert (result.portfolio_returns.loc["2021-07":] < base.portfolio_returns.loc["2021-07":]).all()
