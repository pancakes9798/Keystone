import asyncio
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from jeanclaude.agents.config import AgentConfig, RebalanceConfig
from jeanclaude.agents.events import (
    NoRebalanceEvent, PriceEvent, RegimeEvent, SignalEvent, WeightsEvent,
)
from jeanclaude.agents.portfolio import PortfolioAgent
from jeanclaude.execution.paper.broker import PortfolioState
from jeanclaude.signals.macro.labels import RegimeLabel


def _make_returns(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n)
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(n, 3)),
        index=dates,
        columns=["XLK", "TLT", "GLD"],
    )


def _make_price_event(returns: pd.DataFrame) -> PriceEvent:
    prices = (1 + returns).cumprod() * 100
    macro = pd.DataFrame({"VIX": np.ones(len(returns)) * 15}, index=returns.index)
    return PriceEvent(date=returns.index[-1], prices=prices, returns=returns, macro=macro)


def _make_regime_event(changed: bool = False) -> RegimeEvent:
    return RegimeEvent(
        date=pd.Timestamp("2024-01-01"),
        label=RegimeLabel.EXPANSION,
        probabilities=np.array([1.0, 0.0, 0.0]),
        labels_history=pd.Series(dtype="int64"),
        changed=changed,
    )


def _make_signal_event(tickers: list[str]) -> SignalEvent:
    views = pd.Series(0.05, index=tickers)
    confidence = pd.Series([1.0, 0.0, 0.0], index=["EXPANSION", "CONTRACTION", "TRANSITION"])
    return SignalEvent(date=pd.Timestamp("2024-01-01"), views=views, confidence=confidence)


def _make_broker_mock(empty_weights: bool = True) -> MagicMock:
    broker = MagicMock()
    if empty_weights:
        broker.get_state.return_value = PortfolioState(
            date=pd.Timestamp("2023-01-01"),
            weights=pd.Series(dtype=float),
            nav=100_000.0,
            cash=100_000.0,
        )
    else:
        broker.get_state.return_value = PortfolioState(
            date=pd.Timestamp("2023-01-01"),
            weights=pd.Series({"XLK": 0.4, "TLT": 0.4, "GLD": 0.2}),
            nav=100_000.0,
            cash=0.0,
        )
    return broker


def test_portfolio_agent_rebalances_on_empty_portfolio(tmp_path):
    returns = _make_returns()
    cfg = AgentConfig(universe=["XLK", "TLT", "GLD"], data_dir=str(tmp_path))
    broker = _make_broker_mock(empty_weights=True)
    agent = PortfolioAgent(cfg, broker)

    price_event = _make_price_event(returns)
    regime_event = _make_regime_event(changed=False)
    signal_event = _make_signal_event(["XLK", "TLT", "GLD"])

    result = asyncio.run(agent.run(signal_event, price_event, regime_event))

    assert isinstance(result, WeightsEvent)
    assert result.reason == "empty_portfolio"
    assert abs(result.weights.sum() - 1.0) < 0.05


def test_portfolio_agent_rebalances_on_regime_change(tmp_path):
    returns = _make_returns()
    cfg = AgentConfig(universe=["XLK", "TLT", "GLD"], data_dir=str(tmp_path))
    broker = _make_broker_mock(empty_weights=False)
    agent = PortfolioAgent(cfg, broker)

    price_event = _make_price_event(returns)
    regime_event = _make_regime_event(changed=True)
    signal_event = _make_signal_event(["XLK", "TLT", "GLD"])

    result = asyncio.run(agent.run(signal_event, price_event, regime_event))

    assert isinstance(result, WeightsEvent)
    assert result.reason == "regime_change"
    assert abs(result.weights.sum() - 1.0) < 0.05


def test_portfolio_agent_no_rebalance_in_cooldown(tmp_path):
    returns = _make_returns()
    cfg = AgentConfig(universe=["XLK", "TLT", "GLD"], data_dir=str(tmp_path))
    broker = _make_broker_mock()
    agent = PortfolioAgent(cfg, broker)
    agent._last_rebalance_date = returns.index[-1]

    price_event = _make_price_event(returns)
    regime_event = _make_regime_event(changed=True)
    signal_event = _make_signal_event(["XLK", "TLT", "GLD"])

    result = asyncio.run(agent.run(signal_event, price_event, regime_event))

    assert isinstance(result, NoRebalanceEvent)
    assert result.reason == "cooldown"


def test_portfolio_agent_no_rebalance_when_no_trigger(tmp_path):
    returns = _make_returns()
    cfg = AgentConfig(universe=["XLK", "TLT", "GLD"], data_dir=str(tmp_path),
                      rebalance=RebalanceConfig(signal_drift_threshold=0.99))
    broker = _make_broker_mock(empty_weights=False)
    agent = PortfolioAgent(cfg, broker)

    price_event = _make_price_event(returns)
    regime_event = _make_regime_event(changed=False)
    signal_event = _make_signal_event(["XLK", "TLT", "GLD"])

    result = asyncio.run(agent.run(signal_event, price_event, regime_event))

    assert isinstance(result, NoRebalanceEvent)
    assert result.reason == "no_trigger"


def test_portfolio_agent_weights_respect_max_weight(tmp_path):
    returns = _make_returns()
    cfg = AgentConfig(universe=["XLK", "TLT", "GLD"], data_dir=str(tmp_path), max_weight=0.4)
    broker = _make_broker_mock()
    agent = PortfolioAgent(cfg, broker)

    price_event = _make_price_event(returns)
    regime_event = _make_regime_event(changed=True)
    signal_event = _make_signal_event(["XLK", "TLT", "GLD"])

    result = asyncio.run(agent.run(signal_event, price_event, regime_event))

    if isinstance(result, WeightsEvent):
        assert (result.weights <= cfg.max_weight + 1e-4).all()
