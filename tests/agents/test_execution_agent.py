import asyncio
from unittest.mock import MagicMock

import pandas as pd
import pytest

from jeanclaude.agents.events import NoRebalanceEvent, PriceEvent, WeightsEvent
from jeanclaude.agents.execution import ExecutionAgent
from jeanclaude.execution.paper import RebalanceResult


def _make_price_event() -> PriceEvent:
    dates = pd.date_range("2024-01-01", periods=5)
    prices = pd.DataFrame(
        {"XLK": [100.0, 101.0, 102.0, 103.0, 104.0],
         "TLT": [90.0, 91.0, 92.0, 93.0, 94.0]},
        index=dates,
    )
    returns = prices.pct_change().dropna()
    macro = pd.DataFrame({"VIX": [15.0] * 5}, index=dates)
    return PriceEvent(date=dates[-1], prices=prices, returns=returns, macro=macro)


def _make_weights_event() -> WeightsEvent:
    return WeightsEvent(
        date=pd.Timestamp("2024-01-05"),
        weights=pd.Series({"XLK": 0.6, "TLT": 0.4}),
        reason="regime_change",
    )


def _make_no_rebalance_event() -> NoRebalanceEvent:
    return NoRebalanceEvent(date=pd.Timestamp("2024-01-05"), reason="cooldown")


def test_execution_agent_calls_execute_rebalance_on_weights_event():
    broker = MagicMock()
    broker.execute_rebalance.return_value = RebalanceResult(orders=[], cost_eur=0.0)
    agent = ExecutionAgent(broker)
    price_event = _make_price_event()
    weights_event = _make_weights_event()

    asyncio.run(agent.run(weights_event, price_event))

    broker.execute_rebalance.assert_called_once()
    call_kwargs = broker.execute_rebalance.call_args
    assert call_kwargs is not None


def test_execution_agent_calls_record_daily_nav_on_no_rebalance_event():
    broker = MagicMock()
    agent = ExecutionAgent(broker)
    price_event = _make_price_event()
    no_event = _make_no_rebalance_event()

    # Should not raise
    asyncio.run(agent.run(no_event, price_event))

    broker.record_daily_nav.assert_called_once()


def test_execution_agent_handles_execute_rebalance_exception_gracefully():
    broker = MagicMock()
    broker.execute_rebalance.side_effect = RuntimeError("broker down")
    agent = ExecutionAgent(broker)
    price_event = _make_price_event()
    weights_event = _make_weights_event()

    # Should not raise — soft error handling
    asyncio.run(agent.run(weights_event, price_event))


def test_execution_agent_no_rebalance_skips_execute_rebalance():
    broker = MagicMock()
    agent = ExecutionAgent(broker)
    price_event = _make_price_event()
    no_event = _make_no_rebalance_event()

    asyncio.run(agent.run(no_event, price_event))

    broker.execute_rebalance.assert_not_called()


def test_execution_agent_handles_record_daily_nav_exception_gracefully():
    broker = MagicMock()
    broker.record_daily_nav.side_effect = RuntimeError("nav update failed")
    agent = ExecutionAgent(broker)
    price_event = _make_price_event()
    no_event = _make_no_rebalance_event()

    # Should not raise — soft error handling
    asyncio.run(agent.run(no_event, price_event))


def test_weights_event_applies_rebalance_cost():
    broker = MagicMock()
    broker.execute_rebalance.return_value = RebalanceResult(orders=[], cost_eur=42.0)
    agent = ExecutionAgent(broker)
    price_event = _make_price_event()
    weights_event = _make_weights_event()

    asyncio.run(agent.run(weights_event, price_event))

    broker.apply_rebalance_cost.assert_called_once()
    assert broker.apply_rebalance_cost.call_args.kwargs["cost_eur"] == 42.0


def test_weights_event_zero_cost_skips_apply():
    broker = MagicMock()
    broker.execute_rebalance.return_value = RebalanceResult(orders=[], cost_eur=0.0)
    agent = ExecutionAgent(broker)
    price_event = _make_price_event()
    weights_event = _make_weights_event()

    asyncio.run(agent.run(weights_event, price_event))

    broker.apply_rebalance_cost.assert_not_called()


def test_weights_event_records_nav_before_rebalance():
    """Il NAV è registrato anche sul percorso WeightsEvent, PRIMA del rebalance."""
    broker = MagicMock()
    calls = []
    broker.record_daily_nav.side_effect = lambda **kw: calls.append("nav") or 100_000.0
    broker.execute_rebalance.side_effect = (
        lambda **kw: calls.append("rebal") or RebalanceResult(orders=[], cost_eur=0.0)
    )
    agent = ExecutionAgent(broker)
    price_event = _make_price_event()
    weights_event = _make_weights_event()

    asyncio.run(agent.run(weights_event, price_event))

    assert calls == ["nav", "rebal"]


def test_failed_nav_record_blocks_rebalance():
    """Integrità: se il NAV non è registrato, il rebalance NON deve procedere."""
    broker = MagicMock()
    broker.record_daily_nav.side_effect = RuntimeError("store giù")
    agent = ExecutionAgent(broker)
    price_event = _make_price_event()
    weights_event = _make_weights_event()

    asyncio.run(agent.run(weights_event, price_event))

    broker.execute_rebalance.assert_not_called()
