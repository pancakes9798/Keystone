"""Un ciclo orchestrato registra il NAV ESATTAMENTE una volta (bug: 13 righe in 7 run)."""
import asyncio

import pandas as pd
import pytest

from jeanclaude.agents.config import AgentConfig
from jeanclaude.agents.events import NoRebalanceEvent, PriceEvent
from jeanclaude.agents.execution import ExecutionAgent
from jeanclaude.agents.report import ReportAgent


class SpyBroker:
    def __init__(self):
        self.record_calls: list = []
        self.nav = 100_000.0

    def record_daily_nav(self, daily_returns, pre_applied_cost=0.0, date=None):
        self.record_calls.append(date)
        return self.nav

    def get_state(self):
        raise AssertionError("non dovrebbe servire")


def _make_price_event() -> PriceEvent:
    idx = pd.DatetimeIndex([pd.Timestamp("2026-06-08"), pd.Timestamp("2026-06-09")])
    prices = pd.DataFrame({"XLK": [100.0, 101.0]}, index=idx)
    returns = prices.pct_change().dropna()
    return PriceEvent(
        date=pd.Timestamp("2026-06-09"),
        prices=prices,
        returns=returns,
        macro=pd.DataFrame(),
    )


def test_cycle_records_nav_exactly_once():
    """ExecutionAgent registra il NAV; ReportAgent legge solo. P0 2026-06-10."""
    broker = SpyBroker()
    price_event = _make_price_event()
    no_rebal = NoRebalanceEvent(date=price_event.date, reason="no_trigger")

    asyncio.run(ExecutionAgent(broker).run(no_rebal, price_event))
    asyncio.run(ReportAgent(AgentConfig(), broker).run(no_rebal, price_event))

    assert len(broker.record_calls) == 1, (
        f"Atteso 1 record_daily_nav call, trovati {len(broker.record_calls)}"
    )
    assert broker.record_calls[0] == price_event.date  # data logica propagata


def test_data_agent_returns_are_simple_not_log():
    """Il broker compone (1+r): servono simple returns, non log. P0 2026-06-10."""
    import inspect
    from jeanclaude.agents import data as data_mod
    src = inspect.getsource(data_mod)
    assert "np.log(" not in src, (
        "DataAgent calcola log returns ma il broker richiede simple returns"
    )
