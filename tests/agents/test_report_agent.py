"""Tests for ReportAgent."""
from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from jeanclaude.agents.config import AgentConfig
from jeanclaude.agents.events import NoRebalanceEvent, PriceEvent, WeightsEvent
from jeanclaude.agents.report import ReportAgent
from jeanclaude.execution.paper.broker import PortfolioState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_event(date: str = "2024-01-05") -> PriceEvent:
    ts = pd.Timestamp(date)
    dates = pd.date_range(end=ts, periods=5)
    prices = pd.DataFrame(
        {"XLK": [100.0, 101.0, 102.0, 103.0, 104.0]},
        index=dates,
    )
    returns = prices.pct_change().dropna()
    macro = pd.DataFrame({"VIX": [15.0] * 5}, index=dates)
    return PriceEvent(date=ts, prices=prices, returns=returns, macro=macro)


def _make_weights_event(date: str = "2024-01-05") -> WeightsEvent:
    return WeightsEvent(
        date=pd.Timestamp(date),
        weights=pd.Series({"XLK": 1.0}),
        reason="regime_change",
    )


def _make_no_event(date: str = "2024-01-04") -> NoRebalanceEvent:
    return NoRebalanceEvent(date=pd.Timestamp(date), reason="cooldown")


def _make_broker_mock() -> MagicMock:
    # P0 2026-06-10: ReportAgent legge broker.nav (proprietà); record_daily_nav è ora
    # chiamato solo da ExecutionAgent. Il mock espone nav come attributo.
    broker = MagicMock()
    broker.nav = 100_000.0
    broker.record_daily_nav.return_value = 100_000.0
    broker.get_state.return_value = PortfolioState(
        date=pd.Timestamp("2024-01-05"),
        weights=pd.Series({"XLK": 1.0}),
        nav=100_000.0,
        cash=0.0,
    )
    return broker


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_report_agent_runs_without_error_on_no_rebalance():
    """ReportAgent deve completare senza eccezione su NoRebalanceEvent.

    P0 2026-06-10: ReportAgent legge broker.nav, non chiama più record_daily_nav.
    """
    cfg = AgentConfig(universe=["XLK"])
    broker = _make_broker_mock()
    agent = ReportAgent(cfg, broker)
    asyncio.run(agent.run(_make_no_event(), _make_price_event("2024-01-04")))
    broker.record_daily_nav.assert_not_called()


def test_report_agent_runs_without_error_on_weights_event():
    """ReportAgent deve completare senza eccezione su WeightsEvent.

    P0 2026-06-10: ReportAgent legge broker.nav, non chiama più record_daily_nav.
    """
    cfg = AgentConfig(universe=["XLK"])
    broker = _make_broker_mock()
    agent = ReportAgent(cfg, broker)
    asyncio.run(agent.run(_make_weights_event(), _make_price_event()))
    broker.record_daily_nav.assert_not_called()


def test_report_agent_does_not_send_email_without_recipients():
    """Senza report_recipients il mailer non viene inizializzato."""
    cfg = AgentConfig(universe=["XLK"], report_recipients=[])
    broker = _make_broker_mock()
    agent = ReportAgent(cfg, broker)
    # 2024-01-05 è venerdì — eppure non deve tentare invio
    asyncio.run(agent.run(_make_no_event("2024-01-05"), _make_price_event("2024-01-05")))
    assert agent._mailer is None


def test_report_agent_handles_nav_read_failure_gracefully():
    """Se broker.nav solleva un'eccezione, ReportAgent restituisce 0.0 e non propaga.

    P0 2026-06-10: ReportAgent legge broker.nav (property), non chiama più
    record_daily_nav né get_state. La proprietà nav è simulata come property che lancia.
    """
    cfg = AgentConfig(universe=["XLK"])
    broker = MagicMock()
    # Simulare broker.nav come property che solleva
    type(broker).nav = property(lambda self: (_ for _ in ()).throw(RuntimeError("store unavailable")))
    agent = ReportAgent(cfg, broker)
    # Non deve propagare — NAV restituisce 0.0 internamente
    asyncio.run(agent.run(_make_no_event(), _make_price_event("2024-01-04")))


def test_report_agent_handles_get_state_failure_gracefully():
    """Se entrambe le chiamate falliscono, non deve propagare l'eccezione."""
    cfg = AgentConfig(universe=["XLK"])
    broker = MagicMock()
    broker.record_daily_nav.side_effect = RuntimeError("nav store down")
    broker.get_state.side_effect = RuntimeError("state store down")
    agent = ReportAgent(cfg, broker)
    asyncio.run(agent.run(_make_no_event(), _make_price_event("2024-01-04")))


def test_report_agent_sends_email_on_friday_with_recipients():
    """Il venerdì con recipients configurati e mailer mock deve chiamare send."""
    cfg = AgentConfig(universe=["XLK"], report_recipients=["test@example.com"])
    broker = _make_broker_mock()
    agent = ReportAgent(cfg, broker)

    mock_mailer = MagicMock()
    agent._mailer = mock_mailer

    # 2024-01-05 = venerdì
    asyncio.run(agent.run(_make_no_event("2024-01-05"), _make_price_event("2024-01-05")))

    mock_mailer.send.assert_called_once()
    call_kwargs = mock_mailer.send.call_args
    assert call_kwargs.kwargs["recipients"] == ["test@example.com"]
    assert "2024-01-05" in call_kwargs.kwargs["subject"]


def test_report_agent_sends_email_on_rebalance_regardless_of_day():
    """Su WeightsEvent deve inviare email anche se non è venerdì."""
    cfg = AgentConfig(universe=["XLK"], report_recipients=["test@example.com"])
    broker = _make_broker_mock()
    agent = ReportAgent(cfg, broker)

    mock_mailer = MagicMock()
    agent._mailer = mock_mailer

    # 2024-01-04 = giovedì, ma c'è stato un ribilanciamento
    asyncio.run(agent.run(_make_weights_event("2024-01-04"), _make_price_event("2024-01-04")))

    mock_mailer.send.assert_called_once()


def test_report_agent_does_not_send_email_on_non_friday_no_rebalance():
    """In un giorno non-venerdì senza ribilanciamento non deve inviare email."""
    cfg = AgentConfig(universe=["XLK"], report_recipients=["test@example.com"])
    broker = _make_broker_mock()
    agent = ReportAgent(cfg, broker)

    mock_mailer = MagicMock()
    agent._mailer = mock_mailer

    # 2024-01-04 = giovedì, nessun ribilanciamento
    asyncio.run(agent.run(_make_no_event("2024-01-04"), _make_price_event("2024-01-04")))

    mock_mailer.send.assert_not_called()


def test_report_agent_handles_email_send_failure_gracefully():
    """Se l'invio email fallisce, non deve propagare l'eccezione."""
    cfg = AgentConfig(universe=["XLK"], report_recipients=["test@example.com"])
    broker = _make_broker_mock()
    agent = ReportAgent(cfg, broker)

    mock_mailer = MagicMock()
    mock_mailer.send.side_effect = ConnectionError("SMTP unreachable")
    agent._mailer = mock_mailer

    # Venerdì — deve tentare invio ma non sollevare
    asyncio.run(agent.run(_make_no_event("2024-01-05"), _make_price_event("2024-01-05")))


def test_report_agent_build_html_no_rebalance():
    """_build_html deve includere 'NO' per NoRebalanceEvent."""
    cfg = AgentConfig(universe=["XLK"])
    broker = _make_broker_mock()
    agent = ReportAgent(cfg, broker)
    html = agent._build_html(_make_no_event(), _make_price_event("2024-01-04"), 98_000.0)
    assert "NO" in html
    assert "98,000.00" in html


def test_report_agent_build_html_weights_event():
    """_build_html deve includere pesi e 'YES' per WeightsEvent."""
    cfg = AgentConfig(universe=["XLK"])
    broker = _make_broker_mock()
    agent = ReportAgent(cfg, broker)
    html = agent._build_html(_make_weights_event(), _make_price_event(), 101_000.0)
    assert "YES" in html
    assert "XLK" in html
    assert "100.0%" in html
