import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from jeanclaude.agents.config import AgentConfig
from jeanclaude.agents.events import (
    NoRebalanceEvent, PriceEvent, RegimeEvent, SignalEvent, WeightsEvent,
)
from jeanclaude.agents.orchestrator import Orchestrator
from jeanclaude.signals.macro.labels import RegimeLabel


def _make_price_event(date: str = "2024-01-05") -> PriceEvent:
    ts = pd.Timestamp(date)
    dates = pd.date_range(end=ts, periods=200)
    rng = __import__("numpy").random.default_rng(42)
    returns = pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(200, 3)),
        index=dates,
        columns=["XLK", "TLT", "GLD"],
    )
    prices = (1 + returns).cumprod() * 100
    macro = pd.DataFrame({"VIX": [15.0] * 200}, index=dates)
    return PriceEvent(date=ts, prices=prices, returns=returns, macro=macro)


def _make_regime_event(date: str = "2024-01-05") -> RegimeEvent:
    return RegimeEvent(
        date=pd.Timestamp(date),
        label=RegimeLabel.EXPANSION,
        probabilities=np.array([1.0, 0.0, 0.0]),
        labels_history=pd.Series(dtype="int64"),
        changed=False,
    )


def _make_signal_event(date: str = "2024-01-05") -> SignalEvent:
    return SignalEvent(
        date=pd.Timestamp(date),
        views=pd.Series(0.05, index=["XLK", "TLT", "GLD"]),
        confidence=pd.Series([1.0, 0.0, 0.0], index=["EXPANSION", "CONTRACTION", "TRANSITION"]),
    )


def _make_no_rebalance() -> NoRebalanceEvent:
    return NoRebalanceEvent(date=pd.Timestamp("2024-01-05"), reason="cooldown")


def _make_orchestrator() -> Orchestrator:
    cfg = AgentConfig(universe=["XLK", "TLT", "GLD"])
    with patch("jeanclaude.agents.orchestrator.ParquetStore"), \
         patch("jeanclaude.agents.orchestrator.PaperBroker"):
        return Orchestrator(cfg)


def test_orchestrator_happy_path():
    price_event = _make_price_event()
    regime_event = _make_regime_event()
    signal_event = _make_signal_event()
    no_event = _make_no_rebalance()

    orch = _make_orchestrator()

    mock_data = AsyncMock(return_value=price_event)
    mock_regime = AsyncMock(return_value=regime_event)
    mock_signal = AsyncMock(return_value=signal_event)

    with patch.object(orch._data_agent, "run", new=mock_data), \
         patch.object(orch._regime_agent, "run", new=mock_regime), \
         patch.object(orch._signal_agent, "run", new=mock_signal), \
         patch.object(orch._portfolio_agent, "run", new=AsyncMock(return_value=no_event)), \
         patch.object(orch._execution_agent, "run", new=AsyncMock()), \
         patch.object(orch._report_agent, "run", new=AsyncMock()):
        asyncio.run(orch.run_once(pd.Timestamp("2024-01-05")))

    mock_data.assert_awaited_once()
    mock_regime.assert_awaited_once()
    mock_signal.assert_awaited_once()


def test_orchestrator_aborts_on_data_agent_failure():
    orch = _make_orchestrator()

    with patch.object(orch._data_agent, "run", side_effect=RuntimeError("no data")), \
         patch.object(orch._regime_agent, "run", new=AsyncMock()) as mock_regime:
        asyncio.run(orch.run_once(pd.Timestamp("2024-01-05")))
        mock_regime.assert_not_awaited()


def test_orchestrator_continues_with_fallback_on_regime_failure():
    price_event = _make_price_event()
    signal_event = _make_signal_event()
    no_event = _make_no_rebalance()

    orch = _make_orchestrator()

    with patch.object(orch._data_agent, "run", new=AsyncMock(return_value=price_event)), \
         patch.object(orch._regime_agent, "run", side_effect=RuntimeError("hmm failed")), \
         patch.object(orch._signal_agent, "run", new=AsyncMock(return_value=signal_event)) as mock_signal, \
         patch.object(orch._portfolio_agent, "run", new=AsyncMock(return_value=no_event)), \
         patch.object(orch._execution_agent, "run", new=AsyncMock()), \
         patch.object(orch._report_agent, "run", new=AsyncMock()):
        asyncio.run(orch.run_once(pd.Timestamp("2024-01-05")))
        mock_signal.assert_awaited_once()


def test_orchestrator_continues_on_report_agent_failure():
    price_event = _make_price_event()
    regime_event = _make_regime_event()
    signal_event = _make_signal_event()
    no_event = _make_no_rebalance()

    orch = _make_orchestrator()

    with patch.object(orch._data_agent, "run", new=AsyncMock(return_value=price_event)), \
         patch.object(orch._regime_agent, "run", new=AsyncMock(return_value=regime_event)), \
         patch.object(orch._signal_agent, "run", new=AsyncMock(return_value=signal_event)), \
         patch.object(orch._portfolio_agent, "run", new=AsyncMock(return_value=no_event)), \
         patch.object(orch._execution_agent, "run", new=AsyncMock()), \
         patch.object(orch._report_agent, "run", side_effect=RuntimeError("smtp down")):
        # Should not raise
        asyncio.run(orch.run_once(pd.Timestamp("2024-01-05")))
