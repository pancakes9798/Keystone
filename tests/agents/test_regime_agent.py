import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from jeanclaude.agents.events import PriceEvent, RegimeEvent
from jeanclaude.agents.regime import RegimeAgent
from jeanclaude.signals.macro.labels import RegimeLabel, RegimeResult


def _make_price_event(n: int = 100) -> PriceEvent:
    dates = pd.date_range("2023-01-01", periods=n)
    prices = pd.DataFrame({"XLK": np.linspace(100, 120, n)}, index=dates)
    returns = prices.pct_change().dropna()
    macro = pd.DataFrame({
        "VIX": np.linspace(15, 20, n),
        "Oil": np.linspace(80, 90, n),
    }, index=dates)
    return PriceEvent(date=dates[-1], prices=prices, returns=returns, macro=macro)


def _make_fake_regime_result(n: int, label: RegimeLabel) -> RegimeResult:
    dates = pd.date_range("2023-01-01", periods=n)
    labels = pd.Series([label] * n, index=dates, name="regime")
    proba = np.zeros((n, 3))
    proba[:, label.value] = 1.0
    proba_df = pd.DataFrame(proba, index=dates, columns=[0, 1, 2])
    model = MagicMock()
    return RegimeResult(labels=labels, probabilities=proba_df, model=model)


def test_regime_agent_first_call_not_changed():
    agent = RegimeAgent()
    price_event = _make_price_event()
    fake_result = _make_fake_regime_result(99, RegimeLabel.EXPANSION)

    with patch("jeanclaude.agents.regime.build_state_variables") as mock_sv, \
         patch.object(agent._detector, "fit", return_value=fake_result):
        mock_sv.return_value = pd.DataFrame(
            {"VIX": np.ones(99)},
            index=pd.date_range("2023-01-01", periods=99),
        )
        event = asyncio.run(agent.run(price_event))

    assert isinstance(event, RegimeEvent)
    assert event.changed is False
    assert event.label == RegimeLabel.EXPANSION


def test_regime_agent_detects_change():
    agent = RegimeAgent()
    agent._last_regime = RegimeLabel.EXPANSION

    price_event = _make_price_event()
    fake_result = _make_fake_regime_result(99, RegimeLabel.CONTRACTION)

    with patch("jeanclaude.agents.regime.build_state_variables") as mock_sv, \
         patch.object(agent._detector, "fit", return_value=fake_result):
        mock_sv.return_value = pd.DataFrame(
            {"VIX": np.ones(99)},
            index=pd.date_range("2023-01-01", periods=99),
        )
        event = asyncio.run(agent.run(price_event))

    assert event.changed is True
    assert event.label == RegimeLabel.CONTRACTION


def test_regime_agent_fallback_on_insufficient_data():
    agent = RegimeAgent()
    dates = pd.date_range("2023-01-01", periods=5)
    prices = pd.DataFrame({"XLK": [100.0] * 5}, index=dates)
    macro = pd.DataFrame({"VIX": [15.0] * 5}, index=dates)
    small_event = PriceEvent(
        date=dates[-1],
        prices=prices,
        returns=prices.pct_change().dropna(),
        macro=macro,
    )

    with patch("jeanclaude.agents.regime.build_state_variables") as mock_sv:
        mock_sv.return_value = pd.DataFrame(
            {"VIX": [15.0] * 5},
            index=dates,
        )
        event = asyncio.run(agent.run(small_event))

    assert event.label == RegimeLabel.TRANSITION
    assert event.changed is False


def test_regime_agent_stores_labels_history():
    agent = RegimeAgent()
    price_event = _make_price_event()
    fake_result = _make_fake_regime_result(99, RegimeLabel.EXPANSION)

    with patch("jeanclaude.agents.regime.build_state_variables") as mock_sv, \
         patch.object(agent._detector, "fit", return_value=fake_result):
        mock_sv.return_value = pd.DataFrame(
            {"VIX": np.ones(99)},
            index=pd.date_range("2023-01-01", periods=99),
        )
        event = asyncio.run(agent.run(price_event))

    assert len(event.labels_history) == 99
