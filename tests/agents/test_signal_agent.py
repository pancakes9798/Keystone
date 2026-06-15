import asyncio

import numpy as np
import pandas as pd

from jeanclaude.agents.events import PriceEvent, RegimeEvent, SignalEvent
from jeanclaude.agents.signal import SignalAgent
from jeanclaude.signals.macro.labels import RegimeLabel


def _make_price_event(n: int = 100) -> PriceEvent:
    dates = pd.date_range("2023-01-01", periods=n)
    prices = pd.DataFrame({
        "XLK": np.linspace(100, 120, n),
        "TLT": np.linspace(90, 95, n),
    }, index=dates)
    returns = prices.pct_change().dropna()
    macro = pd.DataFrame({"VIX": np.ones(n) * 15}, index=dates)
    return PriceEvent(date=dates[-1], prices=prices, returns=returns, macro=macro)


def _make_regime_event(label: RegimeLabel, n: int = 99) -> RegimeEvent:
    dates = pd.date_range("2023-01-02", periods=n)
    history = pd.Series([label] * n, index=dates, name="regime")
    proba = np.zeros(3)
    proba[label.value] = 1.0
    return RegimeEvent(
        date=dates[-1],
        label=label,
        probabilities=proba,
        labels_history=history,
        changed=False,
    )


def test_signal_agent_returns_signal_event():
    agent = SignalAgent()
    price_event = _make_price_event()
    regime_event = _make_regime_event(RegimeLabel.EXPANSION)

    event = asyncio.run(agent.run(regime_event, price_event))

    assert isinstance(event, SignalEvent)
    assert set(event.views.index) == {"XLK", "TLT"}
    assert not event.views.isnull().any()


def test_signal_agent_views_aligned_to_universe():
    agent = SignalAgent()
    price_event = _make_price_event()
    regime_event = _make_regime_event(RegimeLabel.CONTRACTION)

    event = asyncio.run(agent.run(regime_event, price_event))

    assert list(event.views.index) == list(price_event.returns.columns)


def test_signal_agent_confidence_sums_to_one():
    agent = SignalAgent()
    price_event = _make_price_event()
    regime_event = _make_regime_event(RegimeLabel.EXPANSION)

    event = asyncio.run(agent.run(regime_event, price_event))

    assert abs(event.confidence.sum() - 1.0) < 1e-6


def test_signal_agent_empty_history_uses_unconditional_mean():
    agent = SignalAgent()
    price_event = _make_price_event()
    regime_event = RegimeEvent(
        date=price_event.date,
        label=RegimeLabel.TRANSITION,
        probabilities=np.array([0.0, 0.0, 1.0]),
        labels_history=pd.Series(dtype="int64"),
        changed=False,
    )

    event = asyncio.run(agent.run(regime_event, price_event))

    assert isinstance(event, SignalEvent)
    assert not event.views.isnull().any()
