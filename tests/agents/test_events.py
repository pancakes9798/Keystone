import numpy as np
import pandas as pd
from jeanclaude.agents.events import (
    PriceEvent, RegimeEvent, SignalEvent, WeightsEvent, NoRebalanceEvent,
)
from jeanclaude.signals.macro.labels import RegimeLabel


def _make_price_event() -> PriceEvent:
    dates = pd.date_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"XLK": [1.0, 1.01, 1.02, 1.01, 1.03]}, index=dates)
    returns = prices.pct_change().dropna()
    macro = pd.DataFrame({"VIX": [15.0, 14.0, 16.0, 15.5, 14.5]}, index=dates)
    return PriceEvent(date=dates[-1], prices=prices, returns=returns, macro=macro)


def test_price_event_fields():
    event = _make_price_event()
    assert event.date == pd.Timestamp("2024-01-05")
    assert "XLK" in event.prices.columns
    assert len(event.returns) == 4


def test_regime_event_fields():
    dates = pd.date_range("2024-01-01", periods=3)
    history = pd.Series(
        [RegimeLabel.EXPANSION, RegimeLabel.EXPANSION, RegimeLabel.CONTRACTION],
        index=dates,
    )
    event = RegimeEvent(
        date=dates[-1],
        label=RegimeLabel.CONTRACTION,
        probabilities=np.array([0.1, 0.8, 0.1]),
        labels_history=history,
        changed=True,
    )
    assert event.label == RegimeLabel.CONTRACTION
    assert event.changed is True
    assert len(event.labels_history) == 3


def test_weights_event_fields():
    weights = pd.Series({"XLK": 0.5, "TLT": 0.5})
    event = WeightsEvent(
        date=pd.Timestamp("2024-01-05"),
        weights=weights,
        reason="regime_change",
    )
    assert abs(event.weights.sum() - 1.0) < 1e-6
    assert event.reason == "regime_change"


def test_no_rebalance_event_fields():
    event = NoRebalanceEvent(date=pd.Timestamp("2024-01-05"), reason="cooldown")
    assert event.reason == "cooldown"


def test_signal_event_fields():
    views = pd.Series({"XLK": 0.08, "TLT": 0.03})
    confidence = pd.Series([0.1, 0.8, 0.1])
    event = SignalEvent(
        date=pd.Timestamp("2024-01-05"),
        views=views,
        confidence=confidence,
    )
    assert "XLK" in event.views.index
    assert len(event.confidence) == 3
