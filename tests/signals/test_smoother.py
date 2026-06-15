"""Unit tests for RegimeSmoother state machine."""
from __future__ import annotations

import pytest

from jeanclaude.signals.macro.smoother import RegimeSmoother, SmootherConfig, SmoothedSignal
from jeanclaude.signals.macro.labels import RegimeLabel


def _make_smoother(min_prob: float = 0.65, min_periods: int = 2) -> RegimeSmoother:
    return RegimeSmoother(SmootherConfig(min_prob=min_prob, min_periods=min_periods))


def test_first_call_accepted_immediately():
    """First signal is always accepted regardless of conf."""
    s = _make_smoother()
    result = s.smooth(RegimeLabel.EXPANSION, 0.50)
    assert result.regime == RegimeLabel.EXPANSION


def test_same_regime_noop():
    """Same regime as current always returns current — no pending state needed."""
    s = _make_smoother()
    s.smooth(RegimeLabel.EXPANSION, 0.90)  # establish current
    result = s.smooth(RegimeLabel.EXPANSION, 0.80)
    assert result.regime == RegimeLabel.EXPANSION


def test_switch_after_min_periods():
    """Regime changes only after min_periods consecutive confirmations at min_prob."""
    s = _make_smoother(min_prob=0.65, min_periods=2)
    s.smooth(RegimeLabel.EXPANSION, 0.90)   # establish EXPANSION

    # First confirmation of CONTRACTION — not enough, still EXPANSION
    r1 = s.smooth(RegimeLabel.CONTRACTION, 0.80)
    assert r1.regime == RegimeLabel.EXPANSION

    # Second confirmation — switch fires
    r2 = s.smooth(RegimeLabel.CONTRACTION, 0.80)
    assert r2.regime == RegimeLabel.CONTRACTION


def test_blocked_below_min_prob():
    """Signal below min_prob never triggers pending, regime stays unchanged."""
    s = _make_smoother(min_prob=0.65, min_periods=2)
    s.smooth(RegimeLabel.EXPANSION, 0.90)

    for _ in range(5):
        result = s.smooth(RegimeLabel.CONTRACTION, 0.40)  # below 0.65
        assert result.regime == RegimeLabel.EXPANSION


def test_pending_reset_on_interruption():
    """If pending regime is interrupted before min_periods, counter resets."""
    s = _make_smoother(min_prob=0.65, min_periods=3)
    s.smooth(RegimeLabel.EXPANSION, 0.90)

    s.smooth(RegimeLabel.CONTRACTION, 0.80)  # pending_count = 1
    s.smooth(RegimeLabel.CONTRACTION, 0.80)  # pending_count = 2

    # Interruption: different pending regime resets counter
    r = s.smooth(RegimeLabel.TRANSITION, 0.80)  # pending resets to TRANSITION, count=1
    assert r.regime == RegimeLabel.EXPANSION     # not yet confirmed

    # Two more CONTRACTION — but counter started fresh from 0, not from 2
    s.smooth(RegimeLabel.CONTRACTION, 0.80)  # count=1
    r2 = s.smooth(RegimeLabel.CONTRACTION, 0.80)  # count=2, still need 3
    assert r2.regime == RegimeLabel.EXPANSION


def test_reset_clears_state():
    """reset() brings smoother back to initial state."""
    s = _make_smoother()
    s.smooth(RegimeLabel.EXPANSION, 0.90)
    s.smooth(RegimeLabel.CONTRACTION, 0.80)
    s.reset()
    # After reset, first call accepted immediately
    result = s.smooth(RegimeLabel.CONTRACTION, 0.30)
    assert result.regime == RegimeLabel.CONTRACTION


def test_smoothed_signal_has_regime_and_confidence():
    """SmoothedSignal exposes .regime and .confidence."""
    s = _make_smoother()
    result = s.smooth(RegimeLabel.EXPANSION, 0.85)
    assert isinstance(result, SmoothedSignal)
    assert result.regime == RegimeLabel.EXPANSION
    assert isinstance(result.confidence, float)
    assert result.confidence == pytest.approx(0.85)  # first call: pass-through


def test_full_confidence_on_pending():
    """Confidence is 1.0 when regime change is pending — overlays stay active."""
    s = _make_smoother(min_prob=0.65, min_periods=2)
    s.smooth(RegimeLabel.EXPANSION, 0.90)  # establish current
    # First confirmation — pending, not yet switched
    result = s.smooth(RegimeLabel.CONTRACTION, 0.80)
    assert result.regime == RegimeLabel.EXPANSION
    assert result.confidence == pytest.approx(1.0)  # full confidence: overlays stay on


def test_min_periods_one_switches_immediately():
    """With min_periods=1, any above-threshold signal switches immediately."""
    s = _make_smoother(min_prob=0.65, min_periods=1)
    s.smooth(RegimeLabel.EXPANSION, 0.90)  # establish EXPANSION
    result = s.smooth(RegimeLabel.CONTRACTION, 0.80)  # should switch immediately
    assert result.regime == RegimeLabel.CONTRACTION
