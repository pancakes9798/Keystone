"""Tests for NoTradeBandOverlay.

Sprint M (2026-06-11). TDD — write FIRST, see RED before implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_series(weights: dict[str, float]) -> pd.Series:
    return pd.Series(weights, dtype=float)


def _make_mock_optimizer(return_sequence: list[pd.Series]) -> MagicMock:
    """Return a mock whose optimize() cycles through the given Series list."""
    mock = MagicMock()
    mock.optimize.side_effect = return_sequence
    return mock


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestNoTradeBandOverlay:
    """All five spec cases from Sprint M brief."""

    def test_first_call_always_trades(self):
        """First rebalance always executes regardless of threshold."""
        from jeanclaude.portfolio.optimizer.no_trade_band import NoTradeBandOverlay

        w0 = _make_series({"A": 0.6, "B": 0.4})
        mock = _make_mock_optimizer([w0])
        overlay = NoTradeBandOverlay(mock, threshold=0.99)

        returns = pd.DataFrame({"A": [0.01], "B": [-0.01]})
        result = overlay.optimize(returns)

        pd.testing.assert_series_equal(result, w0)
        mock.optimize.assert_called_once()

    def test_second_call_below_threshold_returns_previous_weights(self):
        """Below-threshold call returns the PREVIOUS weights; base is still consulted."""
        from jeanclaude.portfolio.optimizer.no_trade_band import NoTradeBandOverlay

        w0 = _make_series({"A": 0.6, "B": 0.4})
        w1 = _make_series({"A": 0.62, "B": 0.38})  # turnover = |0.02|+|0.02| = 0.04
        mock = _make_mock_optimizer([w0, w1])
        overlay = NoTradeBandOverlay(mock, threshold=0.10)

        returns = pd.DataFrame({"A": [0.01], "B": [-0.01]})

        overlay.optimize(returns)             # call 1 — establishes last_weights = w0
        result = overlay.optimize(returns)    # call 2 — proposed turnover 0.04 < 0.10

        # should return w0 (previous), not w1
        pd.testing.assert_series_equal(result, w0)
        # base optimizer was still consulted on call 2
        assert mock.optimize.call_count == 2

    def test_second_call_above_threshold_returns_new_weights(self):
        """Above-threshold call returns the new weights and updates state."""
        from jeanclaude.portfolio.optimizer.no_trade_band import NoTradeBandOverlay

        w0 = _make_series({"A": 0.6, "B": 0.4})
        w1 = _make_series({"A": 0.30, "B": 0.70})  # turnover = 0.30+0.30 = 0.60
        w2 = _make_series({"A": 0.20, "B": 0.80})  # turnover from w1 = 0.10+0.10 = 0.20 > 0.10
        mock = _make_mock_optimizer([w0, w1, w2])
        overlay = NoTradeBandOverlay(mock, threshold=0.10)

        returns = pd.DataFrame({"A": [0.01], "B": [-0.01]})

        overlay.optimize(returns)   # call 1 → last=w0
        result2 = overlay.optimize(returns)  # call 2 → proposed 0.60 ≥ 0.10 → trade

        pd.testing.assert_series_equal(result2, w1)

        # call 3: from w1, proposed turnover = 0.20 > 0.10 → trade
        result3 = overlay.optimize(returns)
        pd.testing.assert_series_equal(result3, w2)

    def test_new_asset_in_universe_counts_in_turnover(self):
        """Asset entering universe (absent in last_weights) raises turnover via reindex fill=0."""
        from jeanclaude.portfolio.optimizer.no_trade_band import NoTradeBandOverlay

        # First call: only A and B
        w0 = _make_series({"A": 0.5, "B": 0.5})
        # Second call: A, B, C — C is new, its contribution to turnover = |0.3 - 0.0| = 0.3
        w1 = _make_series({"A": 0.4, "B": 0.3, "C": 0.3})
        mock = _make_mock_optimizer([w0, w1])
        overlay = NoTradeBandOverlay(mock, threshold=0.05)

        returns = pd.DataFrame({"A": [0.01], "B": [-0.01]})

        overlay.optimize(returns)  # establishes last=w0
        result = overlay.optimize(returns)  # turnover ≥ 0.05 because of new asset C

        # Should trade because new asset C pushes total turnover high
        pd.testing.assert_series_equal(result, w1)

    def test_threshold_zero_always_trades(self):
        """threshold=0 means any proposed change passes through (0 < 0 is always False)."""
        from jeanclaude.portfolio.optimizer.no_trade_band import NoTradeBandOverlay

        w0 = _make_series({"A": 1.0})
        w1 = _make_series({"A": 1.0})  # identical weights → turnover = 0.0
        w2 = _make_series({"A": 0.9, "B": 0.1})
        mock = _make_mock_optimizer([w0, w1, w2])
        overlay = NoTradeBandOverlay(mock, threshold=0.0)

        returns = pd.DataFrame({"A": [0.01]})

        overlay.optimize(returns)            # call 1
        result2 = overlay.optimize(returns)  # call 2: turnover=0.0, threshold=0.0 → 0.0 < 0.0 is False → trade
        pd.testing.assert_series_equal(result2, w1)

        result3 = overlay.optimize(returns)  # call 3: different weights → trade
        pd.testing.assert_series_equal(result3, w2)
