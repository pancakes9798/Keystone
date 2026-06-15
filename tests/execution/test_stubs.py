"""Tests for execution stubs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jeanclaude.execution.paper.stubs import (
    NoopRegimeDetector,
    NoopRiskFilter,
    NoopReportBuilder,
    YahooExtendedDataLoader,
)


# ── NoopRegimeDetector ───────────────────────────────────────────────
def test_noop_regime_detector_current_regime_returns_label_with_name():
    det = NoopRegimeDetector()
    label, arr = det.current_regime(pd.DataFrame())
    assert hasattr(label, "name")
    assert isinstance(label.name, str)


def test_noop_regime_detector_fit_returns_none():
    det = NoopRegimeDetector()
    result = det.fit(pd.DataFrame())
    assert result is None


# ── NoopRiskFilter ───────────────────────────────────────────────────
def test_noop_risk_filter_returns_weights_unchanged():
    rf = NoopRiskFilter()
    w = pd.Series({"A": 0.6, "B": 0.4})
    result = rf.apply(w, pd.DataFrame())
    pd.testing.assert_series_equal(result, w)


# ── NoopReportBuilder ────────────────────────────────────────────────
def test_noop_report_builder_returns_string():
    rb = NoopReportBuilder()
    result = rb.build(as_of=pd.Timestamp("2026-05-01"), regime="EXPANSION")
    assert isinstance(result, str)


# ── YahooExtendedDataLoader ──────────────────────────────────────────
def test_yahoo_extended_data_loader_get_macro_returns_empty_df():
    class _FakeLoader:
        def get_prices(self, tickers, start, end):
            return pd.DataFrame()
        def get_returns(self, prices, method="log"):
            return pd.DataFrame()

    loader = YahooExtendedDataLoader(
        loader=_FakeLoader(),
        supplements={},
        start="2024-01-01",
        end="2024-12-31",
    )
    macro = loader.get_macro(series=["VIXCLS"], start="2024-01-01", end="2024-12-31")
    assert isinstance(macro, pd.DataFrame)
    assert macro.empty


def test_yahoo_extended_data_loader_delegates_get_returns():
    expected = pd.DataFrame({"A": [0.01, -0.02]})

    class _FakeLoader:
        def get_prices(self, tickers, start, end):
            return pd.DataFrame()
        def get_returns(self, prices, method="log"):
            return expected

    loader = YahooExtendedDataLoader(
        loader=_FakeLoader(),
        supplements={},
        start="2024-01-01",
        end="2024-12-31",
    )
    result = loader.get_returns(pd.DataFrame())
    pd.testing.assert_frame_equal(result, expected)
