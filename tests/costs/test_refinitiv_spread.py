"""Tests for RefinitivSource BID/ASK and get_spread()."""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from jeanclaude.data.ingestion.refinitiv import RefinitivSource


def _make_source() -> RefinitivSource:
    """Create a RefinitivSource instance without opening a session."""
    src = RefinitivSource.__new__(RefinitivSource)
    src._session = MagicMock()
    return src


def test_bid_ask_in_field_map():
    """Verify BID and ASK are in the price field map."""
    from jeanclaude.data.ingestion.refinitiv import _PRICE_FIELD_MAP
    assert "bid" in _PRICE_FIELD_MAP
    assert "ask" in _PRICE_FIELD_MAP
    assert _PRICE_FIELD_MAP["bid"] == "BID"
    assert _PRICE_FIELD_MAP["ask"] == "ASK"


def test_get_spread_shape():
    """Test that get_spread returns correct shape and values."""
    src = _make_source()
    tickers = ["AAPL.O", "MSFT.O"]
    dates = pd.date_range("2026-01-02", periods=5, freq="B")

    bid_df = pd.DataFrame(
        {"AAPL.O": [149.9, 150.0, 150.1, 150.2, 150.3],
         "MSFT.O": [299.8, 300.0, 300.2, 300.4, 300.6]},
        index=dates,
    )
    ask_df = pd.DataFrame(
        {"AAPL.O": [150.1, 150.2, 150.3, 150.4, 150.5],
         "MSFT.O": [300.2, 300.4, 300.6, 300.8, 301.0]},
        index=dates,
    )

    with patch.object(src, "get_prices", side_effect=[bid_df, ask_df]):
        spread = src.get_spread(tickers, "2026-01-02", "2026-01-08")

    assert spread.shape == (5, 2)
    assert list(spread.columns) == tickers
    # spread = (ask - bid) / mid; all values should be small positive numbers
    assert (spread > 0).all().all()
    assert (spread < 0.01).all().all()  # <1% spread for liquid equities


def test_get_spread_handles_zero_mid():
    """Mid=0 rows should be dropped by dropna(how='all'); normal rows preserved."""
    src = _make_source()
    tickers = ["A.O"]
    dates = pd.date_range("2026-01-02", periods=2, freq="B")
    bid_df = pd.DataFrame({"A.O": [0.0, 10.0]}, index=dates)
    ask_df = pd.DataFrame({"A.O": [0.0, 10.2]}, index=dates)

    with patch.object(src, "get_prices", side_effect=[bid_df, ask_df]):
        spread = src.get_spread(tickers, "2026-01-02", "2026-01-04")

    # First row (2026-01-02) has mid=0 → spread=NaN → dropped by dropna(how="all")
    # Result has only second row (2026-01-05) with valid spread value
    assert spread.shape[0] == 1
    assert spread.iloc[0, 0] > 0        # remaining row has valid spread
