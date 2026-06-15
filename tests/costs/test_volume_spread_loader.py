import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from jeanclaude.costs.data import VolumeSpreadLoader


def _mock_source(vol_df: pd.DataFrame, spread_df: pd.DataFrame):
    src = MagicMock()
    src.get_prices.return_value = vol_df
    src.get_spread.return_value = spread_df
    return src


def _make_vol_df(tickers, n_rows=30):
    return pd.DataFrame(
        {t: [1_000_000.0] * n_rows for t in tickers},
        index=pd.date_range("2026-01-02", periods=n_rows, freq="B"),
    )


def _make_spread_df(tickers, n_rows=5):
    return pd.DataFrame(
        {t: [0.001] * n_rows for t in tickers},
        index=pd.date_range("2026-04-01", periods=n_rows, freq="B"),
    )


def test_get_adv_returns_series():
    tickers = ["AAPL.O", "MSFT.O"]
    vol_df = _make_vol_df(tickers)
    src = _mock_source(vol_df, _make_spread_df(tickers))
    loader = VolumeSpreadLoader(source=src)

    adv = loader.get_adv(tickers, "2026-01-02", "2026-02-10", window=20)

    assert isinstance(adv, pd.Series)
    assert set(adv.index) == set(tickers)
    assert (adv == 1_000_000.0).all()


def test_get_adv_uses_last_n_rows():
    tickers = ["A.O"]
    # Last 5 rows have volume=500, rest=1000
    dates = pd.date_range("2026-01-02", periods=25, freq="B")
    volumes = [1000.0] * 20 + [500.0] * 5
    vol_df = pd.DataFrame({"A.O": volumes}, index=dates)
    src = _mock_source(vol_df, _make_spread_df(tickers))
    loader = VolumeSpreadLoader(source=src)

    adv = loader.get_adv(tickers, "2026-01-02", "2026-02-06", window=5)

    assert adv["A.O"] == pytest.approx(500.0)


def test_get_spread_returns_series():
    tickers = ["AAPL.O", "TLT.O"]
    spread_df = _make_spread_df(tickers)
    src = _mock_source(_make_vol_df(tickers), spread_df)
    loader = VolumeSpreadLoader(source=src)

    spread = loader.get_spread(tickers, "2026-01-02", "2026-04-10")

    assert isinstance(spread, pd.Series)
    assert set(spread.index) == set(tickers)
    assert (spread == 0.001).all()


def test_get_adv_uses_cache(tmp_path):
    from jeanclaude.data.storage.parquet_store import ParquetStore

    tickers = ["X.O"]
    vol_df = _make_vol_df(tickers)
    src = _mock_source(vol_df, _make_spread_df(tickers))
    store = ParquetStore(str(tmp_path))
    loader = VolumeSpreadLoader(source=src, store=store)

    # First call — fetches from source
    loader.get_adv(tickers, "2026-01-02", "2026-02-10", window=20)
    assert src.get_prices.call_count == 1

    # Second call — should use cache, not re-fetch
    loader.get_adv(tickers, "2026-01-02", "2026-02-10", window=20)
    assert src.get_prices.call_count == 1  # still 1
