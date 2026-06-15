"""
Tests for the data module.

All tests use synthetic / local data — no external API calls.
The Refinitiv connector is tested only for its interface contract
(mocked); actual Refinitiv integration requires lseg-data + credentials.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from jeanclaude.data import (
    DataConfig,
    DataLoader,
    FREDSource,
    ParquetStore,
    RefinitivSource,
    YahooSource,
    to_log_returns,
    to_simple_returns,
    resample_returns,
    annualized_stats,
    build_state_variables,
    align_to_prices,
)


# ======================================================================
# Fixtures
# ======================================================================

ASSETS = ["SPY", "TLT", "GLD"]
DATES = pd.date_range("2020-01-02", periods=252, freq="B")


@pytest.fixture
def price_df():
    """Synthetic price DataFrame."""
    np.random.seed(0)
    prices = 100 * np.exp(
        np.cumsum(np.random.randn(252, len(ASSETS)) * 0.01, axis=0)
    )
    df = pd.DataFrame(prices, index=DATES, columns=ASSETS)
    df.index.name = "date"
    return df


@pytest.fixture
def macro_df():
    """Synthetic macro DataFrame."""
    np.random.seed(1)
    return pd.DataFrame(
        {
            "VIX": np.abs(np.random.randn(252)) * 5 + 15,
            "T10Y2Y": np.random.randn(252) * 0.3 + 1.0,
            "credit_spread": np.abs(np.random.randn(252)) * 0.2 + 1.5,
        },
        index=DATES,
    )


@pytest.fixture
def tmp_store(tmp_path):
    return ParquetStore(tmp_path)


# ======================================================================
# Tests: transforms — returns
# ======================================================================


class TestReturns:

    def test_log_returns_shape(self, price_df):
        rets = to_log_returns(price_df)
        assert rets.shape == (251, 3)

    def test_log_returns_approx(self, price_df):
        """Log returns ≈ simple returns for small daily moves."""
        log_r = to_log_returns(price_df)
        simple_r = to_simple_returns(price_df)
        # For daily returns < 5%, they should be very close
        assert np.allclose(log_r.values, simple_r.values, atol=0.005)

    def test_simple_returns_no_nan(self, price_df):
        rets = to_simple_returns(price_df)
        assert not rets.isnull().any().any()

    def test_resample_weekly_compound(self, price_df):
        daily = to_simple_returns(price_df)
        weekly = resample_returns(daily, freq="W", method="compound")
        assert len(weekly) < len(daily)
        assert weekly.shape[1] == daily.shape[1]

    def test_resample_monthly_sum(self, price_df):
        log_r = to_log_returns(price_df)
        monthly = resample_returns(log_r, freq="ME", method="sum")
        assert monthly.shape[1] == log_r.shape[1]

    def test_annualized_stats(self, price_df):
        rets = to_simple_returns(price_df)
        stats = annualized_stats(rets)
        assert list(stats.columns) == ["ann_return", "ann_vol", "sharpe"]
        assert list(stats.index) == ASSETS
        assert (stats["ann_vol"] > 0).all()


# ======================================================================
# Tests: transforms — features
# ======================================================================


class TestFeatures:

    def test_build_state_variables_shape(self, macro_df):
        sv = build_state_variables(macro_df, normalize=False)
        assert sv.shape == macro_df.shape

    def test_build_state_variables_normalized(self, macro_df):
        sv = build_state_variables(macro_df, normalize=True)
        # Expanding normalisation (default): first 59 rows are NaN (insufficient
        # history for a stable median/IQR estimate — callers must dropna).
        # Rows from index 59 onward are fully normalised.
        assert sv.shape == macro_df.shape
        assert not sv.dropna().isnull().any().any()
        # Verify expanding window produces NaN leading rows and valid values after
        from jeanclaude.data.transform.features import _EXPANDING_MIN_OBS
        assert sv.iloc[:_EXPANDING_MIN_OBS - 1].isnull().all().all()
        assert not sv.iloc[_EXPANDING_MIN_OBS:].isnull().any().any()

    def test_build_state_variables_feature_selection(self, macro_df):
        sv = build_state_variables(macro_df, feature_cols=["VIX", "T10Y2Y"])
        assert list(sv.columns) == ["VIX", "T10Y2Y"]

    def test_build_state_variables_unknown_col(self, macro_df):
        with pytest.raises(ValueError, match="not found"):
            build_state_variables(macro_df, feature_cols=["NONEXISTENT"])

    def test_align_to_prices(self, macro_df, price_df):
        # Macro with gaps (monthly-ish)
        sparse_macro = macro_df.iloc[::21]  # every 21 days
        aligned = align_to_prices(sparse_macro, price_df.index)
        assert len(aligned) == len(price_df)
        # Forward-filled: no NaN except possibly at the start
        assert not aligned.iloc[21:].isnull().any().any()


# ======================================================================
# Tests: storage — ParquetStore
# ======================================================================


class TestParquetStore:

    def test_save_and_load_roundtrip(self, tmp_store, price_df):
        tmp_store.save("prices", "SPY_TLT_GLD", "2020-01-01", "2020-12-31", price_df)
        loaded = tmp_store.load("prices", "SPY_TLT_GLD", "2020-01-01", "2020-12-31")

        assert loaded is not None
        assert loaded.shape == price_df.shape
        pd.testing.assert_frame_equal(loaded, price_df, check_freq=False)

    def test_load_miss_returns_none(self, tmp_store):
        result = tmp_store.load("prices", "NONEXISTENT", "2020-01-01", "2020-12-31")
        assert result is None

    def test_exists(self, tmp_store, price_df):
        assert not tmp_store.exists("prices", "test_key", "2020-01-01", "2020-12-31")
        tmp_store.save("prices", "test_key", "2020-01-01", "2020-12-31", price_df)
        assert tmp_store.exists("prices", "test_key", "2020-01-01", "2020-12-31")

    def test_delete(self, tmp_store, price_df):
        tmp_store.save("prices", "del_key", "2020-01-01", "2020-12-31", price_df)
        assert tmp_store.exists("prices", "del_key", "2020-01-01", "2020-12-31")
        deleted = tmp_store.delete("prices", "del_key", "2020-01-01", "2020-12-31")
        assert deleted
        assert not tmp_store.exists("prices", "del_key", "2020-01-01", "2020-12-31")

    def test_long_key_hashed(self, tmp_store, price_df):
        """Keys longer than 60 chars are hashed without error."""
        long_key = "A_" * 50  # 100-char key
        tmp_store.save("prices", long_key, "2020-01-01", "2020-12-31", price_df)
        loaded = tmp_store.load("prices", long_key, "2020-01-01", "2020-12-31")
        assert loaded is not None

    def test_list_files(self, tmp_store, price_df):
        tmp_store.save("prices", "file1", "2020-01-01", "2020-12-31", price_df)
        tmp_store.save("prices", "file2", "2021-01-01", "2021-12-31", price_df)
        files = tmp_store.list_files("prices")
        assert len(files) == 2


# ======================================================================
# Tests: DataLoader — with mocked sources
# ======================================================================


class TestDataLoader:

    def _make_loader(self, tmp_path) -> DataLoader:
        config = DataConfig(
            cache_dir=str(tmp_path),
            price_source="yahoo",   # no Refinitiv in tests
            macro_source="fred",
            use_cache=True,
        )
        return DataLoader(config)

    def test_get_returns_log(self, price_df, tmp_path):
        loader = self._make_loader(tmp_path)
        rets = loader.get_returns(price_df, method="log")
        assert rets.shape == (251, 3)

    def test_get_returns_simple(self, price_df, tmp_path):
        loader = self._make_loader(tmp_path)
        rets = loader.get_returns(price_df, method="simple")
        assert rets.shape == (251, 3)

    def test_get_returns_invalid_method(self, price_df, tmp_path):
        loader = self._make_loader(tmp_path)
        with pytest.raises(ValueError):
            loader.get_returns(price_df, method="invalid")

    def test_cache_roundtrip_via_loader(self, price_df, tmp_path):
        """Prices saved and re-loaded via DataLoader cache."""
        loader = self._make_loader(tmp_path)

        # Mock Yahoo to return our synthetic data
        loader._yahoo.get_prices = MagicMock(return_value=price_df)

        df1 = loader.get_prices(ASSETS, "2020-01-01", "2020-12-31")
        df2 = loader.get_prices(ASSETS, "2020-01-01", "2020-12-31")

        # Second call should hit cache, not call the source again
        assert loader._yahoo.get_prices.call_count == 1
        pd.testing.assert_frame_equal(df1, df2, check_freq=False)

    def test_refinitiv_not_available_without_lseg(self):
        """lseg-data is optional — likely not installed in CI."""
        src = RefinitivSource()
        result = src.is_available()
        assert isinstance(result, bool)
