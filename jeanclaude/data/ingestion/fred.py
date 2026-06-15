"""
FRED (Federal Reserve Economic Data) connector.

Uses the `fredapi` Python library. API key is read from the
``FRED_API_KEY`` environment variable or passed explicitly.

A free API key is available at: https://fred.stlouisfed.org/docs/api/api_key.html

Usage::

    from jeanclaude.data.ingestion import FREDSource

    src = FREDSource()   # reads FRED_API_KEY from env
    macro = src.get_macro(
        series=["FEDFUNDS", "T10Y2Y", "BAMLH0A0HYM2"],
        start="2010-01-01",
        end="2024-12-31",
    )

Common series IDs
-----------------
Rates & spreads:
    FEDFUNDS      — Fed Funds Rate
    DGS10         — 10-Year Treasury Yield
    DGS2          — 2-Year Treasury Yield
    T10Y2Y        — 10Y–2Y Treasury Spread (yield curve slope)
    BAMLH0A0HYM2  — ICE BofA High Yield OAS (credit spread)
    TEDRATE       — TED Spread

Activity / inflation:
    CPIAUCSL      — CPI (YoY)
    CPILFESL      — Core CPI
    UNRATE        — Unemployment Rate
    GDPC1         — Real GDP (quarterly)
    INDPRO        — Industrial Production Index
    ISM_MAN_PMI   — ISM Manufacturing PMI (via FRED)

Risk / sentiment:
    VIXCLS        — CBOE VIX
    UMCSENT       — U Michigan Consumer Sentiment
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

from .base import DataSource

logger = logging.getLogger(__name__)


class FREDSource(DataSource):
    """FRED macro data connector.

    Parameters
    ----------
    api_key : str, optional
        FRED API key. Falls back to ``FRED_API_KEY`` environment variable.
    publication_lag_days : int, default 1
        Giorni di mercato tra la data dell'osservazione e la sua disponibilità reale.
        1 è conservativo per serie daily (VIXCLS, DGS10); le serie mensili
        richiederebbero lag maggiori — definirli per-serie quando verranno
        usate in backtest.
    """

    def __init__(self, api_key: Optional[str] = None, publication_lag_days: int = 1) -> None:
        self._api_key = api_key or os.environ.get("FRED_API_KEY")
        self._publication_lag_days = publication_lag_days

    @property
    def name(self) -> str:
        return "FRED"

    def is_available(self) -> bool:
        try:
            import fredapi  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_fred(self):
        try:
            from fredapi import Fred
        except ImportError as e:
            raise ImportError(
                "fredapi is not installed. Run: poetry add fredapi"
            ) from e

        if not self._api_key:
            logger.warning(
                "FRED_API_KEY not set — requests may be rate-limited. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        return Fred(api_key=self._api_key)

    def _fetch_series(self, sid: str, start: str, end: str) -> pd.Series:
        """Fetch a single FRED series.

        Parameters
        ----------
        sid : str
            FRED series ID.
        start, end : str
            Date range, ISO format.

        Returns
        -------
        pd.Series
            Raw series as returned by fredapi (index is DatetimeIndex).
        """
        fred = self._get_fred()
        return fred.get_series(sid, observation_start=start, observation_end=end)

    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str,
        field: str = "close",
    ) -> pd.DataFrame:
        """FRED does not serve price data — delegates to get_macro()."""
        return self.get_macro(tickers, start, end)

    def get_macro(
        self,
        series: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch one or more FRED series.

        Parameters
        ----------
        series : list[str]
            FRED series IDs (e.g. ``['FEDFUNDS', 'T10Y2Y', 'VIXCLS']``).
        start, end : str
            Date range, ISO format.

        Returns
        -------
        pd.DataFrame
            Daily (or original frequency) DataFrame, one column per series.
            Missing values forward-filled to business-day frequency.
            Observations shifted forward by publication_lag_days business days.

        Notes
        -----
        The publication lag applies before resampling/forward-fill: observations
        from date t become available starting at t+lag. This prevents look-ahead
        bias in backtests. Vintage revisions (ALFRED) are not handled here.
        """
        frames = {}

        for sid in series:
            try:
                s = self._fetch_series(sid, start, end)
                # Apply publication lag before resampling
                if self._publication_lag_days > 0:
                    s.index = s.index + pd.tseries.offsets.BDay(self._publication_lag_days)
                frames[sid] = s
                logger.debug("FRED: fetched %s (%d obs)", sid, len(s))
            except Exception as exc:
                logger.warning("FRED: failed to fetch '%s': %s", sid, exc)
                frames[sid] = pd.Series(dtype=float, name=sid)

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"

        # Resample to business days and forward-fill (FRED series have mixed freq)
        bday_idx = pd.date_range(start=start, end=end, freq="B")
        df = df.reindex(bday_idx).ffill()
        df.index.name = "date"

        return df
