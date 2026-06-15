"""
Yahoo Finance fallback connector.

Uses `yfinance`. No API key required. Use only as fallback when
Refinitiv is not available, or for quick prototyping.

Usage::

    from jeanclaude.data.ingestion import YahooSource

    src = YahooSource()
    prices = src.get_prices(
        tickers=["SPY", "TLT", "GLD", "EEM"],
        start="2020-01-01",
        end="2024-12-31",
    )
"""

from __future__ import annotations

import logging

import pandas as pd

from .base import DataSource

logger = logging.getLogger(__name__)

# Yahoo Finance macro proxies (ticker → FRED-style label)
# These are tradable proxies, not the underlying macro series.
# For actual macro data (rates, spreads), use FREDSource.
YAHOO_MACRO_PROXIES: dict[str, str] = {
    "^VIX":   "VIX",
    "^TNX":   "US10Y_yield",
    "^IRX":   "US3M_yield",
    "^TYX":   "US30Y_yield",
    "DX-Y.NYB": "DXY",
}


class YahooSource(DataSource):
    """Yahoo Finance connector — fallback / prototype only.

    Parameters
    ----------
    auto_adjust : bool
        Apply split and dividend adjustments (default True).
    """

    def __init__(self, auto_adjust: bool = True) -> None:
        self._auto_adjust = auto_adjust

    @property
    def name(self) -> str:
        return "Yahoo Finance"

    def is_available(self) -> bool:
        try:
            import yfinance  # noqa: F401
            return True
        except ImportError:
            return False

    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str,
        field: str = "close",
    ) -> pd.DataFrame:
        """Download OHLCV data from Yahoo Finance.

        Parameters
        ----------
        tickers : list[str]
            Yahoo Finance ticker symbols (e.g. ``'SPY'``, ``'TLT'``).
        start, end : str
            ISO date strings.
        field : str
            One of ``'close'``, ``'open'``, ``'high'``, ``'low'``, ``'volume'``.

        Returns
        -------
        pd.DataFrame
            Date-indexed DataFrame, one column per ticker.
        """
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError("yfinance is not installed. Run: poetry add yfinance") from e

        yf_field = field.capitalize()
        if self._auto_adjust and field == "close":
            yf_field = "Close"  # yfinance uses 'Close' post-adjustment

        logger.info(
            "Yahoo: fetching %s for %d tickers (%s → %s)",
            field, len(tickers), start, end,
        )

        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=self._auto_adjust,
            progress=False,
            threads=True,
        )

        if raw.empty:
            logger.warning("Yahoo Finance returned empty DataFrame")
            return pd.DataFrame(index=pd.DatetimeIndex([]), columns=tickers)

        # yfinance multi-ticker: columns are MultiIndex (field, ticker)
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw[yf_field]
        else:
            # Single ticker: flat columns
            df = raw[[yf_field]].rename(columns={yf_field: tickers[0]})

        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.reindex(columns=tickers)

        return df

    def get_macro(
        self,
        series: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch tradable macro proxies from Yahoo (^VIX, ^TNX, DXY, etc.).

        Note: for actual FRED macro data (spreads, inflation, GDP),
        use FREDSource — Yahoo only provides tradable indices here.
        """
        return self.get_prices(series, start=start, end=end, field="close")
