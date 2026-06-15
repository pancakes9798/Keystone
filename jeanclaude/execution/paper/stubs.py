"""Lightweight stubs for DailyRunner dependencies not used by Sprint E."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class _DummyRegimeLabel:
    """Minimal stand-in for RegimeLabel enum values."""
    name: str = "EXPANSION"


class NoopRegimeDetector:
    """Regime detector stub for strategies with no macro dependency."""

    def fit(self, macro: pd.DataFrame) -> None:
        pass

    def current_regime(self, macro: pd.DataFrame) -> tuple[_DummyRegimeLabel, Any]:
        return _DummyRegimeLabel(), np.array([])


class NoopRiskFilter:
    """Risk filter stub — passes weights through unchanged."""

    def apply(self, weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
        return weights


class NoopReportBuilder:
    """Report builder stub — returns an empty string."""

    def build(self, as_of: pd.Timestamp, regime: str | None = None) -> str:
        return ""


class YahooExtendedDataLoader:
    """DataLoader wrapper that applies Yahoo Finance fallback after Refinitiv fetch.

    Also overrides ``get_macro`` to return an empty DataFrame, so strategies
    without a macro dependency do not require FRED credentials.

    Parameters
    ----------
    loader :
        A real DataLoader instance (must implement get_prices, get_returns).
    supplements : dict[str, str]
        Mapping of Refinitiv RIC -> Yahoo Finance ticker for Yahoo fallback.
    start : str
        ISO date string — earliest date to request from Yahoo.
    end : str
        ISO date string — latest date to request from Yahoo.
    """

    def __init__(
        self,
        loader: Any,
        supplements: dict[str, str],
        start: str,
        end: str,
    ) -> None:
        self._loader = loader
        self._supplements = supplements
        self._start = start
        self._end = end

    def get_prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        prices = self._loader.get_prices(tickers, start=start, end=end)
        return self._extend_with_yahoo(prices, start, end)

    def get_returns(self, prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
        return self._loader.get_returns(prices, method=method)

    def get_macro(self, series: list[str], start: str, end: str) -> pd.DataFrame:
        return pd.DataFrame()

    def _extend_with_yahoo(
        self, prices: pd.DataFrame, start: str, end: str
    ) -> pd.DataFrame:
        """Prepend Yahoo Finance history where Refinitiv data is missing or short."""
        prices = prices.copy()
        for ric, ticker in self._supplements.items():
            if ric not in prices.columns:
                continue
            try:
                raw = yf.download(
                    ticker, start=start, end=end, progress=False, auto_adjust=True
                )
                if raw.empty:
                    continue
                yh = raw["Close"]
                if isinstance(yh, pd.DataFrame):
                    logger.warning("Unexpected multi-column Close for %s, skipping", ticker)
                    continue
                yh = yh.squeeze().rename(ric)
                yh.index = pd.to_datetime(yh.index).tz_localize(None)
                ref_series = prices[ric].dropna()
                if ref_series.empty:
                    prices[ric] = yh
                    logger.info("Yahoo full-replace %s -> %s", ticker, ric)
                else:
                    ref_first = ref_series.index[0]
                    yh_before = yh.loc[:ref_first].dropna().iloc[:-1]
                    if yh_before.empty:
                        continue
                    yh_at_join = yh.reindex([ref_first]).dropna()
                    if yh_at_join.empty:
                        continue
                    scale = float(ref_series.iloc[0]) / float(yh_at_join.iloc[0])
                    prices[ric] = pd.concat([yh_before * scale, ref_series]).sort_index()
                    logger.info(
                        "Yahoo extended %s -> %s: prepended to %s",
                        ticker, ric, ref_first.date(),
                    )
            except Exception as exc:
                logger.warning(
                    "Yahoo supplement failed for %s (%s): %s", ric, ticker, exc
                )
        return prices
