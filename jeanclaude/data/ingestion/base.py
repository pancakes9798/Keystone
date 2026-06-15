"""
Abstract base class for all data source connectors.

Every connector must implement this interface so that the DataLoader
can swap sources transparently (Refinitiv → Yahoo fallback, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class DataSource(ABC):
    """Protocol for all market data connectors."""

    @abstractmethod
    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str,
        field: str = "close",
    ) -> pd.DataFrame:
        """Fetch daily adjusted close prices.

        Parameters
        ----------
        tickers : list[str]
            List of instrument identifiers (format depends on source).
        start, end : str
            Date range, ISO format (e.g. '2020-01-01').
        field : str
            Price field to retrieve ('close', 'open', 'high', 'low', 'volume').

        Returns
        -------
        pd.DataFrame
            Date-indexed DataFrame, one column per ticker.
            Index: pd.DatetimeIndex (business days).
        """

    @abstractmethod
    def get_macro(
        self,
        series: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch macro time series (rates, spreads, inflation, etc.).

        Parameters
        ----------
        series : list[str]
            Source-specific identifiers for macro indicators.
        start, end : str
            Date range, ISO format.

        Returns
        -------
        pd.DataFrame
            Date-indexed DataFrame, one column per series.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this data source."""

    def is_available(self) -> bool:
        """Return True if this source can be used (credentials, library installed, etc.)."""
        return True
