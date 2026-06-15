"""
DataLoader — unified entry point for market data.

Orchestrates source selection (Refinitiv → Yahoo fallback),
local Parquet caching, and returns transformation.

Usage::

    from jeanclaude.data import DataLoader

    loader = DataLoader(cache_dir="~/jeanclaude_data")

    # Price data
    prices = loader.get_prices(
        tickers=["SPY", "TLT", "GLD", "EEM", "EFA"],
        start="2015-01-01",
        end="2024-12-31",
    )

    # Macro state variables (FRED)
    macro = loader.get_macro(
        series=["VIXCLS", "TEDRATE", "T10Y2Y", "BAMLH0A0HYM2"],
        start="2015-01-01",
        end="2024-12-31",
    )

    # Daily log returns
    returns = loader.get_returns(prices)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .ingestion import DataSource, FREDSource, RefinitivSource, YahooSource
from .storage import ParquetStore
from .transform import to_log_returns, to_simple_returns

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for DataLoader.

    Parameters
    ----------
    cache_dir : str
        Root directory for Parquet cache. Set to '' to disable caching.
    price_source : str
        Primary source for price data: ``'refinitiv'`` or ``'yahoo'``.
    macro_source : str
        Primary source for macro data: ``'fred'`` or ``'refinitiv'``.
    refinitiv_session : str
        LSEG session type:
        - ``'desktop'``  — Eikon/Workspace app must be running locally.
        - ``'platform'`` — direct API, no desktop app needed.
    refinitiv_app_key : str, optional
        LSEG app key. Falls back to ``LSEG_APP_KEY`` env var.
    refinitiv_username : str, optional
        LSEG username (platform session only). Falls back to ``LSEG_USERNAME`` env var.
    refinitiv_password : str, optional
        LSEG password (platform session only). Falls back to ``LSEG_PASSWORD`` env var.
    fred_api_key : str, optional
        FRED API key (if not in FRED_API_KEY env var).
    use_cache : bool
        Whether to use local Parquet cache (default True).
    price_adjustments : tuple[str, ...]
        CORAX adjustments forwarded to RefinitivSource.get_prices.
        Default: ``("CCH", "CRE", "RTS", "RPO")`` — corporate cash events,
        capital reorganizations, rights issues, and reversal-of-price-adjustments.
        Set to ``()`` to fetch unadjusted (raw) prices.
        Note: Yahoo Finance always uses auto_adjust=True (splits + dividends);
        this setting only affects the Refinitiv data path.
    """
    cache_dir: str = "~/jeanclaude_data"
    price_source: str = "refinitiv"
    macro_source: str = "fred"
    refinitiv_session: str = "desktop"
    refinitiv_app_key: Optional[str] = None
    refinitiv_username: Optional[str] = None
    refinitiv_password: Optional[str] = None
    fred_api_key: Optional[str] = None
    use_cache: bool = True
    price_adjustments: tuple = ("CCH", "CRE", "RTS", "RPO")


class DataLoader:
    """Unified market data access layer.

    Automatically falls back to Yahoo Finance if Refinitiv is not available.

    Parameters
    ----------
    config : DataConfig, optional
        Configuration. Defaults to ``DataConfig()`` (Refinitiv primary, FRED macro).
    """

    def __init__(self, config: Optional[DataConfig] = None) -> None:
        self.config = config or DataConfig()

        # Instantiate sources
        self._refinitiv = RefinitivSource(
            app_key=self.config.refinitiv_app_key,
            session_type=self.config.refinitiv_session,
            username=self.config.refinitiv_username,
            password=self.config.refinitiv_password,
        )
        self._yahoo = YahooSource()
        self._fred = FREDSource(api_key=self.config.fred_api_key)

        # Cache
        self._store: Optional[ParquetStore] = None
        if self.config.use_cache and self.config.cache_dir:
            self._store = ParquetStore(self.config.cache_dir)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str,
        field: str = "close",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch adjusted close prices with caching and automatic fallback.

        Tries Refinitiv first; falls back to Yahoo Finance if unavailable.

        Parameters
        ----------
        tickers : list[str]
            - Refinitiv RIC codes when using Refinitiv (e.g. ``'SPY.P'``)
            - Yahoo symbols when using Yahoo fallback (e.g. ``'SPY'``)
        start, end : str
            Date range, ISO format.
        field : str
            Price field (``'close'``, ``'open'``, ``'high'``, ``'low'``, ``'volume'``).
        force_refresh : bool
            Bypass cache and re-fetch from source.

        Returns
        -------
        pd.DataFrame
            Date-indexed prices, one column per ticker.
        """
        # Pick source first, before cache key creation — ensures cache key
        # reflects the actual source that will be used.
        source = self._pick_price_source()
        cache_key = self._price_cache_key(tickers, field, source.name)

        if self._store and not force_refresh:
            cached = self._store.load("prices", cache_key, start, end)
            if cached is not None:
                return cached

        logger.info("Fetching prices via %s", source.name)

        # Propagate adjustments only to Refinitiv — Yahoo uses auto_adjust=True
        # internally and its get_prices signature does not accept adjustments.
        if isinstance(source, RefinitivSource) and self.config.price_adjustments:
            df = source.get_prices(
                tickers, start=start, end=end, field=field,
                adjustments=self.config.price_adjustments,
            )
        else:
            df = source.get_prices(tickers, start=start, end=end, field=field)

        if self._store:
            self._store.save("prices", cache_key, start, end, df)

        return df

    def get_macro(
        self,
        series: list[str],
        start: str,
        end: str,
        source: str = "auto",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch macro time series.

        Parameters
        ----------
        series : list[str]
            - FRED series IDs when using FRED (e.g. ``'VIXCLS'``, ``'T10Y2Y'``)
            - Refinitiv RIC codes when using Refinitiv
        start, end : str
            Date range, ISO format.
        source : str
            ``'fred'``, ``'refinitiv'``, or ``'auto'`` (uses config).
        force_refresh : bool
            Bypass cache.

        Returns
        -------
        pd.DataFrame
            Date-indexed macro DataFrame, one column per series.
        """
        cache_key = f"{'_'.join(sorted(series))}"
        src_name = source if source != "auto" else self.config.macro_source

        if self._store and not force_refresh:
            cached = self._store.load(f"macro_{src_name}", cache_key, start, end)
            if cached is not None:
                return cached

        data_source = self._pick_macro_source(src_name)
        logger.info("Fetching macro via %s", data_source.name)
        df = data_source.get_macro(series, start=start, end=end)

        if self._store:
            self._store.save(f"macro_{src_name}", cache_key, start, end, df)

        return df

    def get_returns(
        self,
        prices: pd.DataFrame,
        method: str = "log",
    ) -> pd.DataFrame:
        """Compute returns from a price DataFrame.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices.
        method : str
            ``'log'`` or ``'simple'``.

        Returns
        -------
        pd.DataFrame
            Daily returns.
        """
        if method == "log":
            return to_log_returns(prices)
        elif method == "simple":
            return to_simple_returns(prices)
        else:
            raise ValueError(f"method must be 'log' or 'simple', got '{method}'")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _price_cache_key(self, tickers: list[str], field: str, source_name: str) -> str:
        """Build a cache key that is sensitive to ticker list, field, source, and adjustments.

        The source name ensures that Yahoo-fallback data is never served as Refinitiv-adjusted
        data, and vice-versa. The adjustments suffix ensures that cached raw prices (adj-raw)
        are never reused when the loader is configured with CORAX adjustments, and vice-versa.

        Format: ``src-<source>_<sorted-tickers>_<field>_adj-<CCH-CRE-RTS-RPO>`` or
        ``src-<source>_<sorted-tickers>_<field>_adj-raw``.

        Parameters
        ----------
        tickers : list[str]
            Ticker list.
        field : str
            Price field (close, open, etc.).
        source_name : str
            Name of the data source (e.g., 'Refinitiv' or 'Yahoo Finance').
        """
        adj = self.config.price_adjustments
        adj_suffix = f"adj-{'-'.join(sorted(adj))}" if adj else "adj-raw"
        source_tag = f"src-{source_name.lower().replace(' ', '_')}"
        return f"{source_tag}_{'_'.join(sorted(tickers))}_{field}_{adj_suffix}"

    def _pick_price_source(self) -> DataSource:
        if self.config.price_source == "refinitiv" and self._refinitiv.is_available():
            return self._refinitiv
        if not self._refinitiv.is_available() and self.config.price_source == "refinitiv":
            logger.warning(
                "Refinitiv (lseg-data) not available — falling back to Yahoo Finance"
            )
        return self._yahoo

    def _pick_macro_source(self, name: str) -> DataSource:
        if name == "refinitiv" and self._refinitiv.is_available():
            return self._refinitiv
        if name == "fred":
            return self._fred
        # Default fallback
        return self._fred
