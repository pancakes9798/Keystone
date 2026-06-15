from __future__ import annotations

import logging

import pandas as pd

from jeanclaude.data.ingestion.refinitiv import RefinitivSource
from jeanclaude.data.storage.parquet_store import ParquetStore

logger = logging.getLogger(__name__)


class VolumeSpreadLoader:
    """Loads ADV and bid-ask spread from Refinitiv with optional Parquet caching.

    Parameters
    ----------
    source : RefinitivSource
    store  : ParquetStore, optional
        When provided, results are cached under 'volume' and 'spread' namespaces.
    """

    def __init__(
        self,
        source: RefinitivSource,
        store: ParquetStore | None = None,
    ) -> None:
        self._source = source
        self._store = store

    def get_adv(
        self,
        tickers: list[str],
        start: str,
        end: str,
        window: int = 20,
    ) -> pd.Series:
        """Return rolling mean ADV (last ``window`` days) per ticker as of ``end``.

        Returns
        -------
        pd.Series
            Index = tickers, values = mean daily volume in currency units.
        """
        cache_key = "_".join(sorted(tickers)) + "_volume"

        vol_data = self._load_or_fetch(
            namespace="volume",
            cache_key=cache_key,
            start=start,
            end=end,
            fetch_fn=lambda: self._source.get_prices(
                tickers, start=start, end=end, field="volume"
            ),
        )
        return vol_data.tail(window).mean()

    def get_spread(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> pd.Series:
        """Return most recent bid-ask spread per ticker.

        Returns
        -------
        pd.Series
            Index = tickers, values = fractional spread (e.g. 0.001 = 10 bps).
        """
        cache_key = "_".join(sorted(tickers)) + "_spread"

        spread_data = self._load_or_fetch(
            namespace="spread",
            cache_key=cache_key,
            start=start,
            end=end,
            fetch_fn=lambda: self._source.get_spread(
                tickers, start=start, end=end
            ),
        )
        return spread_data.iloc[-1]

    def _load_or_fetch(
        self,
        namespace: str,
        cache_key: str,
        start: str,
        end: str,
        fetch_fn,
    ) -> pd.DataFrame:
        if self._store:
            cached = self._store.load(namespace, cache_key, start, end)
            if cached is not None:
                logger.debug("Cache hit: %s/%s", namespace, cache_key)
                return cached

        data = fetch_fn()
        if self._store:
            self._store.save(namespace, cache_key, start, end, data)
        return data
