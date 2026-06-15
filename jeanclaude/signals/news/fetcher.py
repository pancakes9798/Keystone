"""RefinitivNewsSource — fetches financial headlines from LSEG/Refinitiv news API."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class RefinitivNewsSource:
    """Scarica headlines di news finanziarie da LSEG/Refinitiv per asset class.

    Usa ``lseg.data.news.Headlines.Definition`` per ogni query e aggrega
    i risultati in un DataFrame con schema uniforme.

    Il modulo ``lseg.data`` è importato lazily: non serve installarlo per
    usare il resto del sistema (es. in test o ambienti senza licenza).

    Parameters
    ----------
    app_key : str, optional
        LSEG app key. Falls back to ``LSEG_APP_KEY`` env var.
    session_type : str
        ``"desktop"`` (default) o ``"platform"``.
    """

    def __init__(
        self,
        app_key: str | None = None,
        session_type: str = "desktop",
    ) -> None:
        import os
        self._app_key = app_key or os.environ.get("LSEG_APP_KEY")
        self._session_type = session_type
        self._lseg_data: Any = None  # lazy-loaded; injectable in tests

    def get_headlines(
        self,
        asset_class_queries: dict[str, list[str]],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Scarica headlines per ogni asset class.

        Parameters
        ----------
        asset_class_queries : dict[str, list[str]]
            Mapping ``asset_class → [query_string, ...]``.
        start, end : str
            ISO date strings ``"YYYY-MM-DD"``.

        Returns
        -------
        pd.DataFrame
            DatetimeIndex (name="date") + colonne:
            ``asset_class | headline | story_id``
        """
        ld = self._ensure_lseg()
        frames: list[pd.DataFrame] = []

        for asset_class, queries in asset_class_queries.items():
            for query in queries:
                try:
                    raw = ld.news.Headlines.Definition(
                        query=query,
                        date_from=start,
                        date_to=end,
                    ).get_data()

                    if raw is None or (hasattr(raw, "empty") and raw.empty):
                        continue

                    df = self._normalize(raw, asset_class)
                    if not df.empty:
                        frames.append(df)

                except Exception as exc:
                    logger.warning(
                        "News fetch failed for query=%r asset_class=%s: %s",
                        query, asset_class, exc,
                    )

        if not frames:
            return pd.DataFrame(
                columns=["asset_class", "headline", "story_id"],
                index=pd.DatetimeIndex([], name="date"),
            )

        result = pd.concat(frames)
        result = result[~result.index.duplicated(keep="first")]
        result = result.sort_index()
        return result

    def get_ticker_headlines(
        self,
        ticker_queries: dict[str, list[str]],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Scarica headlines per ogni singolo ticker (RIC).

        Parameters
        ----------
        ticker_queries : dict[str, list[str]]
            Mapping ``RIC → [query_string, ...]``.
        start, end : str
            ISO date strings ``"YYYY-MM-DD"``.

        Returns
        -------
        pd.DataFrame
            DatetimeIndex (name="date") + colonne:
            ``ticker | headline | story_id``
        """
        ld = self._ensure_lseg()
        frames: list[pd.DataFrame] = []

        for ticker, queries in ticker_queries.items():
            for query in queries:
                try:
                    raw = ld.news.Headlines.Definition(
                        query=query,
                        date_from=start,
                        date_to=end,
                    ).get_data()

                    if raw is None or (hasattr(raw, "empty") and raw.empty):
                        continue

                    df = self._normalize_ticker(raw, ticker)
                    if not df.empty:
                        frames.append(df)

                except Exception as exc:
                    logger.warning(
                        "Ticker news fetch failed for query=%r ticker=%s: %s",
                        query, ticker, exc,
                    )

        if not frames:
            return pd.DataFrame(
                columns=["ticker", "headline", "story_id"],
                index=pd.DatetimeIndex([], name="date"),
            )

        result = pd.concat(frames)
        result = result[~result.index.duplicated(keep="first")]
        result = result.sort_index()
        return result

    @staticmethod
    def _normalize_ticker(raw: Any, ticker: str) -> pd.DataFrame:
        """Normalizza output LSEG news per ticker → schema uniforme."""
        if isinstance(raw, pd.DataFrame):
            df = raw.copy()
        else:
            try:
                df = pd.DataFrame(raw)
            except Exception:
                return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        rename_map = {
            "storyId": "story_id", "story_id": "story_id",
            "headLine": "headline", "headline": "headline", "Headline": "headline",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        if "headline" not in df.columns:
            df["headline"] = ""
        if "story_id" not in df.columns:
            df["story_id"] = ""

        df["ticker"] = ticker

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        df.index.name = "date"
        df = df.dropna(subset=["headline"])

        return df[["ticker", "headline", "story_id"]]

    def _ensure_lseg(self) -> Any:
        """Return lseg.data module, importing lazily."""
        if self._lseg_data is not None:
            return self._lseg_data

        try:
            import lseg.data as ld  # noqa: PLC0415
        except (ImportError, TypeError) as exc:
            raise ImportError(
                "lseg-data is not installed. Run: poetry add lseg-data --optional"
            ) from exc

        self._lseg_data = ld
        return ld

    @staticmethod
    def _normalize(raw: Any, asset_class: str) -> pd.DataFrame:
        """Normalizza output LSEG news → schema uniforme."""
        if isinstance(raw, pd.DataFrame):
            df = raw.copy()
        else:
            try:
                df = pd.DataFrame(raw)
            except Exception:
                return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        rename_map = {
            "storyId": "story_id",
            "story_id": "story_id",
            "headLine": "headline",
            "headline": "headline",
            "Headline": "headline",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        if "headline" not in df.columns:
            df["headline"] = ""
        if "story_id" not in df.columns:
            df["story_id"] = ""

        df["asset_class"] = asset_class

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        df.index.name = "date"
        df = df.dropna(subset=["headline"])

        return df[["asset_class", "headline", "story_id"]]
