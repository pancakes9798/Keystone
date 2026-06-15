"""
Refinitiv / LSEG Data connector.

Uses the `lseg-data` Python library (formerly `refinitiv-data`).
Credentials are read from environment variables or passed explicitly —
never hardcoded here.

Session types
-------------
desktop (default)
    Requires Eikon or LSEG Workspace desktop app running locally.
    Only needs LSEG_APP_KEY (or config file).

    .env::
        LSEG_APP_KEY=your_app_key

platform
    Direct API access — no desktop app needed. Requires app-key +
    username + password from your LSEG Workspace API account.

    .env::
        LSEG_APP_KEY=your_app_key
        LSEG_USERNAME=your_username@domain.com
        LSEG_PASSWORD=your_password

    Get credentials at: https://developers.lseg.com/en/api-catalog/refinitiv-data-platform/refinitiv-data-platform-apis/quick-start

Config file alternative (LSEG default location):
    ~/.config/refinitiv/refinitiv-data.config.json

Install:
    poetry add lseg-data --optional

Usage::

    from jeanclaude.data.ingestion import RefinitivSource

    # Desktop (Eikon/Workspace running)
    src = RefinitivSource()

    # Platform API (no desktop app)
    src = RefinitivSource(session_type="platform")
    # reads LSEG_APP_KEY, LSEG_USERNAME, LSEG_PASSWORD from env

    prices = src.get_prices(
        tickers=["SPY.P", "TLT.P", "GLD.P"],
        start="2020-01-01",
        end="2024-12-31",
    )
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import pandas as pd

from .base import DataSource

logger = logging.getLogger(__name__)

# Module-level reference to lseg.data — set lazily on first _ensure_session()
# call.  Exposed at module scope so tests can patch it (patch.object(rmod, "ld")).
ld = None  # type: ignore[assignment]

# Refinitiv / LSEG field mapping
_PRICE_FIELD_MAP = {
    "close":  "TRDPRC_1",
    "open":   "OPEN_PRC",
    "high":   "HIGH_1",
    "low":    "LOW_1",
    "volume": "ACVOL_1",
    "bid":    "BID",
    "ask":    "ASK",
}


class RefinitivSource(DataSource):
    """Refinitiv / LSEG market data connector.

    Parameters
    ----------
    app_key : str, optional
        LSEG application key. Falls back to ``LSEG_APP_KEY`` env var,
        then to the LSEG config file.
    session_type : str
        ``"desktop"``  — requires Eikon/Workspace app running locally.
        ``"platform"`` — direct API, no desktop app needed.
    username : str, optional
        LSEG username (platform session only).
        Falls back to ``LSEG_USERNAME`` env var.
    password : str, optional
        LSEG password (platform session only).
        Falls back to ``LSEG_PASSWORD`` env var.
    """

    def __init__(
        self,
        app_key: Optional[str] = None,
        session_type: str = "desktop",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self._app_key = app_key or os.environ.get("LSEG_APP_KEY")
        self._session_type = session_type
        self._username = username or os.environ.get("LSEG_USERNAME")
        self._password = password or os.environ.get("LSEG_PASSWORD")
        self._session = None

    @property
    def name(self) -> str:
        return "Refinitiv/LSEG"

    def is_available(self) -> bool:
        """True if lseg-data is installed."""
        try:
            import lseg.data  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_session(self) -> None:
        """Open LSEG session lazily (once per instance).

        Desktop session: only app_key needed (Eikon/Workspace must be running).
        Platform session: app_key + username + password required.

        Side-effect: sets the module-level ``ld`` reference so callers and
        tests can reference / patch ``rmod.ld`` without a local import.
        """
        import jeanclaude.data.ingestion.refinitiv as _self_module

        if self._session is not None:
            return

        try:
            import lseg.data as _ld
        except ImportError as e:
            raise ImportError(
                "lseg-data is not installed. Run: poetry add lseg-data --optional\n"
                "Or add it to pyproject.toml under [tool.poetry.extras]."
            ) from e

        # Expose at module scope for testability
        _self_module.ld = _ld

        if self._session_type == "platform":
            self._session = self._open_platform_session(_ld)
        else:
            self._session = self._open_desktop_session(_ld)

        logger.info("LSEG session opened (type=%s)", self._session_type)

    def _open_desktop_session(self, ld):
        """Open desktop session (Eikon / LSEG Workspace app required)."""
        kwargs = {"name": "desktop"}
        if self._app_key:
            kwargs["app_key"] = self._app_key
        return ld.open_session(**kwargs)

    def _open_platform_session(self, ld):
        """Open platform session (direct API, no desktop app needed).

        Requires app_key + username + password.
        Get credentials at: https://developers.lseg.com/en/api-catalog/
        """
        missing = []
        if not self._app_key:
            missing.append("LSEG_APP_KEY")
        if not self._username:
            missing.append("LSEG_USERNAME")
        if not self._password:
            missing.append("LSEG_PASSWORD")

        if missing:
            raise ValueError(
                f"Platform session requires: {', '.join(missing)}.\n"
                "Set them in your .env file or pass explicitly to RefinitivSource().\n"
                "Get credentials at: https://developers.lseg.com/en/api-catalog/"
            )

        session = ld.session.platform.Definition(
            app_key=self._app_key,
            grant=ld.session.platform.GrantPassword(
                username=self._username,
                password=self._password,
            ),
            signon_control=True,  # force-close any existing session
        ).get_session()
        session.open()
        ld.session.set_default(session)
        return session

    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str,
        field: str = "close",
        adjustments: list[str] | tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """Fetch daily prices from Refinitiv.

        Parameters
        ----------
        tickers : list[str]
            Refinitiv RIC codes (e.g. ``'SPY.P'``, ``'TLT.P'``, ``'.SPX'``).
        start, end : str
            ISO date strings (``'YYYY-MM-DD'``).
        field : str
            One of ``'close'``, ``'open'``, ``'high'``, ``'low'``, ``'volume'``.
        adjustments : list[str] | tuple[str, ...] | None
            CORAX (Corporate Actions) adjustments to apply.  Supported values:
            ``'CCH'`` (cash corporate actions, e.g. return-of-capital / special
            dividends), ``'CRE'`` (capital reorganization events such as splits),
            ``'RTS'`` (rights issues), ``'RPO'`` (reversal of price adjustments).
            Example: ``adjustments=["CCH", "CRE", "RTS", "RPO"]``.

            **Important:** these adjustments do NOT automatically include ordinary
            dividend distributions — they cover extraordinary corporate events only.
            Whether the resulting series approximates total return depends on the
            specific CORAX configuration of each instrument.  Use
            ``scripts/validate_total_return.py`` to verify empirically.

            The lseg-data API accepts a comma-separated string; this method joins
            the list automatically (``",".join(adjustments)``).

        Returns
        -------
        pd.DataFrame
            Date-indexed DataFrame with one column per ticker.
        """
        self._ensure_session()
        import jeanclaude.data.ingestion.refinitiv as _self_module

        ric_field = _PRICE_FIELD_MAP.get(field)
        if ric_field is None:
            raise ValueError(f"Unknown field '{field}'. Valid: {list(_PRICE_FIELD_MAP)}")

        logger.info(
            "Fetching %s prices for %d tickers (%s → %s) adjustments=%s",
            field, len(tickers), start, end, adjustments,
        )

        # Build kwargs — adjustments is str (comma-separated) per lseg-data API
        history_kwargs: dict = {
            "universe": tickers,
            "fields": [ric_field],
            "interval": "daily",
            "start": start,
            "end": end,
        }
        if adjustments:
            # lseg-data summaries.Definition accepts a list of individual values;
            # the top-level get_history() type-hint says str but the actual
            # historical_pricing layer uses try_copy_to_list internally.
            # Pass a list — each element is validated against the Adjustments enum.
            history_kwargs["adjustments"] = list(adjustments)

        df = _self_module.ld.get_history(**history_kwargs)

        # lseg-data returns a MultiIndex or single-level depending on fields
        # Normalize to: DatetimeIndex rows, ticker columns
        df = self._normalize_history(df, tickers, ric_field)
        df.index.name = "date"
        return df

    def get_total_returns(
        self,
        rics: list[str],
        start: str,
        end: str,
        chunk_size: int = 10,
    ) -> pd.DataFrame:
        """Fetch daily total-return series (dividends included) via TR.TotalReturn1D.

        Campo validato 2026-06-11: KO CAGR 8.06% vs Yahoo 8.09% — include dividendi;
        CORAX no.

        The field ``TR.TotalReturn1D`` returns the **percentage** daily total return
        (e.g. ``0.5`` = +0.5%).  This method converts it to a fractional return
        (e.g. ``0.005``) by dividing by 100, consistent with the rest of the
        JeanClaude pipeline.

        Requests are chunked in batches of ``chunk_size`` RICs (default 10) to
        respect the Refinitiv rate limit validated in the 2026-06-11 probe (B3:
        10 RICs × 5 years in 1.90s, no throttling observed).

        Parameters
        ----------
        rics : list[str]
            Refinitiv RIC codes.  Includes delisted RICs (suffix ``^MMMYY``).
        start, end : str
            ISO date strings (``'YYYY-MM-DD'``).
        chunk_size : int
            Number of RICs per API call.  Default 10.

        Returns
        -------
        pd.DataFrame
            Date-indexed DataFrame, one column per RIC.
            Values are fractional daily total returns (e.g. 0.005 = +0.5%).
            Dates with no data for any RIC are dropped (``dropna(how='all')``).

        Notes
        -----
        - Uses ``ld.get_history`` (timeseries engine → TRUE trading dates).
          NEVER use ``ld.get_data`` with ``Frq=D`` for TR series: the formula
          engine LEFT-ANCHORS the dates to SDate — with SDate earlier than the
          listing date the whole series is shifted back in time (probe
          2026-06-12: ABNB.OQ post-IPO returns labelled 2002-2008), and the
          generated calendar contains phantom holiday rows.
        - CORAX adjustments are NOT applied — ordinary dividends are already
          embedded in the TR.TotalReturn1D computation.
        - A chunk that fails is retried once; if it fails again its RICs come
          back as all-NaN columns (loud warning in the log).

        Examples
        --------
        >>> src = RefinitivSource(session_type="platform")
        >>> tr = src.get_total_returns(["KO.N", "LEH.N^I08"], "2007-01-01", "2009-01-01")
        >>> tr.columns.tolist()
        ['KO.N', 'LEH.N^I08']
        """
        self._ensure_session()
        import jeanclaude.data.ingestion.refinitiv as _self_module

        logger.info(
            "Fetching TR.TotalReturn1D for %d RICs (%s → %s) chunk_size=%d",
            len(rics),
            start,
            end,
            chunk_size,
        )

        frames: list[pd.DataFrame] = []

        for chunk_start in range(0, len(rics), chunk_size):
            chunk = rics[chunk_start : chunk_start + chunk_size]
            chunk_df = self._fetch_total_returns_chunk(
                _self_module.ld,
                chunk,
                start,
                end,
            )
            frames.append(chunk_df)

        if not frames:
            return pd.DataFrame(dtype=float)

        result = pd.concat(frames, axis=1)
        result = result.reindex(columns=rics)
        # get_history può restituire l'ultima barra disponibile anche se
        # PRECEDE la finestra richiesta (es. delisted con window post-vita):
        # le righe fuori [start, end] vanno escluse.
        if isinstance(result.index, pd.DatetimeIndex):
            result = result.loc[
                (result.index >= pd.Timestamp(start))
                & (result.index <= pd.Timestamp(end))
            ]
        result.index.name = "date"
        result = result.dropna(how="all")
        return result

    @staticmethod
    def _fetch_total_returns_chunk(
        ld,
        rics: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch TR.TotalReturn1D for a single chunk of RICs via get_history.

        Returns a date-indexed DataFrame of fractional returns (÷100), with
        the TRUE trading dates from the timeseries engine.  One retry on
        failure; a chunk that fails twice returns empty (logged loudly).
        """
        raw = None
        for attempt in (1, 2):
            try:
                raw = ld.get_history(
                    universe=list(rics),
                    fields=["TR.TotalReturn1D"],
                    start=start,
                    end=end,
                )
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "get_history TR.TotalReturn1D tentativo %d/2 fallito per %s: %s",
                    attempt, rics, exc,
                )
                if attempt == 1:
                    time.sleep(2.0)

        if raw is None or len(raw) == 0:
            logger.warning("TR.TotalReturn1D: nessun dato per chunk %s", rics)
            return pd.DataFrame(dtype=float)

        df = pd.DataFrame(raw).copy()
        # Risposta single-RIC: unica colonna 'Daily Total Return' → nome RIC
        if len(rics) == 1 and len(df.columns) == 1:
            df.columns = [rics[0]]
        df = df.apply(pd.to_numeric, errors="coerce") / 100.0
        df.index = pd.to_datetime(df.index)
        return df

    def get_macro(
        self,
        series: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch macro time series using Refinitiv RIC codes.

        Parameters
        ----------
        series : list[str]
            Refinitiv RIC codes for macro indicators.
            Examples:
            - ``'VXc1'``      — VIX futures front-month (field: TRDPRC_1)
            - ``'US10YT=RR'`` — 10Y UST price (field: BID)
            - ``'US2YT=RR'``  — 2Y UST price (field: BID)
            - ``'XAU='``      — Gold spot (field: BID)
            - ``'CLc1'``      — WTI Oil futures (field: BID)
            - ``'.MOVE'``     — MOVE index (field: TRDPRC_1)
            - ``'EUR='``      — EURUSD spot (field: BID)

        Returns
        -------
        pd.DataFrame
            Date-indexed DataFrame, one column per series.

        Notes
        -----
        LSEG uses different price fields depending on instrument type:
        - Futures and indices: ``TRDPRC_1`` (last trade price)
        - Rates, FX, bond prices: ``BID``

        This method tries ``TRDPRC_1`` first, then falls back to ``BID``
        automatically, so mixed universes (e.g. VIX + yields) work in
        a single call by fetching each RIC individually.
        """
        self._ensure_session()
        import jeanclaude.data.ingestion.refinitiv as _self_module

        logger.info(
            "Fetching %d macro series (%s → %s)", len(series), start, end
        )

        frames: list[pd.DataFrame] = []
        for ric in series:
            df_ric = self._fetch_one_macro(_self_module.ld, ric, start, end)
            frames.append(df_ric)

        result = pd.concat(frames, axis=1)
        result.columns = series
        result.index.name = "date"
        result = result.ffill().dropna(how="all")
        return result

    def get_spread(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch bid-ask spread: (ask - bid) / mid per ticker.

        Parameters
        ----------
        tickers : list[str]
            Refinitiv RIC codes (e.g. ``'SPY.P'``, ``'TLT.P'``).
        start, end : str
            ISO date strings (``'YYYY-MM-DD'``).

        Returns
        -------
        pd.DataFrame
            Date-indexed DataFrame, columns = tickers.
            Values are fractional spreads (e.g. 0.001 = 10 bps).
            Rows where mid == 0 are NaN.
        """
        bid = self.get_prices(tickers, start=start, end=end, field="bid")
        ask = self.get_prices(tickers, start=start, end=end, field="ask")
        mid = (bid + ask) / 2.0
        spread = (ask - bid) / mid.where(mid != 0, other=float("nan"))
        return spread.ffill().dropna(how="all")

    def _fetch_one_macro(self, ld, ric: str, start: str, end: str) -> pd.DataFrame:
        """Fetch a single macro RIC, trying TRDPRC_1 then BID.

        lseg.get_history with a single RIC returns columns named after the
        *field* (e.g. 'TRDPRC_1'), not the RIC. We rename to the RIC name
        so all callers get a consistently-named Series.
        """
        for field in ("TRDPRC_1", "BID"):
            try:
                df = ld.get_history(
                    universe=[ric],
                    fields=[field],
                    interval="daily",
                    start=start,
                    end=end,
                )
                if df is not None and len(df) > 0:
                    df = df[[field]].rename(columns={field: ric})
                    df.index = pd.to_datetime(df.index)
                    df.index.name = "date"
                    logger.debug("RIC %s fetched with field=%s", ric, field)
                    return df
            except Exception:
                continue

        raise ValueError(
            f"Cannot fetch RIC '{ric}': neither TRDPRC_1 nor BID returned data. "
            "Check that the RIC code is correct and your subscription covers it."
        )

    @staticmethod
    def _normalize_history(
        df: pd.DataFrame,
        universe: list[str],
        field: str,
    ) -> pd.DataFrame:
        """Normalize lseg-data output to (date × ticker) shape.

        lseg.data.get_history() can return:
        - MultiIndex columns: (ticker, field)
        - Single-level columns: ticker names (multi-ticker) or field name (single-ticker)
        """
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(field, axis=1, level=1)

        df.columns = [str(c) for c in df.columns]

        # Single-ticker: lseg returns the column named after the field (e.g. "TRDPRC_1")
        # rather than the RIC. Rename it to the requested ticker so reindex works.
        if len(universe) == 1 and len(df.columns) == 1 and df.columns[0] != universe[0]:
            df = df.rename(columns={df.columns[0]: universe[0]})

        # Reindex to requested universe (fill missing with NaN)
        df = df.reindex(columns=universe)

        # Forward-fill weekends/holidays, then drop leading NaN rows
        df = df.ffill().dropna(how="all")

        return df


# ---------------------------------------------------------------------------
# Module-level helpers for get_total_returns
# ---------------------------------------------------------------------------
