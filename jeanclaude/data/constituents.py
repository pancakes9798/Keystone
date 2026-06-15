"""
Point-in-time S&P 500 constituent reconstruction.

Algorithm
---------
The Refinitiv JL (Joined-Leavers) change-log records **every membership
change** (join OR leave) with the date on which it occurred.  The
direction of each change is requested explicitly:

  - the ``IC=B`` parameter returns **B**oth joiners and leavers — without
    it the API silently returns leavers only (~824 events vs ~1670 for
    .SPX since 1995), which inflates reconstructed baskets going backward;
  - the ``TR.IndexJLConstituentRIC.change`` field returns a ``Change``
    column with ``Joiner``/``Leaver`` per event.

Backward reconstruction (basket_at convention)
----------------------------------------------
We define ``basket_at(T)`` as the composition **at close of T** — i.e.
all events with ``date <= T`` have been applied.

Starting from ``current_basket`` (the basket TODAY) we walk backward
through time, un-applying each JL event ``(date=D, ric=R)``:

  - explicit semantics (``change`` column present): before a JOIN the RIC
    was absent → remove R; before a LEAVE it was present → add R.  This
    does not depend on the running state, so same-direction duplicates
    are harmless.  Same-day join+leave pairs for the same RIC are
    cancelled as net no-ops BEFORE the walk: under the close-of-day
    convention the pair leaves membership unchanged whatever the intraday
    order, while resolving it event-by-event would depend on API row
    order (probe 2026-06-12: 10 such pairs in the .SPX log, e.g. the
    EVHC.N^L16 flash entry of 2016-12-02).
  - toggle fallback (no ``change`` column, e.g. legacy caches): membership
    of R is flipped.  Correct only if the log is complete and consistent
    with the current basket.

A RIC whose join pre-dates the log window (e.g. GM, member since before
1995) simply has no join event: it stays in the basket all the way back.

Reliability
-----------
The JL change-log via Refinitiv covers events from ~1995, with reliable
coverage from 2000 and best accuracy from **2003 onwards**.  For dates
before 2003 the reconstruction may undercount members due to missing
historical events.  Document any anomalies in validation output.

Usage
-----
Pure algorithm (no network)::

    from jeanclaude.data.constituents import reconstruct_baskets
    import pandas as pd

    current = frozenset(["AAPL.OQ", "MSFT.OQ"])
    events = pd.DataFrame({
        "date": [pd.Timestamp("2020-06-01")],
        "ric":  ["AAPL.OQ"],
    })
    snapshot_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")]
    baskets = reconstruct_baskets(current, events, snapshot_dates)

Refinitiv-backed (requires lseg-data + credentials)::

    from jeanclaude.data.constituents import ConstituentHistory
    from jeanclaude.data.ingestion import RefinitivSource
    from jeanclaude.data.storage import ParquetStore

    src = RefinitivSource(session_type="platform")
    store = ParquetStore("~/jeanclaude_data")
    ch = ConstituentHistory(source=src, store=store)
    baskets = ch.monthly_baskets("2003-01", "2026-06")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .ingestion.refinitiv import RefinitivSource
    from .storage.parquet_store import ParquetStore

logger = logging.getLogger(__name__)

# Default index RIC for S&P 500
_DEFAULT_INDEX_RIC = ".SPX"

# JL change-log Refinitiv fields.  The `.change` suffix on the RIC field
# yields the `Change` column (Joiner/Leaver) — the explicit ChangeType
# field is NOT returned by the API (probe 2026-06-11/12).
_JL_FIELDS = [
    "TR.IndexJLConstituentChangeDate",
    "TR.IndexJLConstituentRIC.change",
    "TR.IndexJLConstituentRIC",
    "TR.IndexJLConstituentName",
]

# IC=B returns Both joiners and leavers; the API default is leavers only.
_JL_PARAMETERS = {"SDate": "1995-01-01", "EDate": "2099-12-31", "IC": "B"}

# Normalization of the API `Change` column to the internal vocabulary.
_CHANGE_MAP = {"joiner": "join", "leaver": "leave"}

# Current constituent field
_CONSTITUENT_FIELD = "TR.RIC"

# Chain RIC for current basket
_CHAIN_RIC = "0#.SPX"


# ---------------------------------------------------------------------------
# Pure algorithm — no network, fully unit-testable
# ---------------------------------------------------------------------------


def reconstruct_baskets(
    current_basket: frozenset[str],
    events: pd.DataFrame,
    snapshot_dates: list[pd.Timestamp],
) -> dict[pd.Timestamp, frozenset[str]]:
    """Reconstruct point-in-time S&P 500 baskets from a JL change-log.

    Convention — ``basket_at(T)``
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The returned basket for date T represents the composition **at close of
    business on T**: all JL events with ``event.date <= T`` have been
    applied.  Events on T itself are considered already effective at T's
    close (they happened during the trading day).

    Algorithm
    ~~~~~~~~~
    Start from ``current_basket`` (today) and walk **backward** through the
    sorted events (most-recent first), un-applying each event ``(D, R)``
    that is more recent than the snapshot being filled.

    With a ``change`` column the un-apply is explicit — before a join R
    was absent (remove), before a leave it was present (add) — and
    same-day join+leave pairs for the same RIC are cancelled as net
    no-ops before the walk, so the result cannot depend on API row order.

    Without it, membership of R is toggled: correct only if the log is
    complete, consistent with the current basket, and free of same-day
    ambiguities (legacy leavers-only caches).

    Parameters
    ----------
    current_basket : frozenset[str]
        The index membership as of today (fetched live from the API).
    events : pd.DataFrame
        JL change-log.  Must have columns:
        - ``date``  : pd.Timestamp — the date the membership change took effect
        - ``ric``   : str          — the RIC of the constituent that changed
        - ``change``: str, optional — ``'join'`` or ``'leave'``.  When present,
          the backward walk uses explicit semantics (robust to duplicate
          events); when absent it falls back to membership toggling.
        Rows must not contain NaT dates or empty RICs; callers should clean
        the DataFrame before passing it here.
    snapshot_dates : list[pd.Timestamp]
        Dates at which to record the reconstructed basket.  May be in any
        order; the function sorts them internally and returns a dict keyed
        by the original (normalized) timestamps.

    Returns
    -------
    dict[pd.Timestamp, frozenset[str]]
        Mapping ``snapshot_date -> basket_at(snapshot_date)``.

    Examples
    --------
    Single-join case — AAPL entered the index on 2020-06-01:

    >>> current = frozenset(["AAPL.OQ", "MSFT.OQ"])
    >>> events = pd.DataFrame({"date": [pd.Timestamp("2020-06-01")],
    ...                        "ric": ["AAPL.OQ"]})
    >>> before = reconstruct_baskets(current, events, [pd.Timestamp("2020-05-31")])
    >>> "AAPL.OQ" in before[pd.Timestamp("2020-05-31")]
    False

    Rejoin case — documented in tests: ric in current, events at D1 < D2.
    Walking backward: at D2 we toggle (ric was present → now absent for
    dates in [D1, D2)); at D1 we toggle again (ric was absent → now
    present for dates < D1).  Interpretation: ric left at D1, rejoined at
    D2.
    """
    has_change = (not events.empty) and ("change" in events.columns)

    if not events.empty:
        # Validate required columns
        for col in ("date", "ric"):
            if col not in events.columns:
                raise ValueError(f"events DataFrame must have column '{col}'")
        if has_change:
            if events["change"].isna().any():
                raise ValueError(
                    "events 'change' column contains null values; "
                    "expected 'join' or 'leave'"
                )
            invalid = set(events["change"].unique()) - {"join", "leave"}
            if invalid:
                raise ValueError(
                    f"events 'change' column has invalid values {sorted(invalid)}; "
                    "expected 'join' or 'leave'"
                )
            # Same-day join+leave pairs are net no-ops on the close-of-day
            # state under either intraday ordering — cancel them so the walk
            # cannot depend on API row order.
            events = _cancel_same_day_pairs(events)

    # Sort snapshot dates descending (most recent first) so we can sweep
    # backward through events once.
    sorted_snapshots = sorted(snapshot_dates, reverse=True)

    # Sort events descending by date (stable, so same-day events keep the
    # API order).
    event_cols = ["date", "ric"] + (["change"] if has_change else [])
    if not events.empty:
        sorted_events = (
            events[event_cols]
            .dropna(subset=["date", "ric"])
            .sort_values("date", ascending=False, kind="stable")
            .reset_index(drop=True)
        )
    else:
        sorted_events = pd.DataFrame(columns=event_cols)

    running_basket: set[str] = set(current_basket)
    event_idx = 0
    n_events = len(sorted_events)

    result: dict[pd.Timestamp, frozenset[str]] = {}

    for snap_date in sorted_snapshots:
        # Un-apply all events with date > snap_date (i.e. events that happened
        # AFTER the snapshot — walking backward, these are events we need
        # to reverse to get the state AT snap_date).
        while event_idx < n_events:
            ev = sorted_events.iloc[event_idx]
            if ev["date"] <= snap_date:
                break  # This event is already in effect at snap_date — stop.
            ev_ric = ev["ric"]
            if has_change:
                # Explicit semantics: before a join the RIC was absent,
                # before a leave it was present.
                if ev["change"] == "join":
                    running_basket.discard(ev_ric)
                else:
                    running_basket.add(ev_ric)
            elif ev_ric in running_basket:
                running_basket.discard(ev_ric)
            else:
                running_basket.add(ev_ric)
            event_idx += 1

        result[snap_date] = frozenset(running_basket)

    return result


def _cancel_same_day_pairs(events: pd.DataFrame) -> pd.DataFrame:
    """Drop matched join+leave pairs sharing the same (date, ric).

    For each (date, ric) group, ``min(n_join, n_leave)`` events of EACH
    direction are removed; any surplus in one direction survives.  Which
    rows of a direction get dropped is irrelevant — they are
    interchangeable in the backward walk.

    Returns a new DataFrame; the input is not mutated.
    """
    sizes = (
        events.groupby(["date", "ric", "change"])
        .size()
        .unstack(fill_value=0)
    )
    if "join" not in sizes.columns or "leave" not in sizes.columns:
        return events  # only one direction present — no pairs possible

    n_cancel = sizes[["join", "leave"]].min(axis=1).rename("_n_cancel")
    if (n_cancel == 0).all():
        return events

    df = events.copy()
    df["_rank"] = df.groupby(["date", "ric", "change"]).cumcount()
    df = df.join(n_cancel, on=["date", "ric"])
    kept = df.loc[df["_rank"] >= df["_n_cancel"].fillna(0), list(events.columns)]

    n_dropped = len(events) - len(kept)
    logger.info(
        "Cancelled %d same-day join+leave events (%d net no-op pairs)",
        n_dropped,
        n_dropped // 2,
    )
    return kept.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Refinitiv-backed layer
# ---------------------------------------------------------------------------


class ConstituentHistory:
    """Fetch and reconstruct S&P 500 constituent history via Refinitiv JL log.

    Parameters
    ----------
    source : RefinitivSource
        Authenticated Refinitiv/LSEG data source.
    store : ParquetStore
        Local Parquet cache.
    index_ric : str
        RIC of the index.  Defaults to ``'.SPX'`` (S&P 500).
    """

    def __init__(
        self,
        source: RefinitivSource,
        store: ParquetStore,
        index_ric: str = _DEFAULT_INDEX_RIC,
    ) -> None:
        self._source = source
        self._store = store
        self._index_ric = index_ric

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def monthly_baskets(
        self,
        start: str,
        end: str,
    ) -> dict[pd.Timestamp, frozenset[str]]:
        """Build month-end point-in-time baskets for the index.

        Fetches (or loads from cache):
        1. The current basket (live from Refinitiv chain).
        2. The JL change-log (all events from 1995 to today).

        Then calls :func:`reconstruct_baskets` on month-end dates from
        *start* to *end*.

        Results are cached to Parquet:
        - ``constituents/events__<fetch_date>.parquet``
        - ``constituents/current_basket__<fetch_date>.parquet``

        Parameters
        ----------
        start, end : str
            ISO year-month strings (``'YYYY-MM'``).  Both are inclusive.
            The function generates month-end snapshot dates in this range.

        Returns
        -------
        dict[pd.Timestamp, frozenset[str]]
            Month-end date → frozenset of RIC codes.
        """
        import datetime

        today = datetime.date.today().isoformat()

        # 1. Load or fetch current basket
        current_basket = self._get_or_fetch_current_basket(today)

        # 2. Load or fetch JL events
        events = self._get_or_fetch_jl_events(today)

        # 3. Build month-end snapshot dates.
        # pd.date_range with freq="ME" generates the last calendar day of each
        # month. We extend the end by one month so that the last requested
        # month is always included (e.g. end="2021-03" → we use "2021-04" so
        # that 2021-03-31 is generated).
        end_extended = (
            pd.Timestamp(end) + pd.offsets.MonthEnd(1)
        ).strftime("%Y-%m-%d")
        snapshot_dates = list(
            pd.date_range(start=start, end=end_extended, freq="ME")
        )
        logger.info(
            "Reconstructing %d month-end baskets (%s → %s)",
            len(snapshot_dates),
            start,
            end,
        )

        # 4. Reconstruct
        baskets = reconstruct_baskets(
            frozenset(current_basket),
            events,
            snapshot_dates,
        )
        return baskets

    # ------------------------------------------------------------------
    # Internal fetch helpers
    # ------------------------------------------------------------------

    def _get_or_fetch_current_basket(self, fetch_date: str) -> list[str]:
        """Return the current S&P 500 basket, from cache or live fetch."""
        cached = self._store.load("constituents", "current_basket", fetch_date, fetch_date)
        if cached is not None:
            return list(cached["ric"].dropna())

        logger.info("Fetching current basket for %s", self._index_ric)
        rics = self._fetch_current_basket_live()
        df = pd.DataFrame({"ric": rics})
        self._store.save("constituents", "current_basket", fetch_date, fetch_date, df)
        return rics

    def _get_or_fetch_jl_events(self, fetch_date: str) -> pd.DataFrame:
        """Return the JL change-log, from cache or live fetch."""
        cached = self._store.load("constituents", "jl_events", fetch_date, fetch_date)
        if cached is not None:
            # ParquetStore.load() may promote 'date' to the index if it finds
            # it; reset to a regular column before returning.
            if "date" not in cached.columns:
                cached = cached.reset_index()
            cached["date"] = pd.to_datetime(cached["date"])
            # Legacy caches (pre IC=B) have no 'change' column — the
            # reconstruction falls back to toggle semantics for those.
            cols = ["date", "ric"] + (["change"] if "change" in cached.columns else [])
            return cached[cols].dropna(subset=["date", "ric"])

        logger.info("Fetching JL change-log for %s", self._index_ric)
        events = self._fetch_jl_events_live()
        self._store.save("constituents", "jl_events", fetch_date, fetch_date, events)
        return events

    def _fetch_current_basket_live(self) -> list[str]:
        """Fetch current basket from Refinitiv chain (0#.SPX)."""
        self._source._ensure_session()
        import jeanclaude.data.ingestion.refinitiv as rmod

        chain_ric = f"0#{self._index_ric}"
        df = rmod.ld.get_data(
            universe=chain_ric,
            fields=[_CONSTITUENT_FIELD],
        )
        if df is None or df.empty:
            raise RuntimeError(
                f"No current basket returned for chain '{chain_ric}'."
            )
        # Column may be named 'RIC' or 'Constituent RIC' — normalize
        col = _find_column(df, ["ric", "constituent ric", "tr.ric"])
        rics = df[col].dropna().astype(str).tolist()
        logger.info("Current basket: %d RICs", len(rics))
        return rics

    def _fetch_jl_events_live(self) -> pd.DataFrame:
        """Fetch full JL change-log from Refinitiv (earliest available → today).

        Requests ``IC=B`` (both joiners and leavers — the API default is
        leavers only) and the ``Change`` column, normalized to
        ``'join'``/``'leave'``.  Unknown change values fail loudly.
        """
        self._source._ensure_session()
        import jeanclaude.data.ingestion.refinitiv as rmod

        df = rmod.ld.get_data(
            universe=self._index_ric,
            fields=_JL_FIELDS,
            parameters=dict(_JL_PARAMETERS),
        )
        if df is None or df.empty:
            raise RuntimeError(
                f"No JL change-log returned for '{self._index_ric}'."
            )

        # Normalize column names to lowercase for robustness
        df.columns = [str(c).lower().strip() for c in df.columns]

        # Extract columns (names vary by API version)
        date_col = _find_column(df, ["date", "constituent change date", "tr.indexjlconstituentchangedate"])
        ric_col = _find_column(df, ["constituent ric", "tr.indexjlconstituentric", "ric"])
        change_col = _find_column(df, ["change", "tr.indexjlconstituentric.change"])

        events = pd.DataFrame({
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "ric": df[ric_col].astype(str),
            "change": df[change_col].astype(str).str.strip().str.lower(),
        }).dropna(subset=["date", "ric"])

        # Drop placeholder strings
        events = events[events["ric"].str.strip() != ""]
        events = events[events["ric"] != "nan"]

        # Normalize Joiner/Leaver → join/leave; anything else is a data
        # contract violation and must fail loudly.
        mapped = events["change"].map(_CHANGE_MAP)
        unknown = sorted(set(events.loc[mapped.isna(), "change"]))
        if unknown:
            raise ValueError(
                f"JL change-log: unexpected Change values {unknown}; "
                "expected 'Joiner' or 'Leaver'"
            )
        events = events.assign(change=mapped).reset_index(drop=True)

        n_join = int((events["change"] == "join").sum())
        n_leave = int((events["change"] == "leave").sum())
        logger.info(
            "JL change-log: %d events (%d join / %d leave, %s → %s)",
            len(events),
            n_join,
            n_leave,
            events["date"].min().date() if len(events) else "N/A",
            events["date"].max().date() if len(events) else "N/A",
        )
        return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first column name (case-insensitive) that matches any candidate.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with already-lowercased column names.
    candidates : list[str]
        Lowercase candidate column names to try in order.

    Raises
    ------
    KeyError
        If no matching column is found.
    """
    cols_lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    raise KeyError(
        f"None of {candidates} found in columns {list(df.columns)}"
    )
