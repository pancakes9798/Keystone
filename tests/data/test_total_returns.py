"""
Tests for RefinitivSource.get_total_returns (TR.TotalReturn1D via get_history).

All tests mock the lseg.data network layer — no external API calls.

Field validation notes:
  2026-06-11 — KO CAGR 8.06% vs Yahoo 8.09% — include dividendi; CORAX no.
  Il campo restituisce percentuali; get_total_returns divide per 100.
  2026-06-12 — get_data con Frq=D ANCORA le date a SDate (con SDate
  precedente alla quotazione l'intera serie risulta spostata indietro di
  anni — caso ABNB.OQ). L'implementazione usa ld.get_history (timeseries
  engine, date di borsa VERE).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

import jeanclaude.data.ingestion.refinitiv as rmod
from jeanclaude.data.ingestion.refinitiv import RefinitivSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source() -> RefinitivSource:
    src = RefinitivSource(session_type="platform")
    src._session = MagicMock()  # skip _ensure_session network call
    return src


def _history_multi(data: dict[str, list[float]], dates: list[str]) -> pd.DataFrame:
    """Fake ld.get_history multi-RIC response: columns = RIC names (pct)."""
    df = pd.DataFrame(data, index=pd.to_datetime(dates))
    df.index.name = "Date"
    df.columns.name = "Daily Total Return"
    return df


def _history_single(ric: str, dates: list[str], pct: list[float]) -> pd.DataFrame:
    """Fake ld.get_history single-RIC response: one 'Daily Total Return' column."""
    df = pd.DataFrame({"Daily Total Return": pct}, index=pd.to_datetime(dates))
    df.index.name = "Date"
    df.columns.name = ric
    return df


# ---------------------------------------------------------------------------
# RefinitivSource.get_total_returns
# ---------------------------------------------------------------------------


class TestGetTotalReturns:
    def test_percent_converted_to_fraction(self):
        src = _make_source()
        fake_ld = MagicMock()
        fake_ld.get_history.return_value = _history_multi(
            {"KO.N": [0.5, -0.25]}, ["2020-01-02", "2020-01-03"]
        )
        with patch.object(rmod, "ld", fake_ld):
            result = src.get_total_returns(["KO.N"], "2020-01-01", "2020-01-31")
        assert abs(result["KO.N"].iloc[0] - 0.005) < 1e-9
        assert abs(result["KO.N"].iloc[1] - (-0.0025)) < 1e-9

    def test_true_dates_preserved_no_reanchoring(self):
        """Il punto del fix: le date della risposta sono quelle VERE di borsa
        (es. IPO 2020-12-10 con richiesta dal 2002) e NON vengono toccate."""
        src = _make_source()
        true_dates = ["2020-12-10", "2020-12-11", "2020-12-14"]
        fake_ld = MagicMock()
        fake_ld.get_history.return_value = _history_multi(
            {"ABNB.OQ": [0.0, -3.77, -6.64]}, true_dates
        )
        with patch.object(rmod, "ld", fake_ld):
            result = src.get_total_returns(["ABNB.OQ"], "2002-09-01", "2026-06-12")
        assert list(result.index) == [pd.Timestamp(d) for d in true_dates]

    def test_single_ric_column_renamed_to_ric(self):
        """Risposta single-RIC: colonna 'Daily Total Return' → nome RIC."""
        src = _make_source()
        fake_ld = MagicMock()
        fake_ld.get_history.return_value = _history_single(
            "AAPL.OQ", ["2022-03-01"], [0.8]
        )
        with patch.object(rmod, "ld", fake_ld):
            result = src.get_total_returns(["AAPL.OQ"], "2022-01-01", "2022-12-31")
        assert list(result.columns) == ["AAPL.OQ"]

    def test_columns_match_requested_rics(self):
        src = _make_source()

        def fake_get_history(universe, fields, start, end):
            return _history_multi(
                {r: [1.0] for r in universe}, ["2021-01-04"]
            )

        fake_ld = MagicMock()
        fake_ld.get_history.side_effect = fake_get_history
        rics = ["AAPL.OQ", "MSFT.OQ", "KO.N"]
        with patch.object(rmod, "ld", fake_ld):
            result = src.get_total_returns(rics, "2021-01-01", "2021-12-31")
        assert list(result.columns) == rics

    def test_chunking_splits_into_batches(self):
        """chunk_size=2 e 5 RIC → 3 chiamate get_history (2+2+1)."""
        src = _make_source()
        call_log: list[list[str]] = []

        def fake_get_history(universe, fields, start, end):
            call_log.append(list(universe))
            return _history_multi({r: [0.5] for r in universe}, ["2021-01-04"])

        fake_ld = MagicMock()
        fake_ld.get_history.side_effect = fake_get_history
        with patch.object(rmod, "ld", fake_ld):
            src.get_total_returns(
                ["A", "B", "C", "D", "E"], "2021-01-01", "2021-12-31", chunk_size=2
            )
        assert call_log == [["A", "B"], ["C", "D"], ["E"]]

    def test_date_index_name_and_type(self):
        src = _make_source()
        fake_ld = MagicMock()
        fake_ld.get_history.return_value = _history_multi(
            {"KO.N": [0.2]}, ["2021-03-01"]
        )
        with patch.object(rmod, "ld", fake_ld):
            result = src.get_total_returns(["KO.N"], "2021-01-01", "2021-12-31")
        assert result.index.name == "date"
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_empty_ric_list_returns_empty_dataframe(self):
        src = _make_source()
        fake_ld = MagicMock()
        with patch.object(rmod, "ld", fake_ld):
            result = src.get_total_returns([], "2021-01-01", "2021-12-31")
        assert result.empty
        assert fake_ld.get_history.call_count == 0

    def test_failed_chunk_other_columns_survive(self):
        """Un chunk che fallisce (anche al retry) non affonda gli altri."""
        src = _make_source()

        def fake_get_history(universe, fields, start, end):
            if "BAD.RIC" in universe:
                raise RuntimeError("not found")
            return _history_multi({r: [0.5] for r in universe}, ["2021-06-01"])

        fake_ld = MagicMock()
        fake_ld.get_history.side_effect = fake_get_history
        with patch.object(rmod, "ld", fake_ld), patch.object(rmod.time, "sleep"):
            result = src.get_total_returns(
                ["GOOD.RIC", "BAD.RIC"], "2021-01-01", "2021-12-31", chunk_size=1
            )
        assert "GOOD.RIC" in result.columns
        if "BAD.RIC" in result.columns:
            assert result["BAD.RIC"].isna().all()

    def test_transient_failure_retried_once(self):
        """Primo tentativo fallisce, il retry riesce: i dati arrivano."""
        src = _make_source()
        calls = {"n": 0}

        def flaky(universe, fields, start, end):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            return _history_multi({r: [0.5] for r in universe}, ["2021-06-01"])

        fake_ld = MagicMock()
        fake_ld.get_history.side_effect = flaky
        with patch.object(rmod, "ld", fake_ld), patch.object(rmod.time, "sleep"):
            result = src.get_total_returns(["KO.N"], "2021-01-01", "2021-12-31")
        assert calls["n"] == 2
        assert not result["KO.N"].isna().all()

    def test_none_response_returns_empty(self):
        src = _make_source()
        fake_ld = MagicMock()
        fake_ld.get_history.return_value = None
        with patch.object(rmod, "ld", fake_ld):
            result = src.get_total_returns(["KO.N"], "2021-01-01", "2021-12-31")
        assert "KO.N" in result.columns
        assert result["KO.N"].isna().all() or result.empty

    def test_rows_outside_requested_window_clipped(self):
        """get_history può restituire l'ultima barra disponibile anche se
        precede la finestra richiesta (LEH richiesto 2015+ → barra 2008):
        le righe fuori [start, end] vanno eliminate."""
        src = _make_source()
        fake_ld = MagicMock()
        fake_ld.get_history.return_value = _history_multi(
            {"LEH.N^I08": [-13.5]}, ["2008-09-12"]
        )
        with patch.object(rmod, "ld", fake_ld):
            result = src.get_total_returns(["LEH.N^I08"], "2015-01-01", "2025-12-31")
        assert result.empty or result["LEH.N^I08"].dropna().empty
