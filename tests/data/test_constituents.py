"""
Tests for jeanclaude.data.constituents

Part 1 — pure algorithm (no network):
  - reconstruct_baskets correctness

Part 2 — thin plumbing tests for ConstituentHistory (mocked network)

All tests use synthetic data — no external API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from jeanclaude.data.constituents import ConstituentHistory, reconstruct_baskets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _events(*rows: tuple) -> pd.DataFrame:
    """Build a minimal events DataFrame from (date_str, ric) tuples."""
    dates, rics = zip(*rows) if rows else ([], [])
    return pd.DataFrame({
        "date": [pd.Timestamp(d) for d in dates],
        "ric": list(rics),
    })


def _events_typed(*rows: tuple) -> pd.DataFrame:
    """Build an events DataFrame from (date_str, ric, change) tuples."""
    dates, rics, changes = zip(*rows) if rows else ([], [], [])
    return pd.DataFrame({
        "date": [pd.Timestamp(d) for d in dates],
        "ric": list(rics),
        "change": list(changes),
    })


def _snap(*date_strs: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(d) for d in date_strs]


# ---------------------------------------------------------------------------
# reconstruct_baskets — Pure algorithm tests
# ---------------------------------------------------------------------------


class TestReconstructBasketsEmptyEvents:
    """When there are no events all snapshots equal the current basket."""

    def test_all_snapshots_equal_current(self):
        current = frozenset(["A", "B", "C"])
        events = pd.DataFrame(columns=["date", "ric"])
        snaps = _snap("2020-01-31", "2021-01-31", "2022-01-31")
        result = reconstruct_baskets(current, events, snaps)

        for s in snaps:
            assert result[s] == current, f"Basket at {s} differs from current"

    def test_single_snapshot_equals_current(self):
        current = frozenset(["X"])
        events = pd.DataFrame(columns=["date", "ric"])
        result = reconstruct_baskets(current, events, _snap("2010-06-30"))
        assert result[pd.Timestamp("2010-06-30")] == current


class TestSingleJoin:
    """A RIC in the current basket joined the index at some point D.

    Walking backward:
      - At/after D → ric is present (it had already joined)
      - Before D   → ric was NOT yet present (it joined at D)
    """

    D = pd.Timestamp("2020-06-01")
    CURRENT = frozenset(["AAPL.OQ", "MSFT.OQ"])
    EVENTS = _events(("2020-06-01", "AAPL.OQ"))

    def test_before_join_date_ric_absent(self):
        snap_before = pd.Timestamp("2020-05-31")
        result = reconstruct_baskets(self.CURRENT, self.EVENTS, [snap_before])
        assert "AAPL.OQ" not in result[snap_before]

    def test_on_join_date_ric_present(self):
        """basket_at(D) = basket at CLOSE of D — event on D is already applied."""
        result = reconstruct_baskets(self.CURRENT, self.EVENTS, [self.D])
        assert "AAPL.OQ" in result[self.D]

    def test_after_join_date_ric_present(self):
        snap_after = pd.Timestamp("2021-01-31")
        result = reconstruct_baskets(self.CURRENT, self.EVENTS, [snap_after])
        assert "AAPL.OQ" in result[snap_after]

    def test_unaffected_ric_always_present(self):
        snaps = _snap("2019-12-31", "2020-05-31", "2020-06-01", "2021-01-01")
        result = reconstruct_baskets(self.CURRENT, self.EVENTS, snaps)
        for s in snaps:
            assert "MSFT.OQ" in result[s], f"MSFT.OQ missing at {s}"


class TestSingleLeave:
    """A RIC NOT in the current basket was delisted/removed at D.

    Walking backward:
      - Before D   → ric WAS present
      - At/after D → ric is absent (it left)
    """

    D = pd.Timestamp("2008-09-17")
    CURRENT = frozenset(["MSFT.OQ", "GE.N"])  # LEH NOT in current
    EVENTS = _events(("2008-09-17", "LEH.N^I08"))

    def test_before_leave_date_ric_present(self):
        snap_before = pd.Timestamp("2008-09-16")
        result = reconstruct_baskets(self.CURRENT, self.EVENTS, [snap_before])
        assert "LEH.N^I08" in result[snap_before]

    def test_on_leave_date_ric_absent(self):
        """On the leave date, the event is applied → ric already gone at close."""
        result = reconstruct_baskets(self.CURRENT, self.EVENTS, [self.D])
        assert "LEH.N^I08" not in result[self.D]

    def test_after_leave_date_ric_absent(self):
        snap_after = pd.Timestamp("2009-01-01")
        result = reconstruct_baskets(self.CURRENT, self.EVENTS, [snap_after])
        assert "LEH.N^I08" not in result[snap_after]

    def test_other_rics_unaffected(self):
        snaps = _snap("2007-12-31", "2008-09-16", "2008-09-17", "2009-06-30")
        result = reconstruct_baskets(self.CURRENT, self.EVENTS, snaps)
        for s in snaps:
            assert "MSFT.OQ" in result[s]
            assert "GE.N" in result[s]


class TestRejoinCase:
    """A RIC is in the current basket but has two JL events: D1 < D2.

    This is the leave-then-rejoin interpretation:
      - ric left the index at D1
      - ric rejoined the index at D2

    Backward toggle logic from today (ric present):
      1. Encounter event at D2: toggle → ric now ABSENT in running basket
         (representing state BEFORE D2, i.e. [D1, D2))
      2. Encounter event at D1: toggle → ric now PRESENT in running basket
         (representing state BEFORE D1)

    Expected baskets:
      - before D1  → present
      - in [D1, D2) → absent
      - at/after D2 → present
    """

    D1 = pd.Timestamp("2005-03-01")   # left
    D2 = pd.Timestamp("2010-08-15")   # rejoined
    RIC = "DELL.OQ"
    CURRENT = frozenset([RIC, "AAPL.OQ"])
    EVENTS = _events(("2005-03-01", RIC), ("2010-08-15", RIC))

    def test_before_d1_ric_present(self):
        snap = pd.Timestamp("2005-02-28")
        result = reconstruct_baskets(self.CURRENT, self.EVENTS, [snap])
        assert self.RIC in result[snap]

    def test_on_d1_ric_absent(self):
        """Leave event at D1 is applied at close of D1 → absent."""
        result = reconstruct_baskets(self.CURRENT, self.EVENTS, [self.D1])
        assert self.RIC not in result[self.D1]

    def test_between_d1_and_d2_ric_absent(self):
        snap_mid = pd.Timestamp("2007-06-30")
        result = reconstruct_baskets(self.CURRENT, self.EVENTS, [snap_mid])
        assert self.RIC not in result[snap_mid]

    def test_on_d2_ric_present(self):
        """Rejoin event at D2 applied at close of D2 → present again."""
        result = reconstruct_baskets(self.CURRENT, self.EVENTS, [self.D2])
        assert self.RIC in result[self.D2]

    def test_after_d2_ric_present(self):
        snap_after = pd.Timestamp("2015-12-31")
        result = reconstruct_baskets(self.CURRENT, self.EVENTS, [snap_after])
        assert self.RIC in result[snap_after]

    def test_unaffected_ric_always_present(self):
        snaps = _snap("2004-12-31", "2005-03-01", "2008-01-31", "2010-08-15", "2020-01-31")
        result = reconstruct_baskets(self.CURRENT, self.EVENTS, snaps)
        for s in snaps:
            assert "AAPL.OQ" in result[s], f"AAPL missing at {s}"


class TestCountProperty:
    """Snapshot sizes change by exactly the number of net flips between dates.

    Setup: current basket = {A, B, C, D} (4 members)
      Event 1: E1 at 2015-06-01 — E1 was NOT in current → it LEFT (contributed
               +1 to basket before that date)
      Event 2: D  at 2020-03-01 — D IS in current → it JOINED (contributed
               -1 to basket before that date)

    Expected counts:
      - Before 2015-06-01:  4 + 1 (E1 added back) = 5
      - [2015-06-01, 2020-03-01): 4 + 1 (E1 still gone from backward perspective?

    Let me re-reason carefully:
    Starting from {A, B, C, D} (today), walk backward:

      Snap at 2021-01-01 (after both events):
        no events to un-apply → basket = {A, B, C, D} → size 4

      Snap at 2018-01-01 (after 2015 event, before 2020 event):
        un-apply 2020-03-01: D is IN basket → toggle → D removed from running
        basket is now {A, B, C} → size 3
        (wait, 2020-03-01 > 2018-01-01, so we un-apply it)

      Snap at 2014-01-01 (before both events):
        un-apply 2020-03-01: D toggled out → {A, B, C}
        un-apply 2015-06-01: E1 NOT in basket → toggle → E1 added → {A, B, C, E1}
        size 4

    So counts: 2021→4, 2018→3, 2014→4.  Net flip between 2021 and 2018 = -1 (D removed).
    Net flip between 2018 and 2014 = +1 (E1 added).
    """

    CURRENT = frozenset(["A", "B", "C", "D"])
    # D joined at 2020-03-01 (D is in current → join event)
    # E1 left at 2015-06-01 (E1 not in current → leave event)
    EVENTS = _events(
        ("2015-06-01", "E1"),
        ("2020-03-01", "D"),
    )

    def test_count_after_both_events(self):
        """After 2020-03-01 — same as current basket."""
        snap = pd.Timestamp("2021-01-01")
        result = reconstruct_baskets(self.CURRENT, self.EVENTS, [snap])
        assert len(result[snap]) == 4

    def test_count_between_events(self):
        """Between 2015 and 2020 — D not yet joined → 3 members."""
        snap = pd.Timestamp("2018-01-01")
        result = reconstruct_baskets(self.CURRENT, self.EVENTS, [snap])
        assert len(result[snap]) == 3, f"Expected 3, got {result[snap]}"

    def test_count_before_both_events(self):
        """Before 2015-06-01 — D not yet joined (−1), E1 still there (+1) → 4 members."""
        snap = pd.Timestamp("2014-01-01")
        result = reconstruct_baskets(self.CURRENT, self.EVENTS, [snap])
        assert len(result[snap]) == 4, f"Expected 4, got {result[snap]}"

    def test_size_changes_by_net_flips_between_adjacent_snapshots(self):
        """Verify count deltas match event count between snapshot pairs."""
        snaps = _snap("2014-01-01", "2018-01-01", "2021-01-01")
        result = reconstruct_baskets(self.CURRENT, self.EVENTS, snaps)
        # 2014→2018: net +1 D enters (from backward perspective: D un-toggled)
        #            Actually: count goes from 4 (2014) to 3 (2018) = delta -1
        delta_a = len(result[pd.Timestamp("2018-01-01")]) - len(result[pd.Timestamp("2014-01-01")])
        # Events between 2014 and 2018 (exclusive of 2014, inclusive of 2018):
        # Only 2015-06-01 (E1 leave). Net change = -1 (one fewer member).
        assert delta_a == -1

        # 2018→2021: D joins → +1
        delta_b = len(result[pd.Timestamp("2021-01-01")]) - len(result[pd.Timestamp("2018-01-01")])
        assert delta_b == 1


class TestEdgeCases:
    """Edge cases: snapshot ordering, same-day events, single member basket."""

    def test_snapshot_order_independent(self):
        """Result must not depend on the order of snapshot_dates passed in."""
        current = frozenset(["X", "Y"])
        events = _events(("2019-06-30", "X"))
        snaps_asc = _snap("2018-12-31", "2019-12-31")
        snaps_desc = _snap("2019-12-31", "2018-12-31")

        result_asc = reconstruct_baskets(current, events, snaps_asc)
        result_desc = reconstruct_baskets(current, events, snaps_desc)

        for s in snaps_asc:
            assert result_asc[s] == result_desc[s]

    def test_two_events_same_day_cancel_each_other(self):
        """Two events for the same RIC on the same day toggle twice → net neutral."""
        current = frozenset(["A"])
        events = _events(("2020-01-15", "A"), ("2020-01-15", "A"))
        snap = pd.Timestamp("2020-01-14")
        result = reconstruct_baskets(current, events, [snap])
        # Two toggles: A (in) → out → in. Net: A still present.
        assert "A" in result[snap]

    def test_snapshot_exactly_at_event_date(self):
        """Event on snap_date is included in basket_at(snap_date)."""
        current = frozenset(["Z"])
        events = _events(("2015-03-31", "Z"))
        snap = pd.Timestamp("2015-03-31")
        result = reconstruct_baskets(current, events, [snap])
        # Z is in current; event at 2015-03-31: snap == event date → event applied
        assert "Z" in result[snap]

    def test_empty_current_basket(self):
        """Empty current basket with leave events adds members to past baskets."""
        current: frozenset[str] = frozenset()
        events = _events(("2010-01-01", "DELISTD.N"))
        snap = pd.Timestamp("2009-12-31")
        result = reconstruct_baskets(current, events, [snap])
        assert "DELISTD.N" in result[snap]

    def test_missing_required_column_raises(self):
        """Events missing 'ric' column raises ValueError."""
        current = frozenset(["A"])
        bad_events = pd.DataFrame({"date": [pd.Timestamp("2020-01-01")]})
        with pytest.raises(ValueError, match="ric"):
            reconstruct_baskets(current, bad_events, _snap("2019-12-31"))


# ---------------------------------------------------------------------------
# reconstruct_baskets — explicit join/leave semantics (change column)
# ---------------------------------------------------------------------------


class TestExplicitChangeSemantics:
    """With a 'change' column the backward walk uses explicit semantics:

      - un-applying a JOIN  → the RIC was NOT a member before the event
      - un-applying a LEAVE → the RIC WAS a member before the event

    Unlike the toggle, this does not depend on the running state, so
    duplicate or redundant events cannot corrupt the reconstruction.
    """

    def test_joiner_removed_before_join_date(self):
        """TSLA case: in current basket, join event 2020-12-21."""
        current = frozenset(["TSLA.OQ", "AAPL.OQ"])
        events = _events_typed(("2020-12-21", "TSLA.OQ", "join"))
        snaps = _snap("2015-12-31", "2020-12-21", "2021-12-31")
        result = reconstruct_baskets(current, events, snaps)
        assert "TSLA.OQ" not in result[pd.Timestamp("2015-12-31")]
        assert "TSLA.OQ" in result[pd.Timestamp("2020-12-21")]
        assert "TSLA.OQ" in result[pd.Timestamp("2021-12-31")]

    def test_leaver_present_before_leave_date(self):
        """GM case: not in current, leave event only (join pre-dates the log)."""
        current = frozenset(["AAPL.OQ"])
        events = _events_typed(("2009-06-03", "GM.N^F09", "leave"))
        snaps = _snap("2000-01-31", "2008-12-31", "2009-12-31")
        result = reconstruct_baskets(current, events, snaps)
        assert "GM.N^F09" in result[pd.Timestamp("2000-01-31")]
        assert "GM.N^F09" in result[pd.Timestamp("2008-12-31")]
        assert "GM.N^F09" not in result[pd.Timestamp("2009-12-31")]

    def test_join_then_leave_membership_window(self):
        """LEH case: join 1998-01-12, leave 2008-09-17, not in current."""
        current = frozenset(["AAPL.OQ"])
        events = _events_typed(
            ("1998-01-12", "LEH.N^I08", "join"),
            ("2008-09-17", "LEH.N^I08", "leave"),
        )
        snaps = _snap("1997-12-31", "1998-01-12", "2008-06-30", "2008-12-31")
        result = reconstruct_baskets(current, events, snaps)
        assert "LEH.N^I08" not in result[pd.Timestamp("1997-12-31")]
        assert "LEH.N^I08" in result[pd.Timestamp("1998-01-12")]
        assert "LEH.N^I08" in result[pd.Timestamp("2008-06-30")]
        assert "LEH.N^I08" not in result[pd.Timestamp("2008-12-31")]

    def test_duplicate_leave_events_do_not_corrupt(self):
        """Two leave events for the same RIC (data quirk): explicit semantics
        keep the RIC present before both — a toggle would wrongly flip it out."""
        current = frozenset(["AAPL.OQ"])
        events = _events_typed(
            ("2005-03-01", "DUP.N", "leave"),
            ("2010-08-15", "DUP.N", "leave"),
        )
        snaps = _snap("2004-12-31", "2007-06-30", "2011-01-31")
        result = reconstruct_baskets(current, events, snaps)
        assert "DUP.N" in result[pd.Timestamp("2004-12-31")]
        assert "DUP.N" in result[pd.Timestamp("2007-06-30")]
        assert "DUP.N" not in result[pd.Timestamp("2011-01-31")]

    def test_leave_then_rejoin_with_explicit_changes(self):
        """Leave at D1, rejoin at D2, in current basket."""
        current = frozenset(["DELL.OQ"])
        events = _events_typed(
            ("2005-03-01", "DELL.OQ", "leave"),
            ("2010-08-15", "DELL.OQ", "join"),
        )
        snaps = _snap("2005-02-28", "2007-06-30", "2010-08-15")
        result = reconstruct_baskets(current, events, snaps)
        assert "DELL.OQ" in result[pd.Timestamp("2005-02-28")]
        assert "DELL.OQ" not in result[pd.Timestamp("2007-06-30")]
        assert "DELL.OQ" in result[pd.Timestamp("2010-08-15")]

    def test_invalid_change_value_raises(self):
        current = frozenset(["A"])
        events = _events_typed(("2020-01-01", "A", "merger"))
        with pytest.raises(ValueError, match="change"):
            reconstruct_baskets(current, events, _snap("2019-12-31"))

    def test_null_change_value_raises(self):
        """A null in 'change' must fail loudly — silently treating it as a
        leave would corrupt the reconstruction in the opposite direction."""
        current = frozenset(["A"])
        events = pd.DataFrame({
            "date": [pd.Timestamp("2020-01-01")],
            "ric": ["A"],
            "change": [None],
        })
        with pytest.raises(ValueError, match="change"):
            reconstruct_baskets(current, events, _snap("2019-12-31"))

    def test_same_day_join_leave_pair_is_net_noop_flash_entry(self):
        """EVHC case: flash entry — join+leave on the SAME day, not in current.

        Under basket_at(T) = close-of-T the pair is a net no-op whatever the
        intraday order: the RIC must be absent both before and at D.  Without
        pair cancellation the backward walk would resolve it by API row
        order, producing a phantom member in every snapshot before D.
        """
        current = frozenset(["AAPL.OQ"])
        for order in [("join", "leave"), ("leave", "join")]:
            events = _events_typed(
                ("2016-12-02", "EVHC.N^L16", order[0]),
                ("2016-12-02", "EVHC.N^L16", order[1]),
            )
            snaps = _snap("2003-12-31", "2016-12-01", "2016-12-02", "2017-12-31")
            result = reconstruct_baskets(current, events, snaps)
            for s in snaps:
                assert "EVHC.N^L16" not in result[s], (
                    f"phantom member at {s} with row order {order}"
                )

    def test_same_day_pair_member_in_current_stays_member(self):
        """PEG case: same-day pair for a long-time member in the current
        basket — must remain present before and at D, in either row order."""
        for order in [("join", "leave"), ("leave", "join")]:
            current = frozenset(["PEG.N"])
            events = _events_typed(
                ("2003-11-28", "PEG.N", order[0]),
                ("2003-11-28", "PEG.N", order[1]),
            )
            snaps = _snap("1998-06-30", "2003-11-28", "2010-12-31")
            result = reconstruct_baskets(current, events, snaps)
            for s in snaps:
                assert "PEG.N" in result[s], (
                    f"member lost at {s} with row order {order}"
                )

    def test_unbalanced_same_day_events_keep_net_direction(self):
        """2 joins + 1 leave on the same day → net one join survives."""
        current = frozenset(["X.N"])
        events = _events_typed(
            ("2010-05-03", "X.N", "join"),
            ("2010-05-03", "X.N", "leave"),
            ("2010-05-03", "X.N", "join"),
        )
        result = reconstruct_baskets(current, events, _snap("2010-05-02", "2010-05-03"))
        assert "X.N" not in result[pd.Timestamp("2010-05-02")]
        assert "X.N" in result[pd.Timestamp("2010-05-03")]

    def test_same_day_pair_does_not_affect_other_events(self):
        """Pair cancellation must not touch events of other RICs or other days."""
        current = frozenset(["KEEP.N"])
        events = _events_typed(
            ("2016-12-02", "EVHC.N^L16", "join"),
            ("2016-12-02", "EVHC.N^L16", "leave"),
            ("2016-12-02", "KEEP.N", "join"),
        )
        result = reconstruct_baskets(current, events, _snap("2016-12-01", "2016-12-02"))
        assert "KEEP.N" not in result[pd.Timestamp("2016-12-01")]
        assert "KEEP.N" in result[pd.Timestamp("2016-12-02")]
        assert "EVHC.N^L16" not in result[pd.Timestamp("2016-12-01")]

    def test_no_change_column_falls_back_to_toggle(self):
        """Without 'change' the toggle behaviour is preserved (rejoin case)."""
        current = frozenset(["DELL.OQ"])
        events = _events(("2005-03-01", "DELL.OQ"), ("2010-08-15", "DELL.OQ"))
        snaps = _snap("2005-02-28", "2007-06-30")
        result = reconstruct_baskets(current, events, snaps)
        assert "DELL.OQ" in result[pd.Timestamp("2005-02-28")]
        assert "DELL.OQ" not in result[pd.Timestamp("2007-06-30")]


# ---------------------------------------------------------------------------
# ConstituentHistory — thin plumbing tests (mocked network)
# ---------------------------------------------------------------------------


class TestConstituentHistoryPlumbing:
    """Verify that ConstituentHistory wires the Refinitiv calls correctly.

    These tests mock the network layer and assert:
    1. The correct Refinitiv API calls are made.
    2. The resulting baskets are constructed from the mocked data.
    3. The cache is populated after the first fetch.
    """

    def _make_history(self, tmp_path):
        from jeanclaude.data.ingestion.refinitiv import RefinitivSource
        from jeanclaude.data.storage.parquet_store import ParquetStore

        source = MagicMock(spec=RefinitivSource)
        source._ensure_session = MagicMock()
        store = ParquetStore(tmp_path)
        return ConstituentHistory(source=source, store=store)

    def test_monthly_baskets_calls_fetch_and_returns_dict(self, tmp_path):
        """monthly_baskets returns a dict keyed by pd.Timestamp."""
        ch = self._make_history(tmp_path)

        # Patch the private fetch helpers
        fake_basket = ["AAPL.OQ", "MSFT.OQ", "GOOG.OQ"]
        fake_events = pd.DataFrame({
            "date": [pd.Timestamp("2020-06-01")],
            "ric": ["AAPL.OQ"],
        })

        with (
            patch.object(ch, "_fetch_current_basket_live", return_value=fake_basket),
            patch.object(ch, "_fetch_jl_events_live", return_value=fake_events),
        ):
            result = ch.monthly_baskets("2021-01", "2021-03")

        # Should have 3 month-end snapshots: Jan, Feb, Mar 2021
        assert len(result) == 3
        expected_dates = pd.date_range("2021-01", "2021-03", freq="ME")
        for d in expected_dates:
            assert d in result
            assert isinstance(result[d], frozenset)

    def test_cache_used_on_second_call(self, tmp_path):
        """Second call should not call _fetch_*_live again (cache hit)."""
        ch = self._make_history(tmp_path)

        fake_basket = ["AAPL.OQ"]
        fake_events = pd.DataFrame({
            "date": pd.Series([], dtype="datetime64[ns]"),
            "ric": pd.Series([], dtype=str),
        })

        live_basket_calls = []
        live_events_calls = []

        def basket_side():
            live_basket_calls.append(1)
            return fake_basket

        def events_side():
            live_events_calls.append(1)
            return fake_events

        with (
            patch.object(ch, "_fetch_current_basket_live", side_effect=basket_side),
            patch.object(ch, "_fetch_jl_events_live", side_effect=events_side),
        ):
            ch.monthly_baskets("2022-01", "2022-01")
            ch.monthly_baskets("2022-01", "2022-01")  # second call

        assert len(live_basket_calls) == 1, "Expected cache hit on second call"
        assert len(live_events_calls) == 1, "Expected cache hit on second call"

    def test_basket_reconstruction_matches_pure_algorithm(self, tmp_path):
        """Results are identical to calling reconstruct_baskets directly."""
        ch = self._make_history(tmp_path)

        current = ["AAPL.OQ", "MSFT.OQ"]
        events = pd.DataFrame({
            "date": [pd.Timestamp("2020-06-01")],
            "ric": ["AAPL.OQ"],
        })

        with (
            patch.object(ch, "_fetch_current_basket_live", return_value=current),
            patch.object(ch, "_fetch_jl_events_live", return_value=events),
        ):
            result = ch.monthly_baskets("2020-04", "2020-07")

        # Directly call the pure function for comparison
        snap_dates = list(pd.date_range("2020-04", "2020-07", freq="ME"))
        expected = reconstruct_baskets(frozenset(current), events, snap_dates)

        for d in snap_dates:
            assert result[d] == expected[d], f"Mismatch at {d}"


class TestFetchJLEventsLive:
    """_fetch_jl_events_live must request IC=B + the .change field and
    normalize the Change column to {'join', 'leave'}."""

    def _make_history(self, tmp_path):
        from jeanclaude.data.ingestion.refinitiv import RefinitivSource
        from jeanclaude.data.storage.parquet_store import ParquetStore

        source = MagicMock(spec=RefinitivSource)
        source._ensure_session = MagicMock()
        store = ParquetStore(tmp_path)
        return ConstituentHistory(source=source, store=store)

    def _fake_api_response(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Instrument": [".SPX"] * 3,
            "Date": ["2020-12-21", "2008-09-17", "2009-06-03"],
            "Change": ["Joiner", "Leaver", "Leaver"],
            "Constituent RIC": ["TSLA.OQ", "LEH.N^I08", "GM.N^F09"],
            "Constituent Name": ["Tesla Inc", "Lehman Brothers", "General Motors"],
        })

    def test_requests_ic_both_and_change_field(self, tmp_path):
        ch = self._make_history(tmp_path)
        fake_ld = MagicMock()
        fake_ld.get_data.return_value = self._fake_api_response()

        with patch("jeanclaude.data.ingestion.refinitiv.ld", fake_ld):
            ch._fetch_jl_events_live()

        kwargs = fake_ld.get_data.call_args.kwargs
        assert kwargs["parameters"]["IC"] == "B"
        assert "TR.IndexJLConstituentRIC.change" in kwargs["fields"]

    def test_normalizes_change_column(self, tmp_path):
        ch = self._make_history(tmp_path)
        fake_ld = MagicMock()
        fake_ld.get_data.return_value = self._fake_api_response()

        with patch("jeanclaude.data.ingestion.refinitiv.ld", fake_ld):
            events = ch._fetch_jl_events_live()

        assert set(events.columns) >= {"date", "ric", "change"}
        assert events["change"].isin(["join", "leave"]).all()
        tsla = events[events["ric"] == "TSLA.OQ"]
        assert len(tsla) == 1
        assert tsla.iloc[0]["change"] == "join"

    def test_unknown_change_value_raises(self, tmp_path):
        ch = self._make_history(tmp_path)
        bad = self._fake_api_response()
        bad.loc[0, "Change"] = "Merger"
        fake_ld = MagicMock()
        fake_ld.get_data.return_value = bad

        with patch("jeanclaude.data.ingestion.refinitiv.ld", fake_ld):
            with pytest.raises(ValueError, match="[Cc]hange"):
                ch._fetch_jl_events_live()

    def test_cache_roundtrip_preserves_change_column(self, tmp_path):
        """Events saved to the Parquet cache must come back with 'change'."""
        ch = self._make_history(tmp_path)
        fake_events = pd.DataFrame({
            "date": [pd.Timestamp("2020-12-21"), pd.Timestamp("2008-09-17")],
            "ric": ["TSLA.OQ", "LEH.N^I08"],
            "change": ["join", "leave"],
        })

        calls = []

        def events_side():
            calls.append(1)
            return fake_events

        with patch.object(ch, "_fetch_jl_events_live", side_effect=events_side):
            first = ch._get_or_fetch_jl_events("2026-06-12")
            second = ch._get_or_fetch_jl_events("2026-06-12")  # cache hit

        assert len(calls) == 1, "Expected cache hit on second call"
        for df in (first, second):
            assert "change" in df.columns
            assert df["change"].isin(["join", "leave"]).all()
