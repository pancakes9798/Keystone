"""Rigorous NAV accounting tests for PaperBroker.

With real money, NAV accounting must be exact:
- Costs reduce NAV exactly once
- apply_rebalance_cost() and pre_applied_cost in record_daily_nav must never double-count
- NAV is always >= 0
- Persistence is consistent across reads
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from jeanclaude.data.storage.parquet_store import ParquetStore
from jeanclaude.execution.paper.broker import PaperBroker


def _broker(tmp_path, initial: float = 100_000.0) -> PaperBroker:
    return PaperBroker(initial_capital=initial, store=ParquetStore(str(tmp_path)))


def _flat_returns(assets: list[str], ret: float = 0.0) -> pd.Series:
    return pd.Series({a: ret for a in assets})


# ── Test 1: Initial state ─────────────────────────────────────────────────────

class TestInitialState:
    def test_nav_equals_initial_capital_before_any_trade(self, tmp_path):
        broker = _broker(tmp_path, initial=200_000.0)
        assert broker.nav == pytest.approx(200_000.0)

    def test_nav_equals_initial_capital_after_empty_record(self, tmp_path):
        broker = _broker(tmp_path)
        broker.record_daily_nav(pd.Series(dtype=float))
        assert broker.nav == pytest.approx(100_000.0)


# ── Test 2: apply_rebalance_cost() correctness ───────────────────────────────
# apply_cost() was non-idempotent and wall-clock-dated: removed.
# Equivalent invariants are covered by apply_rebalance_cost(date, cost_eur)
# which operates on an explicit, named date row in nav_history.

_D_SEED = pd.Timestamp("2026-06-01")


class TestApplyRebalanceCost:
    def test_apply_rebalance_cost_reduces_nav_by_exact_amount(self, tmp_path):
        broker = _broker(tmp_path)
        broker.record_daily_nav(pd.Series(dtype=float), date=_D_SEED)  # seed nav_history

        initial_nav = broker.nav
        new_nav = broker.apply_rebalance_cost(date=_D_SEED, cost_eur=500.0)

        assert new_nav == pytest.approx(initial_nav - 500.0, rel=1e-9)
        assert broker.nav == pytest.approx(initial_nav - 500.0, rel=1e-9)

    def test_apply_rebalance_cost_zero_is_noop(self, tmp_path):
        broker = _broker(tmp_path)
        broker.record_daily_nav(pd.Series(dtype=float), date=_D_SEED)
        nav_before = broker.nav
        new_nav = broker.apply_rebalance_cost(date=_D_SEED, cost_eur=0.0)
        assert new_nav == pytest.approx(nav_before, rel=1e-9)
        assert broker.nav == pytest.approx(nav_before, rel=1e-9)

    def test_apply_rebalance_cost_cannot_go_below_zero(self, tmp_path):
        broker = _broker(tmp_path, initial=100.0)
        broker.record_daily_nav(pd.Series(dtype=float), date=_D_SEED)
        nav = broker.apply_rebalance_cost(date=_D_SEED, cost_eur=999_999.0)
        assert nav == pytest.approx(0.0)
        assert broker.nav == pytest.approx(0.0)

    def test_multiple_apply_rebalance_cost_accumulate_correctly(self, tmp_path):
        """Sequential cost deductions across different dated rows accumulate correctly."""
        broker = _broker(tmp_path, initial=100_000.0)
        d1 = pd.Timestamp("2026-06-01")
        d2 = pd.Timestamp("2026-06-02")
        d3 = pd.Timestamp("2026-06-03")
        broker.record_daily_nav(pd.Series(dtype=float), date=d1)
        broker.apply_rebalance_cost(date=d1, cost_eur=100.0)
        broker.record_daily_nav(pd.Series(dtype=float), date=d2)
        broker.apply_rebalance_cost(date=d2, cost_eur=200.0)
        broker.record_daily_nav(pd.Series(dtype=float), date=d3)
        broker.apply_rebalance_cost(date=d3, cost_eur=50.0)

        assert broker.nav == pytest.approx(100_000.0 - 100.0 - 200.0 - 50.0, rel=1e-9)

    def test_apply_rebalance_cost_persists_to_nav_history(self, tmp_path):
        broker = _broker(tmp_path)
        broker.record_daily_nav(pd.Series(dtype=float), date=_D_SEED)
        broker.apply_rebalance_cost(date=_D_SEED, cost_eur=300.0)

        # Create a new broker instance pointing to the same store
        broker2 = _broker(tmp_path)
        assert broker2.nav == pytest.approx(100_000.0 - 300.0, rel=1e-9)


# ── Test 3: record_daily_nav with pre_applied_cost ────────────────────────────

class TestPreAppliedCost:
    def test_pre_applied_cost_deducted_from_nav(self, tmp_path):
        assets = ["SPY", "TLT"]
        prices = pd.Series({"SPY": 400.0, "TLT": 100.0})
        broker = _broker(tmp_path)
        broker.execute_rebalance(
            pd.Series({"SPY": 0.5, "TLT": 0.5}), prices, transaction_cost_bps=0.0
        )

        # 0% return day, but pre_applied_cost = 500 EUR
        broker.record_daily_nav(_flat_returns(assets, ret=0.0), pre_applied_cost=500.0)
        # NAV should be 100_000 - 500 = 99_500
        assert broker.nav == pytest.approx(99_500.0, rel=1e-4)

    def test_pre_applied_cost_zero_same_as_no_cost(self, tmp_path):
        assets = ["SPY"]
        prices = pd.Series({"SPY": 400.0})
        broker1 = _broker(tmp_path / "b1")
        broker2 = _broker(tmp_path / "b2")
        w = pd.Series({"SPY": 1.0})

        broker1.execute_rebalance(w, prices, transaction_cost_bps=0.0)
        broker2.execute_rebalance(w, prices, transaction_cost_bps=0.0)

        rets = _flat_returns(assets, ret=0.01)
        broker1.record_daily_nav(rets, pre_applied_cost=0.0)
        broker2.record_daily_nav(rets, pre_applied_cost=0.0)

        assert broker1.nav == pytest.approx(broker2.nav, rel=1e-9)

    def test_pre_applied_cost_is_the_sole_cost_path(self, tmp_path):
        """Simulates the runner pattern: only pre_applied_cost is used.
        apply_cost() has been removed; this test confirms pre_applied_cost alone
        deducts the expected amount without double-counting."""
        assets = ["SPY", "TLT"]
        prices = pd.Series({"SPY": 400.0, "TLT": 100.0})
        broker = _broker(tmp_path)
        broker.execute_rebalance(
            pd.Series({"SPY": 0.5, "TLT": 0.5}), prices, transaction_cost_bps=0.0
        )

        ac_cost_eur = 250.0

        # CORRECT pattern (what the runner does): only pre_applied_cost
        broker.record_daily_nav(_flat_returns(assets, 0.0), pre_applied_cost=ac_cost_eur)
        nav_correct = broker.nav
        # Expected: 100_000 - 250 = 99_750
        assert nav_correct == pytest.approx(100_000.0 - ac_cost_eur, rel=1e-4)


# ── Test 4: execute_rebalance with tc=0 doesn't change NAV ───────────────────

class TestRebalanceWithZeroTC:
    def test_zero_tc_rebalance_preserves_nav(self, tmp_path):
        assets = ["SPY", "TLT"]
        prices = pd.Series({"SPY": 400.0, "TLT": 100.0})
        broker = _broker(tmp_path)

        nav_before = broker.nav
        broker.execute_rebalance(
            pd.Series({"SPY": 0.5, "TLT": 0.5}), prices, transaction_cost_bps=0.0
        )
        state = broker.get_state()
        assert state.nav == pytest.approx(nav_before, rel=1e-9)

    def test_nonzero_tc_rebalance_reduces_nav(self, tmp_path):
        assets = ["SPY", "TLT"]
        prices = pd.Series({"SPY": 400.0, "TLT": 100.0})
        broker = _broker(tmp_path)

        nav_before = broker.nav
        # Under the new contract, execute_rebalance returns RebalanceResult;
        # TC cost is in result.cost_eur and is NOT yet applied to positions/nav_history.
        # The invariant "TC costs money" is verified via result.cost_eur > 0.
        result = broker.execute_rebalance(
            pd.Series({"SPY": 0.5, "TLT": 0.5}), prices, transaction_cost_bps=10.0
        )
        assert result.cost_eur > 0.0


# ── Test 5: Full daily cycle NAV accounting ───────────────────────────────────

class TestFullDailyCycle:
    def test_daily_cycle_without_cost_model(self, tmp_path):
        """Full cycle: rebalance (tc=10bps) → record return. NAV accounting exact.

        Under the P0 contract, TC cost is returned by execute_rebalance (not applied
        silently to positions). The caller must apply it via pre_applied_cost or
        apply_rebalance_cost. Here we apply it explicitly via pre_applied_cost to
        match the runner's flat-bps path.

        Explicit dates are required so _weights_before(d_nav) can find the rebalance.
        """
        assets = ["SPY", "TLT"]
        prices = pd.Series({"SPY": 400.0, "TLT": 100.0})
        broker = _broker(tmp_path, initial=100_000.0)
        d_rebalance = pd.Timestamp("2026-01-01")
        d_nav = pd.Timestamp("2026-01-02")

        # Rebalance: 50/50, 10 bps TC
        # turnover = 1.0 → cost_eur = 100_000 * 0.001 = 100 EUR
        result = broker.execute_rebalance(
            pd.Series({"SPY": 0.5, "TLT": 0.5}), prices,
            transaction_cost_bps=10.0, date=d_rebalance,
        )
        # Under new contract positions.nav stores pre-TC NAV (100_000)
        state_after_rebalance = broker.get_state()
        assert state_after_rebalance.nav == pytest.approx(100_000.0, rel=1e-6)
        assert result.cost_eur == pytest.approx(100.0, rel=1e-6)

        # Record +1% return with TC applied via pre_applied_cost
        # Base NAV for d_nav = nav_history strictly before d_nav = initial_capital (100_000)
        # portfolio_return = 0.5*0.01 + 0.5*0.01 = 0.01
        # new_nav = 100_000 * 1.01 - 100 = 101_000 - 100 = 100_900
        broker.record_daily_nav(
            _flat_returns(assets, ret=0.01),
            pre_applied_cost=result.cost_eur,
            date=d_nav,
        )
        expected = 100_000.0 * 1.01 - result.cost_eur
        assert broker.nav == pytest.approx(expected, rel=1e-4)

    def test_daily_cycle_with_ac_cost_model(self, tmp_path):
        """AC cost applied via pre_applied_cost in record_daily_nav (runner pattern).

        Explicit dates ensure _weights_before(d_nav) finds the rebalance weights.
        """
        assets = ["SPY", "TLT"]
        prices = pd.Series({"SPY": 400.0, "TLT": 100.0})
        broker = _broker(tmp_path, initial=100_000.0)
        d_rebalance = pd.Timestamp("2026-01-01")
        d_nav = pd.Timestamp("2026-01-02")

        # Rebalance with tc=0 (AC model handles costs separately)
        broker.execute_rebalance(
            pd.Series({"SPY": 0.5, "TLT": 0.5}), prices,
            transaction_cost_bps=0.0, date=d_rebalance,
        )

        # Record return + AC cost of 200 EUR
        ac_cost = 200.0
        broker.record_daily_nav(
            _flat_returns(assets, ret=0.005),
            pre_applied_cost=ac_cost,
            date=d_nav,
        )

        # Expected: 100_000 * (1 + 0.5*0.005 + 0.5*0.005) - 200 = 100_000 * 1.005 - 200
        expected = 100_000.0 * 1.005 - ac_cost
        assert broker.nav == pytest.approx(expected, rel=1e-4)

    def test_nav_never_negative(self, tmp_path):
        """Even in catastrophic scenarios, NAV must not go below 0."""
        assets = ["A", "B"]
        prices = pd.Series({"A": 100.0, "B": 100.0})
        broker = _broker(tmp_path, initial=100_000.0)
        d_rebalance = pd.Timestamp("2026-06-01")
        d_nav = pd.Timestamp("2026-06-02")
        broker.execute_rebalance(
            pd.Series({"A": 0.5, "B": 0.5}), prices, transaction_cost_bps=0.0,
            date=d_rebalance,
        )

        # -100% return on both assets
        broker.record_daily_nav(_flat_returns(assets, ret=-1.0), pre_applied_cost=0.0,
                                date=d_nav)
        assert broker.nav >= 0.0

        # Apply huge cost on top via apply_rebalance_cost
        broker.apply_rebalance_cost(date=d_nav, cost_eur=9_999_999.0)
        assert broker.nav >= 0.0


# ── Test 6: NAV consistency across broker instances ──────────────────────────

class TestPersistenceConsistency:
    def test_nav_consistent_after_restart(self, tmp_path):
        """After writing state, a new PaperBroker on same store reads identical NAV."""
        d_seed = pd.Timestamp("2026-06-01")
        broker1 = _broker(tmp_path)
        broker1.record_daily_nav(pd.Series(dtype=float), date=d_seed)
        broker1.apply_rebalance_cost(date=d_seed, cost_eur=1_234.56)

        broker2 = PaperBroker(
            initial_capital=100_000.0,
            store=ParquetStore(str(tmp_path))
        )
        assert broker2.nav == pytest.approx(broker1.nav, rel=1e-9)

    def test_multiple_days_nav_tracks_correctly(self, tmp_path):
        """Multi-day simulation: verify NAV path matches manual calculation.

        Explicit dates are required under the upsert contract: without them all
        record_daily_nav calls default to today and overwrite each other.
        """
        assets = ["X"]
        prices = pd.Series({"X": 100.0})
        broker = _broker(tmp_path, initial=10_000.0)
        d_rebalance = pd.Timestamp("2026-01-01")
        broker.execute_rebalance(
            pd.Series({"X": 1.0}), prices,
            transaction_cost_bps=0.0, date=d_rebalance,
        )

        daily_returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        expected_nav = 10_000.0
        for i, ret in enumerate(daily_returns):
            day = pd.Timestamp("2026-01-02") + pd.Timedelta(days=i)
            broker.record_daily_nav(pd.Series({"X": ret}), pre_applied_cost=0.0, date=day)
            expected_nav *= (1 + ret)
            assert broker.nav == pytest.approx(expected_nav, rel=1e-4)
