"""No-double-counting and sign-correctness tests for DailyRunner + cost model.

The runner applies AC costs exactly once per rebalance:
- execute_rebalance receives transaction_cost_bps=0.0 (no flat bps cost)
- record_daily_nav receives pre_applied_cost=ac_cost_eur

These tests verify that constraint is never violated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, call, patch

from jeanclaude.costs import AlmgrenChrissModel, CostConfig
from jeanclaude.costs.data import VolumeSpreadLoader
from jeanclaude.data.storage.parquet_store import ParquetStore
from jeanclaude.execution.paper.broker import PaperBroker
from jeanclaude.execution.paper.runner import DailyRunner, DailyRunConfig
from jeanclaude.execution.paper.stubs import (
    NoopRegimeDetector, NoopRiskFilter, NoopReportBuilder,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

TICKERS = ["AAPL.O", "TLT.O"]


def _make_prices(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    data = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, (n, 2)), axis=0))
    return pd.DataFrame(data, index=idx, columns=TICKERS)


def _mock_data_loader(prices: pd.DataFrame) -> MagicMock:
    dl = MagicMock()
    dl.get_prices.return_value = prices
    dl.get_returns.return_value = prices.pct_change().dropna()
    dl.get_macro.return_value = pd.DataFrame()
    return dl


def _mock_vsl(tickers: list[str] = TICKERS) -> MagicMock:
    vsl = MagicMock(spec=VolumeSpreadLoader)
    vsl.get_adv.return_value    = pd.Series({t: 1_000_000.0 for t in tickers})
    vsl.get_spread.return_value = pd.Series({t: 0.001 for t in tickers})
    return vsl


def _make_runner(
    broker: PaperBroker,
    prices: pd.DataFrame,
    cost_model=None,
    vsl=None,
    tc_bps: float = 10.0,
) -> DailyRunner:
    opt = MagicMock()
    opt.optimize.return_value = pd.Series({t: 1.0 / len(TICKERS) for t in TICKERS})

    config = DailyRunConfig(
        assets=TICKERS,
        macro_series=[],
        rebalance_freq="ME",
        send_report=False,
        transaction_cost_bps=tc_bps,
        min_history=60,
        history_start="2024-01-02",
    )
    return DailyRunner(
        broker=broker,
        optimizer=opt,
        risk_filter=NoopRiskFilter(),
        regime_detector=NoopRegimeDetector(),
        data_loader=_mock_data_loader(prices),
        report_builder=NoopReportBuilder(),
        mailer=None,
        config=config,
        cost_model=cost_model,
        volume_spread_loader=vsl,
    )


# ── Test 1: No double-counting on rebalance days ──────────────────────────────

class TestNoDoubleCountOnRebalance:
    def test_execute_rebalance_receives_real_tc_when_ac_model_active(self, tmp_path):
        """P0 2026-06-10: con il nuovo contratto, execute_rebalance riceve il tc_bps reale
        (non 0.0) e il costo AC viene applicato separatamente via apply_rebalance_cost.
        Questo evita che un failure dell'AC model addebitasse zero invece del flat bps."""
        broker = PaperBroker(100_000.0, ParquetStore(str(tmp_path)))
        prices = _make_prices()
        cost_model = AlmgrenChrissModel(CostConfig())
        vsl = _mock_vsl()
        runner = _make_runner(broker, prices, cost_model=cost_model, vsl=vsl)

        # Prima allocazione (consuma last_rebalance=None trigger)
        runner.run(pd.Timestamp("2024-01-02"))
        # Rebalance date: primo run con periodo completato e ≥60 ritorni
        rebalance_date = pd.Timestamp("2024-07-31")

        with patch.object(broker, "execute_rebalance", wraps=broker.execute_rebalance) as mock_exec:
            result = runner.run(rebalance_date)

        if result.rebalanced:
            assert mock_exec.called, "execute_rebalance must be called on rebalance day"
            args, kwargs = mock_exec.call_args
            tc_actual = kwargs.get("transaction_cost_bps", args[2] if len(args) > 2 else None)
            # Nuova semantica: tc_bps reale passato a execute_rebalance (non 0.0)
            assert tc_actual == pytest.approx(10.0), (
                f"execute_rebalance deve ricevere il tc_bps reale ({tc_actual}), "
                "non 0.0 — P0 2026-06-10: il fallback AC failure addebitava 0"
            )

    def test_record_daily_nav_receives_no_pre_applied_cost(self, tmp_path):
        """P0 2026-06-10: nel nuovo contratto record_daily_nav NON riceve pre_applied_cost.
        Il costo è applicato dopo via apply_rebalance_cost (separazione record→cost)."""
        broker = PaperBroker(100_000.0, ParquetStore(str(tmp_path)))
        prices = _make_prices()
        cost_model = AlmgrenChrissModel(CostConfig())
        vsl = _mock_vsl()
        runner = _make_runner(broker, prices, cost_model=cost_model, vsl=vsl)

        rebalance_date = pd.Timestamp("2024-07-31")

        with patch.object(broker, "record_daily_nav", wraps=broker.record_daily_nav) as mock_rec:
            result = runner.run(rebalance_date)

        assert mock_rec.called, "record_daily_nav must always be called"
        args, kwargs = mock_rec.call_args
        pre_applied = kwargs.get("pre_applied_cost", args[1] if len(args) > 1 else 0.0)
        # Nuovo contratto: record_daily_nav riceve pre_applied_cost=0 (default);
        # il costo transazionale è applicato separatamente via apply_rebalance_cost
        assert pre_applied == pytest.approx(0.0), (
            "record_daily_nav non deve ricevere pre_applied_cost — "
            "P0 2026-06-10: il costo è applicato via apply_rebalance_cost"
        )

    def test_cost_applied_via_apply_rebalance_cost_not_pre_applied(self, tmp_path):
        """P0 2026-06-10: il costo deve fluire via apply_rebalance_cost (non pre_applied_cost).
        Il contratto è: record_daily_nav PRIMA (P&L sui pesi vecchi), poi execute_rebalance,
        poi apply_rebalance_cost per il costo reale (flat o AC)."""
        broker = PaperBroker(100_000.0, ParquetStore(str(tmp_path)))
        prices = _make_prices()
        cost_model = AlmgrenChrissModel(CostConfig())
        vsl = _mock_vsl()
        runner = _make_runner(broker, prices, cost_model=cost_model, vsl=vsl)

        # Prima allocazione (consuma last_rebalance=None trigger)
        runner.run(pd.Timestamp("2024-01-02"))

        with patch.object(broker, "apply_rebalance_cost", wraps=broker.apply_rebalance_cost) as mock_arc:
            result = runner.run(pd.Timestamp("2024-07-31"))

        assert result.rebalanced, "il run doveva ribilanciare"
        # Nuovo contratto: apply_rebalance_cost DEVE essere chiamato con costo > 0
        assert mock_arc.called, (
            "broker.apply_rebalance_cost() deve essere chiamato dal runner — "
            "P0 2026-06-10: è qui che il costo viene addebitato al NAV"
        )

    def test_nav_decreases_after_rebalance_with_ac_model(self, tmp_path):
        """After a rebalance day, NAV should be lower than without costs (all else equal)."""
        prices = _make_prices()
        rebalance_date = pd.Timestamp("2024-07-31")

        broker_no_cost = PaperBroker(100_000.0, ParquetStore(str(tmp_path / "no_cost")))
        runner_no_cost = _make_runner(broker_no_cost, prices, cost_model=None, tc_bps=0.0)
        runner_no_cost.run(rebalance_date)
        nav_no_cost = broker_no_cost.nav

        broker_ac = PaperBroker(100_000.0, ParquetStore(str(tmp_path / "ac")))
        cost_model = AlmgrenChrissModel(CostConfig())
        vsl = _mock_vsl()
        runner_ac = _make_runner(broker_ac, prices, cost_model=cost_model, vsl=vsl, tc_bps=0.0)
        result_ac = runner_ac.run(rebalance_date)
        nav_ac = broker_ac.nav

        if result_ac.rebalanced:
            assert nav_ac <= nav_no_cost, (
                "NAV with AC costs must be <= NAV without costs on rebalance day"
            )


# ── Test 2: No cost on non-rebalance days ─────────────────────────────────────

class TestNoCostOnNonRebalanceDays:
    def test_no_cost_on_non_rebalance_day(self, tmp_path):
        """P0 2026-06-10: su un giorno di non-rebalance, apply_rebalance_cost non deve
        essere chiamato e record_daily_nav non riceve pre_applied_cost.
        Strategia: rebalance su Jul 31 (consuma tutti i month-end mancati), poi Aug 15
        è mid-agosto → nessun periodo completato → non-rebalance day."""
        broker = PaperBroker(100_000.0, ParquetStore(str(tmp_path)))
        prices = _make_prices()
        cost_model = AlmgrenChrissModel(CostConfig())
        vsl = _mock_vsl()
        runner = _make_runner(broker, prices, cost_model=cost_model, vsl=vsl)

        # Rebalance su Jul 31 (consuma tutti i periodi mancati da Jan)
        runner.run(pd.Timestamp("2024-07-31"))
        vsl.get_adv.reset_mock()
        vsl.get_spread.reset_mock()

        # Aug 15: nessun periodo completato da ribilanciare dopo Jul 31
        non_rebalance_date = pd.Timestamp("2024-08-15")

        with patch.object(broker, "record_daily_nav", wraps=broker.record_daily_nav) as mock_rec:
            with patch.object(broker, "apply_rebalance_cost", wraps=broker.apply_rebalance_cost) as mock_arc:
                result = runner.run(non_rebalance_date)

        assert not result.rebalanced, "2024-08-15 should not be a rebalance day after Jul-31"
        assert not mock_arc.called, "apply_rebalance_cost must not be called on non-rebalance days"
        args, kwargs = mock_rec.call_args
        pre_applied = kwargs.get("pre_applied_cost", args[1] if len(args) > 1 else 0.0)
        assert pre_applied == pytest.approx(0.0), "pre_applied_cost must be 0 on non-rebalance days"

    def test_vsl_not_called_on_non_rebalance_day(self, tmp_path):
        """VolumeSpreadLoader should not be queried on non-rebalance days.
        P0 2026-06-10: rebalance su Jul 31 (catch-up), poi Aug 15 è mid-periodo."""
        broker = PaperBroker(100_000.0, ParquetStore(str(tmp_path)))
        prices = _make_prices()
        cost_model = AlmgrenChrissModel(CostConfig())
        vsl = _mock_vsl()
        runner = _make_runner(broker, prices, cost_model=cost_model, vsl=vsl)

        # Rebalance su Jul 31 (consuma tutti i periodi mancati da Jan)
        runner.run(pd.Timestamp("2024-07-31"))
        vsl.get_adv.reset_mock()
        vsl.get_spread.reset_mock()

        runner.run(pd.Timestamp("2024-08-15"))

        vsl.get_adv.assert_not_called()
        vsl.get_spread.assert_not_called()


# ── Test 3: Backward compatibility ───────────────────────────────────────────

class TestBackwardCompatibility:
    def test_runner_without_cost_model_uses_tc_bps(self, tmp_path):
        """Runner without cost_model must use transaction_cost_bps as before."""
        broker_tc = PaperBroker(100_000.0, ParquetStore(str(tmp_path / "tc")))
        broker_no = PaperBroker(100_000.0, ParquetStore(str(tmp_path / "no")))
        prices = _make_prices()
        rebalance_date = pd.Timestamp("2024-07-31")

        runner_tc = _make_runner(broker_tc, prices, cost_model=None, tc_bps=10.0)
        runner_no = _make_runner(broker_no, prices, cost_model=None, tc_bps=0.0)

        result_tc = runner_tc.run(rebalance_date)
        result_no = runner_no.run(rebalance_date)

        if result_tc.rebalanced and result_no.rebalanced:
            assert broker_tc.nav < broker_no.nav, (
                "With 10 bps TC, NAV must be lower than with 0 bps TC"
            )

    def test_runner_without_cost_model_works_without_vsl(self, tmp_path):
        """No VolumeSpreadLoader needed when cost_model is None."""
        broker = PaperBroker(100_000.0, ParquetStore(str(tmp_path)))
        prices = _make_prices()
        runner = _make_runner(broker, prices, cost_model=None, vsl=None, tc_bps=5.0)
        # Should not raise even without vsl
        result = runner.run(pd.Timestamp("2024-07-15"))
        assert result is not None

    def test_cost_model_without_vsl_falls_back_to_tc_bps(self, tmp_path):
        """cost_model present but vsl=None → falls back to tc_bps (not AC formula)."""
        broker_ac_no_vsl = PaperBroker(100_000.0, ParquetStore(str(tmp_path / "acnv")))
        broker_flat     = PaperBroker(100_000.0, ParquetStore(str(tmp_path / "flat")))
        prices = _make_prices()
        cost_model = AlmgrenChrissModel(CostConfig())
        rebalance_date = pd.Timestamp("2024-07-31")

        runner_ac_no_vsl = _make_runner(
            broker_ac_no_vsl, prices, cost_model=cost_model, vsl=None, tc_bps=10.0
        )
        runner_flat = _make_runner(broker_flat, prices, cost_model=None, vsl=None, tc_bps=10.0)

        r_ac   = runner_ac_no_vsl.run(rebalance_date)
        r_flat = runner_flat.run(rebalance_date)

        # Both should rebalance; NAVs should be equal (same fallback tc_bps path)
        if r_ac.rebalanced and r_flat.rebalanced:
            assert broker_ac_no_vsl.nav == pytest.approx(broker_flat.nav, rel=1e-6)


# ── Test 4: Multi-day simulation ──────────────────────────────────────────────

class TestMultiDaySimulation:
    def test_multi_month_simulation_nav_non_negative(self, tmp_path):
        """Run 6 months of daily paper trading, NAV must stay non-negative."""
        broker = PaperBroker(100_000.0, ParquetStore(str(tmp_path)))
        prices = _make_prices(n=130)
        cost_model = AlmgrenChrissModel(CostConfig())
        vsl = _mock_vsl()
        runner = _make_runner(broker, prices, cost_model=cost_model, vsl=vsl)

        # Simulate ~6 months of trading days
        trading_days = prices.index[-60:]  # last 60 days in our synthetic history
        for day in trading_days:
            runner.run(day)
            assert broker.nav >= 0.0, f"NAV went negative on {day}"

    def test_costs_reduce_nav_vs_zero_cost_over_multiple_months(self, tmp_path):
        """Over 6 months, AC costs reduce final NAV vs zero-cost simulation."""
        prices = _make_prices(n=130)
        trading_days = prices.index[-60:]

        # With AC costs
        broker_ac = PaperBroker(100_000.0, ParquetStore(str(tmp_path / "ac")))
        cost_model = AlmgrenChrissModel(CostConfig())
        vsl = _mock_vsl()
        runner_ac = _make_runner(broker_ac, prices, cost_model=cost_model, vsl=vsl, tc_bps=0.0)
        for day in trading_days:
            runner_ac.run(day)

        # Without costs
        broker_zero = PaperBroker(100_000.0, ParquetStore(str(tmp_path / "zero")))
        runner_zero = _make_runner(broker_zero, prices, cost_model=None, vsl=None, tc_bps=0.0)
        for day in trading_days:
            runner_zero.run(day)

        assert broker_ac.nav <= broker_zero.nav, (
            "After multiple months, AC costs must not increase NAV vs zero-cost"
        )
