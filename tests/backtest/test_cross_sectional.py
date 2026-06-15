"""Tests for jeanclaude.backtest.cross_sectional — synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jeanclaude.backtest.cross_sectional import (
    align_to_calendar,
    momentum_12_1,
    realized_volatility,
    simulate_monthly,
    top_decile_weights,
)


def _returns_frame(n_days: int, rics: dict[str, float], start="2020-01-01") -> pd.DataFrame:
    """Daily returns: constant per-RIC daily return for n_days business days."""
    idx = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame({ric: [r] * n_days for ric, r in rics.items()}, index=idx)


class TestMomentum121:
    def test_winner_ranks_above_loser(self):
        rets = _returns_frame(300, {"WIN.N": 0.002, "LOSE.N": -0.002, "FLAT.N": 0.0})
        sig = momentum_12_1(rets, asof=rets.index[-1])
        assert sig["WIN.N"] > sig["FLAT.N"] > sig["LOSE.N"]

    def test_skip_window_excluded(self):
        """Un crollo SOLO negli ultimi 21 giorni non tocca il segnale 12-1."""
        rets = _returns_frame(300, {"A.N": 0.001})
        rets.iloc[-21:] = -0.05  # crash nel mese skippato
        sig = momentum_12_1(rets, asof=rets.index[-1])
        expected = (1.001 ** 231) - 1  # solo la finestra (T-252, T-21]
        assert sig["A.N"] == pytest.approx(expected, rel=1e-9)

    def test_insufficient_history_is_nan(self):
        rets = _returns_frame(100, {"NEW.N": 0.001})  # < 200 obs nella finestra
        sig = momentum_12_1(rets, asof=rets.index[-1])
        assert np.isnan(sig["NEW.N"])

    def test_no_lookahead_beyond_asof(self):
        """Dati dopo asof NON devono cambiare il segnale."""
        rets = _returns_frame(300, {"A.N": 0.001})
        asof = rets.index[260]
        sig_a = momentum_12_1(rets, asof=asof)
        rets_mut = rets.copy()
        rets_mut.iloc[261:] = 99.0  # garbage nel futuro
        sig_b = momentum_12_1(rets_mut, asof=asof)
        assert sig_a["A.N"] == pytest.approx(sig_b["A.N"], rel=1e-12)


class TestRealizedVolatility:
    def test_constant_returns_have_zero_vol(self):
        rets = _returns_frame(300, {"FLAT.N": 0.0})
        vol = realized_volatility(rets, asof=rets.index[-1])
        assert vol["FLAT.N"] == pytest.approx(0.0)

    def test_more_volatile_ranks_higher(self):
        idx = pd.bdate_range("2020-01-01", periods=300)
        hi = np.where(np.arange(300) % 2 == 0, 0.03, -0.03)
        lo = np.where(np.arange(300) % 2 == 0, 0.005, -0.005)
        rets = pd.DataFrame({"HI.N": hi, "LO.N": lo}, index=idx)
        vol = realized_volatility(rets, asof=idx[-1])
        assert vol["HI.N"] > vol["LO.N"]

    def test_neg_vol_score_prefers_low_vol(self):
        """Il punteggio usato dalla strategia è -vol: il meno volatile ha score più alto."""
        idx = pd.bdate_range("2020-01-01", periods=300)
        hi = np.where(np.arange(300) % 2 == 0, 0.03, -0.03)
        lo = np.where(np.arange(300) % 2 == 0, 0.005, -0.005)
        rets = pd.DataFrame({"HI.N": hi, "LO.N": lo}, index=idx)
        score = -realized_volatility(rets, asof=idx[-1])
        assert score["LO.N"] > score["HI.N"]

    def test_insufficient_history_is_nan(self):
        rets = _returns_frame(100, {"NEW.N": 0.001})  # < 200 obs
        vol = realized_volatility(rets, asof=rets.index[-1])
        assert np.isnan(vol["NEW.N"])

    def test_no_lookahead_beyond_asof(self):
        rets = _returns_frame(300, {"A.N": 0.001})
        asof = rets.index[260]
        v_a = realized_volatility(rets, asof=asof)
        rets_mut = rets.copy()
        rets_mut.iloc[261:] = 99.0
        v_b = realized_volatility(rets_mut, asof=asof)
        assert v_a["A.N"] == pytest.approx(v_b["A.N"], rel=1e-12)


class TestTopDecileWeights:
    def test_top_decile_equal_weight(self):
        sig = pd.Series({f"R{i}.N": float(i) for i in range(200)})  # R199 il migliore
        w = top_decile_weights(sig, min_names=20)
        assert len(w) == 20  # ceil(200 * 0.10)
        assert w.sum() == pytest.approx(1.0)
        assert set(w.index) == {f"R{i}.N" for i in range(180, 200)}
        assert all(v == pytest.approx(1 / 20) for v in w)

    def test_nan_signals_excluded(self):
        sig = pd.Series({"A.N": 1.0, "B.N": np.nan, "C.N": 0.5})
        w = top_decile_weights(sig, min_names=1)
        assert "B.N" not in w.index

    def test_below_min_names_returns_empty(self):
        sig = pd.Series({f"R{i}.N": float(i) for i in range(50)})  # decile = 5 < 20
        w = top_decile_weights(sig, min_names=20)
        assert w.empty


def _baskets_for(dates: list[str], members: list[str]) -> dict[pd.Timestamp, frozenset[str]]:
    return {pd.Timestamp(d): frozenset(members) for d in dates}


class TestSimulateMonthly:
    def _flat_world(self):
        """26 mesi di dati, 3 RIC: WIN sale, LOSE scende, MID piatto."""
        idx = pd.bdate_range("2020-01-01", "2022-02-28")
        rets = pd.DataFrame(
            {"WIN.N": 0.001, "LOSE.N": -0.001, "MID.N": 0.0}, index=idx
        )
        formations = pd.date_range("2021-01-31", "2021-12-31", freq="ME")
        baskets = _baskets_for(
            [str(d.date()) for d in formations], ["WIN.N", "LOSE.N", "MID.N"]
        )
        return rets, baskets

    def test_holds_top_names_and_reports_diagnostics(self):
        rets, baskets = self._flat_world()
        res = simulate_monthly(rets, baskets, decile=0.34, min_names=1, tc_bps=0.0)
        # decile 34% di 3 eleggibili = ceil(1.02) = 2 nomi: WIN e MID
        assert res.monthly_returns.index[0] == pd.Timestamp("2021-02-28")
        assert (res.diagnostics["n_held"] == 2).all()
        # WIN (+0.1%/g) pesato 0.5 domina MID (0) → ritorno positivo ogni mese
        assert (res.monthly_returns > 0).all()

    def test_execution_lag_skips_first_day(self):
        """Il primo giorno di borsa del mese di holding non viene accreditato."""
        idx = pd.bdate_range("2020-01-01", "2021-03-31")
        rets = pd.DataFrame({"A.N": 0.0, "B.N": -0.001}, index=idx)
        # A: +10% SOLO il primo giorno di borsa di feb 2021 — non deve contare
        first_feb = idx[(idx >= "2021-02-01")][0]
        rets.loc[first_feb, "A.N"] = 0.10
        baskets = _baskets_for(["2021-01-31"], ["A.N", "B.N"])
        res = simulate_monthly(rets, baskets, decile=0.5, min_names=1, tc_bps=0.0)
        assert res.monthly_returns.loc[pd.Timestamp("2021-02-28")] == pytest.approx(0.0)

    def test_costs_deducted_on_turnover(self):
        rets, baskets = self._flat_world()
        res0 = simulate_monthly(rets, baskets, decile=0.34, min_names=1, tc_bps=0.0)
        res10 = simulate_monthly(rets, baskets, decile=0.34, min_names=1, tc_bps=10.0)
        # primo mese: costo 10bps sull'intero nozionale
        diff = res0.monthly_returns.iloc[0] - res10.monthly_returns.iloc[0]
        assert diff == pytest.approx(10 / 1e4, rel=1e-6)
        # mesi successivi: stesso portafoglio → turnover da drift ~0 → costo ~0
        diff_later = res0.monthly_returns.iloc[3] - res10.monthly_returns.iloc[3]
        assert abs(diff_later) < 2e-5

    def test_delisted_partial_month_return_is_credited(self):
        """Un nome che muore a metà mese deve portare il suo ritorno parziale."""
        idx = pd.bdate_range("2020-01-01", "2021-03-31")
        rets = pd.DataFrame({"DEAD.N": 0.002, "ALIVE.N": 0.001}, index=idx)
        crash_day = pd.Timestamp("2021-02-10")
        rets.loc[crash_day, "DEAD.N"] = -0.50
        rets.loc[rets.index > crash_day, "DEAD.N"] = np.nan
        baskets = _baskets_for(["2021-01-31"], ["DEAD.N", "ALIVE.N"])
        res = simulate_monthly(rets, baskets, decile=1.0, min_names=1, tc_bps=0.0)
        feb = res.monthly_returns.loc[pd.Timestamp("2021-02-28")]
        assert feb < -0.20  # il crollo è nel P&L (peso 0.5)

    def test_coverage_reported_per_month(self):
        idx = pd.bdate_range("2020-01-01", "2021-03-31")
        rets = pd.DataFrame({"A.N": 0.001, "B.N": 0.001}, index=idx)
        baskets = _baskets_for(["2021-01-31"], ["A.N", "B.N", "GHOST.N"])
        res = simulate_monthly(rets, baskets, decile=1.0, min_names=1, tc_bps=0.0)
        cov = res.diagnostics["coverage"].iloc[0]
        assert cov == pytest.approx(2 / 3)

    def test_cash_month_when_too_few_names(self):
        """Sotto min_names il mese va in cash (ritorno = 0) e n_held = 0."""
        rets, baskets = self._flat_world()
        res = simulate_monthly(rets, baskets, decile=0.34, min_names=10, tc_bps=0.0)
        assert (res.monthly_returns == 0.0).all()
        assert (res.diagnostics["n_held"] == 0).all()

    def test_signal_fn_low_vol_selects_least_volatile(self):
        """Con signal_fn = -realized_volatility, il portafoglio tiene i nomi meno volatili."""
        idx = pd.bdate_range("2020-01-01", "2022-02-28")
        n = len(idx)
        steady = np.full(n, 0.0004)                                  # vol ~0
        choppy = np.where(np.arange(n) % 2 == 0, 0.02, -0.0192)      # vol alta, drift ~steady
        rets = pd.DataFrame({"STEADY.N": steady, "CHOPPY.N": choppy}, index=idx)
        formations = pd.date_range("2021-01-31", "2021-12-31", freq="ME")
        baskets = _baskets_for([str(d.date()) for d in formations], ["STEADY.N", "CHOPPY.N"])

        def neg_vol(daily_returns, asof):
            return -realized_volatility(daily_returns, asof)

        res = simulate_monthly(rets, baskets, signal_fn=neg_vol,
                               decile=0.5, min_names=1, tc_bps=0.0)
        # decile 50% di 2 eleggibili = 1 nome: il meno volatile (STEADY)
        for w in res.weights_history.values():
            assert set(w.index) == {"STEADY.N"}

    def test_default_signal_is_momentum_unchanged(self):
        """Senza signal_fn il comportamento è identico al momentum (regressione Sprint O)."""
        rets, baskets = self._flat_world()
        res_default = simulate_monthly(rets, baskets, decile=0.34, min_names=1, tc_bps=0.0)

        def mom(daily_returns, asof):
            return momentum_12_1(daily_returns, asof)

        res_explicit = simulate_monthly(rets, baskets, signal_fn=mom,
                                        decile=0.34, min_names=1, tc_bps=0.0)
        pd.testing.assert_series_equal(res_default.monthly_returns, res_explicit.monthly_returns)
        # Guardia forte: anche i pesi (e quindi turnover/costi futuri) devono coincidere.
        assert res_default.weights_history.keys() == res_explicit.weights_history.keys()
        for t in res_default.weights_history:
            pd.testing.assert_series_equal(
                res_default.weights_history[t], res_explicit.weights_history[t]
            )


class TestAlignToCalendar:
    """align_to_calendar: righe fantasma (festivi) eliminate, giorni di borsa
    mancanti inseriti come NaN — i valori veri restano intatti."""

    def test_phantom_rows_dropped(self):
        cal = pd.DatetimeIndex(["2004-01-16", "2004-01-20", "2004-01-21"])
        idx = pd.DatetimeIndex(["2004-01-16", "2004-01-19", "2004-01-20", "2004-01-21"])
        df = pd.DataFrame({"A.N": [0.01, 0.01, 0.02, 0.03]}, index=idx)  # 01-19 = MLK fantasma
        aligned, extra, missing = align_to_calendar(df, cal)
        assert list(aligned.index) == list(cal)
        assert pd.Timestamp("2004-01-19") in extra
        assert len(missing) == 0
        assert aligned.loc["2004-01-20", "A.N"] == pytest.approx(0.02)

    def test_missing_trading_days_inserted_as_nan(self):
        cal = pd.DatetimeIndex(["2006-10-06", "2006-10-09", "2006-10-10"])
        idx = pd.DatetimeIndex(["2006-10-06", "2006-10-10"])  # manca Columbus Day
        df = pd.DataFrame({"A.N": [0.01, 0.02]}, index=idx)
        aligned, extra, missing = align_to_calendar(df, cal)
        assert list(aligned.index) == list(cal)
        assert pd.Timestamp("2006-10-09") in missing
        assert np.isnan(aligned.loc["2006-10-09", "A.N"])
