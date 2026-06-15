"""Tests for jeanclaude.backtest.trend_overlay — synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jeanclaude.backtest.trend_overlay import evaluate_overlay_on_core, tsmom_position


def _spx(n_days: int, daily: float, start="2018-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame({"SPX": [daily] * n_days}, index=idx)


class TestTsmomPosition:
    def test_long_when_uptrend_and_calm(self):
        spx = _spx(300, 0.001)
        assert tsmom_position(spx, "EXPANSION", asof=spx.index[-1]) == 1

    def test_short_when_downtrend_and_stress(self):
        spx = _spx(300, -0.001)
        assert tsmom_position(spx, "CONTRACTION", asof=spx.index[-1]) == -1

    def test_flat_when_uptrend_but_stress(self):
        spx = _spx(300, 0.001)
        assert tsmom_position(spx, "CONTRACTION", asof=spx.index[-1]) == 0

    def test_flat_when_downtrend_but_calm(self):
        spx = _spx(300, -0.001)
        assert tsmom_position(spx, "EXPANSION", asof=spx.index[-1]) == 0

    def test_unknown_regime_counts_as_non_stress(self):
        spx = _spx(300, 0.001)
        assert tsmom_position(spx, "UNKNOWN", asof=spx.index[-1]) == 1

    def test_insufficient_history_is_flat(self):
        spx = _spx(100, 0.001)  # < 200 obs in the 12-1 window
        assert tsmom_position(spx, "EXPANSION", asof=spx.index[-1]) == 0

    def test_unknown_regime_downtrend_is_flat(self):
        """UNKNOWN = non-stress → blocca lo short anche in downtrend."""
        spx = _spx(300, -0.001)
        assert tsmom_position(spx, "UNKNOWN", asof=spx.index[-1]) == 0

    def test_multi_column_input_raises(self):
        idx = pd.bdate_range("2018-01-01", periods=300)
        df = pd.DataFrame({"SPX": [0.001] * 300, "OTHER": [0.0] * 300}, index=idx)
        with pytest.raises(ValueError, match="colonna singola"):
            tsmom_position(df, "EXPANSION", asof=df.index[-1])


class TestEvaluateOverlayOnCore:
    def _series(self, vals):
        idx = pd.date_range("2010-01-31", periods=len(vals), freq="ME")
        return pd.Series(vals, index=idx)

    def test_anticorrelated_overlay_passes_all_criteria(self):
        # core scende e risale; overlay è il suo opposto (decorrelato, difensivo)
        core = self._series([0.02, -0.10, -0.08, 0.05, 0.03, -0.06, 0.04] * 6)
        overlay = -core  # perfettamente anti-correlato, cumulato > 0
        res = evaluate_overlay_on_core(core, overlay, overlay_frac=0.5)
        assert res["combined_maxdd"] < res["core_maxdd"]      # (a)
        assert res["combined_calmar"] > res["core_calmar"]    # (b)
        assert res["criterio_c"] is True                      # (c) overlay cum >= 0
        assert res["corr_full"] < 0 and res["criterio_d"] is True   # (d)
        assert res["criterio_a"] is True and res["criterio_b"] is True
        assert res["verdict"] == "PASS"

    def test_positively_correlated_overlay_fails_on_corr(self):
        # overlay = core (correlato positivamente) → corr>0 → criterio (d) FAIL → verdict FAIL
        core = self._series([0.02, -0.10, -0.08, 0.05, 0.03, -0.06, 0.04] * 6)
        res = evaluate_overlay_on_core(core, core.copy(), overlay_frac=0.5)
        assert res["corr_full"] > 0
        assert res["criterio_d"] is False
        assert res["verdict"] == "FAIL"

    def test_criteria_keys_present(self):
        core = self._series([0.01, -0.02, 0.03, -0.01] * 10)
        overlay = self._series([0.0, 0.01, 0.0, 0.02] * 10)
        res = evaluate_overlay_on_core(core, overlay, overlay_frac=0.15)
        for k in ("core_maxdd", "combined_maxdd", "core_calmar", "combined_calmar",
                  "overlay_cum", "corr_full", "corr_stress",
                  "criterio_a", "criterio_b", "criterio_c", "criterio_d", "verdict"):
            assert k in res
