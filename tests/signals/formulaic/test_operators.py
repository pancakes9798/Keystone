"""Tests for jeanclaude.signals.formulaic.operators."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from jeanclaude.signals.formulaic import operators as ops


def test_cs_rank_in_unit_interval_and_monotone():
    s = pd.Series({"A": 1.0, "B": 3.0, "C": 2.0})
    r = ops.cs_rank(s)
    assert r.min() >= 0 and r.max() <= 1
    assert r["B"] > r["C"] > r["A"]

def test_cs_zscore_zero_mean_unit_std():
    s = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
    z = ops.cs_zscore(s)
    assert z.mean() == pytest.approx(0.0, abs=1e-9)
    assert z.std(ddof=0) == pytest.approx(1.0, abs=1e-9)

def test_ts_rank_last_value_percentile():
    x = pd.Series(range(10), dtype=float)
    assert ops.ts_rank(x, 10) == pytest.approx(1.0)

def test_decay_linear_weights_recent_more():
    x = pd.Series([0.0, 0.0, 1.0])
    assert ops.decay_linear(x, 3) == pytest.approx(3 / 6)

def test_ts_argmax_recency_of_extreme():
    x = pd.Series([5.0, 1.0, 1.0])
    assert ops.ts_argmax(x, 3) == 2

def test_signed_log1p_tames_tails_keeps_sign():
    assert ops.signed_log1p(pd.Series([-3.0])).iloc[0] == pytest.approx(-np.log(4))


def test_signedpower_keeps_sign():
    s = pd.Series({"A": -4.0, "B": 9.0})
    p = ops.signedpower(s, 0.5)
    assert p["A"] == pytest.approx(-2.0) and p["B"] == pytest.approx(3.0)


def test_cs_demean_zero_mean():
    s = pd.Series({"A": 1.0, "B": 3.0})
    assert ops.cs_demean(s).mean() == pytest.approx(0.0)


def test_ts_argmin_distance_zero_today():
    assert ops.ts_argmin(pd.Series([1.0, 5.0, 0.0]), 3) == 0  # min è oggi


def test_cs_zscore_constant_series_is_zero_not_nan():
    s = pd.Series({"A": 2.0, "B": 2.0})
    z = ops.cs_zscore(s)
    assert (z == 0.0).all()


def test_scale_all_zeros_is_safe():
    s = pd.Series({"A": 0.0, "B": 0.0})
    assert (ops.scale(s) == 0.0).all()


def test_ts_std_single_obs_is_nan():
    assert np.isnan(ops.ts_std(pd.Series([1.0, 2.0, 3.0]), 1))
