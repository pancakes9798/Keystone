"""Tests for jeanclaude.backtest.ic — Spearman rank IC."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from jeanclaude.backtest.ic import rank_ic, rank_ic_by_year


def test_perfect_alignment_ic_one():
    sig = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
    fwd = pd.Series({"A": 0.1, "B": 0.2, "C": 0.3})
    assert rank_ic(sig, fwd) == pytest.approx(1.0)


def test_inverse_alignment_ic_minus_one():
    sig = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
    fwd = pd.Series({"A": 0.3, "B": 0.2, "C": 0.1})
    assert rank_ic(sig, fwd) == pytest.approx(-1.0)


def test_by_year_groups_correctly():
    idx = pd.to_datetime(["2016-01-31", "2016-02-29", "2017-01-31"])
    ics = pd.Series([0.1, 0.2, -0.1], index=idx)
    by = rank_ic_by_year(ics)
    assert by.loc[2016] == pytest.approx(0.15)
    assert by.loc[2017] == pytest.approx(-0.1)
