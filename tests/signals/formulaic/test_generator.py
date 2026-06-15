"""Tests for jeanclaude.signals.formulaic.generator."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from jeanclaude.signals.formulaic.generator import build_search_space, StrategyConfig


def _panel(n=400, k=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-01", periods=n)
    cols = [f"R{i}.N" for i in range(k)]
    return pd.DataFrame(rng.normal(0.0003, 0.02, size=(n, k)), index=idx, columns=cols)


def test_space_is_exactly_54():
    space = build_search_space()
    assert len(space) == 54  # dichiarato/contato: 3 (A) + 48 (B) + 3 (C)
    assert all(isinstance(c, StrategyConfig) for c in space)


def test_names_are_unique():
    space = build_search_space()
    names = [c.name for c in space]
    assert len(names) == len(set(names))


def test_all_configs_return_nonempty_series_on_long_panel():
    panel = _panel(n=3000, k=12)  # abbastanza per le finestre lunghe (36m=756g)
    asof = panel.index[-1]
    for c in build_search_space():
        s = c.signal_fn(panel, asof)
        assert isinstance(s, pd.Series), f"{c.name}: non è una Series"
        assert len(s) >= 1 and not s.isna().all(), f"{c.name}: segnale vuoto/all-NaN"
        assert s.index.isin(panel.columns).all(), f"{c.name}: indice non RIC"


def test_tier_c_no_nan_from_index_misalignment():
    """Tier C: footprint-NaN divergenti NON devono ridurre l'universo via prodotto."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2003-01-01", periods=2500)
    cols = [f"R{i}.N" for i in range(8)]
    data = rng.normal(0.0003, 0.01, (2500, 8))
    data[-150:, 6] = np.nan  # R6: NaN recente (vol NaN, cumret no) → potenziale disallineamento
    panel = pd.DataFrame(data, index=idx, columns=cols)
    asof = panel.index[-1]
    for c in [c for c in build_search_space() if c.tier == "C"]:
        s = c.signal_fn(panel, asof)
        assert not s.isna().any(), f"{c.name}: NaN da disallineamento indici"
