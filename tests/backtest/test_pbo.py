"""Tests for jeanclaude.backtest.pbo — CSCV Probability of Backtest Overfitting."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from jeanclaude.backtest.pbo import probability_of_backtest_overfitting


def test_pure_noise_has_high_pbo_mean_over_seeds():
    """Robusto: la PBO media del rumore puro converge a ~0.5 (non un singolo seed)."""
    pbos = []
    for seed in range(15):
        rng = np.random.default_rng(seed)
        mat = pd.DataFrame(rng.normal(0, 0.05, size=(120, 50)))
        pbos.append(probability_of_backtest_overfitting(mat, n_partitions=10).pbo)
    assert np.mean(pbos) > 0.4  # teoria: 0.5 per rumore


def test_pbo_raises_with_single_config():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match=">= 2"):
        probability_of_backtest_overfitting(pd.DataFrame(rng.normal(0, 1, (60, 1))))


def test_genuine_signal_has_low_pbo():
    rng = np.random.default_rng(1)
    T, N = 120, 50
    mat = rng.normal(0, 0.05, size=(T, N))
    mat[:, 0] += 0.03
    res = probability_of_backtest_overfitting(pd.DataFrame(mat), n_partitions=10)
    assert res.pbo < 0.2


def test_result_fields():
    rng = np.random.default_rng(2)
    res = probability_of_backtest_overfitting(pd.DataFrame(rng.normal(0, 1, (80, 20))), n_partitions=8)
    assert 0.0 <= res.pbo <= 1.0
    assert len(res.logits) == res.n_combinations
