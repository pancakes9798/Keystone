"""Tests for Combinatorial Purged Cross-Validation."""
from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
import pytest

from jeanclaude.backtest.cpcv import CPCV, CPCVConfig


@pytest.fixture
def daily_index():
    return pd.date_range("2020-01-01", periods=300, freq="B")


@pytest.fixture
def eval_times(daily_index):
    """eval_times[i] = daily_index[i] + 20 business days (simulates fwd_window=20)."""
    return pd.Series([
        daily_index[min(i + 20, len(daily_index) - 1)]
        for i in range(len(daily_index))
    ])


def test_number_of_folds(daily_index, eval_times):
    """C(6, 2) = 15 folds for default config."""
    cpcv = CPCV(CPCVConfig(n_splits=6, n_test_splits=2))
    folds = list(cpcv.split(daily_index, eval_times))
    assert len(folds) == comb(6, 2)


def test_train_test_disjoint(daily_index, eval_times):
    """Train and test indices must never overlap."""
    cpcv = CPCV(CPCVConfig(n_splits=6, n_test_splits=2))
    for train_idx, test_idx in cpcv.split(daily_index, eval_times):
        assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_purging_removes_overlapping_labels(daily_index, eval_times):
    """No training obs should have eval_time inside the test window."""
    cpcv = CPCV(CPCVConfig(n_splits=6, n_test_splits=2, embargo_pct=0.0))
    for train_idx, test_idx in cpcv.split(daily_index, eval_times):
        test_idx_sorted = np.sort(test_idx)
        breaks = np.where(np.diff(test_idx_sorted) > 1)[0] + 1
        segments = np.split(test_idx_sorted, breaks)
        train_evals = eval_times.iloc[train_idx].values
        for seg in segments:
            seg_start = daily_index[seg[0]]
            seg_end = daily_index[seg[-1]]
            overlap = (train_evals >= seg_start) & (train_evals <= seg_end)
            assert not overlap.any(), "Purging failed: training label overlaps test window"


def test_embargo_removes_observations_after_test(daily_index, eval_times):
    """Train set must not contain obs in the embargo zone after each test segment."""
    embargo_pct = 0.05
    cpcv = CPCV(CPCVConfig(n_splits=6, n_test_splits=2, embargo_pct=embargo_pct))
    n = len(daily_index)
    embargo_n = int(n * embargo_pct)
    for train_idx, test_idx in cpcv.split(daily_index, eval_times):
        test_idx_sorted = np.sort(test_idx)
        breaks = np.where(np.diff(test_idx_sorted) > 1)[0] + 1
        segments = np.split(test_idx_sorted, breaks)
        for seg in segments:
            seg_max = int(seg[-1])
            embargo_zone = set(range(seg_max + 1, min(seg_max + embargo_n + 1, n)))
            assert len(embargo_zone.intersection(set(train_idx.tolist()))) == 0


def test_all_observations_covered_across_folds(daily_index, eval_times):
    """Every index position appears in some test set."""
    cpcv = CPCV(CPCVConfig(n_splits=6, n_test_splits=2))
    all_test = set()
    for _, test_idx in cpcv.split(daily_index, eval_times):
        all_test.update(test_idx.tolist())
    assert all_test == set(range(len(daily_index)))


def test_purging_correct_for_non_contiguous_groups(daily_index, eval_times):
    """Groups 0 and 5 are maximally separated — purge must not over-exclude middle obs."""
    cpcv = CPCV(CPCVConfig(n_splits=6, n_test_splits=2, embargo_pct=0.0))
    n = len(daily_index)
    # Find the fold where test groups are groups 0 and 5 (first and last)
    groups = np.array_split(np.arange(n), 6)
    target_test = np.concatenate([groups[0], groups[5]])
    target_test_set = set(target_test.tolist())

    for train_idx, test_idx in cpcv.split(daily_index, eval_times):
        if set(test_idx.tolist()) == target_test_set:
            # Middle groups (1,2,3,4) should mostly be in train (minus purged obs)
            middle_obs = np.concatenate([groups[g] for g in [1, 2, 3, 4]])
            train_set = set(train_idx.tolist())
            # At least 50% of middle obs should be in train
            # (some may be purged due to eval_time overlap with test segments)
            middle_in_train = sum(1 for i in middle_obs if i in train_set)
            assert middle_in_train > len(middle_obs) * 0.5, (
                f"Too many middle obs purged: {middle_in_train}/{len(middle_obs)}"
            )
            break
