"""
Combinatorial Purged Cross-Validation (López de Prado, AFML ch.12).

Generates train/test splits for financial time series with:
- Purging: removes training obs whose label evaluation overlaps the test window
- Embargo: removes a buffer of obs after each test period (serial correlation)
- Combinatorial: produces C(k, n_test) fold combinations for robust evaluation
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CPCVConfig:
    n_splits: int = 6           # k: divide series into k contiguous groups
    n_test_splits: int = 2      # how many groups form the test set each fold
    embargo_pct: float = 0.01   # fraction of total obs to embargo after test


class CPCV:
    """Combinatorial Purged Cross-Validation.

    Parameters
    ----------
    config : CPCVConfig
        Hyperparameters for splitting.
    """

    def __init__(self, config: CPCVConfig | None = None) -> None:
        self.config = config or CPCVConfig()

    def split(
        self,
        index: pd.DatetimeIndex,
        eval_times: pd.Series,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, test_indices) with purging and embargo.

        Parameters
        ----------
        index : pd.DatetimeIndex
            Observation timestamps (length T).
        eval_times : pd.Series
            Integer-indexed Series of length T. eval_times[i] is the datetime
            when observation i's label is evaluated (typically index[i] + fwd_window).
            Used for purging: any training obs whose eval_time falls inside
            the test window is excluded.

        Yields
        ------
        train_idx : np.ndarray of int
            Training observation indices (purged + embargoed).
        test_idx : np.ndarray of int
            Test observation indices.

        Raises
        ------
        ValueError
            If eval_times length does not match index length.
        """
        if len(eval_times) != len(index):
            raise ValueError(
                f"eval_times length ({len(eval_times)}) must match index length ({len(index)})"
            )

        n = len(index)
        k = self.config.n_splits
        n_test = self.config.n_test_splits
        embargo_n = int(n * self.config.embargo_pct)  # 0 means no embargo

        # Divide the index into k contiguous groups
        groups = np.array_split(np.arange(n), k)

        # Precompute eval_times as numpy array for vectorized ops
        eval_times_arr = eval_times.values

        for test_group_ids in combinations(range(k), n_test):
            test_idx = np.concatenate([groups[g] for g in test_group_ids])

            # Candidate train: everything not in test
            all_idx = np.arange(n)
            candidate = np.setdiff1d(all_idx, test_idx)

            # Build excluded set per contiguous test segment to avoid
            # over-purging when test groups are non-contiguous
            excluded = set()

            # Identify contiguous segments in test_idx
            test_idx_sorted = np.sort(test_idx)
            breaks = np.where(np.diff(test_idx_sorted) > 1)[0] + 1
            segments = np.split(test_idx_sorted, breaks)

            for seg in segments:
                seg_start = index[seg[0]]
                seg_end = index[seg[-1]]
                seg_max = int(seg[-1])

                # Purge (vectorized): candidate obs whose eval_time falls in this segment
                candidate_evals = eval_times_arr[candidate]
                purge_mask = (candidate_evals >= seg_start) & (candidate_evals <= seg_end)
                excluded.update(candidate[purge_mask].tolist())

                # Embargo: obs in the buffer zone immediately after this segment
                if embargo_n > 0:
                    embargo_end = min(seg_max + embargo_n + 1, n)
                    excluded.update(range(seg_max + 1, embargo_end))

            train_idx = np.array([i for i in candidate if i not in excluded])
            yield train_idx, test_idx
