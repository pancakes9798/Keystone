"""
Regime labels and result container for HMM regime detection.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pomegranate.hmm import DenseHMM

logger = logging.getLogger(__name__)


class RegimeLabel(IntEnum):
    """Macro regime classification."""

    EXPANSION = 0
    CONTRACTION = 1
    TRANSITION = 2


@dataclass
class RegimeResult:
    """Output of RegimeDetector.fit / RegimeDetector.predict.

    Attributes
    ----------
    labels : pd.Series
        Regime for each date (same DatetimeIndex as input). Values are ``int64``
        integers compatible with ``RegimeLabel`` via equality (e.g.
        ``label == RegimeLabel.EXPANSION``). Use ``RegimeLabel(int(v))`` to get
        a typed enum from individual values.
    probabilities : pd.DataFrame
        Posterior state probabilities, shape (T, 3).
        Column names are RegimeLabel integers: 0, 1, 2.
    model : DenseHMM
        The fitted pomegranate DenseHMM model (serialisable via pickle).
    """

    labels: pd.Series
    probabilities: pd.DataFrame
    model: DenseHMM


def _build_label_map(
    X: np.ndarray,
    raw_states: np.ndarray,
    state_vars: pd.DataFrame,
    label_feature: str,
    label_order: str,
    n_components: int,
) -> dict[int, RegimeLabel]:
    """Map HMM state indices to RegimeLabel using per-state empirical means.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix used for fitting, shape (T, n_features).
    raw_states : np.ndarray
        Viterbi state sequence, shape (T,). Integer indices into [0, n_components).
    state_vars : pd.DataFrame
        The DataFrame used for fitting (to resolve column index from label_feature).
    label_feature : str
        Column name used to order states. Falls back to first column if absent.
    label_order : str
        ``"ascending"`` → lowest mean = EXPANSION (e.g. VIX).
        ``"descending"`` → highest mean = EXPANSION (e.g. yield_slope).
    n_components : int
        Number of HMM states.

    Returns
    -------
    dict[int, RegimeLabel]
        Mapping from HMM state index to RegimeLabel.
    """
    if label_feature not in state_vars.columns:
        feature_idx = 0
    else:
        feature_idx = list(state_vars.columns).index(label_feature)

    state_means_list = []
    for s in range(n_components):
        mask = raw_states == s
        if mask.any():
            state_means_list.append(float(X[mask, feature_idx].mean()))
        else:
            logger.warning(
                "_build_label_map: state %d has no Viterbi observations. "
                "Using 0.0 as mean — label assignment for this state may be unreliable.",
                s,
            )
            state_means_list.append(0.0)  # 0.0 = global center on standardized data
    state_means = np.array(state_means_list)

    sorted_states = np.argsort(state_means)
    if label_order == "descending":
        sorted_states = sorted_states[::-1]

    # sorted_states[0] → lowest mean → EXPANSION
    # sorted_states[1] → middle mean → TRANSITION
    # sorted_states[2] → highest mean → CONTRACTION
    regime_order = [RegimeLabel.EXPANSION, RegimeLabel.TRANSITION, RegimeLabel.CONTRACTION]
    return {int(hmm_state): regime for hmm_state, regime in zip(sorted_states, regime_order)}
