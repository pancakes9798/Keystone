"""
Composite signal types — CompositeSignal dataclass e RegimeTable config.
"""
from __future__ import annotations

import types as _types
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from jeanclaude.signals.macro.labels import RegimeLabel

# RegimeTable: ogni regime → {asset_class: score ∈ [-1, 1]}
RegimeTable = Mapping[RegimeLabel, Mapping[str, float]]

DEFAULT_REGIME_TABLE: RegimeTable = _types.MappingProxyType({
    RegimeLabel.EXPANSION:   _types.MappingProxyType({"equity":  0.8, "bonds": -0.3, "gold":  0.1, "cash": -0.2}),
    RegimeLabel.CONTRACTION: _types.MappingProxyType({"equity": -0.8, "bonds":  0.6, "gold":  0.5, "cash":  0.4}),
    RegimeLabel.TRANSITION:  _types.MappingProxyType({"equity":  0.0, "bonds":  0.1, "gold":  0.3, "cash":  0.2}),
})


@dataclass(frozen=True)
class CompositeSignal:
    """Output di CompositeSignalBuilder.build().

    Attributes
    ----------
    scores : pd.Series
        Score per asset class in [-1, +1]. index = nomi asset class.
        Positivo = sovrappeso, negativo = sottopeso vs. allocazione neutra.
    confidence : float
        Certezza del regime in [0, 1]. Derivata dall'entropia di regime_proba.
        1 = un regime domina nettamente; 0 = massima incertezza.
        Usata da Black-Litterman per calibrare Omega.
    regime : RegimeLabel
        Regime dominante (argmax di regime_proba).
    regime_proba : np.ndarray
        Vettore di probabilità posteriori, shape (3,):
        [p(EXPANSION), p(CONTRACTION), p(TRANSITION)].
    timestamp : pd.Timestamp
        Data dell'ultima osservazione nel RegimeResult usato per costruire questo segnale.

    Notes
    -----
    ``scores`` and ``regime_proba`` are mutable objects. The frozen dataclass
    prevents field reassignment but not in-place mutation. Treat them as
    read-only; never modify their contents after construction.
    """

    scores: pd.Series
    confidence: float
    regime: RegimeLabel
    regime_proba: np.ndarray
    timestamp: pd.Timestamp
