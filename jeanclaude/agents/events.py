from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from jeanclaude.signals.macro.labels import RegimeLabel


@dataclass
class PriceEvent:
    date: pd.Timestamp
    prices: pd.DataFrame
    returns: pd.DataFrame
    macro: pd.DataFrame


@dataclass
class RegimeEvent:
    date: pd.Timestamp
    label: RegimeLabel
    probabilities: np.ndarray      # shape (3,): [p_expansion, p_contraction, p_transition]
    labels_history: pd.Series      # full regime label history aligned to returns index
    changed: bool                  # True se diverso dall'ultimo regime registrato


@dataclass
class SignalEvent:
    date: pd.Timestamp
    views: pd.Series               # expected annual returns per asset (BL Q)
    confidence: pd.Series          # regime probabilities come peso di confidenza


@dataclass
class WeightsEvent:
    date: pd.Timestamp
    weights: pd.Series
    reason: str                    # "regime_change" | "signal_drift" | "risk_breach"


@dataclass
class NoRebalanceEvent:
    date: pd.Timestamp
    reason: str                    # "cooldown" | "no_trigger" | "optimization_failed"
