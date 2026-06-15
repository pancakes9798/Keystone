"""Probability of Backtest Overfitting via CSCV (Bailey & López de Prado 2014).

Input: matrice T x N dei ritorni per periodo (T periodi, N configurazioni di
strategia). Misura la probabilità che la configurazione migliore in-sample
finisca sotto la mediana out-of-sample.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    logits: list[float]
    n_combinations: int


def _sharpe(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, np.nan)
    return mu / sd


def probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame, n_partitions: int = 16
) -> PBOResult:
    """CSCV PBO. ``n_partitions`` (S, pari) partizioni contigue dei T periodi.

    Per ogni combinazione di S/2 partizioni come IS: rank delle N config per
    Sharpe IS, presa la migliore IS, misurato il suo rank relativo OOS → logit.
    PBO = frazione di combinazioni con logit <= 0.

    Costo: C(S, S/2) combinazioni — S=16 → 12.870 (varianza bassa, ~1-2s per chiamata);
    S=10 → 252 (più veloce ma varianza ~0.18 sul rumore). Per un singolo verdetto usa
    S alto; per chiamate ripetute in loop preferisci S=10.
    """
    M = returns_matrix.to_numpy(dtype=float)
    T, N = M.shape
    if N < 2:
        raise ValueError(f"PBO richiede >= 2 configurazioni (ricevute N={N}).")
    S = n_partitions - (n_partitions % 2)  # forza pari
    S = max(2, min(S, (T // 2) * 2))
    parts = np.array_split(np.arange(T), S)

    logits: list[float] = []
    for is_idx in combinations(range(S), S // 2):
        oos_idx = [p for p in range(S) if p not in is_idx]
        is_rows = np.concatenate([parts[p] for p in is_idx])
        oos_rows = np.concatenate([parts[p] for p in oos_idx])
        sr_is = _sharpe(M[is_rows])
        sr_oos = _sharpe(M[oos_rows])
        if np.all(np.isnan(sr_is)):
            continue
        best = int(np.nanargmax(sr_is))
        order = pd.Series(sr_oos).rank(pct=True)
        w = float(order.iloc[best])
        if np.isnan(w):  # la migliore-IS ha varianza OOS nulla → split non informativo
            continue
        w = min(max(w, 1e-6), 1 - 1e-6)
        logits.append(float(np.log(w / (1.0 - w))))

    pbo = float(np.mean([1.0 if l <= 0 else 0.0 for l in logits])) if logits else float("nan")
    return PBOResult(pbo=pbo, logits=logits, n_combinations=len(logits))
