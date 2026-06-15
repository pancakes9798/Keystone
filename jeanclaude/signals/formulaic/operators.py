"""Operatori formulaici puri returns-only (Sprint T).

CS = cross-sezionali (su una pd.Series indicizzata per RIC, a un asof).
TS = time-series (su una pd.Series temporale di un singolo RIC, ritornano scalare
     riferito a `asof` = ultimo valore della finestra).

Comportamento di warm-up (documentato): se `d > len(x)` la finestra è troncata ai dati
disponibili (nessun errore); `ts_std(x, 1)` su una finestra di 1 elemento ritorna NaN
(dev.std indefinita). I chiamanti devono trattare i NaN come "non eleggibile".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---- cross-sezionali (Series RIC->valore) ----

def cs_rank(s: pd.Series) -> pd.Series:
    """Rank percentile in [0,1] (NaN ignorati)."""
    return s.rank(pct=True)

def cs_zscore(s: pd.Series) -> pd.Series:
    m, sd = s.mean(), s.std(ddof=0)
    return (s - m) / sd if sd > 0 else s * 0.0

def cs_demean(s: pd.Series) -> pd.Series:
    return s - s.mean()

def scale(s: pd.Series, a: float = 1.0) -> pd.Series:
    denom = s.abs().sum()
    return a * s / denom if denom > 0 else s * 0.0

# ---- pointwise ----

def signed_log1p(s: pd.Series) -> pd.Series:
    return np.sign(s) * np.log1p(s.abs())

def signedpower(s: pd.Series, a: float) -> pd.Series:
    return np.sign(s) * s.abs() ** a

# ---- time-series (scalare riferito all'ultimo punto della finestra) ----

def ts_window(x: pd.Series, d: int) -> pd.Series:
    return x.iloc[-d:]

def ts_std(x: pd.Series, d: int) -> float:
    return float(ts_window(x, d).std(ddof=1))

def ts_mean(x: pd.Series, d: int) -> float:
    return float(ts_window(x, d).mean())

def ts_skew(x: pd.Series, d: int) -> float:
    return float(ts_window(x, d).skew())

def ts_rank(x: pd.Series, d: int) -> float:
    w = ts_window(x, d)
    return float(w.rank(pct=True).iloc[-1]) if len(w) else float("nan")

def ts_argmax(x: pd.Series, d: int) -> float:
    """Distanza (in giorni) dal punto di massimo nella finestra: 0 = oggi."""
    w = ts_window(x, d)
    return float(len(w) - 1 - int(np.argmax(w.values))) if len(w) else float("nan")

def ts_argmin(x: pd.Series, d: int) -> float:
    w = ts_window(x, d)
    return float(len(w) - 1 - int(np.argmin(w.values))) if len(w) else float("nan")

def decay_linear(x: pd.Series, d: int) -> float:
    """Media lineare pesata: pesi 1,2,...,k (più recente = peso k)."""
    w = ts_window(x, d).values
    k = len(w)
    weights = np.arange(1, k + 1, dtype=float)
    return float(np.dot(weights, w) / weights.sum()) if k else float("nan")

def ts_corr(x: pd.Series, y: pd.Series, d: int) -> float:
    xw, yw = ts_window(x, d), ts_window(y, d)
    return float(xw.corr(yw)) if len(xw) >= 2 else float("nan")
