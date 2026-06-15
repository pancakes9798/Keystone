"""Campi di input formulaici returns-only (Sprint T). Puri, PIT (loc[:asof]).

Tutto derivato dalla matrice di total return giornalieri (date x RIC).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_price(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """logP_t = somma cumulata di log(1+r) per RIC (prezzo log ricostruito)."""
    return np.log1p(daily_returns).cumsum()


def market_return(daily_returns: pd.DataFrame) -> pd.Series:
    """Ritorno equal-weight del paniere a ogni data (media cross-sezionale)."""
    return daily_returns.mean(axis=1)


def cumret(daily_returns: pd.DataFrame, asof: pd.Timestamp,
           months_back: int, months_skip: int, days_per_month: int = 21) -> pd.Series:
    """Ritorno composto sulla finestra [T-months_back, T-months_skip] (in giorni borsa).

    Storia insufficiente (< months_back*days_per_month righe ≤ asof) → NaN per ogni
    nome: evita di spacciare una finestra parziale per una piena (garbage sui nomi
    a storia corta).
    """
    window = daily_returns.loc[:asof]
    if len(window) < months_back * days_per_month:
        return pd.Series(np.nan, index=daily_returns.columns)
    start = -months_back * days_per_month
    end = -months_skip * days_per_month if months_skip > 0 else None
    seg = window.iloc[start:end]
    return (1.0 + seg).prod(skipna=True) - 1.0


def resid_return(daily_returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Ritorno idiosincratico r - beta*mkt_r, beta da regressione rolling `window`g.

    beta_i = cov(r_i, mkt) / var(mkt) sul rolling window; resid = r - beta*mkt.
    """
    mkt = market_return(daily_returns)
    var_m = mkt.rolling(window).var()
    # Vettorizzato (era un loop su ~987 colonne, dominante nel mining Sprint T):
    # rolling cov di ogni colonna con mkt in una sola chiamata.
    cov = daily_returns.rolling(window).cov(mkt)
    beta = cov.div(var_m, axis=0)
    return daily_returns.sub(beta.mul(mkt, axis=0))
