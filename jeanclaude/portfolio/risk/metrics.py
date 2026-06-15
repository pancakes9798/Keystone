"""Metriche di rischio: VaR, CVaR, Component VaR, VaR Decomposition.

Tutte le funzioni usano historical simulation — nessuna assunzione distributiva.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def var_historical(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """VaR storico al livello di confidenza dato.

    Parameters
    ----------
    returns : pd.Series
        Serie di rendimenti giornalieri del portafoglio.
    confidence : float
        Livello di confidenza (default 0.95 = 95%).

    Returns
    -------
    float
        Perdita positiva (es. 0.02 = 2% al 95° percentile).
        Restituisce 0.0 su serie vuota.
    """
    if len(returns) == 0:
        return 0.0
    var = -float(np.quantile(returns, 1.0 - confidence, method="lower"))
    return max(var, 0.0)


def cvar_historical(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """CVaR (Expected Shortfall) storico — media delle perdite oltre VaR.

    Parameters
    ----------
    returns : pd.Series
        Serie di rendimenti giornalieri del portafoglio.
    confidence : float
        Livello di confidenza (default 0.95).

    Returns
    -------
    float
        CVaR positivo. Sempre >= var_historical per la stessa confidence.
    """
    if len(returns) == 0:
        return 0.0
    var = var_historical(returns, confidence)
    tail = returns[returns <= -var]
    if len(tail) == 0:
        return var
    cvar = -float(tail.mean())
    return max(cvar, var)


def component_var(
    weights: pd.Series,
    returns: pd.DataFrame,
    confidence: float = 0.95,
) -> pd.Series:
    """Component VaR per asset tramite aspettative condizionate.

    Ogni componente misura il contributo dell'asset al CVaR del portafoglio.
    La somma è esattamente uguale al CVaR del portafoglio.

    Parameters
    ----------
    weights : pd.Series
        Pesi del portafoglio, index = nomi asset.
    returns : pd.DataFrame
        Rendimenti giornalieri, colonne = nomi asset.
    confidence : float
        Livello di confidenza (default 0.95).

    Returns
    -------
    pd.Series
        Contributo al CVaR per asset, index = nomi asset.
    """
    assets = list(weights.index)
    r_p = returns[assets] @ weights
    if len(returns) == 0:
        return pd.Series(0.0, index=assets)
    var_p = var_historical(r_p, confidence)
    tail_mask = r_p <= -var_p

    contributions = pd.Series(index=assets, dtype=float)
    for asset in assets:
        if tail_mask.any():
            mean_tail = float(returns.loc[tail_mask, asset].mean())
        else:
            mean_tail = 0.0
        contributions[asset] = -float(weights[asset]) * mean_tail

    return contributions


def var_decomposition(
    weights: pd.Series,
    returns: pd.DataFrame,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Decomposizione completa del VaR per asset.

    Parameters
    ----------
    weights : pd.Series
        Pesi del portafoglio, index = nomi asset.
    returns : pd.DataFrame
        Rendimenti giornalieri, colonne = nomi asset.
    confidence : float
        Livello di confidenza (default 0.95).

    Returns
    -------
    pd.DataFrame
        Colonne: asset | weight | component_var | pct_contribution.
        Ordinato per component_var decrescente.
    """
    comp = component_var(weights, returns, confidence)
    total = comp.sum()
    n = len(comp)
    pct = (comp / total) if total != 0.0 else pd.Series(1.0 / n, index=comp.index)

    df = pd.DataFrame({
        "asset": comp.index,
        "weight": weights[comp.index].values,
        "component_var": comp.values,
        "pct_contribution": pct.values,
    })
    return df.sort_values("component_var", ascending=False).reset_index(drop=True)
