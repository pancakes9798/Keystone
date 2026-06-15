"""Spearman rank Information Coefficient (Sprint T)."""
from __future__ import annotations

import pandas as pd


def rank_ic(signal: pd.Series, fwd_returns: pd.Series) -> float:
    """Spearman corr tra il segnale e i ritorni forward, su nomi comuni."""
    df = pd.concat([signal, fwd_returns], axis=1, join="inner").dropna()
    if len(df) < 3:
        return float("nan")
    return float(df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman"))


def rank_ic_by_year(monthly_ic: pd.Series) -> pd.Series:
    """Media annuale di una serie mensile di IC (indice datetime)."""
    return monthly_ic.groupby(monthly_ic.index.year).mean()
