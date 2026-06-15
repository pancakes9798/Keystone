"""CPCV con refit per fold — la validazione che il tearsheet chiamava CPCV ma non era.

Per ogni fold combinatorio: i parametri vengono ri-selezionati con calibrate_fn
sui SOLI dati di train (purged + embargoed), poi valutati con evaluate_fn sul
test fold. eval_times = data + eval_horizon_days (orizzonte della label, qui
il periodo di holding mensile).

Nota metodologica documentata: con strategie a finestra rolling lunga (HRP 756d)
il train di ogni fold è un'unione di segmenti contigui; calibrate_fn riceve la
concatenazione dei segmenti e deve trattarli come tali (la calibrazione Sprint E
su sub-finestre lo fa naturalmente). L'overlap di covarianza tra fold è
accettato (LdP: il purging riguarda le label, non le feature).
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from .cpcv import CPCV, CPCVConfig


def cpcv_refit_validation(
    returns: pd.DataFrame,
    calibrate_fn: Callable[[pd.DataFrame], dict],
    evaluate_fn: Callable[[pd.DataFrame, dict], float],
    config: CPCVConfig | None = None,
    eval_horizon_days: int = 21,
) -> pd.DataFrame:
    """Run Combinatorial Purged Cross-Validation with per-fold parameter refit.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns with DatetimeIndex and assets as columns.
    calibrate_fn : Callable[[pd.DataFrame], dict]
        Called once per fold on train data; returns a parameter dict.
    evaluate_fn : Callable[[pd.DataFrame, dict], float]
        Called once per fold on test data with the calibrated params;
        returns a scalar score (e.g. Sharpe ratio).
    config : CPCVConfig, optional
        CPCV configuration. Defaults to CPCVConfig().
    eval_horizon_days : int
        Number of business days in the label evaluation horizon. Used to
        build eval_times for purging. Default 21 (~1 month).

    Returns
    -------
    pd.DataFrame
        One row per fold with columns: fold, n_train, n_test, params, test_sharpe.
    """
    config = config or CPCVConfig()
    cpcv = CPCV(config)
    index = returns.index
    n = len(index)

    # Build eval_times: for each obs i, the label evaluates at i + eval_horizon_days
    # (clamped to the last available index position).
    horizon_pos = [min(i + eval_horizon_days, n - 1) for i in range(n)]
    eval_times = pd.Series([index[j] for j in horizon_pos])

    rows = []
    for fold, (train_idx, test_idx) in enumerate(cpcv.split(index, eval_times)):
        train = returns.iloc[train_idx]
        test = returns.iloc[test_idx]
        params = calibrate_fn(train)
        sharpe = evaluate_fn(test, params)
        rows.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "params": params,
            "test_sharpe": sharpe,
        })
    return pd.DataFrame(rows)
