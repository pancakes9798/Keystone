"""
Price → returns transformations.

All functions take and return pd.DataFrame with DatetimeIndex.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns: ln(P_t / P_{t-1}).

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices, date-indexed.

    Returns
    -------
    pd.DataFrame
        Log returns (first row is NaN, dropped).
    """
    return np.log(prices / prices.shift(1)).dropna(how="all")


def to_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily simple returns: (P_t - P_{t-1}) / P_{t-1}.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices, date-indexed.

    Returns
    -------
    pd.DataFrame
        Simple returns (first row is NaN, dropped).
    """
    return prices.pct_change().dropna(how="all")


def resample_returns(
    returns: pd.DataFrame,
    freq: str = "W",
    method: str = "compound",
) -> pd.DataFrame:
    """Resample daily returns to a lower frequency.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns (log or simple).
    freq : str
        Pandas offset alias: ``'W'`` weekly, ``'ME'`` month-end, ``'QE'`` quarterly.
    method : str
        ``'compound'``: (1+r1)*(1+r2)*... - 1  (correct for simple returns)
        ``'sum'``: r1 + r2 + ...               (correct for log returns)

    Returns
    -------
    pd.DataFrame
        Resampled returns.
    """
    if method == "compound":
        return returns.resample(freq).apply(lambda x: (1 + x).prod() - 1)
    elif method == "sum":
        return returns.resample(freq).sum()
    else:
        raise ValueError(f"method must be 'compound' or 'sum', got '{method}'")


def rolling_covariance(
    returns: pd.DataFrame,
    window: int = 60,
    min_periods: int = 20,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Compute rolling covariance matrices.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns.
    window : int
        Rolling window in business days.
    min_periods : int
        Minimum observations required to produce a non-NaN result.

    Returns
    -------
    dict[pd.Timestamp, pd.DataFrame]
        Date → covariance matrix for each date with enough data.
    """
    result = {}
    dates = returns.index[window - 1:]

    for date in dates:
        window_data = returns.loc[:date].tail(window)
        if len(window_data) >= min_periods:
            result[date] = window_data.cov()

    return result


def annualized_stats(returns: pd.DataFrame, freq: int = 252) -> pd.DataFrame:
    """Compute annualized return, volatility, and Sharpe ratio.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns.
    freq : int
        Trading days per year (252 for daily, 52 for weekly).

    Returns
    -------
    pd.DataFrame
        Columns: ann_return, ann_vol, sharpe. Index: asset names.
    """
    ann_ret = returns.mean() * freq
    ann_vol = returns.std() * np.sqrt(freq)
    sharpe = ann_ret / ann_vol

    return pd.DataFrame({
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
    })
