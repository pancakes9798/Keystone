"""Portfolio performance metrics.

All functions accept a pd.Series of daily returns (not prices).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe Ratio (zero risk-free rate assumption)."""
    std = returns.std()
    if std == 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sortino Ratio (downside deviation denominator)."""
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("inf")
    return float(returns.mean() / downside.std() * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown as a positive fraction (e.g. 0.15 = 15% drawdown)."""
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(-drawdown.min())


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound Annual Growth Rate (CAGR) from simple daily returns.

    Parameters
    ----------
    returns : pd.Series
        Simple (not log) daily returns.
    periods_per_year : int
        Trading days per year used for annualisation (default 252).

    Returns
    -------
    float
        Annualised return as a fraction (e.g. 0.12 = 12%).
        Returns 0.0 for an empty Series.
        Returns nan if ``returns`` contains NaN values.
    """
    n = len(returns)
    if n == 0:
        return 0.0
    return float((1 + returns).prod() ** (periods_per_year / n) - 1)


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calmar Ratio = annualized return / max drawdown."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return float("inf")
    ann_return = annualized_return(returns, periods_per_year)
    return ann_return / mdd


def deflated_sharpe_ratio(
    sharpe_obs: float,
    n_trials: int,
    obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    periods_per_year: int = 252,
) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014 / AFML ch.8).

    Adjusts the observed Sharpe Ratio for the multiple-testing problem and
    returns the probability that the strategy's true SR > 0.

    The computation follows the original paper which works on per-observation
    (non-annualized) SR units.  The annualized ``sharpe_obs`` is de-annualized
    internally so that it is on the same scale as the expected-maximum-SR
    benchmark.

    Parameters
    ----------
    sharpe_obs : float
        Observed annualized Sharpe Ratio.
    n_trials : int
        Number of independent strategy variants / parameter sets tried.
    obs : int
        Number of daily return observations in the sample.
    skewness : float
        Skewness of returns (default 0 = Gaussian).
    kurtosis : float
        Kurtosis of returns (default 3 = Gaussian).
    periods_per_year : int
        Trading days per year used to de-annualize ``sharpe_obs`` (default 252).
    """
    # De-annualize so both SR and benchmark are in per-observation units
    sr_daily = sharpe_obs / np.sqrt(periods_per_year)
    sr_benchmark = _expected_max_sr(n_trials, obs)
    sr_std = np.sqrt(
        (1 - skewness * sr_daily + ((kurtosis - 1) / 4) * sr_daily ** 2)
        / max(obs - 1, 1)
    )
    if sr_std == 0:
        return 0.0
    z = (sr_daily - sr_benchmark) / sr_std
    return float(stats.norm.cdf(z))


def _expected_max_sr(n_trials: int, obs: int) -> float:
    """Expected maximum SR from n_trials independent tests (Bailey & LdP 2014)."""
    if n_trials <= 1:
        return 0.0
    euler = 0.5772156649  # Euler-Mascheroni constant
    e_max = (
        (1 - euler) * stats.norm.ppf(1 - 1.0 / n_trials)
        + euler * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    return float(e_max / np.sqrt(obs))
