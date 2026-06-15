"""BacktestResult summary report."""
from __future__ import annotations

import pandas as pd

from .engine import BacktestResult
from .metrics import sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown


def summary(result: BacktestResult, periods_per_year: int = 252) -> pd.Series:
    """Compute key performance metrics from a BacktestResult.

    Parameters
    ----------
    result : BacktestResult
    periods_per_year : int
        Trading days per year (default 252).

    Returns
    -------
    pd.Series with labelled metrics.
    """
    r = result.portfolio_returns
    ann_ret = float((1 + r.mean()) ** periods_per_year - 1)
    ann_vol = float(r.std() * (periods_per_year ** 0.5))

    return pd.Series({
        "total_return":   float((1 + r).prod() - 1),
        "ann_return":     ann_ret,
        "ann_volatility": ann_vol,
        "sharpe_ratio":   sharpe_ratio(r, periods_per_year),
        "sortino_ratio":  sortino_ratio(r, periods_per_year),
        "calmar_ratio":   calmar_ratio(r, periods_per_year),
        "max_drawdown":   max_drawdown(r),
        "avg_turnover":   float(result.turnover.mean()) if len(result.turnover) else 0.0,
    })
