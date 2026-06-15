"""TrendMomentumOptimizer — HRP + MA200 trend filter + 12-1m momentum tilt.

All filters are computed from the ``returns`` DataFrame so no separate
price feed is required. Compatible with the Optimizer protocol.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

from jeanclaude.portfolio.covariance.ledoit_wolf import LedoitWolfCovariance
from jeanclaude.portfolio.optimizer.hrp import HRPOptimizer

_LOOKBACK  = 756   # 3-year rolling window for HRP covariance
_MIN_HIST  = 252   # minimum days per asset to include in HRP
_MA_WINDOW = 200   # days for trend filter
_MOM_LONG  = 252   # lookback for 12-month momentum
_MOM_SHORT = 21    # skip period (avoid 1-month reversal)


@dataclass(frozen=True)
class TrendMomentumParams:
    """Calibrated parameters for one strategy variant."""
    name: str
    damp_factor: float         # MA200 damping: 0.0 = off, 1.0 = full zero-out
    momentum_strength: float   # cross-sectional tilt: 0.0 = off
    equity_min: float = 0.0    # minimum equity allocation (0.0 = no floor)
    max_weight: float = 0.20   # per-asset weight cap


def _apply_trend_filter(
    weights: pd.Series,
    returns: pd.DataFrame,
    damp_factor: float,
) -> pd.Series:
    """Dampen weight of assets trading below their 200-day MA.

    Uses cumulative returns as a price proxy — equivalent to raw price
    levels for the purpose of the MA200 comparison.
    """
    w = weights.copy().astype(float)
    cum = (1.0 + returns).cumprod()
    for asset in w.index:
        if asset not in cum.columns:
            continue
        series = cum[asset].dropna()
        if len(series) < _MA_WINDOW:
            continue
        ma   = float(series.iloc[-_MA_WINDOW:].mean())
        last = float(series.iloc[-1])
        if last < ma:
            w[asset] *= damp_factor
    total = w.sum()
    if total > 0:
        w /= total
    else:
        w = pd.Series(1.0 / len(w), index=w.index)
    return w


def _apply_momentum_tilt(
    weights: pd.Series,
    returns: pd.DataFrame,
    strength: float,
) -> pd.Series:
    """12-1 cross-sectional momentum rank tilt.

    Uses a blend of baseline (equal-weight) and rank score to ensure assets
    with strong momentum can gain weight even from a zero HRP starting point.
    Multiplier applied to a baseline-inflated weight vector, then renormed.
    """
    if len(returns) < _MOM_LONG:
        return weights
    sub = returns.iloc[-_MOM_LONG:-_MOM_SHORT] if _MOM_SHORT > 0 else returns.iloc[-_MOM_LONG:]
    mom = ((1.0 + sub).prod() - 1.0).reindex(weights.index).dropna()
    if len(mom) < 2:
        return weights
    ranks = mom.rank(pct=True)
    n = len(weights)
    baseline = 1.0 / n  # equal-weight baseline ensures non-zero base
    w = weights.copy().astype(float)
    for asset in w.index:
        if asset in ranks.index:
            blended = 0.5 * w[asset] + 0.5 * baseline
            w[asset] = blended * max(1.0 + strength * (ranks[asset] - 0.5), 0.0)
    total = w.sum()
    if total > 0:
        w /= total
    return w


def _apply_equity_floor(
    weights: pd.Series,
    equity_min: float,
    equity_rics: frozenset[str],
) -> pd.Series:
    """Ensure minimum equity allocation (scale equity up, non-equity down)."""
    w = weights.copy().astype(float)
    eq  = [t for t in w.index if t in equity_rics]
    neq = [t for t in w.index if t not in equity_rics]
    current_eq = float(w[eq].sum()) if eq else 0.0
    if current_eq >= equity_min or not eq:
        return w
    shortfall  = equity_min - current_eq
    neq_total  = float(w[neq].sum()) if neq else 0.0
    _EQUITY_FLOOR_TOL = 0.01
    if neq_total < shortfall + _EQUITY_FLOOR_TOL:
        logger.warning(
            "equity_floor: cannot meet %.1f%% floor (shortfall=%.4f, neq_total=%.4f); "
            "skipping floor adjustment",
            equity_min * 100, shortfall, neq_total,
        )
        return w
    scale = (neq_total - shortfall) / neq_total
    for t in neq:
        w[t] *= scale
    for t in eq:
        w[t] += shortfall * (float(weights[t]) / current_eq) if current_eq > 0 else shortfall / len(eq)
    total = w.sum()
    if total > 0:
        w /= total
    return w


class TrendMomentumOptimizer:
    """Walk-forward optimizer: HRP → equity floor → trend filter → momentum tilt → cap.

    Implements the Optimizer protocol: optimize(returns) -> pd.Series.

    Parameters
    ----------
    params : TrendMomentumParams
        Strategy configuration.
    equity_rics : frozenset[str]
        Set of ticker names considered equity (used only when params.equity_min > 0).
    """

    def __init__(
        self,
        params: TrendMomentumParams,
        equity_rics: frozenset[str] = frozenset(),
    ) -> None:
        self._p = params
        self._equity_rics = equity_rics

    def optimize(self, returns: pd.DataFrame) -> pd.Series:
        """Compute portfolio weights for the given returns history.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily log/simple returns, columns = asset tickers, rows = trading days.

        Returns
        -------
        pd.Series
            Non-negative weights summing to 1.0, indexed by ticker.
        """
        # 1. HRP on rolling window — only include assets with enough history
        window    = returns.iloc[-_LOOKBACK:]
        available = window.columns[window.notna().sum() >= _MIN_HIST]
        clean     = window[available].dropna()

        if len(clean) < _MIN_HIST or clean.shape[1] < 2:
            cols = available if len(available) >= 1 else returns.columns
            n = max(len(cols), 1)
            return pd.Series(1.0 / n, index=cols)

        weights = HRPOptimizer(LedoitWolfCovariance()).optimize(clean)

        # If HRP is degenerate (all weight on one asset), use equal weights
        # as a more neutral starting point before applying overlays.
        n_assets = len(weights)
        if n_assets > 1 and float(weights.max()) > (1.0 - 1e-6):
            weights = pd.Series(1.0 / n_assets, index=weights.index)

        # 2. Equity floor (no-op when equity_min == 0)
        if self._p.equity_min > 0.0 and self._equity_rics:
            weights = _apply_equity_floor(weights, self._p.equity_min, self._equity_rics)

        # 3. MA200 trend filter
        if self._p.damp_factor > 0.0:
            weights = _apply_trend_filter(weights, returns[available], self._p.damp_factor)

        # 4. 12-1 momentum tilt
        if self._p.momentum_strength > 0.0:
            weights = _apply_momentum_tilt(weights, returns[available], self._p.momentum_strength)

        # 5. Per-asset cap + renorm
        # The effective cap is at least 1/n_assets so we never clip below equal weight
        # (avoids destroying filter effects in small universes).
        effective_cap = max(self._p.max_weight, 1.0 / max(len(weights), 1))
        weights = weights.clip(upper=effective_cap)
        s = weights.sum()
        if s > 0:
            weights /= s
        return weights
