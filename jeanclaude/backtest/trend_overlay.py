"""Trend/regime overlay signal and overlay-on-core evaluation (Sprint S).

Pure functions — no network. The overlay is a long/flat/short time-series
momentum on a single equity index, gated by the macro HMM regime, sized as a
small notional fraction added on top of a fully-invested core (futures overlay).
"""
from __future__ import annotations

import pandas as pd

from jeanclaude.backtest.cross_sectional import LOOKBACK_DAYS, MIN_OBS, SKIP_DAYS, momentum_12_1
from jeanclaude.backtest.metrics import calmar_ratio, max_drawdown

_STRESS_REGIME = "CONTRACTION"


def tsmom_position(
    spx_daily_returns: pd.DataFrame,
    regime_label: str,
    asof: pd.Timestamp,
    lookback: int = LOOKBACK_DAYS,
    skip: int = SKIP_DAYS,
    min_obs: int = MIN_OBS,
) -> int:
    """Posizione overlay in {-1, 0, +1} per il mese di holding dopo ``asof``.

    Trend = segno del momentum 12-1 (riusa :func:`momentum_12_1`) sull'unica
    colonna di ``spx_daily_returns``. Gate regime: stress sse
    ``regime_label == "CONTRACTION"`` (``"UNKNOWN"`` conta come non-stress).

    - ``+1`` se trend > 0 E non-stress;
    - ``-1`` se trend < 0 E stress;
    - ``0``  altrimenti (segnali discordi o storia insufficiente).
    """
    if spx_daily_returns.shape[1] > 1:
        raise ValueError(
            "tsmom_position attende un DataFrame a colonna singola; "
            f"ricevute colonne {list(spx_daily_returns.columns)}"
        )
    mom = momentum_12_1(spx_daily_returns, asof=asof, lookback=lookback, skip=skip, min_obs=min_obs)
    trend = float(mom.iloc[0]) if len(mom) else float("nan")
    if pd.isna(trend):
        return 0
    stress = regime_label == _STRESS_REGIME
    if trend > 0 and not stress:
        return 1
    if trend < 0 and stress:
        return -1
    return 0


def evaluate_overlay_on_core(
    core_returns: pd.Series,
    overlay_excess: pd.Series,
    overlay_frac: float,
    stress_quantile: float = 0.10,
) -> dict:
    """Valuta i criteri ex-ante overlay-su-core (spec §4).

    combined = core + overlay_frac * overlay_excess.
    (a) MaxDD(combined) < MaxDD(core); (b) Calmar(combined) > Calmar(core);
    (c) overlay_excess cumulato >= 0; (d) corr(overlay, core) <= 0 full E nei mesi
    di stress (peggiore decile del core).

    Edge degenere: se il core non ha drawdown (MaxDD=0, serie monotòna su) Calmar
    è inf e (a)/(b) risultano False (verdict FAIL) — corretto come "nessun
    miglioramento dimostrabile". Non si verifica sul core 60/40 reale, che ha
    sempre drawdown nel campione.
    """
    core_returns, overlay_excess = core_returns.align(overlay_excess, join="inner")
    combined = core_returns + overlay_frac * overlay_excess

    core_maxdd = max_drawdown(core_returns)
    combined_maxdd = max_drawdown(combined)
    core_calmar = calmar_ratio(core_returns, periods_per_year=12)
    combined_calmar = calmar_ratio(combined, periods_per_year=12)
    overlay_cum = float((1.0 + overlay_excess).prod() - 1.0)
    corr_full = float(overlay_excess.corr(core_returns))

    stress = core_returns <= core_returns.quantile(stress_quantile)
    if stress.sum() >= 3:
        corr_stress = float(overlay_excess[stress].corr(core_returns[stress]))
    else:
        corr_stress = float("nan")

    crit_a = combined_maxdd < core_maxdd
    crit_b = combined_calmar > core_calmar
    crit_c = overlay_cum >= 0.0
    crit_d = (corr_full <= 0.0) and (not pd.isna(corr_stress)) and (corr_stress <= 0.0)
    return {
        "core_maxdd": core_maxdd, "combined_maxdd": combined_maxdd,
        "core_calmar": core_calmar, "combined_calmar": combined_calmar,
        "overlay_cum": overlay_cum, "corr_full": corr_full, "corr_stress": corr_stress,
        "criterio_a": bool(crit_a), "criterio_b": bool(crit_b),
        "criterio_c": bool(crit_c), "criterio_d": bool(crit_d),
        "verdict": "PASS" if (crit_a and crit_b and crit_c and crit_d) else "FAIL",
    }
