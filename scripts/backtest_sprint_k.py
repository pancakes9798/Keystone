#!/usr/bin/env python3
"""Sprint K — Banda Equity Regime-Condizionata su base Sprint I congelato.

Risponde alla domanda: vincolare la quota equity a una banda [floor, cap]
condizionata al regime HMM (alta in EXPANSION, difensiva in CONTRACTION)
alza il CAGR verso SPY mantenendo il MaxDD sotto quello di SPY?

Flusso (7 punti):
1. Config + hash + registry
2. Prezzi Yahoo 20 ETF (2002-2026) + macro Yahoo (VIX, Oil, Copper, Gold, EURUSD)
3. Precompute walk-forward regimes (UNA volta, cache parquet, riusato da tutte le combo)
4. IS grid: 3 mappe banda × 2 basi (Balanced/Aggressive frozen da Sprint I) = 6 combo
   → registry; selezione max IS Sharpe per base
5. OOS una volta per base con la banda congelata
6. Metriche + DSR + due verdetti espliciti + distribuzione regimi + turnover medio
7. Report docs/reports/sprint_k_results.md + chart

Run:
    uv run python scripts/backtest_sprint_k.py
Output:
    docs/backtest/backtest_sprint_k.png
    docs/reports/sprint_k_results.md

Nota causalità regime:
    Il regime al segnale t (ultimo giorno di trading del mese) è fittato su
    macro ≤ as_of(t) dove as_of(t) = t − 1 giorno di trading.
    Questo è l'esatto as_of che l'overlay vede al momento del segnale
    (il BacktestEngine passa history < signal_date, quindi il giorno
    precedente l'ultimo trading day del mese).
    Risultato: zero staleness, strettamente causale anche rispetto
    all'esecuzione che avviene a t+1.
"""
from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning)

from jeanclaude.backtest.engine import BacktestConfig, BacktestEngine
from jeanclaude.backtest.metrics import (
    annualized_return, calmar_ratio, deflated_sharpe_ratio,
    max_drawdown, sharpe_ratio, sortino_ratio,
)
from jeanclaude.portfolio.optimizer import (
    RegimeBandOverlay, TrendMomentumOptimizer, TrendMomentumParams,
    walkforward_regimes,
)
from jeanclaude.research import ExperimentConfig, TrialRegistry

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sprint_k")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _REPO_ROOT / "configs" / "experiments" / "2026-06-10-sprint-k-regime-band.json"
_OUT_DIR_BACKTEST = _REPO_ROOT / "docs" / "backtest"
_OUT_DIR_REPORTS = _REPO_ROOT / "docs" / "reports"
_OUT_CHART = _OUT_DIR_BACKTEST / "backtest_sprint_k.png"
_OUT_REPORT = _OUT_DIR_REPORTS / "sprint_k_results.md"
_REGISTRY_PATH = _REPO_ROOT / "data" / "research" / "trials.jsonl"
_CACHE_DIR = _REPO_ROOT / "data" / "research"

# Macro tickers — same as run_paper_etf.py
_MACRO_TICKERS: dict[str, str] = {
    "VIX":    "^VIX",
    "Oil":    "CL=F",
    "Copper": "HG=F",
    "Gold":   "GC=F",
    "EURUSD": "EURUSD=X",
}

# Banda IS grid (3 mappe) — chiavi = nomi reali di RegimeLabel
_BAND_MAPS: dict[str, dict[str, tuple[float, float]]] = {
    "K1": {   # aggressiva
        "EXPANSION":   (0.8, 1.0),
        "TRANSITION":  (0.4, 0.7),
        "CONTRACTION": (0.0, 0.2),
    },
    "K2": {   # moderata
        "EXPANSION":   (0.7, 1.0),
        "TRANSITION":  (0.5, 0.8),
        "CONTRACTION": (0.0, 0.3),
    },
    "K3": {   # tenue
        "EXPANSION":   (0.9, 1.0),
        "TRANSITION":  (0.5, 0.9),
        "CONTRACTION": (0.1, 0.4),
    },
}

# UNKNOWN → nessuna banda: regime_lookup ritorna "UNKNOWN" se non in mappa → pesi base
# (RegimeBandOverlay usa pesi base per regimi non in bands dict)

# ---------------------------------------------------------------------------
# RIC → Yahoo ticker
# ---------------------------------------------------------------------------
_RIC_TO_YAHOO: dict[str, str] = {}


def _ric_to_yahoo(ric: str) -> str:
    """Convert Refinitiv RIC to Yahoo Finance ticker (strip exchange suffix)."""
    if ric in _RIC_TO_YAHOO:
        return _RIC_TO_YAHOO[ric]
    return ric.split(".")[0]


# ---------------------------------------------------------------------------
# Data loading — Yahoo prices (same as sprint_j, copied here verbatim)
# ---------------------------------------------------------------------------

def _load_yahoo_prices(
    rics: tuple[str, ...],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download Yahoo Finance adjusted close prices for universe RICs.

    Copied from scripts/backtest_sprint_j.py (same logic, no module-level
    side effects — the original cannot be imported safely).
    """
    yahoo_tickers = [_ric_to_yahoo(r) for r in rics]
    raw = yf.download(
        yahoo_tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned empty DataFrame")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()

    close.index = pd.to_datetime(close.index).tz_localize(None)

    yahoo_to_ric = {_ric_to_yahoo(r): r for r in rics}
    close = close.rename(columns=yahoo_to_ric)

    for r in rics:
        if r not in close.columns:
            close[r] = np.nan

    return close[list(rics)].ffill()


# ---------------------------------------------------------------------------
# Macro fetch — Yahoo Finance (same tickers as run_paper_etf.py)
# ---------------------------------------------------------------------------

def _fetch_macro(start: str, end: str) -> pd.DataFrame:
    """Fetch macro indicators from Yahoo Finance with ffill and dropna(how='all')."""
    frames = {}
    for name, ticker in _MACRO_TICKERS.items():
        try:
            raw = yf.download(ticker, start=start, end=end,
                              progress=False, auto_adjust=True)
            if raw.empty:
                logger.warning("Macro fetch empty for %s (%s)", name, ticker)
                continue
            s = raw["Close"].squeeze()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            frames[name] = s
        except Exception as exc:
            logger.warning("Macro fetch failed for %s (%s): %s", name, ticker, exc)
    df = pd.DataFrame(frames).ffill().dropna(how="all")
    return df


# ---------------------------------------------------------------------------
# Regime fit dates — trading day BEFORE each month's last trading day
# ---------------------------------------------------------------------------

def _regime_fit_dates(price_index: pd.DatetimeIndex, start: str, end: str) -> list[pd.Timestamp]:
    """Per ogni mese: il giorno di trading PRECEDENTE all'ultimo giorno di trading
    del mese — è l'as_of che l'overlay vede al segnale (engine passa history < signal).
    Fit su macro ≤ as_of → regime fresco e strettamente causale."""
    idx = price_index[(price_index >= pd.Timestamp(start)) & (price_index <= pd.Timestamp(end))]
    monthly_last = pd.Series(idx, index=idx).resample("ME").last().dropna()
    out = []
    for signal_day in monthly_last.values:
        pos = idx.get_loc(pd.Timestamp(signal_day))
        if pos >= 1:
            out.append(pd.Timestamp(idx[pos - 1]))
    return out


# ---------------------------------------------------------------------------
# regime_lookup closure — ffill semantics
# ---------------------------------------------------------------------------

def _make_regime_lookup(regimes: pd.Series):
    """Return a callable date → regime_name with ffill semantics.

    For a date not exactly in the precomputed Series, returns the regime of
    the most recent precomputed date ≤ as_of. Returns 'UNKNOWN' if none.
    """
    # Sort index to enable searchsorted
    sr = regimes.sort_index()

    def lookup(as_of: pd.Timestamp) -> str:
        if sr.empty:
            return "UNKNOWN"
        pos = sr.index.searchsorted(as_of, side="right") - 1
        if pos < 0:
            return "UNKNOWN"
        return str(sr.iloc[pos])

    return lookup


# ---------------------------------------------------------------------------
# Backtest runner helper
# ---------------------------------------------------------------------------

def _run_backtest(
    returns: pd.DataFrame,
    optimizer,
    bc: BacktestConfig,
    min_history: int,
) -> tuple["pd.Series", "pd.DataFrame"]:
    eng = BacktestEngine(optimizer, config=bc)
    result = eng.run(returns, min_history=min_history)
    return result.portfolio_returns, result.weights_history


# ---------------------------------------------------------------------------
# FixedWeightsOptimizer — 60/40 benchmark
# (copied from backtest_sprint_j.py — same class, no import side-effects)
# ---------------------------------------------------------------------------

class FixedWeightsOptimizer:
    """Always returns a fixed normalised weight vector."""

    def __init__(self, weights: dict[str, float]) -> None:
        self._w = pd.Series(weights, dtype=float)
        self._w /= self._w.sum()

    def optimize(self, returns: pd.DataFrame) -> pd.Series:  # noqa: ARG002
        return self._w.reindex(returns.columns, fill_value=0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drawdown_series(rets: pd.Series) -> pd.Series:
    cum = (1 + rets).cumprod()
    return (cum - cum.cummax()) / cum.cummax()


def _metrics_row(name: str, rets: pd.Series) -> dict:
    final_val = float((1 + rets).prod())
    return {
        "Strategy": name,
        "CAGR": annualized_return(rets),
        "Sharpe": sharpe_ratio(rets),
        "Sortino": sortino_ratio(rets),
        "MaxDD": max_drawdown(rets),
        "Calmar": calmar_ratio(rets),
        "$10k_final": 10_000 * final_val,
    }


def _avg_annual_turnover(weights_history: pd.DataFrame) -> float:
    """Average annual one-way turnover from weights history diffs.

    Turnover at each rebalance = 0.5 * sum(|w_new - w_prev|).
    Annual = mean_rebalance_turnover * rebalances_per_year.
    """
    if weights_history.empty or len(weights_history) < 2:
        return float("nan")
    diffs = weights_history.diff().iloc[1:].abs().sum(axis=1) / 2.0
    # Estimate rebalances per year from index spacing
    days_between = (weights_history.index[-1] - weights_history.index[0]).days
    years = max(days_between / 365.25, 1e-6)
    rebalances_per_year = (len(weights_history) - 1) / years
    return float(diffs.mean() * rebalances_per_year)


def _regime_distribution(regimes: pd.Series) -> dict[str, int]:
    """Count occurrences of each regime label."""
    return regimes.value_counts().to_dict()


# ---------------------------------------------------------------------------
# IS calibration — grid over band maps for one arm
# ---------------------------------------------------------------------------

def _calibrate_is_arm(
    arm_name: str,
    base_params: TrendMomentumParams,
    equity_rics: frozenset[str],
    returns_is: pd.DataFrame,
    regime_lookup,
    band_maps: dict[str, dict[str, tuple[float, float]]],
    registry: TrialRegistry,
    config_hash: str,
    min_history: int,
) -> tuple[str, dict[str, tuple[float, float]]]:
    """IS grid search over band maps for one arm.

    Returns (best_band_name, best_band_map) — max IS Sharpe.
    Every combo registered for honest DSR.
    """
    bc_is = BacktestConfig(
        rebalance_freq="ME",
        transaction_cost_bps=10.0,
        execution_lag=1,
        max_leverage=1.0,
    )

    best_sharpe = -np.inf
    best_band_name = list(band_maps.keys())[0]
    best_band_map = band_maps[best_band_name]

    for band_name, band_map in band_maps.items():
        base_opt = TrendMomentumOptimizer(base_params, equity_rics=equity_rics)
        overlay = RegimeBandOverlay(
            base_opt,
            equity_rics=equity_rics,
            bands=band_map,
            regime_lookup=regime_lookup,
        )
        try:
            rets, _ = _run_backtest(returns_is, overlay, bc_is, min_history)
        except Exception as exc:
            logger.debug("IS grid arm=%s band=%s failed: %s", arm_name, band_name, exc)
            continue
        if rets.empty:
            continue

        sh = sharpe_ratio(rets)
        logger.info("IS grid arm=%s  band=%s  Sharpe=%.3f", arm_name, band_name, sh)

        # Record every combo (honest DSR)
        registry.record(
            config_hash,
            f"sprint-k-regime-band-{arm_name}",
            {
                "arm": arm_name,
                "band_map": band_name,
                "bands": {k: list(v) for k, v in band_map.items()},
                "damp": base_params.damp_factor,
                "mom": base_params.momentum_strength,
            },
            "IS 2005-2011",
            {"sharpe": round(sh, 4)},
        )

        if sh > best_sharpe:
            best_sharpe = sh
            best_band_name = band_name
            best_band_map = band_map

    logger.info("IS arm=%s best_band=%s  IS Sharpe=%.3f", arm_name, best_band_name, best_sharpe)
    return best_band_name, best_band_map


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _plot(
    oos_results: list[tuple[str, pd.Series]],
    spy_rets: pd.Series,
    bh6040_rets: pd.Series,
    sprint_i_bal_rets: pd.Series,
    oos_start: str,
    oos_end: str,
    initial: float = 10_000.0,
) -> None:
    palette = {
        "Arm A (Balanced+Band)": "#2563eb",
        "Arm B (Aggressive+Band)": "#dc2626",
        "SPY B&H": "#9ca3af",
        "60/40 SPY/IEF": "#f59e0b",
        "Sprint I Balanced (unlevered)": "#6b7280",
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1.5]})
    fig.suptitle(
        f"JeanClaude Sprint K — Banda Equity Regime-Condizionata | OOS {oos_start[:4]}–{oos_end[:4]}\n"
        "Yahoo auto_adjust (total return) | HMM walk-forward causale | IS 2005-2011 | no leva",
        fontsize=12,
    )

    all_series = [
        ("SPY B&H", spy_rets),
        ("60/40 SPY/IEF", bh6040_rets),
        ("Sprint I Balanced (unlevered)", sprint_i_bal_rets),
        *oos_results,
    ]

    ax = axes[0]
    for label, rets in all_series:
        if rets.empty:
            continue
        cum = (1 + rets).cumprod() * initial
        final = float(cum.iloc[-1])
        is_benchmark = label in ("SPY B&H", "60/40 SPY/IEF", "Sprint I Balanced (unlevered)")
        ax.plot(
            cum.index, cum,
            color=palette.get(label, "#000"),
            lw=1.2 if is_benchmark else 1.8,
            linestyle="--" if is_benchmark else "-",
            label=f"{label}  →  ${final:,.0f}",
            alpha=0.85,
        )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_ylabel(f"Valore portafoglio ($ da ${initial:,.0f})")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = axes[1]
    for label, rets in all_series:
        if rets.empty:
            continue
        dd = _drawdown_series(rets) * 100
        is_benchmark = label in ("SPY B&H", "60/40 SPY/IEF", "Sprint I Balanced (unlevered)")
        if is_benchmark:
            ax.plot(dd.index, dd, color=palette.get(label, "#888"), lw=0.9,
                    linestyle="--", label=label, alpha=0.7)
        else:
            ax.fill_between(dd.index, dd, 0, alpha=0.20, color=palette.get(label, "#000"))
            ax.plot(dd.index, dd, color=palette.get(label, "#000"), lw=1.0, label=label)
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    _OUT_DIR_BACKTEST.mkdir(parents=True, exist_ok=True)
    plt.savefig(_OUT_CHART, dpi=150, bbox_inches="tight")
    logger.info("Chart: %s", _OUT_CHART)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(
    cfg: ExperimentConfig,
    config_hash: str,
    frozen_bands: dict[str, tuple[str, dict[str, tuple[float, float]]]],
    all_metrics: list[dict],
    dsr_rows: list[dict],
    n_trials_k: int,
    n_trials_cumulative: int,
    verdicts: dict[str, str],
    regime_dist_is: dict[str, int],
    regime_dist_oos: dict[str, int],
    turnover_table: list[dict],
) -> None:
    _OUT_DIR_REPORTS.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Sprint K — Banda Equity Regime-Condizionata: CAGR vs SPY con DD contenuto?\n",
        f"**Config hash:** `{config_hash}`  ",
        f"**Universo:** {len(cfg.universe)} ETF (identico a Sprint I — ex-ante dal design doc)  ",
        f"**IS:** {cfg.is_start} → {cfg.is_end}  |  **OOS:** {cfg.oos_start} → {cfg.oos_end}  ",
        f"**Fonte prezzi:** Yahoo auto_adjust=True (total return)  ",
        f"**Macro:** VIX, Oil, Copper, Gold, EURUSD (Yahoo Finance)  ",
        f"**Note:** {cfg.notes}  ",
        "",
        "## Parametri base congelati (da Sprint I — NON ricalibrati)\n",
        "- Arm A (Balanced): damp=0.25, mom=0.5",
        "- Arm B (Aggressive): damp=0.25, mom=0.7",
        "",
        "**IS grid → mappa banda congelata:**  ",
    ]
    for arm, (band_name, band_map) in frozen_bands.items():
        lines.append(f"- {arm}: mappa = {band_name}")
        for regime, (floor, cap) in band_map.items():
            lines.append(f"  - {regime}: [{floor:.1f}, {cap:.1f}]")
    lines.append("")

    lines += [
        "## Distribuzione Regimi\n",
        "### IS (2005-2011)\n",
        "| Regime | # date rebalance |",
        "|---|---|",
    ]
    for regime, count in sorted(regime_dist_is.items()):
        lines.append(f"| {regime} | {count} |")
    lines.append("")

    lines += [
        "### OOS (2012-2026)\n",
        "| Regime | # date rebalance |",
        "|---|---|",
    ]
    for regime, count in sorted(regime_dist_oos.items()):
        lines.append(f"| {regime} | {count} |")
    lines.append("")

    lines += [
        "## Turnover Medio Annuo\n",
        "| Strategy | Turnover annuo (one-way) |",
        "|---|---|",
    ]
    for row in turnover_table:
        lines.append(f"| {row['strategy']} | {row['turnover']:.1%} |")
    lines.append("")
    lines.append(
        "> **Nota whipsaw:** alto turnover con basso miglioramento sullo Sharpe indica che "
        "le band producono segnali contraddittori (whipsaw) — la modalità di fallimento attesa."
    )
    lines.append("")

    lines += [
        "## Metriche OOS\n",
        "| Strategy | CAGR | Sharpe | Sortino | MaxDD | Calmar | $10k→ |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in all_metrics:
        lines.append(
            f"| {m['Strategy']} | {m['CAGR']:.1%} | {m['Sharpe']:.2f} | "
            f"{m['Sortino']:.2f} | {-m['MaxDD']:.1%} | {m['Calmar']:.2f} | "
            f"${m['$10k_final']:,.0f} |"
        )
    lines.append("")

    lines += [
        "## Deflated Sharpe Ratio\n",
        f"**n_trials questo esperimento (config_hash):** {n_trials_k}  ",
        f"**n_trials cumulativi (+ 27 da I+J):** {n_trials_cumulative}  ",
        "",
        "| Strategy | n_trials | DSR | Pass (≥0.75)? |",
        "|---|---|---|---|",
    ]
    for row in dsr_rows:
        flag = "✓" if row["dsr"] >= 0.75 else "✗"
        lines.append(f"| {row['strategy']} | {row['n_trials']} | {row['dsr']:.3f} | {flag} |")
    lines.append("")

    lines += [
        "## VERDETTI (due criteri espliciti ex-ante)\n",
        "> Nessuna selezione a posteriori del criterio — entrambi riportati.",
        "",
        "### Criterio A: CAGR OOS ≥ SPY CAGR?\n",
        verdicts["cagr"],
        "",
        "### Criterio B: Sharpe ≥ SPY Sharpe E MaxDD < SPY MaxDD?\n",
        verdicts["sharpe_dd"],
        "",
        "### Confronto con Sprint I (contributo della banda)\n",
        verdicts["vs_sprint_i"],
        "",
        "---",
        "_Report generato automaticamente da scripts/backtest_sprint_k.py_",
    ]

    _OUT_REPORT.write_text("\n".join(lines))
    logger.info("Report: %s", _OUT_REPORT)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Determinism
    np.random.seed(42)
    import random
    random.seed(42)

    t0 = time.time()
    print(f"\n{'═' * 70}")
    print("  JeanClaude Sprint K — Banda Equity Regime-Condizionata")
    print(f"{'═' * 70}\n")

    # ── 1. Config + hash + registry ──────────────────────────────────────
    print("1/7  Caricamento config congelata...")
    cfg = ExperimentConfig.from_json(_CONFIG_PATH)
    config_hash = cfg.config_hash()
    print(f"     Config hash: {config_hash}")
    print(f"     Universo: {len(cfg.universe)} ETF  |  IS: {cfg.is_start}→{cfg.is_end}  |  OOS: {cfg.oos_start}→{cfg.oos_end}")

    registry = TrialRegistry(_REGISTRY_PATH)
    equity_rics = frozenset(cfg.equity_rics)
    MIN_HIST = 252

    # ── 2. Prezzi Yahoo 20 ETF + macro Yahoo ─────────────────────────────
    print("2/7  Download prezzi Yahoo 20 ETF + macro Yahoo...")
    FETCH_START = "2002-01-01"
    FETCH_END = cfg.oos_end

    prices = _load_yahoo_prices(cfg.universe, start=FETCH_START, end=FETCH_END)
    returns_all = prices.pct_change().iloc[1:]
    returns_all = returns_all.dropna(how="all")

    print(f"     Prezzi: {len(prices)} righe  ({prices.index[0].date()} → {prices.index[-1].date()})")

    macro = _fetch_macro(FETCH_START, FETCH_END)
    print(f"     Macro: {len(macro)} righe, features={list(macro.columns)}")

    returns_is = returns_all.loc[cfg.is_start:cfg.is_end]
    returns_oos = returns_all.loc[cfg.oos_start:cfg.oos_end]

    # ── 3. Precompute walk-forward regimes ───────────────────────────────
    print("3/7  Precompute walk-forward regimes (HMM causale, cache parquet)...")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"wf_regimes_{config_hash}_42.parquet"

    # Regime fit dates: trading day before each month's last trading day (exact as_of for the overlay)
    reb_dates = _regime_fit_dates(prices.index, cfg.is_start, cfg.oos_end)
    print(f"     Date rebalance precompute: {len(reb_dates)} ({reb_dates[0].date()} → {reb_dates[-1].date()})")

    regimes_all = walkforward_regimes(
        macro, reb_dates, random_state=42, cache_path=cache_path,
    )
    print(f"     Regimes precomputed: {len(regimes_all)}")
    dist_all = _regime_distribution(regimes_all)
    print(f"     Distribuzione completa: {dist_all}")

    # Regime distributions for IS and OOS
    regimes_is = regimes_all[
        (regimes_all.index >= cfg.is_start) & (regimes_all.index <= cfg.is_end)
    ]
    regimes_oos = regimes_all[
        (regimes_all.index >= cfg.oos_start) & (regimes_all.index <= cfg.oos_end)
    ]
    regime_dist_is = _regime_distribution(regimes_is)
    regime_dist_oos = _regime_distribution(regimes_oos)
    print(f"     IS: {regime_dist_is}  |  OOS: {regime_dist_oos}")

    # Build regime_lookup closure (ffill semantics)
    regime_lookup = _make_regime_lookup(regimes_all)

    # ── 4. IS grid: 3 mappe × 2 basi = 6 combo ───────────────────────────
    print(f"4/7  IS calibration ({cfg.is_start[:4]}–{cfg.is_end[:4]}) — 3 mappe × 2 basi = 6 combo...")

    # Frozen base params from Sprint I (NOT recalibrated)
    params_balanced = TrendMomentumParams(
        name="Balanced", damp_factor=0.25, momentum_strength=0.5,
    )
    params_aggressive = TrendMomentumParams(
        name="Aggressive", damp_factor=0.25, momentum_strength=0.7,
    )

    best_band_name_a, best_band_map_a = _calibrate_is_arm(
        "ArmA-Balanced", params_balanced, equity_rics,
        returns_is, regime_lookup, _BAND_MAPS, registry, config_hash, MIN_HIST,
    )
    best_band_name_b, best_band_map_b = _calibrate_is_arm(
        "ArmB-Aggressive", params_aggressive, equity_rics,
        returns_is, regime_lookup, _BAND_MAPS, registry, config_hash, MIN_HIST,
    )

    frozen_bands = {
        "Arm A (Balanced)": (best_band_name_a, best_band_map_a),
        "Arm B (Aggressive)": (best_band_name_b, best_band_map_b),
    }

    n_trials_k = registry.n_trials(config_hash=config_hash)
    n_trials_cumulative = n_trials_k + 27  # + 27 from Sprint I + J (documented)

    print(f"     Arm A: best_band={best_band_name_a}")
    print(f"     Arm B: best_band={best_band_name_b}")
    print(f"     Trials (questo esperimento): {n_trials_k}")
    print(f"     Trials cumulativi (+ 27 I+J): {n_trials_cumulative}")

    # ── 5. OOS — una volta per arm con banda congelata ────────────────────
    print(f"5/7  OOS ({cfg.oos_start[:4]}–{cfg.oos_end[:4]}) — un run per arm...")
    bc_oos = BacktestConfig(
        rebalance_freq=cfg.rebalance_freq,
        transaction_cost_bps=cfg.tc_bps,
        execution_lag=cfg.execution_lag,
        min_history=MIN_HIST,
        max_leverage=1.0,
    )

    # Arm A: Balanced base + best band
    opt_a = RegimeBandOverlay(
        TrendMomentumOptimizer(params_balanced, equity_rics=equity_rics),
        equity_rics=equity_rics,
        bands=best_band_map_a,
        regime_lookup=regime_lookup,
    )
    rets_a, wh_a = _run_backtest(returns_oos, opt_a, bc_oos, MIN_HIST)

    # Arm B: Aggressive base + best band
    opt_b = RegimeBandOverlay(
        TrendMomentumOptimizer(params_aggressive, equity_rics=equity_rics),
        equity_rics=equity_rics,
        bands=best_band_map_b,
        regime_lookup=regime_lookup,
    )
    rets_b, wh_b = _run_backtest(returns_oos, opt_b, bc_oos, MIN_HIST)

    logger.info("OOS Arm A done: Sharpe=%.3f", sharpe_ratio(rets_a))
    logger.info("OOS Arm B done: Sharpe=%.3f", sharpe_ratio(rets_b))

    # Turnover
    to_a = _avg_annual_turnover(wh_a)
    to_b = _avg_annual_turnover(wh_b)
    print(f"     Arm A turnover: {to_a:.1%}/anno  |  Arm B: {to_b:.1%}/anno")

    # ── Benchmarks ────────────────────────────────────────────────────────
    print("     Benchmark (SPY B&H, 60/40, Sprint I Balanced unlevered)...")
    spy_ric, ief_ric = "SPY.P", "IEF.O"

    spy_rets_raw = returns_oos[spy_ric].dropna() if spy_ric in returns_oos.columns else pd.Series(dtype=float)
    common_idx = rets_a.index
    spy_rets_aligned = (spy_rets_raw.reindex(common_idx).fillna(0.0)
                        if not spy_rets_raw.empty else spy_rets_raw)

    bh6040_rets = pd.Series(dtype=float)
    if spy_ric in returns_oos.columns and ief_ric in returns_oos.columns:
        opt6040 = FixedWeightsOptimizer({spy_ric: 0.6, ief_ric: 0.4})
        bc_bench = BacktestConfig(
            rebalance_freq="ME",
            transaction_cost_bps=10.0,
            execution_lag=1,
            min_history=1,
        )
        bh6040_result = BacktestEngine(opt6040, config=bc_bench).run(
            returns_oos[[spy_ric, ief_ric]].dropna(how="all"), min_history=1,
        )
        bh6040_rets = bh6040_result.portfolio_returns

    # Sprint I Balanced unlevered: damp=0.25, mom=0.5, no overlay
    params_sprint_i_bal = TrendMomentumParams(
        name="Sprint I Balanced (unlevered)", damp_factor=0.25, momentum_strength=0.5,
    )
    bc_unlevered = BacktestConfig(
        rebalance_freq=cfg.rebalance_freq,
        transaction_cost_bps=cfg.tc_bps,
        execution_lag=cfg.execution_lag,
        min_history=MIN_HIST,
    )
    opt_sprint_i_bal = TrendMomentumOptimizer(params_sprint_i_bal, equity_rics=equity_rics)
    sprint_i_bal_rets, wh_si = _run_backtest(returns_oos, opt_sprint_i_bal, bc_unlevered, MIN_HIST)
    to_si = _avg_annual_turnover(wh_si)

    # Turnover table
    turnover_table = [
        {"strategy": "Arm A (Balanced+Band)", "turnover": to_a},
        {"strategy": "Arm B (Aggressive+Band)", "turnover": to_b},
        {"strategy": "Sprint I Balanced (unlevered)", "turnover": to_si},
    ]

    # ── 6. Metrics + DSR + verdetti ───────────────────────────────────────
    print("6/7  Calcolo metriche, DSR, verdetti...")

    spy_m = _metrics_row("SPY B&H", spy_rets_aligned)
    bh6040_m = (_metrics_row("60/40 SPY/IEF", bh6040_rets)
                if not bh6040_rets.empty else _metrics_row("60/40 SPY/IEF", spy_rets_aligned * 0))
    spi_bal_m = _metrics_row("Sprint I Balanced (unlevered)", sprint_i_bal_rets)
    arm_a_m = _metrics_row("Arm A (Balanced+Band)", rets_a)
    arm_b_m = _metrics_row("Arm B (Aggressive+Band)", rets_b)

    all_metrics = [spy_m, bh6040_m, spi_bal_m, arm_a_m, arm_b_m]

    print(f"\n  {'Strategy':<42} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8}")
    print(f"  {'─' * 66}")
    for m in all_metrics:
        print(f"  {m['Strategy']:<42} {m['CAGR']:>7.1%} {m['Sharpe']:>7.2f} {-m['MaxDD']:>8.1%}")

    # DSR
    spy_sh = float(spy_m["Sharpe"])
    dsr_rows: list[dict] = []
    arm_results = [
        ("Arm A (Balanced+Band)", rets_a),
        ("Arm B (Aggressive+Band)", rets_b),
    ]

    print(f"\n  DSR (n_trials_k={n_trials_k}, cumulativo={n_trials_cumulative}):")
    for name, rets in arm_results:
        sr_oos = float(sharpe_ratio(rets))
        obs = len(rets.dropna())
        sk = float(rets.dropna().skew())
        ku = float(rets.dropna().kurt()) + 3.0

        for nt, label in [(n_trials_k, "K"), (n_trials_cumulative, "K+SpI+SpJ")]:
            nt = max(nt, 1)
            dsr = deflated_sharpe_ratio(sr_oos, n_trials=nt, obs=obs, skewness=sk, kurtosis=ku)
            flag = "✓ PASS" if dsr >= 0.75 else "  FAIL"
            tag = f"{name} (n={label})"
            print(f"  {tag:<56} DSR={dsr:.3f}  {flag}")
            dsr_rows.append({"strategy": tag, "n_trials": nt, "dsr": dsr})

    # Verdetti
    spy_cagr = float(spy_m["CAGR"])
    spy_sharpe = float(spy_m["Sharpe"])
    spy_maxdd = float(spy_m["MaxDD"])

    verdetto_cagr_lines = ["| Arm | CAGR | SPY CAGR | Batte SPY? |", "|---|---|---|---|"]
    for m in [arm_a_m, arm_b_m]:
        beats = m["CAGR"] >= spy_cagr
        verdetto_cagr_lines.append(
            f"| {m['Strategy']} | {m['CAGR']:.1%} | {spy_cagr:.1%} | {'SI' if beats else 'NO'} |"
        )
    verdetto_cagr = "\n".join(verdetto_cagr_lines)

    verdetto_shdd_lines = [
        "| Arm | Sharpe | SPY Sharpe | MaxDD | SPY MaxDD | Criterio B? |",
        "|---|---|---|---|---|---|",
    ]
    for m in [arm_a_m, arm_b_m]:
        ok = m["Sharpe"] >= spy_sharpe and m["MaxDD"] < spy_maxdd
        verdetto_shdd_lines.append(
            f"| {m['Strategy']} | {m['Sharpe']:.2f} | {spy_sharpe:.2f} | "
            f"{-m['MaxDD']:.1%} | {-spy_maxdd:.1%} | {'SI' if ok else 'NO'} |"
        )
    verdetto_shdd = "\n".join(verdetto_shdd_lines)

    # Confronto con Sprint I (isolate contributo banda)
    spi_cagr = float(spi_bal_m["CAGR"])
    spi_sharpe = float(spi_bal_m["Sharpe"])
    spi_dd = float(spi_bal_m["MaxDD"])
    vs_spi_lines = [
        "| Arm | CAGR | SpI CAGR | Sharpe | SpI Sharpe | MaxDD | SpI MaxDD | Migliora SpI? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in [arm_a_m, arm_b_m]:
        better = (m["CAGR"] >= spi_cagr and
                  (m["Sharpe"] >= spi_sharpe or m["MaxDD"] < spi_dd))
        vs_spi_lines.append(
            f"| {m['Strategy']} | {m['CAGR']:.1%} | {spi_cagr:.1%} | "
            f"{m['Sharpe']:.2f} | {spi_sharpe:.2f} | "
            f"{-m['MaxDD']:.1%} | {-spi_dd:.1%} | "
            f"{'SI' if better else 'NO'} |"
        )
    verdetto_vs_spi = "\n".join(vs_spi_lines)

    verdicts = {
        "cagr": verdetto_cagr,
        "sharpe_dd": verdetto_shdd,
        "vs_sprint_i": verdetto_vs_spi,
    }

    # ── 7. Chart + report ─────────────────────────────────────────────────
    print("\n7/7  Chart + report...")
    oos_results = [
        ("Arm A (Balanced+Band)", rets_a),
        ("Arm B (Aggressive+Band)", rets_b),
    ]
    _plot(oos_results, spy_rets_aligned, bh6040_rets, sprint_i_bal_rets,
          cfg.oos_start, cfg.oos_end)

    _write_report(
        cfg=cfg,
        config_hash=config_hash,
        frozen_bands=frozen_bands,
        all_metrics=all_metrics,
        dsr_rows=dsr_rows,
        n_trials_k=n_trials_k,
        n_trials_cumulative=n_trials_cumulative,
        verdicts=verdicts,
        regime_dist_is=regime_dist_is,
        regime_dist_oos=regime_dist_oos,
        turnover_table=turnover_table,
    )

    print(f"  Chart  → {_OUT_CHART}")
    print(f"  Report → {_OUT_REPORT}")
    print(f"  Cache  → {cache_path}")
    print(f"  Trials → {_REGISTRY_PATH}")
    print(f"  Tempo totale: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
