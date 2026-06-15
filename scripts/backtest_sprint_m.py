#!/usr/bin/env python3
"""Sprint M — No-Trade Band su Base Sprint L congelata.

Risponde alla domanda: una no-trade band (ribilancia solo se il turnover
proposto supera una soglia) riduce il turnover di almeno il 50% e porta
la base Sprint L (em=0.5, damp=0.25, mom=0.5 CONGELATI) a Sharpe ≥ 1.00
mantenendo MaxDD ≤ 60/40 e CAGR ≥ 8%?

Ipotesi ex-ante (dichiarata PRIMA del run OOS):
  "Una no-trade band sull'intero vettore pesi (si ribilancia solo se il
  turnover proposto supera una soglia) riduce il turnover di almeno il 50%
  e porta la base Sprint L (em=0.5, damp=0.25, mom=0.5 CONGELATI) a
  Sharpe ≥ 1.00 mantenendo MaxDD ≤ 60/40 e avvicinando CAGR 8%."

Criteri di successo DICHIARATI EX-ANTE (valutati dopo l'unico run OOS):
  (a) Sharpe OOS ≥ Sharpe 60/40 (10bps)
  (b) MaxDD OOS non più profondo del MaxDD 60/40
  (c) CAGR OOS ≥ 8%
Tutti e tre devono essere soddisfatti.
Verifica addizionale dell'ipotesi: turnover con band ≤ 50% del turnover senza band.

Flusso (7 punti):
1. Config + hash + registry (data/research/trials.jsonl)
2. Prezzi Yahoo 20 ETF (2002-2026)
3. IS grid: 3 combo threshold ∈ {0.05, 0.10, 0.20}
   base CONGELATA: TrendMomentumOptimizer(em=0.5, damp=0.25, mom=0.5)
   ATTENZIONE: istanza FRESH di NoTradeBandOverlay per ogni run (statefulness!)
   → engine IS (lag=1, 10bps) → registry → selezione max IS Sharpe
4. OOS UNA volta col threshold congelato
5. Benchmark: SPY B&H, 60/40 10bps, «Sprint L no band» (stessa base SENZA overlay)
6. DSR: n_trials questo hash + cumulativo (+39 da I+J+K+L → 42 prima di IS)
   Verdetto TRE criteri ex-ante (a/b/c), ciascuno stampato pass/fail
   Verifica ipotesi turnover: confronto con/senza band (≥50% riduzione?)
   Report: n. rebalance skippati, turnover comparison
7. Report docs/reports/sprint_m_results.md + chart

Run:
    uv run python scripts/backtest_sprint_m.py
Output:
    docs/backtest/backtest_sprint_m.png
    docs/reports/sprint_m_results.md
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
    NoTradeBandOverlay,
    TrendMomentumOptimizer,
    TrendMomentumParams,
)
from jeanclaude.research import ExperimentConfig, TrialRegistry

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sprint_m")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _REPO_ROOT / "configs" / "experiments" / "2026-06-11-sprint-m-no-trade-band.json"
_OUT_DIR_BACKTEST = _REPO_ROOT / "docs" / "backtest"
_OUT_DIR_REPORTS = _REPO_ROOT / "docs" / "reports"
_OUT_CHART = _OUT_DIR_BACKTEST / "backtest_sprint_m.png"
_OUT_REPORT = _OUT_DIR_REPORTS / "sprint_m_results.md"
_REGISTRY_PATH = _REPO_ROOT / "data" / "research" / "trials.jsonl"

# IS threshold grid
_THRESHOLD_GRID: tuple[float, ...] = (0.05, 0.10, 0.20)
# Base Sprint L params (CONGELATI)
_BASE_EQUITY_MIN: float = 0.5
_BASE_DAMP: float = 0.25
_BASE_MOM: float = 0.5
# Cumulative trials from Sprint I + J + K + L before this experiment
_PRIOR_TRIALS = 39

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
# Data loading — Yahoo prices
# ---------------------------------------------------------------------------

def _load_yahoo_prices(
    rics: tuple[str, ...],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download Yahoo Finance adjusted close prices for universe RICs."""
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
# FixedWeightsOptimizer — 60/40 benchmark
# ---------------------------------------------------------------------------

class FixedWeightsOptimizer:
    """Always returns a fixed normalised weight vector."""

    def __init__(self, weights: dict[str, float]) -> None:
        self._w = pd.Series(weights, dtype=float)
        self._w /= self._w.sum()

    def optimize(self, returns: pd.DataFrame) -> pd.Series:  # noqa: ARG002
        return self._w.reindex(returns.columns, fill_value=0.0)


# ---------------------------------------------------------------------------
# Backtest runner helper
# ---------------------------------------------------------------------------

def _run_backtest(
    returns: pd.DataFrame,
    optimizer,
    bc: BacktestConfig,
    min_history: int,
) -> tuple[pd.Series, pd.DataFrame]:
    eng = BacktestEngine(optimizer, config=bc)
    result = eng.run(returns, min_history=min_history)
    return result.portfolio_returns, result.weights_history


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


def _annual_turnover(weights_history: pd.DataFrame) -> float:
    """Estimate annualised turnover from weights_history (rebalance dates only)."""
    if weights_history.empty or len(weights_history) < 2:
        return float("nan")
    diffs = weights_history.diff().iloc[1:].abs().sum(axis=1)
    avg_per_rebalance = float(diffs.mean())
    n_rebalances = len(diffs)
    years = n_rebalances / 12.0
    return avg_per_rebalance * n_rebalances / max(years, 1e-9)


def _count_skipped_rebalances(weights_history: pd.DataFrame) -> int:
    """Count rebalances where weights did not change (band blocked the trade)."""
    if weights_history.empty or len(weights_history) < 2:
        return 0
    diffs = weights_history.diff().iloc[1:].abs().sum(axis=1)
    return int((diffs == 0.0).sum())


# ---------------------------------------------------------------------------
# IS calibration — grid over threshold, base params CONGELATI
# ---------------------------------------------------------------------------

def _calibrate_is(
    returns_is: pd.DataFrame,
    equity_rics: frozenset[str],
    threshold_grid: tuple[float, ...],
    registry: TrialRegistry,
    config_hash: str,
    min_history: int,
) -> float:
    """IS grid search over threshold (base em=0.5, damp=0.25, mom=0.5 CONGELATI).

    Returns best_threshold with highest IS Sharpe.
    CRITICAL: Fresh NoTradeBandOverlay instance per combo (statefulness).
    Every combo is registered for honest DSR.
    """
    bc_is = BacktestConfig(
        rebalance_freq="ME",
        transaction_cost_bps=10.0,
        execution_lag=1,
    )

    best_sharpe = -np.inf
    best_threshold = threshold_grid[0]

    print(f"\n     IS grid ({len(threshold_grid)} threshold):  base em={_BASE_EQUITY_MIN}, "
          f"damp={_BASE_DAMP}, mom={_BASE_MOM} CONGELATI")
    print(f"     {'Threshold':<15} {'IS Sharpe':>10} {'IS MaxDD':>9} {'Skipped RB':>12}")
    print(f"     {'─' * 49}")

    for thr in threshold_grid:
        # FRESH instance per combo — essential for correct state isolation
        base_params = TrendMomentumParams(
            name=f"M_base_thr{thr:.2f}",
            damp_factor=_BASE_DAMP,
            momentum_strength=_BASE_MOM,
            equity_min=_BASE_EQUITY_MIN,
        )
        base_opt = TrendMomentumOptimizer(base_params, equity_rics=equity_rics)
        overlay = NoTradeBandOverlay(base_opt, threshold=thr)

        try:
            rets, wh = _run_backtest(returns_is, overlay, bc_is, min_history)
        except Exception as exc:
            logger.debug("IS grid threshold=%.2f failed: %s", thr, exc)
            continue
        if rets.empty:
            continue

        sh = sharpe_ratio(rets)
        mdd = max_drawdown(rets)
        skipped = _count_skipped_rebalances(wh)
        print(f"     threshold={thr:.2f}          {sh:>10.3f} {-mdd:>9.1%} {skipped:>12d}")

        registry.record(
            config_hash,
            "sprint-m-no-trade-band",
            {
                "threshold": thr,
                "equity_min": _BASE_EQUITY_MIN,
                "damp": _BASE_DAMP,
                "mom": _BASE_MOM,
            },
            "IS 2005-2011",
            {"sharpe": round(sh, 4), "max_dd": round(float(mdd), 4)},
        )

        if sh > best_sharpe:
            best_sharpe = sh
            best_threshold = thr

    print(f"\n     Selected: threshold={best_threshold:.2f}  (IS Sharpe={best_sharpe:.3f})")
    return best_threshold


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _plot(
    strategy_name: str,
    oos_rets: pd.Series,
    spy_rets: pd.Series,
    bh6040_rets: pd.Series,
    sprint_l_noband_rets: pd.Series,
    oos_start: str,
    oos_end: str,
    initial: float = 10_000.0,
) -> None:
    palette = {
        strategy_name: "#2563eb",
        "SPY B&H": "#9ca3af",
        "60/40 SPY/IEF": "#f59e0b",
        "Sprint L (no band)": "#6b7280",
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1.5]})
    fig.suptitle(
        f"JeanClaude Sprint M — No-Trade Band su Base Sprint L | OOS {oos_start[:4]}–{oos_end[:4]}\n"
        "Yahoo auto_adjust (total return) | IS 2005-2011 | em=0.5 damp=0.25 mom=0.5 CONGELATI | lag=1 | 10bps",
        fontsize=12,
    )

    all_series = [
        ("SPY B&H", spy_rets),
        ("60/40 SPY/IEF", bh6040_rets),
        ("Sprint L (no band)", sprint_l_noband_rets),
        (strategy_name, oos_rets),
    ]

    ax = axes[0]
    for label, rets in all_series:
        if rets.empty:
            continue
        cum = (1 + rets).cumprod() * initial
        final = float(cum.iloc[-1])
        is_benchmark = label in ("SPY B&H", "60/40 SPY/IEF", "Sprint L (no band)")
        ax.plot(
            cum.index, cum,
            color=palette.get(label, "#000"),
            lw=1.2 if is_benchmark else 2.0,
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
        is_benchmark = label in ("SPY B&H", "60/40 SPY/IEF", "Sprint L (no band)")
        if is_benchmark:
            ax.plot(dd.index, dd, color=palette.get(label, "#888"), lw=0.9,
                    linestyle="--", label=label, alpha=0.7)
        else:
            ax.fill_between(dd.index, dd, 0, alpha=0.20, color=palette.get(label, "#000"))
            ax.plot(dd.index, dd, color=palette.get(label, "#000"), lw=1.0, label=label)
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", fontsize=8, ncol=4)
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
    best_threshold: float,
    strategy_name: str,
    all_metrics: list[dict],
    dsr_rows: list[dict],
    n_trials_m: int,
    n_trials_cumulative: int,
    verdict_a: str,
    verdict_b: str,
    verdict_c: str,
    overall_pass: bool,
    turnover_with_band: float,
    turnover_without_band: float,
    turnover_reduction_pct: float,
    turnover_hypothesis_pass: bool,
    skipped_rebalances: int,
    total_rebalances: int,
) -> None:
    _OUT_DIR_REPORTS.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Sprint M — No-Trade Band su Base Sprint L: abbattere il friction cost?\n",
        f"**Config hash:** `{config_hash}`  ",
        f"**Universo:** {len(cfg.universe)} ETF (identico a Sprint L — ex-ante dal design doc)  ",
        f"**IS:** {cfg.is_start} → {cfg.is_end}  |  **OOS:** {cfg.oos_start} → {cfg.oos_end}  ",
        f"**Fonte prezzi:** Yahoo auto_adjust=True (total return)  ",
        f"**Note:** {cfg.notes}  ",
        "",
        "## Ipotesi ex-ante\n",
        "> Una no-trade band sull'intero vettore pesi (si ribilancia solo se il turnover proposto "
        "supera una soglia) riduce il turnover di almeno il 50% e porta la base Sprint L "
        "(em=0.5, damp=0.25, mom=0.5 CONGELATI) a Sharpe ≥ 1.00 mantenendo MaxDD ≤ 60/40 "
        "e avvicinando CAGR 8%.",
        "",
        "## Parametri selezionati (IS)\n",
        f"- `threshold` = {best_threshold:.2f}  (IS grid: {list(_THRESHOLD_GRID)})",
        f"- `equity_min` = {_BASE_EQUITY_MIN}  (CONGELATO da Sprint L)",
        f"- `damp_factor` = {_BASE_DAMP}  (CONGELATO da Sprint L)",
        f"- `momentum_strength` = {_BASE_MOM}  (CONGELATO da Sprint I/L)",
        "",
        "## Statistiche Turnover OOS\n",
        f"| | Turnover annuo |",
        f"|---|---|",
        f"| Sprint M (con band, threshold={best_threshold:.2f}) | {turnover_with_band:.1%} |",
        f"| Sprint L base (senza band) | {turnover_without_band:.1%} |",
        f"| Riduzione | {turnover_reduction_pct:.1%} |",
        f"| Ipotesi ≥50% riduzione | {'**PASS ✓**' if turnover_hypothesis_pass else '**FAIL ✗**'} |",
        "",
        f"- **Rebalance skippati OOS:** {skipped_rebalances} su {total_rebalances} totali "
        f"({skipped_rebalances / max(total_rebalances, 1):.1%})",
        "",
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
        f"**n_trials questo esperimento (config_hash):** {n_trials_m}  ",
        f"**n_trials cumulativi (+ {_PRIOR_TRIALS} da I+J+K+L):** {n_trials_cumulative}  ",
        "",
        "| Strategy | n_trials | DSR | Pass (≥0.75)? |",
        "|---|---|---|---|",
    ]
    for row in dsr_rows:
        flag = "✓" if row["dsr"] >= 0.75 else "✗"
        lines.append(f"| {row['strategy']} | {row['n_trials']} | {row['dsr']:.3f} | {flag} |")
    lines.append("")

    overall_label = "PASS (tutti e tre)" if overall_pass else "FAIL (almeno uno non soddisfatto)"
    lines += [
        "## VERDETTI (tre criteri ex-ante — dichiarati PRIMA del run OOS)\n",
        "> Nessuna selezione a posteriori del criterio — tutti e tre riportati.",
        "",
        "### Criterio (a): Sharpe OOS ≥ Sharpe 60/40\n",
        verdict_a,
        "",
        "### Criterio (b): MaxDD OOS non più profondo del MaxDD 60/40\n",
        verdict_b,
        "",
        "### Criterio (c): CAGR OOS ≥ 8%\n",
        verdict_c,
        "",
        f"### Verdetto complessivo: **{overall_label}**\n",
        "",
        "---",
        "_Report generato automaticamente da scripts/backtest_sprint_m.py_",
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
    print("  JeanClaude Sprint M — No-Trade Band su Base Sprint L congelata")
    print(f"{'═' * 70}\n")

    # ── 1. Config + hash + registry ──────────────────────────────────────
    print("1/7  Caricamento config congelata...")
    cfg = ExperimentConfig.from_json(_CONFIG_PATH)
    config_hash = cfg.config_hash()
    print(f"     Config hash: {config_hash}")
    print(f"     Universo: {len(cfg.universe)} ETF  |  IS: {cfg.is_start}→{cfg.is_end}  |  OOS: {cfg.oos_start}→{cfg.oos_end}")
    print(f"     threshold_grid={list(_THRESHOLD_GRID)}  base: em={_BASE_EQUITY_MIN}, damp={_BASE_DAMP}, mom={_BASE_MOM} CONGELATI")

    registry = TrialRegistry(_REGISTRY_PATH)
    equity_rics = frozenset(cfg.equity_rics)
    MIN_HIST = 252

    # ── 2. Prezzi Yahoo 20 ETF ───────────────────────────────────────────
    print("\n2/7  Download prezzi Yahoo 20 ETF (2002→OOS end)...")
    FETCH_START = "2002-01-01"
    FETCH_END = cfg.oos_end

    prices = _load_yahoo_prices(cfg.universe, start=FETCH_START, end=FETCH_END)
    returns_all = prices.pct_change().iloc[1:]
    returns_all = returns_all.dropna(how="all")

    print(f"     Prezzi: {len(prices)} righe  ({prices.index[0].date()} → {prices.index[-1].date()})")

    returns_is = returns_all.loc[cfg.is_start:cfg.is_end]
    returns_oos = returns_all.loc[cfg.oos_start:cfg.oos_end]
    print(f"     IS rows: {len(returns_is)}  |  OOS rows: {len(returns_oos)}")

    # ── 3. IS grid: 3 combo threshold ──────────────────────────────────
    print(f"\n3/7  IS calibration ({cfg.is_start[:4]}–{cfg.is_end[:4]}) — "
          f"3 combo threshold ∈ {set(_THRESHOLD_GRID)}, base CONGELATA...")

    best_threshold = _calibrate_is(
        returns_is=returns_is,
        equity_rics=equity_rics,
        threshold_grid=_THRESHOLD_GRID,
        registry=registry,
        config_hash=config_hash,
        min_history=MIN_HIST,
    )

    n_trials_m = registry.n_trials(config_hash=config_hash)
    n_trials_cumulative = n_trials_m + _PRIOR_TRIALS

    print(f"\n     Trials questo hash: {n_trials_m}")
    print(f"     Trials cumulativi (+{_PRIOR_TRIALS} da I+J+K+L): {n_trials_cumulative}")

    # ── 4. OOS — una volta con threshold congelato ──────────────────────
    strategy_name = f"Sprint M (band={best_threshold:.2f})"
    print(f"\n4/7  OOS ({cfg.oos_start[:4]}–{cfg.oos_end[:4]}) — run unico: {strategy_name}...")

    bc_oos = BacktestConfig(
        rebalance_freq=cfg.rebalance_freq,
        transaction_cost_bps=cfg.tc_bps,
        execution_lag=cfg.execution_lag,
        min_history=MIN_HIST,
    )

    # FRESH overlay instance for OOS (state isolation!)
    base_params_oos = TrendMomentumParams(
        name=strategy_name,
        damp_factor=_BASE_DAMP,
        momentum_strength=_BASE_MOM,
        equity_min=_BASE_EQUITY_MIN,
    )
    base_opt_oos = TrendMomentumOptimizer(base_params_oos, equity_rics=equity_rics)
    overlay_oos = NoTradeBandOverlay(base_opt_oos, threshold=best_threshold)
    rets_m, wh_m = _run_backtest(returns_oos, overlay_oos, bc_oos, MIN_HIST)

    turnover_with_band = _annual_turnover(wh_m)
    skipped_rebalances = _count_skipped_rebalances(wh_m)
    total_rebalances = len(wh_m) - 1 if len(wh_m) > 1 else 0
    print(f"     Turnover annuo OOS (con band): {turnover_with_band:.1%}")
    print(f"     Rebalance skippati: {skipped_rebalances} su {total_rebalances}")

    # ── 5. Benchmark: SPY B&H, 60/40 10bps, Sprint L (no band) ─────────
    print("\n5/7  Benchmark (SPY B&H, 60/40 10bps, Sprint L no band)...")
    spy_ric, ief_ric = "SPY.P", "IEF.O"

    spy_rets_raw = returns_oos[spy_ric].dropna() if spy_ric in returns_oos.columns else pd.Series(dtype=float)
    common_idx = rets_m.index
    spy_rets_aligned = (
        spy_rets_raw.reindex(common_idx).fillna(0.0)
        if not spy_rets_raw.empty
        else pd.Series(0.0, index=common_idx)
    )

    # 60/40 benchmark via engine (10bps, ME, lag=1)
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
            returns_oos[[spy_ric, ief_ric]].dropna(how="all"), min_history=1
        )
        bh6040_rets = bh6040_result.portfolio_returns

    # Sprint L no band — FRESH instance, same base params, NO overlay
    base_params_noband = TrendMomentumParams(
        name="Sprint L (no band)",
        damp_factor=_BASE_DAMP,
        momentum_strength=_BASE_MOM,
        equity_min=_BASE_EQUITY_MIN,
    )
    opt_noband = TrendMomentumOptimizer(base_params_noband, equity_rics=equity_rics)
    rets_noband, wh_noband = _run_backtest(returns_oos, opt_noband, bc_oos, MIN_HIST)
    turnover_without_band = _annual_turnover(wh_noband)
    print(f"     Turnover annuo OOS (senza band): {turnover_without_band:.1%}")

    # Turnover reduction
    if turnover_without_band > 0 and not np.isnan(turnover_without_band):
        turnover_reduction_pct = (turnover_without_band - turnover_with_band) / turnover_without_band
    else:
        turnover_reduction_pct = float("nan")
    turnover_hypothesis_pass = (
        not np.isnan(turnover_reduction_pct) and turnover_reduction_pct >= 0.50
    )

    print(f"     Riduzione turnover: {turnover_reduction_pct:.1%} "
          f"({'PASS ≥50% ✓' if turnover_hypothesis_pass else 'FAIL <50% ✗'})")

    # ── 6. Metrics + DSR + verdetti ex-ante ─────────────────────────────
    print("\n6/7  Calcolo metriche, DSR, verdetti ex-ante...")

    spy_m = _metrics_row("SPY B&H", spy_rets_aligned)
    bh6040_m = (
        _metrics_row("60/40 SPY/IEF", bh6040_rets)
        if not bh6040_rets.empty
        else _metrics_row("60/40 SPY/IEF", spy_rets_aligned * 0)
    )
    noband_m = _metrics_row("Sprint L (no band)", rets_noband)
    sprint_m_m = _metrics_row(strategy_name, rets_m)

    all_metrics = [spy_m, bh6040_m, noband_m, sprint_m_m]

    print(f"\n  {'Strategy':<45} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8}")
    print(f"  {'─' * 69}")
    for m in all_metrics:
        print(f"  {m['Strategy']:<45} {m['CAGR']:>7.1%} {m['Sharpe']:>7.2f} {-m['MaxDD']:>8.1%}")

    # DSR
    dsr_rows: list[dict] = []
    sr_oos = float(sharpe_ratio(rets_m))
    obs = len(rets_m.dropna())
    sk = float(rets_m.dropna().skew())
    ku = float(rets_m.dropna().kurt()) + 3.0

    print(f"\n  DSR (n_trials_m={n_trials_m}, cumulativo={n_trials_cumulative}):")
    for nt, label in [(n_trials_m, "M"), (n_trials_cumulative, "M+prior")]:
        nt_safe = max(nt, 1)
        dsr = deflated_sharpe_ratio(sr_oos, n_trials=nt_safe, obs=obs, skewness=sk, kurtosis=ku)
        flag = "✓ PASS" if dsr >= 0.75 else "  FAIL"
        tag = f"{strategy_name} (n={label})"
        print(f"  {tag:<55} DSR={dsr:.3f}  {flag}")
        dsr_rows.append({"strategy": tag, "n_trials": nt_safe, "dsr": dsr})

    # ── Three explicit ex-ante verdicts ────────────────────────────────
    bh6040_sharpe = float(bh6040_m["Sharpe"])
    bh6040_maxdd = float(bh6040_m["MaxDD"])
    m_sharpe = float(sprint_m_m["Sharpe"])
    m_maxdd = float(sprint_m_m["MaxDD"])
    m_cagr = float(sprint_m_m["CAGR"])

    pass_a = m_sharpe >= bh6040_sharpe
    pass_b = m_maxdd <= bh6040_maxdd
    pass_c = m_cagr >= 0.08
    overall_pass = pass_a and pass_b and pass_c

    verdict_a = (
        f"**{'PASS ✓' if pass_a else 'FAIL ✗'}** — "
        f"Sprint M Sharpe={m_sharpe:.2f} vs 60/40 Sharpe={bh6040_sharpe:.2f}"
    )
    verdict_b = (
        f"**{'PASS ✓' if pass_b else 'FAIL ✗'}** — "
        f"Sprint M MaxDD={-m_maxdd:.1%} vs 60/40 MaxDD={-bh6040_maxdd:.1%} "
        f"({'non più profondo' if pass_b else 'più profondo'})"
    )
    verdict_c = (
        f"**{'PASS ✓' if pass_c else 'FAIL ✗'}** — "
        f"Sprint M CAGR={m_cagr:.1%} vs soglia=8.0%"
    )

    print(f"\n  ──── VERDETTI EX-ANTE ────")
    print(f"  (a) Sharpe ≥ 60/40 Sharpe:     {'PASS ✓' if pass_a else 'FAIL ✗'}  "
          f"(M={m_sharpe:.2f}, ref={bh6040_sharpe:.2f})")
    print(f"  (b) MaxDD ≤ 60/40 MaxDD:       {'PASS ✓' if pass_b else 'FAIL ✗'}  "
          f"(M={-m_maxdd:.1%}, ref={-bh6040_maxdd:.1%})")
    print(f"  (c) CAGR ≥ 8%:                 {'PASS ✓' if pass_c else 'FAIL ✗'}  "
          f"(M={m_cagr:.1%})")
    overall_label = "PASS (tutti e tre)" if overall_pass else "FAIL (almeno uno non soddisfatto)"
    print(f"  ──── OVERALL: {overall_label} ────")

    print(f"\n  ──── VERIFICA IPOTESI TURNOVER ────")
    print(f"  Turnover con band:    {turnover_with_band:.1%}")
    print(f"  Turnover senza band:  {turnover_without_band:.1%}")
    print(f"  Riduzione:            {turnover_reduction_pct:.1%}  "
          f"({'≥50% PASS ✓' if turnover_hypothesis_pass else '<50% FAIL ✗'})")

    # ── 7. Chart + report ────────────────────────────────────────────────
    print("\n7/7  Chart + report...")
    _plot(
        strategy_name=strategy_name,
        oos_rets=rets_m,
        spy_rets=spy_rets_aligned,
        bh6040_rets=bh6040_rets,
        sprint_l_noband_rets=rets_noband,
        oos_start=cfg.oos_start,
        oos_end=cfg.oos_end,
    )

    _write_report(
        cfg=cfg,
        config_hash=config_hash,
        best_threshold=best_threshold,
        strategy_name=strategy_name,
        all_metrics=all_metrics,
        dsr_rows=dsr_rows,
        n_trials_m=n_trials_m,
        n_trials_cumulative=n_trials_cumulative,
        verdict_a=verdict_a,
        verdict_b=verdict_b,
        verdict_c=verdict_c,
        overall_pass=overall_pass,
        turnover_with_band=turnover_with_band,
        turnover_without_band=turnover_without_band,
        turnover_reduction_pct=turnover_reduction_pct,
        turnover_hypothesis_pass=turnover_hypothesis_pass,
        skipped_rebalances=skipped_rebalances,
        total_rebalances=total_rebalances,
    )

    print(f"  Chart  → {_OUT_CHART}")
    print(f"  Report → {_OUT_REPORT}")
    print(f"  Trials → {_REGISTRY_PATH}")
    print(f"  Tempo totale: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
