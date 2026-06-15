#!/usr/bin/env python3
"""Sprint J — Vol Targeting + Leva su base Sprint I congelato.

Risponde alla domanda: HRP+trend+momentum + vol targeting/leva batte SPY B&H?

Flusso (7 punti):
1. Config + hash + registry (data/research/trials.jsonl)
2. Prezzi Yahoo 20 ETF (2002-2026) + funding DGS3MO via FRED
3. IS calibration 2005-2011 — griglia SOLO su target_vol {0.08,0.10,0.12}
   × Arm A (Balanced base) e Arm B (Aggressive + equity_min=0.60)
4. OOS UNA volta per arm con il target_vol congelato
5. Benchmark: SPY B&H, 60/40, Sprint I Balanced unlevered
6. Metriche + DSR + due verdetti espliciti + report
7. Assert leva ≤ 2.0+eps su ogni risultato

Run:
    uv run python scripts/backtest_sprint_j.py
Output:
    docs/backtest/backtest_sprint_j.png
    docs/reports/sprint_j_results.md
"""
from __future__ import annotations

import logging
import os
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
    TrendMomentumOptimizer, TrendMomentumParams, VolTargetOverlay,
)
from jeanclaude.research import ExperimentConfig, TrialRegistry

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sprint_j")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _REPO_ROOT / "configs" / "experiments" / "2026-06-10-sprint-j-voltarget.json"
_OUT_DIR_BACKTEST = _REPO_ROOT / "docs" / "backtest"
_OUT_DIR_REPORTS = _REPO_ROOT / "docs" / "reports"
_OUT_CHART = _OUT_DIR_BACKTEST / "backtest_sprint_j.png"
_OUT_REPORT = _OUT_DIR_REPORTS / "sprint_j_results.md"
_REGISTRY_PATH = _REPO_ROOT / "data" / "research" / "trials.jsonl"

# Vol targeting grid (IS-only — not OOS)
_TV_GRID: tuple[float, ...] = (0.08, 0.10, 0.12)

# ---------------------------------------------------------------------------
# RIC → Yahoo ticker (copied from backtest_sprint_i.py — same mapping logic)
# ---------------------------------------------------------------------------
_RIC_TO_YAHOO: dict[str, str] = {}


def _ric_to_yahoo(ric: str) -> str:
    """Convert Refinitiv RIC to Yahoo Finance ticker (strip exchange suffix)."""
    if ric in _RIC_TO_YAHOO:
        return _RIC_TO_YAHOO[ric]
    return ric.split(".")[0]


# ---------------------------------------------------------------------------
# Data loading — Yahoo prices (same as sprint_i._load_yahoo_prices, copied
# here verbatim so this script has no runtime import side-effects from sprint_i)
# ---------------------------------------------------------------------------

def _load_yahoo_prices(
    rics: tuple[str, ...],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download Yahoo Finance adjusted close prices for universe RICs.

    Copied from scripts/backtest_sprint_i.py (same logic, no module-level
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
# Funding series — DGS3MO via FRED + flat fallback
# ---------------------------------------------------------------------------

def _load_funding_series(start: str, end: str) -> tuple[pd.Series, str]:
    """Load DGS3MO from FRED, convert to fraction + 50bps spread, bdate ffill.

    Returns (series, source_description).  Falls back to flat 0.02 if FRED_API_KEY
    is missing or the fetch fails.
    """
    fred_key = os.environ.get("FRED_API_KEY", "")

    if not fred_key:
        logger.warning(
            "FRED_API_KEY not set — using FLAT FALLBACK 2%% for funding rate. "
            "Results will differ from FRED-sourced run."
        )
        bday_idx = pd.bdate_range(start=start, end=end)
        flat = pd.Series(0.02, index=bday_idx, name="DGS3MO_funding")
        return flat, "FLAT_FALLBACK_2pct (FRED_API_KEY missing)"

    try:
        from jeanclaude.data.ingestion.fred import FREDSource
        src = FREDSource(publication_lag_days=1)
        df = src.get_macro(["DGS3MO"], start=start, end=end)
        raw = df["DGS3MO"]
        # Convert percent to fraction and add 50bps spread
        funding = raw / 100.0 + 0.005
        # Re-index to bdate range and ffill any remaining gaps
        bday_idx = pd.bdate_range(start=start, end=end)
        funding = funding.reindex(bday_idx).ffill().fillna(0.02 + 0.005)
        funding.name = "DGS3MO_funding"
        logger.info("Funding series loaded from FRED: DGS3MO+50bps, %d obs", len(funding))
        return funding, "FRED DGS3MO/100 + 0.005 (publication_lag=1)"
    except Exception as exc:
        logger.warning(
            "FRED fetch failed (%s) — using FLAT FALLBACK 2%% for funding rate.", exc
        )
        bday_idx = pd.bdate_range(start=start, end=end)
        flat = pd.Series(0.02, index=bday_idx, name="DGS3MO_funding")
        return flat, f"FLAT_FALLBACK_2pct (FRED error: {exc})"


# ---------------------------------------------------------------------------
# Backtest runner helper
# ---------------------------------------------------------------------------

def _run_backtest(
    returns: pd.DataFrame,
    optimizer,
    bc: BacktestConfig,
    min_history: int,
) -> "pd.Series":
    eng = BacktestEngine(optimizer, config=bc)
    result = eng.run(returns, min_history=min_history)
    return result.portfolio_returns, result.weights_history


# ---------------------------------------------------------------------------
# IS calibration — grid only on target_vol
# ---------------------------------------------------------------------------

def _calibrate_is_arm(
    arm_name: str,
    returns_is: pd.DataFrame,
    base_params: TrendMomentumParams,
    equity_rics: frozenset[str],
    funding: pd.Series,
    tv_grid: tuple[float, ...],
    registry: TrialRegistry,
    config_hash: str,
    min_history: int,
) -> float:
    """IS grid search over target_vol for one arm.

    Returns the target_vol with highest IS Sharpe (net of funding).
    Every combo is registered for honest DSR.
    """
    bc_is = BacktestConfig(
        rebalance_freq="ME",
        transaction_cost_bps=10.0,
        execution_lag=1,
        max_leverage=2.0,
        funding_rate_annual=funding,
    )

    best_sharpe = -np.inf
    best_tv = tv_grid[0]

    for tv in tv_grid:
        base_opt = TrendMomentumOptimizer(base_params, equity_rics=equity_rics)
        overlay = VolTargetOverlay(base_opt, target_vol=tv, max_leverage=2.0)
        try:
            rets, _ = _run_backtest(returns_is, overlay, bc_is, min_history)
        except Exception as exc:
            logger.debug("IS grid arm=%s tv=%.2f failed: %s", arm_name, tv, exc)
            continue
        if rets.empty:
            continue

        sh = sharpe_ratio(rets)
        logger.info("IS grid arm=%s  tv=%.2f  Sharpe=%.3f", arm_name, tv, sh)

        # Record every combo (honest DSR)
        registry.record(
            config_hash,
            f"sprint-j-voltarget-{arm_name}",
            {
                "arm": arm_name,
                "target_vol": tv,
                "damp": base_params.damp_factor,
                "mom": base_params.momentum_strength,
                "equity_min": base_params.equity_min,
            },
            "IS 2005-2011",
            {"sharpe": round(sh, 4)},
        )

        if sh > best_sharpe:
            best_sharpe = sh
            best_tv = tv

    logger.info("IS arm=%s best_tv=%.2f  IS Sharpe=%.3f", arm_name, best_tv, best_sharpe)
    return best_tv


# ---------------------------------------------------------------------------
# FixedWeightsOptimizer — 60/40 benchmark
# (copied from backtest_sprint_i.py — same class, no import side-effects)
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
        "Arm A": "#2563eb",
        "Arm B": "#dc2626",
        "SPY B&H": "#9ca3af",
        "60/40 SPY/IEF": "#f59e0b",
        "Sprint I Balanced (unlevered)": "#6b7280",
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1.5]})
    fig.suptitle(
        f"JeanClaude Sprint J — Vol Targeting + Leva | OOS {oos_start[:4]}–{oos_end[:4]}\n"
        "Yahoo auto_adjust (total return) | max_leverage=2.0 | funding=DGS3MO+50bps | IS 2005-2011",
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
    frozen_tv: dict[str, float],
    all_metrics: list[dict],
    dsr_rows: list[dict],
    n_trials_j: int,
    n_trials_cumulative: int,
    verdicts: dict[str, str],
    funding_source: str,
) -> None:
    _OUT_DIR_REPORTS.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Sprint J — Vol Targeting + Leva: battere SPY B&H?\n",
        f"**Config hash:** `{config_hash}`  ",
        f"**Universo:** {len(cfg.universe)} ETF (identico a Sprint I — ex-ante dal design doc)  ",
        f"**IS:** {cfg.is_start} → {cfg.is_end}  |  **OOS:** {cfg.oos_start} → {cfg.oos_end}  ",
        f"**Fonte prezzi:** Yahoo auto_adjust=True (total return)  ",
        f"**Funding source:** {funding_source}  ",
        f"**Note:** {cfg.notes}  ",
        "",
        "## Parametri congelati\n",
        "**Base da Sprint I (NON ricalibrati):**  ",
        "- Arm A (Balanced): damp=0.25, mom=0.5",
        "- Arm B (Aggressive + equity_floor): damp=0.25, mom=0.7, equity_min=0.60",
        "",
        "**IS grid → target_vol congelato:**  ",
    ]
    for arm, tv in frozen_tv.items():
        lines.append(f"- {arm}: target_vol = {tv:.2f}")
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
        f"**n_trials questo esperimento (config_hash):** {n_trials_j}  ",
        f"**n_trials cumulativi (+ 21 Sprint I):** {n_trials_cumulative}  ",
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
        "### Criterio B: Sharpe netto ≥ SPY Sharpe E MaxDD < SPY MaxDD?\n",
        verdicts["sharpe_dd"],
        "",
        "---",
        "_Report generato automaticamente da scripts/backtest_sprint_j.py_",
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
    print("  JeanClaude Sprint J — Vol Targeting + Leva su base Sprint I")
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

    # ── 2. Prezzi Yahoo + funding FRED ───────────────────────────────────
    print("2/7  Download prezzi Yahoo + funding FRED DGS3MO...")
    FETCH_START = "2002-01-01"
    FETCH_END = cfg.oos_end

    prices = _load_yahoo_prices(cfg.universe, start=FETCH_START, end=FETCH_END)
    returns_all = prices.pct_change().iloc[1:]
    returns_all = returns_all.dropna(how="all")

    print(f"     Prezzi: {len(prices)} righe  ({prices.index[0].date()} → {prices.index[-1].date()})")

    returns_is = returns_all.loc[cfg.is_start:cfg.is_end]
    returns_oos = returns_all.loc[cfg.oos_start:cfg.oos_end]

    # Funding series: FRED DGS3MO with pub lag 1 bday, /100 + 0.005 spread
    funding, funding_source = _load_funding_series(FETCH_START, FETCH_END)
    print(f"     Funding: {funding_source}")

    # ── 3. IS calibration — griglia su target_vol × arm ─────────────────
    print(f"3/7  IS calibration ({cfg.is_start[:4]}–{cfg.is_end[:4]}) — tv grid {_TV_GRID}...")

    # Frozen base params from Sprint I (NOT recalibrated)
    params_arm_a = TrendMomentumParams(
        name="Arm A (Balanced)", damp_factor=0.25, momentum_strength=0.5,
    )
    params_arm_b = TrendMomentumParams(
        name="Arm B (Aggressive+floor)", damp_factor=0.25, momentum_strength=0.7,
        equity_min=0.60,
    )

    # IS funding slice
    funding_is = funding.loc[cfg.is_start:cfg.is_end]

    best_tv_a = _calibrate_is_arm(
        "ArmA", returns_is, params_arm_a, equity_rics,
        funding_is, _TV_GRID, registry, config_hash, MIN_HIST,
    )
    best_tv_b = _calibrate_is_arm(
        "ArmB", returns_is, params_arm_b, equity_rics,
        funding_is, _TV_GRID, registry, config_hash, MIN_HIST,
    )

    frozen_tv = {"Arm A": best_tv_a, "Arm B": best_tv_b}
    n_trials_j = registry.n_trials(config_hash=config_hash)
    n_trials_cumulative = n_trials_j + 21  # + 21 from Sprint I (documented)

    print(f"     Arm A: best target_vol={best_tv_a:.2f}")
    print(f"     Arm B: best target_vol={best_tv_b:.2f}")
    print(f"     Trials (questo esperimento): {n_trials_j}")
    print(f"     Trials cumulativi (+ 21 Sprint I): {n_trials_cumulative}")

    # ── 4. OOS — una volta per arm con tv congelato ───────────────────────
    print(f"4/7  OOS ({cfg.oos_start[:4]}–{cfg.oos_end[:4]}) — un run per arm...")
    bc_oos = BacktestConfig(
        rebalance_freq=cfg.rebalance_freq,
        transaction_cost_bps=cfg.tc_bps,
        execution_lag=cfg.execution_lag,
        min_history=MIN_HIST,
        max_leverage=2.0,
        funding_rate_annual=funding,
    )

    # Arm A: Balanced base + VolTargetOverlay
    opt_a = VolTargetOverlay(
        TrendMomentumOptimizer(params_arm_a, equity_rics=equity_rics),
        target_vol=best_tv_a,
        max_leverage=2.0,
    )
    rets_a, wh_a = _run_backtest(returns_oos, opt_a, bc_oos, MIN_HIST)

    # Arm B: Aggressive base + equity_min=0.60 + VolTargetOverlay
    opt_b = VolTargetOverlay(
        TrendMomentumOptimizer(params_arm_b, equity_rics=equity_rics),
        target_vol=best_tv_b,
        max_leverage=2.0,
    )
    rets_b, wh_b = _run_backtest(returns_oos, opt_b, bc_oos, MIN_HIST)

    logger.info("OOS Arm A done: Sharpe=%.3f", sharpe_ratio(rets_a))
    logger.info("OOS Arm B done: Sharpe=%.3f", sharpe_ratio(rets_b))

    # ── 5. Benchmarks ─────────────────────────────────────────────────────
    print("5/7  Benchmark (SPY B&H, 60/40, Sprint I Balanced unlevered)...")
    spy_ric, ief_ric = "SPY.P", "IEF.O"

    spy_rets_raw = returns_oos[spy_ric].dropna() if spy_ric in returns_oos.columns else pd.Series(dtype=float)
    common_idx = rets_a.index
    spy_rets_aligned = spy_rets_raw.reindex(common_idx).fillna(0.0) if not spy_rets_raw.empty else spy_rets_raw

    # 60/40 benchmark via engine (10bps, same as Sprint I)
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

    # Sprint I Balanced unlevered: damp=0.25, mom=0.5, no overlay
    # Uses same OOS window and config (no leverage/funding)
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
    sprint_i_bal_rets, _ = _run_backtest(returns_oos, opt_sprint_i_bal, bc_unlevered, MIN_HIST)

    # ── 6. Metrics + DSR + verdetti ───────────────────────────────────────
    print("6/7  Calcolo metriche, DSR, verdetti...")

    spy_m = _metrics_row("SPY B&H", spy_rets_aligned)
    bh6040_m = _metrics_row("60/40 SPY/IEF", bh6040_rets) if not bh6040_rets.empty else _metrics_row("60/40 SPY/IEF", spy_rets_aligned * 0)
    spi_bal_m = _metrics_row("Sprint I Balanced (unlevered)", sprint_i_bal_rets)
    arm_a_m = _metrics_row("Arm A (VolTarget Balanced)", rets_a)
    arm_b_m = _metrics_row("Arm B (VolTarget Aggressive+floor)", rets_b)

    all_metrics = [spy_m, bh6040_m, spi_bal_m, arm_a_m, arm_b_m]

    print(f"\n  {'Strategy':<38} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8}")
    print(f"  {'─' * 62}")
    for m in all_metrics:
        print(f"  {m['Strategy']:<38} {m['CAGR']:>7.1%} {m['Sharpe']:>7.2f} {-m['MaxDD']:>8.1%}")

    # DSR
    spy_sh = float(spy_m["Sharpe"])
    dsr_rows: list[dict] = []
    arm_results = [("Arm A (VolTarget Balanced)", rets_a), ("Arm B (VolTarget Aggressive+floor)", rets_b)]

    print(f"\n  DSR (n_trials_j={n_trials_j}, cumulativo={n_trials_cumulative}):")
    for name, rets in arm_results:
        sr_oos = float(sharpe_ratio(rets))
        obs = len(rets.dropna())
        sk = float(rets.dropna().skew())
        ku = float(rets.dropna().kurt()) + 3.0

        for nt, label in [(n_trials_j, "J"), (n_trials_cumulative, "J+SpI")]:
            nt = max(nt, 1)
            dsr = deflated_sharpe_ratio(sr_oos, n_trials=nt, obs=obs, skewness=sk, kurtosis=ku)
            flag = "✓ PASS" if dsr >= 0.75 else "  FAIL"
            tag = f"{name} (n={label})"
            print(f"  {tag:<50} DSR={dsr:.3f}  {flag}")
            dsr_rows.append({"strategy": tag, "n_trials": nt, "dsr": dsr})

    # ── 7. Leverage assertion ─────────────────────────────────────────────
    for arm_label, wh in [("Arm A", wh_a), ("Arm B", wh_b)]:
        if not wh.empty:
            max_lev = float(wh.sum(axis=1).max())
            assert max_lev <= 2.0 + 1e-6, (
                f"VIOLAZIONE LEVA: {arm_label} max={max_lev:.4f} > 2.0"
            )
            logger.info("%s max leverage observed: %.4f", arm_label, max_lev)
    print("     Leverage assertion OK (max ≤ 2.0+eps)")

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

    verdetto_shdd_lines = ["| Arm | Sharpe | SPY Sharpe | MaxDD | SPY MaxDD | Criterio B? |", "|---|---|---|---|---|---|"]
    for m in [arm_a_m, arm_b_m]:
        ok = m["Sharpe"] >= spy_sharpe and m["MaxDD"] < spy_maxdd
        verdetto_shdd_lines.append(
            f"| {m['Strategy']} | {m['Sharpe']:.2f} | {spy_sharpe:.2f} | "
            f"{-m['MaxDD']:.1%} | {-spy_maxdd:.1%} | {'SI' if ok else 'NO'} |"
        )
    verdetto_shdd = "\n".join(verdetto_shdd_lines)

    verdicts = {"cagr": verdetto_cagr, "sharpe_dd": verdetto_shdd}

    # ── Output ────────────────────────────────────────────────────────────
    print("\n7/7  Chart + report...")
    oos_results = [
        ("Arm A", rets_a),
        ("Arm B", rets_b),
    ]
    _plot(oos_results, spy_rets_aligned, bh6040_rets, sprint_i_bal_rets,
          cfg.oos_start, cfg.oos_end)

    _write_report(
        cfg=cfg,
        config_hash=config_hash,
        frozen_tv=frozen_tv,
        all_metrics=all_metrics,
        dsr_rows=dsr_rows,
        n_trials_j=n_trials_j,
        n_trials_cumulative=n_trials_cumulative,
        verdicts=verdicts,
        funding_source=funding_source,
    )

    print(f"  Chart  → {_OUT_CHART}")
    print(f"  Report → {_OUT_REPORT}")
    print(f"  Trials → {_REGISTRY_PATH}")
    print(f"  Tempo totale: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
