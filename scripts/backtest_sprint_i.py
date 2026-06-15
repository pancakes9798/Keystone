#!/usr/bin/env python3
"""Sprint I — validazione onesta su universo ex-ante 20 ETF (config congelata).

Differenze metodologiche vs Sprint E (audit 2026-06-10):
- universo dal design doc 2026-04-28, definibile nel 2005 senza conoscere i vincitori
- griglie ORIGINALI (niente estensioni post-risultato), config hash registrato
- execution_lag=1, costi 10bps, total return via Yahoo auto_adjust=True
  (Refinitiv CORAX è solo price return — verificato 2026-06-10 con
   scripts/validate_total_return.py)
- ogni combinazione provata finisce nel TrialRegistry → DSR con n_trials reale
- OOS 2012+ eseguito UNA volta con i parametri congelati dall'IS
- CPCV con refit (cpcv_refit_validation) sull'intero periodo IS+OOS
- benchmark: SPY total return B&H e 60/40 SPY/IEF ribilanciato mensile

Run:
    uv run python scripts/backtest_sprint_i.py [--source yahoo|refinitiv]
Output:
    docs/backtest/backtest_sprint_i.png
    docs/reports/sprint_i_results.md
"""
from __future__ import annotations

import argparse
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

from jeanclaude.backtest.bootstrap import stationary_bootstrap_ci
from jeanclaude.backtest.cpcv import CPCVConfig
from jeanclaude.backtest.cpcv_runner import cpcv_refit_validation
from jeanclaude.backtest.engine import BacktestConfig, BacktestEngine
from jeanclaude.backtest.metrics import (
    annualized_return, calmar_ratio, deflated_sharpe_ratio,
    max_drawdown, sharpe_ratio, sortino_ratio,
)
from jeanclaude.portfolio.optimizer import TrendMomentumOptimizer, TrendMomentumParams
from jeanclaude.research import ExperimentConfig, TrialRegistry

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sprint_i")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _REPO_ROOT / "configs" / "experiments" / "2026-06-10-sprint-i-20etf.json"
_OUT_DIR_BACKTEST = _REPO_ROOT / "docs" / "backtest"
_OUT_DIR_REPORTS = _REPO_ROOT / "docs" / "reports"
_OUT_CHART = _OUT_DIR_BACKTEST / "backtest_sprint_i.png"
_OUT_REPORT = _OUT_DIR_REPORTS / "sprint_i_results.md"
_REGISTRY_PATH = _REPO_ROOT / "data" / "research" / "trials.jsonl"

# ---------------------------------------------------------------------------
# RIC → Yahoo ticker mapping (strip suffix: SPY.P → SPY, QQQ.O → QQQ)
# ---------------------------------------------------------------------------
# Explicit overrides for non-trivial mappings; all others strip after "."
_RIC_TO_YAHOO: dict[str, str] = {}


def _ric_to_yahoo(ric: str) -> str:
    """Convert Refinitiv RIC to Yahoo Finance ticker by stripping exchange suffix."""
    if ric in _RIC_TO_YAHOO:
        return _RIC_TO_YAHOO[ric]
    return ric.split(".")[0]


# ---------------------------------------------------------------------------
# IS calibration (compact re-implementation of the Sprint E methodology)
# The original implementation lived in scripts/backtest_sprint_e.py (removed
# 2026-06-10, audit P2 dead-code cleanup; git history has the original).
# This file reproduces the stability-constrained logic compactly without the
# module-level side effects that made backtest_sprint_e.py non-importable.
# ---------------------------------------------------------------------------

# 4 rolling ~18-month sub-windows covering IS 2005-2011
_IS_SUBWINDOWS: list[tuple[str, str]] = [
    ("2005-01-01", "2006-06-30"),
    ("2006-07-01", "2007-12-31"),
    ("2008-01-01", "2009-06-30"),
    ("2009-07-01", "2011-12-31"),
]


def _run_backtest(
    returns: pd.DataFrame,
    params: TrendMomentumParams,
    bc: BacktestConfig,
    equity_rics: frozenset[str],
    min_history: int,
) -> pd.Series:
    """Run engine and return portfolio_returns."""
    opt = TrendMomentumOptimizer(params, equity_rics=equity_rics)
    eng = BacktestEngine(opt, config=bc)
    result = eng.run(returns, min_history=min_history)
    return result.portfolio_returns


def calibrate_is(
    returns_is: pd.DataFrame,
    strategy_name: str,
    mom_grid: list[float],
    damp_grid: list[float],
    equity_rics: frozenset[str],
    registry: TrialRegistry,
    config_hash: str,
    min_history: int,
) -> tuple[float, float]:
    """Stability-constrained IS grid search. Returns (damp_factor, momentum_strength).

    Every evaluated (damp, mom) combo is recorded in registry for honest DSR.
    Selection: highest IS Sharpe with Sharpe > 0 in at least 3/4 sub-windows.
    Falls back to best-overall (relaxation) if best stable combo has Sharpe < 0,
    then to hardcoded (0.5, 0.5)/(0.5, 1.0) as last resort if nothing passes.
    Relaxation branch faithfully re-implements the Sprint E methodology
    (original in git history as scripts/backtest_sprint_e.py, removed 2026-06-10).

    execution_lag=1: la calibrazione IS usa lo stesso regime di esecuzione del
    backtest OOS (default BacktestConfig lag=1).
    """
    _fallback_mom = {"Balanced": 0.5, "Aggressive": 1.0}
    best_sharpe = -np.inf
    best_combo: tuple[float, float] = (0.5, _fallback_mom.get(strategy_name, 0.0))

    # IS config usa execution_lag=1 — stesso regime di esecuzione del backtest OOS.
    # Le comparazioni IS-vs-IS rimangono apples-to-apples perché tutti i combo
    # usano lo stesso lag.
    bc_is = BacktestConfig(
        rebalance_freq="ME",
        transaction_cost_bps=10.0,
        execution_lag=1,
    )

    for damp in damp_grid:
        mom_candidates = [0.0] if strategy_name == "Conservative" else mom_grid
        for mom in mom_candidates:
            params = TrendMomentumParams(
                name=strategy_name, damp_factor=damp, momentum_strength=mom
            )
            try:
                rets = _run_backtest(returns_is, params, bc_is, equity_rics, min_history)
            except Exception as exc:
                logger.debug("IS grid (%s, d=%.2f, m=%.2f) failed: %s", strategy_name, damp, mom, exc)
                continue
            if rets.empty:
                continue

            full_sh = sharpe_ratio(rets)

            # Stability: Sharpe > 0 in ≥ 3/4 sub-windows
            stable = sum(
                1
                for sw_start, sw_end in _IS_SUBWINDOWS
                if len(rets.loc[sw_start:sw_end]) > 20 and sharpe_ratio(rets.loc[sw_start:sw_end]) > 0
            )

            # Record EVERY evaluated combo in the registry (honest DSR)
            registry.record(
                config_hash,
                f"sprint-i-20etf-{strategy_name}",
                {"damp": damp, "mom": mom, "strategy": strategy_name},
                "IS 2005-2011",
                {"sharpe": round(full_sh, 4), "stable_windows": stable},
            )

            logger.info(
                "IS grid %s  damp=%.2f  mom=%.2f  Sharpe=%.3f  stable=%d/4",
                strategy_name, damp, mom, full_sh, stable,
            )

            if stable >= 3 and full_sh > best_sharpe:
                best_sharpe = full_sh
                best_combo = (damp, mom)

    if best_sharpe == -np.inf:
        # Nessun combo ha passato il filtro di stabilità — fallback hardcoded (ultima spiaggia)
        logger.warning(
            "%s: no combo passed stability filter — using fallback (damp=%.2f, mom=%.2f)",
            strategy_name, best_combo[0], best_combo[1],
        )
    elif best_sharpe < 0:
        # Il miglior combo STABILE ha Sharpe < 0 — rilassa il filtro di stabilità
        # e scegli il miglior combo overall (relaxation branch Sprint E)
        logger.warning(
            "%s: best stable combo has Sharpe=%.3f < 0 — relaxing stability filter",
            strategy_name, best_sharpe,
        )
        best_sharpe_relaxed = -np.inf
        best_combo_relaxed = best_combo
        for damp in damp_grid:
            mom_candidates = [0.0] if strategy_name == "Conservative" else mom_grid
            for mom in mom_candidates:
                params = TrendMomentumParams(
                    name=strategy_name, damp_factor=damp, momentum_strength=mom
                )
                try:
                    rets = _run_backtest(returns_is, params, bc_is, equity_rics, min_history)
                except Exception:
                    continue
                if rets.empty:
                    continue
                sh = sharpe_ratio(rets)
                if sh > best_sharpe_relaxed:
                    best_sharpe_relaxed = sh
                    best_combo_relaxed = (damp, mom)
        best_combo = best_combo_relaxed
        best_sharpe = best_sharpe_relaxed
        logger.info(
            "%s: relaxed → selected damp=%.2f, mom=%.2f  (IS Sharpe=%.3f)",
            strategy_name, best_combo[0], best_combo[1], best_sharpe,
        )
    else:
        logger.info("%s: selected damp=%.2f, mom=%.2f (IS Sharpe=%.3f)", strategy_name, *best_combo, best_sharpe)

    return best_combo


# ---------------------------------------------------------------------------
# FixedWeightsOptimizer — 60/40 benchmark
# ---------------------------------------------------------------------------
class FixedWeightsOptimizer:
    """Optimizer that always returns a fixed weight vector.

    Used for the 60% SPY / 40% IEF benchmark.
    """

    def __init__(self, weights: dict[str, float]) -> None:
        self._w = pd.Series(weights, dtype=float)
        self._w /= self._w.sum()  # normalize

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


def _load_yahoo_prices(
    rics: tuple[str, ...],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download Yahoo Finance adjusted close prices for universe RICs.

    Maps RICs to Yahoo tickers, downloads, and returns a DataFrame indexed
    by date with RIC columns. Assets enter when they have data (point-in-time
    inclusion — columns with no data are kept as NaN, not dropped).
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

    # Handle single-ticker vs multi-ticker response shape
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()

    close.index = pd.to_datetime(close.index).tz_localize(None)

    # Rename Yahoo tickers back to RICs
    yahoo_to_ric = {_ric_to_yahoo(r): r for r in rics}
    close = close.rename(columns=yahoo_to_ric)

    # Ensure all RICs present as columns (NaN if not available)
    for r in rics:
        if r not in close.columns:
            close[r] = np.nan

    return close[list(rics)].ffill()


# ---------------------------------------------------------------------------
# CPCV helpers
# ---------------------------------------------------------------------------

def _make_cpcv_calibrate_fn(
    damp_grid: list[float],
    mom_grid_balanced: list[float],
    equity_rics: frozenset[str],
    min_history: int,
) -> object:
    """Return a calibrate_fn that picks params on a compact train fold.

    Approximation documented in the module docstring: train data is the
    concatenation of CPCV train segments (purged + embargoed). We calibrate
    over the aggregate by running the IS engine on each contiguous segment
    of ≥ 252 obs and averaging the Sharpe across segments.
    """

    def calibrate_fn(train_returns: pd.DataFrame) -> dict:
        """Select best (damp, mom) for Balanced strategy on train fold."""
        # Use "Balanced" as representative for CPCV (grid covers the meaningful range).
        # execution_lag=1: stesso regime di esecuzione della evaluate_fn — coerente
        # con la calibrazione IS principale (entrambe usano lag=1).
        bc = BacktestConfig(
            rebalance_freq="ME",
            transaction_cost_bps=10.0,
            execution_lag=1,
        )
        best_sh = -np.inf
        best_combo = {"damp": 0.5, "mom": 0.5}

        # Find contiguous segments ≥ 252 obs
        segments: list[pd.DataFrame] = []
        if len(train_returns) >= min_history:
            # CPCV delivers potentially non-contiguous rows; detect gaps > 5 bdays
            idx = train_returns.index
            gaps = np.where(np.diff(idx.asi8) > pd.Timedelta(days=7).value)[0] + 1
            segs = np.split(np.arange(len(idx)), gaps)
            for seg in segs:
                if len(seg) >= min_history:
                    segments.append(train_returns.iloc[seg])

        if not segments:
            return best_combo

        for damp in damp_grid:
            for mom in mom_grid_balanced:
                params = TrendMomentumParams("Balanced", damp_factor=damp, momentum_strength=mom)
                sharpes: list[float] = []
                for seg_df in segments:
                    try:
                        rets = _run_backtest(seg_df, params, bc, equity_rics, min_history)
                        if not rets.empty:
                            sharpes.append(sharpe_ratio(rets))
                    except Exception:
                        pass
                if sharpes:
                    mean_sh = float(np.mean(sharpes))
                    if mean_sh > best_sh:
                        best_sh = mean_sh
                        best_combo = {"damp": damp, "mom": mom}

        return best_combo

    return calibrate_fn


def _make_cpcv_evaluate_fn(
    equity_rics: frozenset[str],
    min_history: int,
) -> object:
    """Return an evaluate_fn that runs the engine on the test fold.

    Concatenates contiguous test segments and returns their aggregate Sharpe.
    Approximation: segments from different test groups are concatenated as-if
    contiguous; the engine treats them as a single stream. This is standard
    practice in CPCV implementations (LdP AFML ch.12).
    """

    def evaluate_fn(test_returns: pd.DataFrame, params: dict) -> float:
        p = TrendMomentumParams(
            name="Balanced",
            damp_factor=float(params.get("damp", 0.5)),
            momentum_strength=float(params.get("mom", 0.5)),
        )
        bc = BacktestConfig(
            rebalance_freq="ME",
            transaction_cost_bps=10.0,
            execution_lag=1,
        )
        try:
            rets = _run_backtest(test_returns, p, bc, equity_rics, min_history)
            if rets.empty:
                return 0.0
            return float(sharpe_ratio(rets))
        except Exception:
            return 0.0

    return evaluate_fn


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(
    oos_results: list[tuple[str, pd.Series]],
    spy_rets: pd.Series,
    bh6040_rets: pd.Series,
    oos_start: str,
    oos_end: str,
    initial: float = 10_000.0,
) -> None:
    palette = {
        "Conservative": "#6b7280",
        "Balanced": "#2563eb",
        "Aggressive": "#dc2626",
        "SPY B&H": "#9ca3af",
        "60/40 SPY/IEF": "#f59e0b",
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1.5]})
    fig.suptitle(
        f"JeanClaude Sprint I — 20 ETF ex-ante, config congelata | OOS {oos_start[:4]}–{oos_end[:4]}\n"
        "Yahoo auto_adjust (total return) | execution_lag=1, TC=10bps | IS 2005-2011",
        fontsize=12,
    )

    ax = axes[0]
    for label, rets in [("SPY B&H", spy_rets), ("60/40 SPY/IEF", bh6040_rets), *oos_results]:
        cum = (1 + rets).cumprod() * initial
        final = float(cum.iloc[-1])
        ax.plot(
            cum.index, cum,
            color=palette.get(label, "#000"),
            lw=1.8 if label not in ("SPY B&H", "60/40 SPY/IEF") else 1.2,
            linestyle="--" if label in ("SPY B&H", "60/40 SPY/IEF") else "-",
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
    for label, rets in [("SPY B&H", spy_rets), ("60/40 SPY/IEF", bh6040_rets), *oos_results]:
        dd = _drawdown_series(rets) * 100
        if label in ("SPY B&H", "60/40 SPY/IEF"):
            ax.plot(dd.index, dd, color=palette.get(label, "#888"), lw=0.9,
                    linestyle="--", label=f"{label}", alpha=0.7)
        else:
            ax.fill_between(dd.index, dd, 0, alpha=0.20, color=palette.get(label, "#000"))
            ax.plot(dd.index, dd, color=palette.get(label, "#000"), lw=1.0, label=f"{label}")
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
    frozen_params: dict[str, dict],
    oos_metrics: list[dict],
    spy_metrics: dict,
    bh6040_metrics: dict,
    dsr_rows: list[dict],
    cpcv_df: pd.DataFrame,
    n_trials_real: int,
    verdict: str,
) -> None:
    _OUT_DIR_REPORTS.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Sprint I — Risultati Validazione Onesta 20 ETF\n",
        f"**Config hash:** `{config_hash}`  ",
        f"**Universo:** {len(cfg.universe)} ETF ex-ante (design doc 2026-04-28)  ",
        f"**IS:** {cfg.is_start} → {cfg.is_end}  |  **OOS:** {cfg.oos_start} → {cfg.oos_end}  ",
        f"**Fonte prezzi:** Yahoo auto_adjust=True (total return)  ",
        f"**Note:** {cfg.notes}  ",
        "",
        "## Parametri IS congelati\n",
    ]

    for strat, p in frozen_params.items():
        lines.append(f"- **{strat}**: damp={p['damp']:.2f}, mom={p['mom']:.2f}")
    lines.append("")

    lines += [
        "## Metriche OOS\n",
        "| Strategy | CAGR | Sharpe | Sortino | MaxDD | Calmar | $10k→ |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in [spy_metrics, bh6040_metrics, *oos_metrics]:
        lines.append(
            f"| {m['Strategy']} | {m['CAGR']:.1%} | {m['Sharpe']:.2f} | "
            f"{m['Sortino']:.2f} | {-m['MaxDD']:.1%} | {m['Calmar']:.2f} | "
            f"${m['$10k_final']:,.0f} |"
        )
    lines.append("")

    lines += [
        "## Deflated Sharpe Ratio (sensibilità)\n",
        f"**n_trials reale (registry):** {n_trials_real}  ",
        "",
        "| Strategy | n_trials | DSR | Pass (≥0.75)? |",
        "|---|---|---|---|",
    ]
    for row in dsr_rows:
        flag = "✓" if row["dsr"] >= 0.75 else "✗"
        lines.append(f"| {row['strategy']} | {row['n_trials']} | {row['dsr']:.3f} | {flag} |")
    lines.append("")

    lines += [
        "## CPCV con refit (proxy: griglia Balanced) — CPCVConfig n_splits=6, n_test_splits=2, 15 fold\n",
        "> Approximazione documentata: train=concatenazione segmenti contigui purged+embargoed; "
        "calibrate_fn = grid-search Balanced su segmenti ≥252 obs, Sharpe medio; "
        "evaluate_fn = engine run sul test fold, Sharpe aggregato. "
        "Overlap di covarianza tra fold accettato (LdP: purging riguarda le label, non le feature).",
        "",
        "| Fold | n_train | n_test | best_damp | best_mom | test_sharpe |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in cpcv_df.iterrows():
        p = row.get("params", {}) or {}
        lines.append(
            f"| {int(row['fold'])} | {int(row['n_train'])} | {int(row['n_test'])} | "
            f"{p.get('damp', '?')} | {p.get('mom', '?')} | {row['test_sharpe']:.3f} |"
        )
    frac_pos = float((cpcv_df["test_sharpe"] > 0).mean())
    lines += [
        "",
        f"**Frazione fold con Sharpe > 0:** {frac_pos:.0%}",
        "",
    ]

    lines += [
        "## VERDETTO\n",
        f"**{verdict}**",
        "",
        "> Criteri: EDGE CONFERMATO se Sharpe OOS di almeno una strategia > Sharpe SPY "
        "E DSR(n_trials reale) ≥ 0.75 per quella strategia.",
        "",
        "_Report generato automaticamente da scripts/backtest_sprint_i.py_",
    ]

    _OUT_REPORT.write_text("\n".join(lines))
    logger.info("Report: %s", _OUT_REPORT)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Fix all seeds for determinism
    np.random.seed(42)
    import random
    random.seed(42)

    parser = argparse.ArgumentParser(description="Sprint I — validazione onesta 20 ETF")
    parser.add_argument(
        "--source", choices=["yahoo", "refinitiv"], default="yahoo",
        help="Fonte prezzi (default: yahoo — total return via auto_adjust=True)"
    )
    args = parser.parse_args()

    if args.source == "refinitiv":
        print(
            "\n⚠  ATTENZIONE: --source refinitiv usa TRDPRC_1 con CORAX adjustments.\n"
            "   Refinitiv CORAX NON include i dividendi ordinari (solo split/eventi\n"
            "   straordinari) — è PRICE RETURN, non total return.\n"
            "   Verificato empiricamente 2026-06-10 con scripts/validate_total_return.py.\n"
            "   I risultati non saranno comparabili con Yahoo auto_adjust.\n"
            "   Usa --source yahoo per la validazione metodologicamente onesta.\n",
            flush=True,
        )

    t0 = time.time()
    print(f"\n{'═' * 70}")
    print("  JeanClaude Sprint I — 20 ETF ex-ante, config congelata")
    print(f"  Fonte: {args.source.upper()}")
    print(f"{'═' * 70}\n")

    # ── 1. Config ─────────────────────────────────────────────────────────
    print("1/8  Caricamento config congelata...")
    cfg = ExperimentConfig.from_json(_CONFIG_PATH)
    config_hash = cfg.config_hash()
    print(f"     Config hash: {config_hash}")
    print(f"     Universo: {len(cfg.universe)} ETF  |  IS: {cfg.is_start}→{cfg.is_end}  |  OOS: {cfg.oos_start}→{cfg.oos_end}")

    registry = TrialRegistry(_REGISTRY_PATH)
    equity_rics = frozenset(cfg.equity_rics)
    MIN_HIST = 252

    # ── 2. Data ───────────────────────────────────────────────────────────
    print("2/8  Download prezzi Yahoo Finance...")
    # Warmup: fetch from 2002-01-01 to provide 3yr HRP window (756d) from IS start 2005-01-01
    FETCH_START = "2002-01-01"
    FETCH_END = cfg.oos_end

    prices = _load_yahoo_prices(cfg.universe, start=FETCH_START, end=FETCH_END)
    # Simple returns (pct_change) — Yahoo auto_adjust already includes dividends
    returns_all = prices.pct_change().iloc[1:]

    # Align index: remove days with all-NaN (non-trading days)
    returns_all = returns_all.dropna(how="all")

    print(f"     Prezzi: {len(prices)} righe  ({prices.index[0].date()} → {prices.index[-1].date()})")
    print(f"     Assets con dati: {prices.notna().any().sum()}/{len(cfg.universe)}")

    # IS and OOS slices
    returns_is = returns_all.loc[cfg.is_start:cfg.is_end]
    returns_oos = returns_all.loc[cfg.oos_start:cfg.oos_end]

    # ── 3. IS Calibration ─────────────────────────────────────────────────
    print(f"3/8  IS calibration ({cfg.is_start[:4]}–{cfg.is_end[:4]})...")

    damp_grid = list(cfg.damp_grid)
    mom_grid_balanced = list(cfg.mom_grid_balanced)
    mom_grid_aggressive = list(cfg.mom_grid_aggressive)

    damp_cons, _ = calibrate_is(
        returns_is, "Conservative", [], damp_grid,
        equity_rics, registry, config_hash, MIN_HIST,
    )
    damp_bal, mom_bal = calibrate_is(
        returns_is, "Balanced", mom_grid_balanced, damp_grid,
        equity_rics, registry, config_hash, MIN_HIST,
    )
    damp_agg, mom_agg = calibrate_is(
        returns_is, "Aggressive", mom_grid_aggressive, damp_grid,
        equity_rics, registry, config_hash, MIN_HIST,
    )

    frozen_params = {
        "Conservative": {"damp": damp_cons, "mom": 0.0},
        "Balanced": {"damp": damp_bal, "mom": mom_bal},
        "Aggressive": {"damp": damp_agg, "mom": mom_agg},
    }
    print(f"     Conservative:  damp={damp_cons:.2f}")
    print(f"     Balanced:      damp={damp_bal:.2f}, mom={mom_bal:.2f}")
    print(f"     Aggressive:    damp={damp_agg:.2f}, mom={mom_agg:.2f}")

    n_trials_real = registry.n_trials(config_hash=cfg.config_hash())
    print(f"     Trials nel registry: {n_trials_real}")

    # ── 4. OOS Backtest (ONE run per strategy) ───────────────────────────
    print(f"4/8  OOS backtest ({cfg.oos_start[:4]}–{cfg.oos_end[:4]})...")
    bc_oos = BacktestConfig(
        rebalance_freq=cfg.rebalance_freq,
        transaction_cost_bps=cfg.tc_bps,
        execution_lag=cfg.execution_lag,
        min_history=MIN_HIST,
    )

    strategy_configs = [
        TrendMomentumParams("Conservative", damp_factor=damp_cons, momentum_strength=0.0),
        TrendMomentumParams("Balanced", damp_factor=damp_bal, momentum_strength=mom_bal),
        TrendMomentumParams("Aggressive", damp_factor=damp_agg, momentum_strength=mom_agg),
    ]

    oos_results: list[tuple[str, pd.Series]] = []
    for p in strategy_configs:
        rets = _run_backtest(returns_oos, p, bc_oos, equity_rics, MIN_HIST)
        oos_results.append((p.name, rets))
        logger.info("OOS done: %s  Sharpe=%.3f", p.name, sharpe_ratio(rets))

    # ── 5. Benchmarks ─────────────────────────────────────────────────────
    print("5/8  Benchmark SPY B&H e 60/40 SPY/IEF...")
    spy_ric, ief_ric = "SPY.P", "IEF.O"
    spy_rets_oos = returns_oos[spy_ric].dropna() if spy_ric in returns_oos.columns else pd.Series(dtype=float)
    ief_rets_oos = returns_oos[ief_ric].dropna() if ief_ric in returns_oos.columns else pd.Series(dtype=float)

    # 60/40: monthly rebalance via engine
    bh6040_rets: pd.Series
    if spy_ric in returns_oos.columns and ief_ric in returns_oos.columns:
        opt6040 = FixedWeightsOptimizer({spy_ric: 0.6, ief_ric: 0.4})
        # 10bps per il 60/40: stessi costi delle strategie (fair comparison).
        # SPY B&H non paga costi perché è buy-and-hold vero (nessun ribilanciamento).
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
    else:
        bh6040_rets = pd.Series(dtype=float)

    # Align SPY B&H to OOS index
    if not spy_rets_oos.empty:
        # Reindex to the common OOS index
        common_idx = oos_results[0][1].index if oos_results else spy_rets_oos.index
        spy_rets_aligned = spy_rets_oos.reindex(common_idx).fillna(0.0)
    else:
        spy_rets_aligned = spy_rets_oos

    # ── 6. Metrics + DSR ──────────────────────────────────────────────────
    print("6/8  Calcolo metriche e DSR...")
    spy_metrics = _metrics_row("SPY B&H", spy_rets_aligned)
    bh6040_metrics = _metrics_row("60/40 SPY/IEF", bh6040_rets) if not bh6040_rets.empty else _metrics_row("60/40 SPY/IEF", spy_rets_aligned * 0)
    oos_metrics = [_metrics_row(name, rets) for name, rets in oos_results]

    print(f"\n  {'Strategy':<22} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>7} {'$10k':>10}")
    print(f"  {'─' * 70}")
    for m in [spy_metrics, bh6040_metrics, *oos_metrics]:
        print(
            f"  {m['Strategy']:<22} {m['CAGR']:>7.1%} {m['Sharpe']:>7.2f} "
            f"{m['Sortino']:>8.2f} {-m['MaxDD']:>8.1%} {m['Calmar']:>7.2f} "
            f"${m['$10k_final']:>9,.0f}"
        )

    # Bootstrap CI
    print(f"\n  Bootstrap CI 95% (seed=42):")
    for name, rets in [("SPY B&H", spy_rets_aligned), *oos_results]:
        clean = rets.dropna()
        if len(clean) > 10:
            ci = stationary_bootstrap_ci(clean, n_samples=1_000, seed=42)
            print(
                f"  {name:<22}  Sharpe {ci.point['sharpe']:>5.2f} "
                f"[{ci.lower['sharpe']:>5.2f}, {ci.upper['sharpe']:>5.2f}]  "
                f"CAGR {ci.point['cagr']:>5.1%} [{ci.lower['cagr']:>5.1%}, {ci.upper['cagr']:>5.1%}]"
            )

    # DSR with n_trials sensitivity
    dsr_rows: list[dict] = []
    print(f"\n  Deflated Sharpe Ratio (n_trials reale={n_trials_real}):")
    print(f"  {'Strategy':<22} {'n_trials':>10} {'DSR':>8}  Pass≥0.75?")
    print(f"  {'─' * 55}")
    spy_sh = float(spy_metrics["Sharpe"])
    for name, rets in oos_results:
        sr_oos = float(sharpe_ratio(rets))
        obs = len(rets.dropna())
        sk = float(rets.dropna().skew())
        ku = float(rets.dropna().kurt()) + 3.0  # excess → total kurtosis
        for n_mult, label in [(1, "real"), (2, "×2"), (4, "×4")]:
            nt = max(n_trials_real * n_mult, 1)
            dsr = deflated_sharpe_ratio(sr_oos, n_trials=nt, obs=obs, skewness=sk, kurtosis=ku)
            flag = "✓ PASS" if dsr >= 0.75 else "  FAIL"
            tag = f"{name} ({label})"
            print(f"  {tag:<22} {nt:>10}   {dsr:>6.3f}  {flag}")
            dsr_rows.append({
                "strategy": tag, "n_trials": nt, "dsr": dsr,
                "sr_oos": sr_oos, "obs": obs,
            })

    # ── 7. CPCV con refit ─────────────────────────────────────────────────
    print("\n7/8  CPCV con refit (proxy: griglia Balanced) — CPCVConfig n_splits=6, n_test_splits=2 → 15 fold...")
    print("     (approssimazione: train=segmenti contigui concatenati, calibrate=Balanced)")
    returns_full = returns_all.loc[cfg.is_start:cfg.oos_end]

    calibrate_fn = _make_cpcv_calibrate_fn(
        damp_grid=damp_grid,
        mom_grid_balanced=mom_grid_balanced,
        equity_rics=equity_rics,
        min_history=MIN_HIST,
    )
    evaluate_fn = _make_cpcv_evaluate_fn(
        equity_rics=equity_rics,
        min_history=MIN_HIST,
    )

    cpcv_cfg = CPCVConfig(n_splits=6, n_test_splits=2)  # C(6,2) = 15 fold
    cpcv_df = cpcv_refit_validation(
        returns_full,
        calibrate_fn=calibrate_fn,
        evaluate_fn=evaluate_fn,
        config=cpcv_cfg,
        eval_horizon_days=21,
    )
    frac_pos = float((cpcv_df["test_sharpe"] > 0).mean())
    print(f"     CPCV fold con Sharpe>0: {frac_pos:.0%}  (mean={cpcv_df['test_sharpe'].mean():.3f})")

    # ── 8. Verdetto + Output ──────────────────────────────────────────────
    print("\n8/8  Verdetto e output...")

    # EDGE CONFERMATO se almeno una strategia ha Sharpe OOS > SPY
    # E DSR(n_trials reale) ≥ 0.75 per quella stessa strategia
    edge_confirmed = False
    for name, rets in oos_results:
        sr_strat = float(sharpe_ratio(rets))
        obs = len(rets.dropna())
        sk = float(rets.dropna().skew())
        ku = float(rets.dropna().kurt()) + 3.0
        dsr_real = deflated_sharpe_ratio(
            sr_strat, n_trials=max(n_trials_real, 1),
            obs=obs, skewness=sk, kurtosis=ku,
        )
        if sr_strat > spy_sh and dsr_real >= 0.75:
            edge_confirmed = True
            print(f"     → {name}: Sharpe={sr_strat:.2f} > SPY={spy_sh:.2f}, DSR={dsr_real:.3f} ≥ 0.75 ✓")

    verdict = (
        "EDGE CONFERMATO — almeno una strategia batte SPY in Sharpe OOS con DSR(n_trials reale) ≥ 0.75"
        if edge_confirmed else
        "EDGE NON CONFERMATO — nessuna strategia soddisfa entrambe le condizioni: "
        "Sharpe OOS > SPY e DSR(n_trials reale) ≥ 0.75"
    )
    print(f"\n  VERDETTO: {verdict}\n")

    # Chart
    _plot(oos_results, spy_rets_aligned, bh6040_rets,
          cfg.oos_start, cfg.oos_end)

    # Report
    _write_report(
        cfg=cfg,
        config_hash=config_hash,
        frozen_params=frozen_params,
        oos_metrics=oos_metrics,
        spy_metrics=spy_metrics,
        bh6040_metrics=bh6040_metrics,
        dsr_rows=dsr_rows,
        cpcv_df=cpcv_df,
        n_trials_real=n_trials_real,
        verdict=verdict,
    )

    print(f"  Chart     → {_OUT_CHART}")
    print(f"  Report    → {_OUT_REPORT}")
    print(f"  Trials    → {_REGISTRY_PATH} ({n_trials_real} distinti)")
    print(f"  Tempo totale: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
