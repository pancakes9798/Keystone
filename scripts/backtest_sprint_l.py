#!/usr/bin/env python3
"""Sprint L — Base Strutturale Equity + De-risk Trend-Only.

Risponde alla domanda: un floor strutturalmente più equity (equity_min ∈ {0.5,0.6,0.7})
abbinato al de-risk MA200 (damp ∈ {0.25, 0.5}) — il meccanismo vincente di Sprint I —
domina il 60/40?

Criteri di successo DICHIARATI EX-ANTE (valutati dopo l'unico run OOS):
  (a) Sharpe OOS ≥ Sharpe 60/40 (10bps)
  (b) MaxDD OOS non più profondo del MaxDD 60/40
  (c) CAGR OOS ≥ 8%
Tutti e tre devono essere soddisfatti.

Semantica equity_min letta da jeanclaude/portfolio/optimizer/trend_momentum.py
(pipeline: HRP → equity_floor → MA200 damp → momentum tilt → cap/renorm):

  Il floor equity_min è applicato al PASSO 2 (_apply_equity_floor), prima del
  damp MA200 (passo 3) e prima del momentum tilt (passo 4). Il damp moltiplicativo
  moltiplica ogni asset sotto la sua MA200 per damp_factor, poi RENORMALIZZA a
  somma 1. Se tutti o molti degli equity_rics sono in downtrend, il renorm del
  damp può ridurre la quota equity effettiva sotto equity_min: il floor non viene
  ri-applicato dopo il passo 3 né dopo il cap/renorm del passo 5. Questo è il
  design voluto (de-risk: in bear market equity scende sotto il floor strutturale).
  Il passo 5 (cap per-asset + renorm finale) può ulteriormente spostare i pesi
  relativi. Quindi equity_min è un floor "post-HRP, pre-damp": è la base
  strutturale di partenza; il damp e il tilt possono violarlo sistematicamente
  nei periodi di risk-off.

Flusso (7 punti):
1. Config + hash + registry (data/research/trials.jsonl)
2. Prezzi Yahoo 20 ETF (2002-2026)
3. IS grid: 6 combo equity_min ∈ {0.5,0.6,0.7} × damp ∈ {0.25,0.5}, mom=0.5 CONGELATO
   → engine IS (lag=1, 10bps) → registry → selezione max IS Sharpe (log MaxDD IS per combo)
4. OOS UNA volta col combo congelato
5. Benchmark: SPY B&H, 60/40 10bps, Sprint I Balanced unlevered
6. DSR: n_trials questo hash + cumulativo (+33 da I+J+K → 39 prima di IS)
   Verdetto TRE criteri ex-ante (a/b/c), ciascuno stampato pass/fail; nota se bordo griglia
7. Report docs/reports/sprint_l_results.md (equity share media OOS + turnover annuo) + chart

Run:
    uv run python scripts/backtest_sprint_l.py
Output:
    docs/backtest/backtest_sprint_l.png
    docs/reports/sprint_l_results.md
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
from jeanclaude.portfolio.optimizer import TrendMomentumOptimizer, TrendMomentumParams
from jeanclaude.research import ExperimentConfig, TrialRegistry

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sprint_l")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _REPO_ROOT / "configs" / "experiments" / "2026-06-11-sprint-l-equity-structural.json"
_OUT_DIR_BACKTEST = _REPO_ROOT / "docs" / "backtest"
_OUT_DIR_REPORTS = _REPO_ROOT / "docs" / "reports"
_OUT_CHART = _OUT_DIR_BACKTEST / "backtest_sprint_l.png"
_OUT_REPORT = _OUT_DIR_REPORTS / "sprint_l_results.md"
_REGISTRY_PATH = _REPO_ROOT / "data" / "research" / "trials.jsonl"

# IS equity_min grid (mom frozen at 0.5 from Sprint I Balanced)
_EQUITY_MIN_GRID: tuple[float, ...] = (0.5, 0.6, 0.7)
# Cumulative trials from Sprint I + J + K before this experiment
_PRIOR_TRIALS = 33

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
# Data loading — Yahoo prices (copied from backtest_sprint_j.py verbatim —
# no runtime import side-effects from sprint_i or sprint_j)
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
    # average one-way turnover per rebalance × 12 monthly rebalances
    avg_per_rebalance = float(diffs.mean())
    n_rebalances = len(diffs)
    years = n_rebalances / 12.0
    return avg_per_rebalance * n_rebalances / max(years, 1e-9)


def _effective_equity_share(
    weights_history: pd.DataFrame,
    equity_rics: frozenset[str],
) -> float:
    """Average effective equity weight over the OOS weights_history."""
    if weights_history.empty:
        return float("nan")
    eq_cols = [c for c in weights_history.columns if c in equity_rics]
    if not eq_cols:
        return 0.0
    return float(weights_history[eq_cols].sum(axis=1).mean())


# ---------------------------------------------------------------------------
# IS calibration — grid over equity_min × damp, mom frozen at 0.5
# ---------------------------------------------------------------------------

def _calibrate_is(
    returns_is: pd.DataFrame,
    equity_rics: frozenset[str],
    damp_grid: tuple[float, ...],
    equity_min_grid: tuple[float, ...],
    registry: TrialRegistry,
    config_hash: str,
    min_history: int,
) -> tuple[float, float]:
    """IS grid search over equity_min × damp (mom=0.5 frozen).

    Returns (best_equity_min, best_damp) with highest IS Sharpe.
    Every combo is registered for honest DSR.
    Logs MaxDD IS per combo (for risk-awareness, selection is on Sharpe only).
    """
    bc_is = BacktestConfig(
        rebalance_freq="ME",
        transaction_cost_bps=10.0,
        execution_lag=1,
    )

    best_sharpe = -np.inf
    best_equity_min = equity_min_grid[0]
    best_damp = damp_grid[0]

    print(f"\n     IS grid ({len(equity_min_grid)} equity_min × {len(damp_grid)} damp = "
          f"{len(equity_min_grid) * len(damp_grid)} combo):")
    print(f"     {'Combo':<30} {'IS Sharpe':>10} {'IS MaxDD':>9}")
    print(f"     {'─' * 51}")

    for eq_min in equity_min_grid:
        for damp in damp_grid:
            params = TrendMomentumParams(
                name=f"L_em{eq_min:.1f}_d{damp:.2f}",
                damp_factor=damp,
                momentum_strength=0.5,
                equity_min=eq_min,
            )
            opt = TrendMomentumOptimizer(params, equity_rics=equity_rics)
            try:
                rets, _ = _run_backtest(returns_is, opt, bc_is, min_history)
            except Exception as exc:
                logger.debug("IS grid eq_min=%.1f damp=%.2f failed: %s", eq_min, damp, exc)
                continue
            if rets.empty:
                continue

            sh = sharpe_ratio(rets)
            mdd = max_drawdown(rets)
            combo_label = f"equity_min={eq_min:.1f}  damp={damp:.2f}"
            print(f"     {combo_label:<30} {sh:>10.3f} {-mdd:>9.1%}")

            registry.record(
                config_hash,
                "sprint-l-equity-structural",
                {
                    "equity_min": eq_min,
                    "damp": damp,
                    "mom": 0.5,
                },
                "IS 2005-2011",
                {"sharpe": round(sh, 4), "max_dd": round(float(mdd), 4)},
            )

            if sh > best_sharpe:
                best_sharpe = sh
                best_equity_min = eq_min
                best_damp = damp

    print(f"\n     Selected: equity_min={best_equity_min:.1f}, damp={best_damp:.2f}  "
          f"(IS Sharpe={best_sharpe:.3f})")
    return best_equity_min, best_damp


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _plot(
    strategy_name: str,
    oos_rets: pd.Series,
    spy_rets: pd.Series,
    bh6040_rets: pd.Series,
    sprint_i_bal_rets: pd.Series,
    oos_start: str,
    oos_end: str,
    initial: float = 10_000.0,
) -> None:
    palette = {
        strategy_name: "#2563eb",
        "SPY B&H": "#9ca3af",
        "60/40 SPY/IEF": "#f59e0b",
        "Sprint I Balanced (unlevered)": "#6b7280",
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1.5]})
    fig.suptitle(
        f"JeanClaude Sprint L — Base Strutturale Equity + De-risk Trend | OOS {oos_start[:4]}–{oos_end[:4]}\n"
        "Yahoo auto_adjust (total return) | IS 2005-2011 | mom=0.5 congelato | lag=1 | 10bps",
        fontsize=12,
    )

    all_series = [
        ("SPY B&H", spy_rets),
        ("60/40 SPY/IEF", bh6040_rets),
        ("Sprint I Balanced (unlevered)", sprint_i_bal_rets),
        (strategy_name, oos_rets),
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
        is_benchmark = label in ("SPY B&H", "60/40 SPY/IEF", "Sprint I Balanced (unlevered)")
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
    best_equity_min: float,
    best_damp: float,
    strategy_name: str,
    all_metrics: list[dict],
    dsr_rows: list[dict],
    n_trials_l: int,
    n_trials_cumulative: int,
    verdict_a: str,
    verdict_b: str,
    verdict_c: str,
    overall_pass: bool,
    grid_border_warning: str,
    avg_equity_share: float,
    annual_turnover: float,
) -> None:
    _OUT_DIR_REPORTS.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Sprint L — Base Strutturale Equity + De-risk Trend: dominare il 60/40?\n",
        f"**Config hash:** `{config_hash}`  ",
        f"**Universo:** {len(cfg.universe)} ETF (identico a Sprint I — ex-ante dal design doc)  ",
        f"**IS:** {cfg.is_start} → {cfg.is_end}  |  **OOS:** {cfg.oos_start} → {cfg.oos_end}  ",
        f"**Fonte prezzi:** Yahoo auto_adjust=True (total return)  ",
        f"**Note:** {cfg.notes}  ",
        "",
        "## Parametri selezionati (IS)\n",
        f"- `equity_min` = {best_equity_min:.1f}  (IS grid: {list(_EQUITY_MIN_GRID)})",
        f"- `damp_factor` = {best_damp:.2f}  (IS grid: {list(cfg.damp_grid)})",
        "- `momentum_strength` = 0.5  (CONGELATO da Sprint I Balanced)",
        "",
        f"{grid_border_warning}",
        "",
        "## Statistiche OOS\n",
        f"- **Equity share media effettiva OOS:** {avg_equity_share:.1%}",
        f"- **Turnover annuo OOS:** {annual_turnover:.1%}",
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
        f"**n_trials questo esperimento (config_hash):** {n_trials_l}  ",
        f"**n_trials cumulativi (+ {_PRIOR_TRIALS} da I+J+K):** {n_trials_cumulative}  ",
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
        f"### Criterio (a): Sharpe OOS ≥ Sharpe 60/40\n",
        verdict_a,
        "",
        f"### Criterio (b): MaxDD OOS non più profondo del MaxDD 60/40\n",
        verdict_b,
        "",
        f"### Criterio (c): CAGR OOS ≥ 8%\n",
        verdict_c,
        "",
        f"### Verdetto complessivo: **{overall_label}**\n",
        "",
        "---",
        "_Report generato automaticamente da scripts/backtest_sprint_l.py_",
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
    print("  JeanClaude Sprint L — Base Strutturale Equity + De-risk Trend")
    print(f"{'═' * 70}\n")

    # ── 1. Config + hash + registry ──────────────────────────────────────
    print("1/7  Caricamento config congelata...")
    cfg = ExperimentConfig.from_json(_CONFIG_PATH)
    config_hash = cfg.config_hash()
    print(f"     Config hash: {config_hash}")
    print(f"     Universo: {len(cfg.universe)} ETF  |  IS: {cfg.is_start}→{cfg.is_end}  |  OOS: {cfg.oos_start}→{cfg.oos_end}")
    print(f"     damp_grid={cfg.damp_grid}  equity_min_grid={_EQUITY_MIN_GRID}  mom_frozen=0.5")

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

    # ── 3. IS grid: 6 combo ──────────────────────────────────────────────
    print(f"\n3/7  IS calibration ({cfg.is_start[:4]}–{cfg.is_end[:4]}) — "
          f"6 combo (3 equity_min × 2 damp), mom=0.5 CONGELATO...")

    best_equity_min, best_damp = _calibrate_is(
        returns_is=returns_is,
        equity_rics=equity_rics,
        damp_grid=cfg.damp_grid,
        equity_min_grid=_EQUITY_MIN_GRID,
        registry=registry,
        config_hash=config_hash,
        min_history=MIN_HIST,
    )

    n_trials_l = registry.n_trials(config_hash=config_hash)
    n_trials_cumulative = n_trials_l + _PRIOR_TRIALS

    print(f"\n     Trials questo hash: {n_trials_l}")
    print(f"     Trials cumulativi (+{_PRIOR_TRIALS} da I+J+K): {n_trials_cumulative}")

    # Grid-border warning (plan requirement)
    grid_border_warning = ""
    at_border = (
        best_equity_min == min(_EQUITY_MIN_GRID) or best_equity_min == max(_EQUITY_MIN_GRID) or
        best_damp == min(cfg.damp_grid) or best_damp == max(cfg.damp_grid)
    )
    if at_border:
        grid_border_warning = (
            f"> ⚠️  **AVVISO BORDO GRIGLIA:** il combo selezionato "
            f"(equity_min={best_equity_min:.1f}, damp={best_damp:.2f}) è sul bordo della griglia IS. "
            "Il parametro ottimale potrebbe essere fuori griglia — interpretare con cautela."
        )
        print(f"\n     *** AVVISO BORDO GRIGLIA: equity_min={best_equity_min:.1f}, "
              f"damp={best_damp:.2f} è sul bordo ***")
    else:
        grid_border_warning = (
            f"> Il combo selezionato (equity_min={best_equity_min:.1f}, damp={best_damp:.2f}) "
            "NON è sul bordo della griglia IS."
        )

    # ── 4. OOS — una volta con combo congelato ───────────────────────────
    strategy_name = f"Sprint L (em={best_equity_min:.1f}, d={best_damp:.2f})"
    print(f"\n4/7  OOS ({cfg.oos_start[:4]}–{cfg.oos_end[:4]}) — run unico: {strategy_name}...")

    params_best = TrendMomentumParams(
        name=strategy_name,
        damp_factor=best_damp,
        momentum_strength=0.5,
        equity_min=best_equity_min,
    )
    bc_oos = BacktestConfig(
        rebalance_freq=cfg.rebalance_freq,
        transaction_cost_bps=cfg.tc_bps,
        execution_lag=cfg.execution_lag,
        min_history=MIN_HIST,
    )

    opt_best = TrendMomentumOptimizer(params_best, equity_rics=equity_rics)
    rets_l, wh_l = _run_backtest(returns_oos, opt_best, bc_oos, MIN_HIST)

    avg_equity_share = _effective_equity_share(wh_l, equity_rics)
    annual_turnover_val = _annual_turnover(wh_l)
    print(f"     Equity share media effettiva OOS: {avg_equity_share:.1%}")
    print(f"     Turnover annuo OOS: {annual_turnover_val:.1%}")

    # ── 5. Benchmark: SPY B&H, 60/40 10bps, Sprint I Balanced unlevered ─
    print("\n5/7  Benchmark (SPY B&H, 60/40 10bps, Sprint I Balanced unlevered)...")
    spy_ric, ief_ric = "SPY.P", "IEF.O"

    spy_rets_raw = returns_oos[spy_ric].dropna() if spy_ric in returns_oos.columns else pd.Series(dtype=float)
    common_idx = rets_l.index
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

    # Sprint I Balanced unlevered: damp=0.25, mom=0.5, no floor, no overlay
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

    # ── 6. Metrics + DSR + verdetti ex-ante ─────────────────────────────
    print("\n6/7  Calcolo metriche, DSR, verdetti ex-ante...")

    spy_m = _metrics_row("SPY B&H", spy_rets_aligned)
    bh6040_m = (
        _metrics_row("60/40 SPY/IEF", bh6040_rets)
        if not bh6040_rets.empty
        else _metrics_row("60/40 SPY/IEF", spy_rets_aligned * 0)
    )
    spi_bal_m = _metrics_row("Sprint I Balanced (unlevered)", sprint_i_bal_rets)
    sprint_l_m = _metrics_row(strategy_name, rets_l)

    all_metrics = [spy_m, bh6040_m, spi_bal_m, sprint_l_m]

    print(f"\n  {'Strategy':<45} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8}")
    print(f"  {'─' * 69}")
    for m in all_metrics:
        print(f"  {m['Strategy']:<45} {m['CAGR']:>7.1%} {m['Sharpe']:>7.2f} {-m['MaxDD']:>8.1%}")

    # DSR
    dsr_rows: list[dict] = []
    sr_oos = float(sharpe_ratio(rets_l))
    obs = len(rets_l.dropna())
    sk = float(rets_l.dropna().skew())
    ku = float(rets_l.dropna().kurt()) + 3.0

    print(f"\n  DSR (n_trials_l={n_trials_l}, cumulativo={n_trials_cumulative}):")
    for nt, label in [(n_trials_l, "L"), (n_trials_cumulative, "L+prior")]:
        nt_safe = max(nt, 1)
        dsr = deflated_sharpe_ratio(sr_oos, n_trials=nt_safe, obs=obs, skewness=sk, kurtosis=ku)
        flag = "✓ PASS" if dsr >= 0.75 else "  FAIL"
        tag = f"{strategy_name} (n={label})"
        print(f"  {tag:<55} DSR={dsr:.3f}  {flag}")
        dsr_rows.append({"strategy": tag, "n_trials": nt_safe, "dsr": dsr})

    # ── Three explicit ex-ante verdicts ────────────────────────────────
    bh6040_sharpe = float(bh6040_m["Sharpe"])
    bh6040_maxdd = float(bh6040_m["MaxDD"])
    l_sharpe = float(sprint_l_m["Sharpe"])
    l_maxdd = float(sprint_l_m["MaxDD"])
    l_cagr = float(sprint_l_m["CAGR"])

    pass_a = l_sharpe >= bh6040_sharpe
    # max_drawdown() ritorna magnitudine POSITIVA (0.164 = -16.4%): ≤ = meno profondo
    pass_b = l_maxdd <= bh6040_maxdd
    pass_c = l_cagr >= 0.08
    overall_pass = pass_a and pass_b and pass_c

    verdict_a = (
        f"**{'PASS ✓' if pass_a else 'FAIL ✗'}** — "
        f"Sprint L Sharpe={l_sharpe:.2f} vs 60/40 Sharpe={bh6040_sharpe:.2f}"
    )
    verdict_b = (
        f"**{'PASS ✓' if pass_b else 'FAIL ✗'}** — "
        f"Sprint L MaxDD={-l_maxdd:.1%} vs 60/40 MaxDD={-bh6040_maxdd:.1%} "
        f"({'non più profondo' if pass_b else 'più profondo'})"
    )
    verdict_c = (
        f"**{'PASS ✓' if pass_c else 'FAIL ✗'}** — "
        f"Sprint L CAGR={l_cagr:.1%} vs soglia=8.0%"
    )

    print(f"\n  ──── VERDETTI EX-ANTE ────")
    print(f"  (a) Sharpe ≥ 60/40 Sharpe:     {'PASS ✓' if pass_a else 'FAIL ✗'}  "
          f"(L={l_sharpe:.2f}, ref={bh6040_sharpe:.2f})")
    print(f"  (b) MaxDD ≤ 60/40 MaxDD:       {'PASS ✓' if pass_b else 'FAIL ✗'}  "
          f"(L={-l_maxdd:.1%}, ref={-bh6040_maxdd:.1%})")
    print(f"  (c) CAGR ≥ 8%:                 {'PASS ✓' if pass_c else 'FAIL ✗'}  "
          f"(L={l_cagr:.1%})")
    overall_label = "PASS (tutti e tre)" if overall_pass else "FAIL (almeno uno non soddisfatto)"
    print(f"  ──── OVERALL: {overall_label} ────")
    if grid_border_warning.startswith("> ⚠️"):
        print(f"\n  *** ATTENZIONE BORDO GRIGLIA ***")

    # ── 7. Chart + report ────────────────────────────────────────────────
    print("\n7/7  Chart + report...")
    _plot(
        strategy_name=strategy_name,
        oos_rets=rets_l,
        spy_rets=spy_rets_aligned,
        bh6040_rets=bh6040_rets,
        sprint_i_bal_rets=sprint_i_bal_rets,
        oos_start=cfg.oos_start,
        oos_end=cfg.oos_end,
    )

    _write_report(
        cfg=cfg,
        config_hash=config_hash,
        best_equity_min=best_equity_min,
        best_damp=best_damp,
        strategy_name=strategy_name,
        all_metrics=all_metrics,
        dsr_rows=dsr_rows,
        n_trials_l=n_trials_l,
        n_trials_cumulative=n_trials_cumulative,
        verdict_a=verdict_a,
        verdict_b=verdict_b,
        verdict_c=verdict_c,
        overall_pass=overall_pass,
        grid_border_warning=grid_border_warning,
        avg_equity_share=avg_equity_share,
        annual_turnover=annual_turnover_val,
    )

    print(f"  Chart  → {_OUT_CHART}")
    print(f"  Report → {_OUT_REPORT}")
    print(f"  Trials → {_REGISTRY_PATH}")
    print(f"  Tempo totale: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
