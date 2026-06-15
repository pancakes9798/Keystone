#!/usr/bin/env python3
"""
Sprint E — Institutional-Grade Tearsheet
=========================================
Walk-forward OOS backtest (2000-2026) con modello Almgren-Chriss.

Analisi:
  1. Core metrics: tutte le strategie vs benchmark (con AC costs)
  2. Bootstrap CI al 95% su Sharpe, CAGR, Max Drawdown (Politis-Romano)
  3. Deflated Sharpe Ratio (Bailey & López de Prado) — corregge per multiple testing
  4. Analisi per periodo: dot-com, GFC, post-GFC, COVID, rate hike
  5. CPCV (Combinatorial Purged Cross-Validation) — distribuzione Sharpe OOS
  6. Griglia parametrica: robustezza intorno ai valori v16
  7. Rolling Sharpe a 12 mesi

Dati: Yahoo Finance (default) o Refinitiv/LSEG (--source=refinitiv).
AUM fisso: €100,000.

Run:
    uv run python scripts/backtest_tearsheet.py                    # Yahoo
    uv run python scripts/backtest_tearsheet.py --source=refinitiv # Refinitiv (prod)
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from jeanclaude.backtest.engine import BacktestConfig, BacktestEngine
from jeanclaude.backtest.bootstrap import stationary_bootstrap_ci
from jeanclaude.backtest.cpcv import CPCV, CPCVConfig
from jeanclaude.backtest.metrics import (
    annualized_return, calmar_ratio, deflated_sharpe_ratio,
    max_drawdown, sharpe_ratio, sortino_ratio,
)
from jeanclaude.costs import AlmgrenChrissModel, CostConfig
from jeanclaude.portfolio.optimizer import TrendMomentumOptimizer, TrendMomentumParams
from jeanclaude.portfolio.optimizer.hrp import HRPOptimizer
from jeanclaude.portfolio.covariance.historical import SampleCovariance

# ── Configurazione ─────────────────────────────────────────────────────────────

START      = "1998-01-01"   # dati storici (alcuni asset non esistono prima)
IS_END     = "2011-12-31"   # fine periodo in-sample
OOS_START  = "2012-01-01"   # inizio OOS genuino (post-IS 2005-2011)
END        = "2026-04-30"
MIN_HIST   = 252
INITIAL_AUM = 100_000.0
# N_TRIALS_IS: usato nella tabella DSR a sensibilità (vedi sezione 5)
# 50 = sole griglie v16; 200 ≈ griglie × 16 versioni; 500 ≈ include altri ~12 script
_DSR_N_TRIALS = (50, 200, 500)
_DSR_N_TRIALS_VERDICT = 200   # n_trials usato per il verdetto PASS/WARN/FAIL
REBAL_FREQ  = "ME"

# Spread fissi per asset class (bps → fraction)
_SPREAD_BPS: dict[str, float] = {
    "AAPL.O": 3, "MSFT.O": 3, "AMZN.O": 4, "NVDA.O": 4,
    "JPM.N":  4, "BRKb.N": 4, "JNJ.N":  4, "PFE.N":  4,
    "PG.N":   4, "KO.N":   4, "WMT.N":  4, "XOM.N":  4,
    "CAT.N":  5, "TM.N":  10, "NVS.N": 10, "SAP.N": 10,
    "GLD.P":  3, "TLT.O":  2,
}

TICKERS_RIC_TO_YAHOO = {
    "AAPL.O": "AAPL",  "MSFT.O": "MSFT",  "AMZN.O": "AMZN",  "NVDA.O": "NVDA",
    "JPM.N":  "JPM",   "BRKb.N": "BRK-B", "JNJ.N":  "JNJ",   "PFE.N":  "PFE",
    "PG.N":   "PG",    "KO.N":   "KO",    "WMT.N":  "WMT",   "XOM.N":  "XOM",
    "CAT.N":  "CAT",   "TM.N":   "TM",    "NVS.N":  "NVS",   "SAP.N":  "SAP",
    "GLD.P":  "GLD",   "TLT.O":  "TLT",
}

_EQUITY_RICS: frozenset[str] = frozenset({
    "AAPL.O","MSFT.O","AMZN.O","NVDA.O","JPM.N","BRKb.N",
    "JNJ.N","PFE.N","PG.N","KO.N","WMT.N","XOM.N",
    "CAT.N","TM.N","NVS.N","SAP.N",
})

# IS-calibrated v16
STRATEGIES: dict[str, TrendMomentumParams] = {
    "Conservative": TrendMomentumParams("Conservative", damp_factor=0.40, momentum_strength=0.00),
    "Balanced":     TrendMomentumParams("Balanced",     damp_factor=0.40, momentum_strength=1.00),
    "Aggressive":   TrendMomentumParams("Aggressive",   damp_factor=0.30, momentum_strength=2.00),
}

# Periodi chiave per l'analisi
PERIODS: dict[str, tuple[str, str]] = {
    "Dot-com crash (2000-02)": ("2000-01-01", "2002-12-31"),
    "Recovery (2003-07)":      ("2003-01-01", "2007-12-31"),
    "GFC (2008-09)":           ("2008-01-01", "2009-12-31"),
    "Post-GFC bull (2010-19)": ("2010-01-01", "2019-12-31"),
    "COVID crash (2020)":      ("2020-01-01", "2020-12-31"),
    "Rate hike (2022)":        ("2022-01-01", "2022-12-31"),
    "Recovery (2023-26)":      ("2023-01-01", "2026-04-30"),
}


# ── Utility ────────────────────────────────────────────────────────────────────

def _fmt(x: float, pct: bool = True, decimals: int = 1) -> str:
    if pct:
        return f"{x*100:{'+' if x > 0 else ''}.{decimals}f}%"
    return f"{x:.{decimals}f}"


def _var_cvar(returns: pd.Series, level: float = 0.05) -> tuple[float, float]:
    """VaR and CVaR at given significance level (left tail)."""
    q = returns.quantile(level)
    cvar = returns[returns <= q].mean()
    return float(q), float(cvar)


def _full_metrics(returns: pd.Series, label: str = "") -> dict:
    if returns.empty or len(returns) < 20:
        return {"Strategy": label, "CAGR%": "n/a", "Sharpe": "n/a"}
    oos = returns.loc[OOS_START:]
    if len(oos) < 20:
        oos = returns
    var95, cvar95 = _var_cvar(oos)
    n_years = len(oos) / 252
    dd = -float(max_drawdown(oos)) * 100
    return {
        "Strategy":     label,
        "CAGR%":        round(annualized_return(oos) * 100, 2),
        "Sharpe":       round(sharpe_ratio(oos), 3),
        "Sortino":      round(sortino_ratio(oos), 3),
        "Calmar":       round(calmar_ratio(oos), 3),
        "MaxDD%":       round(dd, 1),
        "VaR95%":       round(var95 * 100, 2),
        "CVaR95%":      round(cvar95 * 100, 2),
        "Turnover%/y":  "n/a",   # filled later if available
    }


# ── Data loading ───────────────────────────────────────────────────────────────

def fetch_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns: prices (RIC cols), adv (RIC cols), spread (RIC cols)."""
    rics  = list(TICKERS_RIC_TO_YAHOO.keys())
    yahoo = list(TICKERS_RIC_TO_YAHOO.values())
    print("  Fetching prices & volumes from Yahoo Finance...")
    raw = yf.download(yahoo, start=START, end=END, progress=False, auto_adjust=True)

    close  = raw["Close"].copy()
    volume = raw["Volume"].copy()

    y2r = {v: k for k, v in TICKERS_RIC_TO_YAHOO.items()}
    close.columns  = [y2r.get(str(c), str(c)) for c in close.columns]
    volume.columns = [y2r.get(str(c), str(c)) for c in volume.columns]

    close  = close.ffill().dropna(how="all")
    volume = volume.ffill().dropna(how="all")

    # GLD proxy pre-2004
    _GLD_LAUNCH = pd.Timestamp("2004-11-18")
    if "GLD.P" in close.columns:
        try:
            gc = yf.download("GC=F", start=START, end=END, progress=False, auto_adjust=True)
            gc_close = gc["Close"].squeeze()
            gc_close.index = pd.to_datetime(gc_close.index).tz_localize(None)
            overlap = close["GLD.P"].dropna().index.intersection(gc_close.dropna().index)
            if len(overlap):
                scale = close.loc[overlap[0], "GLD.P"] / gc_close.loc[overlap[0]]
                gc_scaled = gc_close * scale
                mask = close.index < _GLD_LAUNCH
                close.loc[mask, "GLD.P"] = gc_scaled.reindex(close.index[mask])
        except Exception:
            pass

    close  = close.ffill().dropna(how="all")
    volume = volume.ffill().dropna(how="all")

    adv = volume.rolling(20, min_periods=5).mean().ffill()
    spread = pd.DataFrame(
        {ric: _SPREAD_BPS.get(ric, 5) / 10_000 for ric in rics},
        index=close.index,
    )
    print(f"  {close.shape[0]} days × {close.shape[1]} assets  [{close.index[0].date()} → {close.index[-1].date()}]")
    return close.dropna(how="all"), adv.dropna(how="all"), spread


def fetch_data_refinitiv() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carica prezzi da Refinitiv/LSEG + Yahoo per estensione ETF.

    Returns: prices (RIC cols), adv (RIC cols), spread (RIC cols fissi).
    ADV = ACVOL_1 da Refinitiv dove disponibile, altrimenti stima costante.
    Spread = fisso per asset class (BID/ASK storico non disponibile per tutto il periodo).
    """
    from jeanclaude.data import DataConfig, DataLoader
    from jeanclaude.data.ingestion.refinitiv import RefinitivSource

    rics = list(TICKERS_RIC_TO_YAHOO.keys())

    # ── Prezzi: Refinitiv primario + Yahoo fallback per ETF storici ────────────
    print("  Fetching prices from Refinitiv/LSEG (platform session)...")
    config = DataConfig(
        price_source="refinitiv",
        macro_source="refinitiv",
        refinitiv_session="platform",
    )
    loader = DataLoader(config)
    prices_raw = loader.get_prices(rics, start=START, end=END)
    n_loaded = prices_raw.notna().any().sum()
    print(f"  Refinitiv: {n_loaded}/{len(rics)} asset caricati")

    # Identifica asset con copertura Refinitiv parziale (>5% NaN) per supplemento Yahoo
    nan_pct = prices_raw.isna().mean()
    print("  Asset con copertura parziale (>5% NaN su Refinitiv):")
    for ric, pct in nan_pct[nan_pct > 0.05].items():
        first = prices_raw[ric].first_valid_index()
        print(f"    {ric:<10} NaN={pct*100:.1f}%  first={str(first)[:10] if first else 'N/A'}")

    # Estendi con Yahoo per asset con copertura storica limitata su Refinitiv.
    # WMT.N: migrazione RIC Refinitiv (dati solo dal 2025-12-09).
    # NVS.N, TM.N: ADR con copertura Refinitiv limitata pre-2000.
    # GLD.P, TLT.O: ETF con storia recente (lancio 2004, 2002).
    _YAHOO_SUPPLEMENTS: dict[str, str] = {
        "GLD.P":  "GLD",    # ETF 2004; GC=F proxy pre-lancio aggiunto sotto
        "TLT.O":  "TLT",    # ETF 2002
        "AAPL.O": "AAPL",
        "MSFT.O": "MSFT",
        "AMZN.O": "AMZN",
        "WMT.N":  "WMT",    # RIC migrato su Refinitiv — usa Yahoo per tutta la storia
        "NVS.N":  "NVS",    # ADR, copertura Refinitiv da 2000-05-11
        "TM.N":   "TM",     # ADR, copertura Refinitiv da 1999-09-29
    }
    _GLD_LAUNCH = pd.Timestamp("2004-11-18")

    prices = prices_raw.copy()
    for ric, ticker in _YAHOO_SUPPLEMENTS.items():
        if ric not in prices.columns:
            continue
        try:
            raw = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
            if raw.empty:
                continue
            yh = raw["Close"].squeeze().rename(ric)
            yh.index = pd.to_datetime(yh.index).tz_localize(None)
            ref_series = prices[ric].dropna()
            if ref_series.empty:
                prices[ric] = yh
            else:
                ref_first = ref_series.index[0]
                yh_before = yh.loc[:ref_first].dropna().iloc[:-1]
                if not yh_before.empty:
                    yh_at_join = yh.reindex([ref_first]).dropna()
                    if not yh_at_join.empty:
                        scale = float(ref_series.iloc[0]) / float(yh_at_join.iloc[0])
                        prices[ric] = pd.concat([yh_before * scale, ref_series]).sort_index()
        except Exception as exc:
            print(f"  ⚠ Yahoo supplement failed per {ric}: {exc}")

    # GLD proxy pre-2004 via gold futures
    if "GLD.P" in prices.columns:
        try:
            gc_raw = yf.download("GC=F", start=START, end=str(_GLD_LAUNCH.date()),
                                 progress=False, auto_adjust=True)
            if not gc_raw.empty:
                gc = gc_raw["Close"].squeeze().rename("GLD.P")
                gc.index = pd.to_datetime(gc.index).tz_localize(None)
                gld_series = prices["GLD.P"].dropna()
                if not gld_series.empty:
                    gld_first = gld_series.index[0]
                    gc_at_join = gc.reindex([gld_first]).dropna()
                    if not gc_at_join.empty:
                        scale = float(gld_series.iloc[0]) / float(gc_at_join.iloc[0])
                        gc_before = gc.loc[:gld_first].iloc[:-1] * scale
                        if not gc_before.empty:
                            prices["GLD.P"] = pd.concat([gc_before, gld_series]).sort_index()
        except Exception:
            pass

    prices = prices.ffill().dropna(how="all")
    print(f"  Prezzi finali: {prices.shape[0]} giorni × {prices.shape[1]} asset"
          f"  [{prices.index[0].date()} → {prices.index[-1].date()}]")

    # ── ADV: tenta ACVOL_1 da Refinitiv, fallback su stima costante ────────────
    print("  Fetching ADV (ACVOL_1) from Refinitiv...")
    try:
        src = RefinitivSource(session_type="platform")
        vol_raw = src.get_prices(rics, start=START, end=END, field="volume")
        vol_raw = vol_raw.reindex(prices.index).ffill()
        adv = vol_raw.rolling(20, min_periods=5).mean().ffill()
        n_vol = adv.notna().any().sum()
        print(f"  ADV: {n_vol}/{len(rics)} asset con volume Refinitiv")
        # Riempi i mancanti con 1M shares (stima conservativa)
        for ric in rics:
            if ric not in adv.columns or adv[ric].isna().all():
                adv[ric] = 1_000_000.0
        adv = adv.reindex(columns=rics, fill_value=1_000_000.0)
    except Exception as exc:
        print(f"  ⚠ ADV Refinitiv fallito: {exc} — uso stima costante 1M shares")
        adv = pd.DataFrame(1_000_000.0, index=prices.index, columns=rics)

    # ── Spread: fisso per asset class ──────────────────────────────────────────
    # BID/ASK storico non disponibile per tutto il periodo OOS (2001-2026).
    # Usiamo le stime conservative per asset class, coerenti con il tearsheet Yahoo.
    spread = pd.DataFrame(
        {ric: _SPREAD_BPS.get(ric, 5) / 10_000 for ric in rics},
        index=prices.index,
    )
    return prices, adv, spread


# ── Engine runner ─────────────────────────────────────────────────────────────

def run_strategy(
    returns: pd.DataFrame,
    params: TrendMomentumParams,
    adv: pd.DataFrame,
    spread: pd.DataFrame,
    with_costs: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (portfolio_returns, turnover, cost_history)."""
    optimizer  = TrendMomentumOptimizer(params=params, equity_rics=_EQUITY_RICS)
    cost_model = AlmgrenChrissModel(CostConfig(fixed_aum=INITIAL_AUM)) if with_costs else None
    cfg        = BacktestConfig(transaction_cost_bps=0.0, rebalance_freq=REBAL_FREQ,
                                initial_capital=INITIAL_AUM)
    engine = BacktestEngine(optimizer=optimizer, config=cfg, cost_model=cost_model)
    result = engine.run(returns, min_history=MIN_HIST,
                        adv=adv if with_costs else None,
                        spread=spread if with_costs else None)
    return result.portfolio_returns, result.turnover, result.cost_history


def run_benchmark_60_40(returns: pd.DataFrame) -> pd.Series:
    """60% SPY / 40% TLT, monthly rebalance, 0 costs."""
    cols = [c for c in ["AAPL.O", "TLT.O"] if c in returns.columns]
    # Use SPY proxy: AAPL is not SPY. We need to construct SPY from the universe.
    # Use SPY/TLT if available, else use first equity + TLT
    spy_ric = next((r for r in returns.columns if r not in ["TLT.O", "GLD.P"]), None)
    tlt_ric = "TLT.O" if "TLT.O" in returns.columns else None
    if spy_ric is None or tlt_ric is None:
        return pd.Series(dtype=float)

    class FixedWeightOptimizer:
        def __init__(self, w): self.w = w
        def optimize(self, _): return self.w

    # Actually use SPY directly
    spy_raw = yf.download("SPY", start=START, end=END, progress=False, auto_adjust=True)
    spy_ret = spy_raw["Close"].squeeze().pct_change().dropna()
    spy_ret.index = pd.to_datetime(spy_ret.index).tz_localize(None)

    tlt_ret = returns[tlt_ric]
    common = spy_ret.index.intersection(tlt_ret.index)
    if len(common) < 100:
        return pd.Series(dtype=float)

    spy_r = spy_ret.loc[common]
    tlt_r = tlt_ret.loc[common]
    combined = pd.DataFrame({"SPY": spy_r, "TLT": tlt_r})

    # Monthly rebalance 60/40
    w60_40 = pd.Series({"SPY": 0.60, "TLT": 0.40})
    opt60 = FixedWeightOptimizer(w60_40)
    cfg60 = BacktestConfig(transaction_cost_bps=0.0, rebalance_freq=REBAL_FREQ)
    eng60 = BacktestEngine(optimizer=opt60, config=cfg60)
    res60 = eng60.run(combined, min_history=MIN_HIST)
    return res60.portfolio_returns


def run_equal_weight(returns: pd.DataFrame, adv: pd.DataFrame, spread: pd.DataFrame) -> pd.Series:
    """Equal weight 18 assets, monthly rebalance, with AC costs."""
    class EqualWeightOptimizer:
        def __init__(self, n): self.n = n
        def optimize(self, df: pd.DataFrame) -> pd.Series:
            return pd.Series(1.0 / len(df.columns), index=df.columns)
    ew = EqualWeightOptimizer(len(returns.columns))
    cost_model = AlmgrenChrissModel(CostConfig(fixed_aum=INITIAL_AUM))
    cfg = BacktestConfig(transaction_cost_bps=0.0, rebalance_freq=REBAL_FREQ)
    engine = BacktestEngine(optimizer=ew, config=cfg, cost_model=cost_model)
    result = engine.run(returns, min_history=MIN_HIST, adv=adv, spread=spread)
    return result.portfolio_returns


# ── Print helpers ─────────────────────────────────────────────────────────────

def _section(title: str, width: int = 80) -> None:
    print(f"\n{'═'*width}")
    print(f"  {title}")
    print(f"{'═'*width}")


def _sub(title: str, width: int = 80) -> None:
    print(f"\n{'─'*width}")
    print(f"  {title}")
    print(f"{'─'*width}")


def _table(rows: list[dict], cols: list[str] | None = None) -> None:
    if not rows:
        return
    cols = cols or list(rows[0].keys())
    widths = {c: max(len(str(c)), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  " + "  ".join(str(c).ljust(widths[c]) for c in cols)
    sep    = "  " + "  ".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for r in rows:
        print("  " + "  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint E Institutional Tearsheet")
    parser.add_argument(
        "--source", choices=["yahoo", "refinitiv"], default="yahoo",
        help="Fonte dati: 'yahoo' (default, survivorship bias) o 'refinitiv' (produzione)",
    )
    args = parser.parse_args()
    source_label = "Refinitiv/LSEG" if args.source == "refinitiv" else "Yahoo Finance"

    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " JEANCLAUDE — SPRINT E INSTITUTIONAL TEARSHEET".center(78) + "║")
    print("║" + f" AUM: €{INITIAL_AUM:,.0f}  |  IS: {START[:4]}-{IS_END[:4]}  |  OOS: {OOS_START} → {END}  |  AC: Almgren-Chriss".center(78) + "║")
    print("║" + f" Fonte dati: {source_label}".center(78) + "║")
    print("╚" + "═"*78 + "╝")

    # ── 1. Data ────────────────────────────────────────────────────────────────
    _section(f"1. DATA LOADING  [{source_label}]")
    if args.source == "refinitiv":
        prices, adv, spread = fetch_data_refinitiv()
    else:
        prices, adv, spread = fetch_data()
    rics = [c for c in TICKERS_RIC_TO_YAHOO if c in prices.columns]
    prices = prices[rics]
    adv    = adv.reindex(columns=rics, fill_value=1_000_000.0)
    spread = spread.reindex(columns=rics)
    returns = prices.pct_change().dropna(how="any")
    if returns.empty:
        print("  ⚠ WARNING: returns DataFrame vuoto dopo dropna — controllare copertura asset.")
        sys.exit(1)
    print(f"  Returns: {returns.shape[0]} giorni × {returns.shape[1]} asset  "
          f"[{returns.index[0].date()} → {returns.index[-1].date()}]")

    # Fetch SPY for benchmarks
    spy_raw = yf.download("SPY", start=START, end=END, progress=False, auto_adjust=True)
    spy_ret = spy_raw["Close"].squeeze().pct_change().dropna()
    spy_ret.index = pd.to_datetime(spy_ret.index).tz_localize(None)
    spy_oos = spy_ret.loc[OOS_START:]

    # ── 2. Run strategies & benchmarks ────────────────────────────────────────
    _section("2. RUNNING BACKTESTS (with Almgren-Chriss costs)")
    strategy_returns: dict[str, pd.Series] = {}
    turnover_series:  dict[str, pd.Series] = {}
    cost_series:      dict[str, pd.Series] = {}

    for name, params in STRATEGIES.items():
        print(f"  [{name}]...", end=" ", flush=True)
        r, to, ch = run_strategy(returns, params, adv, spread, with_costs=True)
        strategy_returns[name] = r.loc[OOS_START:]
        turnover_series[name]  = to
        cost_series[name]      = ch
        print(f"CAGR {annualized_return(r.loc[OOS_START:])*100:.1f}%  Sharpe {sharpe_ratio(r.loc[OOS_START:]):.2f}")

    print("  [60/40 Benchmark]...", end=" ", flush=True)
    r60 = run_benchmark_60_40(returns)
    r60 = r60.loc[OOS_START:] if not r60.empty else r60
    strategy_returns["60/40 Bench"] = r60
    if not r60.empty:
        print(f"CAGR {annualized_return(r60)*100:.1f}%  Sharpe {sharpe_ratio(r60):.2f}")
    else:
        print("SKIPPED (no SPY data)")

    print("  [Equal Weight +AC]...", end=" ", flush=True)
    rew = run_equal_weight(returns, adv, spread)
    strategy_returns["Equal Weight"] = rew.loc[OOS_START:]
    print(f"CAGR {annualized_return(rew.loc[OOS_START:])*100:.1f}%  Sharpe {sharpe_ratio(rew.loc[OOS_START:]):.2f}")

    strategy_returns["SPY B&H"] = spy_oos

    # ── 3. Core Metrics Table ─────────────────────────────────────────────────
    _section("3. CORE METRICS  (OOS: 2012 → 2026, AC costs included for Sprint E & EW)")
    rows = []
    for name, r in strategy_returns.items():
        m = _full_metrics(r, label=name)
        # Turnover
        if name in turnover_series and not turnover_series[name].empty:
            ann_to = turnover_series[name].mean() * 12 * 100
            m["Turnover%/y"] = f"{ann_to:.0f}%"
        # Cost
        if name in cost_series and not cost_series[name].empty:
            ann_bps = cost_series[name].mean() * 12 * 10_000
            m["AC bps/y"] = f"{ann_bps:.0f}"
        else:
            m["AC bps/y"] = "—"
        rows.append(m)
    cols = ["Strategy","CAGR%","Sharpe","Sortino","Calmar","MaxDD%","VaR95%","CVaR95%","Turnover%/y","AC bps/y"]
    _table(rows, cols)

    # ── 4. Bootstrap Confidence Intervals (95%) ───────────────────────────────
    _section("4. BOOTSTRAP CIs — 95% CONFIDENCE  (Politis-Romano stationary bootstrap, N=1000)")
    boot_rows = []
    for name in list(STRATEGIES.keys()) + ["SPY B&H"]:
        r = strategy_returns.get(name, pd.Series(dtype=float))
        if r.empty or len(r) < 60:
            continue
        print(f"  Bootstrap [{name}]...", end=" ", flush=True)
        res = stationary_bootstrap_ci(r, n_samples=1000, confidence=0.95, seed=42)
        print("done")
        boot_rows.append({
            "Strategy":       name,
            "CAGR% (point)":  f"{res.point['cagr']*100:.2f}%",
            "CAGR CI 95%":    f"[{res.lower['cagr']*100:.2f}%, {res.upper['cagr']*100:.2f}%]",
            "Sharpe (point)": f"{res.point['sharpe']:.3f}",
            "Sharpe CI 95%":  f"[{res.lower['sharpe']:.3f}, {res.upper['sharpe']:.3f}]",
            "MaxDD% (point)": f"{res.point['max_dd']*100:.1f}%",
            "MaxDD CI 95%":   f"[{res.lower['max_dd']*100:.1f}%, {res.upper['max_dd']*100:.1f}%]",
        })
    _table(boot_rows)

    # ── 5. Deflated Sharpe Ratio ───────────────────────────────────────────────
    _section("5. DEFLATED SHARPE RATIO  (Bailey & LdP) — tabella di sensibilità su n_trials")
    print("  Corregge per multiple testing: quante varianti IS sono state provate?")
    print("  50  = sole griglie v16 (stima ottimistica)")
    print("  200 ≈ griglie × 16 versioni di sviluppo IS (stima centrale — usata per il verdetto)")
    print("  500 ≈ include gli altri ~12 script di backtest del progetto (stima conservativa)")
    print()
    print(f"  {'Strategy':<18}  {'Sharpe OOS':>10}  {'Skew':>6}  {'Kurt':>6}", end="")
    for nt in _DSR_N_TRIALS:
        label = f"DSR(n={nt})"
        print(f"  {label:>11}", end="")
    print(f"  {'Verdict':>8}")
    print(f"  {'─'*18}  {'─'*10}  {'─'*6}  {'─'*6}", end="")
    for _ in _DSR_N_TRIALS:
        print(f"  {'─'*11}", end="")
    print(f"  {'─'*8}")
    for name in STRATEGIES:
        r = strategy_returns.get(name, pd.Series(dtype=float))
        if r.empty or len(r) < 60:
            continue
        sr = sharpe_ratio(r)
        sk = float(stats.skew(r.dropna()))
        ku = float(stats.kurtosis(r.dropna(), fisher=False))
        print(f"  {name:<18}  {sr:>10.3f}  {sk:>6.3f}  {ku:>6.3f}", end="")
        dsr_verdict = None
        for nt in _DSR_N_TRIALS:
            dsr = deflated_sharpe_ratio(sr, n_trials=nt, obs=len(r), skewness=sk, kurtosis=ku)
            print(f"  {dsr:>11.4f}", end="")
            if nt == _DSR_N_TRIALS_VERDICT:
                dsr_verdict = dsr
        verdict = "✓ PASS" if dsr_verdict >= 0.95 else ("⚠ WARN" if dsr_verdict >= 0.75 else "✗ FAIL")
        print(f"  {verdict:>8}")
    print()
    print("  Lettura: DSR = P(true Sharpe > 0) — soglie: ≥0.95 PASS, ≥0.75 WARN, <0.75 FAIL")
    print(f"  Verdetto basato su n_trials={_DSR_N_TRIALS_VERDICT} (stima centrale).")

    # ── 6. Period Analysis ────────────────────────────────────────────────────
    _section("6. ANALISI PER PERIODO")
    period_data: list[dict] = []
    for period_name, (p_start, p_end) in PERIODS.items():
        row: dict = {"Periodo": period_name}
        for name in list(STRATEGIES.keys()) + ["SPY B&H"]:
            r = strategy_returns.get(name, pd.Series(dtype=float))
            if r.empty:
                row[name] = "—"
                continue
            sub = r.loc[p_start:p_end]
            if len(sub) < 20:
                row[name] = "—"
                continue
            cagr = annualized_return(sub) * 100
            sr   = sharpe_ratio(sub)
            row[name] = f"{cagr:+.1f}% / SR {sr:.2f}"
        period_data.append(row)
    _table(period_data, ["Periodo"] + list(STRATEGIES.keys()) + ["SPY B&H"])

    # ── 7. Diagnostica combinatoria OOS ──────────────────────────────────────
    _section("7. Diagnostica combinatoria OOS (NO refit — non è CPCV)  — Strategia Aggressive")
    print("  k=6 gruppi, n_test=2  →  C(6,2) = 15 path  |  ogni path ≈ 33% dei dati")
    print("  Distribuzione del Sharpe sulle 15 combinazioni OOS")
    print("  NOTA: questa sezione NON ricalibra i parametri per fold (no refit).")
    print("  Il vero CPCV con refit per fold è in scripts/backtest_sprint_i.py (Task 9).")

    r_agg = strategy_returns["Aggressive"]
    if len(r_agg) > 60:
        oos_idx  = r_agg.index
        n        = len(oos_idx)
        k_splits = 6
        groups   = np.array_split(np.arange(n), k_splits)

        cpcv_sharpes = []
        cpcv_cagrs   = []
        cpcv_paths   = []

        for test_group_ids in combinations(range(k_splits), 2):
            test_pos = np.concatenate([groups[g] for g in test_group_ids])
            test_pos = np.sort(test_pos)
            sub_ret  = r_agg.iloc[test_pos]
            if len(sub_ret) < 20:
                continue
            sr  = sharpe_ratio(sub_ret)
            cag = annualized_return(sub_ret) * 100
            cpcv_sharpes.append(sr)
            cpcv_cagrs.append(cag)
            start_d = sub_ret.index[0].strftime("%Y")
            end_d   = sub_ret.index[-1].strftime("%Y")
            cpcv_paths.append({
                "Gruppi": f"{test_group_ids[0]+1},{test_group_ids[1]+1}",
                "Periodo": f"{start_d}~{end_d}",
                "N giorni": len(sub_ret),
                "CAGR%": f"{cag:+.1f}%",
                "Sharpe": f"{sr:.3f}",
                "Verdict": "✓" if sr > 0 else "✗",
            })

        _table(cpcv_paths)

        sharpes_arr = np.array(cpcv_sharpes)
        n_pos = (sharpes_arr > 0).sum()
        print(f"\n  Sharpe: mean={sharpes_arr.mean():.3f}  std={sharpes_arr.std():.3f}"
              f"  min={sharpes_arr.min():.3f}  max={sharpes_arr.max():.3f}")
        print(f"  Path positivi: {n_pos}/{len(sharpes_arr)} ({n_pos/len(sharpes_arr)*100:.0f}%)")
        verdict_cpcv = "✓ ROBUSTA" if n_pos >= 12 else ("⚠ DEBOLE" if n_pos >= 9 else "✗ NON ROBUSTA")
        print(f"  Valutazione CPCV: {verdict_cpcv}  (soglia: ≥80% path positivi = robusta)")

    # ── 8. Parameter Sensitivity Grid ────────────────────────────────────────
    _section("8. ROBUSTEZZA PARAMETRICA  (griglia intorno ai valori v16 Aggressive)")
    print("  damp_factor × momentum_strength  →  Sharpe OOS con AC costs")
    print("  Valore IS-calibrato v16: damp=0.30, momentum=2.00  (marcato con *)\n")

    damp_vals = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    mom_vals  = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # Header
    header_line = f"  {'damp↓ / mom→':>14} " + " ".join(f"{m:>6.1f}" for m in mom_vals)
    print(header_line)
    print("  " + "─" * (15 + 7 * len(mom_vals)))

    for damp in damp_vals:
        row_vals = []
        for mom in mom_vals:
            p = TrendMomentumParams("grid", damp_factor=damp, momentum_strength=mom)
            r_g, _, _ = run_strategy(returns, p, adv, spread, with_costs=True)
            sr_g = sharpe_ratio(r_g.loc[OOS_START:])
            marker = "*" if abs(damp - 0.30) < 0.01 and abs(mom - 2.00) < 0.01 else " "
            row_vals.append(f"{sr_g:>5.2f}{marker}")
        print(f"  {damp:>14.2f}  " + "  ".join(row_vals))

    print(f"\n  Leggenda: * = parametri v16 (IS-calibrati)")
    print("  Verde semantico: >0.50 = buono, 0.30-0.50 = accettabile, <0.30 = scarso")

    # ── 9. Rolling 12-month Sharpe ────────────────────────────────────────────
    _section("9. ROLLING SHARPE A 12 MESI  (Aggressive vs SPY)")
    r_agg = strategy_returns["Aggressive"]
    if len(r_agg) > 252:
        roll_sr_agg = r_agg.rolling(252).apply(lambda x: sharpe_ratio(pd.Series(x)), raw=False).dropna()
        roll_sr_spy = spy_oos.rolling(252).apply(lambda x: sharpe_ratio(pd.Series(x)), raw=False).dropna()

        common_idx = roll_sr_agg.index.intersection(roll_sr_spy.index)
        if len(common_idx) > 0:
            r_a = roll_sr_agg.loc[common_idx]
            r_s = roll_sr_spy.loc[common_idx]
            pct_positive_agg = (r_a > 0).mean() * 100
            pct_positive_spy = (r_s > 0).mean() * 100
            pct_beats_spy    = (r_a > r_s).mean() * 100
            print(f"  {'Anno':>6}  {'Aggressive':>10}  {'SPY B&H':>10}  {'Diff':>8}")
            print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*8}")
            years = sorted(r_a.resample("YE").last().dropna().index)
            for yr in years:
                yr_str = yr.strftime("%Y")
                a_val = r_a.resample("YE").last().loc[yr] if yr in r_a.resample("YE").last().index else float("nan")
                s_val = r_s.resample("YE").last().loc[yr] if yr in r_s.resample("YE").last().index else float("nan")
                diff  = a_val - s_val if not (np.isnan(a_val) or np.isnan(s_val)) else float("nan")
                print(f"  {yr_str:>6}  {a_val:>10.3f}  {s_val:>10.3f}  {diff:>+8.3f}")
            print(f"\n  Rolling Sharpe > 0: Aggressive {pct_positive_agg:.0f}%  |  SPY {pct_positive_spy:.0f}%")
            print(f"  Aggressive batte SPY rolling: {pct_beats_spy:.0f}% del tempo")

    # ── 10. Verdict ────────────────────────────────────────────────────────────
    _section("10. VERDICT COMPLESSIVO")
    r_agg = strategy_returns["Aggressive"]
    sr_agg = sharpe_ratio(r_agg)
    dsr_agg = deflated_sharpe_ratio(
        sr_agg, n_trials=_DSR_N_TRIALS_VERDICT, obs=len(r_agg),
        skewness=float(stats.skew(r_agg)),
        kurtosis=float(stats.kurtosis(r_agg, fisher=False)),
    )
    dd_agg = max_drawdown(r_agg)
    sr_spy = sharpe_ratio(spy_oos.reindex(r_agg.index, fill_value=0.0))
    alpha_sr = sr_agg - sr_spy

    # CPCV
    cpcv_n_pos = sum(1 for s in cpcv_sharpes if s > 0) if 'cpcv_sharpes' in dir() else 0
    cpcv_total = len(cpcv_sharpes) if 'cpcv_sharpes' in dir() else 15

    checks = [
        ("DSR ≥ 0.95",           dsr_agg >= 0.95,   f"{dsr_agg:.4f}"),
        ("Sharpe > 0.50",        sr_agg > 0.50,     f"{sr_agg:.3f}"),
        ("Alpha vs SPY > 0.10",  alpha_sr > 0.10,   f"{alpha_sr:.3f}"),
        ("Max DD < 35%",         abs(dd_agg) < 0.35, f"{abs(dd_agg)*100:.1f}%"),
        (f"CPCV ≥80% path pos",  cpcv_n_pos / max(cpcv_total,1) >= 0.80,
                                  f"{cpcv_n_pos}/{cpcv_total}"),
    ]

    all_pass = all(ok for _, ok, _ in checks)
    print()
    for check, ok, val in checks:
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {check:<30}  {val}")
    print()
    if all_pass:
        print("  ✓✓ TUTTI I CHECK SUPERATI — strategia Aggressive pronta per valutazione live.")
    else:
        n_fail = sum(1 for _, ok, _ in checks if not ok)
        print(f"  ⚠  {n_fail}/{len(checks)} check falliti — rivedere prima di procedere al live trading.")

    print("\n" + "═"*80)
    if args.source == "yahoo":
        print("  ⚠ SURVIVORSHIP BIAS: dati Yahoo Finance con 18 asset selezionati EX-POST")
        print("    (NVDA, AMZN, AAPL proiettati al 2000). I numeri Sharpe sono SOVRASTIMATI.")
        print("    Rilancia con --source=refinitiv per numeri production-grade.")
    else:
        print("  ⚠ UNIVERSO EX-POST: 16 mega-cap selezionate conoscendo i vincitori — i risultati")
        print("    sono ottimisti per selection bias, qualunque sia il vendor dei prezzi.")
        print("  ⚠ TOTAL RETURN: Refinitiv TRDPRC_1 (anche adjusted) è PRICE return — i dividendi")
        print("    mancano (~2-4%/anno sui titoli value): CAGR equity sottostimati.")
        print("    Verificato empiricamente con scripts/validate_total_return.py (2026-06-10).")
    print()
    print("  NOTA: spread bid-ask = fisso per asset class (4 bps equity US, 10 bps ADR,")
    print("    3 bps GLD/TLT). ADV da Refinitiv ACVOL_1 (volume in shares, non in EUR).")
    print("═"*80 + "\n")


if __name__ == "__main__":
    main()
