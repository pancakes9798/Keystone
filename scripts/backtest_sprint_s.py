#!/usr/bin/env python3
"""Sprint S — Trend/Regime equity-index overlay (convessità difensiva, BACKTEST-ONLY).

Spec congelata: docs/superpowers/specs/2026-06-14-sprint-s-trend-overlay-design.md
Nessun broker/conto/denaro. Overlay long/flat/short TSMOM 12m + gate regime HMM su
S&P 500 (1 MES), valutato su un core 60/40. Criteri ex-ante (a)-(d) overlay-su-core.
Run UNA VOLTA dopo pre-run review:  .venv/bin/python scripts/backtest_sprint_s.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from jeanclaude.backtest.metrics import (  # noqa: E402
    annualized_return, calmar_ratio, deflated_sharpe_ratio, max_drawdown,
    sharpe_ratio, sortino_ratio,
)
from jeanclaude.backtest.trend_overlay import evaluate_overlay_on_core, tsmom_position  # noqa: E402
from jeanclaude.costs.futures import MicroFuturesCost  # noqa: E402
from jeanclaude.data.ingestion.fred import FREDSource  # noqa: E402
from jeanclaude.portfolio.optimizer import walkforward_regimes  # noqa: E402
from jeanclaude.research import ExperimentConfig, TrialRegistry  # noqa: E402

logging.basicConfig(level=logging.WARNING)

HISTORY_START = "1998-01-01"
SUBPERIOD_START = "2011-01-01"
OVERLAY_FRAC = 0.15
MES_MULTIPLIER = 5.0
RF_SERIES = "DGS3MO"
MACRO_TICKERS = {"VIX": "^VIX", "Oil": "CL=F", "Copper": "HG=F", "Gold": "GC=F", "EURUSD": "EURUSD=X"}

DATA_DIR = Path(__file__).parent.parent / "data"
REGISTRY_PATH = DATA_DIR / "research" / "trials.jsonl"
REGIME_CACHE = DATA_DIR / "research" / "wf_regimes_sprint_s_seed42.parquet"
REPORT_PATH = Path(__file__).parent.parent / "docs" / "reports" / "sprint_s_results.md"
CHART_PATH = Path(__file__).parent.parent / "docs" / "backtest" / "backtest_sprint_s.png"
ROLL_MONTHS = {3, 6, 9, 12}

CONFIG = ExperimentConfig(
    name="sprint_s_trend_overlay",
    universe=("S&P 500 (SPY TR) + 60/40 SPY/IEF core; MES overlay",),
    equity_rics=(), is_start="", is_end="",
    oos_start="1999-01", oos_end="LAST_COMPLETE_MONTH",
    rebalance_freq="M", tc_bps=0.0, execution_lag=1,
    damp_grid=(), mom_grid_balanced=(), mom_grid_aggressive=(),
    price_field="adj_close", price_adjustments=("total_return",),
    notes=(
        "Overlay long/flat/short TSMOM 12-1 + gate regime HMM (stress=CONTRACTION) su "
        "S&P 500, espresso con 1 MES in excess return (spx_tr - rf, rf=FRED DGS3MO), "
        "costi MicroFuturesCost pessimistici, roll trimestrale. OVERLAY_FRAC=0.15 su core "
        "60/40 SPY/IEF. Criteri ex-ante: (a) MaxDD(combined)<MaxDD(core); (b) Calmar su; "
        "(c) overlay net>=0; (d) corr<=0 full E stress. BACKTEST-ONLY."
    ),
)


def _spy_daily(start, end):
    px = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if px is None or px.empty:
        raise RuntimeError("Download SPY fallito")
    close = px["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    daily = close.pct_change().dropna()
    daily.index = pd.to_datetime(daily.index).tz_localize(None).normalize()
    return daily


def _ief_daily(start, end):
    px = yf.download("IEF", start=start, end=end, auto_adjust=True, progress=False)
    if px is None or px.empty:
        raise RuntimeError("Download IEF fallito")
    close = px["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    daily = close.pct_change().dropna()
    daily.index = pd.to_datetime(daily.index).tz_localize(None).normalize()
    return daily


def _fetch_macro(start, end):
    """Macro per l'HMM (yfinance), come run_paper_etf._fetch_macro."""
    frames = {}
    for name, ticker in MACRO_TICKERS.items():
        try:
            s = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)["Close"].squeeze()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            frames[name] = s
        except Exception as e:  # noqa: BLE001
            logging.warning("Macro fetch fallita per %s (%s): %s", name, ticker, e)
    return pd.DataFrame(frames).ffill()


def _risk_free_monthly(start, end, month_ends):
    """rf mensile (frazione) da FRED DGS3MO (yield annuo %). Reindex sui month_ends."""
    src = FREDSource()
    df = src.get_macro([RF_SERIES], start, end)
    annual = df[RF_SERIES].copy()
    annual.index = pd.to_datetime(annual.index).tz_localize(None)
    monthly_pct = annual.resample("ME").last().ffill()
    rf = (monthly_pct / 100.0 / 12.0).reindex(month_ends).ffill().fillna(0.0)
    return rf


def yf_spy_level(spy_daily_returns: pd.Series, asof: pd.Timestamp) -> float:
    """Ricostruisce un livello SPY proxy da rendimenti cumulati (base 100 al primo giorno)."""
    cum = 100.0 * (1 + spy_daily_returns.loc[:asof]).cumprod()
    return float(cum.iloc[-1]) if len(cum) else 100.0


def main() -> None:
    print("=" * 72)
    print("SPRINT S — Trend/Regime equity-index overlay (BACKTEST-ONLY)")
    print("=" * 72)
    cfg_hash = CONFIG.config_hash()
    print(f"Config hash: {cfg_hash}")

    registry = TrialRegistry(REGISTRY_PATH)
    already = any(r.get("config_hash") == cfg_hash and r.get("window") == "OOS"
                  for r in registry._rows())
    if already and os.environ.get("JC_FORCE_RERUN") != "1":
        raise SystemExit(f"Config {cfg_hash} ha gia un run OOS — un solo run. Override: JC_FORCE_RERUN=1.")

    today = pd.Timestamp.today()
    end_str = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    last_complete = (today - pd.offsets.MonthEnd(1)).normalize()

    spy_d = _spy_daily(HISTORY_START, end_str)
    ief_d = _ief_daily(HISTORY_START, end_str)
    macro = _fetch_macro(HISTORY_START, end_str)

    spx_df = pd.DataFrame({"SPX": spy_d})
    spy_m = (1 + spy_d).resample("ME").prod() - 1
    ief_m = (1 + ief_d).resample("ME").prod() - 1
    core = (0.6 * spy_m + 0.4 * ief_m).dropna()
    core = core[core.index <= last_complete]

    month_ends = core.index
    first_ok = spx_df.index[0] + pd.offsets.MonthEnd(14)
    formations = [m for m in month_ends if m >= first_ok and m < last_complete]

    print(f"Regimi HMM walk-forward su {len(formations)} mesi (puo richiedere minuti, cache attiva)...")
    regimes = walkforward_regimes(macro, formations, random_state=42, cache_path=REGIME_CACHE)

    rf_m = _risk_free_monthly(HISTORY_START, end_str, month_ends)
    cost = MicroFuturesCost()

    overlay_rows = {}
    pos_prev = 0
    pos_series = {}
    for t in formations:
        regime = regimes.get(t, "UNKNOWN")
        pos = tsmom_position(spx_df, regime, asof=t)
        pos_series[t] = pos
        hold_month = t + pd.offsets.MonthEnd(1)
        if hold_month not in spy_m.index or hold_month not in rf_m.index:
            continue
        excess = float(spy_m.loc[hold_month] - rf_m.loc[hold_month])
        # Nozionale MES ~ moltiplicatore x livello indice SPX. SPY~SPX/10 -> SPX~SPY*10.
        # Serve solo l'ORDINE di grandezza (i costi sono una frazione minima del nozionale).
        spy_level = yf_spy_level(spy_d, t)
        notional = MES_MULTIPLIER * spy_level * 10.0
        trade_c = cost.trade_cost_return(pos - pos_prev, notional)
        roll_c = cost.roll_cost_return(pos, notional) if hold_month.month in ROLL_MONTHS else 0.0
        overlay_rows[hold_month] = pos * excess - trade_c - roll_c
        pos_prev = pos

    overlay_excess = pd.Series(overlay_rows, dtype=float).sort_index()
    pos_ser = pd.Series(pos_series, dtype=float).sort_index()
    core_eval = core.reindex(overlay_excess.index).dropna()
    overlay_excess = overlay_excess.reindex(core_eval.index)

    n_trials = registry.n_trials() + 1
    res = evaluate_overlay_on_core(core_eval, overlay_excess, OVERLAY_FRAC)
    res_sub = evaluate_overlay_on_core(
        core_eval[core_eval.index >= SUBPERIOD_START],
        overlay_excess[overlay_excess.index >= SUBPERIOD_START], OVERLAY_FRAC,
    )

    combined = core_eval + OVERLAY_FRAC * overlay_excess
    sk = float(stats.skew(overlay_excess)); ku = float(stats.kurtosis(overlay_excess, fisher=False))
    dsr = deflated_sharpe_ratio(sharpe_ratio(overlay_excess, 12), n_trials=n_trials,
                                obs=len(overlay_excess), skewness=sk, kurtosis=ku, periods_per_year=12)
    # NOTA (review 2026-06-14): yf_spy_level è un indice base-100 TOTAL-RETURN (~1250 nel 2026),
    # NON il livello SPX reale (~5400). Questo SOVRASTIMA implied_core ~2.3x. Il valore onesto
    # del run 2026-06-14 è ~€170k (5×5400 ≈ $27k / 0.15), non €421k — vedi la correzione nel
    # report. Per un futuro re-run sostituire con il livello SPX reale (prezzo SPY×10).
    spx_notional_for_1mes = MES_MULTIPLIER * yf_spy_level(spy_d, last_complete) * 10.0
    implied_core = spx_notional_for_1mes / OVERLAY_FRAC

    _print_and_report(res, res_sub, core_eval, combined, overlay_excess, pos_ser, regimes,
                      cfg_hash, n_trials, dsr, implied_core, registry)


def _print_and_report(res, res_sub, core, combined, overlay, pos, regimes,
                      cfg_hash, n_trials, dsr, implied_core, registry):
    print(f"\nMesi valutati: {len(overlay)} ({overlay.index[0].date()} -> {overlay.index[-1].date()})")
    print("\n--- Full sample ---")
    print(f"core   : CAGR {annualized_return(core,12):+.1%}  MaxDD -{res['core_maxdd']:.1%}  Calmar {res['core_calmar']:.2f}")
    print(f"combo  : CAGR {annualized_return(combined,12):+.1%}  MaxDD -{res['combined_maxdd']:.1%}  Calmar {res['combined_calmar']:.2f}")
    print(f"overlay: cum {res['overlay_cum']:+.1%}  corr_full {res['corr_full']:+.2f}  corr_stress {res['corr_stress']:+.2f}")
    print(f"(a) MaxDD {res['criterio_a']} | (b) Calmar {res['criterio_b']} | (c) overlay>=0 {res['criterio_c']} | (d) corr<=0 {res['criterio_d']}")
    print(f"VERDETTO (full): {res['verdict']}   | sub-2011: {res_sub['verdict']}")
    print(f"DSR overlay (n_trials={n_trials}): {dsr:.3f}")
    print(f"Nozionale di core implicito per 1 MES @ OVERLAY_FRAC: EUR {implied_core:,.0f}")

    registry.record(
        config_hash=cfg_hash, experiment=CONFIG.name,
        params={"overlay_frac": OVERLAY_FRAC, "signal": "tsmom12_1_regime_gated",
                "stress_regime": "CONTRACTION", "rf": RF_SERIES},
        window="OOS",
        metrics={**{f"full_{k}": v for k, v in res.items()},
                 **{f"sub_{k}": v for k, v in res_sub.items()},
                 "dsr": dsr, "n_months": len(overlay),
                 "implied_core_for_1mes": implied_core},
    )
    _write_figure(core, combined, overlay, pos, regimes, res)
    _write_report(res, res_sub, core, combined, overlay, cfg_hash, n_trials, dsr, implied_core)
    print(f"\nReport: {REPORT_PATH}\nChart:  {CHART_PATH}")


def _write_figure(core, combined, overlay, pos, regimes, res):
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    nav_core = (1 + core).cumprod()
    nav_combo = (1 + combined).cumprod()
    nav_ov = (1 + overlay).cumprod()
    dd_core = nav_core / nav_core.cummax() - 1
    dd_combo = nav_combo / nav_combo.cummax() - 1
    roll_corr = overlay.rolling(12).corr(core)
    stress = core <= core.quantile(0.10)

    fig, ax = plt.subplots(5, 1, figsize=(13, 17), sharex=False,
                           gridspec_kw={"height_ratios": [3, 1.6, 1.6, 1.2, 1.6]})
    ax[0].plot(nav_core.index, nav_core, label="Core 60/40", lw=1.4)
    ax[0].plot(nav_combo.index, nav_combo, label=f"Combined (+{OVERLAY_FRAC:.0%} overlay)", lw=1.6)
    ax[0].plot(nav_ov.index, nav_ov, label="Overlay standalone (excess)", lw=1.0, alpha=0.7, ls="--")
    ax[0].set_yscale("log"); ax[0].set_title("Sprint S — Trend/Regime overlay su core 60/40 (mensile)")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].fill_between(dd_core.index, dd_core, 0, alpha=0.4, label="Core")
    ax[1].plot(dd_combo.index, dd_combo, lw=1.2, color="tab:orange", label="Combined")
    ax[1].set_ylabel("Drawdown"); ax[1].legend(); ax[1].grid(alpha=0.3)

    ax[2].axhspan(-1, 0, color="tab:green", alpha=0.07)
    ax[2].plot(roll_corr.index, roll_corr, lw=1.0, color="tab:purple")
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_ylabel("Corr 12m overlay-core"); ax[2].grid(alpha=0.3)

    # Sfondo ombreggiato per regime HMM di stress (CONTRACTION) su NAV e posizione.
    stress_regime_dates = [pd.Timestamp(d) for d, r in regimes.items() if r == "CONTRACTION"]
    for axi in (ax[0], ax[3]):
        for d in stress_regime_dates:
            axi.axvspan(d, d + pd.offsets.MonthEnd(1), color="tab:red", alpha=0.08, lw=0)
    if stress_regime_dates:
        ax[3].axvspan(stress_regime_dates[0], stress_regime_dates[0] + pd.offsets.MonthEnd(1),
                      color="tab:red", alpha=0.08, lw=0, label="Regime stress (CONTRACTION)")

    ax[3].step(pos.index, pos.values, where="post", lw=0.9, color="tab:blue", label="Posizione")
    ax[3].set_ylabel("Posizione (-1/0/+1)"); ax[3].set_yticks([-1, 0, 1])
    ax[3].legend(loc="lower left", fontsize=8); ax[3].grid(alpha=0.3)

    worst = core[stress]
    ov_in_stress = overlay.reindex(worst.index)
    ax[4].bar(range(len(worst)), ov_in_stress.values, color=np.where(ov_in_stress.values >= 0, "tab:green", "tab:red"))
    ax[4].axhline(0, color="k", lw=0.6)
    ax[4].set_title("Crisi-alpha: ritorno overlay nei mesi peggiori del core")
    ax[4].set_ylabel("Overlay excess"); ax[4].grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(CHART_PATH, dpi=120); plt.close(fig)


def _write_report(res, res_sub, core, combined, overlay, cfg_hash, n_trials, dsr, implied_core):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    def row(label, r, c):
        return (f"| {label} | {annualized_return(c,12):+.1%} | -{r['combined_maxdd']:.1%} | "
                f"{r['combined_calmar']:.2f} | {r['overlay_cum']:+.1%} | {r['corr_full']:+.2f} | {r['corr_stress']:+.2f} |")
    lines = [
        "# Sprint S — Trend/Regime Equity-Index Overlay (convessita difensiva)", "",
        f"**Data run:** {today}  ", f"**Config hash:** `{cfg_hash}`  ",
        f"**Trial cumulativi (DSR):** {n_trials}  ",
        f"**Finestra:** {overlay.index[0].strftime('%Y-%m')} -> {overlay.index[-1].strftime('%Y-%m')} ({len(overlay)} mesi)  ",
        "", "**BACKTEST-ONLY** — nessun broker/conto/denaro. Spec: "
        "`docs/superpowers/specs/2026-06-14-sprint-s-trend-overlay-design.md`.", "",
        "## Criteri ex-ante (overlay-su-core, OVERLAY_FRAC=0.15)", "",
        f"- **(a) MaxDD(combined) < MaxDD(core):** -{res['combined_maxdd']:.1%} vs -{res['core_maxdd']:.1%} -> **{'PASS' if res['criterio_a'] else 'FAIL'}**",
        f"- **(b) Calmar(combined) > Calmar(core):** {res['combined_calmar']:.2f} vs {res['core_calmar']:.2f} -> **{'PASS' if res['criterio_b'] else 'FAIL'}**",
        f"- **(c) Overlay net >= 0:** {res['overlay_cum']:+.1%} -> **{'PASS' if res['criterio_c'] else 'FAIL'}**",
        f"- **(d) corr <= 0 (full {res['corr_full']:+.2f}, stress {res['corr_stress']:+.2f}):** -> **{'PASS' if res['criterio_d'] else 'FAIL'}**",
        "", f"**VERDETTO (full): {res['verdict']}**  |  **sub-2011: {res_sub['verdict']}**",
        f"  -  DSR overlay (n_trials={n_trials}): {dsr:.3f}", "",
        f"Nozionale di core implicito per **1 MES** a OVERLAY_FRAC=0.15: **EUR {implied_core:,.0f}** "
        "(sotto questa taglia anche 1 contratto pesa piu del 15% -> granularita sfavorevole).", "",
        "## Metriche", "",
        "| | CAGR | MaxDD | Calmar | overlay cum | corr full | corr stress |",
        "|---|---|---|---|---|---|---|",
        row("Combined full", res, combined),
        row("Combined 2011+", res_sub, combined[combined.index >= SUBPERIOD_START]),
        f"| Core full | {annualized_return(core,12):+.1%} | -{res['core_maxdd']:.1%} | {res['core_calmar']:.2f} | — | — | — |",
        "", "![chart](../backtest/backtest_sprint_s.png)", "",
        "*Script: `scripts/backtest_sprint_s.py` — spec congelata, un solo run OOS. BACKTEST-ONLY.*",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
