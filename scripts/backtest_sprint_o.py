#!/usr/bin/env python3
"""Sprint O — Momentum 12-1 cross-sectional su universo S&P 500 point-in-time.

VALIDATION GATE della pipeline PIT (constituent JL IC=B + TR.TotalReturn1D),
non una strategia finale. Piano e spec congelata:
docs/superpowers/plans/2026-06-12-sprint-o-momentum-pit.md

Ipotesi ex-ante (dichiarata PRIMA del run OOS):
  "Il momentum 12-1 long-only top-decile sull'universo S&P 500 point-in-time,
  con costi 10bps ed esecuzione t+1, consegna il premio noto in letteratura:
  CAGR entro 2pt da SPY buy&hold con drawdown non peggiore."

Criteri di successo DICHIARATI EX-ANTE (valutati sull'UNICO run OOS):
  (a) CAGR strategia >= CAGR SPY - 2 punti percentuali
  (b) MaxDD strategia <= MaxDD SPY (magnitudine non più profonda)
Gate dati (PRECONDIZIONE, non criterio): coverage mensile >= 90% per OGNI
mese. Se il gate fallisce il run è INVALIDO (problema pipeline): nessun
verdetto, né in positivo né in negativo.

Spec congelata: segnale 12-1 (252 gg lookback, skip 21, min 200 obs),
top decile ceil(10%) equal-weight, min 20 nomi (altrimenti cash, flaggato),
ribilancio mensile, esecuzione t+1 (primo giorno holding non accreditato),
costi 10bps one-way sul nozionale scambiato. NESSUNA GRIGLIA IS: zero
parametri liberi, un solo trial nel registry.

Run (UNA VOLTA, dopo la pre-run review):
    uv run python scripts/backtest_sprint_o.py
Output:
    docs/backtest/backtest_sprint_o.png
    docs/reports/sprint_o_results.md
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from jeanclaude.backtest.cross_sectional import (  # noqa: E402
    LOOKBACK_DAYS,
    MIN_OBS,
    SKIP_DAYS,
    align_to_calendar,
    simulate_monthly,
)
from jeanclaude.backtest.metrics import (  # noqa: E402
    annualized_return,
    deflated_sharpe_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from jeanclaude.data.constituents import ConstituentHistory  # noqa: E402
from jeanclaude.data.ingestion.refinitiv import RefinitivSource  # noqa: E402
from jeanclaude.data.storage.parquet_store import ParquetStore  # noqa: E402
from jeanclaude.research import ExperimentConfig, TrialRegistry  # noqa: E402

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Config CONGELATA (hash nel registry — modificarla = nuovo esperimento)
# ---------------------------------------------------------------------------

BASKETS_START = "2003-01"
FIRST_FORMATION = "2003-12-31"
DECILE = 0.10
MIN_NAMES = 20
TC_BPS = 10.0
COVERAGE_GATE = 0.90
CAGR_TOLERANCE = 0.02  # criterio (a): CAGR >= CAGR_SPY - 2pt
MAX_MISSING_TRADING_DAYS = 10  # giorni di borsa senza alcun dato TR tollerati

DATA_DIR = Path(__file__).parent.parent / "data"
RETURNS_PATH = DATA_DIR / "constituents" / "total_returns_pit.parquet"
MISSING_SIDECAR = DATA_DIR / "constituents" / "total_returns_pit_missing.json"
REGISTRY_PATH = DATA_DIR / "research" / "trials.jsonl"
REPORT_PATH = Path(__file__).parent.parent / "docs" / "reports" / "sprint_o_results.md"
CHART_PATH = Path(__file__).parent.parent / "docs" / "backtest" / "backtest_sprint_o.png"

CONFIG = ExperimentConfig(
    name="sprint_o_momentum_pit",
    universe=("0#.SPX point-in-time (JL change-log IC=B, validato 2026-06-12)",),
    equity_rics=(),
    is_start="",          # NESSUNA finestra IS: zero parametri liberi
    is_end="",
    oos_start="2004-01",
    oos_end="LAST_COMPLETE_MONTH",
    rebalance_freq="M",
    tc_bps=TC_BPS,
    execution_lag=1,
    damp_grid=(),
    mom_grid_balanced=(),
    mom_grid_aggressive=(),
    price_field="TR.TotalReturn1D",
    price_adjustments=("total_return",),
    notes=(
        "Momentum 12-1 cross-sectional long-only: lookback 252 gg, skip 21, "
        "min_obs 200; top decile ceil(10%) equal-weight, min 20 nomi (else cash); "
        "esecuzione t+1 (primo giorno holding non accreditato); costi 10bps "
        "one-way su |dw| vs pesi driftati. Gate dati: coverage mensile >=90% "
        "ogni mese. Criteri ex-ante: (a) CAGR >= CAGR_SPY - 2pt; "
        "(b) MaxDD <= MaxDD_SPY. Benchmark SPY B&H Yahoo total return. "
        "VALIDATION GATE pipeline PIT, non strategia finale."
    ),
)


def _load_baskets() -> dict[pd.Timestamp, frozenset[str]]:
    src = RefinitivSource(session_type="platform")
    store = ParquetStore(DATA_DIR)
    ch = ConstituentHistory(source=src, store=store)
    today = pd.Timestamp.today()
    return ch.monthly_baskets(BASKETS_START, today.strftime("%Y-%m"))


def _spy_daily(start: str, end: str) -> pd.Series:
    """Ritorni giornalieri SPY (Yahoo auto_adjust = total return).

    Doppio uso: calendario di borsa NYSE autorevole (pre-run review
    2026-06-12: il parquet TR ha righe fantasma sui festivi e 4 giorni di
    borsa mancanti) e benchmark B&H.
    """
    px = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if px is None or px.empty:
        raise RuntimeError("Download SPY fallito — calendario/benchmark non disponibili")
    close = px["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance MultiIndex
        close = close.iloc[:, 0]
    daily = close.pct_change().dropna()
    daily.index = pd.to_datetime(daily.index).tz_localize(None).normalize()
    return daily


def _spy_monthly(spy_daily: pd.Series, holding_months: pd.DatetimeIndex) -> pd.Series:
    """SPY B&H mensilizzato sugli stessi mesi di holding (mese pieno: B&H)."""
    monthly = (1 + spy_daily).resample("ME").prod() - 1
    monthly = monthly.reindex(holding_months)
    if monthly.isna().any():
        missing = [str(d.date()) for d in monthly.index[monthly.isna()]]
        raise RuntimeError(f"SPY: mesi mancanti nel benchmark: {missing}")
    return monthly


def main() -> None:
    print("=" * 72)
    print("SPRINT O — Momentum 12-1 PIT (validation gate)")
    print("=" * 72)
    cfg_hash = CONFIG.config_hash()
    print(f"Config hash: {cfg_hash}")

    # ----- Guardia anti doppio run (spec: UN SOLO run OOS) -----
    registry = TrialRegistry(REGISTRY_PATH)
    already_run = any(
        r.get("config_hash") == cfg_hash and r.get("window") == "OOS"
        for r in registry._rows()
    )
    if already_run and os.environ.get("JC_FORCE_RERUN") != "1":
        raise SystemExit(
            f"Config {cfg_hash} ha GIÀ un run OOS nel registry — la spec prevede "
            "un solo run. Un nuovo esperimento richiede una nuova config (nuovo "
            "hash). Override esplicito: JC_FORCE_RERUN=1."
        )

    if not RETURNS_PATH.exists():
        raise SystemExit(
            f"Manca {RETURNS_PATH} — eseguire prima scripts/fetch_pit_total_returns.py"
        )
    daily_returns = pd.read_parquet(RETURNS_PATH)
    daily_returns.index = pd.to_datetime(daily_returns.index)
    daily_returns = daily_returns.sort_index()
    print(f"Total return (raw): {daily_returns.shape[0]} giorni x "
          f"{daily_returns.shape[1]} RIC")

    baskets_all = _load_baskets()

    # Formations: dalla prima con 12 mesi di dati all'ultima con mese di
    # holding COMPLETO. Convenzione conservativa: l'ultimo mese completo è
    # SEMPRE il mese precedente a oggi (un run l'ultimo giorno del mese non
    # tratta il mese corrente come chiuso — dati intraday parziali).
    today = pd.Timestamp.today()
    last_complete_holding = (today - pd.offsets.MonthEnd(1)).normalize()
    last_formation = last_complete_holding - pd.offsets.MonthEnd(1)
    first_formation = pd.Timestamp(FIRST_FORMATION)
    baskets = {
        t: members
        for t, members in baskets_all.items()
        if first_formation <= t <= last_formation
    }
    print(f"Formations: {len(baskets)} mesi "
          f"({min(baskets).date()} → {max(baskets).date()})")

    # ----- Calendario di borsa autorevole (SPY) + allineamento TR -----
    spy_daily = _spy_daily(
        start="2002-09-01",
        end=(last_complete_holding + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d"),
    )
    calendar = spy_daily.index[spy_daily.index <= last_complete_holding]
    daily_returns, extra_rows, missing_days = align_to_calendar(daily_returns, calendar)
    print(f"Allineamento calendario NYSE: {len(extra_rows)} righe fantasma rimosse, "
          f"{len(missing_days)} giorni di borsa senza dati TR (NaN)")
    if len(missing_days):
        print("  Giorni mancanti:", ", ".join(str(d.date()) for d in missing_days[:10]))
    if len(missing_days) > MAX_MISSING_TRADING_DAYS:
        raise SystemExit(
            f"RUN INVALIDO: {len(missing_days)} giorni di borsa senza dati TR "
            f"(max tollerato {MAX_MISSING_TRADING_DAYS}) — fetch incompleto?"
        )

    # ----- Precondizioni di completezza e freschezza (pre-run review) -----
    required = sorted(set().union(*baskets.values()))
    no_data_known: set[str] = set()
    if MISSING_SIDECAR.exists():
        no_data_known = set(json.loads(MISSING_SIDECAR.read_text()))
    absent = [r for r in required if r not in daily_returns.columns
              and r not in no_data_known]
    if absent:
        raise SystemExit(
            f"RUN INVALIDO: {len(absent)} RIC dei panieri assenti dal parquet e "
            f"non dichiarati senza-dati dal fetch (primi 10: {absent[:10]}) — "
            "fetch incompleto: rieseguire scripts/fetch_pit_total_returns.py"
        )
    if no_data_known:
        n_required_missing = len([r for r in required if r in no_data_known])
        print(f"RIC dichiarati senza dati dal fetch: {len(no_data_known)} "
              f"(di cui {n_required_missing} nei panieri della finestra)")
    if daily_returns.index.max() < calendar.max():
        raise SystemExit(
            f"RUN INVALIDO: dati TR stantii — ultimo giorno {daily_returns.index.max().date()} "
            f"< ultimo giorno di borsa del mese di holding finale {calendar.max().date()}"
        )

    # ----- Conteggio trial cumulativo onesto -----
    n_trials_prior = registry.n_trials()
    n_trials = n_trials_prior + 1
    print(f"Trial cumulativi nel registry: {n_trials_prior} (+1 questo run = {n_trials})")

    # ----- UNICO RUN OOS -----
    result = simulate_monthly(
        daily_returns, baskets,
        decile=DECILE, min_names=MIN_NAMES, tc_bps=TC_BPS,
        lookback=LOOKBACK_DAYS, skip=SKIP_DAYS, min_obs=MIN_OBS,
    )
    rets = result.monthly_returns
    diags = result.diagnostics
    print(f"Mesi di holding simulati: {len(rets)} "
          f"({rets.index[0].date()} → {rets.index[-1].date()})")

    # ----- Nessuna formation saltata (bypasserebbe il gate coverage) -----
    expected_months = pd.date_range(
        first_formation + pd.offsets.MonthEnd(1), last_complete_holding, freq="ME"
    )
    if not rets.index.equals(expected_months):
        skipped = expected_months.difference(rets.index)
        raise SystemExit(
            f"RUN INVALIDO: {len(skipped)} mesi di holding saltati dalla "
            f"simulazione ({[str(d.date()) for d in skipped[:5]]}) — "
            "buco nei dati che il gate coverage non può vedere"
        )

    # ----- GATE DATI (precondizione) -----
    min_cov = float(diags["coverage"].min())
    worst_formation = diags["coverage"].idxmin()
    worst_month = worst_formation + pd.offsets.MonthEnd(1)  # mese di HOLDING
    gate_pass = min_cov >= COVERAGE_GATE
    print(f"\nGate dati: coverage minima {min_cov:.1%} "
          f"(mese di holding peggiore: {worst_month.strftime('%Y-%m')}) — "
          f"soglia {COVERAGE_GATE:.0%} → {'PASS' if gate_pass else 'FAIL'}")
    if not gate_pass:
        _write_report(rets, None, diags, cfg_hash, n_trials, gate_pass, min_cov,
                      worst_month, None)
        raise SystemExit(
            "RUN INVALIDO: gate dati fallito — problema pipeline, nessun "
            "verdetto sui criteri. Vedi report."
        )

    # ----- Benchmark e metriche -----
    spy = _spy_monthly(spy_daily, rets.index)

    metrics = {}
    for label, series in (("strategy", rets), ("spy", spy)):
        metrics[label] = {
            "cagr": annualized_return(series, periods_per_year=12),
            "sharpe": sharpe_ratio(series, periods_per_year=12),
            "sortino": sortino_ratio(series, periods_per_year=12),
            "maxdd": max_drawdown(series),
        }

    sk = float(stats.skew(rets))
    ku = float(stats.kurtosis(rets, fisher=False))
    dsr = deflated_sharpe_ratio(
        metrics["strategy"]["sharpe"], n_trials=n_trials, obs=len(rets),
        skewness=sk, kurtosis=ku, periods_per_year=12,
    )

    # ----- Verdetto ex-ante -----
    crit_a = metrics["strategy"]["cagr"] >= metrics["spy"]["cagr"] - CAGR_TOLERANCE
    crit_b = metrics["strategy"]["maxdd"] <= metrics["spy"]["maxdd"]

    print("\n--- Risultati OOS ---")
    for label in ("strategy", "spy"):
        m = metrics[label]
        print(f"{label:>9}: CAGR {m['cagr']:+.1%}  Sharpe {m['sharpe']:.2f}  "
              f"Sortino {m['sortino']:.2f}  MaxDD -{m['maxdd']:.1%}")
    print(f"DSR (n_trials={n_trials}, mensile): {dsr:.3f}")
    print(f"\nCriterio (a) CAGR >= SPY-2pt: {'PASS' if crit_a else 'FAIL'}")
    print(f"Criterio (b) MaxDD <= SPY:    {'PASS' if crit_b else 'FAIL'}")
    verdict = "PASS" if (crit_a and crit_b) else "FAIL"
    print(f"\nVERDETTO VALIDATION GATE: {verdict}")

    # ----- Registry -----
    registry.record(
        config_hash=cfg_hash,
        experiment=CONFIG.name,
        params={"decile": DECILE, "min_names": MIN_NAMES, "tc_bps": TC_BPS,
                "lookback": LOOKBACK_DAYS, "skip": SKIP_DAYS, "min_obs": MIN_OBS,
                "coverage_gate": COVERAGE_GATE, "cagr_tolerance": CAGR_TOLERANCE},
        window="OOS",
        metrics={
            "cagr": metrics["strategy"]["cagr"],
            "sharpe": metrics["strategy"]["sharpe"],
            "maxdd": metrics["strategy"]["maxdd"],
            "dsr": dsr,
            "spy_cagr": metrics["spy"]["cagr"],
            "spy_maxdd": metrics["spy"]["maxdd"],
            "min_coverage": min_cov,
            "criterio_a": bool(crit_a),
            "criterio_b": bool(crit_b),
            "oos_start_eff": str(rets.index[0].date()),
            "oos_end_eff": str(rets.index[-1].date()),
            "n_months": len(rets),
        },
    )

    _write_chart(rets, spy, diags)
    _write_report(rets, spy, diags, cfg_hash, n_trials, gate_pass, min_cov,
                  worst_month, {"metrics": metrics, "dsr": dsr, "skew": sk,
                                "kurt": ku, "crit_a": crit_a, "crit_b": crit_b})
    print(f"\nReport: {REPORT_PATH}")
    print(f"Chart:  {CHART_PATH}")


def _write_chart(rets: pd.Series, spy: pd.Series, diags: pd.DataFrame) -> None:
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    nav_s = (1 + rets).cumprod()
    nav_b = (1 + spy).cumprod()
    dd_s = nav_s / nav_s.cummax() - 1
    dd_b = nav_b / nav_b.cummax() - 1

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    axes[0].plot(nav_s.index, nav_s, label="Momentum 12-1 PIT (net 10bps)", lw=1.6)
    axes[0].plot(nav_b.index, nav_b, label="SPY B&H (TR)", lw=1.2, alpha=0.8)
    axes[0].set_yscale("log")
    axes[0].set_title("Sprint O — Momentum 12-1 su S&P 500 point-in-time (OOS, mensile)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(dd_s.index, dd_s, 0, alpha=0.5, label="Strategia")
    axes[1].plot(dd_b.index, dd_b, lw=1, alpha=0.8, label="SPY", color="tab:orange")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(diags.index, diags["coverage"], lw=1, label="Coverage paniere")
    axes[2].axhline(COVERAGE_GATE, color="red", ls="--", lw=0.8, label="Gate 90%")
    ax2b = axes[2].twinx()
    ax2b.plot(diags.index, diags["n_eligible"], lw=0.8, color="tab:green",
              alpha=0.6, label="Eleggibili")
    axes[2].set_ylabel("Coverage")
    ax2b.set_ylabel("N eleggibili")
    axes[2].legend(loc="lower left")
    ax2b.legend(loc="lower right")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=120)
    plt.close(fig)


def _write_report(
    rets: pd.Series,
    spy: pd.Series | None,
    diags: pd.DataFrame,
    cfg_hash: str,
    n_trials: int,
    gate_pass: bool,
    min_cov: float,
    worst_month: pd.Timestamp,
    res: dict | None,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    avg_turnover_y = float(diags["turnover"].mean()) * 12
    cash_months = int((diags["n_held"] == 0).sum())

    lines = [
        "# Sprint O — Momentum 12-1 Cross-Sectional su Universo Point-in-Time",
        "",
        f"**Data run:** {today}  ",
        f"**Config hash:** `{cfg_hash}`  ",
        f"**Trial cumulativi (DSR):** {n_trials}  ",
        f"**Finestra OOS:** {rets.index[0].strftime('%Y-%m')} → "
        f"{rets.index[-1].strftime('%Y-%m')} ({len(rets)} mesi)  ",
        "",
        "**Natura dell'esperimento:** validation gate della pipeline PIT "
        "(constituent JL IC=B + TR.TotalReturn1D), zero parametri liberi, "
        "spec e criteri congelati ex-ante nel piano "
        "`docs/superpowers/plans/2026-06-12-sprint-o-momentum-pit.md`.",
        "",
        "---",
        "",
        "## Gate dati (precondizione)",
        "",
        f"Coverage minima mensile: **{min_cov:.1%}** "
        f"(mese peggiore: {worst_month.strftime('%Y-%m')}) — soglia 90% → "
        f"**{'PASS' if gate_pass else 'FAIL — RUN INVALIDO'}**",
        "",
    ]

    if not gate_pass or res is None:
        lines += [
            "Il gate dati è fallito: il run è INVALIDO e nessun verdetto viene",
            "emesso sui criteri (a)/(b). Indagare la pipeline (copertura prezzi",
            "dei membri) prima di rieseguire — la riesecuzione post-fix è un",
            "nuovo trial nel registry.",
        ]
    else:
        m = res["metrics"]
        lines += [
            "## Risultati OOS (mensile, net 10bps)",
            "",
            "| | CAGR | Sharpe | Sortino | MaxDD |",
            "|---|---|---|---|---|",
            f"| Momentum 12-1 PIT | {m['strategy']['cagr']:+.1%} | "
            f"{m['strategy']['sharpe']:.2f} | {m['strategy']['sortino']:.2f} | "
            f"-{m['strategy']['maxdd']:.1%} |",
            f"| SPY B&H (TR) | {m['spy']['cagr']:+.1%} | {m['spy']['sharpe']:.2f} | "
            f"{m['spy']['sortino']:.2f} | -{m['spy']['maxdd']:.1%} |",
            "",
            f"**DSR** (n_trials={n_trials}, obs={len(rets)} mesi, "
            f"skew={res['skew']:.2f}, kurt={res['kurt']:.2f}): **{res['dsr']:.3f}**",
            "",
            "## Verdetto sui criteri ex-ante",
            "",
            f"- **(a) CAGR ≥ CAGR_SPY − 2pt:** "
            f"{m['strategy']['cagr']:+.1%} vs {m['spy']['cagr'] - 0.02:+.1%} → "
            f"**{'PASS' if res['crit_a'] else 'FAIL'}**",
            f"- **(b) MaxDD ≤ MaxDD_SPY:** "
            f"-{m['strategy']['maxdd']:.1%} vs -{m['spy']['maxdd']:.1%} → "
            f"**{'PASS' if res['crit_b'] else 'FAIL'}**",
            "",
            f"**VALIDATION GATE: {'PASS' if (res['crit_a'] and res['crit_b']) else 'FAIL'}**",
            "",
            "Interpretazione dichiarata ex-ante: PASS ⇒ pipeline PIT validata, "
            "si procede con FinBERT cross-sectional e doppi sort. FAIL con gate "
            "dati OK ⇒ indagare la pipeline prima di concludere sul segnale.",
        ]

    lines += [
        "",
        "## Diagnostica",
        "",
        f"- Membri medi del paniere: {diags['n_members'].mean():.0f}  ",
        f"- Eleggibili medi (≥200 obs): {diags['n_eligible'].mean():.0f}  ",
        f"- Nomi detenuti medi: {diags['n_held'].mean():.0f}  ",
        f"- Mesi in cash (decile < 20 nomi): {cash_months}  ",
        f"- Turnover medio annualizzato (one-way): {avg_turnover_y:.0%}  ",
        f"- Costo medio mensile: {diags['cost'].mean() * 1e4:.1f} bps  ",
        "",
        "![chart](../backtest/backtest_sprint_o.png)",
        "",
        "*Script: `scripts/backtest_sprint_o.py` — spec congelata, un solo run OOS.*",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
