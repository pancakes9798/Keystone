#!/usr/bin/env python3
"""Sprint T — Mining formulaico returns-only disciplinato (BACKTEST-ONLY).

Spec congelata: docs/superpowers/specs/2026-06-15-sprint-t-disciplined-mining-design.md

Spazio dichiarato N config (3 tier: A=3, B=48, C=3 = 54 totali).
Addestramento SOLO su TRAIN (2004-2015). Holdout 2016-LAST BLOCCATO: aperto
UNA VOLTA sul config selezionata dal TRAIN; usato per diagnostica scatter
post-verdetto ma MAI per selezione.

Criteri kill (tutti e tre devono passare per CANDIDATO):
  - PBO (CSCV, 16 partizioni) < 0.5
  - DSR (n_trials=N interi, non len(keep)) > 0.95
  - Holdout Sharpe > 60/40 AND RankIC/anno > 0 per ogni anno

Run (UNA VOLTA, dopo pre-run review):
    .venv/bin/python scripts/backtest_sprint_t.py
Output:
    docs/backtest/backtest_sprint_t.png
    docs/reports/sprint_t_results.md
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

from jeanclaude.backtest.cross_sectional import (  # noqa: E402
    align_to_calendar,
    simulate_monthly,
)
from jeanclaude.backtest.ic import rank_ic, rank_ic_by_year  # noqa: E402
from jeanclaude.backtest.metrics import (  # noqa: E402
    annualized_return,
    deflated_sharpe_ratio,
    max_drawdown,
    sharpe_ratio,
)
from jeanclaude.backtest.pbo import probability_of_backtest_overfitting  # noqa: E402
from jeanclaude.data.constituents import ConstituentHistory  # noqa: E402
from jeanclaude.data.ingestion.refinitiv import RefinitivSource  # noqa: E402
from jeanclaude.data.storage.parquet_store import ParquetStore  # noqa: E402
from jeanclaude.research import ExperimentConfig, TrialRegistry  # noqa: E402
from jeanclaude.signals.formulaic.generator import (  # noqa: E402
    StrategyConfig,
    build_search_space,
)

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Costanti e config CONGELATE
# ---------------------------------------------------------------------------

BASKETS_START = "2003-01"
TRAIN_END = pd.Timestamp("2015-12-31")
HOLDOUT_START = pd.Timestamp("2016-01-01")
DECILE = 0.10
MIN_NAMES = 20
TC_BPS = 10.0
PBO_PARTITIONS = 16
PBO_MAX = 0.5
DSR_MIN = 0.95

DATA_DIR = Path(__file__).parent.parent / "data"
RETURNS_PATH = DATA_DIR / "constituents" / "total_returns_pit.parquet"
REGISTRY_PATH = DATA_DIR / "research" / "trials.jsonl"
REPORT_PATH = Path(__file__).parent.parent / "docs" / "reports" / "sprint_t_results.md"
CHART_PATH = Path(__file__).parent.parent / "docs" / "backtest" / "backtest_sprint_t.png"

CONFIG = ExperimentConfig(
    name="sprint_t_formulaic_mining",
    oos_start="2016-01",
    oos_end="LAST_COMPLETE_MONTH",
    rebalance_freq="M",
    tc_bps=TC_BPS,
    execution_lag=1,
    price_field="TR.TotalReturn1D",
    price_adjustments=("total_return",),
    universe=("0#.SPX PIT (JL IC=B)",),
    equity_rics=(),
    is_start="2004-01",
    is_end="2015-12",
    damp_grid=(),
    mom_grid_balanced=(),
    mom_grid_aggressive=(),
    notes=(
        "Mining formulaico returns-only disciplinato: spazio dichiarato N config "
        "(3 tier), PBO(CSCV)+DSR(N)+holdout bloccato 2016-2026. "
        "Kill: PBO<0.5 & DSR>0.95 & holdout Sharpe>60/40 & RankIC/anno>0. "
        "BACKTEST-ONLY."
    ),
)


# ---------------------------------------------------------------------------
# Boilerplate PIT-loading (verbatim da Sprint O)
# ---------------------------------------------------------------------------

def _load_baskets() -> dict[pd.Timestamp, frozenset[str]]:
    src = RefinitivSource(session_type="platform")
    store = ParquetStore(DATA_DIR)
    ch = ConstituentHistory(source=src, store=store)
    today = pd.Timestamp.today()
    return ch.monthly_baskets(BASKETS_START, today.strftime("%Y-%m"))


def _spy_daily(start: str, end: str) -> pd.Series:
    """Ritorni giornalieri SPY (Yahoo auto_adjust = total return).

    Doppio uso: calendario di borsa NYSE autorevole e benchmark B&H (quota SPY
    nel 60/40). Verbatim da Sprint O / Sprint S.
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


def _ief_daily(start: str, end: str) -> pd.Series:
    """Ritorni giornalieri IEF (Yahoo auto_adjust = total return). Da Sprint S."""
    px = yf.download("IEF", start=start, end=end, auto_adjust=True, progress=False)
    if px is None or px.empty:
        raise RuntimeError("Download IEF fallito")
    close = px["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    daily = close.pct_change().dropna()
    daily.index = pd.to_datetime(daily.index).tz_localize(None).normalize()
    return daily


# ---------------------------------------------------------------------------
# Helpers forward return per RankIC
# ---------------------------------------------------------------------------

def _fwd_monthly_returns(
    daily_returns: pd.DataFrame, formation: pd.Timestamp
) -> pd.Series:
    """Ritorno composito per-RIC nel mese di holding che segue la formation.

    Usa tutti i giorni del mese di holding (senza esecuzione lag — qui ci interessa
    il ritorno del prezzo, non il P&L della strategia che ha il lag t+1).
    """
    month_end = formation + pd.offsets.MonthEnd(1)
    hold = daily_returns.loc[
        formation + pd.Timedelta(days=1) : month_end
    ]
    if hold.empty:
        return pd.Series(dtype=float)
    return (1.0 + hold).prod(skipna=True) - 1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("SPRINT T — Mining formulaico returns-only (BACKTEST-ONLY)")
    print("=" * 72)
    cfg_hash = CONFIG.config_hash()
    print(f"Config hash: {cfg_hash}")

    # ----- Guardia anti doppio run -----
    registry = TrialRegistry(REGISTRY_PATH)
    already_run = any(
        r.get("config_hash") == cfg_hash and r.get("window") == "OOS"
        for r in registry._rows()
    )
    if already_run and os.environ.get("JC_FORCE_RERUN") != "1":
        raise SystemExit(
            f"Config {cfg_hash} ha GIÀ un run OOS nel registry — la spec prevede "
            "un solo run. Override esplicito: JC_FORCE_RERUN=1."
        )

    if not RETURNS_PATH.exists():
        raise SystemExit(
            f"Manca {RETURNS_PATH} — eseguire prima scripts/fetch_pit_total_returns.py"
        )

    # ----- Caricamento dati -----
    daily_returns = pd.read_parquet(RETURNS_PATH)
    daily_returns.index = pd.to_datetime(daily_returns.index)
    daily_returns = daily_returns.sort_index()
    print(f"Total return (raw): {daily_returns.shape[0]} giorni x {daily_returns.shape[1]} RIC")

    baskets_all = _load_baskets()

    today = pd.Timestamp.today()
    last_complete_holding = (today - pd.offsets.MonthEnd(1)).normalize()
    last_formation = last_complete_holding - pd.offsets.MonthEnd(1)
    print(f"Ultimo mese di holding completo: {last_complete_holding.date()}")

    # ----- Calendario NYSE + allineamento TR (verbatim Sprint O) -----
    end_str = (last_complete_holding + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
    spy_daily = _spy_daily(start="2002-09-01", end=end_str)
    ief_daily = _ief_daily(start="2002-09-01", end=end_str)
    calendar = spy_daily.index[spy_daily.index <= last_complete_holding]
    daily_returns, extra_rows, missing_days = align_to_calendar(daily_returns, calendar)
    print(
        f"Allineamento calendario NYSE: {len(extra_rows)} righe fantasma rimosse, "
        f"{len(missing_days)} giorni senza dati TR (NaN)"
    )

    # ----- Split FISICO baskets TRAIN vs HOLDOUT -----
    # TRAIN: formation <= TRAIN_END → holding <= 2016-01-31
    # HOLDOUT: formation >= HOLDOUT_START e holding <= last_complete_holding
    #   (la formation di gennaio 2016 ha holding in febbraio 2016 → si forma
    #    con dati fino al 2015-12-31 ma fa parte del HOLDOUT dal lato holding).
    # Convenzione: baskets_hold = formations in [HOLDOUT_START, last_formation].
    # Il TRAIN INCLUDE tutte le formations fino a TRAIN_END (31-12-2015).
    baskets_train = {
        t: m for t, m in baskets_all.items() if t <= TRAIN_END
    }
    baskets_hold = {
        t: m for t, m in baskets_all.items()
        if t >= HOLDOUT_START and t <= last_formation
    }
    print(
        f"Baskets TRAIN: {len(baskets_train)} formations "
        f"({min(baskets_train).date()} → {max(baskets_train).date()})"
    )
    print(
        f"Baskets HOLDOUT: {len(baskets_hold)} formations "
        f"({min(baskets_hold).date()} → {max(baskets_hold).date()})"
    )

    # ----- Spazio di ricerca (dichiarato, N contati prima della ricerca) -----
    space: list[StrategyConfig] = build_search_space()
    N = len(space)
    print(f"\nSpazio di ricerca: N = {N} configurazioni (Tier A+B+C)")

    # ----- Search loop su TRAIN -----
    print("\n--- Simulazioni TRAIN (solo su baskets_train) ---")
    monthly_by_name: dict[str, pd.Series] = {}
    for i, cfg in enumerate(space):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{N}] {cfg.name}")
        try:
            res = simulate_monthly(
                daily_returns,
                baskets_train,
                signal_fn=cfg.signal_fn,
                decile=DECILE,
                min_names=MIN_NAMES,
                tc_bps=TC_BPS,
            )
            if not res.monthly_returns.empty:
                monthly_by_name[cfg.name] = res.monthly_returns
        except Exception as exc:  # noqa: BLE001
            logging.warning("Config %s fallita: %s", cfg.name, exc)

    print(f"Config con ritorni TRAIN non vuoti: {len(monthly_by_name)}/{N}")
    returns_matrix = pd.DataFrame(monthly_by_name).dropna(how="all")
    print(f"Matrice ritorni TRAIN: {returns_matrix.shape} (periodi x config)")

    # ----- Corr-pruning -----
    corr = returns_matrix.corr()
    keep: list[str] = []
    for col in returns_matrix.columns:
        if all(abs(corr.loc[col, k]) <= 0.7 for k in keep):
            keep.append(col)
    pruned = returns_matrix[keep]
    print(f"Corr-pruning (|corr|<=0.7): {len(keep)} config kept da {len(returns_matrix.columns)}")

    # ----- PBO (CSCV su TRAIN pruned) -----
    print(f"\nCalcolo PBO (CSCV, {PBO_PARTITIONS} partizioni) su {len(keep)} config...")
    pbo = probability_of_backtest_overfitting(pruned, n_partitions=PBO_PARTITIONS)
    print(f"PBO: {pbo.pbo:.3f} ({pbo.n_combinations} combinazioni)")

    # ----- Selezione migliore IS + DSR -----
    sharpes = pruned.apply(lambda s: sharpe_ratio(s.dropna(), 12))
    best_name = str(sharpes.idxmax())
    best_is = pruned[best_name].dropna()
    sk = float(stats.skew(best_is))
    ku = float(stats.kurtosis(best_is, fisher=False))
    # n_trials = N INTERO (tutto lo spazio provato), NON len(keep)
    dsr = deflated_sharpe_ratio(
        sharpe_ratio(best_is, 12),
        n_trials=N,
        obs=len(best_is),
        skewness=sk,
        kurtosis=ku,
        periods_per_year=12,
    )
    print(f"\nMigliore IS: {best_name}")
    print(f"  Sharpe IS: {float(sharpes.max()):.3f}")
    print(f"  DSR (n_trials={N}): {dsr:.3f}")

    # ----- Holdout: ONE-SHOT sulla config selezionata -----
    print("\n--- Holdout (ONE-SHOT, config selezionata) ---")
    best_cfg: StrategyConfig = next(c for c in space if c.name == best_name)
    hold_res = simulate_monthly(
        daily_returns,
        baskets_hold,
        signal_fn=best_cfg.signal_fn,
        decile=DECILE,
        min_names=MIN_NAMES,
        tc_bps=TC_BPS,
    )
    hold_rets = hold_res.monthly_returns
    if hold_rets.empty:
        raise RuntimeError("Holdout vuoto — la config selezionata produce 0 mesi su baskets_hold.")
    hold_sharpe = sharpe_ratio(hold_rets, 12)
    print(f"Mesi holdout: {len(hold_rets)} ({hold_rets.index[0].date()} → {hold_rets.index[-1].date()})")
    print(f"Sharpe holdout: {hold_sharpe:.3f}")

    # ----- Benchmark 60/40 sul periodo di holdout -----
    spy_m = (1 + spy_daily).resample("ME").prod() - 1
    ief_m = (1 + ief_daily).resample("ME").prod() - 1
    core_6040 = 0.6 * spy_m + 0.4 * ief_m
    core_hold = core_6040.reindex(hold_rets.index).dropna()
    sharpe_6040 = sharpe_ratio(core_hold, 12) if not core_hold.empty else float("nan")
    print(f"Sharpe 60/40 (holdout): {sharpe_6040:.3f}")

    # ----- RankIC per anno sul holdout -----
    ic_monthly_rows: dict[pd.Timestamp, float] = {}
    for t in sorted(baskets_hold):
        month_end = t + pd.offsets.MonthEnd(1)
        if month_end not in hold_rets.index:
            continue
        try:
            sig_t = best_cfg.signal_fn(daily_returns, t)
            fwd_t = _fwd_monthly_returns(daily_returns, t)
            ic_val = rank_ic(sig_t, fwd_t)
            ic_monthly_rows[month_end] = ic_val
        except Exception as exc:  # noqa: BLE001
            logging.warning("RankIC fallito per formation %s: %s", t.date(), exc)

    ic_monthly = pd.Series(ic_monthly_rows, dtype=float).sort_index()
    ic_by_year = rank_ic_by_year(ic_monthly.dropna())
    print(f"\nRankIC per anno (holdout):")
    for yr, val in ic_by_year.items():
        print(f"  {yr}: {val:+.3f}")

    # ----- Criteri Kill -----
    crit_pbo = bool(pbo.pbo < PBO_MAX)
    crit_dsr = bool(dsr > DSR_MIN)
    # ic_by_year vuoto (tutti i calcoli IC falliti) → NON possiamo verificare il decay → FAIL.
    ic_all_positive = (not ic_by_year.empty) and bool((ic_by_year > 0).all())
    crit_oos = bool(hold_sharpe > sharpe_6040) and ic_all_positive
    verdict = "CANDIDATO" if (crit_pbo and crit_dsr and crit_oos) else "KILL"

    print(f"\n--- Verdetto ---")
    print(f"PBO < {PBO_MAX}: {pbo.pbo:.3f} → {'PASS' if crit_pbo else 'FAIL'}")
    print(f"DSR > {DSR_MIN}: {dsr:.3f} → {'PASS' if crit_dsr else 'FAIL'}")
    print(f"Holdout Sharpe > 60/40: {hold_sharpe:.3f} vs {sharpe_6040:.3f} → {'PASS' if bool(hold_sharpe > sharpe_6040) else 'FAIL'}")
    print(f"RankIC/anno > 0 tutti: {(ic_by_year > 0).all()} → {'PASS' if bool((ic_by_year > 0).all()) else 'FAIL'}")
    print(f"\nVERDETTO: {verdict}  (probabile KILL — dichiarato ex-ante)")

    # ----- Sharpe OOS di tutte le config per lo scatter diagnostico -----
    # NOTA: il loop OOS sotto è SOLO per il panel (2) del grafico diagnostico,
    # eseguito POST-verdetto. NON è selezione.
    print("\n--- Calcolo Sharpe OOS per scatter diagnostico (post-verdetto) ---")
    sharpe_oos_dict: dict[str, float] = {}
    for cfg in space:
        try:
            res_h = simulate_monthly(
                daily_returns,
                baskets_hold,
                signal_fn=cfg.signal_fn,
                decile=DECILE,
                min_names=MIN_NAMES,
                tc_bps=TC_BPS,
            )
            if not res_h.monthly_returns.empty:
                sharpe_oos_dict[cfg.name] = sharpe_ratio(res_h.monthly_returns, 12)
        except Exception as exc:  # noqa: BLE001
            logging.warning("OOS scatter: config %s fallita: %s", cfg.name, exc)
    print(f"Config con Sharpe OOS disponibile: {len(sharpe_oos_dict)}/{N}")

    # ----- Registry (un solo trial) -----
    registry.record(
        config_hash=cfg_hash,
        experiment="sprint_t_formulaic_mining",
        window="OOS",
        params={
            "N": N,
            "n_kept": len(keep),
            "decile": DECILE,
            "tc_bps": TC_BPS,
            "pbo_partitions": PBO_PARTITIONS,
        },
        metrics={
            "pbo": pbo.pbo,
            "dsr": dsr,
            "best_name": best_name,
            "best_is_sharpe": float(sharpes.max()),
            "holdout_sharpe": hold_sharpe,
            "sharpe_6040": sharpe_6040,
            "min_year_ic": float(ic_by_year.min()) if not ic_by_year.empty else float("nan"),
            "verdict": verdict,
        },
    )

    _write_chart(
        pbo=pbo,
        sharpes_is=sharpes,
        sharpe_oos_dict=sharpe_oos_dict,
        best_name=best_name,
        best_is=best_is,
        hold_rets=hold_rets,
        core_hold=core_hold,
        ic_by_year=ic_by_year,
        N=N,
    )
    _write_report(
        N=N,
        n_kept=len(keep),
        pbo=pbo.pbo,
        dsr=dsr,
        best_name=best_name,
        best_is_sharpe=float(sharpes.max()),
        hold_sharpe=hold_sharpe,
        sharpe_6040=sharpe_6040,
        ic_by_year=ic_by_year,
        verdict=verdict,
        cfg_hash=cfg_hash,
        hold_rets=hold_rets,
    )
    print(f"\nReport: {REPORT_PATH}")
    print(f"Chart:  {CHART_PATH}")


# ---------------------------------------------------------------------------
# Output: figura 5 pannelli
# ---------------------------------------------------------------------------

def _write_chart(
    *,
    pbo,
    sharpes_is: pd.Series,
    sharpe_oos_dict: dict[str, float],
    best_name: str,
    best_is: pd.Series,
    hold_rets: pd.Series,
    core_hold: pd.Series,
    ic_by_year: pd.Series,
    N: int,
) -> None:
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(13, 22),
                             gridspec_kw={"height_ratios": [2, 2.5, 3, 2, 2]})

    # Panel 1: istogramma logits PBO
    ax = axes[0]
    ax.hist(pbo.logits, bins=30, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="red", lw=1.5, ls="--", label="logit=0 (PBO boundary)")
    ax.set_title(f"Panel 1 — Distribuzione logit PBO (CSCV, {pbo.n_combinations} comb.)\n"
                 f"PBO = {pbo.pbo:.3f}")
    ax.set_xlabel("logit(rank OOS)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: scatter Sharpe IS vs Sharpe OOS (tutte le config)
    ax = axes[1]
    common_names = [n for n in sharpes_is.index if n in sharpe_oos_dict]
    is_vals = [float(sharpes_is[n]) for n in common_names]
    oos_vals = [sharpe_oos_dict[n] for n in common_names]
    colors = ["tab:orange" if n == best_name else "steelblue" for n in common_names]
    ax.scatter(is_vals, oos_vals, c=colors, alpha=0.6, s=30)
    lim_is = [min(is_vals) - 0.1, max(is_vals) + 0.1] if is_vals else [-1, 2]
    ax.plot(lim_is, lim_is, "k--", lw=0.8, alpha=0.5, label="IS=OOS")
    ax.set_xlabel("Sharpe IS (TRAIN 2004-2015)")
    ax.set_ylabel("Sharpe OOS (HOLDOUT 2016-)")
    ax.set_title(
        "Panel 2 — Scatter Sharpe IS vs OOS (TUTTE le config; OOS = diagnostico post-verdetto)\n"
        "Arancione = selezionata"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: NAV selezionata IS + holdout + 60/40 holdout
    ax = axes[2]
    nav_is = (1 + best_is).cumprod()
    nav_hold = (1 + hold_rets).cumprod()
    nav_6040 = (1 + core_hold).cumprod()
    ax.plot(nav_is.index, nav_is, label=f"IS {best_name}", lw=1.4)
    ax.axvline(pd.Timestamp("2016-01-01"), color="gray", ls="--", lw=0.8, label="Train/Holdout split")
    ax.plot(nav_hold.index, nav_hold, label="Holdout (selezionata)", lw=1.6, color="tab:orange")
    ax.plot(nav_6040.index, nav_6040, label="60/40 holdout", lw=1.2, color="tab:green", alpha=0.8)
    ax.set_yscale("log")
    ax.set_title(f"Panel 3 — NAV: IS cumprod | Holdout selezionata | 60/40 holdout\n"
                 f"Holdout Sharpe: {sharpe_ratio(hold_rets, 12):.3f} vs 60/40: {sharpe_ratio(core_hold, 12):.3f}")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 4: RankIC per anno (holdout)
    ax = axes[3]
    colors_ic = ["tab:green" if v >= 0 else "tab:red" for v in ic_by_year.values]
    ax.bar(ic_by_year.index.astype(str), ic_by_year.values, color=colors_ic, alpha=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Panel 4 — RankIC per anno (holdout)")
    ax.set_ylabel("IC medio")
    ax.grid(alpha=0.3)

    # Panel 5: istogramma Sharpe IS di tutte le config con la migliore evidenziata
    ax = axes[4]
    all_is_sharpes = sharpes_is.values.astype(float)
    best_sharpe_val = float(sharpes_is[best_name])
    ax.hist(all_is_sharpes, bins=20, color="steelblue", alpha=0.7, edgecolor="white",
            label=f"N={N} config")
    ax.axvline(best_sharpe_val, color="tab:orange", lw=2.0,
               label=f"Migliore IS: {best_name} ({best_sharpe_val:.3f})")
    ax.set_title(f"Panel 5 — Distribuzione Sharpe IS (TRAIN, N={N})")
    ax.set_xlabel("Sharpe IS (annualizzato, 12 periodi/anno)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output: report markdown
# ---------------------------------------------------------------------------

def _write_report(
    *,
    N: int,
    n_kept: int,
    pbo: float,
    dsr: float,
    best_name: str,
    best_is_sharpe: float,
    hold_sharpe: float,
    sharpe_6040: float,
    ic_by_year: pd.Series,
    verdict: str,
    cfg_hash: str,
    hold_rets: pd.Series,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    hold_start = hold_rets.index[0].strftime("%Y-%m") if not hold_rets.empty else "N/A"
    hold_end = hold_rets.index[-1].strftime("%Y-%m") if not hold_rets.empty else "N/A"

    ic_table_rows = "\n".join(
        f"| {int(yr)} | {val:+.3f} |" for yr, val in ic_by_year.items()
    )

    lines = [
        "# Sprint T — Mining Formulaico Returns-Only Disciplinato",
        "",
        f"**Data run:** {today}  ",
        f"**Config hash:** `{cfg_hash}`  ",
        f"**Spec:** `docs/superpowers/specs/2026-06-15-sprint-t-disciplined-mining-design.md`  ",
        "",
        "**BACKTEST-ONLY** — nessun broker/conto/denaro reale.",
        "",
        "---",
        "",
        "## Framing onesto (dichiarato ex-ante)",
        "",
        "Questo sprint è un esercizio di disciplina metodologica sul mining. "
        "L'esito probabile è **KILL**: lo spazio di ricerca returns-only su un "
        "singolo universo azionario (S&P 500 PIT) senza dati fondamentali, "
        "senza news, con segnali puramente tecnici, difficilmente supera tutti e "
        "tre i gate (PBO, DSR, holdout) nello stesso esperimento. "
        "Un verdetto KILL è informativo e onesto.",
        "",
        "---",
        "",
        "## Parametri",
        "",
        f"| Parametro | Valore |",
        "|---|---|",
        f"| N (spazio dichiarato) | {N} |",
        f"| n_kept (corr-pruning ≤0.7) | {n_kept} |",
        f"| TRAIN | 2004-01 → 2015-12 |",
        f"| HOLDOUT | 2016-01 → ultimo mese completo |",
        f"| Decile | {DECILE:.0%} |",
        f"| Min nomi | {MIN_NAMES} |",
        f"| TC (bps one-way) | {TC_BPS} |",
        f"| PBO partizioni | {PBO_PARTITIONS} |",
        "",
        "---",
        "",
        "## Risultati",
        "",
        f"| Metrica | Valore |",
        "|---|---|",
        f"| PBO (CSCV) | {pbo:.3f} (soglia < {PBO_MAX}) |",
        f"| DSR (n_trials=N={N}) | {dsr:.3f} (soglia > {DSR_MIN}) |",
        f"| Migliore config IS | `{best_name}` |",
        f"| Sharpe IS migliore | {best_is_sharpe:.3f} |",
        f"| Sharpe holdout ({hold_start}→{hold_end}) | {hold_sharpe:.3f} |",
        f"| Sharpe 60/40 (holdout) | {sharpe_6040:.3f} |",
        "",
        "### RankIC per anno (holdout)",
        "",
        "**NOTA:** lo scatter IS vs OOS nel grafico usa il holdout solo per "
        "diagnostica post-verdetto — non è selezione.",
        "",
        "| Anno | IC medio |",
        "|---|---|",
        ic_table_rows,
        "",
        "---",
        "",
        "## Criteri Kill",
        "",
        f"- **PBO < {PBO_MAX}:** {pbo:.3f} → **{'PASS' if pbo < PBO_MAX else 'FAIL'}**  ",
        f"- **DSR > {DSR_MIN}:** {dsr:.3f} → **{'PASS' if dsr > DSR_MIN else 'FAIL'}**  ",
        f"- **Holdout Sharpe > 60/40:** {hold_sharpe:.3f} vs {sharpe_6040:.3f} → "
        f"**{'PASS' if hold_sharpe > sharpe_6040 else 'FAIL'}**  ",
        f"- **RankIC/anno > 0 (tutti):** {bool((ic_by_year > 0).all())} → "
        f"**{'PASS' if bool((ic_by_year > 0).all()) else 'FAIL'}**  ",
        "",
        f"## VERDETTO: **{verdict}**",
        "",
        "---",
        "",
        "![chart](../backtest/backtest_sprint_t.png)",
        "",
        "*Script: `scripts/backtest_sprint_t.py` — spec congelata, un solo run OOS. BACKTEST-ONLY.*",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
