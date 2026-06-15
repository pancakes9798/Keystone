#!/usr/bin/env python3
"""Validazione live della constituent history S&P 500 ricostruita dal JL change-log.

Cosa fa
-------
1. Costruisce i panieri mensili dal 2003-01 a oggi via ConstituentHistory.
2. Stampa il conteggio dei membri per ogni year-end (2003-2025).
3. Spot check su eventi noti:
   - LEH.N^I08 in basket 2008-06 (Lehman prima del crack)
   - LEH.N^I08 NON in basket 2008-12 (Lehman uscita il 17 settembre 2008)
   - GM.N^F09  in basket 2008-12 (GM prima del bankruptcy)
   - GM.N^F09  NON in basket 2009-12 (GM uscita il 3 giugno 2009)
   - TSLA.OQ   NON in basket 2015-12 (TSLA entrata S&P 500 solo a dic 2020)
   - TSLA.OQ   in basket 2021-12 (TSLA presente dopo ingresso dic 2020)
4. Scrive docs/research/<oggi>-validazione-constituents.md con la tabella
   e PASS/FAIL per ogni spot check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Bootstrap path
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from jeanclaude.data.constituents import ConstituentHistory  # noqa: E402
from jeanclaude.data.ingestion.refinitiv import RefinitivSource  # noqa: E402
from jeanclaude.data.storage.parquet_store import ParquetStore  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

START = "2003-01"
TODAY = pd.Timestamp.today().strftime("%Y-%m")

# ParquetStore appende la categoria 'constituents' alla base — la base è data/
DATA_DIR = Path(__file__).parent.parent / "data"
REPORT_PATH = (
    Path(__file__).parent.parent
    / "docs"
    / "research"
    / f"{pd.Timestamp.today().strftime('%Y-%m-%d')}-validazione-constituents.md"
)

# Conteggi year-end ESATTI dove la verità storica è nota con certezza:
# l'S&P 500 aveva esattamente 500 ticker fino ad aprile 2014 (sostituzioni
# one-for-one); 502 a fine 2014 (split GOOG/GOOGL + DISCK), 504 a fine 2015.
# Una tolleranza a banda (±10) non rileva errori sistematici da ±1-2 membri
# fantasma — dove la verità è esatta, il check deve essere esatto.
EXACT_COUNTS: dict[int, int] = {
    **{year: 500 for year in range(2003, 2014)},
    2014: 502,
    2015: 504,
}

# Banda di tolleranza per gli anni senza ancora esatta (share class multiple)
BAND_LO, BAND_HI = 490, 515

# Spot checks: (ric, snapshot_date, expected_present, description)
SPOT_CHECKS = [
    ("LEH.N^I08", "2008-06-30", True,  "Lehman in basket giugno 2008 (prima del crack)"),
    ("LEH.N^I08", "2008-12-31", False, "Lehman OUT basket dic 2008 (uscita 17 set 2008)"),
    ("GM.N^F09",  "2008-12-31", True,  "GM in basket dic 2008 (prima del bankruptcy)"),
    ("GM.N^F09",  "2009-12-31", False, "GM OUT basket dic 2009 (uscita 3 giu 2009)"),
    ("TSLA.OQ",   "2015-12-31", False, "TSLA assente nel 2015 (entrata dic 2020)"),
    ("TSLA.OQ",   "2021-12-31", True,  "TSLA presente nel 2021 (dopo ingresso dic 2020)"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_basket(
    baskets: dict[pd.Timestamp, frozenset[str]],
    date_str: str,
) -> frozenset[str] | None:
    """Return basket for the given date string (exact match)."""
    ts = pd.Timestamp(date_str)
    # Month-end dates in baskets may differ from exact date_str by a day;
    # look for the closest month-end within the same year-month.
    target_ym = (ts.year, ts.month)
    for k in baskets:
        if (k.year, k.month) == target_ym:
            return baskets[k]
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("Validazione constituent history S&P 500 (JL change-log)")
    print("=" * 70)

    # 1. Connect to Refinitiv
    src = RefinitivSource(session_type="platform")
    store = ParquetStore(DATA_DIR)
    ch = ConstituentHistory(source=src, store=store)

    print(f"\nCostruisco panieri mensili {START} → {TODAY}...")
    baskets = ch.monthly_baskets(START, TODAY)
    print(f"Panieri costruiti: {len(baskets)} snapshot mensili")

    # 2. Year-end counts table
    print("\n--- Conteggio membri per year-end ---")
    header = f"{'Anno':<6} {'Data':<12} {'Membri':>7}"
    print(header)
    print("-" * len(header))

    year_end_rows: list[tuple[int, str, int, bool]] = []
    for year in range(2003, pd.Timestamp.today().year + 1):
        date_str = f"{year}-12-31"
        basket = _find_basket(baskets, date_str)
        if basket is None:
            continue
        count = len(basket)
        if year in EXACT_COUNTS:
            ok = count == EXACT_COUNTS[year]
            flag = "" if ok else f"  ← ANOMALIA (atteso esatto {EXACT_COUNTS[year]})"
        else:
            ok = BAND_LO <= count <= BAND_HI
            flag = "" if ok else f"  ← ANOMALIA (atteso [{BAND_LO},{BAND_HI}])"
        year_end_rows.append((year, date_str, count, ok))
        print(f"{year:<6} {date_str:<12} {count:>7}{flag}")

    # 3. Spot checks
    print("\n--- Spot checks ---")
    spot_results: list[tuple[str, str, bool, bool, str]] = []  # ric, date, expected, actual_pass, desc

    for ric, date_str, expected_present, desc in SPOT_CHECKS:
        basket = _find_basket(baskets, date_str)
        if basket is None:
            status = "SKIP (basket non disponibile)"
            passed = False
        else:
            actually_present = ric in basket
            passed = (actually_present == expected_present)
            presence_str = "PRESENTE" if actually_present else "ASSENTE"
            expected_str = "PRESENTE" if expected_present else "ASSENTE"
            status = f"{'PASS' if passed else 'FAIL'} — {ric} è {presence_str} (atteso: {expected_str})"

        spot_results.append((ric, date_str, expected_present, passed, desc))
        symbol = "PASS" if passed else "FAIL"
        print(f"[{symbol}] {desc}")
        print(f"       {status}")

    # Summary
    n_pass = sum(1 for *_, passed, _ in spot_results if passed)
    n_total = len(spot_results)
    print(f"\nSpot checks: {n_pass}/{n_total} PASS")

    # 4. Write markdown report
    _write_report(year_end_rows, spot_results, baskets)
    print(f"\nReport scritto: {REPORT_PATH}")


def _write_report(
    year_end_rows: list[tuple[int, str, int, bool]],
    spot_results: list[tuple[str, str, bool, bool, str]],
    baskets: dict[pd.Timestamp, frozenset[str]],
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_pass = sum(1 for *_, passed, _ in spot_results if passed)
    n_total = len(spot_results)
    n_count_fail = sum(1 for *_, ok in year_end_rows if not ok)
    overall = (
        "PASS"
        if n_pass == n_total and n_count_fail == 0
        else f"PARZIALE (spot {n_pass}/{n_total}, conteggi KO: {n_count_fail})"
    )

    lines = [
        "# Validazione Live — Constituent History S&P 500",
        "",
        f"**Data esecuzione:** {pd.Timestamp.today().strftime('%Y-%m-%d')}  ",
        "**Sorgente:** Refinitiv JL change-log (`TR.IndexJL*`, `IC=B`)  ",
        f"**Panieri ricostruiti:** {len(baskets)} snapshot mensili ({START} → oggi)  ",
        f"**Spot checks:** {n_pass}/{n_total} PASS  ",
        f"**Conteggi year-end fuori attesa:** {n_count_fail}  ",
        f"**Esito complessivo:** **{overall}**  ",
        "",
        "---",
        "",
        "## Conteggio membri per year-end",
        "",
        "Dove la verità storica è nota con certezza il check è ESATTO: 500 ticker",
        "fino al 2013 (sostituzioni one-for-one), 502 a fine 2014 (GOOG/GOOGL +",
        "DISCK), 504 a fine 2015. Per gli anni successivi (share class multiple)",
        f"si usa la banda [{BAND_LO}, {BAND_HI}].",
        "",
        "| Anno | Data | Membri | Atteso | Esito |",
        "|------|------|--------|--------|-------|",
    ]

    for year, date_str, count, ok in year_end_rows:
        atteso = (
            str(EXACT_COUNTS[year])
            if year in EXACT_COUNTS
            else f"[{BAND_LO}, {BAND_HI}]"
        )
        esito = "OK" if ok else "**ANOMALIA**"
        lines.append(f"| {year} | {date_str} | {count} | {atteso} | {esito} |")

    lines += [
        "",
        "---",
        "",
        "## Spot checks",
        "",
        "| # | RIC | Data | Atteso | Esito | Descrizione |",
        "|---|-----|------|--------|-------|-------------|",
    ]

    for i, (ric, date_str, expected_present, passed, desc) in enumerate(spot_results, 1):
        atteso = "PRESENTE" if expected_present else "ASSENTE"
        esito = "PASS" if passed else "FAIL"
        lines.append(f"| {i} | `{ric}` | {date_str} | {atteso} | **{esito}** | {desc} |")

    lines += [
        "",
        "---",
        "",
        "## Note metodologiche",
        "",
        "- **Algoritmo:** backward walk dal paniere corrente (basket_at(T) = composizione alla chiusura di T)",
        "  con semantica esplicita join/leave: il fetch usa `IC=B` (joiners+leavers) e il campo",
        "  `TR.IndexJLConstituentRIC.change` (colonna `Change`: Joiner/Leaver).",
        "- **Copertura:** con `IC=B` il log è essenzialmente completo dal 1995 (densità eventi 1995-2002",
        "  coerente col turnover storico pubblico). La finestra di backtest dal 2003 resta una convenzione",
        "  conservativa, non un limite del log.",
        "- **Join pre-1995:** i membri entrati prima dell'inizio del log (es. GM) non hanno evento join:",
        "  restano nel paniere fino all'inizio della finestra — comportamento corretto.",
        "- **Coppie same-day:** le coppie join+leave stesso giorno/stesso RIC (10 nel log, es. EVHC.N^L16)",
        "  sono cancellate come no-op netti prima del walk — l'esito non dipende dall'ordine riga API.",
        "- **Anomalia nota:** FJ.N^K00 (Fort James) ha due leave consecutivi (1997, 2000) senza join in mezzo —",
        "  unico gap genuino del log; con la semantica esplicita non corrompe il walk (add idempotente).",
        "- **Spot check RIC delistati:** i RIC con suffisso `^MMMYY` (es. `LEH.N^I08`) sono identificatori",
        "  Refinitiv per titoli rimossi; la loro presenza nel JL log è stata verificata nella probe 2026-06-11.",
        "",
        "*Script: `scripts/validate_constituents.py`*",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
