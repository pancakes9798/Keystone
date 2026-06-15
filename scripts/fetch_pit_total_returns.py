#!/usr/bin/env python3
"""Fetch resumabile dei total return giornalieri per l'universo S&P 500 point-in-time.

Cosa fa
-------
1. Ricostruisce i panieri mensili 2003-01 → oggi (ConstituentHistory, JL IC=B).
2. Unione di tutti i RIC mai membri nel periodo (~1100+).
3. Fetch TR.TotalReturn1D giornaliero (RefinitivSource.get_total_returns) a
   blocchi di 50 RIC, con salvataggio progressivo su parquet: al riavvio le
   colonne già presenti vengono saltate (resumabile).
4. Stampa il report di copertura: RIC con dati / RIC richiesti, ed elenca i
   RIC senza alcun dato.

Output: data/constituents/total_returns_pit.parquet (gitignorato — dati LSEG).

Run:
    uv run python scripts/fetch_pit_total_returns.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from jeanclaude.data.constituents import ConstituentHistory  # noqa: E402
from jeanclaude.data.ingestion.refinitiv import RefinitivSource  # noqa: E402
from jeanclaude.data.storage.parquet_store import ParquetStore  # noqa: E402

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Configurazione (allineata alla spec congelata Sprint O)
# ---------------------------------------------------------------------------

BASKETS_START = "2003-01"
FETCH_START = "2002-09-01"   # >=252 giorni di borsa prima della prima formation
BLOCK_SIZE = 50              # RIC per blocco (internamente chunk da 10)

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_PATH = DATA_DIR / "constituents" / "total_returns_pit.parquet"


def main() -> None:
    today = pd.Timestamp.today()
    fetch_end = today.strftime("%Y-%m-%d")
    baskets_end = today.strftime("%Y-%m")

    print("=" * 70)
    print("Fetch total return PIT — TR.TotalReturn1D")
    print("=" * 70)

    src = RefinitivSource(session_type="platform")
    store = ParquetStore(DATA_DIR)
    ch = ConstituentHistory(source=src, store=store)

    baskets = ch.monthly_baskets(BASKETS_START, baskets_end)
    all_rics = sorted(set().union(*baskets.values()))
    print(f"Panieri: {len(baskets)} mesi — RIC unici nel periodo: {len(all_rics)}")

    # Resume: colonne già scaricate
    existing: pd.DataFrame | None = None
    done: set[str] = set()
    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        done = set(existing.columns)
        print(f"Resume: {len(done)} RIC già in cache ({OUT_PATH.name})")

    todo = [r for r in all_rics if r not in done]
    print(f"Da scaricare: {len(todo)} RIC ({FETCH_START} → {fetch_end})\n")

    frames: list[pd.DataFrame] = [existing] if existing is not None else []
    t0 = time.time()
    for i in range(0, len(todo), BLOCK_SIZE):
        block = todo[i: i + BLOCK_SIZE]
        df = src.get_total_returns(block, FETCH_START, fetch_end)
        frames.append(df)
        merged = pd.concat(frames, axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(OUT_PATH)
        frames = [merged]
        elapsed = time.time() - t0
        n_done = min(i + BLOCK_SIZE, len(todo))
        print(
            f"  blocco {i // BLOCK_SIZE + 1}: {n_done}/{len(todo)} RIC "
            f"({elapsed:.0f}s, {merged.shape[0]} righe x {merged.shape[1]} col)"
        )

    final = pd.read_parquet(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()

    # Report copertura + sidecar dei RIC senza dati (letto dalle precondizioni
    # di backtest_sprint_o.py: un RIC assente e NON dichiarato qui = fetch rotto)
    have_data = [r for r in all_rics if r in final.columns and final[r].notna().any()]
    missing = [r for r in all_rics if r not in have_data]
    sidecar = OUT_PATH.parent / "total_returns_pit_missing.json"
    sidecar.write_text(json.dumps(sorted(missing), indent=2))
    print("\n--- Copertura ---")
    print(f"RIC con almeno un dato: {len(have_data)}/{len(all_rics)} "
          f"({len(have_data) / len(all_rics):.1%})")
    if missing:
        print(f"RIC SENZA dati ({len(missing)}) — scritti in {sidecar.name}:")
        for r in missing:
            print(f"  {r}")
    print(f"\nParquet: {OUT_PATH} — {final.shape[0]} righe x {final.shape[1]} colonne")


if __name__ == "__main__":
    main()
