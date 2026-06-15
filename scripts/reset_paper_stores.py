#!/usr/bin/env python3
"""Reset degli store di paper trading corrotti (audit 2026-06-10).

Sposta i file di stato (__state__state.parquet) in una directory di backup e
lascia che i broker si re-inizializzino al capitale iniziale al prossimo run.
Le cache prezzi/macro NON vengono toccate.

Default: dry-run (mostra cosa sposterebbe). Eseguire con --apply per agire.

Usage:
    uv run python scripts/reset_paper_stores.py            # dry-run
    uv run python scripts/reset_paper_stores.py --apply
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("reset_paper_stores")

_REPO_ROOT = Path(__file__).parent.parent
_STORES = [
    _REPO_ROOT / "data" / "paper_trading",
    _REPO_ROOT / "data" / "paper_etf",
    _REPO_ROOT / "data" / "agent",
]
_STATE_GLOB = "**/*__state__state.parquet"
_COOLDOWN_FILE = "last_rebalance_date.txt"


def reset_store(store_root: Path, backup_root: Path, dry_run: bool) -> list[Path]:
    """Sposta i file di stato di ``store_root`` sotto ``backup_root``.

    Returns la lista dei file individuati (spostati se ``dry_run=False``).
    """
    if not store_root.exists():
        return []
    targets = sorted(store_root.glob(_STATE_GLOB))
    cooldown = store_root / _COOLDOWN_FILE
    if cooldown.exists():
        targets.append(cooldown)

    for src in targets:
        rel = src.relative_to(store_root.parent)
        dst = backup_root / rel
        action = "SPOSTEREBBE" if dry_run else "sposto"
        logger.info("%s %s → %s", action, src, dst)
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset paper trading stores (backup + remove)")
    parser.add_argument("--apply", action="store_true", help="esegue davvero (default: dry-run)")
    args = parser.parse_args()
    dry_run = not args.apply

    backup_root = _REPO_ROOT / "data" / f"_backup_corrupt_{date.today().isoformat()}"
    total = 0
    for store in _STORES:
        total += len(reset_store(store, backup_root, dry_run=dry_run))

    if total == 0:
        logger.info("Nessun file di stato trovato — niente da fare.")
        sys.exit(0)
    if dry_run:
        logger.info("DRY-RUN: %d file individuati. Rilanciare con --apply per procedere.", total)
    else:
        logger.info("Reset completato: %d file in %s", total, backup_root)


if __name__ == "__main__":
    main()
