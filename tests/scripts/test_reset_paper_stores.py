"""Test del reset degli store di paper trading."""
from pathlib import Path

from scripts.reset_paper_stores import reset_store


def test_reset_moves_state_to_backup(tmp_path):
    store_dir = tmp_path / "paper_trading" / "paper_trading"
    store_dir.mkdir(parents=True)
    state_file = store_dir / "nav_history__state__state.parquet"
    state_file.write_bytes(b"corrupt")
    backup_root = tmp_path / "_backup"

    moved = reset_store(tmp_path / "paper_trading", backup_root, dry_run=False)

    assert state_file.exists() is False
    assert len(moved) == 1
    backed_up = backup_root / "paper_trading" / "paper_trading" / "nav_history__state__state.parquet"
    assert backed_up.exists()


def test_dry_run_moves_nothing(tmp_path):
    store_dir = tmp_path / "paper_trading" / "paper_trading"
    store_dir.mkdir(parents=True)
    state_file = store_dir / "nav_history__state__state.parquet"
    state_file.write_bytes(b"corrupt")

    moved = reset_store(tmp_path / "paper_trading", tmp_path / "_backup", dry_run=True)

    assert state_file.exists()
    assert len(moved) == 1  # riporta cosa SPOSTEREBBE


def test_cache_files_not_touched(tmp_path):
    """I parquet di cache prezzi NON devono essere spostati."""
    store_dir = tmp_path / "paper_trading"
    (store_dir / "prices").mkdir(parents=True)
    cache = store_dir / "prices" / "AAPL.O__2010-01-01__2026-06-10.parquet"
    cache.write_bytes(b"prices")

    moved = reset_store(store_dir, tmp_path / "_backup", dry_run=False)

    assert cache.exists()
    assert moved == []
