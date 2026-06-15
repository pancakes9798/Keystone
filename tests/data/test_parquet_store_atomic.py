"""Test scritture atomiche di ParquetStore."""
import pandas as pd
import pytest

from jeanclaude.data.storage.parquet_store import ParquetStore


@pytest.fixture
def store(tmp_path):
    return ParquetStore(tmp_path)


def _df(values):
    return pd.DataFrame(
        {"nav": values},
        index=pd.DatetimeIndex(pd.date_range("2026-01-01", periods=len(values)), name="date"),
    )


def test_save_leaves_no_tmp_files(store, tmp_path):
    store.save("paper_trading", "nav_history", "state", "state", _df([100.0]))
    leftovers = list(tmp_path.rglob("*.tmp"))
    assert leftovers == []


def test_save_overwrite_is_atomic_and_readable(store):
    store.save("paper_trading", "nav_history", "state", "state", _df([100.0]))
    store.save("paper_trading", "nav_history", "state", "state", _df([100.0, 101.0]))
    loaded = store.load("paper_trading", "nav_history", "state", "state")
    assert len(loaded) == 2
    assert loaded["nav"].iloc[-1] == 101.0


def test_failed_write_preserves_existing_file(store, monkeypatch):
    """Se la scrittura del file temporaneo esplode, il file esistente resta intatto."""
    import jeanclaude.data.storage.parquet_store as ps_module

    store.save("paper_trading", "nav_history", "state", "state", _df([100.0]))

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(ps_module.pq, "write_table", boom)
    with pytest.raises(OSError):
        store.save("paper_trading", "nav_history", "state", "state", _df([1.0, 2.0]))

    loaded = store.load("paper_trading", "nav_history", "state", "state")
    assert len(loaded) == 1
    assert loaded["nav"].iloc[0] == 100.0

    leftovers = list(store.root.rglob("*.tmp"))
    assert leftovers == [], f"Stale tmp files found: {leftovers}"
