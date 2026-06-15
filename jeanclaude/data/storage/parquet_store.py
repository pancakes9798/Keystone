"""
Local Parquet storage layer.

Persists DataFrames to disk with a consistent naming scheme.
Acts as a transparent cache in front of remote data sources.

Directory layout::

    data_root/
    ├── prices/
    │   ├── SPY__2020-01-01__2024-12-31.parquet
    │   └── ...
    ├── macro/
    │   ├── FRED__FEDFUNDS__2010-01-01__2024-12-31.parquet
    │   └── ...
    └── transforms/
        └── returns__daily__2020-01-01__2024-12-31.parquet

Usage::

    store = ParquetStore("~/jeanclaude_data")

    # Save
    store.save("prices", "SPY_TLT", start, end, df)

    # Load (returns None if not found)
    df = store.load("prices", "SPY_TLT", start, end)
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class ParquetStore:
    """File-based Parquet cache for market data.

    Parameters
    ----------
    root : str or Path
        Root directory for all stored files. Created if it doesn't exist.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        category: str,
        key: str,
        start: str,
        end: str,
        df: pd.DataFrame,
    ) -> Path:
        """Persist a DataFrame to Parquet.

        Parameters
        ----------
        category : str
            Subdirectory (e.g. ``'prices'``, ``'macro'``, ``'transforms'``).
        key : str
            Descriptive key for the dataset (used in filename).
        start, end : str
            Date range — stored in filename for quick identification.
        df : pd.DataFrame
            Data to persist. Index must be a DatetimeIndex.

        Returns
        -------
        Path
            Path of the saved file.
        """
        path = self._path(category, key, start, end)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Scrittura atomica: tmp nello stesso filesystem + os.replace.
        # Se il processo muore a metà scrittura, il file esistente resta valido.
        tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        table = pa.Table.from_pandas(df, preserve_index=True)
        try:
            pq.write_table(table, tmp_path, compression="snappy")
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)

        logger.info("Saved %s (%d rows × %d cols) → %s", key, len(df), len(df.columns), path)
        return path

    def load(
        self,
        category: str,
        key: str,
        start: str,
        end: str,
    ) -> Optional[pd.DataFrame]:
        """Load a previously saved DataFrame, or None if not found.

        Parameters
        ----------
        category, key, start, end : str
            Same parameters used when saving.

        Returns
        -------
        pd.DataFrame or None
        """
        path = self._path(category, key, start, end)
        if not path.exists():
            logger.debug("Cache miss: %s", path)
            return None

        logger.info("Cache hit: %s", path)
        table = pq.read_table(path)
        df = table.to_pandas()

        # Restore DatetimeIndex if pandas stored it as a column
        if "__index_level_0__" in df.columns:
            df = df.set_index("__index_level_0__")
        elif "date" in df.columns:
            df = df.set_index("date")

        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df

    def exists(self, category: str, key: str, start: str, end: str) -> bool:
        """Return True if this dataset is already cached."""
        return self._path(category, key, start, end).exists()

    def list_files(self, category: Optional[str] = None) -> list[Path]:
        """List all cached files, optionally filtered by category."""
        base = self.root / category if category else self.root
        return sorted(base.rglob("*.parquet"))

    def delete(self, category: str, key: str, start: str, end: str) -> bool:
        """Delete a cached file. Returns True if it existed."""
        path = self._path(category, key, start, end)
        if path.exists():
            path.unlink()
            logger.info("Deleted cache: %s", path)
            return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path(self, category: str, key: str, start: str, end: str) -> Path:
        """Build the canonical file path for a dataset.

        Uses a short hash of the key to avoid filesystem issues with
        long ticker lists while keeping the filename human-readable.
        """
        safe_key = key.replace("/", "-").replace(" ", "_")
        if len(safe_key) > 60:
            # Hash long keys (many tickers)
            h = hashlib.md5(safe_key.encode()).hexdigest()[:8]
            safe_key = f"{safe_key[:52]}_{h}"

        filename = f"{safe_key}__{start}__{end}.parquet"
        return self.root / category / filename
