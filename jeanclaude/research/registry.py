"""TrialRegistry — log append-only di ogni combinazione provata.

Serve a contare ONESTAMENTE n_trials per il Deflated Sharpe Ratio: ogni
(esperimento, parametri) valutato è un trial, anche se poi scartato.

Il file è append-only e scritto solo da questa classe: nessun recovery da
righe corrotte.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class TrialRegistry:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def record(
        self,
        config_hash: str,
        experiment: str,
        params: dict,
        window: str,
        metrics: dict,
    ) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_hash": config_hash,
            "experiment": experiment,
            "params": params,
            "window": window,
            "metrics": metrics,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)

        def _coerce(o):
            if hasattr(o, "item"):  # numpy scalar (int64, float64, bool_)
                return o.item()
            raise TypeError(f"Non serializzabile nel registry: {type(o).__name__}")

        with self._path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True, default=_coerce) + "\n")

    def _rows(self) -> list[dict]:
        if not self._path.exists():
            return []
        return [json.loads(line) for line in self._path.read_text().strip().splitlines() if line]

    def n_records(self) -> int:
        return len(self._rows())

    def n_trials(self, config_hash: str | None = None) -> int:
        """Trial distinti su (experiment, params) — input per il DSR.

        Args:
            config_hash: se fornito, conta solo le righe con quel config_hash.
                         Se None (default), conta tutti i trial nel registry.
        """
        rows = self._rows()
        if config_hash is not None:
            rows = [r for r in rows if r.get("config_hash") == config_hash]
        seen = {
            (r["experiment"], json.dumps(r["params"], sort_keys=True))
            for r in rows
        }
        return len(seen)
