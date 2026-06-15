"""ExperimentConfig — configurazione CONGELATA di un esperimento di backtest.

Regola P1 (audit 2026-06-10): universo, griglie e finestre IS/OOS si fissano
QUI, in un file versionato, PRIMA di girare il backtest. Estendere una griglia
dopo aver visto i risultati richiede un nuovo file config (nuovo hash) e
incrementa il conteggio trial del DSR.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    universe: tuple[str, ...]
    equity_rics: tuple[str, ...]
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    rebalance_freq: str
    tc_bps: float
    execution_lag: int
    damp_grid: tuple[float, ...]
    mom_grid_balanced: tuple[float, ...]
    mom_grid_aggressive: tuple[float, ...]
    price_field: str
    price_adjustments: tuple[str, ...]
    notes: str = ""

    _TUPLE_FIELDS: ClassVar[tuple[str, ...]] = (
        "universe", "equity_rics", "damp_grid",
        "mom_grid_balanced", "mom_grid_aggressive", "price_adjustments",
    )

    def config_hash(self) -> str:
        """SHA-256 della rappresentazione canonica — identifica l'esperimento."""
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True))
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        data = json.loads(Path(path).read_text())
        for f in cls._TUPLE_FIELDS:
            if f in data and isinstance(data[f], list):
                data[f] = tuple(data[f])
        return cls(**data)
