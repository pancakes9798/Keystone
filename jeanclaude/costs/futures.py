"""Per-contract cost model for micro futures (Sprint S).

The Almgren-Chriss bps/ADV model cannot represent micro futures: costs are
fixed-per-contract (commission + exchange fee folded into commission) plus a
spread cross (in ticks) per side, plus a quarterly roll. This expresses those
dollar costs as a fraction of the contract notional (return-space), pessimistic
by default.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MicroFuturesCost:
    """Costo per-contratto in return-space sul nozionale del contratto.

    Parameters (default pessimistici per un micro tipo MES):
    - commission_per_side : commissione + fee per contratto per lato ($).
    - spread_ticks : tick attraversati per lato (>=1, mai mid).
    - tick_value : valore di 1 tick ($) — MES = $1.25 (0.25 pt x $5).
    """

    commission_per_side: float = 0.50
    spread_ticks: float = 1.0
    tick_value: float = 1.25

    def _dollar_per_contract(self) -> float:
        return self.commission_per_side + self.spread_ticks * self.tick_value

    def trade_cost_return(self, position_change_contracts: float, notional_per_contract: float) -> float:
        """Costo di un cambio di posizione (|Δcontratti|), in frazione di nozionale.

        Ogni contratto in |Δ| è UNA transazione (un lato): un open+close nel tempo
        è modellato come due eventi Δ separati (es. Δ=+1 poi Δ=−1), ciascuno
        addebitato qui una volta. Il chiamante NON deve raddoppiare per il round-trip.
        """
        if notional_per_contract <= 0:
            return 0.0
        contracts = abs(position_change_contracts)
        return contracts * self._dollar_per_contract() / notional_per_contract

    def roll_cost_return(self, position_contracts: float, notional_per_contract: float) -> float:
        """Costo di un roll trimestrale per i contratti detenuti, in frazione di nozionale.

        Assume roll via gambe OUTRIGHT (chiudi front + apri back ~1 spread+commissione
        per contratto) — stesso costo per-contratto di un trade. È una stima
        pessimistica: un calendar-spread su Globex costerebbe meno.
        """
        if notional_per_contract <= 0:
            return 0.0
        contracts = abs(position_contracts)
        return contracts * self._dollar_per_contract() / notional_per_contract
