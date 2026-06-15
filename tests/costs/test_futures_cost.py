"""Tests for jeanclaude.costs.futures.MicroFuturesCost."""
from __future__ import annotations

import pytest

from jeanclaude.costs.futures import MicroFuturesCost


NOTIONAL = 27_000.0  # ~1 MES a indice 5400 (moltiplicatore $5)


class TestMicroFuturesCost:
    def test_trade_cost_is_commission_plus_spread_over_notional(self):
        c = MicroFuturesCost(commission_per_side=0.50, spread_ticks=1.0, tick_value=1.25)
        # 1 contratto scambiato: (0.50 + 1*1.25) / 27000
        assert c.trade_cost_return(1, NOTIONAL) == pytest.approx(1.75 / NOTIONAL)

    def test_trade_cost_uses_absolute_change(self):
        c = MicroFuturesCost(commission_per_side=0.50, spread_ticks=1.0, tick_value=1.25)
        # da +1 a -1 = |Δ| 2 contratti
        assert c.trade_cost_return(-2, NOTIONAL) == pytest.approx(2 * 1.75 / NOTIONAL)

    def test_no_change_no_cost(self):
        c = MicroFuturesCost()
        assert c.trade_cost_return(0, NOTIONAL) == 0.0

    def test_roll_cost_scales_with_contracts_held(self):
        c = MicroFuturesCost(commission_per_side=0.50, spread_ticks=1.0, tick_value=1.25)
        assert c.roll_cost_return(1, NOTIONAL) == pytest.approx(1.75 / NOTIONAL)
        assert c.roll_cost_return(0, NOTIONAL) == 0.0

    def test_zero_notional_is_safe(self):
        c = MicroFuturesCost()
        assert c.trade_cost_return(1, 0.0) == 0.0
        assert c.roll_cost_return(1, 0.0) == 0.0
