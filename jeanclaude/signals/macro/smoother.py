"""Regime smoothing — requires min_periods consecutive confirmations to switch."""
from __future__ import annotations

from dataclasses import dataclass

from .labels import RegimeLabel


@dataclass(frozen=True)
class SmootherConfig:
    min_prob: float = 0.65    # minimum P(new_regime) to consider a change
    min_periods: int = 2      # consecutive confirmations required to switch


@dataclass(frozen=True)
class SmoothedSignal:
    """Drop-in replacement for any signal with .regime and .confidence."""
    regime: RegimeLabel
    confidence: float


class RegimeSmoother:
    """State machine that confirms regime changes over multiple periods.

    A regime change fires only when the new regime is signalled with
    confidence >= min_prob for min_periods consecutive rebalancing periods.
    """

    def __init__(self, config: SmootherConfig | None = None) -> None:
        self._config = config if config is not None else SmootherConfig()
        self._current_regime: RegimeLabel | None = None
        self._pending_regime: RegimeLabel | None = None
        self._pending_count: int = 0

    def smooth(self, regime: RegimeLabel, confidence: float) -> SmoothedSignal:
        """Apply smoothing filter to one period's regime signal.

        Parameters
        ----------
        regime : RegimeLabel
            Raw regime predicted by the HMM for this period.
        confidence : float
            P(regime) from the HMM posterior, must be in [0.0, 1.0].

        Returns
        -------
        SmoothedSignal
            .regime — the confirmed (smoothed) regime for this period.
            .confidence — when the confirmed regime matches the input, this is the
            raw upstream confidence. When a different regime is held (pending or
            blocked), confidence = 1.0 so downstream consumers continue applying
            regime overlays at full strength for the confirmed regime.
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {confidence!r}")

        # First call: accept immediately
        if self._current_regime is None:
            self._current_regime = regime
            return SmoothedSignal(regime=regime, confidence=confidence)

        # Same as current: reset pending
        if regime == self._current_regime:
            self._pending_regime = None
            self._pending_count = 0
            return SmoothedSignal(regime=self._current_regime, confidence=confidence)

        # Below threshold: reset pending, hold current at full confidence
        if confidence < self._config.min_prob:
            self._pending_regime = None
            self._pending_count = 0
            return SmoothedSignal(regime=self._current_regime, confidence=1.0)

        # Above threshold, different regime
        if regime == self._pending_regime:
            self._pending_count += 1
        else:
            self._pending_regime = regime
            self._pending_count = 1

        if self._pending_count >= self._config.min_periods:
            self._current_regime = regime
            self._pending_regime = None
            self._pending_count = 0
            return SmoothedSignal(regime=regime, confidence=confidence)

        # Not yet confirmed: hold current at full confidence so overlays remain active
        return SmoothedSignal(regime=self._current_regime, confidence=1.0)

    def reset(self) -> None:
        """Reset state — call at start of each walk-forward run."""
        self._current_regime = None
        self._pending_regime = None
        self._pending_count = 0
