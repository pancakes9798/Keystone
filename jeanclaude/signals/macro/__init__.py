"""
Macro regime detection via Hidden Markov Model.

Usage::

    from jeanclaude.signals.macro import RegimeDetector, RegimeLabel, RegimeResult
    from jeanclaude.data.transform.features import build_state_variables

    state_vars = build_state_variables(macro_df)
    detector = RegimeDetector()
    result = detector.fit(state_vars)

    label, proba = detector.current_regime(state_vars)
"""
from __future__ import annotations

from .labels import RegimeLabel, RegimeResult
from .smoother import RegimeSmoother, SmootherConfig, SmoothedSignal

__all__ = [
    "RegimeDetector",
    "RegimeLabel",
    "RegimeResult",
    "RegimeSmoother",
    "SmootherConfig",
    "SmoothedSignal",
]


def __getattr__(name: str):
    if name == "RegimeDetector":
        from .detector import RegimeDetector  # noqa: PLC0415
        return RegimeDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
