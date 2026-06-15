"""Tests for CompositeSignal types and DEFAULT_REGIME_TABLE."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from jeanclaude.signals.composite.types import (
    CompositeSignal,
    DEFAULT_REGIME_TABLE,
    RegimeTable,
)
from jeanclaude.signals.macro.labels import RegimeLabel


def test_composite_signal_is_immutable():
    """CompositeSignal è un frozen dataclass — i campi non possono essere riassegnati."""
    sig = CompositeSignal(
        scores=pd.Series({"equity": 0.5, "bonds": -0.2}),
        confidence=0.8,
        regime=RegimeLabel.EXPANSION,
        regime_proba=np.array([0.9, 0.05, 0.05]),
        timestamp=pd.Timestamp("2024-01-01"),
    )
    with pytest.raises(AttributeError):
        sig.confidence = 0.0  # type: ignore[misc]


def test_composite_signal_fields():
    """CompositeSignal ha tutti i campi con tipi corretti."""
    sig = CompositeSignal(
        scores=pd.Series({"equity": 0.5}),
        confidence=0.7,
        regime=RegimeLabel.CONTRACTION,
        regime_proba=np.array([0.1, 0.8, 0.1]),
        timestamp=pd.Timestamp("2024-06-01"),
    )
    assert isinstance(sig.scores, pd.Series)
    assert isinstance(sig.confidence, float)
    assert isinstance(sig.regime, RegimeLabel)
    assert sig.regime_proba.shape == (3,)
    assert isinstance(sig.timestamp, pd.Timestamp)


def test_default_regime_table_structure():
    """DEFAULT_REGIME_TABLE ha tutti e 3 i regimi e asset class consistenti."""
    table = DEFAULT_REGIME_TABLE
    assert set(table.keys()) == {
        RegimeLabel.EXPANSION,
        RegimeLabel.CONTRACTION,
        RegimeLabel.TRANSITION,
    }
    asset_sets = [set(v.keys()) for v in table.values()]
    assert asset_sets[0] == asset_sets[1] == asset_sets[2]


def test_default_regime_table_scores_in_range():
    """Tutti gli score in DEFAULT_REGIME_TABLE sono in [-1, 1]."""
    for regime, assets in DEFAULT_REGIME_TABLE.items():
        for asset, score in assets.items():
            assert -1.0 <= score <= 1.0, (
                f"{regime.name}/{asset} score={score} fuori range"
            )


from jeanclaude.signals.composite.builder import CompositeSignalBuilder


def test_invalid_table_missing_regime():
    """Manca un regime nella table → ValueError che menziona il nome del regime."""
    bad_table = {
        RegimeLabel.EXPANSION:   {"equity": 0.5},
        RegimeLabel.CONTRACTION: {"equity": -0.5},
        # TRANSITION mancante
    }
    with pytest.raises(ValueError, match="TRANSITION"):
        CompositeSignalBuilder(bad_table)


def test_invalid_table_score_out_of_range():
    """Score > 1 → ValueError che menziona il nome dell'asset."""
    bad_table = {
        RegimeLabel.EXPANSION:   {"equity": 1.5},   # fuori range
        RegimeLabel.CONTRACTION: {"equity": -0.5},
        RegimeLabel.TRANSITION:  {"equity":  0.0},
    }
    with pytest.raises(ValueError, match="equity"):
        CompositeSignalBuilder(bad_table)


def test_invalid_table_mismatched_assets():
    """Asset class diverse tra regimi → ValueError che menziona 'asset'."""
    bad_table = {
        RegimeLabel.EXPANSION:   {"equity": 0.5, "bonds": -0.3},
        RegimeLabel.CONTRACTION: {"equity": -0.5},          # manca bonds
        RegimeLabel.TRANSITION:  {"equity": 0.0, "bonds": 0.1},
    }
    with pytest.raises(ValueError, match="asset"):
        CompositeSignalBuilder(bad_table)


def test_valid_table_does_not_raise():
    """Una table valida si costruisce senza errori."""
    builder = CompositeSignalBuilder(DEFAULT_REGIME_TABLE)
    assert builder is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_regime_result(proba_last: list[float], n: int = 5) -> RegimeResult:
    """Crea un RegimeResult minimale con probabilità controllate sull'ultima riga."""
    from jeanclaude.signals.macro.labels import RegimeResult
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    proba_arr = np.tile(proba_last, (n, 1)).astype(np.float32)
    probabilities = pd.DataFrame(proba_arr, index=idx, columns=[0, 1, 2])
    dominant = int(np.argmax(proba_last))
    labels = pd.Series(
        [RegimeLabel(dominant)] * n,
        index=idx,
        name="regime",
    )
    return RegimeResult(labels=labels, probabilities=probabilities, model=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test build()
# ---------------------------------------------------------------------------

def test_build_returns_composite_signal():
    """build() restituisce un CompositeSignal con le asset class corrette."""
    builder = CompositeSignalBuilder(DEFAULT_REGIME_TABLE)
    sig = builder.build(_make_regime_result([0.9, 0.05, 0.05]))
    assert isinstance(sig, CompositeSignal)
    assert set(sig.scores.index) == {"equity", "bonds", "gold", "cash"}


def test_scores_in_range():
    """Tutti gli score prodotti da build() sono in [-1, 1]."""
    builder = CompositeSignalBuilder(DEFAULT_REGIME_TABLE)
    for proba in [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1/3, 1/3, 1/3],
    ]:
        sig = builder.build(_make_regime_result(proba))
        assert sig.scores.between(-1.0, 1.0).all(), (
            f"Score fuori range per proba={proba}: {sig.scores.to_dict()}"
        )


def test_scores_weighted_by_proba():
    """Score = media pesata per probabilità posteriori (calcolo esplicito su equity)."""
    builder = CompositeSignalBuilder(DEFAULT_REGIME_TABLE)
    proba = [0.7, 0.2, 0.1]
    sig = builder.build(_make_regime_result(proba))
    # equity: 0.7*0.8 + 0.2*(-0.8) + 0.1*0.0 = 0.56 - 0.16 + 0.0 = 0.40
    expected_equity = (
        proba[0] * DEFAULT_REGIME_TABLE[RegimeLabel.EXPANSION]["equity"]
        + proba[1] * DEFAULT_REGIME_TABLE[RegimeLabel.CONTRACTION]["equity"]
        + proba[2] * DEFAULT_REGIME_TABLE[RegimeLabel.TRANSITION]["equity"]
    )
    np.testing.assert_allclose(sig.scores["equity"], expected_equity, atol=1e-6)


def test_confidence_pure_regime():
    """Probabilità [1, 0, 0] → confidence = 1.0 (entropia = 0)."""
    builder = CompositeSignalBuilder(DEFAULT_REGIME_TABLE)
    sig = builder.build(_make_regime_result([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(sig.confidence, 1.0, atol=1e-6)


def test_confidence_uniform():
    """Probabilità [1/3, 1/3, 1/3] → confidence = 0.0 (entropia massima)."""
    builder = CompositeSignalBuilder(DEFAULT_REGIME_TABLE)
    sig = builder.build(_make_regime_result([1/3, 1/3, 1/3]))
    np.testing.assert_allclose(sig.confidence, 0.0, atol=1e-6)


def test_dominant_regime():
    """Il campo regime = argmax delle probabilità posteriori."""
    builder = CompositeSignalBuilder(DEFAULT_REGIME_TABLE)
    sig = builder.build(_make_regime_result([0.1, 0.8, 0.1]))
    assert sig.regime == RegimeLabel.CONTRACTION


def test_timestamp():
    """Il campo timestamp = ultimo indice del RegimeResult."""
    builder = CompositeSignalBuilder(DEFAULT_REGIME_TABLE)
    result = _make_regime_result([0.5, 0.3, 0.2], n=10)
    sig = builder.build(result)
    assert sig.timestamp == result.labels.index[-1]
