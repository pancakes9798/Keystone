"""ExperimentConfig: congelata, serializzabile, hashabile in modo stabile."""
import dataclasses
import pytest

from jeanclaude.research.experiment import ExperimentConfig


def _cfg(**overrides):
    base = dict(
        name="test-exp",
        universe=("SPY.P", "TLT.O"),
        equity_rics=("SPY.P",),
        is_start="2005-01-01", is_end="2011-12-31",
        oos_start="2012-01-01", oos_end="2026-04-30",
        rebalance_freq="ME", tc_bps=10.0, execution_lag=1,
        damp_grid=(0.0, 0.25, 0.5),
        mom_grid_balanced=(0.3, 0.5, 0.7),
        mom_grid_aggressive=(0.7, 1.0, 1.3),
        price_field="close",
        price_adjustments=("CCH", "CRE", "RTS", "RPO"),
        notes="",
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def test_config_is_frozen():
    cfg = _cfg()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.tc_bps = 0.0


def test_config_hash_stable_and_sensitive():
    assert _cfg().config_hash() == _cfg().config_hash()
    assert _cfg().config_hash() != _cfg(tc_bps=0.0).config_hash()


def test_roundtrip_json(tmp_path):
    cfg = _cfg()
    path = tmp_path / "exp.json"
    cfg.to_json(path)
    loaded = ExperimentConfig.from_json(path)
    assert loaded == cfg
    assert loaded.config_hash() == cfg.config_hash()


def test_grids_are_tuples_after_load(tmp_path):
    cfg = _cfg()
    path = tmp_path / "exp.json"
    cfg.to_json(path)
    loaded = ExperimentConfig.from_json(path)
    assert isinstance(loaded.universe, tuple)
    assert isinstance(loaded.damp_grid, tuple)
