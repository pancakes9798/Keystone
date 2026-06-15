"""La config dell'esperimento è quella del design doc — guard contro modifiche silenziose."""
from jeanclaude.research.experiment import ExperimentConfig

_PATH = "configs/experiments/2026-06-10-sprint-i-20etf.json"


def test_universe_is_the_20_etf_exante():
    cfg = ExperimentConfig.from_json(_PATH)
    assert len(cfg.universe) == 20
    assert "NVDA.O" not in cfg.universe          # niente vincitori ex-post
    assert "AAPL.O" not in cfg.universe


def test_grids_are_the_original_design_grids():
    cfg = ExperimentConfig.from_json(_PATH)
    assert cfg.damp_grid == (0.0, 0.25, 0.5)
    assert cfg.mom_grid_aggressive == (0.7, 1.0, 1.3)   # NON la griglia estesa v16
    assert cfg.execution_lag == 1
    assert cfg.tc_bps == 10.0
