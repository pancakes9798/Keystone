"""Guard test — config Sprint K congelata: regime-band su parametri base da Sprint I."""
from jeanclaude.research.experiment import ExperimentConfig

_PATH = "configs/experiments/2026-06-10-sprint-k-regime-band.json"


def test_sprint_k_config_guard():
    cfg = ExperimentConfig.from_json(_PATH)
    assert "regime-band" in cfg.name
    assert len(cfg.universe) == 20
    assert cfg.damp_grid == (0.25,)            # parametri base congelati da Sprint I
    assert cfg.mom_grid_balanced == (0.5,)
    assert cfg.mom_grid_aggressive == (0.7,)
    assert cfg.execution_lag == 1
    assert cfg.tc_bps == 10.0
