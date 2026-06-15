"""Guard test — config Sprint M congelata: no-trade band su base Sprint L."""
from jeanclaude.research.experiment import ExperimentConfig

_PATH = "configs/experiments/2026-06-11-sprint-m-no-trade-band.json"


def test_sprint_m_config_frozen():
    cfg = ExperimentConfig.from_json(_PATH)
    assert "sprint-m" in cfg.name
    assert len(cfg.universe) == 20
    assert cfg.mom_grid_balanced == (0.5,)          # mom congelato da Sprint I/L
    assert cfg.damp_grid == (0.25,)                 # singleton — damp congelato
    assert cfg.execution_lag == 1 and cfg.tc_bps == 10.0
    assert "threshold" in cfg.notes                 # griglia threshold documentata
    assert "0.05" in cfg.notes and "0.10" in cfg.notes and "0.20" in cfg.notes
