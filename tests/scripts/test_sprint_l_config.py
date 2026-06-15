"""Guard test — config Sprint L congelata: base strutturale equity + de-risk trend."""
from jeanclaude.research.experiment import ExperimentConfig

_PATH = "configs/experiments/2026-06-11-sprint-l-equity-structural.json"


def test_sprint_l_config_frozen():
    cfg = ExperimentConfig.from_json(_PATH)
    assert "equity-structural" in cfg.name
    assert len(cfg.universe) == 20
    assert cfg.mom_grid_balanced == (0.5,)        # mom congelato da Sprint I
    assert cfg.damp_grid == (0.25, 0.5)           # damp in griglia (2 valori)
    assert cfg.execution_lag == 1 and cfg.tc_bps == 10.0
    assert "equity_min" in cfg.notes              # la griglia floor è documentata
    assert "0.5" in cfg.notes and "0.7" in cfg.notes
