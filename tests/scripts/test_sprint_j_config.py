"""Guard test — config Sprint J congelata: parametri base da Sprint I non ricalibrati."""
from jeanclaude.research.experiment import ExperimentConfig

_PATH = "configs/experiments/2026-06-10-sprint-j-voltarget.json"


def test_base_params_frozen_from_sprint_i():
    cfg = ExperimentConfig.from_json(_PATH)
    assert cfg.damp_grid == (0.25,)           # nessuna ricalibrazione del base
    assert cfg.mom_grid_balanced == (0.5,)
    assert cfg.mom_grid_aggressive == (0.7,)
    assert cfg.execution_lag == 1 and cfg.tc_bps == 10.0
    assert len(cfg.universe) == 20
    assert "voltarget" in cfg.name
