"""TrialRegistry: append-only, conteggio trial per il DSR."""
import json

import pytest

from jeanclaude.research.registry import TrialRegistry


def test_record_appends_jsonl(tmp_path):
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    reg.record(config_hash="abc", experiment="exp", params={"damp": 0.2},
               window="IS 2005-2011", metrics={"sharpe": 0.5})
    reg.record(config_hash="abc", experiment="exp", params={"damp": 0.3},
               window="IS 2005-2011", metrics={"sharpe": 0.6})
    lines = (tmp_path / "trials.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert row["params"]["damp"] == 0.2
    assert "timestamp" in row


def test_n_trials_counts_distinct_params(tmp_path):
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    reg.record("abc", "exp", {"damp": 0.2}, "IS", {"sharpe": 0.5})
    reg.record("abc", "exp", {"damp": 0.2}, "IS", {"sharpe": 0.5})  # ripetuto
    reg.record("abc", "exp", {"damp": 0.3}, "IS", {"sharpe": 0.6})
    assert reg.n_trials() == 2          # distinti su (experiment, params)
    assert reg.n_records() == 3


def test_n_trials_empty_registry(tmp_path):
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    assert reg.n_trials() == 0


def test_record_accepts_numpy_scalars(tmp_path):
    np = pytest.importorskip("numpy")
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    reg.record("abc", "exp", {"damp": np.float64(0.2), "n": np.int64(3)},
               "IS", {"sharpe": np.float64(0.5), "trades": np.int64(120)})
    assert reg.n_records() == 1
    assert reg.n_trials() == 1


def test_n_trials_filtered_by_config_hash(tmp_path):
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    reg.record("hash_a", "exp", {"d": 1}, "IS", {"s": 0.1})
    reg.record("hash_b", "exp", {"d": 2}, "IS", {"s": 0.2})
    assert reg.n_trials() == 2
    assert reg.n_trials(config_hash="hash_a") == 1
