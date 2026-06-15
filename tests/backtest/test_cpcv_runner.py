"""CPCV con refit: i parametri vengono RISELEZIONATI su ogni train fold."""
import numpy as np
import pandas as pd

from jeanclaude.backtest.cpcv import CPCVConfig
from jeanclaude.backtest.cpcv_runner import cpcv_refit_validation


def _returns(n=600):
    idx = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(5)
    return pd.DataFrame(rng.normal(0.0003, 0.01, (n, 3)), index=idx,
                        columns=["A", "B", "C"])


def test_refit_called_per_fold_and_results_shape():
    calls = []

    def calibrate(train_returns: pd.DataFrame) -> dict:
        calls.append(len(train_returns))
        return {"damp": 0.2}

    def evaluate(test_returns: pd.DataFrame, params: dict) -> float:
        assert params == {"damp": 0.2}
        mean, std = test_returns.mean(axis=1).mean(), test_returns.mean(axis=1).std()
        return float(mean / std * np.sqrt(252)) if std > 0 else 0.0

    res = cpcv_refit_validation(
        _returns(), calibrate, evaluate,
        config=CPCVConfig(n_splits=4, n_test_splits=1),
        eval_horizon_days=21,
    )
    assert len(res) == 4                      # C(4,1) = 4 fold
    assert set(res.columns) >= {"fold", "test_sharpe", "params"}
    assert len(calls) == 4                    # un refit per fold
    # i train fold devono essere più corti del campione (purge+embargo)
    assert all(c < 600 for c in calls)


def test_summary_fraction_positive():
    def calibrate(tr):
        return {}

    def evaluate(te, p):
        return 1.0

    res = cpcv_refit_validation(_returns(), calibrate, evaluate,
                                config=CPCVConfig(n_splits=4, n_test_splits=1))
    assert (res["test_sharpe"] > 0).mean() == 1.0
