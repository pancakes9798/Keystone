"""Due fit identici devono produrre labels identiche (stesso seed)."""
import numpy as np
import pandas as pd

from jeanclaude.signals.macro.detector import RegimeDetector


def _sv(n=400):
    idx = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(11)
    return pd.DataFrame(rng.normal(0, 1, (n, 4)), index=idx,
                        columns=["a", "b", "c", "d"])


def test_same_seed_same_labels():
    sv = _sv()
    r1 = RegimeDetector(random_state=7).fit(sv)
    r2 = RegimeDetector(random_state=7).fit(sv)
    assert (r1.labels == r2.labels).all()
