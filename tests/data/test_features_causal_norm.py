"""La normalizzazione delle feature deve essere causale: il valore a t non cambia se estendo il campione."""
import numpy as np
import pandas as pd

from jeanclaude.data.transform.features import build_state_variables


def _macro(n):
    idx = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "VIX": 15 + rng.normal(0, 3, n).cumsum() * 0.05 + 5,
        "Oil": 60 + rng.normal(0, 1, n).cumsum() * 0.1,
        "Copper": 3 + rng.normal(0, 0.05, n).cumsum() * 0.01,
        "Gold": 1500 + rng.normal(0, 10, n).cumsum() * 0.1,
        "EURUSD": 1.1 + rng.normal(0, 0.005, n).cumsum() * 0.001,
    }, index=idx).abs()


def test_expanding_norm_is_causal():
    full = _macro(700)
    short = full.iloc[:500]
    sv_full = build_state_variables(full, normalization="expanding").dropna()
    sv_short = build_state_variables(short, normalization="expanding").dropna()
    common = sv_short.index.intersection(sv_full.index)
    pd.testing.assert_frame_equal(
        sv_full.loc[common], sv_short.loc[common], check_exact=False, atol=1e-10
    )


def test_full_sample_norm_still_available_with_warning():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        build_state_variables(_macro(300), normalization="full")
        assert any("look-ahead" in str(x.message) for x in w)


def test_default_is_expanding():
    full = _macro(700)
    sv_default = build_state_variables(full)
    sv_expanding = build_state_variables(full, normalization="expanding")
    pd.testing.assert_frame_equal(sv_default, sv_expanding)
