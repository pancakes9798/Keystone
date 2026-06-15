# tests/portfolio/test_ledoit_wolf.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf as _SklearnLW

from jeanclaude.portfolio.covariance.ledoit_wolf import LedoitWolfCovariance

N_ASSETS = 5
N_OBS = 200


@pytest.fixture
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = rng.standard_normal((N_OBS, N_ASSETS)) * 0.01
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(N_ASSETS)])


def test_matrix_shape(returns):
    cov = LedoitWolfCovariance().fit(returns).matrix()
    assert cov.shape == (N_ASSETS, N_ASSETS)


def test_matrix_symmetric(returns):
    cov = LedoitWolfCovariance().fit(returns).matrix()
    np.testing.assert_array_almost_equal(cov, cov.T)


def test_matrix_positive_definite(returns):
    cov = LedoitWolfCovariance().fit(returns).matrix()
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues > 0), f"Not PD, min eigenvalue: {eigenvalues.min()}"


def test_shrinkage_in_range(returns):
    est = LedoitWolfCovariance().fit(returns)
    assert 0 < est.shrinkage < 1


def test_identical_to_sklearn_direct(returns):
    lw = _SklearnLW()
    lw.fit(returns.values.astype(float))
    cov_lw = LedoitWolfCovariance().fit(returns).matrix()
    np.testing.assert_array_almost_equal(cov_lw, lw.covariance_)


def test_raises_before_fit():
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        LedoitWolfCovariance().matrix()


def test_shrinkage_raises_before_fit():
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        _ = LedoitWolfCovariance().shrinkage


def test_matrix_returns_copy(returns):
    est = LedoitWolfCovariance().fit(returns)
    m1 = est.matrix()
    m1[0, 0] = 9999.0
    m2 = est.matrix()
    assert m2[0, 0] != 9999.0


def test_assume_centered_passthrough(returns):
    default_cov = LedoitWolfCovariance().fit(returns).matrix()
    centered_cov = LedoitWolfCovariance(assume_centered=True).fit(returns).matrix()
    assert not np.allclose(default_cov, centered_cov)


def test_raises_on_nan_input():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((200, 5)) * 0.01
    df = pd.DataFrame(data, columns=list("ABCDE"))
    df.iloc[10, 2] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        LedoitWolfCovariance().fit(df)


def test_raises_when_obs_less_than_assets():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((4, 5)) * 0.01, columns=list("ABCDE"))
    with pytest.raises(ValueError, match="n_obs"):
        LedoitWolfCovariance().fit(df)
