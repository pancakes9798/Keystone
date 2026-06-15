"""Tests for historical covariance estimators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jeanclaude.portfolio.covariance.historical import EWMACovariance, SampleCovariance


N_ASSETS = 5
N_OBS = 200


@pytest.fixture
def returns():
    np.random.seed(42)
    data = np.random.randn(N_OBS, N_ASSETS) * 0.01
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(N_ASSETS)])


def test_ewma_output_shape(returns):
    cov = EWMACovariance().fit(returns).matrix()
    assert cov.shape == (N_ASSETS, N_ASSETS)


def test_ewma_is_symmetric(returns):
    cov = EWMACovariance().fit(returns).matrix()
    np.testing.assert_array_almost_equal(cov, cov.T)


def test_ewma_is_positive_definite(returns):
    cov = EWMACovariance().fit(returns).matrix()
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues > 0), f"Not PD, min eigenvalue: {eigenvalues.min()}"


def test_ewma_raises_before_fit():
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        EWMACovariance().matrix()


def test_ewma_higher_lambda_gives_more_historical_weight():
    np.random.seed(0)
    r = pd.DataFrame(np.random.randn(300, 3) * 0.01, columns=list("ABC"))
    cov_long = EWMACovariance(lambda_=0.99).fit(r).matrix()
    cov_short = EWMACovariance(lambda_=0.50).fit(r).matrix()
    assert not np.allclose(cov_long, cov_short)


def test_sample_covariance_output_shape(returns):
    cov = SampleCovariance().fit(returns).matrix()
    assert cov.shape == (N_ASSETS, N_ASSETS)


def test_sample_covariance_is_positive_semidefinite(returns):
    cov = SampleCovariance().fit(returns).matrix()
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues >= -1e-10)


def test_sample_covariance_raises_before_fit():
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        SampleCovariance().matrix()


def test_matrix_returns_copy(returns):
    """Mutating the returned matrix must not affect internal state."""
    est = EWMACovariance().fit(returns)
    m1 = est.matrix()
    m1[0, 0] = 9999.0
    m2 = est.matrix()
    assert m2[0, 0] != 9999.0
