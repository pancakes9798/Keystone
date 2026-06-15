"""Hierarchical Risk Parity (HRP) optimizer.

Pure numpy/scipy implementation of López de Prado's HRP algorithm
(Machine Learning for Asset Managers, ch.2).

Architecture:
    1. Compute distance matrix from correlation matrix
    2. Hierarchical clustering (Ward linkage via scipy)
    3. Quasi-diagonalization: reorder assets by cluster proximity
    4. Recursive bisection: allocate inverse-variance weights per cluster

The covariance estimator is injected — any object with:
    .fit(returns: pd.DataFrame) -> self
    .matrix() -> np.ndarray
qualifies (EWMACovariance, SampleCovariance).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


class HRPOptimizer:
    """Hierarchical Risk Parity optimizer.

    Parameters
    ----------
    cov_estimator : EWMACovariance | SampleCovariance
        Any estimator with fit(returns) -> self and matrix() -> np.ndarray.
    linkage_method : str
        Scipy linkage method (default "ward").

    Nota determinismo
    -----------------
    scipy.cluster.hierarchy.linkage può rompere i pareggi di distanza in modo
    dipendente dalla versione/piattaforma — il determinismo cross-machine non è
    garantito sui tie.
    """

    def __init__(
        self,
        cov_estimator,
        linkage_method: str = "ward",
    ) -> None:
        self.cov_estimator = cov_estimator
        self.linkage_method = linkage_method

    def optimize(self, returns: pd.DataFrame) -> pd.Series:
        """Compute HRP weights from return history.

        Parameters
        ----------
        returns : pd.DataFrame
            Date-indexed DataFrame (rows=dates, columns=asset names).

        Returns
        -------
        weights : pd.Series
            Non-negative weights indexed by asset name, summing to 1.
        """
        cov = self.cov_estimator.fit(returns).matrix()
        assets = list(returns.columns)

        # Correlation matrix from covariance
        std = np.sqrt(np.diag(cov))
        std = np.where(std == 0, 1e-10, std)
        corr = cov / np.outer(std, std)
        corr = np.clip(corr, -1.0, 1.0)

        # Distance matrix: d = sqrt(0.5 * (1 - corr))
        dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, 1))
        np.fill_diagonal(dist, 0.0)

        # Hierarchical clustering
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method=self.linkage_method)

        # Quasi-diagonalization
        sorted_idx = _quasi_diag(link)

        # Recursive bisection
        cov_df = pd.DataFrame(cov, index=assets, columns=assets)
        weights = _recursive_bisect(cov_df, sorted_idx)

        # Reindex to original asset order and normalize
        weights = weights.reindex(assets).fillna(0.0)
        weights = weights / weights.sum()
        return weights


def _quasi_diag(link: np.ndarray) -> list[int]:
    """Sort leaf nodes so similar assets are adjacent (quasi-diagonalization)."""
    link = link.astype(int)
    n_leaves = link[-1, 3]

    sort_ix = [int(link[-1, 0]), int(link[-1, 1])]

    while max(sort_ix) >= n_leaves:
        new_ix = []
        for item in sort_ix:
            if item >= n_leaves:
                row = item - n_leaves
                new_ix.extend([int(link[row, 0]), int(link[row, 1])])
            else:
                new_ix.append(item)
        sort_ix = new_ix

    return sort_ix


def _recursive_bisect(cov: pd.DataFrame, sort_ix: list[int]) -> pd.Series:
    """Allocate weights via recursive bisection of the quasi-diagonal covariance."""
    assets = [cov.columns[i] for i in sort_ix]
    weights = pd.Series(1.0, index=assets)
    clusters = [assets]

    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            var_left = _cluster_var(cov, left)
            var_right = _cluster_var(cov, right)

            total_var = var_left + var_right
            alpha_left = 1 - var_left / total_var
            alpha_right = 1 - var_right / total_var

            weights[left] *= alpha_left
            weights[right] *= alpha_right

            next_clusters.extend([left, right])
        clusters = next_clusters

    return weights


def _cluster_var(cov: pd.DataFrame, assets: list[str]) -> float:
    """Variance of inverse-variance portfolio within a cluster."""
    sub_cov = cov.loc[assets, assets].values
    inv_diag = 1.0 / np.maximum(np.diag(sub_cov), 1e-10)
    ivp = inv_diag / inv_diag.sum()
    return float(ivp @ sub_cov @ ivp)
