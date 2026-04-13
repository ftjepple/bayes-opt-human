"""KNN coverage metrics, comparison to optimal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.neighbors import NearestNeighbors

from bayesopt_human.gp.surrogate_space import SurrogateSpaceTransform


@dataclass
class CoverageReport:
    """Container for coverage diagnostic results."""

    avg_knn_distance_flat: float
    min_knn_distance_flat: float
    avg_knn_distance_surrogate: float | None
    min_knn_distance_surrogate: float | None
    optimal_baseline_flat: float
    ratio_to_optimal_flat: float
    optimal_baseline_surrogate: float | None
    ratio_to_optimal_surrogate: float | None
    k: int


def _knn_distances(X: np.ndarray, k: int) -> np.ndarray:
    """Compute mean KNN distance for each point.

    Args:
        X: Points, shape (N, D).
        k: Number of neighbors.

    Returns:
        Mean distance to K nearest neighbors for each point, shape (N,).
    """
    k_actual = min(k, len(X) - 1)
    if k_actual < 1:
        return np.zeros(len(X))

    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric="euclidean")
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    # Exclude self-distance (first column)
    return distances[:, 1:].mean(axis=1)


def _optimal_space_filling_distance(n_obs: int, n_dim: int) -> float:
    """Estimate expected average nearest-neighbor distance for an optimal
    space-filling design in [0,1]^D.

    Uses the asymptotic formula for maximin distance in a unit hypercube:
    d ~ c * n^(-1/D) where c depends on D.
    """
    if n_obs <= 1 or n_dim <= 0:
        return 0.0
    # Approximate constant for uniform distribution in unit hypercube
    # Expected NN distance ~ Gamma(1 + 1/D) / (n * V_D)^(1/D)
    # Simplified: ~ 0.5 * n^(-1/D) for reasonable D
    return 0.5 * (n_obs ** (-1.0 / n_dim))


def coverage_metrics(
    X: np.ndarray,
    k: int = 5,
    surrogate_transform: SurrogateSpaceTransform | None = None,
) -> CoverageReport:
    """Compute KNN-based coverage metrics in flat and/or surrogate space.

    Args:
        X: Normalized input coordinates in flat space, shape (N, D).
        k: Number of nearest neighbors.
        surrogate_transform: If provided, also compute surrogate-space metrics.

    Returns:
        CoverageReport with avg/min KNN distances and comparison to optimal.
    """
    n_obs, n_dim = X.shape
    k_actual = min(k, n_obs - 1)

    # Flat space
    flat_dists = _knn_distances(X, k_actual)
    avg_flat = float(np.mean(flat_dists))
    min_flat = float(np.min(flat_dists))
    opt_flat = _optimal_space_filling_distance(n_obs, n_dim)
    ratio_flat = avg_flat / opt_flat if opt_flat > 0 else float("inf")

    # Surrogate space
    avg_sur = None
    min_sur = None
    opt_sur = None
    ratio_sur = None
    if surrogate_transform is not None:
        X_sur = surrogate_transform.to_surrogate(X)
        sur_dists = _knn_distances(X_sur, k_actual)
        avg_sur = float(np.mean(sur_dists))
        min_sur = float(np.min(sur_dists))
        opt_sur = _optimal_space_filling_distance(n_obs, n_dim)
        ratio_sur = avg_sur / opt_sur if opt_sur > 0 else float("inf")

    return CoverageReport(
        avg_knn_distance_flat=avg_flat,
        min_knn_distance_flat=min_flat,
        avg_knn_distance_surrogate=avg_sur,
        min_knn_distance_surrogate=min_sur,
        optimal_baseline_flat=opt_flat,
        ratio_to_optimal_flat=ratio_flat,
        optimal_baseline_surrogate=opt_sur,
        ratio_to_optimal_surrogate=ratio_sur,
        k=k_actual,
    )


def coverage_gain(
    X_existing: np.ndarray,
    x_new: np.ndarray,
    k: int = 5,
    surrogate_transform: SurrogateSpaceTransform | None = None,
) -> tuple[float, float]:
    """Compute improvement in average KNN distance if x_new is added.

    Args:
        X_existing: Current observations in flat space, shape (N, D).
        x_new: New candidate point, shape (D,).
        k: Number of neighbors.
        surrogate_transform: If provided, also compute surrogate-space gain.

    Returns:
        Tuple of (flat_gain, surrogate_gain). Surrogate gain is 0.0 if no transform.
    """
    x_new = x_new.reshape(1, -1)
    k_actual = min(k, len(X_existing))

    # Flat space
    before_flat = float(np.mean(_knn_distances(X_existing, k_actual)))
    X_aug = np.vstack([X_existing, x_new])
    after_flat = float(np.mean(_knn_distances(X_aug, k_actual)))
    flat_gain = before_flat - after_flat  # Reduction is improvement

    # Surrogate space
    sur_gain = 0.0
    if surrogate_transform is not None:
        X_sur = surrogate_transform.to_surrogate(X_existing)
        x_new_sur = surrogate_transform.to_surrogate(x_new)
        before_sur = float(np.mean(_knn_distances(X_sur, k_actual)))
        X_aug_sur = np.vstack([X_sur, x_new_sur])
        after_sur = float(np.mean(_knn_distances(X_aug_sur, k_actual)))
        sur_gain = before_sur - after_sur

    return flat_gain, sur_gain
