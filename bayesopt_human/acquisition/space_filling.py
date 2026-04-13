"""Max-min distance in flat and surrogate space."""

from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist

from bayesopt_human.gp.surrogate_space import SurrogateSpaceTransform


def max_min_distance_flat(
    X_existing: np.ndarray,
    n_dim: int,
) -> np.ndarray:
    """Find point maximizing minimum distance to existing points in flat space.

    Uses differential evolution with a large candidate pool for global optimization.

    Args:
        X_existing: Existing observation coordinates in flat space, shape (N, D).
        n_dim: Number of dimensions.

    Returns:
        Coordinates of the space-filling candidate in flat space, shape (D,).
    """
    bounds = [(0.0, 1.0 - 1e-10)] * n_dim

    def neg_min_dist(x):
        dists = cdist(x.reshape(1, -1), X_existing).flatten()
        return -np.min(dists)

    result = differential_evolution(
        neg_min_dist,
        bounds,
        seed=42,
        maxiter=200,
        popsize=max(15, 10 * n_dim),
        tol=1e-6,
    )
    return np.clip(result.x, 0.0, 1.0 - 1e-10)


def max_min_distance_surrogate(
    X_existing: np.ndarray,
    surrogate_transform: SurrogateSpaceTransform,
    n_dim: int,
) -> np.ndarray:
    """Find point maximizing minimum distance in surrogate space.

    Optimizes in flat space but evaluates distances in surrogate space.

    Args:
        X_existing: Existing observations in flat space, shape (N, D).
        surrogate_transform: Transform from flat to surrogate space.
        n_dim: Number of dimensions.

    Returns:
        Coordinates in flat space that maximize min surrogate-space distance.
    """
    X_existing_sur = surrogate_transform.to_surrogate(X_existing)
    bounds = [(0.0, 1.0 - 1e-10)] * n_dim

    def neg_min_dist(x):
        x_sur = surrogate_transform.to_surrogate(x.reshape(1, -1))
        dists = cdist(x_sur, X_existing_sur).flatten()
        return -np.min(dists)

    result = differential_evolution(
        neg_min_dist,
        bounds,
        seed=42,
        maxiter=200,
        popsize=max(15, 10 * n_dim),
        tol=1e-6,
    )
    return np.clip(result.x, 0.0, 1.0 - 1e-10)
