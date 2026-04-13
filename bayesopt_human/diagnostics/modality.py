"""KNN-based local optima detection and basin counting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

from bayesopt_human.gp.surrogate_space import SurrogateSpaceTransform


@dataclass
class ModalityReport:
    """Container for modality diagnostic results."""

    n_local_optima_flat: int
    n_basins_flat: int
    local_optima_indices_flat: np.ndarray
    local_optima_coords_flat: np.ndarray
    n_local_optima_surrogate: int | None
    n_basins_surrogate: int | None
    local_optima_indices_surrogate: np.ndarray | None
    local_optima_coords_surrogate: np.ndarray | None
    reliability_warning: bool
    reliability_message: str


def _find_local_optima(
    X: np.ndarray,
    y: np.ndarray,
    direction: str,
    k: int,
) -> np.ndarray:
    """Find candidate local optima via KNN heuristic.

    A point is a candidate local optimum if its objective value is
    better than all of its K nearest neighbors.

    Args:
        X: Coordinates, shape (N, D).
        y: Objective values, shape (N,).
        direction: 'minimize' or 'maximize'.
        k: Number of neighbors.

    Returns:
        Indices of candidate local optima.
    """
    n = len(X)
    k_actual = min(k, n - 1)
    if k_actual < 1:
        return np.array([], dtype=int)

    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric="euclidean")
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    local_optima = []
    for i in range(n):
        neighbor_idx = indices[i, 1:]  # Exclude self
        if direction == "minimize":
            if np.all(y[i] < y[neighbor_idx]):
                local_optima.append(i)
        else:
            if np.all(y[i] > y[neighbor_idx]):
                local_optima.append(i)

    return np.array(local_optima, dtype=int)


def _count_basins(
    X: np.ndarray,
    optima_indices: np.ndarray,
    merge_radius: float | None = None,
) -> int:
    """Count distinct basins by clustering local optima.

    Uses a simple distance-based merging: two optima are in the same
    basin if they are within merge_radius of each other.

    Args:
        X: Coordinates, shape (N, D).
        optima_indices: Indices of candidate local optima.
        merge_radius: Distance threshold for merging. If None, uses
                      0.2 (in the coordinate space provided).

    Returns:
        Number of distinct basins.
    """
    if len(optima_indices) <= 1:
        return len(optima_indices)

    if merge_radius is None:
        merge_radius = 0.2

    X_optima = X[optima_indices]

    # Simple greedy clustering
    from scipy.spatial.distance import pdist, squareform

    dists = squareform(pdist(X_optima))
    visited = np.zeros(len(optima_indices), dtype=bool)
    n_basins = 0

    for i in range(len(optima_indices)):
        if visited[i]:
            continue
        n_basins += 1
        # Mark all optima within merge_radius as same basin
        queue = [i]
        while queue:
            curr = queue.pop()
            if visited[curr]:
                continue
            visited[curr] = True
            neighbors = np.where(
                (dists[curr] < merge_radius) & (~visited)
            )[0]
            queue.extend(neighbors.tolist())

    return n_basins


def detect_local_optima(
    X: np.ndarray,
    y: np.ndarray,
    direction: str,
    k: int = 5,
    surrogate_transform: SurrogateSpaceTransform | None = None,
) -> ModalityReport:
    """Detect candidate local optima via KNN heuristic.

    Computes in both flat and (optionally) surrogate space.
    Includes reliability gate: warns if N_OBS / N_DIM < 5.

    Args:
        X: Normalized inputs in flat space, shape (N, D).
        y: Objective values, shape (N,).
        direction: 'minimize' or 'maximize'.
        k: Number of neighbors.
        surrogate_transform: If provided, also analyze in surrogate space.

    Returns:
        ModalityReport with local optima candidates and basin counts.
    """
    n_obs, n_dim = X.shape
    reliability_warning = n_obs / max(n_dim, 1) < 5
    reliability_msg = ""
    if reliability_warning:
        reliability_msg = (
            f"Modality diagnostics may be unreliable: only {n_obs} observations "
            f"for {n_dim} dimensions (ratio={n_obs/max(n_dim,1):.1f}, need >= 5). "
            "Interpret with caution."
        )

    # Flat space
    optima_flat = _find_local_optima(X, y, direction, k)
    n_basins_flat = _count_basins(X, optima_flat)
    coords_flat = X[optima_flat] if len(optima_flat) > 0 else np.empty((0, n_dim))

    # Surrogate space
    n_opt_sur = None
    n_bas_sur = None
    idx_sur = None
    coords_sur = None
    if surrogate_transform is not None:
        X_sur = surrogate_transform.to_surrogate(X)
        optima_sur = _find_local_optima(X_sur, y, direction, k)
        n_opt_sur = len(optima_sur)
        n_bas_sur = _count_basins(X_sur, optima_sur)
        idx_sur = optima_sur
        coords_sur = X_sur[optima_sur] if len(optima_sur) > 0 else np.empty((0, n_dim))

    return ModalityReport(
        n_local_optima_flat=len(optima_flat),
        n_basins_flat=n_basins_flat,
        local_optima_indices_flat=optima_flat,
        local_optima_coords_flat=coords_flat,
        n_local_optima_surrogate=n_opt_sur,
        n_basins_surrogate=n_bas_sur,
        local_optima_indices_surrogate=idx_sur,
        local_optima_coords_surrogate=coords_sur,
        reliability_warning=reliability_warning,
        reliability_message=reliability_msg,
    )
