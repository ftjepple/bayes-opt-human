"""Flat-to-surrogate and surrogate-to-flat transforms."""

from __future__ import annotations

import numpy as np
import torch

from bayesopt_human.gp.model import FittedGP
from bayesopt_human.transforms.warping import InputWarping


class SurrogateSpaceTransform:
    """Transforms between flat (normalized) space and surrogate space.

    Surrogate space: x_sur_i = w_i(x_flat_i) / l_i
    where w_i is the warp function and l_i is the length scale.
    """

    def __init__(self, fitted_gp: FittedGP) -> None:
        """Initialize from a fitted GP (requires learned parameters)."""
        self.length_scales = fitted_gp.length_scales.copy()
        self.warp_transform = fitted_gp.warp_transform
        self.has_warping = self.warp_transform is not None

    def to_surrogate(self, X_flat: np.ndarray) -> np.ndarray:
        """Transform flat-space coordinates to surrogate space.

        Args:
            X_flat: Coordinates in flat (normalized) space, shape (N, D) or (D,).

        Returns:
            Coordinates in surrogate space.
        """
        squeeze = False
        if X_flat.ndim == 1:
            X_flat = X_flat.reshape(1, -1)
            squeeze = True

        X_flat = np.asarray(X_flat, dtype=np.float64)

        # Apply warping if available
        if self.has_warping:
            X_t = torch.tensor(X_flat, dtype=torch.float64)
            self.warp_transform.eval()
            with torch.no_grad():
                X_warped = self.warp_transform(X_t).cpu().numpy()
        else:
            X_warped = X_flat

        # Divide by length scales
        X_surrogate = X_warped / self.length_scales

        return X_surrogate[0] if squeeze else X_surrogate

    def flat_to_surrogate_distances(self, X_flat: np.ndarray) -> np.ndarray:
        """Compute pairwise distances in surrogate space from flat-space points.

        Args:
            X_flat: Flat-space coordinates, shape (N, D).

        Returns:
            Pairwise distance matrix, shape (N, N).
        """
        X_sur = self.to_surrogate(X_flat)
        from scipy.spatial.distance import pdist, squareform

        return squareform(pdist(X_sur))
