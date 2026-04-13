"""Beta-CDF input warping (wraps BoTorch Warp)."""

from __future__ import annotations

import math

import torch
from botorch.models.transforms.input import Warp
from gpytorch.priors import LogNormalPrior


class InputWarping:
    """Manages Beta-CDF input warping via BoTorch's Warp transform.

    Wraps BoTorch's Warp input transform, providing a clean interface
    for enabling/disabling warping and extracting learned parameters.
    """

    def __init__(self, n_dims: int) -> None:
        """
        Args:
            n_dims: Number of input dimensions.
        """
        self.n_dims = n_dims

    def create_warp_transform(
        self,
        prior_median: float = 1.0,
        prior_scale: float = 0.75,
    ) -> Warp:
        """Create a BoTorch Warp input transform with configurable priors.

        Args:
            prior_median: Median of LogNormal prior on concentration parameters.
                1.0 centers the prior on the identity warp.
            prior_scale: Scale (spread) of LogNormal prior.
                Lower values = stronger regularization toward the median.
        """
        prior = LogNormalPrior(math.log(prior_median), prior_scale)
        return Warp(
            d=self.n_dims,
            indices=list(range(self.n_dims)),
            concentration0_prior=prior,
            concentration1_prior=prior,
        )

    @staticmethod
    def get_learned_parameters(warp: Warp) -> dict:
        """Extract concentration0 and concentration1 per dimension.

        Returns:
            Dict with keys 'concentration0' and 'concentration1',
            each a numpy array of shape (n_dims,).
        """
        c0 = warp.concentration0.detach().cpu().numpy().flatten()
        c1 = warp.concentration1.detach().cpu().numpy().flatten()
        return {"concentration0": c0, "concentration1": c1}

    @staticmethod
    def sufficient_data(n_obs: int, n_dims: int, min_ratio: int = 10) -> bool:
        """Check if N_OBS >= min_ratio * N_DIM for reliable warping."""
        return n_obs >= min_ratio * n_dims

    @staticmethod
    def apply_warp(warp: Warp, X: torch.Tensor) -> torch.Tensor:
        """Apply the warp transform to input coordinates.

        Args:
            warp: A fitted BoTorch Warp transform.
            X: Input tensor, shape (..., n_dims).

        Returns:
            Warped coordinates.
        """
        warp.eval()
        with torch.no_grad():
            return warp(X)
