"""GP model construction and fitting."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import math

import numpy as np
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Warp
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior
from linear_operator.utils.errors import NotPSDError

from bayesopt_human.config import OptimizationConfig
from bayesopt_human.gp.jitter import AdaptiveJitter
from bayesopt_human.transforms.normalization import IQRNormalizer
from bayesopt_human.transforms.output import OutputTransform
from bayesopt_human.transforms.warping import InputWarping

logger = logging.getLogger("bayesopt_human")


@dataclass
class FittedGP:
    """Container for a fitted GP and its associated transforms."""

    model: SingleTaskGP
    output_transform: OutputTransform
    iqr_normalizer: IQRNormalizer
    input_warping: InputWarping | None
    warp_transform: Warp | None
    jitter_used: float
    length_scales: np.ndarray
    warp_parameters: dict | None
    conditioning: float
    conditioning_warning: bool
    ard: bool = True

    def predict(
        self, X_new: np.ndarray, return_std: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Posterior mean (and optionally std) at new points.

        Predictions are returned in surrogate space (IQR-normalized,
        output-transformed) which is the natural scale for model diagnostics.
        """
        self.model.eval()
        X_t = torch.tensor(X_new, dtype=torch.float64)
        with torch.no_grad():
            posterior = self.model.posterior(X_t)
            mu = posterior.mean.squeeze(-1).cpu().numpy()
            if not return_std:
                return mu
            sigma = posterior.variance.squeeze(-1).sqrt().cpu().numpy()
        return mu, sigma


class GPModel:
    """Constructs and fits a GP model with Matern 5/2 kernel.

    Wraps BoTorch's SingleTaskGP with the configured output transform,
    IQR normalization, optional input warping, and adaptive jitter.
    Supports both ARD (per-dimension) and isotropic length scales.
    """

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        output_transform: str = "none",
        warping: bool = False,
        ard: bool = True,
    ) -> FittedGP:
        """Fit GP to observations.

        Args:
            X: Normalized input coordinates, shape (N_OBS, N_DIM).
            y: Raw objective values, shape (N_OBS,).
            output_transform: Which output transform to apply.
            warping: Whether to enable input warping.
            ard: Whether to use ARD (per-dimension length scales).

        Returns:
            FittedGP containing the fitted model, transforms, and metadata.
        """
        n_obs, n_dim = X.shape

        # Apply output transform
        ot = OutputTransform(output_transform, self.config.clipped_log_epsilon)
        y_transformed = ot.forward(y)

        # IQR normalization
        iqr = IQRNormalizer().fit(y_transformed)
        y_norm = iqr.transform(y_transformed)

        # Adaptive jitter
        jitter_mgr = AdaptiveJitter(
            base=self.config.jitter_base,
            ceiling_factor=self.config.jitter_ceiling_factor,
        )
        jitter = jitter_mgr.initial_jitter(y_norm)

        # Convert to torch
        train_X = torch.tensor(X, dtype=torch.float64)
        train_Y = torch.tensor(y_norm, dtype=torch.float64).unsqueeze(-1)

        # Input warping
        iw: InputWarping | None = None
        warp_tf: Warp | None = None
        if warping:
            iw = InputWarping(n_dim)
            warp_tf = iw.create_warp_transform(
                prior_median=self.config.warp_prior_median,
                prior_scale=self.config.warp_prior_scale,
            )

        # Fit with Cholesky fallback loop
        model = self._fit_with_jitter(
            train_X, train_Y, jitter, jitter_mgr, y_norm, warp_tf, ard=ard
        )

        # Extract learned parameters
        length_scales_raw = (
            model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()
        )
        # Non-ARD has a single shared length scale — broadcast to all dims
        if not ard:
            length_scales = np.full(n_dim, length_scales_raw[0])
        else:
            length_scales = length_scales_raw

        warp_params = None
        if warp_tf is not None:
            warp_params = InputWarping.get_learned_parameters(warp_tf)

        # Conditioning: log determinant of kernel matrix
        model.eval()
        with torch.no_grad():
            K = model.covar_module(train_X).to_dense()
            noise = jitter_mgr.current_jitter * torch.eye(
                n_obs, dtype=torch.float64
            )
            K_noisy = K + noise
            try:
                L = torch.linalg.cholesky(K_noisy)
                log_det = 2.0 * torch.sum(torch.log(torch.diag(L))).item()
            except RuntimeError:
                # Cholesky raises ``torch._C._LinAlgError`` (a RuntimeError
                # subclass) when the kernel is not PSD. Flag by setting
                # log_det to +inf so diagnostics show a conditioning warning.
                log_det = float("inf")

        return FittedGP(
            model=model,
            output_transform=ot,
            iqr_normalizer=iqr,
            input_warping=iw,
            warp_transform=warp_tf,
            jitter_used=jitter_mgr.current_jitter,
            length_scales=length_scales,
            warp_parameters=warp_params,
            conditioning=log_det,
            conditioning_warning=jitter_mgr.conditioning_warning,
            ard=ard,
        )

    def _fit_with_jitter(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        jitter: float,
        jitter_mgr: AdaptiveJitter,
        y_norm: np.ndarray,
        warp_tf: Warp | None,
        ard: bool = True,
    ) -> SingleTaskGP:
        """Try fitting, escalating jitter on Cholesky failures."""
        n_dim = train_X.shape[1]
        max_attempts = 5
        for attempt in range(max_attempts):
            train_Yvar = torch.full_like(train_Y, jitter)

            ls_prior = LogNormalPrior(
                math.log(self.config.length_scale_prior_median),
                self.config.length_scale_prior_scale,
            )
            base_kernel = (
                MaternKernel(nu=2.5, ard_num_dims=n_dim, lengthscale_prior=ls_prior)
                if ard
                else MaternKernel(nu=2.5, lengthscale_prior=ls_prior)
            )
            covar = ScaleKernel(base_kernel)

            kwargs = {"covar_module": covar}
            if warp_tf is not None:
                kwargs["input_transform"] = warp_tf

            model = SingleTaskGP(
                train_X, train_Y, train_Yvar, **kwargs
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            try:
                fit_gpytorch_mll(mll)
                logger.info(
                    "GP fitted successfully (jitter=%.2e, attempt=%d).",
                    jitter, attempt + 1,
                )
                return model
            except (NotPSDError, RuntimeError, ModelFittingError) as e:
                if isinstance(e, ModelFittingError) or "not positive definite" in str(e).lower() or isinstance(
                    e, NotPSDError
                ):
                    new_jitter = jitter_mgr.increase(y_norm)
                    if new_jitter is None or jitter_mgr.conditioning_warning:
                        # Use ceiling jitter and try one more time
                        jitter = jitter_mgr.current_jitter
                        if attempt == max_attempts - 1:
                            raise
                    else:
                        jitter = new_jitter
                else:
                    raise

        raise RuntimeError("GP fitting failed after maximum jitter attempts.")

    def predict(
        self, fitted_gp: FittedGP, X_new: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation at new points.

        Returns predictions in the raw objective scale (inverse-transformed).
        """
        model = fitted_gp.model
        model.eval()
        X_new_t = torch.tensor(X_new, dtype=torch.float64)

        with torch.no_grad():
            posterior = model.posterior(X_new_t)
            mean_norm = posterior.mean.squeeze(-1).cpu().numpy()
            std_norm = posterior.variance.squeeze(-1).sqrt().cpu().numpy()

        # Inverse IQR normalization
        mean_transformed = fitted_gp.iqr_normalizer.inverse_transform(mean_norm)
        # For std, scale by IQR (since IQR normalization is linear)
        std_transformed = std_norm * fitted_gp.iqr_normalizer.iqr_

        # Inverse output transform for mean
        mean_raw = fitted_gp.output_transform.inverse(mean_transformed)

        return mean_raw, std_transformed
