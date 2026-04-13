"""Analytical LOO-MAE computation."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import torch

from bayesopt_human.gp.model import FittedGP

logger = logging.getLogger("bayesopt_human")


def _compute_loo_internals(
    fitted_gp: FittedGP,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared LOO computation returning (mu_loo, K_inv_diag, y_t).

    All returned tensors are in IQR-normalized space.
    """
    model = fitted_gp.model
    model.eval()

    y_transformed = fitted_gp.output_transform.forward(y)
    y_norm = fitted_gp.iqr_normalizer.transform(y_transformed)

    train_X = torch.tensor(X, dtype=torch.float64)

    with torch.no_grad():
        K = model.covar_module(train_X).to_dense()
        noise_var = fitted_gp.jitter_used
        K_noisy = K + noise_var * torch.eye(len(X), dtype=torch.float64)

        K_inv = torch.linalg.inv(K_noisy)

        y_t = torch.tensor(y_norm, dtype=torch.float64)
        K_inv_y = K_inv @ y_t
        K_inv_diag = torch.diag(K_inv)

        mu_loo = y_t - K_inv_y / K_inv_diag

    return mu_loo, K_inv_diag, y_t


def analytical_loo_mae(
    fitted_gp: FittedGP,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Compute analytical leave-one-out MAE, LPD, and calibration variance.

    Uses the standard LOO formulas:
        mu_LOO_i = y_i - (K^{-1} y)_i / (K^{-1})_{ii}
        sigma2_LOO_i = 1 / (K^{-1})_{ii}

    Calibration variance: Var(r) where r_i = (y_i - mu_LOO_i) / sigma_LOO_i.
    For a well-calibrated model, Var(r) ≈ 1.

    Returns (mae, std, lpd, calibration_var, mean_r):
        - mae: mean absolute LOO error (normalized space)
        - std: standard deviation of absolute LOO errors
        - lpd: mean log predictive density
        - calibration_var: variance of standardized LOO residuals
        - mean_r: mean raw (signed) LOO residual
    """
    mu_loo, K_inv_diag, y_t = _compute_loo_internals(fitted_gp, X, y)

    residuals = y_t - mu_loo
    loo_errors = torch.abs(residuals).cpu().numpy()
    mae = float(np.mean(loo_errors))
    std = float(np.std(loo_errors))
    mean_r = float(np.mean(residuals.cpu().numpy()))

    # Log predictive density
    sigma2_loo = torch.clamp(1.0 / K_inv_diag, min=1e-12)
    lpd_i = -0.5 * torch.log(2 * np.pi * sigma2_loo) - 0.5 * residuals**2 / sigma2_loo
    lpd = float(lpd_i.mean().cpu().item())

    # Standardized residuals and calibration variance
    sigma_loo = torch.sqrt(sigma2_loo)
    standardized = residuals / sigma_loo
    calibration_var = float(torch.var(standardized).cpu().item())

    return mae, std, lpd, calibration_var, mean_r


def analytical_loo_details(
    fitted_gp: FittedGP,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute detailed LOO predictions for diagnostic plots.

    Returns dict with numpy arrays:
        mu_loo: LOO predicted means (normalized/surrogate space)
        sigma_loo: LOO predicted standard deviations
        residuals: y_norm - mu_loo (surrogate space)
        residuals_raw: y - mu_loo_raw (raw/original space)
        standardized_residuals: residuals / sigma_loo (should be ~N(0,1))
        calibration_var: Var(standardized_residuals) (should be ~1.0)
        lpd_i: per-point log predictive density (for paired model comparison)
        y_norm: normalized targets
    """
    mu_loo, K_inv_diag, y_t = _compute_loo_internals(fitted_gp, X, y)

    sigma2_loo = torch.clamp(1.0 / K_inv_diag, min=1e-12)
    sigma_loo = torch.sqrt(sigma2_loo)
    residuals = y_t - mu_loo

    # Standardized residuals
    standardized = residuals / sigma_loo

    # Per-point log predictive density
    lpd_i = -0.5 * torch.log(2 * np.pi * sigma2_loo) - 0.5 * residuals**2 / sigma2_loo

    # Invert transforms to get raw-space LOO predictions
    mu_loo_np = mu_loo.cpu().numpy()
    mu_loo_transformed = fitted_gp.iqr_normalizer.inverse_transform(mu_loo_np)
    mu_loo_raw = fitted_gp.output_transform.inverse(mu_loo_transformed)
    residuals_raw = y - mu_loo_raw

    return {
        "mu_loo": mu_loo_np,
        "sigma_loo": sigma_loo.cpu().numpy(),
        "residuals": residuals.cpu().numpy(),
        "residuals_raw": residuals_raw,
        "standardized_residuals": standardized.cpu().numpy(),
        "calibration_var": float(torch.var(standardized).cpu().item()),
        "lpd_i": lpd_i.cpu().numpy(),
        "y_norm": y_t.cpu().numpy(),
    }


def loo_parameter_stability(
    model_fit_fn: Callable,
    X: np.ndarray,
    y: np.ndarray,
    **kwargs,
) -> dict:
    """Compute LOO standard deviation of hyperparameters.

    EXPENSIVE: Refits the GP N times. Gate behind explicit user request.

    Args:
        model_fit_fn: Callable that takes (X, y) and returns a FittedGP.
        X: Normalized inputs, shape (N, D).
        y: Raw objectives, shape (N,).
        **kwargs: Additional arguments passed to model_fit_fn.

    Returns:
        Dict mapping parameter names to their LOO standard deviations.
    """
    n = len(X)
    logger.warning(
        "Running LOO parameter stability analysis (%d refits). "
        "This may take a while for large datasets.",
        n,
    )

    length_scales_all = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_loo = X[mask]
        y_loo = y[mask]
        try:
            fitted = model_fit_fn(X_loo, y_loo, **kwargs)
            length_scales_all.append(fitted.length_scales)
        except (RuntimeError, ArithmeticError, ValueError) as e:
            # Individual LOO folds can fail when the reduced data set is
            # ill-conditioned (Cholesky / numerical errors) or when botorch
            # raises ``ModelFittingError`` (a RuntimeError subclass).
            # Skip the fold and continue so the diagnostic still reports the
            # variance of the folds that did fit.
            logger.warning("LOO fold %d failed: %s", i, e)
            continue

    n_dim = X.shape[1]
    if len(length_scales_all) < 2:
        return {
            "length_scales_std": np.full(n_dim, float("nan")),
            "length_scales_mean": np.full(n_dim, float("nan")),
            "length_scales_summary": [
                f"l_{i+1} = nan \u00b1 nan" for i in range(n_dim)
            ],
            "n_successful_folds": len(length_scales_all),
        }

    ls_array = np.array(length_scales_all)
    means = np.mean(ls_array, axis=0)
    stds = np.std(ls_array, axis=0)
    summary = [
        f"l_{i+1} = {means[i]:.4f} \u00b1 {stds[i]:.4f}" for i in range(n_dim)
    ]
    return {
        "length_scales_std": stds,
        "length_scales_mean": means,
        "length_scales_summary": summary,
        "n_successful_folds": len(length_scales_all),
    }
