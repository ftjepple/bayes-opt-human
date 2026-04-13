"""GP fit diagnostics: LOO-MAE, conditioning, length scale analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bayesopt_human.config import OptimizationConfig
from bayesopt_human.gp.loo import analytical_loo_mae
from bayesopt_human.gp.model import FittedGP


@dataclass
class GPFitReport:
    """Container for GP fit diagnostic results."""

    loo_mae: float
    loo_std: float
    loo_lpd: float
    calibration_var: float
    conditioning: float
    jitter_used: float
    conditioning_warning: bool
    length_scales: np.ndarray
    feature_importances: np.ndarray
    length_scale_flags: dict[int, str] = field(default_factory=dict)
    warp_parameters: dict | None = None
    output_transform: str = "none"
    warping_enabled: bool = False


def gp_fit_summary(
    fitted_gp: FittedGP,
    X: np.ndarray,
    y: np.ndarray,
    length_scale_bounds: tuple[float, float] | None = None,
) -> GPFitReport:
    """Compute full GP fit diagnostics.

    Args:
        fitted_gp: A fitted GP model.
        X: Normalized inputs, shape (N, D).
        y: Raw objective values, shape (N,).
        length_scale_bounds: (lower, upper) bounds for length scale flagging.

    Returns:
        GPFitReport containing LOO-MAE, conditioning, jitter used,
        per-dimension length scales, feature importances, and
        boundary flags for length scales.
    """
    loo_mae, loo_std, loo_lpd, cal_var, _mean_r = analytical_loo_mae(fitted_gp, X, y)

    ls = fitted_gp.length_scales
    feature_imp = 1.0 / np.clip(ls, 1e-10, None)

    # Flag length scales near bounds
    flags: dict[int, str] = {}
    if length_scale_bounds is None:
        lo = OptimizationConfig.length_scale_lower_bound
        hi = OptimizationConfig.length_scale_upper_bound
    else:
        lo, hi = length_scale_bounds
    for i, l in enumerate(ls):
        if l < lo * 2:
            flags[i] = "near_lower"
        elif l > hi * 0.5:
            flags[i] = "near_upper"

    return GPFitReport(
        loo_mae=loo_mae,
        loo_std=loo_std,
        loo_lpd=loo_lpd,
        calibration_var=cal_var,
        conditioning=fitted_gp.conditioning,
        jitter_used=fitted_gp.jitter_used,
        conditioning_warning=fitted_gp.conditioning_warning,
        length_scales=ls,
        feature_importances=feature_imp,
        length_scale_flags=flags,
        warp_parameters=fitted_gp.warp_parameters,
        output_transform=fitted_gp.output_transform.transform_type,
        warping_enabled=fitted_gp.warp_transform is not None,
    )
