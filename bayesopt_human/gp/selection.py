"""Model selection workflow (compare transforms/warping)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from botorch.exceptions.errors import ModelFittingError

from bayesopt_human.config import OptimizationConfig
from bayesopt_human.gp.loo import analytical_loo_mae
from bayesopt_human.gp.model import FittedGP, GPModel
from bayesopt_human.transforms.output import OutputTransform
from bayesopt_human.transforms.warping import InputWarping

logger = logging.getLogger("bayesopt_human")


@dataclass
class ModelCandidate:
    """A candidate GP configuration with evaluation metrics."""

    output_transform: str
    warping: bool
    ard: bool
    loo_mae: float
    loo_std: float
    loo_lpd: float
    calibration_var: float
    fitted_gp: FittedGP
    censored_fraction: float = 0.0
    mean_r: float = float('nan')
    loo_lpd_i: np.ndarray | None = None
    loo_residuals: np.ndarray | None = None


def _clipped_log_eligible(y: np.ndarray, direction: str = "minimize") -> bool:
    """Check if clipped_log transform is eligible for this data.

    clipped_log censors all non-positive values to log(epsilon), collapsing
    them in surrogate space. This is only appropriate when:

    - **Maximizing**: extreme negative outliers dwarf the positive signal
      we're trying to optimize. The negatives are noise to suppress.
    - **Minimizing**: clipped_log censors negatives (where the optimum
      lives), so it is generally inappropriate. Only allowed when the
      objective is strictly positive with extreme positive outliers —
      i.e. clipped_log would function like log but is more robust.

    Requires at least 5 positive values plus a direction-dependent
    extremity check (100x ratio).
    """
    n_positive = int(np.sum(y > 0))
    if n_positive < 5:
        return False
    y_max = float(np.max(y))
    y_min = float(np.min(y))
    if y_max <= 0:
        return False

    if direction == "maximize":
        # We want positive peaks; negatives are noise to censor.
        # Require: negative extreme >> positive max.
        return abs(y_min) >= 100 * y_max
    else:
        # Minimizing: we want small values.  clipped_log censors
        # non-positives, which is only safe when ALL values are positive
        # (making it equivalent to log) and there are extreme positive
        # outliers dominating the range.  In practice, log already
        # handles this case, so we simply disallow clipped_log.
        return False


def length_scale_flags(
    candidate: ModelCandidate,
    ls_lower_bound: float = 1e-3,
    ls_upper_bound: float = 1e3,
    severe_ratio: float = 1e3,
    advisory_ratio: float = 100.0,
) -> tuple[list[str], str]:
    """Compute length-scale diagnostic flags and RAG status.

    Boundary detection uses log-space: a length scale is "at boundary"
    if it falls within 5% of the log-range from either bound.

    Returns:
        (flags, rag) where flags is a list of human-readable strings
        and rag is 'red', 'amber', or 'green'.
    """
    ls = getattr(candidate.fitted_gp, 'length_scales', None)
    if ls is None:
        return [], 'green'

    n_dims = len(ls)
    log_ls = np.log(np.clip(ls, 1e-30, None))
    log_lb = np.log(ls_lower_bound)
    log_ub = np.log(ls_upper_bound)
    log_range = log_ub - log_lb
    tol = 0.05 * log_range  # 5% of log-range, symmetric

    at_lower = log_ls <= (log_lb + tol)
    at_upper = log_ls >= (log_ub - tol)
    n_at_boundary = int(np.sum(at_lower | at_upper))

    max_ls = float(np.max(ls))
    min_ls = float(np.min(ls))
    ratio = max_ls / min_ls if min_ls > 0 else np.inf

    flags: list[str] = []
    ls_rag = 'green'

    if n_at_boundary > 0:
        if n_at_boundary >= (n_dims + 2) // 2:  # strict majority
            flags.append(f"LS: {n_at_boundary} at boundary")
            ls_rag = 'red'
        else:
            flags.append(f"LS: {n_at_boundary} at boundary")
            ls_rag = 'amber'

    if ratio > severe_ratio:
        flags.append("LS: max/min ratio > 1e3")
        if ls_rag != 'red':
            ls_rag = 'amber'
    elif ratio > advisory_ratio:
        flags.append("LS: max/min ratio > 100")
        if ls_rag == 'green':
            ls_rag = 'amber'

    return flags, ls_rag


def warp_flags(
    candidate: ModelCandidate,
    low_threshold: float = 0.1,
    high_threshold: float = 20.0,
) -> tuple[list[str], str]:
    """Compute warp-parameter diagnostic flags and RAG status.

    A Beta(a,b) CDF is the identity when a=b=1. Extreme warps occur when
    either concentration parameter is very small (steep/step-like transform)
    or very large (compresses the input range). Both cases can effectively
    collapse a dimension, similar to an extreme length scale.

    Args:
        candidate: A ModelCandidate (may or may not have warping).
        low_threshold: Flag dimension if either c0 or c1 < this value.
        high_threshold: Flag dimension if either c0 or c1 > this value.

    Returns:
        (flags, rag) where flags is a list of human-readable strings
        and rag is 'red', 'amber', or 'green'.
    """
    if not candidate.warping:
        return [], 'green'

    wp = getattr(candidate.fitted_gp, 'warp_parameters', None)
    if wp is None:
        return [], 'green'

    c0 = wp['concentration0']
    c1 = wp['concentration1']
    n_dims = len(c0)

    n_extreme = 0
    for i in range(n_dims):
        if (c0[i] < low_threshold or c0[i] > high_threshold
                or c1[i] < low_threshold or c1[i] > high_threshold):
            n_extreme += 1

    flags: list[str] = []
    wrp_rag = 'green'

    if n_extreme > 0:
        flags.append(f"WRP: {n_extreme} extreme")
        if n_extreme >= (n_dims + 2) // 2:  # strict majority
            wrp_rag = 'red'
        else:
            wrp_rag = 'amber'

    return flags, wrp_rag


def model_n_params(candidate: ModelCandidate) -> int:
    """Count effective GP hyperparameters for a model candidate.

    ISO:  1 length scale + 1 outputscale + 1 noise = 3
    ARD:  D length scales + 1 outputscale + 1 noise = D + 2
    WARP: D length scales + 2D warp params + 1 outputscale + 1 noise = 3D + 2
    """
    n_dim = len(candidate.fitted_gp.length_scales)
    if candidate.warping:
        return 3 * n_dim + 2
    elif candidate.ard:
        return n_dim + 2
    else:
        return 3


class ModelSelector:
    """Evaluates and compares GP configurations.

    Tests output transforms (with ARD and non-ARD kernels) and
    optionally input warping, ranking by LOO-LPD.
    """

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config
        self._gp_model = GPModel(config)

    def evaluate_transforms(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[ModelCandidate]:
        """Fit GP with each output transform x kernel type, compute LOO-MAE.

        Evaluates all transforms with both ARD and non-ARD (isotropic) kernels.
        Skips clipped_log when data has insufficient positive values.

        Returns list of ModelCandidate objects sorted by LOO-LPD descending.
        """
        candidates: list[ModelCandidate] = []
        clipped_eligible = _clipped_log_eligible(y, self.config.direction)
        clipped_censored_frac = float(np.sum(y <= 0)) / len(y) if len(y) > 0 else 0.0

        for ard in [True, False]:
            kernel_label = "ARD" if ard else "iso"
            for tf_name in OutputTransform.VALID_TRANSFORMS:
                if tf_name == "clipped_log" and not clipped_eligible:
                    logger.info(
                        "Skipping clipped_log (%s): eligibility not met.",
                        kernel_label,
                    )
                    continue

                try:
                    fitted = self._gp_model.fit(
                        X, y, output_transform=tf_name, warping=False, ard=ard
                    )
                    loo_mae, loo_std, loo_lpd, cal_var, mean_r = analytical_loo_mae(fitted, X, y)
                    # Compute loo_residuals for diagnostics
                    from bayesopt_human.gp.loo import analytical_loo_details
                    loo_details = analytical_loo_details(fitted, X, y)
                    loo_residuals = loo_details.get("residuals", None)
                    cf = clipped_censored_frac if tf_name == "clipped_log" else 0.0
                    mc = ModelCandidate(
                        output_transform=tf_name,
                        warping=False,
                        ard=ard,
                        loo_mae=loo_mae,
                        loo_std=loo_std,
                        loo_lpd=loo_lpd,
                        calibration_var=cal_var,
                        fitted_gp=fitted,
                        censored_fraction=cf,
                        mean_r=mean_r,
                        loo_lpd_i=loo_details.get("lpd_i", None),
                    )
                    mc.loo_residuals = loo_residuals
                    candidates.append(mc)
                    logger.info(
                        "Transform %r (%s): LOO-MAE = %.4f, LPD = %.4f, Var(r) = %.3f",
                        tf_name, kernel_label, loo_mae, loo_lpd, cal_var,
                    )
                except ValueError as e:
                    # Expected: data incompatible with transform (e.g. mixed signs for log)
                    logger.debug(
                        "Transform %r (%s) skipped: %s", tf_name, kernel_label, e
                    )
                except (RuntimeError, ModelFittingError) as e:
                    logger.warning(
                        "Transform %r (%s) failed: %s", tf_name, kernel_label, e
                    )

        candidates.sort(key=lambda c: -c.loo_lpd)

        return candidates

    def evaluate_warping(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transform: str,
        ard: bool = True,
    ) -> ModelCandidate | None:
        """Test input warping for a given transform.

        Returns None if insufficient data (N_OBS < min_ratio * N_DIM).
        """
        n_obs, n_dim = X.shape
        min_ratio = self.config.warping_min_ratio
        if not InputWarping.sufficient_data(n_obs, n_dim, min_ratio):
            logger.info(
                "Insufficient data for warping (%d obs, %d dims, need %d).",
                n_obs, n_dim, min_ratio * n_dim,
            )
            return None

        try:
            fitted = self._gp_model.fit(
                X, y, output_transform=transform, warping=True, ard=ard
            )
            loo_mae, loo_std, loo_lpd, cal_var, mean_r = analytical_loo_mae(fitted, X, y)
            from bayesopt_human.gp.loo import analytical_loo_details
            loo_details = analytical_loo_details(fitted, X, y)
            loo_residuals = loo_details.get("residuals", None)
            kernel_label = "ARD" if ard else "iso"
            logger.info(
                "Transform %r + warping (%s): LOO-MAE = %.4f, Var(r) = %.3f",
                transform, kernel_label, loo_mae, cal_var,
            )
            mc = ModelCandidate(
                output_transform=transform,
                warping=True,
                ard=ard,
                loo_mae=loo_mae,
                loo_std=loo_std,
                loo_lpd=loo_lpd,
                calibration_var=cal_var,
                fitted_gp=fitted,
                mean_r=mean_r,
                loo_lpd_i=loo_details.get("lpd_i", None),
            )
            mc.loo_residuals = loo_residuals
            return mc
        except ValueError as e:
            logger.debug("Warping for %r skipped: %s", transform, e)
        except (RuntimeError, ModelFittingError) as e:
            logger.warning("Warping evaluation failed for %r: %s", transform, e)
            return None
