"""Acquisition functions: EI, UCB/LCB, MPV, Max Mean (wraps BoTorch)."""

from __future__ import annotations

import torch
from botorch.acquisition import (
    LogExpectedImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.analytic import PosteriorMean

from bayesopt_human.gp.model import FittedGP


def compute_ei(
    fitted_gp: FittedGP,
    best_value: float,
    direction: str,
) -> LogExpectedImprovement:
    """Construct Expected Improvement acquisition function.

    Uses LogExpectedImprovement for better numerical stability.

    Args:
        fitted_gp: Fitted GP model.
        best_value: Best observed value (in IQR-normalized space).
        direction: 'minimize' or 'maximize'.

    Returns:
        BoTorch LogExpectedImprovement acquisition function.
    """
    model = fitted_gp.model
    model.eval()
    maximize = direction == "maximize"
    return LogExpectedImprovement(
        model=model,
        best_f=best_value,
        maximize=maximize,
    )


def compute_ucb_lcb(
    fitted_gp: FittedGP,
    kappa: float,
    direction: str,
) -> UpperConfidenceBound:
    """Construct UCB (maximize) or LCB (minimize) acquisition function.

    When minimizing, we negate the model and use UCB, which is equivalent to LCB.
    BoTorch's UpperConfidenceBound supports this via the maximize flag.

    Args:
        fitted_gp: Fitted GP model.
        kappa: Confidence bound parameter (beta in BoTorch = kappa^2).
        direction: 'minimize' or 'maximize'.

    Returns:
        BoTorch UpperConfidenceBound acquisition function.
    """
    model = fitted_gp.model
    model.eval()
    # BoTorch UCB uses beta (= kappa^2 for standard UCB formulation)
    # but actually BoTorch uses beta directly as the multiplier on sigma
    maximize = direction == "maximize"
    return UpperConfidenceBound(
        model=model,
        beta=kappa ** 2 if not maximize else kappa ** 2,
        maximize=maximize,
    )


def compute_mpv(fitted_gp: FittedGP) -> UpperConfidenceBound:
    """Construct Max Posterior Variance acquisition function.

    We use UCB with beta set very high and maximize=True to effectively
    maximize posterior variance. Alternatively we implement a custom
    acquisition that directly targets variance.

    Actually, BoTorch doesn't have a dedicated MPV. We implement it
    as a simple callable that returns posterior variance.
    """
    return _PosteriorVariance(fitted_gp.model)


class _PosteriorVariance:
    """Custom acquisition function that returns posterior variance."""

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            posterior = self.model.posterior(X)
            # Shape: (batch, q, 1) -> (batch,)
            variance = posterior.variance.squeeze(-1).squeeze(-1)
        return variance


def compute_max_mean(
    fitted_gp: FittedGP,
    direction: str,
) -> PosteriorMean:
    """Construct acquisition function that optimizes the GP posterior mean.

    Args:
        fitted_gp: Fitted GP model.
        direction: 'minimize' or 'maximize'.

    Returns:
        BoTorch PosteriorMean acquisition function.
    """
    model = fitted_gp.model
    model.eval()
    maximize = direction == "maximize"
    acqf = PosteriorMean(model=model, maximize=maximize)
    return acqf
