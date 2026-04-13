"""Shared utilities for BayesOpt-Human."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("bayesopt_human")


def apply_seed(seed: int | None) -> None:
    """Reseed torch and numpy global RNGs from a single integer seed.

    ``None`` is a no-op, so callers can pass
    ``OptimizationConfig.random_seed`` directly and let the user opt out of
    determinism by setting it to ``None``. Seeding is deliberately scoped
    to the global RNGs because the downstream callers
    (``botorch.optim.optimize_acqf``, ``fit_gpytorch_mll``, ``torch.rand``,
    ``sklearn`` estimators) all pull from them when no explicit RNG is
    provided.
    """
    if seed is None:
        return
    # Import inside the function to avoid making torch a hard dependency
    # of ``utils`` at import time; utils is imported from places that load
    # before torch in some tests.
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure package-level logging."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)


def stagnation_length(
    y: np.ndarray, direction: str, warmstart: int = 0,
) -> int:
    """Compute the number of consecutive observations without improvement.

    When ``warmstart > 0``, only the optimization-phase observations
    (``y[warmstart:]``) are considered.

    Args:
        y: Objective values in observation order.
        direction: 'minimize' or 'maximize'.
        warmstart: Number of warmstart observations to skip.

    Returns:
        Number of consecutive observations since last improvement
        (within the optimization phase).
    """
    y_opt = y[warmstart:]
    if len(y_opt) <= 1:
        return 0

    if direction == "minimize":
        best_so_far = np.minimum.accumulate(y_opt)
    else:
        best_so_far = np.maximum.accumulate(y_opt)

    # Find indices where improvement occurred
    improvements = np.where(np.diff(best_so_far) != 0)[0] + 1
    if len(improvements) == 0:
        return len(y_opt) - 1
    return len(y_opt) - 1 - improvements[-1]


def format_value(value: float, precision: int = 4) -> str:
    """Format a float for display."""
    if abs(value) < 1e-3 or abs(value) > 1e4:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"
