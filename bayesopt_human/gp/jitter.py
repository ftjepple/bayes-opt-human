"""Adaptive jitter logic with Cholesky fallback."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("bayesopt_human")


class AdaptiveJitter:
    """Manages adaptive regularization jitter for GP fitting.

    The jitter adapts to the scale of the transformed, normalized targets:
    - Initial: max(base, 1e-4 * var(y))
    - Cholesky fallback: multiply by 10 on NotPSDError
    - Ceiling: ceiling_factor * var(y)
    """

    def __init__(
        self,
        base: float = 1e-6,
        ceiling_factor: float = 1e-2,
    ) -> None:
        self.base = base
        self.ceiling_factor = ceiling_factor
        self.current_jitter: float | None = None
        self.conditioning_warning = False

    def initial_jitter(self, y_normalized: np.ndarray) -> float:
        """Compute initial jitter from normalized target variance."""
        var_y = float(np.var(y_normalized))
        jitter = max(self.base, 1e-4 * var_y)
        self.current_jitter = jitter
        self.conditioning_warning = False
        return jitter

    def ceiling(self, y_normalized: np.ndarray) -> float:
        """Compute the hard jitter ceiling."""
        var_y = float(np.var(y_normalized))
        return self.ceiling_factor * max(var_y, 1e-10)

    def increase(self, y_normalized: np.ndarray) -> float | None:
        """Increase jitter by 10x after a Cholesky failure.

        Returns:
            New jitter value, or None if ceiling is reached.
        """
        if self.current_jitter is None:
            raise RuntimeError("Must call initial_jitter before increase.")
        new_jitter = self.current_jitter * 10.0
        ceil = self.ceiling(y_normalized)
        if new_jitter > ceil:
            self.conditioning_warning = True
            logger.warning(
                "Jitter ceiling reached (%.2e). GP conditioning is poor.",
                ceil,
            )
            self.current_jitter = ceil
            return ceil
        logger.info("Increasing jitter to %.2e after Cholesky failure.", new_jitter)
        self.current_jitter = new_jitter
        return new_jitter
