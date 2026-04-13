"""Bounds-based normalization and denormalization."""

from __future__ import annotations

import numpy as np


class BoundsNormalizer:
    """Normalizes raw parameter values to [0, 1) using provided bounds.

    The library owns normalization. Users provide raw values and bounds;
    all internal operations use normalized coordinates.
    """

    def __init__(self, bounds: dict[str, tuple[float, float]]) -> None:
        """
        Args:
            bounds: Dict mapping parameter names to (lower, upper) bounds.

        Raises:
            ValueError: If any lower >= upper.
        """
        self.param_names = list(bounds.keys())
        self._lower = np.array([bounds[n][0] for n in self.param_names])
        self._upper = np.array([bounds[n][1] for n in self.param_names])
        self._range = self._upper - self._lower

        if np.any(self._range <= 0):
            bad = [
                n for n, r in zip(self.param_names, self._range) if r <= 0
            ]
            raise ValueError(
                f"Lower bound must be strictly less than upper bound for: {bad}"
            )

    @property
    def n_dim(self) -> int:
        return len(self.param_names)

    @property
    def lower(self) -> np.ndarray:
        return self._lower.copy()

    @property
    def upper(self) -> np.ndarray:
        return self._upper.copy()

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize raw coordinates to [0, 1).

        Args:
            X: Raw coordinates, shape (N, D) or (D,).

        Returns:
            Normalized coordinates in [0, 1).

        Raises:
            ValueError: If any value falls outside bounds.
        """
        X = np.asarray(X, dtype=np.float64)
        squeeze = False
        if X.ndim == 1:
            X = X.reshape(1, -1)
            squeeze = True

        X_norm = (X - self._lower) / self._range

        if np.any(X_norm < 0) or np.any(X_norm >= 1.0):
            # Find offending points
            below = np.any(X_norm < 0, axis=1)
            above = np.any(X_norm >= 1.0, axis=1)
            bad_rows = np.where(below | above)[0]
            raise ValueError(
                f"Observations at indices {bad_rows.tolist()} fall outside "
                f"the provided bounds. Ensure all values are in [lower, upper)."
            )

        return X_norm[0] if squeeze else X_norm

    def denormalize(self, X_norm: np.ndarray) -> np.ndarray:
        """Convert normalized coordinates back to raw space.

        Args:
            X_norm: Normalized coordinates, shape (N, D) or (D,).

        Returns:
            Raw-space coordinates.
        """
        X_norm = np.asarray(X_norm, dtype=np.float64)
        squeeze = False
        if X_norm.ndim == 1:
            X_norm = X_norm.reshape(1, -1)
            squeeze = True

        X_raw = X_norm * self._range + self._lower
        return X_raw[0] if squeeze else X_raw

    def botorch_bounds(self) -> np.ndarray:
        """Return bounds in BoTorch format: shape (2, D), rows = [lower, upper].

        In normalized space this is [[0,...,0], [1,...,1]].
        """
        return np.vstack([
            np.zeros(self.n_dim),
            np.ones(self.n_dim),
        ])
