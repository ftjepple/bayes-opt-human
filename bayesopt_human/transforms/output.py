"""Output transforms (none, log, clipped_log, signed_log1p)."""

from __future__ import annotations

import numpy as np


class OutputTransform:
    """Applies and inverts output transforms on objective values.

    ``log`` auto-detects sign on the first call to :meth:`forward`: strictly
    positive data uses ``log(y)`` while strictly negative data uses the
    mirrored form ``-log(-y)``. Mixed-sign data raises ``ValueError``. The
    sign mode is then locked for the lifetime of the instance so
    :meth:`inverse` is round-trip safe.

    ``signed_log1p`` computes ``sign(y) · log(1 + |y|)`` in one shot, so it
    handles mixed-sign data (and zero) without needing a sign mode — it is
    the recommended transform whenever the objective spans zero.
    """

    VALID_TRANSFORMS = ("none", "log", "clipped_log", "signed_log1p")

    def __init__(
        self, transform_type: str, clip_epsilon: float = 1e-8
    ) -> None:
        """
        Args:
            transform_type: One of VALID_TRANSFORMS.
            clip_epsilon: Clipping threshold for clipped_log.
        """
        if transform_type not in self.VALID_TRANSFORMS:
            raise ValueError(
                f"Invalid transform {transform_type!r}. "
                f"Must be one of {self.VALID_TRANSFORMS}."
            )
        self.transform_type = transform_type
        self.clip_epsilon = clip_epsilon
        # Only the ``log`` transform cares about sign mode. None means "not yet
        # fitted"; locked to True/False on the first ``forward`` call so that
        # subsequent ``inverse`` calls are reproducible and round-trip safe.
        self._negative_mode: bool | None = None

    def forward(self, y: np.ndarray) -> np.ndarray:
        """Apply transform to raw objective values."""
        y = np.asarray(y, dtype=np.float64)
        if self.transform_type == "none":
            return y.copy()
        elif self.transform_type == "log":
            if np.all(y > 0):
                mode = False
            elif np.all(y < 0):
                mode = True
            else:
                raise ValueError(
                    "log transform requires strictly positive or "
                    "strictly negative values."
                )
            if self._negative_mode is None:
                self._negative_mode = mode
            elif self._negative_mode != mode:
                raise ValueError(
                    "log transform sign mode was locked to "
                    f"{'negative' if self._negative_mode else 'positive'} "
                    "on an earlier forward() call; cannot apply to data of "
                    "the opposite sign."
                )
            return -np.log(-y) if mode else np.log(y)
        elif self.transform_type == "clipped_log":
            return np.log(np.maximum(y, self.clip_epsilon))
        elif self.transform_type == "signed_log1p":
            return np.sign(y) * np.log1p(np.abs(y))
        raise RuntimeError(f"Unhandled transform type: {self.transform_type}")

    def inverse(self, y_transformed: np.ndarray) -> np.ndarray:
        """Invert transform to recover raw-scale values."""
        y_transformed = np.asarray(y_transformed, dtype=np.float64)
        if self.transform_type == "none":
            return y_transformed.copy()
        elif self.transform_type == "log":
            if self._negative_mode is None:
                raise RuntimeError(
                    "OutputTransform('log').inverse() called before "
                    "forward(); the sign mode is not yet determined."
                )
            if self._negative_mode:
                return -np.exp(-y_transformed)
            return np.exp(y_transformed)
        elif self.transform_type == "clipped_log":
            return np.exp(y_transformed)
        elif self.transform_type == "signed_log1p":
            return np.sign(y_transformed) * np.expm1(np.abs(y_transformed))
        raise RuntimeError(f"Unhandled transform type: {self.transform_type}")

    def __repr__(self) -> str:
        return f"OutputTransform({self.transform_type!r})"
