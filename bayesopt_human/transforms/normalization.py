"""IQR normalization of transformed targets."""

from __future__ import annotations

import numpy as np


class IQRNormalizer:
    """Robust IQR normalization of transformed targets.

    Applied after the output transform and before GP fitting.
    y_norm = (y' - median(y')) / IQR(y')
    """

    def __init__(self) -> None:
        self.median_: float | None = None
        self.iqr_: float | None = None
        self._fitted = False

    def fit(self, y_transformed: np.ndarray) -> IQRNormalizer:
        """Compute median and IQR from training data."""
        y_transformed = np.asarray(y_transformed, dtype=np.float64)
        self.median_ = float(np.median(y_transformed))
        q75, q25 = np.percentile(y_transformed, [75, 25])
        self.iqr_ = float(q75 - q25)
        if self.iqr_ == 0:
            # Fallback: use standard deviation if IQR is zero (constant data)
            self.iqr_ = float(np.std(y_transformed))
        if self.iqr_ == 0:
            # All values identical: use 1.0 to avoid division by zero
            self.iqr_ = 1.0
        self._fitted = True
        return self

    def transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """Normalize using fitted median and IQR."""
        if not self._fitted:
            raise RuntimeError("IQRNormalizer must be fit before transform.")
        y_transformed = np.asarray(y_transformed, dtype=np.float64)
        return (y_transformed - self.median_) / self.iqr_

    def inverse_transform(self, y_normalized: np.ndarray) -> np.ndarray:
        """Recover transformed-scale values from normalized."""
        if not self._fitted:
            raise RuntimeError("IQRNormalizer must be fit before inverse_transform.")
        y_normalized = np.asarray(y_normalized, dtype=np.float64)
        return y_normalized * self.iqr_ + self.median_
