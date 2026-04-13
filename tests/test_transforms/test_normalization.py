"""Tests for IQR normalization."""

import numpy as np
import pytest

from bayesopt_human.transforms.normalization import IQRNormalizer


class TestIQRNormalizer:
    def test_basic_fit_transform(self):
        iqr = IQRNormalizer()
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        iqr.fit(y)
        y_norm = iqr.transform(y)
        # Median should map to 0
        np.testing.assert_allclose(
            iqr.transform(np.array([iqr.median_])), [0.0], atol=1e-10
        )

    def test_inverse_roundtrip(self):
        iqr = IQRNormalizer()
        y = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
        iqr.fit(y)
        y_norm = iqr.transform(y)
        y_back = iqr.inverse_transform(y_norm)
        np.testing.assert_allclose(y_back, y, atol=1e-10)

    def test_constant_values(self):
        iqr = IQRNormalizer()
        y = np.array([5.0, 5.0, 5.0, 5.0])
        iqr.fit(y)
        # Should not crash; IQR fallback should handle this
        y_norm = iqr.transform(y)
        assert np.isfinite(y_norm).all()

    def test_transform_before_fit_raises(self):
        iqr = IQRNormalizer()
        with pytest.raises(RuntimeError, match="fit"):
            iqr.transform(np.array([1.0]))

    def test_inverse_before_fit_raises(self):
        iqr = IQRNormalizer()
        with pytest.raises(RuntimeError, match="fit"):
            iqr.inverse_transform(np.array([1.0]))

    def test_outlier_robustness(self):
        iqr = IQRNormalizer()
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        iqr.fit(y)
        # The outlier should not dominate the scaling
        y_norm = iqr.transform(y)
        # Most values should be within a reasonable range
        assert abs(y_norm[0]) < 10
