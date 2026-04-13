"""Tests for output transforms."""

import numpy as np
import pytest

from bayesopt_human.transforms.output import OutputTransform


class TestOutputTransform:
    def test_none_roundtrip(self):
        ot = OutputTransform("none")
        y = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(ot.inverse(ot.forward(y)), y)

    def test_log_roundtrip(self):
        ot = OutputTransform("log")
        y = np.array([0.5, 1.0, 2.0, 10.0])
        np.testing.assert_allclose(ot.inverse(ot.forward(y)), y, atol=1e-10)

    def test_clipped_log_roundtrip(self):
        ot = OutputTransform("clipped_log")
        y = np.array([1.0, 2.0, 10.0])
        result = ot.inverse(ot.forward(y))
        np.testing.assert_allclose(result, y, atol=1e-10)

    def test_log_negative_roundtrip(self):
        ot = OutputTransform("log")
        y = np.array([-0.5, -1.0, -2.0, -10.0])
        np.testing.assert_allclose(ot.inverse(ot.forward(y)), y, atol=1e-10)

    def test_log_negative_preserves_order(self):
        ot = OutputTransform("log")
        y = np.array([-10.0, -2.0, -0.5])
        result = ot.forward(y)
        # Monotonic: more negative raw -> more negative transformed
        assert result[0] < result[1] < result[2]

    def test_log_mixed_sign_raises(self):
        ot = OutputTransform("log")
        with pytest.raises(ValueError, match="positive.*negative"):
            ot.forward(np.array([-1.0, 1.0]))

    def test_clipped_log_handles_near_zero(self):
        ot = OutputTransform("clipped_log", clip_epsilon=1e-8)
        y = np.array([0.0, -1.0, 1.0])
        result = ot.forward(y)
        assert np.isfinite(result).all()

    def test_signed_log1p_roundtrip(self):
        ot = OutputTransform("signed_log1p")
        y = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
        np.testing.assert_allclose(ot.inverse(ot.forward(y)), y, atol=1e-10)

    def test_signed_log1p_preserves_sign(self):
        ot = OutputTransform("signed_log1p")
        y = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        result = ot.forward(y)
        assert result[0] < 0
        assert result[1] < 0
        assert result[2] == 0.0
        assert result[3] > 0
        assert result[4] > 0

    def test_signed_log1p_compresses_magnitude(self):
        ot = OutputTransform("signed_log1p")
        y = np.array([1000.0, -1000.0])
        result = ot.forward(y)
        assert abs(result[0]) < abs(y[0])
        assert abs(result[1]) < abs(y[1])

    def test_invalid_transform(self):
        with pytest.raises(ValueError, match="Invalid"):
            OutputTransform("sqrt")

    def test_repr(self):
        ot = OutputTransform("log")
        assert "log" in repr(ot)
