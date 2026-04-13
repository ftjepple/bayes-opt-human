"""Tests for bounds normalization."""

import numpy as np
import pytest

from bayesopt_human.data.normalizer import BoundsNormalizer


class TestBoundsNormalizer:
    def test_basic_normalization(self):
        norm = BoundsNormalizer({"x1": (0.0, 10.0), "x2": (-5.0, 5.0)})
        X = np.array([[5.0, 0.0], [0.0, -5.0]])
        X_norm = norm.normalize(X)
        np.testing.assert_allclose(X_norm[0], [0.5, 0.5])
        np.testing.assert_allclose(X_norm[1], [0.0, 0.0])

    def test_denormalization_roundtrip(self):
        norm = BoundsNormalizer({"x1": (1.0, 100.0), "x2": (-3.0, 7.0)})
        X = np.array([[50.0, 2.0], [10.0, -1.0]])
        X_norm = norm.normalize(X)
        X_back = norm.denormalize(X_norm)
        np.testing.assert_allclose(X_back, X, atol=1e-10)

    def test_1d_input(self):
        norm = BoundsNormalizer({"x1": (0.0, 10.0)})
        x = np.array([5.0])
        x_norm = norm.normalize(x)
        assert x_norm.ndim == 1
        np.testing.assert_allclose(x_norm, [0.5])

    def test_out_of_bounds_raises(self):
        norm = BoundsNormalizer({"x1": (0.0, 10.0)})
        with pytest.raises(ValueError, match="outside"):
            norm.normalize(np.array([[11.0]]))

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            BoundsNormalizer({"x1": (10.0, 5.0)})

    def test_n_dim(self):
        norm = BoundsNormalizer({"a": (0.0, 1.0), "b": (0.0, 1.0), "c": (0.0, 1.0)})
        assert norm.n_dim == 3

    def test_botorch_bounds(self):
        norm = BoundsNormalizer({"x1": (0.0, 10.0), "x2": (0.0, 5.0)})
        bb = norm.botorch_bounds()
        assert bb.shape == (2, 2)
        np.testing.assert_allclose(bb[0], [0.0, 0.0])
        np.testing.assert_allclose(bb[1], [1.0, 1.0])
