"""Tests for surrogate space transforms."""

import numpy as np
import pytest

from bayesopt_human.config import OptimizationConfig
from bayesopt_human.gp.model import GPModel
from bayesopt_human.gp.surrogate_space import SurrogateSpaceTransform


@pytest.fixture
def surrogate_transform():
    config = OptimizationConfig(
        name="test",
        horizon=50,
        bounds={"x1": (0.0, 1.0), "x2": (0.0, 1.0)},
        objective_column="y",
    )
    np.random.seed(42)
    X = np.random.rand(10, 2) * 0.99
    y = X[:, 0] * 2 + X[:, 1]  # Dim 0 is more important
    gp = GPModel(config)
    fitted = gp.fit(X, y)
    return SurrogateSpaceTransform(fitted), X


class TestSurrogateSpaceTransform:
    def test_to_surrogate_shape(self, surrogate_transform):
        st, X = surrogate_transform
        X_sur = st.to_surrogate(X)
        assert X_sur.shape == X.shape

    def test_to_surrogate_1d(self, surrogate_transform):
        st, X = surrogate_transform
        x_sur = st.to_surrogate(X[0])
        assert x_sur.ndim == 1
        assert len(x_sur) == 2

    def test_pairwise_distances(self, surrogate_transform):
        st, X = surrogate_transform
        dists = st.flat_to_surrogate_distances(X)
        assert dists.shape == (len(X), len(X))
        # Diagonal should be zero
        np.testing.assert_allclose(np.diag(dists), 0.0, atol=1e-10)
        # Symmetric
        np.testing.assert_allclose(dists, dists.T, atol=1e-10)
