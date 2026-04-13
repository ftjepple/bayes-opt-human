"""Tests for GP model fitting."""

import numpy as np
import pytest

from bayesopt_human.config import OptimizationConfig
from bayesopt_human.gp.model import GPModel, FittedGP


@pytest.fixture
def gp_config():
    return OptimizationConfig(
        name="test",
        horizon=50,
        bounds={"x1": (0.0, 1.0), "x2": (0.0, 1.0), "x3": (0.0, 1.0)},
        objective_column="y",
    )


class TestGPModel:
    def test_fit_basic(self, gp_config, sample_normalized_data):
        X, y = sample_normalized_data
        gp = GPModel(gp_config)
        fitted = gp.fit(X, y, output_transform="none", warping=False)
        assert isinstance(fitted, FittedGP)
        assert fitted.length_scales.shape == (3,)
        assert fitted.jitter_used > 0
        assert np.isfinite(fitted.conditioning)

    def test_fit_with_log_transform(self, gp_config):
        np.random.seed(42)
        X = np.random.rand(15, 3) * 0.99
        y = np.exp(np.sum(X, axis=1))  # Positive values
        gp = GPModel(gp_config)
        fitted = gp.fit(X, y, output_transform="log", warping=False)
        assert fitted.output_transform.transform_type == "log"

    def test_predict(self, gp_config, sample_normalized_data):
        X, y = sample_normalized_data
        gp = GPModel(gp_config)
        fitted = gp.fit(X, y)
        X_new = np.array([[0.5, 0.5, 0.5]])
        mean, std = gp.predict(fitted, X_new)
        assert mean.shape == (1,)
        assert std.shape == (1,)
        assert np.isfinite(mean).all()
        assert np.all(std >= 0)

    def test_fit_produces_reasonable_predictions(self, gp_config):
        """GP should predict near training data at training points."""
        np.random.seed(42)
        X = np.random.rand(10, 3) * 0.99
        y = np.sum(X**2, axis=1)
        gp = GPModel(gp_config)
        fitted = gp.fit(X, y)
        mean, std = gp.predict(fitted, X)
        # Predictions at training points should be close to observed
        np.testing.assert_allclose(mean, y, rtol=0.5)

    def test_fit_non_ard(self, gp_config, sample_normalized_data):
        """Non-ARD fitting should produce identical length scales."""
        X, y = sample_normalized_data
        gp = GPModel(gp_config)
        fitted = gp.fit(X, y, ard=False)
        assert isinstance(fitted, FittedGP)
        assert fitted.ard is False
        assert fitted.length_scales.shape == (3,)
        # All length scales should be identical (broadcast from single value)
        assert np.all(fitted.length_scales == fitted.length_scales[0])
