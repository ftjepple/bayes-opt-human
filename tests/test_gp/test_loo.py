"""Tests for LOO-MAE computation."""

import numpy as np
import pytest

from bayesopt_human.config import OptimizationConfig
from bayesopt_human.gp.loo import analytical_loo_mae, analytical_loo_details
from bayesopt_human.gp.model import GPModel


@pytest.fixture
def fitted_gp_simple():
    config = OptimizationConfig(
        name="test",
        horizon=50,
        bounds={"x1": (0.0, 1.0), "x2": (0.0, 1.0)},
        objective_column="y",
    )
    np.random.seed(42)
    X = np.random.rand(10, 2) * 0.99
    y = np.sum(X**2, axis=1)
    gp = GPModel(config)
    fitted = gp.fit(X, y)
    return fitted, X, y


class TestAnalyticalLOO:
    def test_loo_returns_finite(self, fitted_gp_simple):
        fitted, X, y = fitted_gp_simple
        mae, std, lpd, cal_var, mean_r = analytical_loo_mae(fitted, X, y)
        assert np.isfinite(mae)
        assert np.isfinite(std)
        assert mae >= 0
        assert std >= 0

    def test_loo_is_nonnegative(self, fitted_gp_simple):
        fitted, X, y = fitted_gp_simple
        mae, std, lpd, cal_var, mean_r = analytical_loo_mae(fitted, X, y)
        assert mae >= 0.0
        assert std >= 0.0

    def test_loo_lpd_finite(self, fitted_gp_simple):
        fitted, X, y = fitted_gp_simple
        mae, std, lpd, cal_var, mean_r = analytical_loo_mae(fitted, X, y)
        assert np.isfinite(lpd)

    def test_calibration_var_finite_and_positive(self, fitted_gp_simple):
        fitted, X, y = fitted_gp_simple
        mae, std, lpd, cal_var, mean_r = analytical_loo_mae(fitted, X, y)
        assert np.isfinite(cal_var)
        assert cal_var > 0

    def test_loo_details_shapes(self, fitted_gp_simple):
        fitted, X, y = fitted_gp_simple
        details = analytical_loo_details(fitted, X, y)
        n = len(y)
        assert details["mu_loo"].shape == (n,)
        assert details["sigma_loo"].shape == (n,)
        assert details["residuals"].shape == (n,)
        assert details["standardized_residuals"].shape == (n,)
        assert details["y_norm"].shape == (n,)
        assert np.all(details["sigma_loo"] > 0)
        assert np.isfinite(details["calibration_var"])
        assert details["calibration_var"] > 0
