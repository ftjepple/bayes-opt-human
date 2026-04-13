"""Tests for acquisition functions."""

import numpy as np
import pytest
import torch

from bayesopt_human.acquisition.functions import (
    compute_ei,
    compute_max_mean,
    compute_mpv,
    compute_ucb_lcb,
)
from bayesopt_human.config import OptimizationConfig
from bayesopt_human.gp.model import GPModel


@pytest.fixture
def fitted_gp():
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
    return gp.fit(X, y), X, y


class TestAcquisitionFunctions:
    def test_ei_returns_finite(self, fitted_gp):
        fgp, X, y = fitted_gp
        best_val = float(np.min(y))
        acqf = compute_ei(fgp, best_val, "minimize")
        x_test = torch.rand(5, 1, 2, dtype=torch.float64)
        with torch.no_grad():
            values = acqf(x_test)
        # LogEI returns log-space values, which are always finite
        assert torch.isfinite(values).all()

    def test_ucb_direction(self, fitted_gp):
        fgp, X, y = fitted_gp
        ucb = compute_ucb_lcb(fgp, 1.0, "maximize")
        lcb = compute_ucb_lcb(fgp, 1.0, "minimize")
        assert ucb is not None
        assert lcb is not None

    def test_mpv_returns_nonnegative(self, fitted_gp):
        fgp, X, y = fitted_gp
        acqf = compute_mpv(fgp)
        x_test = torch.rand(5, 1, 2, dtype=torch.float64)
        with torch.no_grad():
            values = acqf(x_test)
        assert (values >= -1e-6).all()

    def test_max_mean(self, fitted_gp):
        fgp, X, y = fitted_gp
        acqf = compute_max_mean(fgp, "minimize")
        assert acqf is not None
