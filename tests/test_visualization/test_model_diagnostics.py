"""Smoke tests for model diagnostic plots."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from bayesopt_human.visualization.model_diagnostics import (
    plot_residual_histogram,
    plot_residuals_vs_x,
    plot_sigma_vs_x,
)


@pytest.fixture
def diagnostic_data():
    np.random.seed(42)
    X = np.random.rand(15, 3)
    residuals = np.random.randn(15) * 0.5
    sigma = np.abs(np.random.randn(15)) + 0.1
    return X, residuals, sigma


class TestResidualHistogram:
    def test_smoke(self, diagnostic_data):
        _, residuals, _ = diagnostic_data
        fig = plot_residual_histogram(residuals)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestResidualsVsX:
    def test_smoke(self, diagnostic_data):
        X, residuals, _ = diagnostic_data
        fig = plot_residuals_vs_x(X, residuals, ["a", "b", "c"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_default_names(self, diagnostic_data):
        X, residuals, _ = diagnostic_data
        fig = plot_residuals_vs_x(X, residuals)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestSigmaVsX:
    def test_smoke(self, diagnostic_data):
        X, _, sigma = diagnostic_data
        fig = plot_sigma_vs_x(X, sigma, ["a", "b", "c"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
