"""Smoke tests for visualization functions."""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests
import matplotlib.pyplot as plt

from bayesopt_human.acquisition.candidates import Candidate
from bayesopt_human.config import OptimizationConfig
from bayesopt_human.gp.model import GPModel
from bayesopt_human.recommendations.schema import Recommendation
from bayesopt_human.visualization.candidates import display_candidate_table
from bayesopt_human.visualization.model_params import plot_length_scales, plot_warp_functions
from bayesopt_human.visualization.progress import plot_progress
from bayesopt_human.visualization.projections import (
    plot_pairwise,
    plot_pca_projection,
    plot_top2_projection,
    short_label,
)


@pytest.fixture
def fitted_gp_2d():
    config = OptimizationConfig(
        name="test",
        horizon=50,
        bounds={"x1": (0.0, 1.0), "x2": (0.0, 1.0)},
        objective_column="y",
    )
    np.random.seed(42)
    X = np.random.rand(15, 2) * 0.99
    y = np.sum(X**2, axis=1)
    gp = GPModel(config)
    fitted = gp.fit(X, y)
    return fitted, X, y


@pytest.fixture
def sample_candidates():
    return [
        Candidate(
            coordinates_normalized=np.array([0.3, 0.3]),
            coordinates_raw=np.array([3.0, 3.0]),
            coordinates_surrogate=np.array([0.3, 0.3]),
            source="EI", category="balanced",
            posterior_mean=1.0, posterior_std=0.5,
            expected_improvement=0.1,
            coverage_gain_flat=0.01, coverage_gain_surrogate=0.01,
            svm_prediction=True,
        ),
        Candidate(
            coordinates_normalized=np.array([0.7, 0.7]),
            coordinates_raw=np.array([7.0, 7.0]),
            coordinates_surrogate=np.array([0.7, 0.7]),
            source="Max GP Mean", category="exploitation",
            posterior_mean=0.8, posterior_std=0.3,
            expected_improvement=0.05,
            coverage_gain_flat=0.02, coverage_gain_surrogate=0.02,
            svm_prediction=False,
        ),
    ]


class TestProgressPlot:
    def test_smoke(self):
        df = pd.DataFrame({
            "obs": range(10),
            "loss": np.random.rand(10) * 10,
        })
        fig = plot_progress(df, "loss", "minimize")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_gp_optima(self):
        df = pd.DataFrame({"loss": np.random.rand(10)})
        gp_opts = list(np.random.rand(10))
        fig = plot_progress(df, "loss", "maximize", gp_predicted_optima=gp_opts)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestLengthScalePlot:
    def test_smoke(self, fitted_gp_2d):
        fitted, X, y = fitted_gp_2d
        fig = plot_length_scales(fitted, ["x1", "x2"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestWarpPlot:
    def test_returns_none_without_warping(self, fitted_gp_2d):
        fitted, X, y = fitted_gp_2d
        fig = plot_warp_functions(fitted, ["x1", "x2"])
        assert fig is None


class TestProjectionPlots:
    def test_top2_smoke(self, fitted_gp_2d, sample_candidates):
        fitted, X, y = fitted_gp_2d
        fig = plot_top2_projection(
            X, y, fitted, sample_candidates,
            direction="maximize", param_names=["x1", "x2"],
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_pca_smoke(self, fitted_gp_2d, sample_candidates):
        fitted, X, y = fitted_gp_2d
        fig = plot_pca_projection(
            X, y, fitted, sample_candidates, direction="maximize",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_pairwise_smoke(self, fitted_gp_2d, sample_candidates):
        fitted, X, y = fitted_gp_2d
        fig = plot_pairwise(
            X, y, fitted, sample_candidates, top_k=2,
            direction="maximize", param_names=["x1", "x2"],
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestCandidateTable:
    def test_display(self, sample_candidates):
        recs = [
            Recommendation(
                candidate=np.array([3.0, 3.0]),
                candidate_normalized=np.array([0.3, 0.3]),
                source="EI", rationale="Good", confidence="high", priority=1,
                arm="balanced", is_arm_winner=True,
            ),
        ]
        df = display_candidate_table(sample_candidates, recs)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "Source" in df.columns


class TestShortLabel:
    def test_ei(self):
        assert short_label("Expected Improvement") == "EI"

    def test_ucb_variants(self):
        assert short_label("UCB (kappa=high)") == "UCB HI"
        assert short_label("UCB (kappa=medium)") == "UCB MID"
        assert short_label("UCB (kappa=low)") == "UCB LO"

    def test_lcb_variants(self):
        assert short_label("LCB (kappa=high)") == "LCB HI"
        assert short_label("LCB (kappa=medium)") == "LCB MID"
        assert short_label("LCB (kappa=low)") == "LCB LO"

    def test_other_strategies(self):
        assert short_label("Max GP Mean") == "MAX MEAN"
        assert short_label("Max-min distance (flat)") == "MAXMIN FLAT"
        assert short_label("Max-min distance (surrogate)") == "MAXMIN SUR"
        assert short_label("Max Posterior Variance") == "MPV"

    def test_fallback(self):
        assert short_label("Some Unknown Strategy") == "Some Unkno"
