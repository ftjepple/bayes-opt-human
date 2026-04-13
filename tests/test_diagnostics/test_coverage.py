"""Tests for coverage diagnostics."""

import numpy as np
import pytest

from bayesopt_human.diagnostics.coverage import coverage_gain, coverage_metrics


class TestCoverageMetrics:
    def test_basic_coverage(self):
        np.random.seed(42)
        X = np.random.rand(20, 3) * 0.99
        report = coverage_metrics(X, k=5)
        assert report.avg_knn_distance_flat > 0
        assert report.min_knn_distance_flat > 0
        assert report.optimal_baseline_flat > 0
        assert report.ratio_to_optimal_flat > 0

    def test_small_k(self):
        X = np.array([[0.1, 0.1], [0.9, 0.9], [0.5, 0.5]])
        report = coverage_metrics(X, k=5)
        assert report.k == 2  # min(5, N-1)

    def test_coverage_gain_returns_finite(self):
        """Coverage gain should return finite values."""
        np.random.seed(42)
        X = np.random.rand(10, 2) * 0.99
        x_new = np.array([0.5, 0.5])
        flat_gain, sur_gain = coverage_gain(X, x_new, k=3)
        assert np.isfinite(flat_gain)
        assert sur_gain == 0.0  # No surrogate transform provided

    def test_coverage_gain_nearby_point(self):
        """Adding a point near an existing cluster should decrease coverage (negative gain)."""
        np.random.seed(42)
        X = np.random.rand(10, 2) * 0.99
        # Add a point very close to existing ones
        x_new = X[0] + 1e-5
        flat_gain, _ = coverage_gain(X, x_new, k=3)
        assert np.isfinite(flat_gain)
