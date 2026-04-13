"""Tests for modality diagnostics."""

import numpy as np
import pytest

from bayesopt_human.diagnostics.modality import detect_local_optima


class TestModalityDetection:
    def test_unimodal_function(self):
        """Simple quadratic should have one local optimum."""
        np.random.seed(42)
        n = 30
        X = np.random.rand(n, 2) * 0.99
        y = np.sum((X - 0.5)**2, axis=1)  # Minimum at center
        report = detect_local_optima(X, y, "minimize", k=5)
        assert report.n_local_optima_flat >= 1
        assert report.n_basins_flat >= 1
        assert not report.reliability_warning  # 30/2 = 15 > 5

    def test_reliability_warning(self):
        """Few points should trigger reliability warning."""
        X = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        y = np.array([1.0, 0.5, 0.8])
        report = detect_local_optima(X, y, "minimize", k=2)
        assert report.reliability_warning  # 3/3 = 1 < 5

    def test_maximize_direction(self):
        np.random.seed(42)
        X = np.random.rand(20, 2) * 0.99
        y = -np.sum((X - 0.5)**2, axis=1)  # Maximum at center
        report = detect_local_optima(X, y, "maximize", k=5)
        assert report.n_local_optima_flat >= 1
