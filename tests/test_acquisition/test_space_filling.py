"""Tests for space-filling candidate generation."""

import numpy as np
import pytest

from bayesopt_human.acquisition.space_filling import max_min_distance_flat


class TestSpaceFilling:
    def test_candidate_in_bounds(self):
        np.random.seed(42)
        X = np.random.rand(10, 2) * 0.99
        cand = max_min_distance_flat(X, n_dim=2)
        assert cand.shape == (2,)
        assert np.all(cand >= 0)
        assert np.all(cand < 1.0)

    def test_candidate_not_too_close(self):
        """Candidate should be reasonably far from existing points."""
        np.random.seed(42)
        X = np.random.rand(5, 2) * 0.99
        cand = max_min_distance_flat(X, n_dim=2)
        from scipy.spatial.distance import cdist
        dists = cdist(cand.reshape(1, -1), X).flatten()
        min_dist = np.min(dists)
        # Should be at least somewhat far away
        assert min_dist > 0.01
