"""Tests for adaptive jitter."""

import numpy as np
import pytest

from bayesopt_human.gp.jitter import AdaptiveJitter


class TestAdaptiveJitter:
    def test_initial_jitter(self):
        aj = AdaptiveJitter(base=1e-6, ceiling_factor=1e-2)
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        jitter = aj.initial_jitter(y)
        assert jitter >= 1e-6
        assert jitter > 0

    def test_increase(self):
        aj = AdaptiveJitter(base=1e-6, ceiling_factor=1e-2)
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        aj.initial_jitter(y)
        new_jitter = aj.increase(y)
        assert new_jitter > aj.base

    def test_ceiling_reached(self):
        aj = AdaptiveJitter(base=1e-2, ceiling_factor=1e-2)
        y = np.array([0.0, 1.0])
        aj.initial_jitter(y)
        # Try increasing past ceiling
        for _ in range(10):
            aj.increase(y)
        assert aj.conditioning_warning is True
