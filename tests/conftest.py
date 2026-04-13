"""Shared test fixtures for BayesOpt-Human tests."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def branin_data():
    """Branin-Hoo function data (2D) for testing.

    The Branin function has 3 global minima at approximately:
    (-pi, 12.275), (pi, 2.275), (9.42478, 2.475)
    with a minimum value of ~0.397887.
    """
    np.random.seed(42)
    n = 20
    # Sample in the standard Branin domain: x1 in [-5, 10], x2 in [0, 15]
    x1 = np.random.uniform(-5, 10, n)
    x2 = np.random.uniform(0, 15, n)

    # Branin function
    a, b, c = 1, 5.1 / (4 * np.pi**2), 5 / np.pi
    r, s, t = 6, 10, 1 / (8 * np.pi)
    y = a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s

    df = pd.DataFrame({"x1": x1, "x2": x2, "objective": y})
    bounds = {"x1": (-5.0, 10.0), "x2": (0.0, 15.0)}

    return df, bounds


@pytest.fixture
def simple_csv(tmp_path):
    """Create a simple CSV file for data loading tests."""
    df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x2": [0.5, 1.5, 2.5, 3.5, 4.5],
        "loss": [10.0, 8.0, 6.0, 7.0, 9.0],
    })
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return str(path), df


@pytest.fixture
def sample_normalized_data():
    """Sample normalized data for GP and acquisition tests."""
    np.random.seed(42)
    n, d = 15, 3
    X = np.random.rand(n, d) * 0.99  # In [0, 1)
    y = np.sum((X - 0.5)**2, axis=1)  # Simple quadratic
    return X, y


@pytest.fixture
def sample_config():
    """Sample optimization config for testing."""
    from bayesopt_human.config import OptimizationConfig
    return OptimizationConfig(
        name="test",
        horizon=50,
        bounds={"x1": (0.0, 10.0), "x2": (0.0, 10.0), "x3": (0.0, 10.0)},
        objective_column="objective",
        direction="minimize",
    )
