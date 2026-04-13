"""Tests for data loading and validation."""

import pandas as pd
import pytest

from bayesopt_human.data.loader import load_observations


class TestLoadObservations:
    def test_load_csv(self, simple_csv):
        path, expected = simple_csv
        df = load_observations(path, "loss")
        assert len(df) == 5
        assert "loss" in df.columns
        assert "x1" in df.columns
        assert "x2" in df.columns

    def test_missing_objective_column(self, simple_csv):
        path, _ = simple_csv
        with pytest.raises(ValueError, match="not found"):
            load_observations(path, "nonexistent")

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_observations("/nonexistent/file.csv", "loss")

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "data.xlsx"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported"):
            load_observations(str(path), "loss")

    def test_missing_values(self, tmp_path):
        df = pd.DataFrame({
            "x1": [1.0, None, 3.0],
            "x2": [1.0, 2.0, 3.0],
            "loss": [1.0, 2.0, 3.0],
        })
        path = tmp_path / "missing.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing values"):
            load_observations(str(path), "loss")

    def test_too_few_observations(self, tmp_path):
        df = pd.DataFrame({"x1": [1.0], "loss": [1.0]})
        path = tmp_path / "single.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="At least 2"):
            load_observations(str(path), "loss")
