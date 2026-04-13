"""Tests for the OptimizationStep get/report/choose API."""

import numpy as np
import pandas as pd
import pytest

from bayesopt_human.config import OptimizationConfig
from bayesopt_human.optimizer.step import (
    CandidateResult,
    DataEntry,
    DataPayload,
    DataResult,
    ModelResult,
    OptimizationStep,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_step():
    """Step with a single config whose name matches 'data.csv'."""
    config = OptimizationConfig(
        name="data",
        horizon=50,
        bounds={"x1": (0.0, 10.0), "x2": (0.0, 10.0)},
        objective_column="loss",
        direction="minimize",
    )
    return OptimizationStep([config])


@pytest.fixture
def multi_step():
    """Step with two configs matching 'problem_0.csv' and 'problem_1.csv'."""
    configs = [
        OptimizationConfig(
            name="problem_0",
            horizon=50,
            bounds={"x1": (0.0, 10.0), "x2": (0.0, 10.0)},
            objective_column="loss",
            direction="minimize",
        ),
        OptimizationConfig(
            name="problem_1",
            horizon=50,
            bounds={"x1": (0.0, 10.0), "x2": (0.0, 10.0)},
            objective_column="loss",
            direction="minimize",
        ),
    ]
    return OptimizationStep(configs)


@pytest.fixture
def single_data_dir(tmp_path):
    """Create a directory with a single CSV file named 'data.csv'."""
    df = pd.DataFrame({
        "x1": np.linspace(1, 9, 10),
        "x2": np.linspace(0.5, 9.5, 10),
        "loss": np.random.RandomState(42).rand(10) * 10,
    })
    df.to_csv(tmp_path / "data.csv", index=False)
    return str(tmp_path)


@pytest.fixture
def multi_data_dir(tmp_path):
    """Create a directory with multiple CSV files."""
    for i, n_rows in enumerate([8, 12]):
        df = pd.DataFrame({
            "x1": np.linspace(0, 10, n_rows),
            "x2": np.linspace(0, 10, n_rows),
            "loss": np.random.RandomState(i).rand(n_rows) * 10,
        })
        df.to_csv(tmp_path / f"problem_{i}.csv", index=False)
    return str(tmp_path)


# ---------------------------------------------------------------------------
# get_data
# ---------------------------------------------------------------------------

class TestGetData:
    def test_single_file(self, single_step, single_data_dir):
        result = single_step.get_data(single_data_dir)
        assert isinstance(result, DataResult)
        assert len(result.entries) == 1
        entry = result.entries[0]
        assert entry.n_obs == 10
        assert entry.n_dim == 2
        assert entry.param_names == ["x1", "x2"]

    def test_directory(self, multi_step, multi_data_dir):
        result = multi_step.get_data(multi_data_dir)
        assert isinstance(result, DataResult)
        assert len(result.entries) == 2
        assert "problem_0" in result.entries[0].filepath
        assert "problem_1" in result.entries[1].filepath

    def test_missing_csv_raises(self, single_step, tmp_path):
        with pytest.raises(FileNotFoundError):
            single_step.get_data(str(tmp_path))


# ---------------------------------------------------------------------------
# report_data
# ---------------------------------------------------------------------------

class TestReportData:
    def test_no_arg_returns_summary_list(self, single_step, single_data_dir):
        result = single_step.get_data(single_data_dir)
        reports = single_step.report_data(result)
        assert isinstance(reports, list)
        assert reports == ["statistics", "progress", "histogram", "correlation", "duplicates",
                           "coverage_metrics", "modality"]

    def test_individual_no_arg_returns_list(self, single_step, single_data_dir):
        result = single_step.get_data(single_data_dir)
        reports = single_step.report_data(result, choice=0)
        assert isinstance(reports, list)
        assert reports == ["observations", "progress", "statistics", "histogram", "coverage",
                           "pca", "coverage_metrics", "modality", "svm"]

    def test_invalid_raises(self, single_step, single_data_dir):
        result = single_step.get_data(single_data_dir)
        with pytest.raises(ValueError, match="Unknown summary report"):
            single_step.report_data(result, report="nonexistent")

    def test_statistics(self, single_step, single_data_dir):
        result = single_step.get_data(single_data_dir)
        reports = single_step.report_data(result, report="statistics", choice=0)
        assert reports == ["statistics"]

    def test_progress(self, single_step, single_data_dir):
        import matplotlib
        matplotlib.use("Agg")
        result = single_step.get_data(single_data_dir)
        reports = single_step.report_data(result, report="progress")
        assert reports == ["progress"]


# ---------------------------------------------------------------------------
# choose_data
# ---------------------------------------------------------------------------

class TestChooseData:
    def test_basic(self, single_step, single_data_dir):
        result = single_step.get_data(single_data_dir)
        data = single_step.choose_data(result, choice=0)
        assert isinstance(data, DataPayload)
        assert data.X_norm.shape == (10, 2)
        assert data.y.shape == (10,)
        assert data.param_names == ["x1", "x2"]

    def test_auto_detect_bounds(self, tmp_path):
        """When no bounds are configured, auto-detect from data."""
        config = OptimizationConfig(
            name="data",
            horizon=50,
            bounds={"x1": (0.0, 10.0), "x2": (0.0, 10.0)},
            objective_column="loss",
        )
        s = OptimizationStep([config])
        df = pd.DataFrame({
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [0.5, 1.5, 2.5, 3.5, 4.5],
            "loss": [10.0, 8.0, 6.0, 7.0, 9.0],
        })
        df.to_csv(tmp_path / "data.csv", index=False)
        result = s.get_data(str(tmp_path))
        # Clear bounds after get_data to trigger auto-detect in choose_data
        s.configs[0].bounds = None
        data = s.choose_data(result, choice=0)
        assert isinstance(data, DataPayload)

    def test_out_of_range(self, single_step, single_data_dir):
        result = single_step.get_data(single_data_dir)
        with pytest.raises(ValueError, match="out of range"):
            single_step.choose_data(result, choice=5)


# ---------------------------------------------------------------------------
# report_models / report_candidates: registry contents
# ---------------------------------------------------------------------------

class TestReportListing:
    def test_report_models_registry_keys(self):
        """reporting.models should expose the expected report keys."""
        from bayesopt_human.reporting.models import (
            INDIVIDUAL_REPORTS,
            SUMMARY_REPORTS,
        )
        assert [r.key for r in SUMMARY_REPORTS] == [
            "residual_histogram", "length_scales",
            "transform_comparison", "gp_fit",
        ]
        assert [r.key for r in INDIVIDUAL_REPORTS] == [
            "surrogate_observations", "surrogate_progress", "surrogate_pca",
            "hyperparameters", "residuals", "observed_vs_predicted",
            "feature_importance", "marginal_effects", "pairwise_projections",
            "gp_fit",
        ]

    def test_report_candidates_registry_keys(self):
        """reporting.candidates should expose the expected report keys."""
        from bayesopt_human.reporting.candidates import (
            INDIVIDUAL_REPORTS,
            SUMMARY_REPORTS,
        )
        assert [r.key for r in SUMMARY_REPORTS] == ["importance_2d", "pca_2d"]
        assert [r.key for r in INDIVIDUAL_REPORTS] == [
            "observations_raw", "observations_sur",
            "importance_2d", "pca_2d",
        ]


# ---------------------------------------------------------------------------
# choose_candidate save_to
# ---------------------------------------------------------------------------

class TestChooseCandidateSaveTo:
    def test_save_to_csv(self, tmp_path):
        """choose_candidate with save_to should write a param-only CSV."""
        from bayesopt_human.acquisition.candidates import Candidate
        from bayesopt_human.recommendations.schema import Recommendation

        cand = Candidate(
            coordinates_normalized=np.array([0.5, 0.3]),
            coordinates_raw=np.array([5.0, 3.0]),
            coordinates_surrogate=np.array([0.5, 0.3]),
            source="EI",
            category="balanced",
            posterior_mean=1.0,
            posterior_std=0.5,
            expected_improvement=0.1,
            coverage_gain_flat=0.01,
            coverage_gain_surrogate=0.01,
            svm_prediction=True,
        )
        rec = Recommendation(
            source="EI",
            priority=1,
            candidate=cand.coordinates_raw,
            candidate_normalized=cand.coordinates_normalized,
            rationale="test",
            confidence="high",
        )
        data = DataPayload(
            df=pd.DataFrame(),
            X_raw=np.array([[5.0, 3.0]]),
            X_norm=np.array([[0.5, 0.3]]),
            y=np.array([1.0]),
            normalizer=None,
            param_names=["x1", "x2"],
        )
        result = CandidateResult(
            candidates=[cand],
            recommendations=[rec],
            gp_report=None,
            coverage=None,
            modality=None,
            svm_report=None,
            candidate_table=None,
            data=data,
            fitted_gp=None,
        )

        config = OptimizationConfig(
            name="test",
            horizon=50,
            bounds={"x1": (0.0, 10.0), "x2": (0.0, 10.0)},
            objective_column="loss",
        )
        s = OptimizationStep([config])

        save_path = str(tmp_path / "next_point.csv")
        coords = s.choose_candidate(result, choice=1, save_to=save_path)

        assert np.allclose(coords, [5.0, 3.0])

        # Verify CSV
        saved = pd.read_csv(save_path)
        assert list(saved.columns) == ["x1", "x2"]
        assert len(saved) == 1
        assert float(saved["x1"].iloc[0]) == pytest.approx(5.0)
        assert float(saved["x2"].iloc[0]) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

class TestStatePersistence:
    """Integration tests for optimization state persistence."""

    def test_choose_candidate_saves_state(self, tmp_path):
        """Full pipeline should create a state JSON file."""
        from bayesopt_human.acquisition.candidates import Candidate
        from bayesopt_human.recommendations.schema import Recommendation
        from bayesopt_human.optimizer.state import load_state

        cand = Candidate(
            coordinates_normalized=np.array([0.5, 0.3]),
            coordinates_raw=np.array([5.0, 3.0]),
            coordinates_surrogate=np.array([0.5, 0.3]),
            source="EI",
            category="balanced",
            posterior_mean=1.0,
            posterior_std=0.5,
            expected_improvement=0.1,
            coverage_gain_flat=0.01,
            coverage_gain_surrogate=0.01,
            svm_prediction=True,
        )
        rec = Recommendation(
            source="EI",
            priority=1,
            candidate=cand.coordinates_raw,
            candidate_normalized=cand.coordinates_normalized,
            rationale="test",
            confidence="high",
        )
        data = DataPayload(
            df=pd.DataFrame(),
            X_raw=np.array([[5.0, 3.0]]),
            X_norm=np.array([[0.5, 0.3]]),
            y=np.array([1.0]),
            normalizer=None,
            param_names=["x1", "x2"],
        )
        result = CandidateResult(
            candidates=[cand],
            recommendations=[rec],
            gp_report=None,
            coverage=None,
            modality=None,
            svm_report=None,
            candidate_table=None,
            data=data,
            fitted_gp=None,
        )

        config = OptimizationConfig(
            name="test",
            horizon=50,
            bounds={"x1": (0.0, 10.0), "x2": (0.0, 10.0)},
            objective_column="loss",
        )
        config.update_from_data(10, ["x1", "x2"])

        s = OptimizationStep([config])
        s._data_dir = str(tmp_path)
        s._locked_config = config

        # Simulate choose_model by setting _chosen_model_record
        from bayesopt_human.optimizer.state import ModelChoiceRecord
        s._chosen_model_record = ModelChoiceRecord(
            output_transform="none", ard=True, warping=False,
            loo_mae=0.05, loo_lpd=1.5, calibration_var=0.8,
            n_obs_when_chosen=10,
        )

        s.choose_candidate(result, choice=1)

        # Verify state file was created
        state = load_state(str(tmp_path), "test")
        assert state is not None
        assert state.config_name == "test"
        assert len(state.rounds) == 1
        assert state.rounds[0].candidate_source == "EI"
        assert state.rounds[0].candidate_category == "balanced"
        assert state.rounds[0].candidate_confidence == "high"
        assert state.model_choice.output_transform == "none"
        assert state.model_choice.ard is True

    def test_choose_candidate_custom_coordinates_saves_state(self, tmp_path):
        """Custom coordinates should record was_custom=True."""
        from bayesopt_human.optimizer.state import load_state, ModelChoiceRecord

        data = DataPayload(
            df=pd.DataFrame(),
            X_raw=np.array([[5.0, 3.0]]),
            X_norm=np.array([[0.5, 0.3]]),
            y=np.array([1.0]),
            normalizer=None,
            param_names=["x1", "x2"],
        )
        result = CandidateResult(
            candidates=[],
            recommendations=[],
            gp_report=None,
            coverage=None,
            modality=None,
            svm_report=None,
            candidate_table=None,
            data=data,
            fitted_gp=None,
        )

        config = OptimizationConfig(
            name="custom_test",
            horizon=50,
            bounds={"x1": (0.0, 10.0), "x2": (0.0, 10.0)},
            objective_column="loss",
        )
        config.update_from_data(10, ["x1", "x2"])

        s = OptimizationStep([config])
        s._data_dir = str(tmp_path)
        s._locked_config = config
        s._chosen_model_record = ModelChoiceRecord(
            output_transform="log", ard=False, warping=False,
            loo_mae=0.1, loo_lpd=0.5, calibration_var=1.0,
            n_obs_when_chosen=10,
        )

        s.choose_candidate(result, choice=np.array([5.0, 3.0]))

        state = load_state(str(tmp_path), "custom_test")
        assert state is not None
        assert len(state.rounds) == 1
        assert state.rounds[0].candidate_was_custom is True
        assert state.rounds[0].candidate_source == "custom"

    def test_get_models_shows_previous_choice(self, single_step, single_data_dir, capsys):
        """When a state file exists, get_models() should show 'Previous choice'."""
        from bayesopt_human.optimizer.state import (
            ModelChoiceRecord, OptimizationState, save_state, record_round,
        )

        # Run pipeline to get data
        data_result = single_step.get_data(single_data_dir)
        data = single_step.choose_data(data_result, choice=0)

        # Create a state file with a known previous model
        state = OptimizationState(
            schema_version=1,
            config_name="data",
            last_updated="2026-01-01T00:00:00",
            model_choice=ModelChoiceRecord(
                output_transform="none", ard=True, warping=False,
                loo_mae=0.05, loo_lpd=1.5, calibration_var=0.8,
                n_obs_when_chosen=8,
            ),
            rounds=[],
        )
        record_round(
            state, n_obs=8,
            model_choice=state.model_choice,
            candidate_source="EI", candidate_category="balanced",
            candidate_confidence="high",
            candidate_posterior_mean=1.0, candidate_posterior_std=0.2,
            candidate_expected_improvement=0.3, was_custom=False,
        )
        save_state(state, single_data_dir)

        # Run get_models — should display previous choice and recommendation
        model_result = single_step.get_models(data)
        captured = capsys.readouterr()
        assert "Previous choice" in captured.out
        assert "Recommendation" in captured.out
        assert model_result.recommendation is not None

    def test_first_round_no_state(self, single_step, single_data_dir, capsys):
        """Without a state file, get_models() should show INITIAL recommendation."""
        data_result = single_step.get_data(single_data_dir)
        data = single_step.choose_data(data_result, choice=0)
        model_result = single_step.get_models(data)
        captured = capsys.readouterr()
        assert "Previous choice" not in captured.out
        assert "Model Selection Results" in captured.out
        assert "Recommendation (INITIAL)" in captured.out
        assert model_result.recommendation is not None
        assert model_result.recommendation.recommended_idx == 0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Two full pipeline runs with the same ``random_seed`` must match."""

    def _run_pipeline(self, data_dir: str, seed: int | None) -> np.ndarray:
        """Run data → model → candidate once; return the top candidate's
        raw coordinates as a numpy array."""
        config = OptimizationConfig(
            name="data",
            horizon=50,
            bounds={"x1": (0.0, 10.0), "x2": (0.0, 10.0)},
            objective_column="loss",
            direction="minimize",
            random_seed=seed,
        )
        step = OptimizationStep([config])
        data_result = step.get_data(data_dir)
        data = step.choose_data(data_result, choice=0)
        model_result = step.get_models(data)
        fitted_gp = step.choose_model(
            model_result, choice=model_result.recommendation.recommended_idx,
        )
        cand_result = step.get_candidates(data, fitted_gp)
        # Flatten every candidate's raw coordinates into one array for a
        # single bit-identical comparison across runs.
        return np.concatenate([
            np.asarray(c.coordinates_raw, dtype=np.float64)
            for c in cand_result.candidates
        ])

    def test_seeded_runs_are_identical(self, single_data_dir):
        """With a fixed ``random_seed`` the full pipeline must produce
        identical candidate coordinates on repeated runs."""
        run_a = self._run_pipeline(single_data_dir, seed=42)
        run_b = self._run_pipeline(single_data_dir, seed=42)
        np.testing.assert_array_equal(run_a, run_b)
