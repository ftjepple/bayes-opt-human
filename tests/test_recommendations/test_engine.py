"""Tests for the 5-phase recommendation engine."""

import numpy as np
import pytest

from bayesopt_human.acquisition.candidates import Candidate
from bayesopt_human.config import OptimizationConfig
from bayesopt_human.diagnostics.coverage import CoverageReport
from bayesopt_human.diagnostics.gp_diagnostics import GPFitReport
from bayesopt_human.diagnostics.svm import SVMReport
from bayesopt_human.recommendations.engine import RecommendationEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    n_obs=20, remaining=10, horizon=50, warmstart=0, direction="minimize",
):
    config = OptimizationConfig(
        name="test",
        horizon=horizon,
        bounds={"x1": (0.0, 10.0), "x2": (0.0, 10.0)},
        objective_column="y",
        direction=direction,
        warmstart=warmstart,
    )
    config.n_obs = n_obs
    config.remaining = remaining
    config.n_dim = 2
    return config


def _make_candidate(
    source="Strategy",
    category="balanced",
    posterior_mean=1.0,
    posterior_std=0.5,
    ei=0.1,
    coverage_flat=0.01,
    coverage_sur=0.01,
    svm_prediction=True,
    coords_norm=None,
):
    if coords_norm is None:
        coords_norm = np.array([0.5, 0.5])
    return Candidate(
        coordinates_normalized=coords_norm,
        coordinates_raw=coords_norm * 10.0,
        coordinates_surrogate=coords_norm,
        source=source,
        category=category,
        posterior_mean=posterior_mean,
        posterior_std=posterior_std,
        expected_improvement=ei,
        coverage_gain_flat=coverage_flat,
        coverage_gain_surrogate=coverage_sur,
        svm_prediction=svm_prediction,
    )


def _make_gp_report(calibration_var=1.0, loo_mae=0.05, conditioning_warning=False):
    return GPFitReport(
        loo_mae=loo_mae,
        loo_std=0.02,
        loo_lpd=-1.0,
        calibration_var=calibration_var,
        conditioning=10.0,
        jitter_used=1e-6,
        conditioning_warning=conditioning_warning,
        length_scales=np.array([0.5, 0.5]),
        feature_importances=np.array([2.0, 2.0]),
    )


def _make_coverage(avg_knn=0.3, ratio_to_optimal=1.5):
    return CoverageReport(
        avg_knn_distance_flat=avg_knn,
        min_knn_distance_flat=0.1,
        avg_knn_distance_surrogate=avg_knn,
        min_knn_distance_surrogate=0.1,
        optimal_baseline_flat=0.2,
        ratio_to_optimal_flat=ratio_to_optimal,
        optimal_baseline_surrogate=0.2,
        ratio_to_optimal_surrogate=ratio_to_optimal,
        k=5,
    )


def _make_svm_report(is_reliable=True):
    from sklearn.svm import SVC
    svm = SVC(kernel="rbf", probability=True)
    svm.fit(np.array([[0, 0], [1, 1]]), np.array([0, 1]))
    return SVMReport(
        svm=svm,
        loo_balanced_accuracy=0.7,
        confidence_interval=(0.5, 0.9),
        is_reliable=is_reliable,
        reliability_message="",
        scaler_mean=np.array([0.5, 0.5]),
        scaler_std=np.array([0.3, 0.3]),
    )


def _improving_y(n, direction="minimize"):
    """Objective values with steady improvement (no stagnation)."""
    if direction == "minimize":
        return np.linspace(10.0, 1.0, n)
    return np.linspace(1.0, 10.0, n)


def _stagnant_y(n, stag_length, direction="minimize"):
    """Objective values that improve then stagnate."""
    n_improve = n - stag_length
    if direction == "minimize":
        improving = np.linspace(10.0, 1.0, max(n_improve, 1))
        flat = np.full(stag_length, 5.0)
    else:
        improving = np.linspace(1.0, 10.0, max(n_improve, 1))
        flat = np.full(stag_length, 5.0)
    return np.concatenate([improving, flat])


# ---------------------------------------------------------------------------
# Phase 1: NORMALIZE
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_basic_ranking(self):
        """Candidates with distinct metrics get distinct percentiles."""
        config = _make_config()
        engine = RecommendationEngine(config)

        candidates = [
            _make_candidate("A", posterior_mean=3.0, posterior_std=0.1, ei=0.01),
            _make_candidate("B", posterior_mean=2.0, posterior_std=0.5, ei=0.05),
            _make_candidate("C", posterior_mean=1.0, posterior_std=1.0, ei=0.10),
        ]

        pct = engine._normalize(candidates, 3)

        # For minimize: lower mean = better = higher percentile
        assert pct["p_exploit"][2] > pct["p_exploit"][0]  # C (mean=1) > A (mean=3)
        # Higher std = higher explore percentile
        assert pct["p_explore"][2] > pct["p_explore"][0]
        # Higher EI = higher EI percentile
        assert pct["p_ei"][2] > pct["p_ei"][0]

    def test_ties_get_average(self):
        """Tied values produce identical percentiles."""
        config = _make_config()
        engine = RecommendationEngine(config)

        candidates = [
            _make_candidate("A", ei=0.1),
            _make_candidate("B", ei=0.1),
            _make_candidate("C", ei=0.1),
        ]

        pct = engine._normalize(candidates, 3)
        assert pct["p_ei"][0] == pct["p_ei"][1] == pct["p_ei"][2]

    def test_single_candidate(self):
        """Single candidate gets 0.5 percentile."""
        config = _make_config()
        engine = RecommendationEngine(config)

        candidates = [_make_candidate("A")]
        pct = engine._normalize(candidates, 1)

        assert pct["p_exploit"][0] == 0.5
        assert pct["p_explore"][0] == 0.5

    def test_maximize_direction(self):
        """For maximize: higher mean = better = higher percentile."""
        config = _make_config(direction="maximize")
        engine = RecommendationEngine(config)

        candidates = [
            _make_candidate("A", posterior_mean=1.0),
            _make_candidate("B", posterior_mean=3.0),
        ]

        pct = engine._normalize(candidates, 2)
        assert pct["p_exploit"][1] > pct["p_exploit"][0]  # B (mean=3) > A (mean=1)


# ---------------------------------------------------------------------------
# Phase 2: ASSESS
# ---------------------------------------------------------------------------

class TestAssess:
    def test_urgency_at_start(self):
        """At start of optimization, urgency is low."""
        config = _make_config(n_obs=5, remaining=45, horizon=50)
        engine = RecommendationEngine(config)
        y = _improving_y(5)

        urgency, _, _ = engine._assess(_make_gp_report(), y)
        assert urgency < 0.2

    def test_urgency_at_end(self):
        """At final evaluation, urgency is 1.0."""
        config = _make_config(n_obs=49, remaining=1, horizon=50)
        engine = RecommendationEngine(config)
        y = _improving_y(49)

        urgency, _, _ = engine._assess(_make_gp_report(), y)
        assert urgency == pytest.approx(1.0)

    def test_gp_quality_perfect(self):
        """calibration_var=1.0 gives gp_quality=1.0."""
        config = _make_config()
        engine = RecommendationEngine(config)
        y = _improving_y(20)

        _, gp_quality, _ = engine._assess(_make_gp_report(calibration_var=1.0), y)
        assert gp_quality == pytest.approx(1.0)

    def test_gp_quality_poor(self):
        """Far-from-1.0 calibration_var gives low gp_quality."""
        config = _make_config()
        engine = RecommendationEngine(config)
        y = _improving_y(20)

        _, gp_quality, _ = engine._assess(_make_gp_report(calibration_var=0.01), y)
        assert gp_quality < 0.2

    def test_stagnation_pressure(self):
        """Stagnating observations produce positive stagnation pressure."""
        config = _make_config(n_obs=20, remaining=30, horizon=50)
        engine = RecommendationEngine(config)
        y = _stagnant_y(20, stag_length=10)

        _, _, stag = engine._assess(_make_gp_report(), y)
        assert stag > 0.3

    def test_no_stagnation(self):
        """Steadily improving observations have near-zero stagnation."""
        config = _make_config(n_obs=20, remaining=30, horizon=50)
        engine = RecommendationEngine(config)
        y = _improving_y(20)

        _, _, stag = engine._assess(_make_gp_report(), y)
        assert stag < 0.1


# ---------------------------------------------------------------------------
# Phase 3: SCORE
# ---------------------------------------------------------------------------

class TestScore:
    def test_late_stage_favors_exploitation(self):
        """Late stage (high urgency) should rank exploitation higher."""
        config = _make_config(n_obs=45, remaining=5, horizon=50)
        engine = RecommendationEngine(config)
        y = _improving_y(45)

        candidates = [
            _make_candidate("Explorer", posterior_mean=5.0, posterior_std=2.0, ei=0.01),
            _make_candidate("Exploiter", posterior_mean=0.5, posterior_std=0.1, ei=0.05),
        ]

        recs = engine.generate(
            candidates, _make_gp_report(), _make_coverage(),
            _make_svm_report(), y,
        )

        assert recs[0].source == "Exploiter"

    def test_poor_gp_forces_exploration(self):
        """Poor GP quality drives α up, favoring exploration."""
        config = _make_config(n_obs=20, remaining=30, horizon=50)
        engine = RecommendationEngine(config)
        y = _improving_y(20)

        candidates = [
            _make_candidate("Explorer", posterior_mean=5.0, posterior_std=2.0, ei=0.01),
            _make_candidate("Exploiter", posterior_mean=0.5, posterior_std=0.1, ei=0.01),
        ]

        recs = engine.generate(
            candidates, _make_gp_report(calibration_var=0.01), _make_coverage(),
            _make_svm_report(), y,
        )

        assert recs[0].source == "Explorer"

    def test_final_evaluation_pure_exploit(self):
        """With remaining=1, exploitation should dominate."""
        config = _make_config(n_obs=49, remaining=1, horizon=50)
        engine = RecommendationEngine(config)
        y = _improving_y(49)

        candidates = [
            _make_candidate("Explorer", posterior_mean=5.0, posterior_std=2.0, ei=0.01),
            _make_candidate("Exploiter", posterior_mean=0.1, posterior_std=0.1, ei=0.01),
        ]

        recs = engine.generate(
            candidates, _make_gp_report(), _make_coverage(),
            _make_svm_report(), y,
        )

        assert recs[0].source == "Exploiter"

    def test_stagnation_forces_exploration(self):
        """Stagnation pressure pushes α up, favoring exploration."""
        config = _make_config(n_obs=20, remaining=30, horizon=50)
        engine = RecommendationEngine(config)
        y = _stagnant_y(20, stag_length=15)

        candidates = [
            _make_candidate("Explorer", posterior_mean=5.0, posterior_std=2.0, ei=0.01),
            _make_candidate("Exploiter", posterior_mean=0.5, posterior_std=0.1, ei=0.01),
        ]

        recs = engine.generate(
            candidates, _make_gp_report(), _make_coverage(),
            _make_svm_report(), y,
        )

        assert recs[0].source == "Explorer"


# ---------------------------------------------------------------------------
# Phase 4: DIVERSIFY
# ---------------------------------------------------------------------------

class TestDiversify:
    def test_near_duplicates_demoted(self):
        """Candidates at same location should not both rank at top."""
        config = _make_config()
        engine = RecommendationEngine(config)
        y = _improving_y(20)

        # Two candidates at nearly identical locations, one better on EI
        candidates = [
            _make_candidate("A", ei=0.3, coords_norm=np.array([0.5, 0.5])),
            _make_candidate("B", ei=0.29, coords_norm=np.array([0.501, 0.501])),
            _make_candidate("C", ei=0.1, coords_norm=np.array([0.1, 0.1])),
        ]

        recs = engine.generate(
            candidates, _make_gp_report(), _make_coverage(avg_knn=0.3),
            _make_svm_report(), y,
        )

        # B should be demoted below C due to proximity to A
        sources = [r.source for r in recs]
        assert sources.index("C") < sources.index("B")

    def test_distant_candidates_not_affected(self):
        """Well-separated candidates maintain score-based ordering."""
        config = _make_config()
        engine = RecommendationEngine(config)
        y = _improving_y(20)

        candidates = [
            _make_candidate("A", ei=0.3, coords_norm=np.array([0.1, 0.1])),
            _make_candidate("B", ei=0.2, coords_norm=np.array([0.5, 0.5])),
            _make_candidate("C", ei=0.1, coords_norm=np.array([0.9, 0.9])),
        ]

        recs = engine.generate(
            candidates, _make_gp_report(), _make_coverage(avg_knn=0.3),
            _make_svm_report(), y,
        )

        sources = [r.source for r in recs]
        assert sources[0] == "A"  # Highest EI


# ---------------------------------------------------------------------------
# Phase 5: ANNOTATE
# ---------------------------------------------------------------------------

class TestAnnotate:
    def test_confidence_varies(self):
        """Different candidates should get different confidence levels."""
        config = _make_config()
        engine = RecommendationEngine(config)
        y = _improving_y(20)

        candidates = [
            _make_candidate("High", posterior_std=0.01, svm_prediction=True,
                            coords_norm=np.array([0.1, 0.1])),
            _make_candidate("Mid", posterior_std=0.5, svm_prediction=True,
                            coords_norm=np.array([0.5, 0.5])),
            _make_candidate("Low", posterior_std=1.0, svm_prediction=False,
                            coords_norm=np.array([0.9, 0.9])),
        ]

        recs = engine.generate(
            candidates, _make_gp_report(), _make_coverage(),
            _make_svm_report(is_reliable=True), y,
        )

        confidences = {r.source: r.confidence for r in recs}
        # With well-calibrated GP, low-std + SVM-agree should have highest confidence
        assert confidences["High"] == "high"
        assert confidences["Low"] == "low"

    def test_rationale_mentions_alpha(self):
        """Every rationale should mention α."""
        config = _make_config()
        engine = RecommendationEngine(config)
        y = _improving_y(20)

        candidates = [_make_candidate("A"), _make_candidate("B")]

        recs = engine.generate(
            candidates, _make_gp_report(), _make_coverage(),
            _make_svm_report(), y,
        )

        for rec in recs:
            assert "α=" in rec.rationale

    def test_stagnation_in_rationale(self):
        """When stagnation is high, rationale should mention it."""
        config = _make_config(n_obs=20, remaining=30, horizon=50)
        engine = RecommendationEngine(config)
        y = _stagnant_y(20, stag_length=15)

        candidates = [_make_candidate("A")]

        recs = engine.generate(
            candidates, _make_gp_report(), _make_coverage(),
            _make_svm_report(), y,
        )

        assert "stagnation" in recs[0].rationale.lower()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_candidates(self):
        """Empty candidate list returns empty recommendations."""
        config = _make_config()
        engine = RecommendationEngine(config)

        recs = engine.generate(
            [], _make_gp_report(), _make_coverage(),
            _make_svm_report(), _improving_y(20),
        )

        assert recs == []

    def test_all_identical_metrics(self):
        """Candidates with identical metrics should all be ranked."""
        config = _make_config()
        engine = RecommendationEngine(config)
        y = _improving_y(20)

        candidates = [
            _make_candidate("A", coords_norm=np.array([0.2, 0.2])),
            _make_candidate("B", coords_norm=np.array([0.5, 0.5])),
            _make_candidate("C", coords_norm=np.array([0.8, 0.8])),
        ]

        recs = engine.generate(
            candidates, _make_gp_report(), _make_coverage(),
            _make_svm_report(), y,
        )

        assert len(recs) == 3
        priorities = {r.priority for r in recs}
        assert priorities == {1, 2, 3}

    def test_single_candidate(self):
        """A single candidate should still produce a valid recommendation."""
        config = _make_config()
        engine = RecommendationEngine(config)
        y = _improving_y(20)

        candidates = [_make_candidate("Only")]

        recs = engine.generate(
            candidates, _make_gp_report(), _make_coverage(),
            _make_svm_report(), y,
        )

        assert len(recs) == 1
        assert recs[0].priority == 1
        assert recs[0].source == "Only"

    def test_unreliable_svm_no_penalty(self):
        """When SVM is unreliable, svm_prediction=False should not penalize."""
        config = _make_config()
        engine = RecommendationEngine(config)
        y = _improving_y(20)

        candidates = [
            _make_candidate("A", svm_prediction=False, posterior_std=0.01,
                            coords_norm=np.array([0.3, 0.3])),
            _make_candidate("B", svm_prediction=True, posterior_std=0.01,
                            coords_norm=np.array([0.7, 0.7])),
        ]

        recs = engine.generate(
            candidates, _make_gp_report(), _make_coverage(),
            _make_svm_report(is_reliable=False), y,
        )

        # With unreliable SVM, both should get same confidence
        conf_a = next(r.confidence for r in recs if r.source == "A")
        conf_b = next(r.confidence for r in recs if r.source == "B")
        assert conf_a == conf_b

    def test_warmstart_respected(self):
        """Warmstart observations should be excluded from stagnation calc."""
        # 5 warmstart + 15 optimization, all stagnant in opt phase
        config = _make_config(n_obs=20, remaining=30, horizon=50, warmstart=5)
        engine = RecommendationEngine(config)

        # First 5 improving (warmstart), then 15 flat (stagnant opt phase)
        y_warm = np.linspace(10.0, 5.0, 5)
        y_opt = np.full(15, 5.0)
        y = np.concatenate([y_warm, y_opt])

        _, _, stag = engine._assess(_make_gp_report(), y)
        # 14 stagnant out of 15 opt observations
        assert stag > 0.5
