"""Tests for model selection workflow."""

from types import SimpleNamespace

import numpy as np
import pytest

from bayesopt_human.config import OptimizationConfig
from bayesopt_human.gp.selection import (
    ModelCandidate,
    ModelSelector,
    _clipped_log_eligible,
    length_scale_flags,
    model_n_params,
    warp_flags,
)


@pytest.fixture
def selection_config():
    return OptimizationConfig(
        name="test",
        horizon=50,
        bounds={"x1": (0.0, 1.0), "x2": (0.0, 1.0)},
        objective_column="y",
    )


@pytest.fixture
def positive_data():
    np.random.seed(42)
    X = np.random.rand(15, 2) * 0.99
    y = np.exp(np.sum(X, axis=1))  # All positive
    return X, y


@pytest.fixture
def negative_data():
    np.random.seed(42)
    X = np.random.rand(15, 2) * 0.99
    y = -np.sum(X**2, axis=1)  # All negative
    return X, y


class TestClippedLogEligibility:
    # -- Maximization tests --

    def test_maximize_all_positive_ineligible(self):
        """All positive: no negative extreme to suppress."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert _clipped_log_eligible(y, "maximize") is False

    def test_maximize_all_negative_ineligible(self):
        y = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
        assert _clipped_log_eligible(y, "maximize") is False

    def test_maximize_too_few_positive_ineligible(self):
        """Only 3 positive values (need >= 5)."""
        y = np.array([1.0, 2.0, 3.0, -1000.0, -2000.0, -3000.0, -4000.0, -5000.0])
        assert _clipped_log_eligible(y, "maximize") is False

    def test_maximize_negatives_not_extreme_enough(self):
        """5 positives but negatives aren't 100x larger."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, -4.0, -5.0])
        assert _clipped_log_eligible(y, "maximize") is False

    def test_maximize_extreme_negatives_eligible(self):
        """5+ positives and abs(min) >= 100 * max."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, -1000.0, -2000.0, -3000.0, -4000.0, -5000.0])
        assert _clipped_log_eligible(y, "maximize") is True

    def test_maximize_boundary_100x_eligible(self):
        """Exactly at 100x boundary."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, -500.0])
        assert _clipped_log_eligible(y, "maximize") is True

    def test_maximize_boundary_just_under_100x_ineligible(self):
        """Just below 100x boundary."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, -499.0])
        assert _clipped_log_eligible(y, "maximize") is False

    # -- Minimization tests --

    def test_minimize_always_ineligible(self):
        """clipped_log censors negatives, which is where the optimum lives."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, -1000.0, -2000.0, -3000.0])
        assert _clipped_log_eligible(y, "minimize") is False

    def test_minimize_all_positive_ineligible(self):
        """Even all-positive data: log already handles this."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert _clipped_log_eligible(y, "minimize") is False

    def test_default_direction_is_minimize(self):
        """Default direction param is minimize."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, -5000.0])
        assert _clipped_log_eligible(y) is False


class TestEvaluateTransforms:
    def test_includes_ard_and_non_ard(self, selection_config, positive_data):
        X, y = positive_data
        selector = ModelSelector(selection_config)
        candidates = selector.evaluate_transforms(X, y)

        ard_candidates = [c for c in candidates if c.ard]
        non_ard_candidates = [c for c in candidates if not c.ard]

        assert len(ard_candidates) > 0
        assert len(non_ard_candidates) > 0

    def test_has_lpd(self, selection_config, positive_data):
        X, y = positive_data
        selector = ModelSelector(selection_config)
        candidates = selector.evaluate_transforms(X, y)

        for c in candidates:
            assert np.isfinite(c.loo_lpd)

    def test_skips_clipped_log_for_negative_data(self, selection_config, negative_data):
        X, y = negative_data
        selector = ModelSelector(selection_config)
        candidates = selector.evaluate_transforms(X, y)

        clipped_log_candidates = [c for c in candidates if c.output_transform == "clipped_log"]
        assert len(clipped_log_candidates) == 0

def _make_candidate(length_scales, ard=True, warping=False,
                    warp_c0=None, warp_c1=None, **overrides):
    """Create a ModelCandidate with a mock FittedGP for unit testing."""
    ls = np.array(length_scales)
    warp_params = None
    if warp_c0 is not None and warp_c1 is not None:
        warp_params = {
            "concentration0": np.array(warp_c0),
            "concentration1": np.array(warp_c1),
        }
    fitted_gp = SimpleNamespace(length_scales=ls, warp_parameters=warp_params)
    defaults = dict(
        output_transform="none",
        warping=warping,
        ard=ard,
        loo_mae=0.1,
        loo_std=0.05,
        loo_lpd=-1.0,
        calibration_var=1.0,
        fitted_gp=fitted_gp,
    )
    defaults.update(overrides)
    return ModelCandidate(**defaults)


class TestLengthScaleFlags:
    """Tests for length_scale_flags with symmetric log-space boundary detection."""

    def test_green_when_clean(self):
        """Length scales well within bounds get green."""
        mc = _make_candidate([0.1, 0.5, 1.0, 10.0])
        flags, rag = length_scale_flags(mc, ls_lower_bound=1e-3, ls_upper_bound=1e3)
        assert rag == 'green'
        assert flags == []

    def test_boundary_detection_symmetric_log_space(self):
        """Values near lower and upper bounds are detected symmetrically.

        With bounds (1e-3, 1e3): log-range = 13.816, tol = 0.691.
        Lower zone: ls <= exp(-6.908 + 0.691) ≈ 0.002.
        Upper zone: ls >= exp(6.908 - 0.691) ≈ 501.
        """
        mc_lower = _make_candidate([0.0015, 0.5])  # 1 near lower, 1 clean
        flags, rag = length_scale_flags(mc_lower, ls_lower_bound=1e-3, ls_upper_bound=1e3)
        assert rag in ('amber', 'red')
        assert any("at boundary" in f for f in flags)

        mc_upper = _make_candidate([0.5, 600.0])  # 1 clean, 1 near upper
        flags, rag = length_scale_flags(mc_upper, ls_lower_bound=1e-3, ls_upper_bound=1e3)
        assert rag in ('amber', 'red')
        assert any("at boundary" in f for f in flags)

    def test_red_threshold_majority_at_bounds(self):
        """Red when a strict majority of length scales are at bounds."""
        # 4 dims, 3 at lower bound → >= (4+2)//2 = 3 → red
        mc = _make_candidate([0.001, 0.001, 0.001, 1.0])
        flags, rag = length_scale_flags(mc, ls_lower_bound=1e-3, ls_upper_bound=1e3)
        assert rag == 'red'
        assert any("boundary" in f for f in flags)

    def test_amber_fewer_than_majority_at_bounds(self):
        """Amber when some (< majority) length scales are at bounds."""
        # 4 dims, 2 at lower → < (4+2)//2 = 3 → amber
        mc = _make_candidate([0.001, 0.001, 1.0, 10.0])
        flags, rag = length_scale_flags(mc, ls_lower_bound=1e-3, ls_upper_bound=1e3)
        assert rag == 'amber'
        assert any("at boundary" in f for f in flags)

    def test_amber_for_high_ratio(self):
        """Amber when max/min ratio exceeds 100."""
        mc = _make_candidate([0.01, 5.0])  # ratio = 500
        flags, rag = length_scale_flags(mc, ls_lower_bound=1e-3, ls_upper_bound=1e3)
        assert rag == 'amber'
        assert any("ratio" in f for f in flags)

    def test_none_length_scales_returns_green(self):
        """Graceful handling when length_scales is None."""
        mc = _make_candidate([1.0])
        mc.fitted_gp.length_scales = None
        flags, rag = length_scale_flags(mc)
        assert rag == 'green'
        assert flags == []


class TestWarpFlags:
    """Tests for warp_flags with concentration threshold detection."""

    def test_green_when_not_warped(self):
        mc = _make_candidate([0.5, 1.0], warping=False)
        flags, rag = warp_flags(mc)
        assert rag == 'green'
        assert flags == []

    def test_green_when_warp_parameters_none(self):
        """Graceful handling when warp_parameters is None despite warping=True."""
        mc = _make_candidate([0.5, 1.0], warping=True)
        flags, rag = warp_flags(mc)
        assert rag == 'green'
        assert flags == []

    def test_green_when_clean(self):
        mc = _make_candidate(
            [0.5, 1.0], warping=True,
            warp_c0=[1.0, 1.5], warp_c1=[1.0, 0.8],
        )
        flags, rag = warp_flags(mc)
        assert rag == 'green'
        assert flags == []

    def test_amber_one_dim_low_c0(self):
        """One of 3 dimensions with c0 < 0.1 gets amber (minority)."""
        mc = _make_candidate(
            [0.5, 1.0, 0.8], warping=True,
            warp_c0=[0.05, 1.0, 1.5], warp_c1=[1.0, 1.0, 1.0],
        )
        flags, rag = warp_flags(mc)
        assert rag == 'amber'
        assert any("extreme" in f for f in flags)

    def test_amber_one_dim_high_c1(self):
        """One of 3 dimensions with c1 > 20 gets amber."""
        mc = _make_candidate(
            [0.5, 1.0, 0.8], warping=True,
            warp_c0=[1.0, 1.0, 1.5], warp_c1=[25.0, 1.0, 1.0],
        )
        flags, rag = warp_flags(mc)
        assert rag == 'amber'
        assert any("extreme" in f for f in flags)

    def test_red_majority_extreme(self):
        """Red when strict majority of dimensions are extreme."""
        # 4 dims, 3 extreme -> >= (4+2)//2 = 3 -> red
        mc = _make_candidate(
            [0.5, 1.0, 0.8, 0.3], warping=True,
            warp_c0=[0.05, 25.0, 0.03, 1.0], warp_c1=[1.0, 1.0, 1.0, 1.0],
        )
        flags, rag = warp_flags(mc)
        assert rag == 'red'

    def test_red_all_extreme(self):
        mc = _make_candidate(
            [0.5, 1.0], warping=True,
            warp_c0=[0.05, 30.0], warp_c1=[0.02, 25.0],
        )
        flags, rag = warp_flags(mc)
        assert rag == 'red'

    def test_amber_minority_extreme(self):
        """Amber when fewer than majority are extreme."""
        # 4 dims, 2 extreme -> < (4+2)//2 = 3 -> amber
        mc = _make_candidate(
            [0.5, 1.0, 0.8, 0.3], warping=True,
            warp_c0=[0.05, 25.0, 1.0, 1.0], warp_c1=[1.0, 1.0, 1.0, 1.0],
        )
        flags, rag = warp_flags(mc)
        assert rag == 'amber'

    def test_both_concentrations_checked(self):
        """c1 extreme triggers flag even if c0 is fine."""
        mc = _make_candidate(
            [0.5, 1.0], warping=True,
            warp_c0=[1.0, 1.0], warp_c1=[0.05, 1.0],
        )
        flags, rag = warp_flags(mc)
        assert rag in ('amber', 'red')
        assert len(flags) > 0

    def test_custom_thresholds(self):
        mc = _make_candidate(
            [0.5, 1.0], warping=True,
            warp_c0=[0.5, 1.0], warp_c1=[1.0, 1.0],
        )
        # Default thresholds: c0=0.5 is fine
        flags, rag = warp_flags(mc)
        assert rag == 'green'
        # Stricter low threshold: c0=0.5 is now extreme
        flags, rag = warp_flags(mc, low_threshold=1.0)
        assert rag in ('amber', 'red')

    def test_boundary_at_low_threshold_not_flagged(self):
        """Value exactly at low_threshold is NOT flagged (strict <)."""
        mc = _make_candidate(
            [0.5, 1.0], warping=True,
            warp_c0=[0.1, 1.0], warp_c1=[1.0, 1.0],
        )
        flags, rag = warp_flags(mc)
        assert rag == 'green'

    def test_boundary_at_high_threshold_not_flagged(self):
        """Value exactly at high_threshold is NOT flagged (strict >)."""
        mc = _make_candidate(
            [0.5, 1.0], warping=True,
            warp_c0=[20.0, 1.0], warp_c1=[1.0, 1.0],
        )
        flags, rag = warp_flags(mc)
        assert rag == 'green'

    def test_flag_text_includes_count(self):
        mc = _make_candidate(
            [0.5, 1.0, 0.8], warping=True,
            warp_c0=[0.05, 25.0, 1.0], warp_c1=[1.0, 1.0, 1.0],
        )
        flags, rag = warp_flags(mc)
        assert any("2" in f for f in flags)


class TestModelNParams:
    def test_iso(self):
        mc = _make_candidate([0.5, 0.5], ard=False, warping=False)
        assert model_n_params(mc) == 3

    def test_ard_2d(self):
        mc = _make_candidate([0.5, 1.0], ard=True, warping=False)
        assert model_n_params(mc) == 4  # D + 2

    def test_ard_5d(self):
        mc = _make_candidate([0.5] * 5, ard=True, warping=False)
        assert model_n_params(mc) == 7  # 5 + 2

    def test_warp_2d(self):
        mc = _make_candidate([0.5, 1.0], ard=True, warping=True)
        assert model_n_params(mc) == 8  # 3*2 + 2

    def test_warp_5d(self):
        mc = _make_candidate([0.5] * 5, ard=True, warping=True)
        assert model_n_params(mc) == 17  # 3*5 + 2
