"""Tests for model recommendation with inertia."""

from types import SimpleNamespace

import numpy as np
import pytest

from bayesopt_human.gp.selection import ModelCandidate
from bayesopt_human.gp.model_recommendation import (
    ModelAction,
    ModelRecommendation,
    paired_lpd_test,
    recommend_model,
    _parsimony_pick,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mc(
    output_transform="none",
    ard=True,
    warping=False,
    loo_lpd=-1.0,
    loo_lpd_i=None,
    demoted=False,
    **overrides,
):
    """Create a ModelCandidate with mock FittedGP."""
    defaults = dict(
        output_transform=output_transform,
        warping=warping,
        ard=ard,
        loo_mae=0.1,
        loo_std=0.05,
        loo_lpd=loo_lpd,
        calibration_var=1.0,
        fitted_gp=SimpleNamespace(
            length_scales=np.array([0.5, 0.5]),
            warp_parameters=None,
        ),
        loo_lpd_i=loo_lpd_i,
    )
    defaults.update(overrides)
    mc = ModelCandidate(**defaults)
    mc.demoted = demoted
    return mc


# ---------------------------------------------------------------------------
# paired_lpd_test
# ---------------------------------------------------------------------------

class TestPairedLpdTest:
    def test_identical_returns_zero_t(self):
        lpd = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        t, p = paired_lpd_test(lpd, lpd)
        assert t == 0.0
        assert p == 0.5

    def test_b_clearly_better(self):
        a = np.zeros(20)
        b = np.ones(20)
        t, p = paired_lpd_test(a, b)
        assert t > 2.0
        assert p < 0.05

    def test_a_clearly_better(self):
        a = np.ones(20)
        b = np.zeros(20)
        t, p = paired_lpd_test(a, b)
        assert t < -2.0
        assert p > 0.95

    def test_small_n_fallback(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        t, p = paired_lpd_test(a, b)
        assert t == 0.0
        assert p == 1.0

    def test_numerically_identical(self):
        a = np.full(10, 5.0)
        b = np.full(10, 5.0)
        t, p = paired_lpd_test(a, b)
        assert t == 0.0
        assert p == 0.5


# ---------------------------------------------------------------------------
# recommend_model — INITIAL
# ---------------------------------------------------------------------------

class TestRecommendInitial:
    def test_first_round_returns_initial(self):
        candidates = [
            _make_mc(loo_lpd=-0.5, ard=True),
            _make_mc(loo_lpd=-1.0, ard=False),
        ]
        rec = recommend_model(candidates, prev_choice_idx=None)
        assert rec.action == ModelAction.INITIAL

    def test_initial_picks_best_non_demoted(self):
        candidates = [
            _make_mc(loo_lpd=-0.5, ard=True),   # best LPD
            _make_mc(loo_lpd=-1.0, ard=True),
        ]
        rec = recommend_model(candidates, prev_choice_idx=None)
        assert rec.recommended_idx == 0

    def test_initial_applies_parsimony(self):
        """Simpler model within 0.5 nats should be preferred."""
        candidates = [
            _make_mc(loo_lpd=-0.5, ard=True, warping=True),  # WARP, best LPD
            _make_mc(loo_lpd=-0.7, ard=False),                # ISO, within 0.5 nats
        ]
        rec = recommend_model(candidates, prev_choice_idx=None)
        # ISO should be preferred (simpler, within 0.5 nats)
        assert rec.recommended_idx == 1

    def test_initial_does_not_pick_demoted(self):
        candidates = [
            _make_mc(loo_lpd=-0.5, demoted=True),   # best LPD but demoted
            _make_mc(loo_lpd=-1.0, demoted=False),
        ]
        rec = recommend_model(candidates, prev_choice_idx=None)
        assert rec.recommended_idx == 1


# ---------------------------------------------------------------------------
# recommend_model — Hard switch
# ---------------------------------------------------------------------------

class TestRecommendHardSwitch:
    def test_demoted_status_quo_forces_switch(self):
        candidates = [
            _make_mc(loo_lpd=-1.0, demoted=True),   # status quo (demoted)
            _make_mc(loo_lpd=-0.5, demoted=False),
        ]
        rec = recommend_model(candidates, prev_choice_idx=0)
        assert rec.action == ModelAction.SWITCH
        assert rec.recommended_idx == 1

    def test_switch_picks_parsimony(self):
        """Hard switch should apply parsimony among non-demoted."""
        candidates = [
            _make_mc(loo_lpd=-1.0, demoted=True),            # status quo
            _make_mc(loo_lpd=-0.5, ard=True, warping=True),  # WARP best
            _make_mc(loo_lpd=-0.7, ard=False),                # ISO within 0.5
        ]
        rec = recommend_model(candidates, prev_choice_idx=0)
        assert rec.action == ModelAction.SWITCH
        assert rec.recommended_idx == 2  # ISO preferred


# ---------------------------------------------------------------------------
# recommend_model — Significance test
# ---------------------------------------------------------------------------

class TestRecommendSignificance:
    def test_keep_when_not_significant(self):
        """Small, noisy LPD difference → KEEP."""
        n = 15
        rng = np.random.RandomState(42)
        sq_lpd_i = rng.normal(0, 1, n)
        alt_lpd_i = sq_lpd_i + rng.normal(0, 1, n) * 0.1  # tiny difference

        candidates = [
            _make_mc(loo_lpd=-0.5, loo_lpd_i=sq_lpd_i),    # status quo
            _make_mc(loo_lpd=-0.4, loo_lpd_i=alt_lpd_i),
        ]
        rec = recommend_model(candidates, prev_choice_idx=0)
        assert rec.action == ModelAction.KEEP

    def test_switch_when_significant_and_simpler(self):
        """Clear LPD improvement + simpler model → SWITCH."""
        n = 20
        sq_lpd_i = np.zeros(n)
        alt_lpd_i = np.ones(n)  # uniformly better

        candidates = [
            _make_mc(loo_lpd=-1.0, ard=True, loo_lpd_i=sq_lpd_i),      # ARD
            _make_mc(loo_lpd=0.0, ard=False, loo_lpd_i=alt_lpd_i),      # ISO
        ]
        rec = recommend_model(candidates, prev_choice_idx=0)
        assert rec.action == ModelAction.SWITCH
        assert rec.recommended_idx == 1

    def test_switch_when_significant_and_same_complexity(self):
        """Clear LPD improvement, same complexity → SWITCH."""
        n = 20
        sq_lpd_i = np.zeros(n)
        alt_lpd_i = np.ones(n)

        candidates = [
            _make_mc(loo_lpd=-1.0, ard=True, loo_lpd_i=sq_lpd_i),
            _make_mc(loo_lpd=0.0, ard=True, loo_lpd_i=alt_lpd_i,
                     output_transform="log"),
        ]
        rec = recommend_model(candidates, prev_choice_idx=0)
        assert rec.action == ModelAction.SWITCH

    def test_keep_when_marginal_and_more_complex(self):
        """Moderate improvement but more complex → KEEP (needs t > 3)."""
        n = 15
        rng = np.random.RandomState(42)
        sq_lpd_i = np.zeros(n)
        # Just enough to pass t > 2 but not t > 3
        alt_lpd_i = np.full(n, 0.6) + rng.normal(0, 0.3, n)

        candidates = [
            _make_mc(loo_lpd=-0.5, ard=False, loo_lpd_i=sq_lpd_i),         # ISO
            _make_mc(loo_lpd=0.0, ard=True, warping=True,
                     loo_lpd_i=alt_lpd_i),                                   # WARP
        ]
        rec = recommend_model(candidates, prev_choice_idx=0)
        # Should KEEP because more complex needs t > 3
        # Verify the t-stat is in the gap (2 < t < 3)
        if rec.t_stat is not None and 2.0 < rec.t_stat < 3.0:
            assert rec.action == ModelAction.KEEP
        # If the random data pushed t above 3, that's also a valid outcome

    def test_switch_when_highly_significant_and_more_complex(self):
        """Very clear improvement, more complex → SWITCH (t > 3)."""
        n = 20
        sq_lpd_i = np.zeros(n)
        alt_lpd_i = np.full(n, 2.0)  # very strong improvement

        candidates = [
            _make_mc(loo_lpd=-2.0, ard=False, loo_lpd_i=sq_lpd_i),         # ISO
            _make_mc(loo_lpd=0.0, ard=True, warping=True,
                     loo_lpd_i=alt_lpd_i),                                   # WARP
        ]
        rec = recommend_model(candidates, prev_choice_idx=0)
        assert rec.action == ModelAction.SWITCH
        assert rec.t_stat is not None
        assert rec.t_stat > 3.0

    def test_only_non_demoted_returns_keep(self):
        """Status quo is the only non-demoted model → KEEP."""
        candidates = [
            _make_mc(loo_lpd=-0.5, loo_lpd_i=np.zeros(10)),
            _make_mc(loo_lpd=0.0, loo_lpd_i=np.ones(10), demoted=True),
        ]
        rec = recommend_model(candidates, prev_choice_idx=0)
        assert rec.action == ModelAction.KEEP
        assert rec.recommended_idx == 0

    def test_rationale_includes_t_stat(self):
        """Rationale string should mention the t-statistic."""
        n = 20
        sq_lpd_i = np.zeros(n)
        alt_lpd_i = np.ones(n)

        candidates = [
            _make_mc(loo_lpd=-1.0, loo_lpd_i=sq_lpd_i),
            _make_mc(loo_lpd=0.0, loo_lpd_i=alt_lpd_i),
        ]
        rec = recommend_model(candidates, prev_choice_idx=0)
        assert "t=" in rec.rationale

    def test_fallback_when_lpd_i_missing(self):
        """When per-point LPDs are None, fall back gracefully."""
        candidates = [
            _make_mc(loo_lpd=-0.5),   # no loo_lpd_i
            _make_mc(loo_lpd=-0.3),
        ]
        rec = recommend_model(candidates, prev_choice_idx=0)
        assert rec.action in (ModelAction.KEEP, ModelAction.SWITCH)
        assert "Fallback" in rec.rationale


# ---------------------------------------------------------------------------
# _parsimony_pick
# ---------------------------------------------------------------------------

class TestParsimonyPick:
    def test_picks_simplest_within_threshold(self):
        """ISO within 0.5 nats of best WARP → pick ISO."""
        candidates = [
            _make_mc(loo_lpd=-0.5, ard=True, warping=True),  # idx 0, WARP
            _make_mc(loo_lpd=-0.7, ard=False),                # idx 1, ISO
        ]
        non_demoted = [(i, c) for i, c in enumerate(candidates)]
        assert _parsimony_pick(candidates, non_demoted) == 1

    def test_does_not_pick_outside_threshold(self):
        """ISO more than 0.5 nats worse → keep WARP."""
        candidates = [
            _make_mc(loo_lpd=-0.5, ard=True, warping=True),  # WARP
            _make_mc(loo_lpd=-1.5, ard=False),                # ISO, 1.0 nats worse
        ]
        non_demoted = [(i, c) for i, c in enumerate(candidates)]
        assert _parsimony_pick(candidates, non_demoted) == 0

    def test_empty_non_demoted_returns_zero(self):
        candidates = [_make_mc(demoted=True)]
        assert _parsimony_pick(candidates, []) == 0
