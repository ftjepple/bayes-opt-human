"""Model recommendation with inertia (KEEP / SWITCH / INITIAL)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import t as t_dist

from bayesopt_human.gp.selection import ModelCandidate

logger = logging.getLogger("bayesopt_human")


class ModelAction(str, Enum):
    INITIAL = "initial"
    KEEP = "keep"
    SWITCH = "switch"


@dataclass
class ModelRecommendation:
    """Result of the model recommendation logic."""

    action: ModelAction
    recommended_idx: int
    rationale: str
    t_stat: float | None = None
    p_value: float | None = None


_COMPLEXITY = {"ISO": 0, "ARD": 1, "WARP": 2}


def _kernel_label(mc: ModelCandidate) -> str:
    return "WARP" if mc.warping else ("ARD" if mc.ard else "ISO")


def _complexity_rank(mc: ModelCandidate) -> int:
    return _COMPLEXITY[_kernel_label(mc)]


def paired_lpd_test(
    lpd_i_a: np.ndarray,
    lpd_i_b: np.ndarray,
) -> tuple[float, float]:
    """Paired t-test: is model B significantly better than model A?

    Tests H0: mean(lpd_B - lpd_A) <= 0 (one-sided).
    Positive t_stat means B has higher per-point LPD on average.

    Returns:
        (t_stat, p_value) where p_value is one-sided (right tail).
    """
    d = lpd_i_b - lpd_i_a
    n = len(d)
    if n < 3:
        return 0.0, 1.0
    d_mean = float(np.mean(d))
    d_std = float(np.std(d, ddof=1))
    if d_std < 1e-15:
        # All differences identical — either no difference or deterministic
        if abs(d_mean) < 1e-15:
            return 0.0, 0.5
        # Deterministic improvement/degradation
        return (np.inf if d_mean > 0 else -np.inf), (0.0 if d_mean > 0 else 1.0)
    t_stat = d_mean / (d_std / np.sqrt(n))
    p_value = 1.0 - t_dist.cdf(t_stat, df=n - 1)
    return float(t_stat), float(p_value)


def recommend_model(
    candidates: list[ModelCandidate],
    prev_choice_idx: int | None,
    t_threshold_same_or_simpler: float = 2.0,
    t_threshold_more_complex: float = 3.0,
) -> ModelRecommendation:
    """Recommend KEEP, SWITCH, or INITIAL for model selection.

    Decision gates (in order):
      1. INITIAL — no previous choice (round 1).
      2. Hard switch — status quo has red RAG flag or is missing.
      3. Significance test — paired LOO-LPD t-test with complexity filter.

    Args:
        candidates: All candidates, sorted by (demoted, -loo_lpd).
        prev_choice_idx: Index of the previous model in the candidate list,
            or None if this is the first round.
        t_threshold_same_or_simpler: t-stat threshold for switching to a
            model of equal or lower complexity.
        t_threshold_more_complex: t-stat threshold for switching to a
            model of higher complexity.

    Returns:
        ModelRecommendation with action, recommended_idx, and rationale.
    """
    non_demoted = [
        (i, c) for i, c in enumerate(candidates)
        if not getattr(c, "demoted", False)
    ]

    # Gate 1: INITIAL
    if prev_choice_idx is None:
        rec_idx = _parsimony_pick(candidates, non_demoted)
        return ModelRecommendation(
            action=ModelAction.INITIAL,
            recommended_idx=rec_idx,
            rationale=(
                "First round. Picked the simplest model within 0.5 nats "
                "of the best LOO-LPD."
            ),
        )

    status_quo = candidates[prev_choice_idx]

    # Gate 2: Hard switch — status quo is demoted
    if getattr(status_quo, "demoted", False):
        rec_idx = _parsimony_pick(candidates, non_demoted)
        mc = candidates[rec_idx]
        label = _kernel_label(mc)
        sq_label = _kernel_label(status_quo)
        return ModelRecommendation(
            action=ModelAction.SWITCH,
            recommended_idx=rec_idx,
            rationale=(
                f"Previous choice {status_quo.output_transform} ({sq_label}) "
                f"has critical diagnostics (red RAG). "
                f"Switching to {mc.output_transform} ({label})."
            ),
        )

    # Gate 3: Significance test
    alternatives = [(i, c) for i, c in non_demoted if i != prev_choice_idx]
    if not alternatives:
        return ModelRecommendation(
            action=ModelAction.KEEP,
            recommended_idx=prev_choice_idx,
            rationale="Only non-demoted model available. Keeping current choice.",
        )

    best_alt_idx, best_alt = max(alternatives, key=lambda pair: pair[1].loo_lpd)

    sq_lpd_i = getattr(status_quo, "loo_lpd_i", None)
    alt_lpd_i = getattr(best_alt, "loo_lpd_i", None)

    if sq_lpd_i is None or alt_lpd_i is None:
        logger.warning(
            "Per-point LPDs not available; falling back to scalar LPD ranking."
        )
        rec_idx = _parsimony_pick(candidates, non_demoted)
        return ModelRecommendation(
            action=(
                ModelAction.KEEP if rec_idx == prev_choice_idx
                else ModelAction.SWITCH
            ),
            recommended_idx=rec_idx,
            rationale="Fallback to scalar LPD ranking (per-point LPDs unavailable).",
        )

    t_stat, p_value = paired_lpd_test(sq_lpd_i, alt_lpd_i)

    sq_complexity = _complexity_rank(status_quo)
    alt_complexity = _complexity_rank(best_alt)

    if alt_complexity <= sq_complexity:
        threshold = t_threshold_same_or_simpler
        complexity_note = "simpler or equal complexity"
    else:
        threshold = t_threshold_more_complex
        complexity_note = "more complex — higher threshold required"

    sq_label = _kernel_label(status_quo)
    alt_label = _kernel_label(best_alt)

    if t_stat >= threshold:
        return ModelRecommendation(
            action=ModelAction.SWITCH,
            recommended_idx=best_alt_idx,
            rationale=(
                f"{best_alt.output_transform} ({alt_label}) is significantly "
                f"better than the previous choice "
                f"{status_quo.output_transform} ({sq_label}) "
                f"(t={t_stat:.2f}, p={p_value:.4f}, {complexity_note})."
            ),
            t_stat=t_stat,
            p_value=p_value,
        )
    else:
        return ModelRecommendation(
            action=ModelAction.KEEP,
            recommended_idx=prev_choice_idx,
            rationale=(
                f"Best alternative {best_alt.output_transform} ({alt_label}) "
                f"not significantly better than the previous choice "
                f"(t={t_stat:.2f}, p={p_value:.4f}, {complexity_note}). "
                f"Keeping {status_quo.output_transform} ({sq_label})."
            ),
            t_stat=t_stat,
            p_value=p_value,
        )


def _parsimony_pick(
    candidates: list[ModelCandidate],
    non_demoted: list[tuple[int, ModelCandidate]],
) -> int:
    """Among non-demoted candidates, pick simplest within 0.5 nats of best LPD."""
    if not non_demoted:
        return 0
    best_idx, best = non_demoted[0]
    simplest_idx = best_idx
    simplest_complexity = _complexity_rank(best)
    for idx, mc in non_demoted:
        if best.loo_lpd - mc.loo_lpd <= 0.5:
            c = _complexity_rank(mc)
            if c < simplest_complexity:
                simplest_idx = idx
                simplest_complexity = c
    return simplest_idx
