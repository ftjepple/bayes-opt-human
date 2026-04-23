"""Arm-based recommendation engine.

Allocates weight across four strategy arms — exploit, balanced, model-explore,
geometric-explore — from four signals (urgency, GP calibration, stagnation,
coverage need). Each candidate is scored as ``w_arm * within-arm percentile``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from bayesopt_human.acquisition.candidates import Candidate
from bayesopt_human.config import OptimizationConfig
from bayesopt_human.diagnostics.coverage import CoverageReport
from bayesopt_human.diagnostics.gp_diagnostics import GPFitReport
from bayesopt_human.diagnostics.svm import SVMReport
from bayesopt_human.recommendations.schema import Recommendation
from bayesopt_human.utils import stagnation_length

logger = logging.getLogger("bayesopt_human")


# Arm keys match the category strings assigned during candidate generation.
ARM_KEYS: tuple[str, ...] = (
    "exploitation",
    "balanced",
    "exploration",
    "space-filling",
)

# User-facing arm labels.
ARM_LABEL: dict[str, str] = {
    "exploitation": "exploit",
    "balanced": "balanced",
    "exploration": "model-explore",
    "space-filling": "geometric-explore",
}

# The natural per-candidate metric used to rank within each arm.
ARM_METRIC_KEY: dict[str, str] = {
    "exploitation": "p_exploit",
    "balanced": "p_ei",
    "exploration": "p_explore",
    "space-filling": "p_coverage",
}


@dataclass
class Signals:
    """Four signals that drive the arm allocation."""

    urgency: float           # u — 0 at start, 1 at final evaluation
    gp_quality: float        # q — 1 when calibration_var=1, decays away
    stagnation: float        # s — deadband-adjusted stagnation fraction
    stagnation_raw: float    # raw stagnation fraction (for reporting)
    coverage_need: float     # c — 0 when coverage at optimum, 1 when far
    exploit_pull: float      # = u
    explore_pull: float      # = clip(1, (1-u) + s + c)


class RecommendationEngine:
    """Arm-based recommendation engine.

    Six-phase pipeline:
      1. NORMALIZE — rank-percentile all candidate metrics
      2. ASSESS    — derive signals (u, q, s, c) and pulls
      3. ALLOCATE  — weights across the four strategy arms
      4. SCORE     — score_i = w_arm(i) · within-arm percentile
      5. DIVERSIFY — demote near-duplicate candidates
      6. ANNOTATE  — per-candidate confidence + rationale
    """

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config

    def generate(
        self,
        candidates: list[Candidate],
        gp_report: GPFitReport,
        coverage: CoverageReport,
        svm_report: SVMReport,
        y: np.ndarray,
    ) -> list[Recommendation]:
        """Generate ranked recommendations."""
        if not candidates:
            return []

        n = len(candidates)

        # Phase 1: NORMALIZE
        percentiles = self._normalize(candidates, n)

        # Phase 2: ASSESS
        signals = self._compute_signals(gp_report, y, coverage)

        # Phase 3: ALLOCATE
        arm_weights = self._compute_arm_weights(signals)

        # Phase 4: SCORE
        scores, within_arm_pct, within_arm_rank, arm_size, arm_winners = (
            self._score(candidates, percentiles, arm_weights)
        )

        # Phase 5: DIVERSIFY
        order = self._diversify(candidates, scores, coverage)

        # Phase 6: ANNOTATE
        recommendations = self._annotate(
            candidates, percentiles, order,
            signals, arm_weights,
            within_arm_pct, within_arm_rank, arm_size, arm_winners,
            gp_report, svm_report,
        )

        return recommendations

    # ------------------------------------------------------------------
    # Phase 1: NORMALIZE
    # ------------------------------------------------------------------

    def _normalize(
        self, candidates: list[Candidate], n: int,
    ) -> dict[str, np.ndarray]:
        """Rank-percentile all candidate metrics across the full slate."""
        means = np.array([c.posterior_mean for c in candidates])
        stds = np.array([c.posterior_std for c in candidates])
        eis = np.array([c.expected_improvement for c in candidates])
        cov_flat = np.array([c.coverage_gain_flat for c in candidates])
        cov_sur = np.array([c.coverage_gain_surrogate for c in candidates])
        coverages = (cov_flat + cov_sur) / 2.0

        def _to_percentile(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
            if n == 1:
                return np.array([0.5])
            if higher_is_better:
                ranks = rankdata(values, method="average")
            else:
                ranks = rankdata(-values, method="average")
            return (ranks - 1) / (n - 1)

        if self.config.direction == "maximize":
            p_exploit = _to_percentile(means, higher_is_better=True)
        else:
            p_exploit = _to_percentile(means, higher_is_better=False)

        return {
            "p_exploit": p_exploit,
            "p_explore": _to_percentile(stds, higher_is_better=True),
            "p_ei": _to_percentile(eis, higher_is_better=True),
            "p_coverage": _to_percentile(coverages, higher_is_better=True),
        }

    # ------------------------------------------------------------------
    # Phase 2: ASSESS
    # ------------------------------------------------------------------

    def _compute_signals(
        self,
        gp_report: GPFitReport,
        y: np.ndarray,
        coverage: CoverageReport,
    ) -> Signals:
        """Derive the four driving signals plus the exploit/explore pulls."""
        remaining = self.config.remaining
        budget = self.config.optimization_budget
        warmstart = self.config.warmstart

        # Urgency: 0 at start, 1 at final evaluation
        u = 1.0 - (remaining - 1) / max(budget - 1, 1)
        u = _clip01(u)

        # GP quality: peaks at calibration_var=1, symmetric in log space.
        cal_var = max(gp_report.calibration_var, 1e-12)
        q = math.exp(-abs(math.log(cal_var)))
        q = _clip01(q)

        # Stagnation: deadband-adjusted — the first 30% of the run's
        # stagnation fraction is ignored so small-sample noise doesn't
        # trigger the signal prematurely.
        n_opt_obs = self.config.n_obs - warmstart
        stag = stagnation_length(y, self.config.direction, warmstart)
        s_raw = stag / max(n_opt_obs, 1)
        s = _clip01(max(0.0, (s_raw - 0.3) / 0.7))

        # Coverage need: 0 when coverage matches an even sample at the
        # current sample size, rising toward 1 when observed points leave
        # large gaps in the search space.
        coverage_ratio = coverage.ratio_to_optimal_flat
        c = _clip01(1.0 - 1.0 / max(coverage_ratio, 1.0))

        exploit_pull = u
        explore_pull = _clip01((1.0 - u) + s + c)

        return Signals(
            urgency=u, gp_quality=q,
            stagnation=s, stagnation_raw=s_raw,
            coverage_need=c,
            exploit_pull=exploit_pull, explore_pull=explore_pull,
        )

    # ------------------------------------------------------------------
    # Phase 3: ALLOCATE
    # ------------------------------------------------------------------

    def _compute_arm_weights(self, sig: Signals) -> dict[str, float]:
        """Allocate weight across the four strategy arms.

        - exploit       = exploit_pull · max(q, 0.2)
        - balanced      = 0.5 · (exploit_pull + explore_pull) · sqrt(q)
        - model-explore = explore_pull · q
        - geometric     = explore_pull · (1 - q) + 0.05

        Weights are then normalized to sum to 1.
        """
        q = sig.gp_quality
        q_floor = max(q, 0.2)

        raw = {
            "exploitation": sig.exploit_pull * q_floor,
            "balanced": 0.5 * (sig.exploit_pull + sig.explore_pull) * math.sqrt(q),
            "exploration": sig.explore_pull * q,
            "space-filling": sig.explore_pull * (1.0 - q) + 0.05,
        }

        total = sum(raw.values())
        if total <= 0:
            # Degenerate (zero pulls everywhere). Fall back to uniform.
            return {k: 0.25 for k in ARM_KEYS}
        return {k: v / total for k, v in raw.items()}

    # ------------------------------------------------------------------
    # Phase 4: SCORE
    # ------------------------------------------------------------------

    def _score(
        self,
        candidates: list[Candidate],
        percentiles: dict[str, np.ndarray],
        arm_weights: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int], set[int]]:
        """Score candidates as ``w_arm · within-arm percentile``.

        Returns:
            scores: score per candidate
            within_arm_pct: [0, 1] percentile within arm (best=1)
            within_arm_rank: 1-based rank within arm (best=1)
            arm_size: count of candidates per arm
            arm_winners: set of indices that are top in their arm
        """
        n = len(candidates)
        within_pct = np.zeros(n)
        within_rank = np.ones(n, dtype=int)
        arm_winners: set[int] = set()

        members: dict[str, list[int]] = {k: [] for k in ARM_KEYS}
        for i, cand in enumerate(candidates):
            if cand.category not in members:
                raise ValueError(
                    f"Candidate '{cand.source}' has unknown category "
                    f"'{cand.category}' (expected one of {ARM_KEYS})."
                )
            members[cand.category].append(i)

        arm_size = {k: len(v) for k, v in members.items()}

        for arm, idxs in members.items():
            if not idxs:
                continue
            metric = percentiles[ARM_METRIC_KEY[arm]]
            # Higher metric is better in all four cases (direction-aware
            # handling is done inside _normalize for p_exploit).
            arm_vals = np.array([metric[i] for i in idxs])
            order = np.argsort(-arm_vals, kind="stable")
            k = len(idxs)
            for rank_pos, local_idx in enumerate(order):
                g = idxs[local_idx]
                within_rank[g] = rank_pos + 1
                within_pct[g] = 1.0 if k == 1 else 1.0 - rank_pos / (k - 1)
            arm_winners.add(idxs[order[0]])

        scores = np.array([
            arm_weights[c.category] * within_pct[i]
            for i, c in enumerate(candidates)
        ])
        return scores, within_pct, within_rank, arm_size, arm_winners

    # ------------------------------------------------------------------
    # Phase 5: DIVERSIFY
    # ------------------------------------------------------------------

    def _diversify(
        self,
        candidates: list[Candidate],
        scores: np.ndarray,
        coverage: CoverageReport,
    ) -> list[int]:
        """Sort by score; demote near-duplicate candidates."""
        order = list(np.argsort(-scores))  # descending
        threshold = coverage.avg_knn_distance_flat / 2.0

        accepted_coords: list[np.ndarray] = []
        final_order: list[int] = []
        deferred: list[int] = []

        for idx in order:
            coord = candidates[idx].coordinates_normalized
            if not accepted_coords:
                accepted_coords.append(coord)
                final_order.append(idx)
                continue

            too_close = any(
                np.linalg.norm(coord - ac) < threshold
                for ac in accepted_coords
            )
            if too_close:
                deferred.append(idx)
            else:
                accepted_coords.append(coord)
                final_order.append(idx)

        final_order.extend(deferred)
        return final_order

    # ------------------------------------------------------------------
    # Phase 6: ANNOTATE
    # ------------------------------------------------------------------

    def _annotate(
        self,
        candidates: list[Candidate],
        percentiles: dict[str, np.ndarray],
        order: list[int],
        signals: Signals,
        arm_weights: dict[str, float],
        within_arm_pct: np.ndarray,
        within_arm_rank: np.ndarray,
        arm_size: dict[str, int],
        arm_winners: set[int],
        gp_report: GPFitReport,
        svm_report: SVMReport,
    ) -> list[Recommendation]:
        """Build Recommendation objects with rationale and confidence."""
        n = len(candidates)

        # Shared header text reused on every recommendation.
        allocation_line = _format_allocation(arm_weights)
        why_line = _format_allocation_reasons(signals, arm_weights)

        # Per-candidate confidence scores (unchanged in intent, now using q).
        q = signals.gp_quality
        conf_scores = np.empty(n)
        for i in range(n):
            p_explore_pct = percentiles["p_explore"][i]
            svm_pred = candidates[i].svm_prediction
            if not svm_report.is_reliable:
                svm_factor = 1.0
            elif svm_pred is True:
                svm_factor = 1.0
            else:
                svm_factor = 0.5
            conf_scores[i] = q * (1 - p_explore_pct) * svm_factor

        if n >= 3:
            sorted_conf = np.sort(conf_scores)
            low_thresh = sorted_conf[n // 3]
            high_thresh = sorted_conf[2 * n // 3]
        else:
            low_thresh = 0.33
            high_thresh = 0.67

        recommendations: list[Recommendation] = []
        for priority, idx in enumerate(order, 1):
            cand = candidates[idx]

            # Confidence label
            cs = conf_scores[idx]
            if cs >= high_thresh:
                confidence = "high"
            elif cs >= low_thresh:
                confidence = "medium"
            else:
                confidence = "low"

            # Per-candidate rationale
            arm = cand.category
            arm_label = ARM_LABEL[arm]
            rank_in_arm = int(within_arm_rank[idx])
            size_of_arm = arm_size[arm]
            is_winner = idx in arm_winners
            winner_tag = " ★" if is_winner else ""

            if is_winner:
                candidate_line = (
                    f"This candidate is the top pick within the "
                    f"{arm_label} arm (rank 1/{size_of_arm}){winner_tag}."
                )
            else:
                candidate_line = (
                    f"This candidate sits at rank {rank_in_arm}/{size_of_arm} "
                    f"within the {arm_label} arm."
                )

            parts = [allocation_line, why_line, candidate_line]

            # SVM note
            if svm_report.is_reliable:
                if cand.svm_prediction is True:
                    parts.append("SVM agrees: likely top quartile.")
                elif cand.svm_prediction is False:
                    parts.append("SVM disagrees: may not be top quartile.")

            rationale = " ".join(parts)

            recommendations.append(
                Recommendation(
                    candidate=cand.coordinates_raw,
                    candidate_normalized=cand.coordinates_normalized,
                    source=cand.source,
                    rationale=rationale,
                    confidence=confidence,
                    priority=priority,
                    arm=arm,
                    is_arm_winner=is_winner,
                )
            )

        return recommendations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def _format_allocation(arm_weights: dict[str, float]) -> str:
    parts = [
        f"{ARM_LABEL[k]} {arm_weights[k] * 100:.0f}%"
        for k in ARM_KEYS
    ]
    return "Strategy allocation: " + " · ".join(parts) + "."


def _format_allocation_reasons(
    signals: Signals, arm_weights: dict[str, float],
) -> str:
    """Explain *why* the allocation came out the way it did.

    We name the active signals (those pulling weight away from a neutral
    allocation) in plain language. Silent when all signals are neutral.
    """
    notes: list[str] = []

    # GP calibration is the strongest discriminator — always call it out.
    if signals.gp_quality < 0.5:
        notes.append(
            "the GP is poorly calibrated, so model-dependent strategies are "
            "downweighted in favour of geometric exploration"
        )
    elif signals.gp_quality >= 0.9:
        notes.append(
            "the GP is well calibrated, so model-based strategies are trusted"
        )

    if signals.urgency >= 0.7:
        notes.append(
            "the evaluation budget is nearly exhausted, pushing weight "
            "toward exploitation"
        )
    elif signals.urgency <= 0.3:
        notes.append(
            "the evaluation budget still has headroom, leaving room for "
            "exploration"
        )

    if signals.stagnation > 0.0:
        notes.append(
            "recent progress has shown stagnation, boosting exploration"
        )

    if signals.coverage_need >= 0.5:
        notes.append(
            "observed points leave large gaps in the search space, boosting "
            "exploration"
        )

    if not notes:
        return "Why: all signals are neutral."

    top_arm = max(arm_weights, key=arm_weights.get)
    return (
        f"Why: {'; '.join(notes)} — the {ARM_LABEL[top_arm]} arm carries "
        f"the most weight."
    )
