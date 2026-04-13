"""Data-driven recommendation engine (5-phase scoring)."""

from __future__ import annotations

import logging
import math

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


class RecommendationEngine:
    """Data-driven recommendation engine.

    Five-phase pipeline:
      1. NORMALIZE — rank-percentile all candidate metrics
      2. ASSESS   — derive urgency, GP quality, stagnation pressure
      3. SCORE    — blend exploitation vs exploration via α
      4. DIVERSIFY — demote near-duplicate candidates
      5. ANNOTATE  — per-candidate confidence + rationale
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
        """Generate ranked recommendations.

        Args:
            candidates: Candidate points from all strategies.
            gp_report: GP fit diagnostics (includes calibration_var).
            coverage: Coverage metrics (KNN distances).
            svm_report: SVM sense-check report.
            y: Raw objective values in observation order.

        Returns:
            Ranked list of Recommendation objects.
        """
        if not candidates:
            return []

        n = len(candidates)

        # Phase 1: NORMALIZE
        percentiles = self._normalize(candidates, n)

        # Phase 2: ASSESS
        urgency, gp_quality, stag_pressure = self._assess(gp_report, y)

        # Phase 3: SCORE
        alpha = max(1 - urgency, 1 - gp_quality, stag_pressure)
        alpha = min(max(alpha, 0.0), 1.0)

        coverage_ratio = coverage.ratio_to_optimal_flat
        coverage_need = 1.0 - 1.0 / max(coverage_ratio, 1.0)

        scores = np.empty(n)
        for i in range(n):
            p_exploit = percentiles["p_exploit"][i]
            p_explore = percentiles["p_explore"][i]
            p_ei = percentiles["p_ei"][i]
            p_coverage = percentiles["p_coverage"][i]

            p_explore_composite = (
                (1 - coverage_need) * p_explore + coverage_need * p_coverage
            )
            scores[i] = (1 - alpha) * p_exploit + alpha * p_explore_composite + p_ei

        # Phase 4: DIVERSIFY
        order = self._diversify(candidates, scores, coverage)

        # Phase 5: ANNOTATE
        recommendations = self._annotate(
            candidates, percentiles, scores, order,
            alpha, urgency, gp_quality, stag_pressure,
            gp_report, svm_report,
        )

        return recommendations

    # ------------------------------------------------------------------
    # Phase 1: NORMALIZE
    # ------------------------------------------------------------------

    def _normalize(
        self, candidates: list[Candidate], n: int,
    ) -> dict[str, np.ndarray]:
        """Rank-percentile all candidate metrics."""
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

        # Direction-aware exploitation percentile
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

    def _assess(
        self, gp_report: GPFitReport, y: np.ndarray,
    ) -> tuple[float, float, float]:
        """Derive urgency, GP quality, and stagnation pressure."""
        remaining = self.config.remaining
        budget = self.config.optimization_budget
        warmstart = self.config.warmstart

        # Urgency: 0 at start, 1 at final evaluation
        urgency = 1.0 - (remaining - 1) / max(budget - 1, 1)
        urgency = min(max(urgency, 0.0), 1.0)

        # GP quality: peaks at calibration_var=1.0, decays away from it
        cal_var = max(gp_report.calibration_var, 1e-12)
        gp_quality = math.exp(-abs(math.log(cal_var)))
        gp_quality = min(max(gp_quality, 0.0), 1.0)

        # Stagnation pressure
        n_opt_obs = self.config.n_obs - warmstart
        stag = stagnation_length(y, self.config.direction, warmstart)
        stag_pressure = stag / max(n_opt_obs, 1)
        stag_pressure = min(max(stag_pressure, 0.0), 1.0)

        return urgency, gp_quality, stag_pressure

    # ------------------------------------------------------------------
    # Phase 4: DIVERSIFY
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
    # Phase 5: ANNOTATE
    # ------------------------------------------------------------------

    def _annotate(
        self,
        candidates: list[Candidate],
        percentiles: dict[str, np.ndarray],
        scores: np.ndarray,
        order: list[int],
        alpha: float,
        urgency: float,
        gp_quality: float,
        stag_pressure: float,
        gp_report: GPFitReport,
        svm_report: SVMReport,
    ) -> list[Recommendation]:
        """Build Recommendation objects with rationale and confidence."""
        # Determine dominant signal for α
        signals = {
            "budget pressure": 1 - urgency,
            "GP quality": 1 - gp_quality,
            "stagnation": stag_pressure,
        }
        dominant = max(signals, key=signals.get)

        # Per-candidate confidence scores
        n = len(candidates)
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
            conf_scores[i] = gp_quality * (1 - p_explore_pct) * svm_factor

        # Tercile thresholds
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

            # Rationale
            parts: list[str] = []

            # State description
            parts.append(
                f"α={alpha:.2f} (driven by {dominant})."
            )

            # Per-candidate rank info
            ei_rank = int(rankdata(-percentiles["p_ei"])[idx])
            exploit_rank = int(rankdata(-percentiles["p_exploit"])[idx])
            explore_rank = int(rankdata(-percentiles["p_explore"])[idx])
            parts.append(
                f"EI rank {ei_rank}/{n}, "
                f"exploit rank {exploit_rank}/{n}, "
                f"explore rank {explore_rank}/{n}."
            )

            # SVM info
            if svm_report.is_reliable:
                if cand.svm_prediction is True:
                    parts.append("SVM agrees: likely top quartile.")
                elif cand.svm_prediction is False:
                    parts.append("SVM disagrees: may not be top quartile.")

            # Stagnation note
            if stag_pressure > 0.3:
                parts.append("Stagnation detected: exploration weighted higher.")

            # Final evaluation note
            if self.config.remaining <= 1:
                parts.append("Final evaluation: exploitation strongly favored.")

            rationale = " ".join(parts)

            recommendations.append(
                Recommendation(
                    candidate=cand.coordinates_raw,
                    candidate_normalized=cand.coordinates_normalized,
                    source=cand.source,
                    rationale=rationale,
                    confidence=confidence,
                    priority=priority,
                )
            )

        return recommendations
