"""Orchestrate candidate generation from all strategies."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from botorch.optim import optimize_acqf

from bayesopt_human.acquisition.functions import (
    compute_ei,
    compute_max_mean,
    compute_mpv,
    compute_ucb_lcb,
)
from bayesopt_human.acquisition.space_filling import (
    max_min_distance_flat,
    max_min_distance_surrogate,
)
from bayesopt_human.config import OptimizationConfig
from bayesopt_human.gp.model import FittedGP
from bayesopt_human.gp.surrogate_space import SurrogateSpaceTransform

logger = logging.getLogger("bayesopt_human")


@dataclass
class Candidate:
    """A single candidate point with metadata."""

    coordinates_normalized: np.ndarray
    coordinates_raw: np.ndarray
    coordinates_surrogate: np.ndarray | None
    source: str
    category: str
    posterior_mean: float
    posterior_std: float
    expected_improvement: float
    coverage_gain_flat: float
    coverage_gain_surrogate: float
    svm_prediction: bool | None


class CandidateGenerator:
    """Orchestrates candidate generation from all strategies."""

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config

    def generate_all(
        self,
        fitted_gp: FittedGP,
        X: np.ndarray,
        y: np.ndarray,
        normalizer,
    ) -> list[Candidate]:
        """Generate one candidate per strategy.

        Args:
            fitted_gp: The fitted GP model.
            X: Normalized inputs, shape (N, D).
            y: Raw objective values, shape (N,).
            normalizer: BoundsNormalizer for denormalization.

        Returns:
            List of 8 Candidate objects (some may be fewer if strategies fail).
        """
        n_dim = X.shape[1]
        direction = self.config.direction

        # Surrogate space transform
        sur_transform = SurrogateSpaceTransform(fitted_gp)

        # Compute best value in IQR-normalized space for EI
        y_transformed = fitted_gp.output_transform.forward(y)
        y_norm = fitted_gp.iqr_normalizer.transform(y_transformed)
        if direction == "minimize":
            best_norm = float(np.min(y_norm))
        else:
            best_norm = float(np.max(y_norm))

        # BoTorch bounds
        bounds_t = torch.tensor(
            [[0.0] * n_dim, [1.0 - 1e-10] * n_dim], dtype=torch.float64
        )

        candidates: list[Candidate] = []

        # Strategy 1: Max-min distance (flat)
        cand = self._try_strategy(
            "Max-min distance (flat)", "space-filling",
            lambda: max_min_distance_flat(X, n_dim),
            fitted_gp, X, y, normalizer, sur_transform,
        )
        if cand:
            candidates.append(cand)

        # Strategy 2: Max-min distance (surrogate)
        cand = self._try_strategy(
            "Max-min distance (surrogate)", "space-filling",
            lambda: max_min_distance_surrogate(X, sur_transform, n_dim),
            fitted_gp, X, y, normalizer, sur_transform,
        )
        if cand:
            candidates.append(cand)

        # Strategy 3: Max Posterior Variance
        cand = self._try_strategy(
            "Max Posterior Variance", "exploration",
            lambda: self._optimize_custom_acqf(
                compute_mpv(fitted_gp), bounds_t, n_dim
            ),
            fitted_gp, X, y, normalizer, sur_transform,
        )
        if cand:
            candidates.append(cand)

        # Strategy 4: Expected Improvement (robust optimizer with extra restarts/samples)
        ei_acqf = compute_ei(fitted_gp, best_norm, direction)
        cand = self._try_strategy(
            "Expected Improvement", "balanced",
            lambda: self._optimize_acqf(
                ei_acqf, bounds_t, n_dim,
                num_restarts=max(20, 4 * n_dim),
                raw_samples=max(2048, 256 * n_dim),
            ),
            fitted_gp, X, y, normalizer, sur_transform,
        )
        if cand:
            candidates.append(cand)

        # Strategies 5-7: UCB/LCB with different kappas. Each candidate is
        # generated at its declared base kappa; no horizon annealing — the
        # recommendation engine handles budget-pressure ranking downstream.
        kappa_labels = [
            (self.config.kappa_values[0], "low", "exploitation"),
            (self.config.kappa_values[1], "medium", "balanced"),
            (self.config.kappa_values[2], "high", "exploration"),
        ]
        ucb_name = "UCB" if direction == "maximize" else "LCB"
        for kappa, label, category in kappa_labels:
            acqf = compute_ucb_lcb(fitted_gp, kappa, direction)
            cand = self._try_strategy(
                f"{ucb_name} (kappa={label})", category,
                lambda acqf=acqf: self._optimize_acqf(acqf, bounds_t, n_dim),
                fitted_gp, X, y, normalizer, sur_transform,
            )
            if cand:
                candidates.append(cand)

        # Strategy 8: Max GP Mean (robust optimizer with extra restarts/samples)
        mean_acqf = compute_max_mean(fitted_gp, direction)
        cand = self._try_strategy(
            "Max GP Mean", "exploitation",
            lambda: self._optimize_acqf(
                mean_acqf, bounds_t, n_dim,
                num_restarts=max(20, 4 * n_dim),
                raw_samples=max(2048, 256 * n_dim),
            ),
            fitted_gp, X, y, normalizer, sur_transform,
        )
        if cand:
            candidates.append(cand)

        # Post-hoc correction for metric-defined strategies: if any competitor
        # scores strictly better on the strategy's own metric, rebuild that
        # strategy's candidate at the competitor's coordinates. This guarantees
        # the label is always accurate even if the optimizer gets stuck.
        self._correct_strategy(
            candidates, "Expected Improvement", "expected_improvement",
            higher_is_better=True,  # log-EI: higher is always better regardless of direction
            fitted_gp=fitted_gp, X=X, y=y,
            normalizer=normalizer, sur_transform=sur_transform,
        )
        self._correct_strategy(
            candidates, "Max GP Mean", "posterior_mean",
            higher_is_better=(direction == "maximize"),
            fitted_gp=fitted_gp, X=X, y=y,
            normalizer=normalizer, sur_transform=sur_transform,
        )

        return candidates

    def _correct_strategy(
        self,
        candidates: list[Candidate],
        source_name: str,
        metric: str,
        higher_is_better: bool,
        fitted_gp: FittedGP,
        X: np.ndarray,
        y: np.ndarray,
        normalizer,
        sur_transform: SurrogateSpaceTransform,
    ) -> None:
        """Rebuild *source_name* candidate at the best competitor's coordinates
        if any candidate scores strictly better on *metric*."""
        target_idx = next(
            (i for i, c in enumerate(candidates) if c.source == source_name), None
        )
        if target_idx is None:
            return
        target = candidates[target_idx]
        target_val = getattr(target, metric)

        best_idx = None
        best_val = target_val
        for i, other in enumerate(candidates):
            if i == target_idx:
                continue
            val = getattr(other, metric)
            if higher_is_better:
                if val > best_val:
                    best_val = val
                    best_idx = i
            else:
                if val < best_val:
                    best_val = val
                    best_idx = i

        if best_idx is None:
            return

        winner = candidates[best_idx]
        logger.info(
            "%s optimizer missed global mode; reusing %s coordinates "
            "(%s %.4g vs %.4g).",
            source_name, winner.source, metric, best_val, target_val,
        )
        candidates[target_idx] = self._build_candidate(
            winner.coordinates_normalized,
            source_name, target.category,
            fitted_gp, X, y, normalizer, sur_transform,
        )

    def _try_strategy(
        self,
        name: str,
        category: str,
        optimize_fn,
        fitted_gp: FittedGP,
        X: np.ndarray,
        y: np.ndarray,
        normalizer,
        sur_transform: SurrogateSpaceTransform,
    ) -> Candidate | None:
        """Try a strategy and return a Candidate or None on failure."""
        try:
            x_norm = optimize_fn()
            x_norm = np.clip(x_norm, 0.0, 1.0 - 1e-10)
            return self._build_candidate(
                x_norm, name, category, fitted_gp, X, y,
                normalizer, sur_transform,
            )
        except Exception as e:
            logger.warning("Strategy %r failed: %s", name, e)
            return None

    def _build_candidate(
        self,
        x_norm: np.ndarray,
        source: str,
        category: str,
        fitted_gp: FittedGP,
        X: np.ndarray,
        y: np.ndarray,
        normalizer,
        sur_transform: SurrogateSpaceTransform,
    ) -> Candidate:
        """Build a Candidate with all supporting metrics."""
        x_norm = x_norm.flatten()
        x_raw = normalizer.denormalize(x_norm)
        x_sur = sur_transform.to_surrogate(x_norm)

        # Posterior predictions in surrogate space (IQR-normalized, output-transformed)
        mean_sur, std_sur = fitted_gp.predict(x_norm.reshape(1, -1))
        post_mean = float(mean_sur[0])
        post_std = float(std_sur[0])

        # EI at this candidate (regardless of source strategy)
        y_transformed = fitted_gp.output_transform.forward(y)
        y_normed = fitted_gp.iqr_normalizer.transform(y_transformed)
        if self.config.direction == "minimize":
            best_norm = float(np.min(y_normed))
        else:
            best_norm = float(np.max(y_normed))

        ei_acqf = compute_ei(fitted_gp, best_norm, self.config.direction)
        x_t = torch.tensor(x_norm, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            ei_val = float(ei_acqf(x_t))

        return Candidate(
            coordinates_normalized=x_norm,
            coordinates_raw=x_raw,
            coordinates_surrogate=x_sur,
            source=source,
            category=category,
            posterior_mean=post_mean,
            posterior_std=post_std,
            expected_improvement=ei_val,
            coverage_gain_flat=0.0,  # Filled in by diagnostics
            coverage_gain_surrogate=0.0,  # Filled in by diagnostics
            svm_prediction=None,  # Filled in by diagnostics
        )

    @staticmethod
    def _optimize_acqf(
        acqf, bounds: torch.Tensor, n_dim: int,
        num_restarts: int | None = None,
        raw_samples: int | None = None,
    ) -> np.ndarray:
        """Optimize a BoTorch acquisition function."""
        if num_restarts is None:
            num_restarts = min(10, max(5, 2 * n_dim))
        if raw_samples is None:
            raw_samples = max(256, 64 * n_dim)
        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        return candidate.detach().cpu().numpy().flatten()

    @staticmethod
    def _optimize_custom_acqf(
        acqf, bounds: torch.Tensor, n_dim: int
    ) -> np.ndarray:
        """Optimize a custom acquisition function using random + local search."""
        n_samples = max(1000, 200 * n_dim)
        X_cand = torch.rand(n_samples, n_dim, dtype=torch.float64)
        X_cand = X_cand * (bounds[1] - bounds[0]) + bounds[0]

        with torch.no_grad():
            values = acqf(X_cand.unsqueeze(1))

        best_idx = values.argmax()
        return X_cand[best_idx].numpy()
