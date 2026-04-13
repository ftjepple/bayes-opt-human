"""Central configuration for BayesOpt-Human."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OptimizationConfig:
    """Central configuration for the optimization."""

    # Required
    name: str  # Name of the function/problem, must match CSV filename (without .csv)
    horizon: int
    bounds: dict[str, tuple[float, float]] | tuple[float, float]
    objective_column: str

    # Prior beliefs
    direction: str = "minimize"
    modality: str = "unknown"

    # Warmstart: number of observations that existed before optimization began
    warmstart: int = 0

    # Tunable parameters
    kappa_values: tuple[float, ...] = (0.5, 1.5, 2.5)
    knn_k: int = 5

    # Reproducibility. When set, the torch and numpy global RNGs are
    # reseeded at the start of every ``get_models`` and ``get_candidates``
    # call, so running the same pipeline on the same data in a fresh
    # Python process produces bit-identical recommendations. Set to
    # ``None`` to leave the global RNG state untouched (useful if the
    # caller wants to stochastically re-run the acquisition optimizer).
    random_seed: int | None = 42
    modality_k: int = 5
    warping_min_ratio: int = 5
    jitter_base: float = 1e-6
    jitter_ceiling_factor: float = 1e-2
    clipped_log_epsilon: float = 1e-120
    pairwise_top_k: int = 4

    # Length-scale bounds for diagnostics and reporting
    length_scale_lower_bound: float = 1e-3
    length_scale_upper_bound: float = 1e3

    # Regularization priors (LogNormal) — applied during GP fitting
    length_scale_prior_median: float = 0.5
    length_scale_prior_scale: float = 1.0
    warp_prior_median: float = 1.0
    warp_prior_scale: float = 0.75

    # Derived (set after data loading)
    n_obs: int = 0
    n_dim: int = 0
    remaining: int = 0
    optimization_budget: int = 0
    param_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if self.warmstart < 0:
            raise ValueError(f"warmstart must be >= 0, got {self.warmstart}")
        if self.warmstart >= self.horizon:
            raise ValueError(
                f"warmstart ({self.warmstart}) must be < horizon ({self.horizon})"
            )
        if self.direction not in ("minimize", "maximize"):
            raise ValueError(
                f"direction must be 'minimize' or 'maximize', got {self.direction!r}"
            )
        if self.modality not in ("unimodal", "multimodal", "unknown"):
            raise ValueError(
                f"modality must be 'unimodal', 'multimodal', or 'unknown', "
                f"got {self.modality!r}"
            )
        # Allow bounds to be a dict or a tuple
        if isinstance(self.bounds, dict):
            for name, (lo, hi) in self.bounds.items():
                if lo >= hi:
                    raise ValueError(
                        f"Bound for {name!r}: lower ({lo}) must be < upper ({hi})"
                    )
        elif isinstance(self.bounds, tuple) and len(self.bounds) == 2:
            lo, hi = self.bounds
            if lo >= hi:
                raise ValueError(f"Bound tuple: lower ({lo}) must be < upper ({hi})")
        else:
            raise ValueError("bounds must be a dict or a tuple of length 2")
        # Validate prior parameters
        if self.length_scale_prior_median <= 0:
            raise ValueError(
                f"length_scale_prior_median must be > 0, got {self.length_scale_prior_median}"
            )
        if self.length_scale_prior_scale <= 0:
            raise ValueError(
                f"length_scale_prior_scale must be > 0, got {self.length_scale_prior_scale}"
            )
        if self.warp_prior_median <= 0:
            raise ValueError(
                f"warp_prior_median must be > 0, got {self.warp_prior_median}"
            )
        if self.warp_prior_scale <= 0:
            raise ValueError(
                f"warp_prior_scale must be > 0, got {self.warp_prior_scale}"
            )
        # Set optimization budget eagerly (also updated in update_from_data)
        self.optimization_budget = self.horizon - self.warmstart

    def update_from_data(self, n_obs: int, param_names: list[str]) -> None:
        """Update derived fields after data loading. Also expands bounds if needed."""
        self.n_obs = n_obs
        self.n_dim = len(param_names)
        self.remaining = self.horizon - n_obs
        self.optimization_budget = self.horizon - self.warmstart
        self.param_names = list(param_names)
        # If bounds is a tuple, expand to dict using param_names
        if isinstance(self.bounds, tuple) and len(self.bounds) == 2:
            lo, hi = self.bounds
            self.bounds = {name: (lo, hi) for name in param_names}
