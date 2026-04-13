"""2D projection plots (top-2, PCA, small-multiples)."""

from __future__ import annotations

import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from bayesopt_human.acquisition.candidates import Candidate
from bayesopt_human.gp.model import FittedGP
from bayesopt_human.gp.surrogate_space import SurrogateSpaceTransform

# Pattern-based mapping from full source strings to short labels
_SHORT_LABEL_PATTERNS: list[tuple[str, str]] = [
    (r"^Expected Improvement$", "EI"),
    (r"^(UCB|LCB) \(kappa=high", lambda m: f"{m.group(1)} HI"),
    (r"^(UCB|LCB) \(kappa=medium", lambda m: f"{m.group(1)} MID"),
    (r"^(UCB|LCB) \(kappa=low", lambda m: f"{m.group(1)} LO"),
    (r"^Max GP Mean$", "MAX MEAN"),
    (r"^Max-min distance \(flat\)$", "MAXMIN FLAT"),
    (r"^Max-min distance \(surrogate\)$", "MAXMIN SUR"),
    (r"^Max Posterior Variance$", "MPV"),
]

# Candidate marker styling
_DIAMOND_GRAY = "#9E9E9E"
_DIAMOND_EDGE = "#616161"


def short_label(source: str) -> str:
    """Convert a full candidate source string to a short display label."""
    for pattern, replacement in _SHORT_LABEL_PATTERNS:
        m = re.match(pattern, source)
        if m:
            return replacement(m) if callable(replacement) else replacement
    # Fallback: first 10 chars
    return source[:10]


def _plot_ranked_diamond(
    ax: plt.Axes, x: float, y: float, rank: int,
    size: int = 180, fontsize: int = 8,
) -> None:
    """Plot a gray diamond marker with the rank number centered inside."""
    ax.scatter(
        x, y, marker="D", s=size,
        c=_DIAMOND_GRAY, edgecolors=_DIAMOND_EDGE,
        linewidth=1, zorder=4,
    )
    ax.text(
        x, y, str(rank),
        ha="center", va="center", fontsize=fontsize,
        fontweight="bold", color="white", zorder=5,
    )


def _get_coordinates(
    X: np.ndarray,
    candidates: list[Candidate],
    local_optima: np.ndarray | None,
    space: str,
    surrogate_transform: SurrogateSpaceTransform | None,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray | None]:
    """Get coordinates in the requested space."""
    if space == "surrogate" and surrogate_transform is not None:
        X_plot = surrogate_transform.to_surrogate(X)
        cand_coords = [
            surrogate_transform.to_surrogate(c.coordinates_normalized.reshape(1, -1)).flatten()
            for c in candidates
        ]
        lo_coords = (
            surrogate_transform.to_surrogate(local_optima)
            if local_optima is not None and len(local_optima) > 0
            else None
        )
    else:
        X_plot = X
        cand_coords = [c.coordinates_normalized for c in candidates]
        lo_coords = local_optima

    return X_plot, cand_coords, lo_coords


def _find_best_obs_idx(y: np.ndarray, direction: str) -> int:
    """Return the index of the best observed point."""
    if direction == "minimize":
        return int(np.argmin(y))
    return int(np.argmax(y))


def _ranked_candidates(
    candidates: list[Candidate],
    ranks: dict[str, int],
) -> list[tuple[Candidate, int]]:
    """Return (candidate, rank) pairs sorted by rank ascending."""
    pairs = [(c, ranks.get(c.source, i + 1)) for i, c in enumerate(candidates)]
    pairs.sort(key=lambda p: p[1])
    return pairs


def _make_legend(
    ax: plt.Axes,
    candidates: list[Candidate],
    ranks: dict[str, int],
) -> None:
    """Build a legend with global best, local optima, and ranked candidate entries."""
    handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="none",
            markeredgecolor="red", markeredgewidth=2.5, markersize=10,
            label="Global best",
        ),
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="none",
            markeredgecolor="red", markeredgewidth=1.2, markersize=9,
            linestyle="--", label="Local optima",
        ),
    ]
    for cand, rank in _ranked_candidates(candidates, ranks):
        label = short_label(cand.source)
        handles.append(
            Line2D(
                [0], [0], marker="D", color="none",
                markerfacecolor=_DIAMOND_GRAY, markeredgecolor=_DIAMOND_EDGE,
                markeredgewidth=0.5, markersize=8, label=f"#{rank} {label}",
            )
        )
    ax.legend(
        handles=handles, loc="best", fontsize=7,
        labelspacing=1.0, handletextpad=0.8,
    )


# ---------------------------------------------------------------------------
# Top-2 Feature Importance Projection
# ---------------------------------------------------------------------------

def _plot_top2_on_ax(
    ax: plt.Axes,
    X_plot: np.ndarray,
    y: np.ndarray,
    top2: np.ndarray,
    candidates: list[Candidate],
    cand_coords: list[np.ndarray],
    lo_coords: np.ndarray | None,
    direction: str,
    ranks: dict[str, int],
    param_names: list[str] | None,
    space_label: str,
):
    """Plot top-2 importance scatter on a given axes. Returns PathCollection."""
    sc = ax.scatter(
        X_plot[:, top2[0]], X_plot[:, top2[1]],
        c=y, cmap="viridis", s=50, zorder=2, edgecolors="black", linewidth=0.5,
    )

    best_idx = _find_best_obs_idx(y, direction)
    ax.scatter(
        X_plot[best_idx, top2[0]], X_plot[best_idx, top2[1]],
        facecolors="none", edgecolors="red", s=200, linewidth=2.5, zorder=3,
    )

    if lo_coords is not None and len(lo_coords) > 0:
        ax.scatter(
            lo_coords[:, top2[0]], lo_coords[:, top2[1]],
            facecolors="none", edgecolors="red", s=150, linewidth=1.2,
            linestyle="--", zorder=3,
        )

    coord_map = {id(c): coord for coord, c in zip(cand_coords, candidates)}
    for cand, rank in reversed(_ranked_candidates(candidates, ranks)):
        coord = coord_map[id(cand)]
        _plot_ranked_diamond(ax, coord[top2[0]], coord[top2[1]], rank)

    names = param_names or [f"x{i}" for i in range(X_plot.shape[1])]
    ax.set_xlabel(names[top2[0]])
    ax.set_ylabel(names[top2[1]])
    ax.set_title(f"Top-2 Importance ({space_label} space)")
    return sc


def plot_top2_projection(
    X: np.ndarray,
    y: np.ndarray,
    fitted_gp: FittedGP,
    candidates: list[Candidate],
    local_optima: np.ndarray | None = None,
    direction: str = "maximize",
    ranks: dict[str, int] | None = None,
    param_names: list[str] | None = None,
    figsize: tuple = (16, 7),
) -> plt.Figure:
    """Side-by-side top-2 importance projection: raw (left) and surrogate (right).

    Args:
        X: Normalized inputs in flat space, shape (N, D).
        y: Objective values, shape (N,).
        fitted_gp: Fitted GP model.
        candidates: List of candidate points.
        local_optima: Coordinates of local optima in flat space.
        direction: 'minimize' or 'maximize'.
        ranks: Mapping from candidate source to recommendation rank.
        param_names: Dimension names.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    ranks = ranks or {}
    importances = 1.0 / fitted_gp.length_scales
    top2 = np.argsort(importances)[-2:][::-1]
    sur_tf = SurrogateSpaceTransform(fitted_gp)

    # Transformed y for surrogate subplot
    y_sur = fitted_gp.iqr_normalizer.transform(fitted_gp.output_transform.forward(y))
    tf_name = fitted_gp.output_transform.transform_type

    fig, (ax_raw, ax_sur) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Raw space — raw objective
    X_raw, cand_raw, lo_raw = _get_coordinates(X, candidates, local_optima, "flat", None)
    sc_raw = _plot_top2_on_ax(
        ax_raw, X_raw, y, top2, candidates, cand_raw, lo_raw,
        direction, ranks, param_names, "Raw",
    )
    fig.colorbar(sc_raw, ax=ax_raw, label="Objective", shrink=0.8)

    # Surrogate space — transformed objective
    X_sur, cand_sur, lo_sur = _get_coordinates(X, candidates, local_optima, "surrogate", sur_tf)
    sc_sur = _plot_top2_on_ax(
        ax_sur, X_sur, y_sur, top2, candidates, cand_sur, lo_sur,
        direction, ranks, param_names, "Surrogate",
    )
    sur_label = f"Objective ({tf_name} + IQR)" if tf_name != "none" else "Objective (IQR)"
    fig.colorbar(sc_sur, ax=ax_sur, label=sur_label, shrink=0.8)

    _make_legend(ax_sur, candidates, ranks)
    fig.suptitle("Top-2 Feature Importance Projection", fontsize=13)
    return fig


# ---------------------------------------------------------------------------
# PCA Projection
# ---------------------------------------------------------------------------

def _plot_pca_on_ax(
    ax: plt.Axes,
    X_plot: np.ndarray,
    y: np.ndarray,
    importances: np.ndarray,
    candidates: list[Candidate],
    cand_coords: list[np.ndarray],
    lo_coords: np.ndarray | None,
    direction: str,
    ranks: dict[str, int],
    space_label: str,
    n_label_recent: int = 5,
):
    """Plot weighted PCA projection on a given axes. Returns PathCollection."""
    weights = importances / importances.sum()
    X_weighted = X_plot * np.sqrt(weights)

    # Pin the solver and seed so the projection is reproducible regardless
    # of input size. ``svd_solver='auto'`` switches to a randomized solver
    # for large matrices, which would make the PCA output depend on the
    # global RNG state.
    pca = PCA(n_components=2, svd_solver="full", random_state=0)
    X_pca = pca.fit_transform(X_weighted)

    sc = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=y, cmap="viridis", s=50, zorder=2, edgecolors="black", linewidth=0.5,
    )

    best_idx = _find_best_obs_idx(y, direction)
    ax.scatter(
        X_pca[best_idx, 0], X_pca[best_idx, 1],
        facecolors="none", edgecolors="red", s=200, linewidth=2.5, zorder=3,
    )

    # Label the last n_label_recent observations with 1-based round numbers
    n_obs = len(y)
    if n_label_recent > 0 and n_obs > 0:
        start = max(0, n_obs - n_label_recent)
        for i in range(start, n_obs):
            ax.annotate(
                str(i + 1),
                (X_pca[i, 0], X_pca[i, 1]),
                textcoords="offset points", xytext=(5, 4),
                fontsize=7, color="#777777", zorder=4,
            )

    if lo_coords is not None and len(lo_coords) > 0:
        lo_weighted = lo_coords * np.sqrt(weights)
        lo_pca = pca.transform(lo_weighted)
        ax.scatter(
            lo_pca[:, 0], lo_pca[:, 1],
            facecolors="none", edgecolors="red", s=150, linewidth=1.2,
            linestyle="--", zorder=3,
        )

    pca_map: dict[int, np.ndarray] = {}
    for coord, cand in zip(cand_coords, candidates):
        coord_weighted = coord * np.sqrt(weights)
        pca_map[id(cand)] = pca.transform(coord_weighted.reshape(1, -1))[0]
    for cand, rank in reversed(_ranked_candidates(candidates, ranks)):
        cp = pca_map[id(cand)]
        _plot_ranked_diamond(ax, cp[0], cp[1], rank)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(f"Weighted PCA ({space_label} space)")
    return sc


def plot_pca_projection(
    X: np.ndarray,
    y: np.ndarray,
    fitted_gp: FittedGP,
    candidates: list[Candidate],
    local_optima: np.ndarray | None = None,
    direction: str = "maximize",
    ranks: dict[str, int] | None = None,
    figsize: tuple = (16, 7),
) -> plt.Figure:
    """Side-by-side PCA projection: raw (left) and surrogate (right).

    Args:
        X: Normalized inputs in flat space, shape (N, D).
        y: Objective values, shape (N,).
        fitted_gp: Fitted GP model.
        candidates: List of candidate points.
        local_optima: Coordinates of local optima in flat space.
        direction: 'minimize' or 'maximize'.
        ranks: Mapping from candidate source to recommendation rank.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    ranks = ranks or {}
    importances = 1.0 / fitted_gp.length_scales
    sur_tf = SurrogateSpaceTransform(fitted_gp)

    # Transformed y for surrogate subplot
    y_sur = fitted_gp.iqr_normalizer.transform(fitted_gp.output_transform.forward(y))
    tf_name = fitted_gp.output_transform.transform_type

    fig, (ax_raw, ax_sur) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Raw space — raw objective
    X_raw, cand_raw, lo_raw = _get_coordinates(X, candidates, local_optima, "flat", None)
    sc_raw = _plot_pca_on_ax(
        ax_raw, X_raw, y, importances, candidates, cand_raw, lo_raw,
        direction, ranks, "Raw",
    )
    fig.colorbar(sc_raw, ax=ax_raw, label="Objective", shrink=0.8)

    # Surrogate space — transformed objective
    X_sur, cand_sur, lo_sur = _get_coordinates(X, candidates, local_optima, "surrogate", sur_tf)
    sc_sur = _plot_pca_on_ax(
        ax_sur, X_sur, y_sur, importances, candidates, cand_sur, lo_sur,
        direction, ranks, "Surrogate",
    )
    sur_label = f"Objective ({tf_name} + IQR)" if tf_name != "none" else "Objective (IQR)"
    fig.colorbar(sc_sur, ax=ax_sur, label=sur_label, shrink=0.8)

    _make_legend(ax_sur, candidates, ranks)
    fig.suptitle("Weighted PCA Projection", fontsize=13)
    return fig


def plot_pca_observations(
    X: np.ndarray,
    y: np.ndarray,
    direction: str = "maximize",
    local_optima: np.ndarray | None = None,
    figsize: tuple = (8, 7),
) -> plt.Figure:
    """Unweighted PCA projection of observations — raw space, single panel.

    Used by the data phase, where no GP has been fitted yet so there are no
    per-dimension length scales to weight the PCA with. Inputs are expected
    to be already normalized to ``[0, 1)`` via the config bounds; outputs
    are colored by raw objective value.

    Args:
        X: Normalized inputs in flat space, shape (N, D).
        y: Raw objective values, shape (N,).
        direction: 'minimize' or 'maximize'. Used to circle the best point.
        local_optima: Optional array of local-optima coordinates in flat
            space, shape (M, D). Drawn as red dashed circles.
        figsize: Figure size.

    Returns:
        Matplotlib Figure with a single PCA axes.
    """
    n_dim = X.shape[1]
    importances = np.ones(n_dim)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    sc = _plot_pca_on_ax(
        ax, X, y, importances,
        candidates=[], cand_coords=[], lo_coords=local_optima,
        direction=direction, ranks={}, space_label="Raw",
    )
    ax.set_title("PCA (raw space, unweighted)")
    fig.colorbar(sc, ax=ax, label="Objective", shrink=0.8)
    return fig


# ---------------------------------------------------------------------------
# Pairwise Small-Multiples
# ---------------------------------------------------------------------------

def plot_pairwise(
    X: np.ndarray,
    y: np.ndarray,
    fitted_gp: FittedGP,
    candidates: list[Candidate],
    local_optima: np.ndarray | None = None,
    top_k: int = 4,
    space: str = "flat",
    direction: str = "maximize",
    ranks: dict[str, int] | None = None,
    param_names: list[str] | None = None,
    figsize: tuple = (12, 12),
) -> plt.Figure:
    """Small-multiples pairwise scatter of top-K important dimensions.

    Uses ranked gray diamond markers per candidate with a shared legend.

    Args:
        X: Normalized inputs in flat space, shape (N, D).
        y: Objective values, shape (N,).
        fitted_gp: Fitted GP model.
        candidates: List of candidate points.
        local_optima: Coordinates of local optima in flat space.
        top_k: Number of top dimensions.
        space: 'flat' or 'surrogate'.
        direction: 'minimize' or 'maximize'.
        ranks: Mapping from candidate source to recommendation rank.
        param_names: Dimension names.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    ranks = ranks or {}
    sur_tf = SurrogateSpaceTransform(fitted_gp) if space == "surrogate" else None
    X_plot, cand_coords, lo_coords = _get_coordinates(
        X, candidates, local_optima, space, sur_tf
    )

    # Use transformed y in surrogate space
    if space == "surrogate":
        y_plot = fitted_gp.iqr_normalizer.transform(fitted_gp.output_transform.forward(y))
    else:
        y_plot = y

    importances = 1.0 / fitted_gp.length_scales
    k = min(top_k, X.shape[1])
    top_dims = np.argsort(importances)[-k:][::-1]

    names = param_names or [f"x{i}" for i in range(X.shape[1])]
    best_idx = _find_best_obs_idx(y_plot, direction)

    fig, axes = plt.subplots(k, k, figsize=figsize, squeeze=False)

    for row_i, d_row in enumerate(top_dims):
        for col_i, d_col in enumerate(top_dims):
            ax = axes[row_i, col_i]

            if row_i == col_i:
                # Diagonal: marginal histogram
                ax.hist(X_plot[:, d_row], bins=15, color="steelblue", alpha=0.7)
                # Candidate locations as vertical lines
                for coord in cand_coords:
                    ax.axvline(
                        coord[d_row], color=_DIAMOND_GRAY,
                        linestyle="--", alpha=0.7,
                    )
            else:
                # Off-diagonal: scatter
                ax.scatter(
                    X_plot[:, d_col], X_plot[:, d_row],
                    c=y_plot, cmap="viridis", s=20, edgecolors="none", alpha=0.7,
                )

                # Global best
                ax.scatter(
                    X_plot[best_idx, d_col], X_plot[best_idx, d_row],
                    facecolors="none", edgecolors="red", s=120, linewidth=2,
                    zorder=3,
                )

                # Local optima
                if lo_coords is not None and len(lo_coords) > 0:
                    ax.scatter(
                        lo_coords[:, d_col], lo_coords[:, d_row],
                        facecolors="none", edgecolors="red", s=100, linewidth=1,
                        linestyle="--", zorder=3,
                    )

                # Candidates: gray diamonds with rank number (worst first)
                coord_map = {id(c): co for co, c in zip(cand_coords, candidates)}
                for cand, rank in reversed(_ranked_candidates(candidates, ranks)):
                    coord = coord_map[id(cand)]
                    _plot_ranked_diamond(
                        ax, coord[d_col], coord[d_row], rank,
                        size=100, fontsize=6,
                    )

            if row_i == k - 1:
                ax.set_xlabel(names[d_col], fontsize=8)
            else:
                ax.set_xticklabels([])
            if col_i == 0:
                ax.set_ylabel(names[d_row], fontsize=8)
            else:
                ax.set_yticklabels([])

    fig.suptitle(f"Pairwise Projections - Top {k} Dimensions ({space} space)", fontsize=12)

    # Add shared legend with spacing
    if candidates:
        legend_handles = []
        for cand, rank in _ranked_candidates(candidates, ranks):
            label = short_label(cand.source)
            legend_handles.append(
                Line2D(
                    [0], [0], marker="D", color="none",
                    markerfacecolor=_DIAMOND_GRAY, markeredgecolor=_DIAMOND_EDGE,
                    markeredgewidth=0.5, markersize=7, label=f"#{rank} {label}",
                )
            )
        fig.legend(
            handles=legend_handles, loc="lower center",
            ncol=min(4, len(legend_handles)), fontsize=7,
            labelspacing=1.0, handletextpad=0.8, columnspacing=2.0,
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    return fig
