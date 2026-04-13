"""Model selection diagnostic plots: residuals, LOO sigma, obs-vs-pred, importance."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_residual_histogram(
    residuals: np.ndarray,
    residuals_raw: np.ndarray | None = None,
    subtitle: str | None = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Side-by-side histograms of LOO residuals in raw and surrogate space.

    Args:
        residuals: LOO residuals in surrogate (GP) space, shape (N,).
        residuals_raw: LOO residuals in raw (original) space, shape (N,).
            If None, only the surrogate-space histogram is shown.
        subtitle: Optional guidance note displayed below the title.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    if residuals_raw is not None:
        fig, (ax_raw, ax_sur) = plt.subplots(1, 2, figsize=figsize)

        ax_raw.hist(residuals_raw, bins=20, color="steelblue", edgecolor="black", alpha=0.8)
        ax_raw.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero")
        ax_raw.set_xlabel("LOO Residual")
        ax_raw.set_ylabel("Count")
        ax_raw.set_title("Raw Space")
        ax_raw.legend(loc="best", fontsize=8)

        ax_sur.hist(residuals, bins=20, color="darkorange", edgecolor="black", alpha=0.8)
        ax_sur.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero")
        ax_sur.set_xlabel("LOO Residual")
        ax_sur.set_ylabel("Count")
        ax_sur.set_title("Surrogate Space")
        ax_sur.legend(loc="best", fontsize=8)

        fig.suptitle("LOO Residual Distribution", fontsize=12)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(residuals, bins=20, color="steelblue", edgecolor="black", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero")
        ax.set_xlabel("LOO Residual")
        ax.set_ylabel("Count")
        ax.set_title("LOO Residual Distribution")
        ax.legend(loc="best", fontsize=8)

    if subtitle:
        fig.text(0.5, 0.01, subtitle, ha="center", fontsize=9, style="italic", color="#666666")
        fig.tight_layout(rect=[0, 0.04, 1, 1])
    else:
        fig.tight_layout()
    return fig


def plot_residuals_vs_x(
    X: np.ndarray,
    residuals: np.ndarray,
    param_names: list[str] | None = None,
    subtitle: str | None = None,
    figsize: tuple = (14, 4),
) -> plt.Figure:
    """Scatter of LOO residuals vs each input dimension.

    Args:
        X: Normalized inputs, shape (N, D).
        residuals: LOO residuals, shape (N,).
        param_names: Dimension names.
        subtitle: Optional guidance note displayed below the title.
        figsize: Figure size (width is scaled by number of dims).

    Returns:
        Matplotlib Figure.
    """
    n_dim = X.shape[1]
    names = param_names or [f"x{i}" for i in range(n_dim)]
    fig_w = max(figsize[0], 3.5 * n_dim)

    fig, axes = plt.subplots(1, n_dim, figsize=(fig_w, figsize[1]), squeeze=False)

    for i in range(n_dim):
        ax = axes[0, i]
        ax.scatter(X[:, i], residuals, s=30, alpha=0.7, color="steelblue", edgecolors="black", linewidth=0.3)
        ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel(names[i])
        if i == 0:
            ax.set_ylabel("LOO Residual")

    fig.suptitle("LOO Residuals vs Input Dimensions", fontsize=12)
    if subtitle:
        fig.text(0.5, 0.01, subtitle, ha="center", fontsize=9, style="italic", color="#666666")
        fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_sigma_vs_x(
    X: np.ndarray,
    sigma_loo: np.ndarray,
    param_names: list[str] | None = None,
    subtitle: str | None = None,
    figsize: tuple = (14, 4),
) -> plt.Figure:
    """Scatter of LOO predicted sigma vs each input dimension.

    Args:
        X: Normalized inputs, shape (N, D).
        sigma_loo: LOO predicted standard deviations, shape (N,).
        param_names: Dimension names.
        subtitle: Optional guidance note displayed below the title.
        figsize: Figure size (width is scaled by number of dims).

    Returns:
        Matplotlib Figure.
    """
    n_dim = X.shape[1]
    names = param_names or [f"x{i}" for i in range(n_dim)]
    fig_w = max(figsize[0], 3.5 * n_dim)

    fig, axes = plt.subplots(1, n_dim, figsize=(fig_w, figsize[1]), squeeze=False)

    for i in range(n_dim):
        ax = axes[0, i]
        ax.scatter(X[:, i], sigma_loo, s=30, alpha=0.7, color="darkorange", edgecolors="black", linewidth=0.3)
        ax.set_xlabel(names[i])
        if i == 0:
            ax.set_ylabel("LOO Sigma")

    fig.suptitle("LOO Predicted Sigma vs Input Dimensions", fontsize=12)
    if subtitle:
        fig.text(0.5, 0.01, subtitle, ha="center", fontsize=9, style="italic", color="#666666")
        fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_observed_vs_predicted(
    y_actual: np.ndarray,
    mu_loo: np.ndarray,
    sigma_loo: np.ndarray | None = None,
    subtitle: str | None = None,
    figsize: tuple = (7, 7),
) -> plt.Figure:
    """Observed vs LOO-predicted scatter with 1:1 line and error bars.

    Args:
        y_actual: Actual observed values (surrogate space), shape (N,).
        mu_loo: LOO predicted means (surrogate space), shape (N,).
        sigma_loo: LOO predicted standard deviations, shape (N,).
            If provided, 2-sigma error bars are drawn.
        subtitle: Optional guidance note displayed below the title.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Error bars (2-sigma)
    if sigma_loo is not None:
        ax.errorbar(
            y_actual, mu_loo, yerr=2 * sigma_loo,
            fmt="none", ecolor="#cccccc", elinewidth=0.8, zorder=1,
        )

    ax.scatter(
        y_actual, mu_loo, s=40, c="steelblue",
        edgecolors="black", linewidth=0.4, zorder=2,
    )

    # 1:1 reference line
    lo = min(y_actual.min(), mu_loo.min())
    hi = max(y_actual.max(), mu_loo.max())
    margin = 0.05 * (hi - lo)
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "r--", linewidth=1.2, label="1:1 line", zorder=0)

    ax.set_xlabel("Observed (surrogate space)")
    ax.set_ylabel("LOO Predicted (surrogate space)")
    ax.set_title("Observed vs LOO-Predicted")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    if subtitle:
        fig.text(0.5, 0.01, subtitle, ha="center", fontsize=9, style="italic", color="#666666")
        fig.tight_layout(rect=[0, 0.04, 1, 1])
    else:
        fig.tight_layout()
    return fig


def plot_feature_importance(
    length_scales: np.ndarray,
    param_names: list[str] | None = None,
    subtitle: str | None = None,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Horizontal bar chart of feature importances (1 / length scale).

    Args:
        length_scales: Per-dimension length scales, shape (D,).
        param_names: Dimension names.
        subtitle: Optional guidance note displayed below the title.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    n_dim = len(length_scales)
    names = param_names or [f"x{i}" for i in range(n_dim)]
    importances = 1.0 / length_scales
    # Normalize to percentages
    total = importances.sum()
    pcts = 100.0 * importances / total if total > 0 else importances

    # Sort by importance ascending (so most important is at top)
    order = np.argsort(pcts)
    sorted_names = [names[i] for i in order]
    sorted_pcts = pcts[order]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(range(n_dim), sorted_pcts, color="steelblue", edgecolor="black", linewidth=0.4)

    # Value labels on bars
    for bar, pct in zip(bars, sorted_pcts):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%", va="center", fontsize=9,
        )

    ax.set_yticks(range(n_dim))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Relative Importance (%)")
    ax.set_title("Feature Importance (1 / length scale)")

    if subtitle:
        fig.text(0.5, 0.01, subtitle, ha="center", fontsize=9, style="italic", color="#666666")
        fig.tight_layout(rect=[0, 0.04, 1, 1])
    else:
        fig.tight_layout()
    return fig


def plot_transform_comparison(
    model_labels: list[str],
    y_actuals: list[np.ndarray],
    mu_loos: list[np.ndarray],
    subtitle: str | None = None,
    figsize: tuple = (7, 7),
) -> plt.Figure:
    """Overlay observed-vs-predicted for multiple models on one plot.

    Args:
        model_labels: Display label for each model (e.g. "log (ARD)").
        y_actuals: List of actual values arrays (surrogate space per model).
        mu_loos: List of LOO predicted means arrays (surrogate space per model).
        subtitle: Optional guidance note displayed below the title.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    colors = ["steelblue", "darkorange", "forestgreen", "crimson", "mediumpurple",
              "saddlebrown", "deeppink", "olive"]
    fig, ax = plt.subplots(figsize=figsize)

    all_vals = np.concatenate(y_actuals + mu_loos)
    lo, hi = float(all_vals.min()), float(all_vals.max())
    margin = 0.05 * (hi - lo)
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", linewidth=1, alpha=0.5, label="1:1 line", zorder=0)

    for i, (label, y_act, mu) in enumerate(zip(model_labels, y_actuals, mu_loos)):
        c = colors[i % len(colors)]
        ax.scatter(y_act, mu, s=30, c=c, edgecolors="black", linewidth=0.3,
                   alpha=0.7, label=label, zorder=2 + i)

    ax.set_xlabel("Observed (surrogate space)")
    ax.set_ylabel("LOO Predicted (surrogate space)")
    ax.set_title("Transform Comparison: Observed vs LOO-Predicted")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    if subtitle:
        fig.text(0.5, 0.01, subtitle, ha="center", fontsize=9, style="italic", color="#666666")
        fig.tight_layout(rect=[0, 0.04, 1, 1])
    else:
        fig.tight_layout()
    return fig
