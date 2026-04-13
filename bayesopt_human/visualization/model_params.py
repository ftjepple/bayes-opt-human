"""Length scale and warp parameter charts."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

from bayesopt_human.gp.model import FittedGP


def plot_length_scales(
    fitted_gp: FittedGP,
    param_names: list[str],
    length_scale_bounds: tuple[float, float] = None,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """Plot learned length scales on log scale with boundary flags.

    Args:
        fitted_gp: Fitted GP model.
        param_names: Names of parameter dimensions.
        length_scale_bounds: (lower, upper) for boundary flagging.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    ls = fitted_gp.length_scales
    n_dim = len(ls)
    if length_scale_bounds is None:
        from bayesopt_human.config import OptimizationConfig
        lo = getattr(OptimizationConfig, 'length_scale_lower_bound', 10e-3)
        hi = getattr(OptimizationConfig, 'length_scale_upper_bound', 1e3)
    else:
        lo, hi = length_scale_bounds

    fig, ax = plt.subplots(figsize=figsize)

    colors = []
    for l in ls:
        if l < lo * 2:
            colors.append("red")
        elif l > hi * 0.5:
            colors.append("orange")
        else:
            colors.append("steelblue")

    x_pos = np.arange(n_dim)
    ax.bar(x_pos, ls, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")

    # Boundary lines
    ax.axhline(lo, color="red", linestyle="--", alpha=0.5, label=f"Lower bound ({lo})")
    ax.axhline(hi, color="orange", linestyle="--", alpha=0.5, label=f"Upper bound ({hi})")

    ax.set_xticks(x_pos)
    labels = param_names if len(param_names) == n_dim else [f"x{i}" for i in range(n_dim)]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Length scale (log)")
    kernel_type = "ARD" if fitted_gp.ard else "Isotropic"
    ax.set_title(f"Learned {kernel_type} Length Scales")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    return fig


def plot_warp_functions(
    fitted_gp: FittedGP,
    param_names: list[str],
    figsize: tuple = (12, 8),
    low_threshold: float = 0.1,
    high_threshold: float = 20.0,
) -> plt.Figure | None:
    """Small-multiples plot of learned Beta CDF warp per dimension.

    Args:
        fitted_gp: Fitted GP model (must have warping enabled).
        param_names: Names of parameter dimensions.
        figsize: Figure size.

    Returns:
        Matplotlib Figure, or None if warping is not enabled.
    """
    if fitted_gp.warp_parameters is None:
        return None

    c0 = fitted_gp.warp_parameters["concentration0"]
    c1 = fitted_gp.warp_parameters["concentration1"]
    n_dim = len(c0)

    n_cols = min(4, n_dim)
    n_rows = (n_dim + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    x = np.linspace(0.001, 0.999, 200)

    def _warp_color(a: float, b: float) -> str:
        a_extreme = (a < low_threshold) or (a > high_threshold)
        b_extreme = (b < low_threshold) or (b > high_threshold)
        if a_extreme and b_extreme:
            return "red"
        elif a_extreme or b_extreme:
            return "orangered"
        else:
            return "steelblue"

    for i in range(n_dim):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]

        # Beta CDF
        cdf_vals = beta_dist.cdf(x, c1[i], c0[i])
        color = _warp_color(c1[i], c0[i])
        ax.plot(x, cdf_vals, color=color, linewidth=2)
        ax.plot(x, x, "--", color="gray", alpha=0.5, label="Identity")

        label = param_names[i] if i < len(param_names) else f"x{i}"
        extreme_marker = " *" if color != "steelblue" else ""
        ax.set_title(f"{label}{extreme_marker}\na={c1[i]:.2f}, b={c0[i]:.2f}", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if row == n_rows - 1:
            ax.set_xlabel("Input")
        if col == 0:
            ax.set_ylabel("Warped")

    # Hide unused axes
    for i in range(n_dim, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle("Learned Input Warp Functions (Beta CDF)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
