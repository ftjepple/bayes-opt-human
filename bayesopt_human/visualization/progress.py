"""Progress chart visualization."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from bayesopt_human.utils import stagnation_length


def plot_progress(
    observations: pd.DataFrame,
    objective_column: str,
    direction: str,
    gp_predicted_optima: list[float] | None = None,
    remaining: int | None = None,
    warmstart: int = 0,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Plot progress chart.

    Shows observation values, best-so-far trace, and optionally
    the GP's predicted optimum at each historical step.
    Warmstart observations are shown in gray; optimization-phase
    observations in steelblue.

    Args:
        observations: DataFrame with observation data.
        objective_column: Name of the objective column.
        direction: 'minimize' or 'maximize'.
        gp_predicted_optima: GP predicted optimum at each historical step.
        remaining: Number of evaluations remaining.
        warmstart: Number of warmstart observations (shown in gray).
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    y = observations[objective_column].values
    indices = np.arange(1, len(y) + 1)

    if direction == "minimize":
        best_so_far = np.minimum.accumulate(y)
    else:
        best_so_far = np.maximum.accumulate(y)

    stag = stagnation_length(y, direction, warmstart=warmstart)

    fig, ax = plt.subplots(figsize=figsize)

    # Observed values — split by warmstart phase
    if warmstart > 0 and warmstart < len(y):
        ax.scatter(indices[:warmstart], y[:warmstart],
                   c="#999999", s=40, zorder=3, label="Warmstart")
        ax.scatter(indices[warmstart:], y[warmstart:],
                   c="steelblue", s=40, zorder=3, label="Observations")
        ax.axvline(warmstart + 0.5, color="#999999", linestyle="--",
                   linewidth=1, alpha=0.5)
    else:
        ax.scatter(indices, y, c="steelblue", s=40, zorder=3, label="Observations")

    # Best-so-far
    ax.step(indices, best_so_far, where="post", color="darkgreen",
            linewidth=2, label="Best so far")

    # GP predicted optima
    if gp_predicted_optima is not None and len(gp_predicted_optima) > 0:
        gp_idx = indices[:len(gp_predicted_optima)]
        ax.plot(gp_idx, gp_predicted_optima, "--", color="orange",
                linewidth=1.5, label="GP predicted optimum")

    ax.set_xlabel("Observation index")
    ax.set_ylabel(objective_column)
    ax.set_title("Optimization Progress")
    ax.legend(loc="best")

    # Annotations
    text_parts = [f"Stagnation: {stag} steps"]
    if remaining is not None:
        text_parts.append(f"Remaining: {remaining}")
    text_parts.append(f"Best: {best_so_far[-1]:.4g}")
    annotation = "\n".join(text_parts)
    ax.annotate(
        annotation,
        xy=(0.02, 0.98), xycoords="axes fraction",
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
    )

    fig.tight_layout()
    return fig
