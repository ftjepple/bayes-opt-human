"""Candidate-phase reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from bayesopt_human.acquisition.candidates import Candidate
from bayesopt_human.config import OptimizationConfig
from bayesopt_human.gp.surrogate_space import SurrogateSpaceTransform
from bayesopt_human.reporting.base import Report
from bayesopt_human.visualization.projections import (
    plot_pca_projection,
    plot_top2_projection,
)

if TYPE_CHECKING:
    from bayesopt_human.optimizer.step import CandidateResult


@dataclass
class CandidateContext:
    """Everything a candidate report needs to render.

    ``candidate`` is ``None`` for summary reports (all candidates plotted at
    once) and a single ``Candidate`` for individual reports.
    """

    result: "CandidateResult"
    config: OptimizationConfig
    candidate: Candidate | None = None
    ranks: dict[str, int] | None = None


# ---------------------------------------------------------------------------
# Projections (shared between summary and individual reports)
# ---------------------------------------------------------------------------


def _plot_projection(ctx: CandidateContext, kind: str) -> plt.Figure:
    result = ctx.result
    data = result.data

    if ctx.candidate is None:
        candidates = result.candidates
        ranks = {rec.source: rec.priority for rec in result.recommendations}
    else:
        candidates = [ctx.candidate]
        ranks = ctx.ranks or {}

    local_optima = (
        result.modality.local_optima_coords_flat
        if len(result.modality.local_optima_coords_flat) > 0
        else None
    )

    common = dict(
        X=data.X_norm,
        y=data.y,
        fitted_gp=result.fitted_gp,
        candidates=candidates,
        local_optima=local_optima,
        direction=ctx.config.direction,
        ranks=ranks,
    )

    if kind == "importance_2d":
        fig = plot_top2_projection(**common, param_names=data.param_names)
    elif kind == "pca_2d":
        fig = plot_pca_projection(**common)
    else:
        raise ValueError(f"Unknown projection kind: {kind}")

    plt.show()
    return fig


def _show_importance_2d(ctx: CandidateContext) -> plt.Figure:
    return _plot_projection(ctx, "importance_2d")


def _show_pca_2d(ctx: CandidateContext) -> plt.Figure:
    return _plot_projection(ctx, "pca_2d")


# ---------------------------------------------------------------------------
# Observation tables (individual reports only)
# ---------------------------------------------------------------------------


def _show_observations_raw(ctx: CandidateContext) -> None:
    assert ctx.candidate is not None, "observations_raw requires an individual candidate"
    result = ctx.result
    data = result.data
    fitted_gp = result.fitted_gp
    candidate = ctx.candidate

    # Projected output at candidate: inverse-transform surrogate mean
    cand_mean_sur = np.array([candidate.posterior_mean])
    cand_mean_tf = fitted_gp.iqr_normalizer.inverse_transform(cand_mean_sur)
    cand_mean_raw = float(fitted_gp.output_transform.inverse(cand_mean_tf)[0])

    tf_name = fitted_gp.output_transform.transform_type
    note = "" if tf_name == "none" else f" (inverse of {tf_name}; approximate)"

    _print_observations_table(
        config=ctx.config,
        X=data.X_raw,
        y=data.y,
        param_names=data.param_names,
        cand_X=candidate.coordinates_raw,
        cand_y=cand_mean_raw,
        candidate_source=candidate.source,
        space_label="raw",
        obj_label="objective",
        candidate_note=f"proposed (projected output{note})",
    )


def _show_observations_sur(ctx: CandidateContext) -> None:
    assert ctx.candidate is not None, "observations_sur requires an individual candidate"
    result = ctx.result
    data = result.data
    fitted_gp = result.fitted_gp
    candidate = ctx.candidate

    sur_tf = SurrogateSpaceTransform(fitted_gp)
    X_sur = sur_tf.to_surrogate(data.X_norm)
    y_sur = fitted_gp.iqr_normalizer.transform(
        fitted_gp.output_transform.forward(data.y)
    )

    cand_X_sur = candidate.coordinates_surrogate
    cand_y_sur = float(candidate.posterior_mean)

    tf_name = fitted_gp.output_transform.transform_type
    tf_label = f"{tf_name} + IQR" if tf_name != "none" else "IQR"
    param_names = [f"{n}*" for n in data.param_names]

    _print_observations_table(
        config=ctx.config,
        X=X_sur,
        y=y_sur,
        param_names=param_names,
        cand_X=cand_X_sur,
        cand_y=cand_y_sur,
        candidate_source=candidate.source,
        space_label="surrogate",
        obj_label=f"obj ({tf_label})",
        candidate_note="proposed (GP posterior mean)",
    )


def _print_observations_table(
    *,
    config: OptimizationConfig,
    X: np.ndarray,
    y: np.ndarray,
    param_names: list[str],
    cand_X: np.ndarray,
    cand_y: float,
    candidate_source: str,
    space_label: str,
    obj_label: str,
    candidate_note: str,
) -> None:
    """Print an observations table with distances in the same space as the
    coordinates, highlight the 5 closest, and append the candidate row."""
    warmstart = config.warmstart
    direction = config.direction
    n_obs = len(y)

    distances = np.linalg.norm(X - cand_X, axis=1)

    n_highlight = min(5, n_obs)
    closest_idx_set = set(np.argsort(distances)[:n_highlight].tolist())

    best_idx = int(np.argmin(y)) if direction == "minimize" else int(np.argmax(y))

    print(f"\nObservations and proposed candidate [{candidate_source}]")
    print(
        f"  {n_obs} observations, {len(param_names)} dimensions, "
        f"warmstart={warmstart}, direction={direction}"
    )
    print(
        f"  Distances computed in {space_label} space. "
        f"{n_highlight} closest highlighted.\n"
    )

    idx_width = max(3, len(str(n_obs - 1)))
    phase_width = 6
    dist_width = 12
    obj_width = max(len(obj_label), 12)
    param_widths = [max(len(name), 10) for name in param_names]

    header_parts = [
        "#".rjust(idx_width),
        "Phase".ljust(phase_width),
        "Distance".rjust(dist_width),
        obj_label.rjust(obj_width),
    ]
    for name, w in zip(param_names, param_widths):
        header_parts.append(name.rjust(w))
    header = "  ".join(header_parts)
    separator = "-" * (len(header) + 16)

    print(header)
    print(separator)

    order = np.argsort(distances)
    for rank, i in enumerate(order):
        phase = "warm" if i < warmstart else "opt"
        row_parts = [
            str(i).rjust(idx_width),
            phase.ljust(phase_width),
            f"{distances[i]:.6g}".rjust(dist_width),
            f"{y[i]:.6g}".rjust(obj_width),
        ]
        for j, w in enumerate(param_widths):
            row_parts.append(f"{X[i, j]:.6g}".rjust(w))
        line = "  ".join(row_parts)
        markers = []
        if i in closest_idx_set:
            markers.append(f"closest #{rank + 1}")
        if i == best_idx:
            markers.append("best")
        if markers:
            line += "  <-- " + ", ".join(markers)
        print(line)

    print(separator)
    cand_row_parts = [
        "*".rjust(idx_width),
        "cand".ljust(phase_width),
        "-".rjust(dist_width),
        f"{cand_y:.6g}".rjust(obj_width),
    ]
    for j, w in enumerate(param_widths):
        cand_row_parts.append(f"{cand_X[j]:.6g}".rjust(w))
    cand_line = "  ".join(cand_row_parts)
    cand_line += f"  <-- {candidate_note}"
    print(cand_line)
    print(separator)


# ---------------------------------------------------------------------------
# Report registries
# ---------------------------------------------------------------------------


SUMMARY_REPORTS: list[Report] = [
    Report(
        "importance_2d",
        "2D projection (top-2 importance, raw + surrogate side-by-side)",
        _show_importance_2d,
    ),
    Report(
        "pca_2d",
        "2D projection (PCA, raw + surrogate side-by-side)",
        _show_pca_2d,
    ),
]

INDIVIDUAL_REPORTS: list[Report] = [
    Report(
        "observations_raw",
        "Table of observations + candidate, sorted by distance (raw space)",
        _show_observations_raw,
    ),
    Report(
        "observations_sur",
        "Table of observations + candidate, sorted by distance (surrogate space)",
        _show_observations_sur,
    ),
    Report(
        "importance_2d",
        "2D projection (top-2 importance, raw + surrogate side-by-side)",
        _show_importance_2d,
    ),
    Report(
        "pca_2d",
        "2D projection (PCA, raw + surrogate side-by-side)",
        _show_pca_2d,
    ),
]
