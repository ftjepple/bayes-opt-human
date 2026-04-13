"""Model-phase reports."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayesopt_human.config import OptimizationConfig
from bayesopt_human.diagnostics.gp_diagnostics import gp_fit_summary
from bayesopt_human.gp.loo import analytical_loo_details
from bayesopt_human.gp.selection import ModelCandidate
from bayesopt_human.reporting.base import Report
from bayesopt_human.visualization.model_diagnostics import (
    plot_feature_importance,
    plot_observed_vs_predicted,
    plot_residual_histogram,
    plot_residuals_vs_x,
    plot_sigma_vs_x,
    plot_transform_comparison,
)
from bayesopt_human.visualization.model_params import (
    plot_length_scales,
    plot_warp_functions,
)
from bayesopt_human.visualization.progress import plot_progress
from bayesopt_human.visualization.projections import plot_pca_projection

if TYPE_CHECKING:
    from bayesopt_human.optimizer.step import ModelResult


# ---------------------------------------------------------------------------
# Contexts
# ---------------------------------------------------------------------------


@dataclass
class ModelSummaryContext:
    result: "ModelResult"


@dataclass
class ModelIndividualContext:
    result: "ModelResult"
    candidate: ModelCandidate
    config: OptimizationConfig
    choice: int


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _fmt(v: float, width: int = 8) -> str:
    """Format a metric value, switching to scientific notation when large."""
    if abs(v) >= 1000:
        return f"{v:{width}.2e}"
    return f"{v:{width}.4f}"


def _kernel_label(mc: ModelCandidate) -> str:
    return "WARP" if mc.warping else ("ARD" if mc.ard else "ISO")


# ---------------------------------------------------------------------------
# Summary reports
# ---------------------------------------------------------------------------


def _show_residual_histogram_overview(ctx: ModelSummaryContext) -> plt.Figure:
    """Grid of LOO residual histograms (surrogate space) for all model candidates."""
    candidates = ctx.result.candidates
    data = ctx.result.data
    n = len(candidates)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()
    for idx, mc in enumerate(candidates):
        ax = axes[idx]
        details = analytical_loo_details(mc.fitted_gp, data.X_norm, data.y)
        residuals = details["residuals"]
        ax.hist(residuals, bins=20, color="darkorange", edgecolor="black", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"[{idx}] {mc.output_transform} ({_kernel_label(mc)})", fontsize=10)
        if idx % cols == 0:
            ax.set_ylabel("Count")
        else:
            ax.set_ylabel("")
        if idx // cols == rows - 1:
            ax.set_xlabel("LOO Residual")
        else:
            ax.set_xlabel("")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("LOO Residual Distributions (Surrogate Space)", fontsize=14)
    fig.text(
        0.5, 0.01,
        "Compare residual spread across models. "
        "Tighter, symmetric distributions indicate better fit.",
        ha="center", fontsize=9, style="italic", color="#666666",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.show()
    return fig


def _show_length_scales_overview(ctx: ModelSummaryContext) -> None:
    """Display a table of all learned length scales for each model and dimension."""
    models = ctx.result.candidates
    param_names = ctx.result.data.param_names
    n_dims = len(param_names)

    in_jupyter = "ipykernel" in sys.modules

    def rag_circle(rag):
        if in_jupyter:
            color = {"red": "#d62728", "amber": "#ffbf00", "green": "#2ca02c"}[rag]
            return f'<span style="font-size:2em; color:{color}; vertical-align:middle;">&#9679;</span>'
        return {"red": "\u25cf", "amber": "\u25cd", "green": "\u2022"}[rag]

    rows = []
    for idx, mc in enumerate(models):
        ls = getattr(mc.fitted_gp, "length_scales", None)
        if ls is None:
            continue
        max_ls = float(np.max(ls))
        min_ls = float(np.min(ls))
        ratio = max_ls / min_ls if min_ls > 0 else np.inf
        ls_rag = getattr(mc, "ls_rag", "green")
        ls_flags = getattr(mc, "length_scale_flags", [])
        rows.append({
            "Idx": idx,
            "Transform": mc.output_transform,
            "Kernel": _kernel_label(mc),
            "LS": rag_circle(ls_rag),
            **{param_names[i]: f"{ls[i]:.4f}" for i in range(n_dims)},
            "max/min": f"{ratio:.1f}",
            "Notes": " | ".join(ls_flags) if ls_flags else "",
        })
    df = pd.DataFrame(rows)

    print("\nLength Scales for All Models:")
    if in_jupyter:
        from IPython.display import display, HTML
        styles = [
            dict(selector="th", props=[("text-align", "left")]),
            dict(selector="td", props=[("text-align", "left")]),
        ]
        display(HTML(
            df.style.hide(axis="index")
            .set_table_styles(styles)
            .to_html(escape=False)
        ))
    else:
        from tabulate import tabulate
        print(tabulate(df.values.tolist(), headers=df.columns.tolist(), tablefmt="pipe"))
    print(
        "  Length scales near bounds or with extreme max/min ratio warrant "
        "investigation (see marginal_effects)."
    )


def _show_transform_comparison(ctx: ModelSummaryContext) -> plt.Figure:
    """Overlay observed-vs-predicted for top non-demoted models."""
    data = ctx.result.data
    non_demoted = [mc for mc in ctx.result.candidates if not mc.demoted]
    to_compare = non_demoted[:3] if non_demoted else ctx.result.candidates[:3]

    labels = []
    y_actuals = []
    mu_loos = []
    for mc in to_compare:
        labels.append(f"{mc.output_transform} ({_kernel_label(mc)})")
        details = analytical_loo_details(mc.fitted_gp, data.X_norm, data.y)
        y_actuals.append(details["y_norm"])
        mu_loos.append(details["mu_loo"])

    fig = plot_transform_comparison(
        labels, y_actuals, mu_loos,
        subtitle=(
            "Compare how well each model predicts held-out points. "
            "Tighter clustering around the 1:1 line = better fit."
        ),
    )
    plt.show()
    return fig


def _show_gp_fit_overview(ctx: ModelSummaryContext) -> None:
    """Table of GP fit diagnostics for all model candidates."""
    data = ctx.result.data
    rows = []
    for idx, mc in enumerate(ctx.result.candidates):
        report = gp_fit_summary(mc.fitted_gp, data.X_norm, data.y)
        rows.append({
            "Idx": idx,
            "Transform": mc.output_transform,
            "Kernel": _kernel_label(mc),
            "LOO-MAE": _fmt(report.loo_mae).strip(),
            "LOO-LPD": _fmt(report.loo_lpd).strip(),
            "Conditioning": _fmt(report.conditioning).strip(),
            "Jitter": f"{report.jitter_used:.2e}",
            "Cond.Warn": "\u26a0" if report.conditioning_warning else "",
        })
    df = pd.DataFrame(rows)
    print("\nGP Fit Diagnostics:")
    print(df.to_string(index=False))
    print("  Lower LOO-MAE = better predictions. Higher LOO-LPD = better calibration.")


# ---------------------------------------------------------------------------
# Individual reports
# ---------------------------------------------------------------------------


def _show_surrogate_observations(ctx: ModelIndividualContext) -> None:
    """Table of all evaluated points with surrogate-space objective values."""
    data = ctx.result.data
    config = ctx.config
    fitted_gp = ctx.candidate.fitted_gp

    y_sur = fitted_gp.iqr_normalizer.transform(
        fitted_gp.output_transform.forward(data.y)
    )
    warmstart = config.warmstart
    direction = config.direction
    param_names = data.param_names
    n_obs = len(data.y)

    best_idx = int(np.argmin(y_sur)) if direction == "minimize" else int(np.argmax(y_sur))

    tf_name = fitted_gp.output_transform.transform_type
    tf_label = f"{tf_name} + IQR" if tf_name != "none" else "IQR"

    print(f"\nObservations in surrogate space ({tf_label})")
    print(
        f"  {n_obs} points, {len(param_names)} dimensions, "
        f"warmstart={warmstart}, direction={direction}\n"
    )

    idx_width = max(3, len(str(n_obs - 1)))
    phase_width = 6
    obj_header = f"obj ({tf_label})"
    obj_width = max(len(obj_header), 12)
    param_widths = [max(len(name), 10) for name in param_names]

    header_parts = [
        "#".rjust(idx_width),
        "Phase".ljust(phase_width),
        obj_header.rjust(obj_width),
    ]
    for name, w in zip(param_names, param_widths):
        header_parts.append(name.rjust(w))
    header = "  ".join(header_parts)
    separator = "-" * len(header)

    print(header)
    print(separator)

    for i in range(n_obs):
        phase = "warm" if i < warmstart else "opt"
        obj_val = f"{y_sur[i]:.6g}".rjust(obj_width)
        row_parts = [
            str(i).rjust(idx_width),
            phase.ljust(phase_width),
            obj_val,
        ]
        for name, w in zip(param_names, param_widths):
            val = data.df[name].iloc[i]
            row_parts.append(f"{val:.6g}".rjust(w))
        line = "  ".join(row_parts)
        if i == best_idx:
            line += "  <-- best"
        print(line)

    print(separator)
    print(f"Best: observation #{best_idx} ({y_sur[best_idx]:.6g})")


def _show_surrogate_progress(ctx: ModelIndividualContext) -> plt.Figure:
    """Progress chart with surrogate-space objective values."""
    data = ctx.result.data
    config = ctx.config
    fitted_gp = ctx.candidate.fitted_gp

    y_sur = fitted_gp.iqr_normalizer.transform(
        fitted_gp.output_transform.forward(data.y)
    )
    tf_name = fitted_gp.output_transform.transform_type
    tf_label = f"{tf_name} + IQR" if tf_name != "none" else "IQR"

    sur_col = f"objective ({tf_label})"
    df_sur = data.df.copy()
    df_sur[sur_col] = y_sur

    fig = plot_progress(
        df_sur,
        sur_col,
        config.direction,
        remaining=config.remaining,
        warmstart=config.warmstart,
    )
    fig.suptitle(f"Progress in Surrogate Space ({tf_label})", fontsize=13, y=1.02)
    plt.show()
    return fig


def _show_surrogate_pca(ctx: ModelIndividualContext) -> plt.Figure:
    """PCA projection of existing observations in surrogate space (no candidates)."""
    data = ctx.result.data
    fitted_gp = ctx.candidate.fitted_gp

    fig = plot_pca_projection(
        X=data.X_norm,
        y=data.y,
        fitted_gp=fitted_gp,
        candidates=[],
        direction=ctx.config.direction,
    )
    fig.suptitle("PCA Projection \u2014 Observations Only", fontsize=13, y=1.02)
    plt.show()
    return fig


def _show_hyperparameters(ctx: ModelIndividualContext) -> None:
    """Plot length scales and (if enabled) warp functions."""
    fitted_gp = ctx.candidate.fitted_gp
    param_names = ctx.result.data.param_names
    plot_length_scales(fitted_gp, param_names)
    plt.show()
    fig_warp = plot_warp_functions(fitted_gp, param_names)
    if fig_warp is not None:
        plt.show()


def _show_residuals(ctx: ModelIndividualContext) -> None:
    """LOO residual diagnostics: histogram, residuals vs x, sigma vs x."""
    data = ctx.result.data
    details = analytical_loo_details(ctx.candidate.fitted_gp, data.X_norm, data.y)

    plot_residual_histogram(
        details["residuals"], details.get("residuals_raw"),
        subtitle=(
            "Symmetric around zero = good. "
            "Heavy tails or skew \u2192 poor output transform."
        ),
    )
    plt.show()

    plot_residuals_vs_x(
        data.X_norm, details["residuals"], data.param_names,
        subtitle=(
            "No trend against any input = good. "
            "Systematic patterns \u2192 missing structure."
        ),
    )
    plt.show()

    plot_sigma_vs_x(
        data.X_norm, details["sigma_loo"], data.param_names,
        subtitle=(
            "Predicted \u03c3 should be larger in sparse regions. "
            "Uniformly tiny \u03c3 \u2192 overconfidence."
        ),
    )


def _show_observed_vs_predicted(ctx: ModelIndividualContext) -> None:
    """Observed vs LOO-predicted scatter."""
    data = ctx.result.data
    details = analytical_loo_details(ctx.candidate.fitted_gp, data.X_norm, data.y)
    plot_observed_vs_predicted(
        details["y_norm"], details["mu_loo"], details["sigma_loo"],
        subtitle=(
            "Points near the 1:1 line = good. "
            "Systematic deviation \u2192 bias in the GP fit."
        ),
    )
    plt.show()


def _show_feature_importance(ctx: ModelIndividualContext) -> None:
    """Relative feature-importance bar chart."""
    plot_feature_importance(
        ctx.candidate.fitted_gp.length_scales,
        ctx.result.data.param_names,
        subtitle=(
            "Higher % = GP is more sensitive to that parameter. "
            "Near-equal bars with ARD \u2192 consider ISO kernel."
        ),
    )
    plt.show()


def _show_marginal_effects(ctx: ModelIndividualContext, n_points: int = 100) -> None:
    """Partial-dependence plots: GP mean +/- 1 sigma with data overlay."""
    fitted_gp = ctx.candidate.fitted_gp
    X = ctx.result.data.X_norm
    y_norm = fitted_gp.iqr_normalizer.transform(
        fitted_gp.output_transform.forward(ctx.result.data.y)
    )
    param_names = ctx.result.data.param_names
    n_dim = X.shape[1]
    ls = fitted_gp.length_scales
    if ls.ndim == 0 or len(ls) == 1:
        ls_per_dim = np.full(n_dim, float(ls))
    else:
        ls_per_dim = ls

    x_median = np.median(X, axis=0)

    fig, axes = plt.subplots(1, n_dim, figsize=(3.5 * n_dim, 4), squeeze=False)
    for i in range(n_dim):
        ax = axes[0, i]
        x_grid = np.linspace(0, 1, n_points)
        X_pred = np.tile(x_median, (n_points, 1))
        X_pred[:, i] = x_grid
        mu, sigma = fitted_gp.predict(X_pred, return_std=True)
        ax.plot(x_grid, mu, color="steelblue", lw=2, label="GP mean")
        ax.fill_between(
            x_grid, mu - sigma, mu + sigma,
            color="steelblue", alpha=0.15, label="\u00b11\u03c3",
        )
        ax.scatter(
            X[:, i], y_norm, s=18, color="black", alpha=0.5,
            zorder=3, label="Observations",
        )
        ax.set_xlabel(param_names[i])
        ax.set_title(f"{param_names[i]}\n\u2113 = {ls_per_dim[i]:.3g}")
        if i == 0:
            ax.set_ylabel("Surrogate space")
            ax.legend(fontsize=7, loc="best")

    fig.suptitle("Marginal Effects (Partial Dependence)", fontsize=14)
    fig.text(
        0.5, 0.01,
        "Flat curve with scattered data \u2192 large \u2113 may be valid. "
        "Structure in data but flat curve \u2192 possible misfit.",
        ha="center", fontsize=9, style="italic", color="#666666",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.show()


def _show_pairwise_projections(ctx: ModelIndividualContext, top_k: int = 4) -> None:
    """Pairwise scatter of top-K dims colored by LOO residual magnitude."""
    fitted_gp = ctx.candidate.fitted_gp
    X = ctx.result.data.X_norm
    param_names = ctx.result.data.param_names
    details = analytical_loo_details(fitted_gp, X, ctx.result.data.y)
    residuals = details["residuals"]

    ls = fitted_gp.length_scales
    if ls.ndim == 0 or len(ls) == 1:
        importances = np.ones(X.shape[1])
    else:
        importances = 1.0 / ls

    k = min(top_k, X.shape[1])
    top_dims = np.argsort(importances)[-k:][::-1]
    names = [param_names[d] for d in top_dims]

    abs_res = np.abs(residuals)
    vmax = np.percentile(abs_res, 95)

    fig, axes = plt.subplots(k, k, figsize=(3 * k + 1.5, 3 * k), squeeze=False)
    sc = None
    for row_i, d_row in enumerate(top_dims):
        for col_i, d_col in enumerate(top_dims):
            ax = axes[row_i, col_i]
            if row_i == col_i:
                ax.hist(X[:, d_row], bins=15, color="steelblue", alpha=0.7)
            else:
                sc = ax.scatter(
                    X[:, d_col], X[:, d_row],
                    c=abs_res, cmap="RdYlGn_r", vmin=0, vmax=vmax,
                    s=25, edgecolors="black", linewidth=0.3, alpha=0.8,
                )
            if row_i == k - 1:
                ax.set_xlabel(names[col_i], fontsize=8)
            else:
                ax.set_xticklabels([])
            if col_i == 0:
                ax.set_ylabel(names[row_i], fontsize=8)
            else:
                ax.set_yticklabels([])

    fig.suptitle(
        f"Pairwise LOO Residuals \u2014 Top {k} Dimensions",
        fontsize=11,
    )
    fig.text(
        0.44, 0.01,
        "Clustered red regions suggest the model is missing structure in those dimensions.",
        ha="center", fontsize=9, style="italic", color="#666666",
    )
    fig.tight_layout(rect=[0, 0.04, 0.90, 0.95])
    if sc is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.65])
        fig.colorbar(sc, cax=cbar_ax, label="|LOO residual|")
    plt.show()


def _show_gp_fit_individual(ctx: ModelIndividualContext) -> None:
    """Print GP fit diagnostics for one model candidate."""
    mc = ctx.candidate
    data = ctx.result.data

    report = gp_fit_summary(mc.fitted_gp, data.X_norm, data.y)

    print(f"\nGP Fit Diagnostics: [{ctx.choice}] {mc.output_transform} ({_kernel_label(mc)})")
    print("=" * 60)
    print(f"  LOO-MAE:          {_fmt(report.loo_mae).strip()} \u00b1 {_fmt(report.loo_std).strip()}")
    print(f"  LOO-LPD:          {_fmt(report.loo_lpd).strip()}")
    print(f"  Conditioning:     {_fmt(report.conditioning).strip()}")
    print(f"  Jitter used:      {report.jitter_used:.2e}")
    if report.conditioning_warning:
        print("  WARNING: GP conditioning is poor!")
    print(f"  Output transform: {report.output_transform}")
    print(f"  Warping:          {'enabled' if report.warping_enabled else 'disabled'}")

    param_names = data.param_names
    print("\n  Feature importances (1/length_scale):")
    for i, (ls, fi) in enumerate(
        zip(report.length_scales, report.feature_importances)
    ):
        flag = report.length_scale_flags.get(i, "")
        flag_str = f" [{flag}]" if flag else ""
        name = param_names[i] if i < len(param_names) else f"x{i}"
        print(f"    {name}: ls={ls:.4f}, importance={fi:.4f}{flag_str}")

    if report.warp_parameters is not None:
        c0 = report.warp_parameters["concentration0"]
        c1 = report.warp_parameters["concentration1"]
        print("\n  Warp parameters:")
        for i in range(len(c0)):
            name = param_names[i] if i < len(param_names) else f"x{i}"
            print(f"    {name}: c0={c0[i]:.4f}, c1={c1[i]:.4f}")


# ---------------------------------------------------------------------------
# Report registries
# ---------------------------------------------------------------------------


SUMMARY_REPORTS: list[Report] = [
    Report("residual_histogram",   "LOO residual histograms for all models (surrogate space)", _show_residual_histogram_overview),
    Report("length_scales",        "Table of all learned length scales with diagnostics", _show_length_scales_overview),
    Report("transform_comparison", "Observed-vs-predicted overlay for top models", _show_transform_comparison),
    Report("gp_fit",               "GP fit diagnostics (conditioning + jitter) for all models", _show_gp_fit_overview),
]


INDIVIDUAL_REPORTS: list[Report] = [
    Report("surrogate_observations", "Table of evaluated points in surrogate space",           _show_surrogate_observations),
    Report("surrogate_progress",     "Progress chart in surrogate space",                      _show_surrogate_progress),
    Report("surrogate_pca",          "PCA projection of observations in surrogate space",      _show_surrogate_pca),
    Report("hyperparameters",        "Length scales + warp function plots",                    _show_hyperparameters),
    Report("residuals",              "LOO residual diagnostics",                               _show_residuals),
    Report("observed_vs_predicted",  "Observed vs LOO-predicted scatter",                      _show_observed_vs_predicted),
    Report("feature_importance",     "Relative feature importance bar chart",                  _show_feature_importance),
    Report("marginal_effects",       "GP mean as each input varies (partial dependence)",      _show_marginal_effects),
    Report("pairwise_projections",   "Pairwise projections of GP predictions (top-K dims)",    _show_pairwise_projections),
    Report("gp_fit",                 "GP fit diagnostics (LOO, conditioning, jitter, flags)",  _show_gp_fit_individual),
]
