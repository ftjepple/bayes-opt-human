"""Data-phase reports."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayesopt_human.config import OptimizationConfig
from bayesopt_human.data.normalizer import BoundsNormalizer
from bayesopt_human.diagnostics.coverage import coverage_metrics
from bayesopt_human.diagnostics.modality import detect_local_optima
from bayesopt_human.diagnostics.svm import svm_sense_check
from bayesopt_human.reporting.base import Report
from bayesopt_human.utils import stagnation_length
from bayesopt_human.visualization.progress import plot_progress
from bayesopt_human.visualization.projections import plot_pca_observations

if TYPE_CHECKING:
    from bayesopt_human.optimizer.step import DataEntry, DataResult


# ---------------------------------------------------------------------------
# Contexts
# ---------------------------------------------------------------------------


@dataclass
class DataSummaryContext:
    """Context for summary (multi-function) data reports."""

    result: "DataResult"
    configs: list[OptimizationConfig] = field(default_factory=list)


@dataclass
class DataIndividualContext:
    """Context for individual (per-function) data reports."""

    result: "DataResult"
    entry: "DataEntry"
    config: OptimizationConfig


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _config_for_entry(
    entry: "DataEntry", configs: list[OptimizationConfig]
) -> OptimizationConfig:
    entry_name = Path(entry.filepath).stem
    config = next((c for c in configs if c.name == entry_name), None)
    if config is None:
        raise ValueError(f"No config found for entry {entry_name}")
    return config


def _normalize_entry(
    entry: "DataEntry", config: OptimizationConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize an entry's inputs using its config bounds.

    Used by data-stage reports (coverage, modality, svm) that need normalized
    coordinates without requiring ``choose_data`` to have been called.
    """
    p_cols = entry.param_names
    df = entry.df
    normalizer = BoundsNormalizer({name: config.bounds[name] for name in p_cols})
    X_norm = normalizer.normalize(df[p_cols].values)
    y = df[config.objective_column].values
    return X_norm, y


# ---------------------------------------------------------------------------
# Summary reports
# ---------------------------------------------------------------------------


def _show_summary_table(ctx: DataSummaryContext) -> None:
    rows = []
    for entry in ctx.result.entries:
        entry_name = Path(entry.filepath).stem
        y = entry.df[entry.objective_column].values
        rows.append({
            "Function": entry_name,
            "#Dims": len(entry.param_names),
            "#Obs": len(y),
            "Min": float(pd.Series(y).min()),
            "Median": float(pd.Series(y).median()),
            "Max": float(pd.Series(y).max()),
        })
    df = pd.DataFrame(rows)
    print("\nSummary of loaded functions:")
    print(df.to_string(index=False))


def _show_progress_overview(ctx: DataSummaryContext) -> plt.Figure:
    result = ctx.result
    n = len(result.entries)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()
    for idx, entry in enumerate(result.entries):
        ax = axes[idx]
        entry_name = Path(entry.filepath).stem
        y = entry.df[entry.objective_column].values
        x = np.arange(1, len(y) + 1)
        config = next((c for c in ctx.configs if c.name == entry_name), None)
        direction = config.direction if config is not None else "maximize"
        ws = config.warmstart if config is not None else 0
        if ws > 0 and ws < len(y):
            ax.scatter(x[:ws], y[:ws], c="#999999", s=40, zorder=3, label="Warmstart")
            ax.scatter(x[ws:], y[ws:], c="steelblue", s=40, zorder=3, label="Observations")
            ax.axvline(ws + 0.5, color="#999999", linestyle="--", linewidth=1, alpha=0.5)
        else:
            ax.scatter(x, y, c="steelblue", s=40, zorder=3, label="Observations")
        if direction == "minimize":
            best_so_far = np.minimum.accumulate(y)
        else:
            best_so_far = np.maximum.accumulate(y)
        ax.step(x, best_so_far, where="post", color="darkgreen", linewidth=2, label="Best so far")
        if idx == 0:
            ax.legend(loc="best")
        ax.set_title(f"[{idx}] {entry_name}")
        if idx % cols == 0:
            ax.set_ylabel("Objective Value")
        else:
            ax.set_ylabel("")
        if idx // cols == rows - 1:
            ax.set_xlabel("Observation index")
        else:
            ax.set_xlabel("")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Progress", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    print("Displayed progress charts for all functions.")
    return fig


def _show_histogram_overview(ctx: DataSummaryContext) -> plt.Figure:
    result = ctx.result
    n = len(result.entries)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()
    for idx, entry in enumerate(result.entries):
        ax = axes[idx]
        entry_name = Path(entry.filepath).stem
        y = entry.df[entry.objective_column].values
        ax.hist(y, bins=20, color="skyblue", edgecolor="black")
        ax.set_title(f"[{idx}] {entry_name}")
        if idx % cols == 0:
            ax.set_ylabel("Count")
        else:
            ax.set_ylabel("")
        if idx // cols == rows - 1:
            ax.set_xlabel(entry.objective_column)
        else:
            ax.set_xlabel("")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Histogram", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    print("Displayed histograms for all functions.")
    return fig


def _show_correlation_overview(ctx: DataSummaryContext) -> plt.Figure:
    """Correlation heatmap (params + objective) for each loaded function."""
    result = ctx.result
    n = len(result.entries)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes_flat = axes.flatten()
    for idx, entry in enumerate(result.entries):
        ax = axes_flat[idx]
        entry_name = Path(entry.filepath).stem
        cols_to_use = entry.param_names + [entry.objective_column]
        corr = entry.df[cols_to_use].corr()
        ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols_to_use)))
        ax.set_yticks(range(len(cols_to_use)))
        ax.set_xticklabels(cols_to_use, fontsize=7, rotation=45, ha="right")
        ax.set_yticklabels(cols_to_use, fontsize=7)
        for i in range(len(cols_to_use)):
            for j in range(len(cols_to_use)):
                val = corr.values[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)
        ax.set_title(f"[{idx}] {entry_name}", fontsize=10)
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.suptitle("Correlation (Parameters + Objective)", fontsize=14)
    fig.text(
        0.5, 0.01,
        "Strong param-objective correlations suggest important dimensions. "
        "High inter-param correlation may cause fitting issues.",
        ha="center", fontsize=9, style="italic", color="#666666",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.show()
    return fig


def _show_duplicates_overview(ctx: DataSummaryContext) -> None:
    """Detect near-duplicate observations across all loaded functions."""
    any_found = False
    for idx, entry in enumerate(ctx.result.entries):
        entry_name = Path(entry.filepath).stem
        X = entry.df[entry.param_names].values
        n = len(X)
        if n < 2:
            continue
        ranges = X.max(axis=0) - X.min(axis=0)
        ranges[ranges == 0] = 1.0
        X_scaled = X / ranges
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((X_scaled[i] - X_scaled[j]) ** 2))
                if dist < 0.02:
                    pairs.append((i, j, dist * 100))
        if pairs:
            if not any_found:
                print("\nNear-Duplicate Detection (within 2% of parameter range):")
                any_found = True
            print(f"\n  [{idx}] {entry_name}:")
            for i, j, pct in sorted(pairs, key=lambda p: p[2]):
                print(f"    Rows {i} and {j}: {pct:.2f}% apart")
    if not any_found:
        print("\nNo near-duplicate observations detected (threshold: 2% of parameter range).")


def _show_coverage_metrics_overview(ctx: DataSummaryContext) -> None:
    rows = []
    for entry in ctx.result.entries:
        config = _config_for_entry(entry, ctx.configs)
        X_norm, _ = _normalize_entry(entry, config)
        report = coverage_metrics(X_norm, k=config.knn_k)
        X_post = X_norm[config.warmstart:]
        if len(X_post) >= 2:
            report_post = coverage_metrics(X_post, k=config.knn_k)
            ratio_post = f"{report_post.ratio_to_optimal_flat:.2f}"
        else:
            ratio_post = "N/A"
        entry_name = Path(entry.filepath).stem
        rows.append({
            "Function": entry_name,
            "Avg KNN": f"{report.avg_knn_distance_flat:.4f}",
            "Min KNN": f"{report.min_knn_distance_flat:.4f}",
            "Optimal": f"{report.optimal_baseline_flat:.4f}",
            "Ratio": f"{report.ratio_to_optimal_flat:.2f}",
            "Post-WS Ratio": ratio_post,
            "K": report.k,
        })
    df = pd.DataFrame(rows)
    print("\nCoverage Metrics (flat space):")
    print(df.to_string(index=False))
    print("  Ratio = Avg KNN / Optimal. Values near 1.0 = well-spread design.")
    print("  Post-WS Ratio = same metric on post-warmstart points only.")
    print("  Lower than Ratio suggests the optimizer concentrated (exploitation).")


def _show_modality_overview(ctx: DataSummaryContext) -> None:
    rows = []
    for entry in ctx.result.entries:
        config = _config_for_entry(entry, ctx.configs)
        X_norm, y = _normalize_entry(entry, config)
        report = detect_local_optima(X_norm, y, config.direction, k=config.modality_k)
        entry_name = Path(entry.filepath).stem
        warn = "\u26a0" if report.reliability_warning else ""
        rows.append({
            "Function": entry_name,
            "Optima": report.n_local_optima_flat,
            "Basins": report.n_basins_flat,
            "Warn": warn,
        })
    df = pd.DataFrame(rows)
    print("\nModality Diagnostics (flat space):")
    print(df.to_string(index=False))
    print("  Multiple basins suggest multimodal landscape. \u26a0 = low data-to-dim ratio.")


# ---------------------------------------------------------------------------
# Individual reports
# ---------------------------------------------------------------------------


def _show_observations(ctx: DataIndividualContext) -> None:
    """Print a table of all evaluated coordinates with function values."""
    entry = ctx.entry
    config = ctx.config
    df = entry.df
    y = df[entry.objective_column].values
    param_names = entry.param_names
    warmstart = config.warmstart
    direction = config.direction

    best_idx = int(np.argmin(y)) if direction == "minimize" else int(np.argmax(y))

    entry_name = Path(entry.filepath).stem
    print(f"\nObservations for function [{entry_name}]: {Path(entry.filepath).name}")
    print(
        f"  {entry.n_obs} points, {entry.n_dim} dimensions, "
        f"warmstart={warmstart}, direction={direction}\n"
    )

    idx_width = max(3, len(str(entry.n_obs - 1)))
    phase_width = 6
    obj_width = max(len(entry.objective_column), 12)
    param_widths = [max(len(name), 10) for name in param_names]

    header_parts = [
        "#".rjust(idx_width),
        "Phase".ljust(phase_width),
        entry.objective_column.rjust(obj_width),
    ]
    for name, w in zip(param_names, param_widths):
        header_parts.append(name.rjust(w))
    header = "  ".join(header_parts)
    separator = "-" * len(header)

    print(header)
    print(separator)

    for i in range(len(df)):
        phase = "warm" if i < warmstart else "opt"
        obj_val = f"{y[i]:.6g}".rjust(obj_width)
        row_parts = [
            str(i).rjust(idx_width),
            phase.ljust(phase_width),
            obj_val,
        ]
        for name, w in zip(param_names, param_widths):
            val = df[name].iloc[i]
            row_parts.append(f"{val:.6g}".rjust(w))
        line = "  ".join(row_parts)
        if i == best_idx:
            line += "  <-- best"
        print(line)

    best_val = y[best_idx]
    print(separator)
    print(f"Best: observation #{best_idx} ({best_val:.6g})")


def _show_progress(ctx: DataIndividualContext) -> plt.Figure:
    fig = plot_progress(
        ctx.entry.df,
        ctx.entry.objective_column,
        ctx.config.direction,
        remaining=ctx.config.remaining,
        warmstart=ctx.config.warmstart,
    )
    plt.show()
    return fig


def _show_statistics(ctx: DataIndividualContext) -> None:
    entry = ctx.entry
    config = ctx.config
    print(f"\nStatistics for function [{Path(entry.filepath).stem}]: {Path(entry.filepath).name}")
    print(entry.df.describe().to_string())
    stag = stagnation_length(
        entry.df[entry.objective_column].values,
        config.direction,
        warmstart=config.warmstart,
    )
    y = entry.df[entry.objective_column].values
    best = float(np.min(y)) if config.direction == "minimize" else float(np.max(y))
    print(f"\nBest-so-far: {best:.4f}")
    print(f"Stagnation length: {stag}")


def _show_histogram(ctx: DataIndividualContext) -> plt.Figure:
    entry = ctx.entry
    y = entry.df[entry.objective_column].values
    fig, ax = plt.subplots()
    ax.hist(y, bins=20, color="skyblue", edgecolor="black")
    ax.set_title(f"Histogram of {entry.objective_column} values")
    ax.set_xlabel(entry.objective_column)
    ax.set_ylabel("Count")
    plt.show()
    return fig


def _show_coverage(ctx: DataIndividualContext) -> plt.Figure:
    """Strip plot of parameter coverage per dimension."""
    entry = ctx.entry
    entry_name = Path(entry.filepath).stem
    X = entry.df[entry.param_names].values
    n_dim = len(entry.param_names)

    fig, axes = plt.subplots(1, n_dim, figsize=(3 * n_dim, 4), squeeze=False)
    for i in range(n_dim):
        ax = axes[0, i]
        vals = X[:, i]
        lo, hi = vals.min(), vals.max()
        rng = hi - lo if hi > lo else 1.0
        jitter = np.random.default_rng(42).normal(0, 0.08, size=len(vals))
        ax.scatter(jitter, vals, s=25, c="steelblue", edgecolors="black",
                   linewidth=0.3, alpha=0.7, zorder=2)
        ax.axhline(lo, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axhline(hi, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        sorted_vals = np.sort(vals)
        for k in range(len(sorted_vals) - 1):
            gap = sorted_vals[k + 1] - sorted_vals[k]
            if gap > 0.25 * rng:
                mid = (sorted_vals[k] + sorted_vals[k + 1]) / 2
                ax.axhspan(sorted_vals[k], sorted_vals[k + 1],
                           color="red", alpha=0.08, zorder=0)
                ax.text(0.5, mid, f"gap {gap / rng:.0%}",
                        ha="center", va="center", fontsize=7,
                        color="red", alpha=0.7,
                        transform=ax.get_yaxis_transform())
        ax.set_title(entry.param_names[i], fontsize=10)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        if i > 0:
            ax.set_yticklabels([])

    fig.suptitle(f"Parameter Coverage: {entry_name}", fontsize=12)
    fig.text(
        0.5, 0.01,
        "Red bands highlight gaps > 25% of the parameter range. "
        "Consider sampling in gap regions.",
        ha="center", fontsize=9, style="italic", color="#666666",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.show()
    return fig


def _show_pca(ctx: DataIndividualContext) -> plt.Figure:
    """Unweighted PCA projection of observations (raw space, single panel).

    Local optima are computed lazily via ``detect_local_optima`` and overlaid
    as red dashed circles when present. The modality reliability flag is
    ignored here: if the heuristic picked anything, it is shown.
    """
    X_norm, y = _normalize_entry(ctx.entry, ctx.config)
    modality = detect_local_optima(
        X_norm, y, ctx.config.direction, k=ctx.config.modality_k
    )
    local_optima = (
        modality.local_optima_coords_flat
        if len(modality.local_optima_coords_flat) > 0
        else None
    )
    fig = plot_pca_observations(
        X_norm, y,
        direction=ctx.config.direction,
        local_optima=local_optima,
    )
    plt.show()
    return fig


def _show_coverage_metrics_individual(ctx: DataIndividualContext) -> None:
    X_norm, _ = _normalize_entry(ctx.entry, ctx.config)
    report = coverage_metrics(X_norm, k=ctx.config.knn_k)
    entry_name = Path(ctx.entry.filepath).stem

    print(f"\nCoverage Metrics: {entry_name}")
    print("-" * 50)
    print(f"  Avg KNN distance (flat):     {report.avg_knn_distance_flat:.4f}")
    print(f"  Min KNN distance (flat):     {report.min_knn_distance_flat:.4f}")
    print(f"  Optimal baseline (flat):     {report.optimal_baseline_flat:.4f}")
    print(f"  Ratio to optimal (flat):     {report.ratio_to_optimal_flat:.2f}")
    print(f"  K neighbors:                 {report.k}")
    if report.ratio_to_optimal_flat > 3.0:
        print("  NOTE: Ratio > 3 suggests significant clustering or gaps in the design.")

    X_post = X_norm[ctx.config.warmstart:]
    print(f"\n  Post-warmstart points:       {len(X_post)} of {len(X_norm)}")
    if len(X_post) >= 2:
        report_post = coverage_metrics(X_post, k=ctx.config.knn_k)
        print(f"  Post-WS ratio to optimal:    {report_post.ratio_to_optimal_flat:.2f}")
        print("  Lower than full ratio = optimizer concentrated (exploitation).")
        print("  Near full ratio = optimizer kept exploring.")
    else:
        print("  Post-WS ratio:               N/A (need >= 2 post-warmstart points)")


def _show_modality_individual(ctx: DataIndividualContext) -> None:
    X_norm, y = _normalize_entry(ctx.entry, ctx.config)
    report = detect_local_optima(X_norm, y, ctx.config.direction, k=ctx.config.modality_k)
    entry_name = Path(ctx.entry.filepath).stem

    print(f"\nModality Diagnostics: {entry_name}")
    print("-" * 50)
    print(f"  Local optima (flat):    {report.n_local_optima_flat}")
    print(f"  Basins (flat):          {report.n_basins_flat}")
    if report.reliability_warning:
        print(f"  WARNING: {report.reliability_message}")


def _show_svm_individual(ctx: DataIndividualContext) -> None:
    X_norm, y = _normalize_entry(ctx.entry, ctx.config)
    report = svm_sense_check(X_norm, y, ctx.config.direction)
    entry_name = Path(ctx.entry.filepath).stem

    print(f"\nSVM Sense-Check: {entry_name}")
    print("-" * 50)
    print(
        f"  LOO Balanced Accuracy: {report.loo_balanced_accuracy:.2f} "
        f"(95% CI: [{report.confidence_interval[0]:.2f}, "
        f"{report.confidence_interval[1]:.2f}])"
    )
    if not report.is_reliable:
        print(f"  WARNING: {report.reliability_message}")


# ---------------------------------------------------------------------------
# Report registries
# ---------------------------------------------------------------------------


SUMMARY_REPORTS: list[Report] = [
    Report("statistics",       "Table of summary statistics for each function",     _show_summary_table),
    Report("progress",         "Compare optimization progress across functions",    _show_progress_overview),
    Report("histogram",        "Histograms for all functions",                      _show_histogram_overview),
    Report("correlation",      "Correlation heatmap (params + objective)",          _show_correlation_overview),
    Report("duplicates",       "Near-duplicate observation detection",              _show_duplicates_overview),
    Report("coverage_metrics", "KNN coverage metrics across functions",             _show_coverage_metrics_overview),
    Report("modality",         "Modality diagnostics across functions",             _show_modality_overview),
]


INDIVIDUAL_REPORTS: list[Report] = [
    Report("observations",      "Table of all evaluated points with function values", _show_observations),
    Report("progress",          "Optimization progress chart",                        _show_progress),
    Report("statistics",        "Summary statistics",                                 _show_statistics),
    Report("histogram",         "Histogram of objective values",                      _show_histogram),
    Report("coverage",          "Parameter coverage (strip plots per dimension)",     _show_coverage),
    Report("pca",               "Unweighted PCA projection (raw space)",              _show_pca),
    Report("coverage_metrics",  "KNN coverage metrics (distances + ratio to optimal)", _show_coverage_metrics_individual),
    Report("modality",          "Local optima and basin detection",                   _show_modality_individual),
    Report("svm",               "SVM sense-check (balanced accuracy + CI)",           _show_svm_individual),
]
