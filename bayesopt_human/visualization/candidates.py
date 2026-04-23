"""Candidate comparison table and overlay on projections."""

from __future__ import annotations

import math

import pandas as pd

from bayesopt_human.acquisition.candidates import Candidate
from bayesopt_human.recommendations.schema import Recommendation


def display_candidate_table(
    candidates: list[Candidate],
    recommendations: list[Recommendation],
) -> pd.DataFrame:
    """Format candidates and recommendations as a styled DataFrame.

    Columns: Source, Category, Posterior Mean, Posterior Std, EI,
    Coverage Gain (flat), Coverage Gain (surrogate), SVM Prediction,
    Recommended (with priority).

    Args:
        candidates: List of candidate points.
        recommendations: Ranked recommendations.

    Returns:
        Formatted DataFrame for display.
    """
    # Build recommendation lookup
    rec_map: dict[str, tuple[int, bool]] = {}
    for rec in recommendations:
        rec_map[rec.source] = (rec.priority, rec.is_arm_winner)

    rows = []
    for cand in candidates:
        entry = rec_map.get(cand.source)
        if entry is not None:
            priority, is_arm_winner = entry
            rec_label = f"#{priority}"
        else:
            priority, is_arm_winner = None, False
            rec_label = ""

        category_label = cand.category
        if is_arm_winner:
            category_label += " ★"

        rows.append({
            "Rank": rec_label,
            "Source": cand.source,
            "Category": category_label,
            "Sur. Mean": f"{cand.posterior_mean:+.4f}",
            "Sur. Std": f"{cand.posterior_std:+.4f}",
            "EI": f"{math.exp(cand.expected_improvement):+.4f}",
            "Cov. Gain (flat)": f"{cand.coverage_gain_flat:.4g}",
            "Cov. Gain (surr.)": f"{cand.coverage_gain_surrogate:.4g}",
            "SVM Pred.": (
                "Top Q" if cand.svm_prediction is True
                else "No" if cand.svm_prediction is False
                else "N/A"
            ),
        })

    df = pd.DataFrame(rows)

    # Sort by recommendation rank
    def _sort_key(x):
        if x == "":
            return 999
        return int(x.replace("#", ""))

    df["_sort"] = df["Rank"].apply(_sort_key)
    df = df.sort_values("_sort").drop(columns=["_sort"])

    return df
