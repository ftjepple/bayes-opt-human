"""SVM sense-check with LOO balanced accuracy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC


@dataclass
class SVMReport:
    """Container for SVM sense-check results."""

    svm: SVC
    loo_balanced_accuracy: float
    confidence_interval: tuple[float, float]
    is_reliable: bool
    reliability_message: str
    scaler_mean: np.ndarray
    scaler_std: np.ndarray


def svm_sense_check(
    X: np.ndarray,
    y: np.ndarray,
    direction: str,
) -> SVMReport:
    """Fit SVM and evaluate as sense-check on GP predictions.

    Binary target: 1 if observation in top quartile, 0 otherwise.
    "Top" is defined by optimization direction (best 25%).

    Uses LOO cross-validation (tractable since SVM is fast and N is small).
    Reports balanced accuracy with binomial confidence interval.

    Args:
        X: Normalized inputs, shape (N, D).
        y: Objective values, shape (N,).
        direction: 'minimize' or 'maximize'.

    Returns:
        SVMReport with fitted SVM, LOO balanced accuracy,
        confidence interval, and reliability flag.
    """
    n = len(X)

    # Define top quartile
    if direction == "minimize":
        threshold = np.percentile(y, 25)
        labels = (y <= threshold).astype(int)
    else:
        threshold = np.percentile(y, 75)
        labels = (y >= threshold).astype(int)

    # Standardize inputs for SVM
    scaler_mean = X.mean(axis=0)
    scaler_std = X.std(axis=0)
    scaler_std[scaler_std == 0] = 1.0
    X_scaled = (X - scaler_mean) / scaler_std

    # LOO cross-validation
    loo_preds = np.zeros(n, dtype=int)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        svm = SVC(kernel="rbf", class_weight="balanced")
        svm.fit(X_scaled[mask], labels[mask])
        loo_preds[i] = svm.predict(X_scaled[i:i+1])[0]

    loo_bacc = balanced_accuracy_score(labels, loo_preds)

    # Binomial confidence interval for balanced accuracy
    # Approximate using Wilson interval on the balanced accuracy
    n_eff = n  # Effective sample size
    z = 1.96  # 95% CI
    p_hat = loo_bacc
    denom = 1 + z**2 / n_eff
    center = (p_hat + z**2 / (2 * n_eff)) / denom
    margin = z * np.sqrt(
        (p_hat * (1 - p_hat) + z**2 / (4 * n_eff)) / n_eff
    ) / denom
    ci = (max(0.0, center - margin), min(1.0, center + margin))

    # Reliability: is accuracy significantly better than chance (50%)?
    is_reliable = ci[0] > 0.5
    reliability_msg = ""
    if not is_reliable:
        reliability_msg = (
            f"SVM balanced accuracy ({loo_bacc:.2f}, 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]) "
            "is not significantly better than chance. "
            "The SVM sense-check should be disregarded for this dataset."
        )

    # Fit final SVM on all data. ``probability=True`` enables Platt scaling,
    # which does an internal 5-fold CV whose fold shuffling draws from the
    # global RNG unless ``random_state`` is fixed. Hard-code it so the SVM
    # sense-check is reproducible regardless of caller seeding.
    final_svm = SVC(
        kernel="rbf", class_weight="balanced", probability=True, random_state=0,
    )
    final_svm.fit(X_scaled, labels)

    return SVMReport(
        svm=final_svm,
        loo_balanced_accuracy=loo_bacc,
        confidence_interval=ci,
        is_reliable=is_reliable,
        reliability_message=reliability_msg,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )


def svm_predict_candidate(
    svm_report: SVMReport,
    x_candidate: np.ndarray,
) -> bool:
    """Predict whether a candidate point falls in the top quartile.

    Args:
        svm_report: Fitted SVM report.
        x_candidate: Candidate coordinates in normalized space, shape (D,).

    Returns:
        True if SVM predicts the candidate is in the top quartile.
    """
    x_scaled = (x_candidate.reshape(1, -1) - svm_report.scaler_mean) / svm_report.scaler_std
    pred = svm_report.svm.predict(x_scaled)
    return bool(pred[0] == 1)
