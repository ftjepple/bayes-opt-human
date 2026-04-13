"""Tests for SVM sense-check."""

import numpy as np
import pytest

from bayesopt_human.diagnostics.svm import svm_predict_candidate, svm_sense_check


class TestSVMSenseCheck:
    def test_basic_svm_check(self):
        np.random.seed(42)
        X = np.random.rand(20, 2) * 0.99
        y = np.sum(X**2, axis=1)
        report = svm_sense_check(X, y, "minimize")
        assert 0.0 <= report.loo_balanced_accuracy <= 1.0
        assert len(report.confidence_interval) == 2
        assert report.confidence_interval[0] <= report.confidence_interval[1]

    def test_svm_predict(self):
        np.random.seed(42)
        X = np.random.rand(20, 2) * 0.99
        y = np.sum(X**2, axis=1)
        report = svm_sense_check(X, y, "minimize")
        pred = svm_predict_candidate(report, np.array([0.1, 0.1]))
        assert isinstance(pred, bool)

    def test_maximize_direction(self):
        np.random.seed(42)
        X = np.random.rand(20, 2) * 0.99
        y = -np.sum(X**2, axis=1)
        report = svm_sense_check(X, y, "maximize")
        assert 0.0 <= report.loo_balanced_accuracy <= 1.0
