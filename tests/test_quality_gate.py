"""Unit tests for the shared quality gate utilities."""

import pytest

from common.quality_gate import evaluate_quality_gate


def test_all_metrics_pass():
    """The candidate should pass when every metric meets its threshold."""
    metrics = {
        "precision": 0.90,
        "recall": 0.85,
        "f1_score": 0.87,
    }
    thresholds = {
        "precision": 0.85,
        "recall": 0.80,
        "f1_score": 0.82,
    }

    result = evaluate_quality_gate(metrics, thresholds)

    assert result["passed"] is True
    assert result["failed_metrics"] == []
    assert result["missing_metrics"] == []


def test_one_metric_below_threshold():
    """The candidate should fail if any required metric is below threshold."""
    metrics = {
        "precision": 0.90,
        "recall": 0.75,
        "f1_score": 0.87,
    }
    thresholds = {
        "precision": 0.85,
        "recall": 0.80,
        "f1_score": 0.82,
    }

    result = evaluate_quality_gate(metrics, thresholds)

    assert result["passed"] is False
    assert result["failed_metrics"] == ["recall"]


def test_metric_equal_to_threshold_passes():
    """A metric exactly equal to its minimum threshold should pass."""
    result = evaluate_quality_gate(
        metrics={"f1_score": 0.80},
        thresholds={"f1_score": 0.80},
    )

    assert result["passed"] is True


def test_missing_required_metric_fails():
    """A missing required metric should cause the gate to fail."""
    result = evaluate_quality_gate(
        metrics={"precision": 0.90},
        thresholds={
            "precision": 0.85,
            "recall": 0.80,
        },
    )

    assert result["passed"] is False
    assert result["failed_metrics"] == ["recall"]
    assert result["missing_metrics"] == ["recall"]
    assert result["checks"]["recall"]["value"] is None


def test_empty_thresholds_are_rejected():
    """An empty threshold configuration should raise ValueError."""
    with pytest.raises(
        ValueError,
        match="At least one quality gate threshold",
    ):
        evaluate_quality_gate(
            metrics={"f1_score": 0.90},
            thresholds={},
        )


def test_non_numeric_metric_is_rejected():
    """A non-numeric metric should raise TypeError."""
    with pytest.raises(
        TypeError,
        match="must have a numeric value",
    ):
        evaluate_quality_gate(
            metrics={"f1_score": "high"},
            thresholds={"f1_score": 0.80},
        )


def test_non_finite_metric_is_rejected():
    """An infinite metric should raise ValueError."""
    with pytest.raises(
        ValueError,
        match="must have a finite value",
    ):
        evaluate_quality_gate(
            metrics={"f1_score": float("inf")},
            thresholds={"f1_score": 0.80},
        )


def test_non_numeric_threshold_is_rejected():
    """A non-numeric threshold should raise TypeError."""
    with pytest.raises(
        TypeError,
        match="must be a numeric value",
    ):
        evaluate_quality_gate(
            metrics={"f1_score": 0.90},
            thresholds={"f1_score": "high"},
        )