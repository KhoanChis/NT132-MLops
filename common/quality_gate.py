"""Reusable quality gate utilities for evaluating ML model candidates."""

from collections.abc import Mapping
from math import isfinite
from numbers import Real


def evaluate_quality_gate(
    metrics: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    """Check whether a model satisfies all minimum metric thresholds.

    Every metric specified in thresholds must exist in metrics and must
    be greater than or equal to its corresponding minimum threshold.

    Args:
        metrics: Metrics produced during model evaluation.
        thresholds: Minimum acceptable value for each required metric.

    Returns:
        A dictionary containing the overall result, individual checks,
        failed metrics, and missing metrics.

    Raises:
        TypeError: If the inputs are not mappings or contain non-numeric values.
        ValueError: If thresholds are empty or contain non-finite values.
    """
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping of metric names to values.")

    if not isinstance(thresholds, Mapping):
        raise TypeError(
            "thresholds must be a mapping of metric names to minimum values."
        )

    if not thresholds:
        raise ValueError("At least one quality gate threshold is required.")

    for metric_name, threshold in thresholds.items():
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ValueError("Metric names must be non-empty strings.")

        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise TypeError(
                f"Threshold for '{metric_name}' must be a numeric value."
            )

        if not isfinite(float(threshold)):
            raise ValueError(
                f"Threshold for '{metric_name}' must be finite."
            )

    checks: dict[str, dict[str, object]] = {}
    failed_metrics: list[str] = []
    missing_metrics: list[str] = []

    for metric_name, threshold in thresholds.items():
        if metric_name not in metrics:
            checks[metric_name] = {
                "value": None,
                "threshold": float(threshold),
                "passed": False,
                "reason": "Metric is missing.",
            }
            failed_metrics.append(metric_name)
            missing_metrics.append(metric_name)
            continue

        value = metrics[metric_name]

        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"Metric '{metric_name}' must have a numeric value."
            )

        if not isfinite(float(value)):
            raise ValueError(
                f"Metric '{metric_name}' must have a finite value."
            )

        passed = float(value) >= float(threshold)

        checks[metric_name] = {
            "value": float(value),
            "threshold": float(threshold),
            "passed": passed,
            "reason": (
                "Threshold satisfied."
                if passed
                else "Metric is below the minimum threshold."
            ),
        }

        if not passed:
            failed_metrics.append(metric_name)

    return {
        "passed": not failed_metrics,
        "checks": checks,
        "failed_metrics": failed_metrics,
        "missing_metrics": missing_metrics,
    }