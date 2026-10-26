"""Reusable MLflow utilities for experiment tracking."""

from collections.abc import Mapping
from math import isfinite
from numbers import Real

import mlflow


def setup_mlflow(
    experiment_name: str,
    tracking_uri: str | None = None,
) -> None:
    """Configure MLflow Tracking for an experiment.

    Args:
        experiment_name: Name of the experiment to use.
        tracking_uri: Optional MLflow Tracking Server URI.
            If omitted, MLflow uses its default tracking location.

    Raises:
        ValueError: If the experiment name is empty.
    """
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("Experiment name must be a non-empty string.")

    if tracking_uri is not None:
        if not isinstance(tracking_uri, str) or not tracking_uri.strip():
            raise ValueError("Tracking URI must be a non-empty string.")

        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)


def log_parameters(parameters: Mapping[str, object]) -> None:
    """Log model parameters to the active MLflow run.

    Args:
        parameters: Parameter names and their values.

    Raises:
        TypeError: If parameters is not a mapping.
        ValueError: If a parameter name is invalid.
    """
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping.")

    for name in parameters:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Parameter names must be non-empty strings.")

    mlflow.log_params(dict(parameters))


def log_metrics(metrics: Mapping[str, float]) -> None:
    """Log finite numeric metrics to the active MLflow run.

    Args:
        metrics: Metric names and their numeric values.

    Raises:
        TypeError: If metrics is not a mapping or contains non-numeric values.
        ValueError: If a metric name is invalid or a value is not finite.
    """
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping.")

    validated_metrics: dict[str, float] = {}

    for name, value in metrics.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Metric names must be non-empty strings.")

        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"Metric '{name}' must have a numeric value."
            )

        numeric_value = float(value)

        if not isfinite(numeric_value):
            raise ValueError(
                f"Metric '{name}' must have a finite value."
            )

        validated_metrics[name] = numeric_value

    if validated_metrics:
        mlflow.log_metrics(validated_metrics)