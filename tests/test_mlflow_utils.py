"""Unit tests for the shared MLflow utilities."""

import mlflow
import pytest

from common.mlflow_utils import (
    log_metrics,
    log_parameters,
    setup_mlflow,
)


@pytest.fixture
def local_mlflow(tmp_path):
    """Configure an isolated SQLite-backed MLflow tracking environment."""
    tracking_db = tmp_path / "mlflow.db"
    tracking_uri = f"sqlite:///{tracking_db.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)

    yield tracking_uri

    mlflow.end_run()
    mlflow.set_tracking_uri("sqlite:///mlflow.db")


def test_setup_mlflow_creates_experiment(local_mlflow):
    """Setup should create or select the requested experiment."""
    setup_mlflow("fraud-detection-test")

    experiment = mlflow.get_experiment_by_name(
        "fraud-detection-test"
    )

    assert experiment is not None
    assert experiment.name == "fraud-detection-test"


def test_setup_mlflow_rejects_empty_experiment_name():
    """An empty experiment name should be rejected."""
    with pytest.raises(
        ValueError,
        match="Experiment name must be a non-empty string",
    ):
        setup_mlflow("")


def test_log_parameters_records_parameters(local_mlflow):
    """Parameters should be recorded in the active MLflow run."""
    setup_mlflow("parameter-test")

    with mlflow.start_run():
        log_parameters(
            {
                "model_type": "RandomForest",
                "n_estimators": 100,
            }
        )

        run_id = mlflow.active_run().info.run_id

    run = mlflow.get_run(run_id)

    assert run.data.params["model_type"] == "RandomForest"
    assert run.data.params["n_estimators"] == "100"


def test_log_metrics_records_metrics(local_mlflow):
    """Metrics should be recorded in the active MLflow run."""
    setup_mlflow("metric-test")

    with mlflow.start_run():
        log_metrics(
            {
                "precision": 0.91,
                "recall": 0.87,
                "f1_score": 0.89,
            }
        )

        run_id = mlflow.active_run().info.run_id

    run = mlflow.get_run(run_id)

    assert run.data.metrics["precision"] == pytest.approx(0.91)
    assert run.data.metrics["recall"] == pytest.approx(0.87)
    assert run.data.metrics["f1_score"] == pytest.approx(0.89)


def test_log_metrics_accepts_empty_mapping(local_mlflow):
    """An empty metrics mapping should not create a logging error."""
    setup_mlflow("empty-metrics-test")

    with mlflow.start_run():
        log_metrics({})


def test_log_parameters_rejects_invalid_input(local_mlflow):
    """Parameters must be supplied as a mapping."""
    setup_mlflow("invalid-parameters-test")

    with mlflow.start_run():
        with pytest.raises(
            TypeError,
            match="parameters must be a mapping",
        ):
            log_parameters(["invalid"])


def test_log_metrics_rejects_non_numeric_value(local_mlflow):
    """Metric values must be numeric."""
    setup_mlflow("non-numeric-metrics-test")

    with mlflow.start_run():
        with pytest.raises(
            TypeError,
            match="must have a numeric value",
        ):
            log_metrics({"f1_score": "high"})


def test_log_metrics_rejects_non_finite_value(local_mlflow):
    """Metric values must be finite numbers."""
    setup_mlflow("non-finite-metrics-test")

    with mlflow.start_run():
        with pytest.raises(
            ValueError,
            match="must have a finite value",
        ):
            log_metrics({"f1_score": float("nan")})