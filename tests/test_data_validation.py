"""Unit tests for the shared data validation utilities."""

import pandas as pd
import pytest

from common.data_validation import validate_dataframe


def test_valid_dataframe_passes():
    """A valid DataFrame should pass validation."""
    df = pd.DataFrame(
        {
            "amount": [100.0, 250.0],
            "label": [0, 1],
        }
    )

    validate_dataframe(
        df,
        required_columns=["amount", "label"],
    )


def test_empty_dataframe_is_rejected():
    """An empty DataFrame should be rejected by default."""
    df = pd.DataFrame(columns=["amount", "label"])

    with pytest.raises(ValueError, match="DataFrame is empty"):
        validate_dataframe(df)


def test_empty_dataframe_can_be_allowed():
    """An empty DataFrame should pass when explicitly allowed."""
    df = pd.DataFrame(columns=["amount", "label"])

    validate_dataframe(df, allow_empty=True)


def test_missing_required_columns_are_rejected():
    """Missing required columns should produce a clear error."""
    df = pd.DataFrame({"amount": [100.0]})

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        validate_dataframe(
            df,
            required_columns=["amount", "label"],
        )


def test_missing_values_are_rejected_by_default():
    """Missing values should be rejected by default."""
    df = pd.DataFrame(
        {
            "amount": [100.0, None],
            "label": [0, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="contains missing values",
    ):
        validate_dataframe(df)


def test_missing_values_can_be_allowed():
    """Missing values should pass when explicitly allowed."""
    df = pd.DataFrame(
        {
            "amount": [100.0, None],
            "label": [0, 1],
        }
    )

    validate_dataframe(df, allow_missing_values=True)


def test_non_dataframe_input_is_rejected():
    """Input that is not a DataFrame should raise TypeError."""
    with pytest.raises(
        TypeError,
        match="must be a pandas DataFrame",
    ):
        validate_dataframe([100, 200])


def test_duplicate_column_names_are_rejected():
    """Duplicate column names should be rejected."""
    df = pd.DataFrame(
        [[100.0, 200.0]],
        columns=["amount", "amount"],
    )

    with pytest.raises(
        ValueError,
        match="duplicate column names",
    ):
        validate_dataframe(df)