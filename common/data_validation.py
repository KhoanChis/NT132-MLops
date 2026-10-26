"""Reusable data validation utilities for the MLOps platform."""

from collections.abc import Sequence

import pandas as pd


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Sequence[str] | None = None,
    allow_missing_values: bool = False,
    allow_empty: bool = False,
) -> None:
    """Validate a pandas DataFrame before downstream processing.

    Args:
        df: DataFrame to validate.
        required_columns: Columns that must exist in the DataFrame.
        allow_missing_values: Whether missing values are permitted.
        allow_empty: Whether an empty DataFrame is permitted.

    Raises:
        TypeError: If the input is not a pandas DataFrame.
        ValueError: If one or more validation checks fail.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input data must be a pandas DataFrame."
        )

    if not allow_empty and df.empty:
        raise ValueError(
            "DataFrame is empty. At least one row is required."
        )

    if not df.columns.is_unique:
        raise ValueError(
            "DataFrame contains duplicate column names."
        )

    if required_columns is not None:
        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

    if not allow_missing_values and df.isnull().values.any():
        missing_counts = (
            df.isnull()
            .sum()
            .loc[lambda counts: counts > 0]
            .to_dict()
        )

        raise ValueError(
            f"DataFrame contains missing values: {missing_counts}"
        )