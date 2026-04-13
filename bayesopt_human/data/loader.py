"""CSV loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_observations(
    filepath: str,
    objective_column: str,
) -> pd.DataFrame:
    """Load observations from a CSV file.

    Args:
        filepath: Path to a ``.csv`` data file.
        objective_column: Name of the column containing objective values.

    Returns:
        DataFrame with validated observations. All columns other than
        ``objective_column`` are treated as parameter columns.

    Raises:
        ValueError: If the file is not a ``.csv``, contains missing values,
                    has non-numeric columns, or has fewer than 2 observations.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Unsupported file format: {path.suffix}. Observations must be "
            f"a .csv file."
        )
    df = pd.read_csv(filepath)

    if objective_column not in df.columns:
        raise ValueError(
            f"Objective column {objective_column!r} not found in data. "
            f"Available columns: {list(df.columns)}"
        )

    param_columns = [c for c in df.columns if c != objective_column]
    if not param_columns:
        raise ValueError("No parameter columns found in data.")

    cols = param_columns + [objective_column]
    df = df[cols].copy()

    # Validate no missing values
    if df.isna().any().any():
        missing_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Data contains missing values in columns: {missing_cols}")

    # Validate numeric
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col!r} contains non-numeric values.")

    # Validate minimum observations
    if len(df) < 2:
        raise ValueError(
            f"At least 2 observations required, got {len(df)}."
        )

    return df
