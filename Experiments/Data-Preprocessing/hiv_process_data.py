"""Print the dimensions and column headers of a HIV CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def print_csv_summary(csv_path: Path) -> None:
    """Print the row count, column count, and column headers of a CSV file."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    data = pd.read_csv(csv_path)

    print(f"CSV file: {csv_path}")
    print(f"Dimensions: {data.shape[0]} rows x {data.shape[1]} columns")

    # go through the column headers and find how many have null values
    null_columns = data.columns[data.isnull().any()].tolist()
    print(f"Columns with null values: {len(null_columns)}")

    # drop any columns that have null values from null_columns
    data = data.drop(columns=null_columns)

    # print new dimensions after dropping columns with null values
    print(f"New dimensions after dropping columns with null values: {data.shape[0]} rows x {data.shape[1]} columns")

    # go through columns and find how many have only leq than two unique values
    low_unique_columns = [col for col in data.columns if data[col].nunique() <= 2]
    print(f"Columns with less than two unique values: {len(low_unique_columns)}")

    # go through and find the more than two unique values columns and print the first 5 unique values for each
    more_unique_columns = [col for col in data.columns if data[col].nunique() > 2]
    print(f"Columns with more than two unique values: {len(more_unique_columns)}")
    for col in more_unique_columns:
        print(f"Column: {col}, Unique values: {data[col].unique()[:5]}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the dimensions and column headers of a CSV file."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the CSV file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print_csv_summary(arguments.csv_path.expanduser().resolve())
