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
    if null_columns:
        print(f"Columns with null values: {len(null_columns)}")
        for column in null_columns:
            null_count = data[column].isnull().sum()
            print(f"  - {column}: {null_count} null values")

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
