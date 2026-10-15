"""Load, validate, and inspect the spinal-surgery NumPy dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATASET_DIRECTORY = Path(__file__).resolve().parent
ROWS_TO_PRINT = 10
REQUIRED_MODELING_FILES = (
    "X.npy",
    "y.npy",
    "feature_names.npy",
    "output_names.npy",
)
FEATURE_NAME_ALIASES = {
    "PLATELET COUNT": "PLATELET_COUNT",
    "WBC COUNT": "WBC_COUNT",
    "RBC COUNT": "RBC_COUNT",
}
OUTCOME_NAME_ALIASES = {
    "los_greater_than_7d": "LOS_extended",
}


def _load_array(dataset_directory: Path, filename: str) -> np.ndarray:
    path = dataset_directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required spinal-surgery file is missing: {path}")
    # feature_names.npy is an object array; the remaining supplied arrays are numeric
    # or fixed-width Unicode arrays. These files are trusted repository data.
    return np.load(path, allow_pickle=filename == "feature_names.npy")


def _normalized_names(values: np.ndarray, aliases: dict[str, str]) -> list[str]:
    names = [aliases.get(str(value), str(value)) for value in values.tolist()]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate names after schema normalization: {duplicates}")
    return names


def load_modeling_data(
    dataset_directory: Path = DEFAULT_DATASET_DIRECTORY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return validated feature and outcome tables built directly from the NumPy files.

    Patient identifiers and surgery timestamps are intentionally not included in the
    modeling features. The returned outcome names use the established experiment schema.
    """
    dataset_directory = Path(dataset_directory).expanduser().resolve()
    if not dataset_directory.is_dir():
        raise FileNotFoundError(
            f"Spinal-surgery dataset directory does not exist: {dataset_directory}"
        )
    missing_files = [
        filename
        for filename in REQUIRED_MODELING_FILES
        if not (dataset_directory / filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Spinal-surgery dataset is incomplete; missing: {missing_files}"
        )

    X = _load_array(dataset_directory, "X.npy")
    y = _load_array(dataset_directory, "y.npy")
    feature_names = _normalized_names(
        _load_array(dataset_directory, "feature_names.npy"),
        FEATURE_NAME_ALIASES,
    )
    output_names = _normalized_names(
        _load_array(dataset_directory, "output_names.npy"),
        OUTCOME_NAME_ALIASES,
    )

    if X.ndim != 2 or y.ndim != 2:
        raise ValueError(f"X and y must be two-dimensional; found {X.shape=} and {y.shape=}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y row counts differ: {X.shape[0]} != {y.shape[0]}"
        )
    if X.shape[1] != len(feature_names):
        raise ValueError(
            "X column count does not match feature_names.npy: "
            f"{X.shape[1]} != {len(feature_names)}"
        )
    if y.shape[1] != len(output_names):
        raise ValueError(
            "y column count does not match output_names.npy: "
            f"{y.shape[1]} != {len(output_names)}"
        )

    return (
        pd.DataFrame(X, columns=feature_names),
        pd.DataFrame(y, columns=output_names),
    )


def print_dataset_contents(dataset_directory: Path) -> None:
    """Print directory contents and a preview of every NumPy data file."""
    if not dataset_directory.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_directory}"
        )

    entries = sorted(dataset_directory.iterdir(), key=lambda path: path.name.lower())
    files = [entry for entry in entries if entry.is_file()]

    print(f"Dataset directory: {dataset_directory}")
    print(f"Files found: {len(files)}")
    for file_path in files:
        print(f"  {file_path.name} ({file_path.stat().st_size:,} bytes)")

    np.set_printoptions(linewidth=160, threshold=np.inf)
    for file_path in files:
        print(f"\n{'=' * 80}")
        print(f"File: {file_path.name}")

        if file_path.suffix.lower() != ".npy":
            print("Preview skipped: unsupported file type.")
            continue

        # Some supplied arrays contain strings stored with NumPy's object dtype.
        data = np.load(file_path, allow_pickle=True)
        row_count = data.shape[0] if data.ndim else 1
        preview = data[:ROWS_TO_PRINT] if data.ndim else data

        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Dimensions: {data.ndim}")
        print(f"Rows/elements shown: {min(ROWS_TO_PRINT, row_count)} of {row_count}")
        print(preview)

    features, outcomes = load_modeling_data(dataset_directory)
    print(f"\n{'=' * 80}")
    print("Validated modeling dataset")
    print(f"Samples: {len(features)}")
    print(f"Features: {features.shape[1]}")
    print(f"Outcomes: {outcomes.columns.tolist()}")
    print("First 10 modeling rows:")
    print(pd.concat([features, outcomes], axis=1).head(ROWS_TO_PRINT).to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List dataset files and print the first 10 rows of each NumPy file."
    )
    parser.add_argument(
        "dataset_directory",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET_DIRECTORY,
        help=f"Dataset directory (default: {DEFAULT_DATASET_DIRECTORY})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print_dataset_contents(arguments.dataset_directory.expanduser().resolve())
