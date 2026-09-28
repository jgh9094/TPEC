import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load and print spine opioid dataset arrays."
    )
    parser.add_argument("--x", required=True, help="Path to X.npy")
    parser.add_argument("--y", required=True, help="Path to Y.npy")
    parser.add_argument(
        "--feature-names", required=True, help="Path to feature_names.npy"
    )
    parser.add_argument(
        "--outcome-names", required=True, help="Path to outcome_names.npy"
    )
    parser.add_argument(
        "--output-csv",
        default="combined.csv",
        help="Path for the combined output CSV",
    )
    return parser.parse_args()


def combine_to_csv(X, Y, feature_names, outcome_names, output_path):
    """Combine X, Y and their column names into a single CSV file.

    Columns are the feature names followed by the outcome names, and each
    row is the corresponding X row concatenated with its Y row.
    """
    # Ensure Y is 2D so it can be horizontally stacked with X.
    Y2d = Y.reshape(-1, 1) if Y.ndim == 1 else Y

    header = list(feature_names) + list(outcome_names)
    combined = np.hstack([X, Y2d])

    np.savetxt(
        output_path,
        combined,
        delimiter=",",
        header=",".join(str(h) for h in header),
        comments="",
        fmt="%s",
    )

    print(f"\nWrote combined CSV with shape {combined.shape} to {output_path}")


def count_missing(X, feature_names):
    """Count rows with any missing value and missing values per column.

    A value is considered missing if it is NaN. Object/string arrays are
    handled by also treating None as missing.
    """
    # Build a boolean mask of missing entries that works for numeric and
    # object dtypes.
    if np.issubdtype(X.dtype, np.floating):
        missing_mask = np.isnan(X)
    else:
        missing_mask = np.zeros(X.shape, dtype=bool)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                val = X[i, j]
                if val is None:
                    missing_mask[i, j] = True
                else:
                    try:
                        missing_mask[i, j] = np.isnan(float(val))
                    except (TypeError, ValueError):
                        missing_mask[i, j] = False

    rows_with_missing = int(np.any(missing_mask, axis=1).sum())
    missing_per_column = missing_mask.sum(axis=0)

    print("\n=== Missing value summary ===")
    print(f"rows with at least one missing value: {rows_with_missing} / {X.shape[0]}")
    if missing_mask.any():
        print("missing values per column:")
        for name, count in zip(feature_names, missing_per_column):
            if count > 0:
                print(f"  {name}: {int(count)}")
    else:
        print("no columns have missing values")

    return rows_with_missing, missing_per_column


def main():
    args = parse_args()

    X = np.load(args.x, allow_pickle=True)
    Y = np.load(args.y, allow_pickle=True)
    feature_names = np.load(args.feature_names, allow_pickle=True)
    outcome_names = np.load(args.outcome_names, allow_pickle=True)

    print("=== X ===")
    print(f"shape: {X.shape}, dtype: {X.dtype}")
    print(X)

    print("\n=== Y ===")
    print(f"shape: {Y.shape}, dtype: {Y.dtype}")
    print(Y)

    print("\n=== feature_names ===")
    print(f"shape: {feature_names.shape}, dtype: {feature_names.dtype}")
    print(feature_names)

    print("\n=== outcome_names ===")
    print(f"shape: {outcome_names.shape}, dtype: {outcome_names.dtype}")
    print(outcome_names)

    count_missing(X, feature_names)

    combine_to_csv(X, Y, feature_names, outcome_names, args.output_csv)


if __name__ == "__main__":
    main()
