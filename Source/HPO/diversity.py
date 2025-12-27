import numpy as np
import pandas as pd
import argparse
import sys
import os
from itertools import combinations
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Source.Base.eval_utils import MODEL_CONFIG

def normalize_hyperparameters(df: pd.DataFrame, param_space: dict) -> tuple:
    """
    Normalize numerical hyperparameters using min-max normalization.
    Keep categorical parameters as-is for distance computation.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing hyperparameter values
    param_space : dict
        Parameter space dictionary from MODEL_CONFIG

    Returns:
    --------
    df_normalized : pd.DataFrame
        DataFrame with normalized numerical values and original categorical values
    numerical_cols : list
        List of column names that are numerical (normalized)
    categorical_cols : list
        List of column names that are categorical
    """
    df_normalized = df.copy()
    numerical_cols = []
    categorical_cols = []

    for param_name, param_spec in param_space.items():
        if param_name not in df.columns:
            assert False, f"Parameter {param_name} not found in DataFrame columns."

        param_type = param_spec['type']

        if param_type == 'float' or param_type == 'int':
            # Min-max normalization for numerical parameters using actual values in CSV
            min_val = df[param_name].min()
            max_val = df[param_name].max()

            # Avoid division by zero if all values are the same
            if max_val == min_val:
                df_normalized[param_name] = 0.5
            else:
                df_normalized[param_name] = (df[param_name] - min_val) / (max_val - min_val)
            numerical_cols.append(param_name)

        elif param_type == 'cat':
            # Keep categorical parameters as-is for binary distance (0 if equal, 1 if not)
            categorical_cols.append(param_name)

        elif param_type == 'bool':
            # Boolean parameters: treat as categorical (0 if equal, 1 if not)
            categorical_cols.append(param_name)

        else: # Unknown parameter type
            raise ValueError(f"Unknown parameter type: {param_type} for parameter {param_name}")

    return df_normalized, numerical_cols, categorical_cols

def compute_pairwise_distances(df_normalized: pd.DataFrame, numerical_cols: list, categorical_cols: list) -> np.ndarray:
    """
    Compute all pairwise distances between rows.
    For numerical columns: use normalized Euclidean distance
    For categorical columns: use 0 if equal, 1 if not equal

    Parameters:
    -----------
    df_normalized : pd.DataFrame
        DataFrame with normalized numerical values and original categorical values
    numerical_cols : list
        List of numerical column names
    categorical_cols : list
        List of categorical column names

    Returns:
    --------
    distances : np.ndarray
        Array of all pairwise distances
    """
    n = len(df_normalized)

    # Extract numerical and categorical data separately
    numerical_data = df_normalized[numerical_cols].values if numerical_cols else np.zeros((n, 0))
    categorical_data = df_normalized[categorical_cols].values if categorical_cols else np.zeros((n, 0))

    # Compute pairwise distances
    distances = []
    for i, j in combinations(range(n), 2):
        # Euclidean distance for numerical parameters
        if len(numerical_cols) > 0:
            numerical_dist_sq = np.sum((numerical_data[i] - numerical_data[j]) ** 2)
        else:
            numerical_dist_sq = 0.0

        # Binary distance for categorical parameters (0 if equal, 1 if not)
        if len(categorical_cols) > 0:
            categorical_dist_sq = np.sum(categorical_data[i] != categorical_data[j])
        else:
            categorical_dist_sq = 0.0

        # Combined distance
        dist = np.sqrt(numerical_dist_sq + categorical_dist_sq)
        distances.append(dist)

    return np.array(distances)

def compute_diversity_metric(csv_path: str, model_type: str) -> float:
    """
    Compute diversity metric as the median of min-max normalized pairwise Euclidean distances.

    Parameters:
    -----------
    csv_path : str
        Path to the archive CSV file
    model_type : str
        Model type identifier (e.g., 'RF', 'DT', etc.)

    Returns:
    --------
    diversity : float
        Median pairwise distance across all evaluated hyperparameters
    """
    # Load the CSV
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        print("Warning: Empty CSV file")
        return 0.0

    if len(df) == 1:
        print("Warning: Only one row in CSV file")
        return 0.0

    # Get parameter space for the model
    if model_type not in MODEL_CONFIG:
        raise ValueError(f"Unknown model type: {model_type}. Valid options: {list(MODEL_CONFIG.keys())}")

    param_class = MODEL_CONFIG[model_type]['param_class']
    param_space = param_class.get_parameter_space()

    # Normalize hyperparameters
    df_normalized, numerical_cols, categorical_cols = normalize_hyperparameters(df, param_space)

    # Compute pairwise distances
    distances = compute_pairwise_distances(df_normalized, numerical_cols, categorical_cols)

    # Return median distance
    diversity = float(np.median(distances))

    return diversity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute diversity metric for hyperparameter archive")

    parser.add_argument('--model_type', type=str, required=True,
                        choices=['RF', 'DT', 'KSVC', 'LSVC', 'LSGD', 'ET', 'GB'],
                        help='Type of model (must match MODEL_CONFIG keys)')

    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to the archive CSV file')

    args = parser.parse_args()

    # Compute diversity
    diversity = compute_diversity_metric(args.csv_path, args.model_type)

    print(f"Model Type: {args.model_type}")
    print(f"CSV Path: {args.csv_path}")
    print(f"Diversity Metric (Median Pairwise Distance): {diversity:.6f}")

    # Save results to JSON file in the same directory as the CSV
    csv_dir = os.path.dirname(args.csv_path)
    json_path = os.path.join(csv_dir, "diversity_metric.json")

    results = {
        "model_type": args.model_type,
        "csv_path": args.csv_path,
        "diversity_metric": diversity
    }

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to: {json_path}")
