##########################################################################################
# Search custom pipelines with auto-sklearn 0.15 and score a held-out test set.
# 
# Usage (built-in classification smoke test):
#     python auto-sklearn_wrapper.py \
#         --task debug --y_label target --seed 42 --train_p 0.8 \
#         --n_folds 3 --n_evaluations 5 --time_limit_mins 5 \
#         --max_eval_time_mins 1 --output_directory results/debug_classification
#
# With --task debug, omit --data_path to use a built-in dataset.
# Classification is the default and uses the breast cancer dataset.
# Add --classification false to use the diabetes regression dataset.
# The CLI requires --y_label even for built-in data. Use "target" as a placeholder.
# To use your own debug CSV, pass --data_path dataset.csv and set --y_label
# to the name of its target column.
#
# --components smoke is the default. It limits the estimator to random forest
# and the feature preprocessor to PCA, while built-in data preprocessing
# remains part of the search.
##########################################################################################

import argparse
import json
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, PredefinedSplit, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.ensembles import SingleBest
from autosklearn.metrics import make_scorer, r2, roc_auc

from Source.Base.model_param_space import DataContext

import param_space_conversion as components

OUTCOME_COLUMNS = ("LOS_extended", "discharge_Home", "HOSP_READM_90")
SEARCH_SPACE_REVISION = "spinal_surgery_numpy_custom_cv_external_preprocessing_v6"
SO_NUMERICAL_COLUMNS = (
    "AGE",
    "BMI",
    "SBP",
    "PAIN_SCORE",
    "WBC_COUNT",
    "HEMOGLOBIN",
    "POTASSIUM",
    "SODIUM",
    "PLATELET_COUNT",
    "RBC_COUNT",
    "CALCIUM",
    "CHLORIDE",
    "BUN",
    "CREATININE",
)

def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, received {value!r}")

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--task', choices=['so', 'debug'], required=True)
    parser.add_argument('--y_label', required=True)
    parser.add_argument(
        "--data_path",
        type=Path,
        help=(
            "Spinal-surgery NumPy dataset directory for task 'so', or a CSV file "
            "for task 'debug'."
        ),
    )
    parser.add_argument('--train_p', type=float, required=True)
    parser.add_argument('--output_directory', type=Path, required=True)
    parser.add_argument("--classification", type=parse_bool, default=True)
    parser.add_argument('--cores', type=int, default=1)
    parser.add_argument('--n_evaluations', type=int, default=500)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument(
        "--max_eval_time_mins",
        type=float,
        default=5.0,
        help="Maximum wall time per candidate pipeline evaluation in minutes. Default 5.",
    )
    parser.add_argument('--time_limit_mins', type=float, default=1440.0)
    parser.add_argument('--memory_limit_mb', type=int, default=4096)
    parser.add_argument("--components", choices=["smoke", "all"], default="smoke",
                    help="Smoke searches custom RF and PCA. All uses the project registries.")
    return parser.parse_args()

def validate_args(args: argparse.Namespace) -> None:
    if args.task == "so" and (args.data_path is None or not args.data_path.is_dir()):
        raise FileNotFoundError(f"Expected a spine data directory, got {args.data_path}")
    if args.task == "debug" and args.data_path is not None and not args.data_path.is_file():
        raise FileNotFoundError(f"Debug CSV does not exist: {args.data_path}")
    if not 0.0 < args.train_p < 1.0:
        raise ValueError("train_p must be between 0 and 1 (exclusive)")
    if args.cores < 1 or args.n_folds < 2 or args.n_evaluations < 1 or args.memory_limit_mb < 1:
        raise ValueError("cores, n_evaluations and memory_limit_mb must be positive; n_folds must be at least 2")
    for name in ("max_eval_time_mins", "time_limit_mins"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
            return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(json_safe(value), file, indent=2)
    os.replace(temporary, path)


# def load_spine_opioid_data(data_path: Path, y_label: str):
#     if y_label not in OUTCOME_COLUMNS:
#         raise ValueError(f"Invalid spine-opioid outcome {y_label!r}")
#     X, outcomes = load_modeling_data(data_path)
#     missing_outcomes = [
#         column for column in OUTCOME_COLUMNS if column not in outcomes
#     ]
#     if missing_outcomes:
#         raise ValueError(f"Missing outcome columns: {missing_outcomes}")
#     missing_scaled = [column for column in SO_NUMERICAL_COLUMNS if column not in X]
#     if missing_scaled:
#         raise ValueError(f"Missing columns that must be scaled: {missing_scaled}")

#     y = outcomes[y_label].to_numpy()
#     return X, y, list(SO_NUMERICAL_COLUMNS), []


# def load_generic_data(data_path: Path, y_label: str):
#     data = pd.read_csv(data_path)
#     if y_label not in data:
#         raise ValueError(f"Target column {y_label!r} is missing from {data_path}")
#     X = data.drop(columns=[y_label])
#     return X, data[y_label].to_numpy(), list(X.columns), []


def load_data(args):
    if args.task == "so":
        from spinal_surgery import load_modeling_data

        if args.y_label not in OUTCOME_COLUMNS:
            raise ValueError(f"Invalid spine-opioid outcome: {args.y_label!r}")
        
        X, outcomes = load_modeling_data(args.data_path)
        if args.y_label not in outcomes:
            raise ValueError(f"Missing spine-opioid outcome: {args.y_label!r}")
        
        if not X.index.equals(outcomes.index):
            raise ValueError("Spine features and outcomes have different row indices")
        
        return X, outcomes[args.y_label]

    # default debug dataset
    if args.data_path is None:
        from sklearn.datasets import load_breast_cancer, load_diabetes

        loader = load_breast_cancer if args.classification else load_diabetes
        X, y = loader(return_X_y=True, as_frame=True)
        return X, y

    data = pd.read_csv(args.data_path)
    if args.y_label not in data:
        raise ValueError(f"Target column {args.y_label!r} is missing from {args.data_path}")
    return data.drop(columns=[args.y_label]), data[args.y_label]



def make_splits(X, y, args):
    if pd.isna(y).any():
        raise ValueError("Target contains missing values")
    labels = None
    if args.classification:
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        labels = encoder.classes_
        if len(labels) < 2:
            raise ValueError("ROC AUC requires at least two target classes")
    else:
        y = pd.to_numeric(y, errors="raise").astype(float)
        if not np.isfinite(y).all():
            raise ValueError("Regression targets must be finite")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=args.train_p, shuffle=True,
        stratify=y if args.classification else None, random_state=args.seed,
    )
    if args.classification:
        if np.bincount(y_train, minlength=len(labels)).min() < args.n_folds:
            raise ValueError("Each class needs at least n_folds rows in the training set")
        if len(np.unique(y_test)) != len(labels):
            raise ValueError("Every class must appear in the held-out test set for ROC AUC")
        splitter = StratifiedKFold
    else:
        if len(y_train) < 2 * args.n_folds or len(y_test) < 2:
            raise ValueError("R² requires at least two rows in each validation fold and test set")
        splitter = KFold

    folds = splitter(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    test_fold = np.empty(len(y_train), dtype=int)
    smallest_fold = len(y_train)
    for fold, (train_indices, valid_indices) in enumerate(folds.split(X_train, y_train)):
        test_fold[valid_indices] = fold
        smallest_fold = min(smallest_fold, len(train_indices))

    return X_train, X_test, y_train, y_test, PredefinedSplit(test_fold), smallest_fold, labels



def register_components(args, data_context):
    """Register custom pipeline components, return the 'include' dictionary for autosklearn's runner"""
    stage = "classifier" if args.classification else "regressor"

    # register TPEC's parameter space, i.e., everything from 'param_space_conversion.py'
    if args.components == "all":
        register_estimators = ( # pick a function!
            components.register_classifiers if args.classification
            else components.register_regressors
        )
        return {
            stage: register_estimators(data_context),
            "feature_preprocessor": components.register_feature_preprocessors(
                data_context, classification=args.classification,
            ) + ["no_preprocessing"], # skipping feature preprocessing is an option
        }

    # smoke test, register one RF and one PCA
    from Source.ML.transformer import PCAParams
    from sklearn.decomposition import PCA

    if args.classification:
        from Source.ML.classifiers import RandomForestParams
        from sklearn.ensemble import RandomForestClassifier

        CustomRF = components.make_classifier_class(
            RandomForestClassifier, RandomForestParams, name="CustomRF",
            data_context=data_context,
        )
        estimator_name = components.register_classifier(CustomRF)
    else:
        from Source.ML.regressor import RandomForestRegressorParams
        from sklearn.ensemble import RandomForestRegressor


        CustomRF = components.make_regressor_class(
            RandomForestRegressor, RandomForestRegressorParams, name="CustomRFRegressor",
            data_context=data_context,
        )
        estimator_name = components.register_regressor(CustomRF)

    CustomPCA = components.make_preprocessor_class(
        PCA, PCAParams, name="CustomPCA", data_context=data_context,
        classification=args.classification,
    )
    return {
        stage: [estimator_name],
        "feature_preprocessor": [components.register_preprocessor(CustomPCA)],
    }


def run(args):
    started = time.monotonic()
    validate_args(args)
    X, y = load_data(args)
    X_train, X_test, y_train, y_test, cv, smallest_fold, labels = make_splits(X, y, args)

    data_context = DataContext(
        n_samples=smallest_fold, n_features=X_train.shape[1], n_classes=len(labels) if args.classification else 0,
    )
    include = register_components(args, data_context)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    work_directory = Path(tempfile.mkdtemp(prefix="search-", dir=args.output_directory))

    estimator_class = AutoSklearnClassifier if args.classification else AutoSklearnRegressor
    if not args.classification:
        metric = r2
    elif len(labels) == 2:
        metric = roc_auc
    else:
        metric = make_scorer(
            "roc_auc_ovo", roc_auc_score, needs_proba=True,
            multi_class="ovo", average="macro",
        )

    automl = estimator_class(
        time_left_for_this_task=math.ceil(60 * args.time_limit_mins),
        per_run_time_limit=math.ceil(60 * args.max_eval_time_mins),
        memory_limit=args.memory_limit_mb,
        n_jobs=args.cores,
        seed=args.seed,
        include=include,
        metric=metric,
        resampling_strategy=cv,
        dataset_compression=False,
        initial_configurations_via_metalearning=0,
        ensemble_class=SingleBest,
        smac_scenario_args={"runcount_limit": args.n_evaluations},
        tmp_folder=str(work_directory / "autosklearn"),
        delete_tmp_folder_after_terminate=False,
    )
    automl.fit(X_train, y_train) # begin search! find a configuration

    leaderboard = automl.leaderboard(detailed=True)
    leaderboard.to_csv(work_directory / "leaderboard.csv")
    trials = pd.DataFrame(automl.cv_results_)
    trials.to_csv(work_directory / "cv_results.csv", index=False)

    automl.refit(X_train, y_train) # fit configuration on all training data

    # score final configuration on the held-out test set
    if args.classification:
        probabilities = automl.predict_proba(X_test)
        if len(labels) == 2:
            score = roc_auc_score(y_test, probabilities[:, 1])
        else:
            score = roc_auc_score(
                y_test, probabilities, multi_class="ovo", average="macro",
            )
        metric_name = metric.name
    else:
        score = r2_score(y_test, automl.predict(X_test))
        metric_name = "r2"
    
    result = {
        "task": args.task,
        "target": (
            ("breast_cancer" if args.classification else "diabetes")
            if args.task == "debug" and args.data_path is None else args.y_label
        ),
        "seed": args.seed,
        "include": include,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_folds": args.n_folds,
        "classification": args.classification,
        "metric": metric_name,
        "test_auc" if args.classification else "test_r2": score,
        "requested_smac_run_limit": args.n_evaluations,
        "cv_results_rows": len(trials),
        "cv_status_counts": trials["status"].value_counts().to_dict() if "status" in trials else {},
        "elapsed_seconds": time.monotonic() - started,
        "search_directory": work_directory,
    }
    if args.classification:
        result["class_labels"] = labels
        if len(labels) == 2:
            result["positive_label"] = labels[1]

    write_json(args.output_directory / "results.json", result)
    print(automl.sprint_statistics(), flush=True)
    print(f"Held-out {metric_name}: {score:.4f}", flush=True)


def main():
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        failure = {
            "error": str(error),
            "trace": traceback.format_exc(),
            "seed": args.seed,
            "task_id": args.y_label,
        }
        write_json(args.output_directory / "failed.json", failure)
        print(failure["trace"], file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()