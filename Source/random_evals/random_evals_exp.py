#!/usr/bin/env python3
"""Run reproducible random-search HPO experiments.

For each replicate, this script makes one stratified train/test split using a
unique seed and runs three independent random searches containing 10, 100, and
1,000 model evaluations. Thus, the default configuration performs 1,110 model
evaluations per replicate. The data split is shared within a replicate, while
each evaluation budget has its own reproducible random-number stream.

Example
-------
python Source/random_evals/random_evals_exp.py \
    --task_id 359955 \
    --data_directory Data/Raw_OpenML_Suite_271_Binary_Classification \
    --output_directory Results/random_evals \
    --model RF
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# Allow this file to be run directly from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Source.Base.model_param_space import (  # noqa: E402
    GradientBoostParams,
    KernelSVCParams,
    KNeighborsClassifierParams,
    MLPClassifierParams,
    RandomForestParams,
)


DEFAULT_BUDGETS = (10, 100, 1000)
DEFAULT_REPLICATES = 20
MODEL_NAMES = ("RF", "KSVC", "GB", "KNN", "MLP")


def json_safe(value: Any) -> Any:
    """Convert NumPy values and other experiment values to JSON-safe objects."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(json_safe(value), output_file, indent=2, sort_keys=True)


def load_categorical_indicator(data_directory: Path, task_id: int, X: pd.DataFrame) -> List[bool]:
    """Load the repository's JSON/PKL indicator, or infer it when absent."""
    json_path = data_directory / f"task_{task_id}_categorical_indicator.json"
    pickle_path = data_directory / f"task_{task_id}_categorical_indicator.pkl"

    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as indicator_file:
            indicator = json.load(indicator_file)
    elif pickle_path.exists():
        with pickle_path.open("rb") as indicator_file:
            indicator = pickle.load(indicator_file)
    else:
        indicator = [
            isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            for dtype in X.dtypes
        ]

    if len(indicator) != X.shape[1]:
        raise ValueError(
            f"Categorical indicator has {len(indicator)} entries, but the dataset "
            f"has {X.shape[1]} feature columns."
        )
    return [bool(value) for value in indicator]


def load_task_data(
    task_id: int,
    data_directory: Path,
) -> Tuple[pd.DataFrame, np.ndarray, List[bool]]:
    """Load one task using the CSV layout already used by this repository."""
    data_path = data_directory / f"task_{task_id}.csv"
    summary_path = data_directory / "tasks_summary.csv"
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {data_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Task summary does not exist: {summary_path}")

    summary = pd.read_csv(summary_path)
    if "task_id" not in summary.columns:
        raise ValueError(f"{summary_path} does not contain a 'task_id' column.")
    task_rows = summary.loc[summary["task_id"].astype(str) == str(task_id)]
    if task_rows.empty:
        raise ValueError(f"Task {task_id} is not present in {summary_path}.")

    data = pd.read_csv(data_path)
    target_name = None
    if "target_name" in task_rows.columns:
        candidate = task_rows.iloc[0]["target_name"]
        if pd.notna(candidate):
            target_name = str(candidate)
    if target_name is None:
        target_name = str(data.columns[-1])
    if target_name not in data.columns:
        raise ValueError(f"Target column '{target_name}' is missing from {data_path}.")

    X = data.drop(columns=[target_name])
    y = data[target_name].to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("The target must contain at least two classes.")
    indicator = load_categorical_indicator(data_directory, task_id, X)
    return X, y, indicator


def make_preprocessor(X: pd.DataFrame, categorical_indicator: Sequence[bool]) -> ColumnTransformer:
    categorical_columns = [
        column for column, is_categorical in zip(X.columns, categorical_indicator) if is_categorical
    ]
    numerical_columns = [
        column for column, is_categorical in zip(X.columns, categorical_indicator) if not is_categorical
    ]
    return ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), numerical_columns),
            (
                "categorical",
                OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore"),
                categorical_columns,
            ),
        ],
        remainder="passthrough",
    )


def prepare_cv_data(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    categorical_indicator: Sequence[bool],
    seed: int,
    cv_folds: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Fit preprocessing within each fold and return transformed CV data."""
    class_counts = pd.Series(y_train).value_counts()
    if class_counts.min() < cv_folds:
        raise ValueError(
            f"Each training class needs at least {cv_folds} observations for "
            f"{cv_folds}-fold stratified CV; the smallest class has {class_counts.min()}."
        )

    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    folds = []
    for train_indices, validation_indices in splitter.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_indices].reset_index(drop=True)
        X_fold_validation = X_train.iloc[validation_indices].reset_index(drop=True)
        preprocessor = make_preprocessor(X_fold_train, categorical_indicator)
        folds.append(
            (
                preprocessor.fit_transform(X_fold_train),
                y_train[train_indices],
                preprocessor.transform(X_fold_validation),
                y_train[validation_indices],
            )
        )
    return folds


def get_parameter_sampler(model_name: str, binary_classification: bool) -> Any:
    samplers = {
        "RF": RandomForestParams,
        "KSVC": KernelSVCParams,
        "GB": lambda: GradientBoostParams(binary_class=binary_classification),
        "KNN": KNeighborsClassifierParams,
        "MLP": MLPClassifierParams,
    }
    return samplers[model_name]()


def make_estimator(
    model_name: str,
    sampled_params: Dict[str, Any],
    model_seed: int,
    n_jobs: int,
    maximum_neighbors: int | None = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Translate sampled repository parameters into a scikit-learn estimator."""
    if model_name == "RF":
        params = RandomForestParams().eval_parameters(sampled_params)
        estimator = RandomForestClassifier(**params, random_state=model_seed, n_jobs=n_jobs)
    elif model_name == "KSVC":
        params = KernelSVCParams().eval_parameters(sampled_params)
        estimator = SVC(**params, random_state=model_seed, probability=True)
    elif model_name == "GB":
        # The sampled loss is already restricted correctly for binary/multiclass data.
        params = dict(sampled_params)
        estimator = GradientBoostingClassifier(**params, random_state=model_seed)
    elif model_name == "KNN":
        params = KNeighborsClassifierParams().eval_parameters(sampled_params)
        if maximum_neighbors is not None:
            params["n_neighbors"] = min(params["n_neighbors"], maximum_neighbors)
        estimator = KNeighborsClassifier(**params, n_jobs=n_jobs)
    elif model_name == "MLP":
        params = MLPClassifierParams().eval_parameters(sampled_params)
        hidden_layers = tuple(params[f"layer_{index}"] for index in range(1, 6))
        estimator = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=params["activation"],
            solver=params["solver"],
            max_iter=params["max_iter"],
            random_state=model_seed,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return estimator, params


def auc_score(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
) -> float:
    probabilities = estimator.predict_proba(X)
    if len(labels) == 2:
        return float(roc_auc_score(y, probabilities[:, 1], labels=labels))
    return float(
        roc_auc_score(y, probabilities, labels=labels, multi_class="ovo", average="macro")
    )


def evaluate_configuration(
    model_name: str,
    sampled_params: Dict[str, Any],
    model_seed: int,
    n_jobs: int,
    cv_data: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    labels: np.ndarray,
) -> Tuple[float, float, Dict[str, Any]]:
    train_scores = []
    validation_scores = []
    maximum_neighbors = min(len(fold[1]) for fold in cv_data)

    for X_fold_train, y_fold_train, X_fold_validation, y_fold_validation in cv_data:
        estimator, effective_params = make_estimator(
            model_name,
            sampled_params,
            model_seed,
            n_jobs,
            maximum_neighbors,
        )
        estimator.fit(X_fold_train, y_fold_train)
        train_scores.append(auc_score(estimator, X_fold_train, y_fold_train, labels))
        validation_scores.append(auc_score(estimator, X_fold_validation, y_fold_validation, labels))

    return float(np.mean(train_scores)), float(np.mean(validation_scores)), effective_params


def evaluate_on_test_set(
    model_name: str,
    params: Dict[str, Any],
    model_seed: int,
    n_jobs: int,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    categorical_indicator: Sequence[bool],
    labels: np.ndarray,
) -> Tuple[float, float]:
    preprocessor = make_preprocessor(X_train, categorical_indicator)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    estimator, _ = make_estimator(
        model_name,
        params,
        model_seed,
        n_jobs,
        maximum_neighbors=len(y_train),
    )
    estimator.fit(X_train_transformed, y_train)
    return (
        auc_score(estimator, X_train_transformed, y_train, labels),
        auc_score(estimator, X_test_transformed, y_test, labels),
    )


def best_successful_record(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [record for record in records if record["succeeded"]]
    if not successful:
        raise RuntimeError("Every random configuration failed for this evaluation budget.")
    return max(successful, key=lambda record: record["validation_cv_score"])


def write_evaluation_records(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    """Persist completed evaluations, including checkpoints during long runs."""
    evaluation_rows = []
    for record in records:
        row = dict(record)
        row["sampled_params"] = json.dumps(record["sampled_params"], sort_keys=True)
        row["effective_params"] = json.dumps(record["effective_params"], sort_keys=True)
        evaluation_rows.append(row)
    pd.DataFrame(evaluation_rows).to_csv(path, index=False)


def run_replicate(
    task_id: int,
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    categorical_indicator: Sequence[bool],
    output_directory: Path,
    replicate: int,
    seed: int,
    train_size: float,
    budgets: Sequence[int],
    cv_folds: int,
    n_jobs: int,
) -> List[Dict[str, Any]]:
    """Run independent random searches for one shared train/test replicate."""
    replicate_directory = output_directory / f"Task_{task_id}" / model_name / f"Replicate_{replicate}"
    replicate_directory.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    labels = np.unique(y)
    binary_classification = len(labels) == 2
    cv_data = prepare_cv_data(X_train, y_train, categorical_indicator, seed, cv_folds)

    all_records: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    total_evaluations = sum(budgets)

    print(
        f"Replicate {replicate + 1}: split seed={seed}, "
        f"independent evaluations={total_evaluations}",
        flush=True,
    )
    for budget in budgets:
        # Derive separate, stable streams for parameter sampling and estimator
        # randomness. Including the budget ensures that no budget is a prefix of
        # another budget's random-search sequence.
        parameter_seed = int(
            np.random.SeedSequence([seed, budget, 0]).generate_state(1, dtype=np.uint32)[0]
        )
        model_seed_rng = np.random.default_rng(np.random.SeedSequence([seed, budget, 1]))
        parameter_rng = np.random.default_rng(parameter_seed)
        parameter_sampler = get_parameter_sampler(model_name, binary_classification)
        records: List[Dict[str, Any]] = []
        progress_interval = max(1, budget // 10)

        print(f"  Starting independent budget {budget} (seed={parameter_seed})", flush=True)
        for evaluation in range(1, budget + 1):
            sampled_params = parameter_sampler.generate_random_parameters(parameter_rng)
            model_seed = int(model_seed_rng.integers(0, np.iinfo(np.int32).max))
            started_at = time.perf_counter()
            record: Dict[str, Any] = {
                "task_id": task_id,
                "model": model_name,
                "replicate": replicate,
                "split_seed": seed,
                "evaluation_budget": budget,
                "random_search_seed": parameter_seed,
                "model_seed": model_seed,
                "evaluation": evaluation,
                "sampled_params": json_safe(sampled_params),
                "effective_params": None,
                "train_cv_score": np.nan,
                "validation_cv_score": np.nan,
                "succeeded": False,
                "error": None,
            }
            try:
                train_score, validation_score, effective_params = evaluate_configuration(
                    model_name,
                    sampled_params,
                    model_seed,
                    n_jobs,
                    cv_data,
                    labels,
                )
                record.update(
                    {
                        "effective_params": json_safe(effective_params),
                        "train_cv_score": train_score,
                        "validation_cv_score": validation_score,
                        "succeeded": True,
                    }
                )
            except Exception as error:  # Keep a failed configuration in the evaluation count.
                record["error"] = f"{type(error).__name__}: {error}"
                print(
                    f"    Budget {budget}, evaluation {evaluation} failed: {record['error']}",
                    flush=True,
                )
            record["elapsed_seconds"] = time.perf_counter() - started_at
            records.append(record)

            if evaluation % progress_interval == 0 or evaluation == budget:
                print(f"    Completed {evaluation}/{budget} evaluations", flush=True)
                write_evaluation_records(
                    replicate_directory / f"budget_{budget}_evaluations.csv",
                    records,
                )

        best_record = best_successful_record(records)
        train_score, test_score = evaluate_on_test_set(
            model_name,
            best_record["effective_params"],
            best_record["model_seed"],
            n_jobs,
            X_train,
            X_test,
            y_train,
            y_test,
            categorical_indicator,
            labels,
        )
        summary = {
            "task_id": task_id,
            "model": model_name,
            "metric": "roc_auc_ovo_macro" if len(labels) > 2 else "roc_auc",
            "replicate": replicate,
            "split_seed": seed,
            "random_search_seed": parameter_seed,
            "train_size": train_size,
            "cv_folds": cv_folds,
            "evaluation_budget": budget,
            "successful_evaluations": sum(record["succeeded"] for record in records),
            "best_evaluation": best_record["evaluation"],
            "best_params": best_record["effective_params"],
            "best_validation_cv_score": best_record["validation_cv_score"],
            "train_score": train_score,
            "test_score": test_score,
        }
        write_json(replicate_directory / f"budget_{budget}_best_results.json", summary)
        summaries.append(summary)
        all_records.extend(records)
        write_evaluation_records(replicate_directory / "evaluations.csv", all_records)
        print(
            f"  Budget {budget}: validation={best_record['validation_cv_score']:.6f}, "
            f"test={test_score:.6f}",
            flush=True,
        )

    write_json(
        replicate_directory / "replicate_summary.json",
        {"replicate": replicate, "split_seed": seed, "budgets": summaries},
    )
    return summaries


def validate_arguments(args: argparse.Namespace) -> Tuple[int, ...]:
    if args.task_id <= 0:
        raise ValueError("--task_id must be positive.")
    if not 0.0 < args.train_size < 1.0:
        raise ValueError("--train_size must be between 0 and 1.")
    if args.replicates <= 0:
        raise ValueError("--replicates must be positive.")
    if args.replicate_start < 0:
        raise ValueError("--replicate_start must be non-negative.")
    if args.base_seed < 0:
        raise ValueError("--base_seed must be non-negative.")
    if args.cv_folds < 2:
        raise ValueError("--cv_folds must be at least 2.")
    if args.n_jobs == 0:
        raise ValueError("--n_jobs cannot be zero.")
    if not args.budgets or any(budget <= 0 for budget in args.budgets):
        raise ValueError("--budgets must contain positive integers.")
    return tuple(sorted(set(args.budgets)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate random ML hyperparameters in independent 10/100/1000 searches "
            "over 20 uniquely seeded train/test replicates."
        )
    )
    parser.add_argument("--task_id", type=int, required=True, help="OpenML task ID to evaluate.")
    parser.add_argument(
        "--data_directory",
        type=Path,
        required=True,
        help="Directory containing task_<id>.csv and tasks_summary.csv.",
    )
    parser.add_argument(
        "--output_directory",
        type=Path,
        required=True,
        help="Root directory for evaluation CSVs and best-result JSON files.",
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default="RF", help="Model to tune (default: RF).")
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_BUDGETS),
        help="Independent random-search budgets (default: 10 100 1000).",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=DEFAULT_REPLICATES,
        help="Number of train/test replicates (default: 20).",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=0,
        help="Seed offset; replicate r uses base_seed + r (default: 0).",
    )
    parser.add_argument(
        "--replicate_start",
        type=int,
        default=0,
        help="First replicate index, useful for scheduler arrays (default: 0).",
    )
    parser.add_argument("--train_size", type=float, default=0.7, help="Training proportion (default: 0.7).")
    parser.add_argument("--cv_folds", type=int, default=5, help="Stratified CV folds (default: 5).")
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Parallel jobs used by estimators that support it (default: 1).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        budgets = validate_arguments(args)
        X, y, categorical_indicator = load_task_data(args.task_id, args.data_directory)
        experiment_directory = args.output_directory / f"Task_{args.task_id}" / args.model
        experiment_directory.mkdir(parents=True, exist_ok=True)
        replicate_indices = list(
            range(args.replicate_start, args.replicate_start + args.replicates)
        )
        split_seeds = [args.base_seed + replicate for replicate in replicate_indices]
        run_label = f"replicates_{replicate_indices[0]}_{replicate_indices[-1]}"
        experiment_config = {
            "task_id": args.task_id,
            "model": args.model,
            "data_directory": args.data_directory.resolve(),
            "budgets": budgets,
            "replicates": args.replicates,
            "replicate_indices": replicate_indices,
            "split_seeds": split_seeds,
            "train_size": args.train_size,
            "cv_folds": args.cv_folds,
            "n_jobs": args.n_jobs,
            "budgets_are_nested": False,
            "total_evaluations_per_replicate": sum(budgets),
        }
        write_json(
            experiment_directory / "run_configs" / f"{run_label}.json",
            experiment_config,
        )

        all_summaries = []
        for replicate, seed in zip(replicate_indices, split_seeds):
            all_summaries.extend(
                run_replicate(
                    task_id=args.task_id,
                    model_name=args.model,
                    X=X,
                    y=y,
                    categorical_indicator=categorical_indicator,
                    output_directory=args.output_directory,
                    replicate=replicate,
                    seed=seed,
                    train_size=args.train_size,
                    budgets=budgets,
                    cv_folds=args.cv_folds,
                    n_jobs=args.n_jobs,
                )
            )

        summary_rows = []
        for summary in all_summaries:
            row = dict(summary)
            row["best_params"] = json.dumps(summary["best_params"], sort_keys=True)
            summary_rows.append(row)
        summary_frame = pd.DataFrame(summary_rows)
        summaries_directory = experiment_directory / "summaries"
        summaries_directory.mkdir(parents=True, exist_ok=True)
        summary_frame.to_csv(summaries_directory / f"{run_label}.csv", index=False)
        if args.replicates > 1:
            # A multi-replicate local run has no concurrent writers, so also
            # provide convenient aggregate files at the experiment root.
            write_json(experiment_directory / "experiment_config.json", experiment_config)
            summary_frame.to_csv(experiment_directory / "summary.csv", index=False)
        print(f"Finished. Results saved under {experiment_directory}", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
