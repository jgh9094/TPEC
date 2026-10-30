"""Run one linear or graph TPOT replicate over a shared operator space."""

import argparse
import fcntl
import json
import os
import pickle
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import tpot
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    get_scorer,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

# SLURM invokes this file directly, so make repository imports independent of cwd.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
DATA_TOOLS_DIRECTORY = REPOSITORY_ROOT / "Data-Tools"
if str(DATA_TOOLS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(DATA_TOOLS_DIRECTORY))

try:
    from .param_space_conversion import (
        generate_tpot_graph_search_space,
        generate_tpot_search_space,
    )
except ImportError:  # direct execution
    from param_space_conversion import (
        generate_tpot_graph_search_space,
        generate_tpot_search_space,
    )

from Source.Base.model_param_space import DataContext
from spinal_surgery import load_modeling_data


OUTCOME_COLUMNS = ("LOS_extended", "discharge_Home", "HOSP_READM_90")
REQUIRED_TPOT_VERSION = "1.1.0"
DEFAULT_MAX_EVAL_TIME_MINS = 5.0
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one TPOT CASH comparison")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task", choices=["so", "debug"], required=True)
    parser.add_argument("--y_label", required=True)
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help=(
            "Spinal-surgery NumPy dataset directory for task 'so', or a CSV file "
            "for task 'debug'."
        ),
    )
    parser.add_argument("--train_p", type=float, required=True)
    parser.add_argument("--output_directory", type=Path, required=True)
    parser.add_argument(
        "--search_space",
        choices=["constrained", "graph"],
        required=True,
    )
    parser.add_argument("--classification", type=parse_bool, default=True)
    parser.add_argument("--cores", type=int, required=True)
    parser.add_argument("--pop_size", type=int, required=True)
    parser.add_argument("--generations", type=int, required=True)
    parser.add_argument("--n_evaluations", type=int, default=500)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument(
        "--max_eval_time_mins",
        type=float,
        default=DEFAULT_MAX_EVAL_TIME_MINS,
        help="Maximum wall time per candidate pipeline evaluation in minutes. Default 5.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    installed_version = getattr(tpot, "__version__", "unknown")
    if installed_version != REQUIRED_TPOT_VERSION:
        raise RuntimeError(
            f"This experiment requires TPOT {REQUIRED_TPOT_VERSION} (the latest "
            f"PyPI release when configured); found {installed_version}."
        )
    if args.task == "so" and not args.data_path.is_dir():
        raise FileNotFoundError(
            f"Spinal-surgery dataset directory does not exist: {args.data_path}"
        )
    if args.task == "debug" and not args.data_path.is_file():
        raise FileNotFoundError(f"Debug data file does not exist: {args.data_path}")
    if not 0.0 < args.train_p < 1.0:
        raise ValueError("train_p must be between 0 and 1 (exclusive)")
    if args.cores < 1 or args.pop_size < 1 or args.n_folds < 2:
        raise ValueError(
            "cores/pop_size must be positive and n_folds must be at least 2"
        )
    if args.generations < 1:
        raise ValueError("generations must be positive")
    if args.max_eval_time_mins <= 0:
        raise ValueError("max_eval_time_mins must be positive")

    # The CLI follows the repository EA convention: `generations` counts only
    # post-initial generations. TPOT itself receives generations + 1 below.
    expected = args.pop_size * (args.generations + 1)
    if expected != args.n_evaluations:
        raise ValueError(
            "TPOT budget mismatch: population_size * (generations + 1) must "
            f"equal n_evaluations ({args.pop_size} * ({args.generations} + 1) "
            f"= {expected}, requested {args.n_evaluations})"
        )


def is_run_complete(output_directory: Path) -> bool:
    result_path = output_directory / "best_results.json"
    if not result_path.is_file():
        return False
    try:
        with result_path.open(encoding="utf-8") as file:
            result = json.load(file)
    except (OSError, ValueError):
        return False
    return result.get("search_space_revision") == SEARCH_SPACE_REVISION


@contextmanager
def acquire_run_lock(output_directory: Path):
    """Prevent duplicate jobs from writing the same replicate concurrently."""
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_directory.parent / f".{output_directory.name}.tpot.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                f"Another job is already running this replicate: {output_directory}",
                flush=True,
            )
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def reset_incomplete_run(output_directory: Path) -> None:
    """Remove runner-owned artifacts from an incomplete or failed replicate."""
    if not output_directory.exists():
        return
    if is_run_complete(output_directory):
        raise RuntimeError("Refusing to reset a completed TPOT replicate")

    owned_files = {
        "archive.json",
        "best_results.json",
        "best_pipeline.json",
        "best_pipeline.pkl",
        "checkpoints.csv",
        "configuration.json",
        "evaluation_budget.json",
        "failed.json",
        "fit_timing.json",
        "pareto_front.json",
        "preprocessor.pkl",
    }
    removed = []
    for name in owned_files:
        path = output_directory / name
        if path.is_file() or path.is_symlink():
            path.unlink()
            removed.append(path.name)
    for path in output_directory.glob("results_eval_*.json"):
        if path.is_file() or path.is_symlink():
            path.unlink()
            removed.append(path.name)

    checkpoint_directory = output_directory / "tpot_state"
    if checkpoint_directory.is_symlink():
        checkpoint_directory.unlink()
        removed.append(checkpoint_directory.name)
    elif checkpoint_directory.is_dir():
        shutil.rmtree(checkpoint_directory)
        removed.append(checkpoint_directory.name)

    if removed:
        print(
            f"Restarting incomplete replicate from scratch; removed "
            f"{len(removed)} TPOT artifact(s) from {output_directory}",
            flush=True,
        )


def load_spine_opioid_data(data_path: Path, y_label: str):
    if y_label not in OUTCOME_COLUMNS:
        raise ValueError(f"Invalid spine-opioid outcome {y_label!r}")
    X, outcomes = load_modeling_data(data_path)
    missing_outcomes = [
        column for column in OUTCOME_COLUMNS if column not in outcomes
    ]
    if missing_outcomes:
        raise ValueError(f"Missing outcome columns: {missing_outcomes}")
    missing_scaled = [column for column in SO_NUMERICAL_COLUMNS if column not in X]
    if missing_scaled:
        raise ValueError(f"Missing columns that must be scaled: {missing_scaled}")

    y = outcomes[y_label].to_numpy()
    return X, y, list(SO_NUMERICAL_COLUMNS), []


def load_generic_data(data_path: Path, y_label: str):
    data = pd.read_csv(data_path)
    if y_label not in data:
        raise ValueError(f"Target column {y_label!r} is missing from {data_path}")
    X = data.drop(columns=[y_label])
    return X, data[y_label].to_numpy(), list(X.columns), []


def make_preprocessor(
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> ColumnTransformer:
    """Match CASH: numericize/reorder only; leave scaling to the evolved stage."""
    transformers = []
    if numerical_columns:
        transformers.append(("num", "passthrough", list(numerical_columns)))
    if categorical_columns:
        transformers.append(
            (
                "cat",
                OneHotEncoder(
                    drop=None,
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                list(categorical_columns),
            )
        )
    return ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
    )


def prepare_data(
    X: pd.DataFrame,
    y: np.ndarray,
    train_p: float,
    n_folds: int,
    seed: int,
    classification: bool,
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_p,
        random_state=seed,
        shuffle=True,
        stratify=y if classification else None,
    )

    if classification:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_splits = list(splitter.split(X_train, y_train))
    fold_data = prepare_cv_folds(
        X_train=X_train,
        y_train=y_train,
        cv_splits=cv_splits,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
    )

    final_preprocessor = make_preprocessor(
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
    )
    X_train_transformed = final_preprocessor.fit_transform(X_train)
    X_test_transformed = final_preprocessor.transform(X_test)
    return (
        X_train,
        X_test,
        X_train_transformed,
        X_test_transformed,
        y_train,
        y_test,
        cv_splits,
        fold_data,
        final_preprocessor,
    )


def prepare_cv_folds(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_splits,
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
):
    """Fit the CASH base preprocessor independently on each CV training fold."""
    fold_data = []
    for train_indices, validation_indices in cv_splits:
        X_fold_train = X_train.iloc[train_indices].reset_index(drop=True)
        X_fold_validation = X_train.iloc[validation_indices].reset_index(drop=True)
        preprocessor = make_preprocessor(numerical_columns, categorical_columns)
        fold_data.append(
            (
                preprocessor.fit_transform(X_fold_train),
                preprocessor.transform(X_fold_validation),
                y_train[train_indices],
                y_train[validation_indices],
            )
        )
    return fold_data


def cross_validated_objectives(
    estimator,
    *,
    fold_data,
    classification: bool,
    classes: Optional[Sequence],
) -> np.ndarray:
    """Return mean validation performance and fitted complexity on CASH folds."""
    performance_scores = []
    complexity_scores = []
    for X_fold_train, X_fold_validation, y_fold_train, y_fold_validation in fold_data:
        fold_estimator = clone(estimator)
        fold_estimator.fit(X_fold_train, y_fold_train)
        performance_scores.append(
            performance_score(
                fold_estimator,
                X_fold_validation,
                y_fold_validation,
                classification,
                classes,
            )
        )
        complexity_scores.append(tpot.objectives.complexity_scorer(fold_estimator))
    return np.asarray(
        [np.mean(performance_scores), np.mean(complexity_scores)],
        dtype=float,
    )


def performance_score(
    estimator,
    X,
    y,
    classification: bool,
    classes: Optional[Sequence],
) -> float:
    if not classification:
        return float(r2_score(y, estimator.predict(X)))

    assert classes is not None
    scorer_name = "roc_auc" if len(classes) == 2 else "roc_auc_ovo"
    return float(get_scorer(scorer_name)(estimator, X, y))


def classification_metrics(
    estimator,
    X,
    y,
    classes: Sequence,
) -> dict[str, float]:
    """Calculate AUROC, AUPRC, and balanced accuracy for a fitted classifier."""
    classes = np.asarray(classes)
    probabilities = np.asarray(estimator.predict_proba(X))
    estimator_classes = np.asarray(getattr(estimator, "classes_", classes))
    column_indices = [
        int(np.flatnonzero(estimator_classes == label)[0]) for label in classes
    ]
    probabilities = probabilities[:, column_indices]

    if len(classes) == 2:
        positive_targets = (np.asarray(y) == classes[1]).astype(int)
        auprc = average_precision_score(positive_targets, probabilities[:, 1])
    else:
        indicator = np.column_stack(
            [(np.asarray(y) == label).astype(int) for label in classes]
        )
        auprc = average_precision_score(
            indicator,
            probabilities,
            average="macro",
        )

    return {
        "auroc": performance_score(estimator, X, y, True, classes),
        "auprc": float(auprc),
        "balanced_accuracy": float(balanced_accuracy_score(y, estimator.predict(X))),
    }


def cross_validated_classification_metrics(
    estimator,
    *,
    fold_data,
    classes: Sequence,
) -> dict[str, float]:
    """Evaluate classification diagnostics on the cached CASH folds."""
    fold_metrics = []
    for X_fold_train, X_fold_validation, y_fold_train, y_fold_validation in fold_data:
        fold_estimator = clone(estimator)
        fold_estimator.fit(X_fold_train, y_fold_train)
        fold_metrics.append(
            classification_metrics(
                fold_estimator,
                X_fold_validation,
                y_fold_validation,
                classes,
            )
        )
    return {
        metric: float(np.mean([values[metric] for values in fold_metrics]))
        for metric in ("auroc", "auprc", "balanced_accuracy")
    }


def json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def save_archive(evaluated_individuals: pd.DataFrame, output_directory: Path) -> None:
    records = []
    for index, row in evaluated_individuals.iterrows():
        record = {"individual_id": str(index)}
        for column, value in row.items():
            is_individual = column in {"Individual", "Instance"}
            record["individual" if is_individual else str(column)] = (
                repr(value) if is_individual else json_safe(value)
            )
        records.append(record)
    write_json(output_directory / "archive.json", records)


def select_best_row(
    history: pd.DataFrame,
    performance_objective: str,
    complexity_objective: str,
):
    ranked = history.copy()
    ranked[performance_objective] = pd.to_numeric(
        ranked[performance_objective], errors="coerce"
    )
    ranked[complexity_objective] = pd.to_numeric(
        ranked[complexity_objective], errors="coerce"
    )
    ranked = ranked.dropna(subset=[performance_objective, complexity_objective])
    if ranked.empty:
        raise ValueError("TPOT history contains no successfully evaluated pipelines")
    return ranked.sort_values(
        [performance_objective, complexity_objective],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]


def save_pareto_front(estimator, output_directory: Path) -> int:
    pareto = estimator.pareto_front
    if pareto is None:
        write_json(output_directory / "pareto_front.json", [])
        return 0

    records = []
    for index, row in pareto.iterrows():
        record = {"individual_id": str(index)}
        for column, value in row.items():
            record[str(column)] = (
                repr(value)
                if column in {"Individual", "Instance"}
                else json_safe(value)
            )
        records.append(record)
    write_json(output_directory / "pareto_front.json", records)
    return len(records)


def evaluation_error_counts(evaluated_individuals: pd.DataFrame) -> dict:
    if "Eval Error" not in evaluated_individuals:
        return {
            "available": False,
            "errored": None,
            "invalid": None,
            "timed_out": None,
        }

    def normalize(value) -> str:
        if value is None:
            return ""
        try:
            if bool(pd.isna(value)):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value).strip()

    values = evaluated_individuals["Eval Error"]
    normalized = values.map(normalize)
    errored = ~normalized.isin({"", "None"})
    upper = normalized.str.upper()
    return {
        "available": True,
        "errored": int(errored.sum()),
        "invalid": int((errored & upper.str.contains("INVALID", regex=False)).sum()),
        "timed_out": int((errored & upper.str.contains("TIMEOUT", regex=False)).sum()),
    }


def write_generation_checkpoints(
    evaluated_individuals: pd.DataFrame,
    output_directory: Path,
    performance_objective: str,
    complexity_objective: str,
    X_train,
    X_test,
    y_train,
    y_test,
    classification: bool,
    classes: Optional[Sequence],
    seed: int,
    task_id: str,
    metric_name: str,
    search_space_name: str,
) -> list[dict]:
    """Reconstruct best-so-far diagnostics at every completed TPOT generation."""
    individual_column = next(
        (
            column
            for column in ("Individual", "Instance")
            if column in evaluated_individuals.columns
        ),
        None,
    )
    if individual_column is None:
        raise ValueError("TPOT history has neither an Individual nor Instance column")

    required = {"Generation", performance_objective, complexity_objective}
    missing = required.difference(evaluated_individuals.columns)
    if missing:
        raise ValueError(
            f"TPOT history is missing checkpoint columns: {sorted(missing)}"
        )

    history = evaluated_individuals.copy()
    history["Generation"] = pd.to_numeric(history["Generation"], errors="coerce")
    history[performance_objective] = pd.to_numeric(
        history[performance_objective], errors="coerce"
    )
    history[complexity_objective] = pd.to_numeric(
        history[complexity_objective], errors="coerce"
    )
    generations = sorted(history["Generation"].dropna().unique())
    if not generations:
        raise ValueError("TPOT history contains no completed generations")

    checkpoints = []
    first_generation = int(generations[0])
    last_best_id = None
    last_train_score = None
    last_test_score = None

    for raw_generation in generations:
        seen = history.loc[history["Generation"] <= raw_generation]
        successful = seen.dropna(subset=[performance_objective, complexity_objective])
        if successful.empty:
            continue

        best_row = select_best_row(
            successful, performance_objective, complexity_objective
        )
        best_id = best_row.name
        if best_id != last_best_id:
            pipeline = best_row[individual_column].export_pipeline()
            pipeline.fit(X_train, y_train)
            last_train_score = performance_score(
                pipeline, X_train, y_train, classification, classes
            )
            last_test_score = performance_score(
                pipeline, X_test, y_test, classification, classes
            )
            last_best_id = best_id

        generation = int(raw_generation) - first_generation - 1
        hard_evals = int(len(seen))
        validation_score = float(best_row[performance_objective])
        complexity = float(best_row[complexity_objective])
        checkpoint = {
            "generation": generation,
            "candidates_considered": hard_evals,
            "hard_evals": hard_evals,
            "train_auc": last_train_score,
            "val_auc": validation_score,
            "test_auc": last_test_score,
            "complexity": complexity,
        }
        checkpoints.append(checkpoint)
        write_json(
            output_directory / f"results_eval_{hard_evals}.json",
            {
                "task_id": task_id,
                "model_type": f"TPOT_{search_space_name.upper()}",
                "seed": seed,
                "metric": metric_name,
                "train_accuracy": last_train_score,
                "validation_accuracy": validation_score,
                "test_accuracy": last_test_score,
                "complexity": complexity,
                "best_params": {"pipeline": repr(best_row[individual_column])},
            },
        )

    pd.DataFrame(checkpoints).to_csv(output_directory / "checkpoints.csv", index=False)
    return checkpoints


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    if is_run_complete(args.output_directory):
        print(f"Run already complete: {args.output_directory / 'best_results.json'}")
        return

    # TPOT 1.1.0 restarts its generation counter when loading population.pkl and
    # consequently runs the full configured budget again. Always discard partial
    # artifacts so every non-complete replicate starts with exactly one clean budget.
    reset_incomplete_run(args.output_directory)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    if args.task == "so":
        X, y, numerical_columns, categorical_columns = load_spine_opioid_data(
            args.data_path, args.y_label
        )
    else:
        X, y, numerical_columns, categorical_columns = load_generic_data(
            args.data_path, args.y_label
        )

    classes = np.unique(y) if args.classification else None
    if args.classification and len(classes) < 2:
        raise ValueError(f"Outcome {args.y_label!r} has fewer than two classes")

    (
        X_train_raw,
        X_test_raw,
        X_train_transformed,
        X_test_transformed,
        y_train,
        y_test,
        cv_splits,
        fold_data,
        base_preprocessor,
    ) = prepare_data(
        X,
        y,
        args.train_p,
        args.n_folds,
        args.seed,
        args.classification,
        numerical_columns,
        categorical_columns,
    )

    smallest_cv_train_size = min(len(train_indices) for train_indices, _ in cv_splits)
    n_classes = len(classes) if classes is not None else 0

    data_context = DataContext(
        n_samples=smallest_cv_train_size,
        n_features=X_train_raw.shape[1],
        n_classes=n_classes,
    )
    has_protected_columns = len(numerical_columns) < X_train_raw.shape[1]
    scale_columns = (
        tuple(range(len(numerical_columns))) if has_protected_columns else None
    )
    performance_scorer = (
        "roc_auc"
        if args.classification and len(classes) == 2
        else "roc_auc_ovo" if args.classification else "r2"
    )
    scorer_object = get_scorer(performance_scorer)
    performance_objective = scorer_object._score_func.__name__
    complexity_objective = "complexity_scorer"
    objective = partial(
        cross_validated_objectives,
        fold_data=fold_data,
        classification=args.classification,
        classes=classes,
    )
    objective.__name__ = "cash_cross_validated_objectives"
    metric_name = performance_scorer
    tpot_generations = args.generations + 1
    cv_splitter = (
        StratifiedKFold(
            n_splits=args.n_folds,
            shuffle=True,
            random_state=args.seed,
        )
        if args.classification
        else KFold(
            n_splits=args.n_folds,
            shuffle=True,
            random_state=args.seed,
        )
    )

    configuration = {
        "seed": args.seed,
        "task": args.task,
        "y_label": args.y_label,
        "data_path": str(args.data_path),
        "train_p": args.train_p,
        "output_directory": str(args.output_directory),
        "search_space": args.search_space,
        "search_space_revision": SEARCH_SPACE_REVISION,
        "operator_space": "Source.ML",
        "hyperparameter_space": "Source.ML",
        "topology": (
            "fixed_scaler_transformer_selector_predictor"
            if args.search_space == "constrained"
            else "flexible_dag"
        ),
        "classification": args.classification,
        "cores": args.cores,
        "population_size": args.pop_size,
        "evolution_generations": args.generations,
        "tpot_generations_including_initial": tpot_generations,
        "requested_evaluations": args.n_evaluations,
        "n_folds": args.n_folds,
        "cv_splitter": repr(cv_splitter),
        "evaluation_engine": "custom_other_objective_functions",
        "base_preprocessing": (
            "external_CASH_preprocessor_fit_on_each_outer_CV_training_fold"
        ),
        "preprocessing_in_candidate_pipeline": False,
        "reported_classification_metrics": [
            "auroc",
            "auprc",
            "balanced_accuracy",
        ],
        "max_eval_time_mins": args.max_eval_time_mins,
        "tpot_checkpointing": False,
        "incomplete_run_policy": "restart_from_scratch",
        "classes": classes,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "scale_columns": scale_columns,
        "objectives": {
            "performance_scorer": performance_scorer,
            "performance_history_column": performance_objective,
            "complexity": complexity_objective,
            "returns_performance_and_fitted_complexity": True,
            "weights": [1.0, -1.0],
        },
        "excluded_operators": ["PolynomialFeatures"],
        "tpot_version": getattr(tpot, "__version__", "unknown"),
    }
    write_json(args.output_directory / "configuration.json", configuration)
    print(json.dumps(json_safe(configuration), indent=2), flush=True)

    if args.search_space == "constrained":
        search_space = generate_tpot_search_space(
            data_context=data_context,
            seed=args.seed,
            classification=args.classification,
            scale_columns=scale_columns,
        )
    else:
        search_space = generate_tpot_graph_search_space(
            data_context=data_context,
            seed=args.seed,
            classification=args.classification,
            n_folds=args.n_folds,
            scale_columns=scale_columns,
        )

    estimator = tpot.TPOTEstimator(
        search_space=search_space,
        scorers=[],
        scorers_weights=[],
        other_objective_functions=[objective],
        other_objective_functions_weights=[1.0, -1.0],
        objective_function_names=[performance_objective, complexity_objective],
        cv=cv_splitter,
        population_size=args.pop_size,
        initial_population_size=args.pop_size,
        generations=tpot_generations,
        classification=args.classification,
        disable_label_encoder=args.classification,
        max_eval_time_mins=args.max_eval_time_mins,
        max_time_mins=None,
        n_jobs=args.cores,
        verbose=5,
        random_state=args.seed,
    )
    fit_started_at = time.time()
    fit_started_counter = time.perf_counter()
    fit_completed = False
    try:
        estimator.fit(X_train_transformed, y_train)
        fit_completed = True
    finally:
        fit_time_seconds = time.perf_counter() - fit_started_counter
        write_json(
            args.output_directory / "fit_timing.json",
            {
                "fit_completed": fit_completed,
                "fit_started_unix_seconds": fit_started_at,
                "fit_time_seconds": fit_time_seconds,
                "fit_time_minutes": fit_time_seconds / 60.0,
                "fit_time_hours": fit_time_seconds / 3600.0,
            },
        )

    # Keep preprocessing outside the candidate pipeline, matching the objective's
    # fold boundary. The outer training preprocessor was fit only on X_train_raw in
    # prepare_data; the selected candidate sees its transformed output.
    best_pipeline = clone(estimator.fitted_pipeline_)
    best_pipeline.fit(X_train_transformed, y_train)
    train_score = performance_score(
        best_pipeline, X_train_transformed, y_train, args.classification, classes
    )
    test_score = performance_score(
        best_pipeline, X_test_transformed, y_test, args.classification, classes
    )
    evaluated = estimator.evaluated_individuals
    requested_best_row = select_best_row(
        evaluated, performance_objective, complexity_objective
    )
    selected_row = estimator.selected_best_score
    selected_rank = (
        float(selected_row[performance_objective]),
        float(selected_row[complexity_objective]),
    )
    requested_rank = (
        float(requested_best_row[performance_objective]),
        float(requested_best_row[complexity_objective]),
    )
    if selected_rank != requested_rank:
        raise RuntimeError(
            "TPOT's fitted pipeline does not match the required AUROC-first, "
            "complexity-tiebreak selection rule"
        )
    validation_score = float(selected_row[performance_objective])
    selected_complexity = float(selected_row[complexity_objective])
    classification_diagnostics = None
    if args.classification:
        assert classes is not None
        validation_metrics = cross_validated_classification_metrics(
            estimator.fitted_pipeline_,
            fold_data=fold_data,
            classes=classes,
        )
        # The archived objective value remains the canonical validation AUROC.
        validation_metrics["auroc"] = validation_score
        classification_diagnostics = {
            "train": classification_metrics(
                best_pipeline,
                X_train_transformed,
                y_train,
                classes,
            ),
            "validation": validation_metrics,
            "test": classification_metrics(
                best_pipeline,
                X_test_transformed,
                y_test,
                classes,
            ),
        }
    error_counts = evaluation_error_counts(evaluated)
    actual_evaluations = len(evaluated)

    save_archive(evaluated, args.output_directory)
    pareto_front_size = save_pareto_front(estimator, args.output_directory)
    write_json(
        args.output_directory / "evaluation_budget.json",
        {
            "requested_evaluations": args.n_evaluations,
            "actual_evaluations": actual_evaluations,
            "exact_budget_met": actual_evaluations == args.n_evaluations,
        },
    )
    if actual_evaluations != args.n_evaluations:
        raise RuntimeError(
            "TPOT did not perform exactly the requested number of distinct pipeline "
            f"evaluations: expected {args.n_evaluations}, observed {actual_evaluations}"
        )
    checkpoints = write_generation_checkpoints(
        evaluated_individuals=evaluated,
        output_directory=args.output_directory,
        performance_objective=performance_objective,
        complexity_objective=complexity_objective,
        X_train=X_train_transformed,
        X_test=X_test_transformed,
        y_train=y_train,
        y_test=y_test,
        classification=args.classification,
        classes=classes,
        seed=args.seed,
        task_id=args.y_label,
        metric_name=metric_name,
        search_space_name=args.search_space,
    )

    with (args.output_directory / "best_pipeline.pkl").open("wb") as file:
        pickle.dump(best_pipeline, file)
    with (args.output_directory / "preprocessor.pkl").open("wb") as file:
        pickle.dump(base_preprocessor, file)

    pipeline_record = {
        "pipeline": repr(best_pipeline),
        "parameters": best_pipeline.get_params(deep=True),
        "expects_preprocessed_input": True,
        "preprocessor_artifact": "preprocessor.pkl",
        "selected_individual": repr(selected_row.get("Individual")),
        "validation_score": validation_score,
        "complexity": selected_complexity,
        "classification_metrics": classification_diagnostics,
    }
    write_json(args.output_directory / "best_pipeline.json", pipeline_record)

    result = {
        "task_id": args.y_label,
        "model_type": f"TPOT_{args.search_space.upper()}",
        "search_space": args.search_space,
        "search_space_revision": SEARCH_SPACE_REVISION,
        "operator_space": "Source.ML",
        "hyperparameter_space": "Source.ML",
        "seed": args.seed,
        "metric": metric_name,
        "train_accuracy": train_score,
        "validation_accuracy": validation_score,
        "test_accuracy": test_score,
        "classification_metrics": classification_diagnostics,
        "complexity": selected_complexity,
        "best_params": json_safe(best_pipeline.get_params(deep=True)),
        "pipeline": repr(best_pipeline),
        "expects_preprocessed_input": True,
        "preprocessor_artifact": "preprocessor.pkl",
        "selected_individual": repr(selected_row.get("Individual")),
        "num_evaluated_individuals": actual_evaluations,
        "evaluation_errors": error_counts,
        "requested_evaluations": args.n_evaluations,
        "num_checkpoints": len(checkpoints),
        "pareto_front_size": pareto_front_size,
        "fit_time_seconds": fit_time_seconds,
        "fit_time_minutes": fit_time_seconds / 60.0,
        "tpot_version": getattr(tpot, "__version__", "unknown"),
    }
    if classification_diagnostics is not None:
        for split_name, split_metrics in classification_diagnostics.items():
            for metric_name, metric_value in split_metrics.items():
                result[f"{split_name}_{metric_name}"] = metric_value
    # Written last: this is the completion marker used on resubmission.
    write_json(args.output_directory / "best_results.json", result)
    print(json.dumps(json_safe(result), indent=2), flush=True)


def main() -> None:
    args = parse_args()
    with acquire_run_lock(args.output_directory) as acquired:
        if not acquired:
            return
        try:
            run(args)
        except Exception as error:
            args.output_directory.mkdir(parents=True, exist_ok=True)
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
