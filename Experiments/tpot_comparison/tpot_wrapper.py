"""Run one TPOT experiment over the project's full linear CASH space."""

import argparse
import json
import os
import pickle
import sys
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import tpot
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

# SLURM invokes this file directly, so make repository imports independent of cwd.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    from .param_space_conversion import generate_tpot_search_space
except ImportError:  # direct execution
    from param_space_conversion import generate_tpot_search_space

from Source.Base.model_param_space import DataContext


OUTCOME_COLUMNS = ("LOS_extended", "discharge_Home", "HOSP_READM_90")
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
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--train_p", type=float, required=True)
    parser.add_argument("--output_directory", type=Path, required=True)
    parser.add_argument("--classification", type=parse_bool, default=True)
    parser.add_argument("--cores", type=int, required=True)
    parser.add_argument("--pop_size", type=int, required=True)
    parser.add_argument("--generations", type=int, required=True)
    parser.add_argument("--n_evaluations", type=int, default=500)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--max_eval_time_mins", type=float, default=60.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_path.is_file():
        raise FileNotFoundError(f"Data file does not exist: {args.data_path}")
    if not 0.0 < args.train_p < 1.0:
        raise ValueError("train_p must be between 0 and 1 (exclusive)")
    if args.cores < 1 or args.pop_size < 1 or args.n_folds < 2:
        raise ValueError("cores/pop_size must be positive and n_folds must be at least 2")
    if args.generations < 1:
        raise ValueError("generations must be positive")

    # TPOT includes its initial population in this generation count. This differs
    # from Source.HPO.EA, whose `gens` counts only post-initial generations.
    expected = args.pop_size * args.generations
    if expected != args.n_evaluations:
        raise ValueError(
            "TPOT budget mismatch: population_size * generations must "
            f"equal n_evaluations ({args.pop_size} * {args.generations} "
            f"= {expected}, requested {args.n_evaluations})"
        )


def is_run_complete(output_directory: Path) -> bool:
    return (output_directory / "best_results.json").is_file()


def load_spine_opioid_data(data_path: Path, y_label: str):
    if y_label not in OUTCOME_COLUMNS:
        raise ValueError(f"Invalid spine-opioid outcome {y_label!r}")
    data = pd.read_csv(data_path)
    missing_outcomes = [column for column in OUTCOME_COLUMNS if column not in data]
    if missing_outcomes:
        raise ValueError(f"Missing outcome columns: {missing_outcomes}")
    missing_scaled = [column for column in SO_NUMERICAL_COLUMNS if column not in data]
    if missing_scaled:
        raise ValueError(f"Missing columns that must be scaled: {missing_scaled}")

    other_outcomes = [column for column in OUTCOME_COLUMNS if column != y_label]
    data = data.drop(columns=other_outcomes)
    X = data.drop(columns=[y_label])
    y = data[y_label].to_numpy()
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
    fold_data = []
    for train_indices, validation_indices in splitter.split(X_train, y_train):
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

    final_preprocessor = make_preprocessor(numerical_columns, categorical_columns)
    X_train_transformed = final_preprocessor.fit_transform(X_train)
    X_test_transformed = final_preprocessor.transform(X_test)
    return (
        X_train,
        X_train_transformed,
        X_test_transformed,
        y_train,
        y_test,
        fold_data,
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
    probabilities = estimator.predict_proba(X)
    estimator_classes = np.asarray(estimator.classes_)
    if len(classes) == 2:
        positive_class = classes[-1]
        positive_index = int(np.flatnonzero(estimator_classes == positive_class)[0])
        return float(roc_auc_score(y, probabilities[:, positive_index]))
    return float(
        roc_auc_score(
            y,
            probabilities,
            multi_class="ovo",
            labels=classes,
        )
    )


def cross_validated_score(
    estimator,
    *,
    fold_data: Iterable,
    classification: bool,
    classes: Optional[Sequence],
) -> float:
    scores = []
    for X_train, X_validation, y_train, y_validation in fold_data:
        fold_estimator = clone(estimator)
        fold_estimator.fit(X_train, y_train)
        scores.append(
            performance_score(
                fold_estimator,
                X_validation,
                y_validation,
                classification,
                classes,
            )
        )
    return float(np.mean(scores))


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


def write_generation_checkpoints(
    evaluated_individuals: pd.DataFrame,
    output_directory: Path,
    objective_name: str,
    X_train,
    X_test,
    y_train,
    y_test,
    classification: bool,
    classes: Optional[Sequence],
    seed: int,
    task_id: str,
    metric_name: str,
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

    required = {"Generation", objective_name}
    missing = required.difference(evaluated_individuals.columns)
    if missing:
        raise ValueError(f"TPOT history is missing checkpoint columns: {sorted(missing)}")

    history = evaluated_individuals.copy()
    history["Generation"] = pd.to_numeric(history["Generation"], errors="coerce")
    history[objective_name] = pd.to_numeric(history[objective_name], errors="coerce")
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
        successful = seen.dropna(subset=[objective_name])
        if successful.empty:
            continue

        best_id = successful[objective_name].idxmax()
        best_row = successful.loc[best_id]
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
        validation_score = float(best_row[objective_name])
        checkpoint = {
            "generation": generation,
            "candidates_considered": hard_evals,
            "hard_evals": hard_evals,
            "train_auc": last_train_score,
            "val_auc": validation_score,
            "test_auc": last_test_score,
        }
        checkpoints.append(checkpoint)
        write_json(
            output_directory / f"results_eval_{hard_evals}.json",
            {
                "task_id": task_id,
                "model_type": "TPOT_CASH",
                "seed": seed,
                "metric": metric_name,
                "train_accuracy": last_train_score,
                "validation_accuracy": validation_score,
                "test_accuracy": last_test_score,
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
        X_train,
        X_test,
        y_train,
        y_test,
        fold_data,
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

    smallest_cv_train_size = min(len(fold[2]) for fold in fold_data)
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
    objective_name = "validation_auc" if args.classification else "validation_r2"
    metric_name = "roc_auc" if args.classification else "r2"
    objective = partial(
        cross_validated_score,
        fold_data=fold_data,
        classification=args.classification,
        classes=classes,
    )
    objective.__name__ = objective_name

    configuration = {
        "seed": args.seed,
        "task": args.task,
        "y_label": args.y_label,
        "data_path": str(args.data_path),
        "train_p": args.train_p,
        "output_directory": str(args.output_directory),
        "classification": args.classification,
        "cores": args.cores,
        "population_size": args.pop_size,
        "generations": args.generations,
        "requested_evaluations": args.n_evaluations,
        "n_folds": args.n_folds,
        "max_eval_time_mins": args.max_eval_time_mins,
        "classes": classes,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "scale_columns": scale_columns,
    }
    write_json(args.output_directory / "configuration.json", configuration)
    print(json.dumps(json_safe(configuration), indent=2), flush=True)

    checkpoint_directory = args.output_directory / "tpot_state"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    estimator = tpot.TPOTEstimator(
        search_space=generate_tpot_search_space(
            data_context=data_context,
            seed=args.seed,
            classification=args.classification,
            scale_columns=scale_columns,
        ),
        scorers=[],
        scorers_weights=[],
        other_objective_functions=[objective],
        other_objective_functions_weights=[1.0],
        objective_function_names=[objective_name],
        population_size=args.pop_size,
        initial_population_size=args.pop_size,
        generations=args.generations,
        classification=args.classification,
        disable_label_encoder=args.classification,
        max_eval_time_mins=args.max_eval_time_mins,
        max_time_mins=None,
        n_jobs=args.cores,
        periodic_checkpoint_folder=str(checkpoint_directory),
        verbose=5,
        random_state=args.seed,
    )
    estimator.fit(X_train, y_train)

    best_pipeline = estimator.fitted_pipeline_
    train_score = performance_score(
        best_pipeline, X_train, y_train, args.classification, classes
    )
    test_score = performance_score(
        best_pipeline, X_test, y_test, args.classification, classes
    )
    evaluated = estimator.evaluated_individuals
    validation_scores = pd.to_numeric(evaluated[objective_name], errors="coerce")
    validation_score = float(validation_scores.max())

    save_archive(evaluated, args.output_directory)
    checkpoints = write_generation_checkpoints(
        evaluated_individuals=evaluated,
        output_directory=args.output_directory,
        objective_name=objective_name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        classification=args.classification,
        classes=classes,
        seed=args.seed,
        task_id=args.y_label,
        metric_name=metric_name,
    )

    with (args.output_directory / "best_pipeline.pkl").open("wb") as file:
        pickle.dump(best_pipeline, file)

    result = {
        "task_id": args.y_label,
        "model_type": "TPOT_CASH",
        "seed": args.seed,
        "metric": metric_name,
        "train_accuracy": train_score,
        "validation_accuracy": validation_score,
        "test_accuracy": test_score,
        "best_params": json_safe(best_pipeline.get_params(deep=True)),
        "pipeline": repr(best_pipeline),
        "num_evaluated_individuals": len(evaluated),
        "requested_evaluations": args.n_evaluations,
        "num_checkpoints": len(checkpoints),
    }
    # Written last: this is the completion marker used on resubmission.
    write_json(args.output_directory / "best_results.json", result)
    print(json.dumps(json_safe(result), indent=2), flush=True)


def main() -> None:
    args = parse_args()
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
