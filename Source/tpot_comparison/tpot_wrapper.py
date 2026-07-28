import argparse
import os
import pickle
import random
import time
import traceback
from functools import partial
from typing import List

import numpy as np
import pandas as pd
import tpot
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from Source.tpot_comparison.param_space_conversion import generate_tpot_search_space


def load_task_data(task_id, data_directory, train_size, seed):
    """Load and split a dataset using the current CASH data format."""
    summary = pd.read_csv(os.path.join(data_directory, "tasks_summary.csv"))
    task_summary = summary.loc[summary["task_id"] == task_id]
    if task_summary.empty:
        raise ValueError(f"Task {task_id} is missing from tasks_summary.csv")

    data = pd.read_csv(os.path.join(data_directory, f"task_{task_id}.csv"))
    target_name = (
        task_summary.iloc[0]["target_name"]
        if "target_name" in task_summary
        else data.columns[-1]
    )
    X = data.drop(columns=[target_name])
    y = data[target_name].to_numpy()
    classes = (
        int(task_summary.iloc[0]["num_classes"])
        if "num_classes" in task_summary
        else len(np.unique(y))
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )

    indicator_path = os.path.join(
        data_directory, f"task_{task_id}_categorical_indicator.pkl"
    )
    if os.path.exists(indicator_path):
        with open(indicator_path, "rb") as file:
            categorical_indicator = pickle.load(file)
    else:
        categorical_indicator = [
            isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            for dtype in X.dtypes
        ]
    return X_train, X_test, y_train, y_test, categorical_indicator, classes


def make_preprocessor(X, categorical_indicator):
    categorical_cols = [col for col, is_cat in zip(X.columns, categorical_indicator) if is_cat]
    numerical_cols = [col for col, is_cat in zip(X.columns, categorical_indicator) if not is_cat]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough",
    )


def get_splits(X_train, y_train, categorical_indicator, seed):
    """Create the same preprocessed five-fold splits used by current CASH."""
    splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train, y_train)
    X_train_splits, X_val_splits = [], []
    y_train_splits, y_val_splits = [], []

    for train_indices, val_indices in splits:
        X_fold_train = X_train.iloc[train_indices].reset_index(drop=True)
        X_fold_val = X_train.iloc[val_indices].reset_index(drop=True)
        preprocessor = make_preprocessor(X_fold_train, categorical_indicator)
        X_train_splits.append(preprocessor.fit_transform(X_fold_train))
        X_val_splits.append(preprocessor.transform(X_fold_val))
        y_train_splits.append(y_train[train_indices])
        y_val_splits.append(y_train[val_indices])

    return X_train_splits, X_val_splits, y_train_splits, y_val_splits


def objective_score(estimator, X, y, classes):
    probabilities = estimator.predict_proba(X)
    if classes == 2:
        return roc_auc_score(y, probabilities[:, 1])
    return -log_loss(y, probabilities, labels=np.arange(classes))


def custom_objective_function(
    estimator,
    X_train_splits,
    X_val_splits,
    y_train_splits,
    y_val_splits,
    classes,
):
    scores = []
    for X_train, X_val, y_train, y_val in zip(
        X_train_splits, X_val_splits, y_train_splits, y_val_splits
    ):
        fold_pipeline = clone(estimator)
        fold_pipeline.fit(X_train, y_train)
        score = objective_score(fold_pipeline, X_val, y_val, classes)
        complexity = tpot.objectives.complexity_scorer(fold_pipeline)
        scores.append([score, complexity])

    return np.mean(scores, axis=0)


def tpot_loop_through_tasks(
    taskids: List[int],
    data_directory: str,
    base_save_folder: str,
    num_reps: int,
    ga_params: dict,
    train_size: float = 0.8,
):
    for task_index, task_id in enumerate(taskids):
        for rep in range(num_reps):
            save_folder = os.path.join(base_save_folder, str(task_id), f"Rep_{rep}")
            time.sleep(random.random() * 5)
            if os.path.exists(save_folder):
                continue
            os.makedirs(save_folder)

            seed = task_index * 1000 + rep
            print(f"Working on {save_folder}")
            print(f"Super Seed: {seed}")

            try:
                X_train, X_test, y_train, y_test, categorical_indicator, classes = load_task_data(
                    task_id, data_directory, train_size, seed
                )
                X_train_splits, X_val_splits, y_train_splits, y_val_splits = get_splits(
                    X_train, y_train, categorical_indicator, seed
                )
                custom_objective = partial(
                    custom_objective_function,
                    X_train_splits=X_train_splits,
                    X_val_splits=X_val_splits,
                    y_train_splits=y_train_splits,
                    y_val_splits=y_val_splits,
                    classes=classes,
                )
                custom_objective.__name__ = "custom_objective"

                estimator = tpot.TPOTEstimator(
                    search_space=generate_tpot_search_space(classes, ga_params["n_jobs"]),
                    scorers=[],
                    scorers_weights=[],
                    other_objective_functions=[custom_objective],
                    other_objective_functions_weights=[1, -1],
                    objective_function_names=["score", "complexity"],
                    population_size=ga_params["population_size"],
                    generations=ga_params["generations"],
                    classification=True,
                    disable_label_encoder=True,
                    max_eval_time_mins=1440,
                    max_time_mins=None,
                    n_jobs=ga_params["n_jobs"],
                    verbose=5,
                    random_state=seed,
                )

                preprocessor = make_preprocessor(X_train, categorical_indicator)
                X_train_transformed = preprocessor.fit_transform(X_train)
                X_test_transformed = preprocessor.transform(X_test)
                estimator.fit(X_train_transformed, y_train)

                best_pipeline = estimator.fitted_pipeline_
                train_score = objective_score(best_pipeline, X_train_transformed, y_train, classes)
                test_score = objective_score(best_pipeline, X_test_transformed, y_test, classes)
                evaluated_individuals = estimator.evaluated_individuals

                classifier_counts = {}
                for pipeline in evaluated_individuals["Instance"]:
                    classifier_name = pipeline.__class__.__name__
                    classifier_counts[classifier_name] = classifier_counts.get(classifier_name, 0) + 1

                result = {
                    "train_score": train_score,
                    "test_score": test_score,
                    "metric": "roc_auc" if classes == 2 else "negative_log_loss",
                    "complexity": tpot.objectives.complexity_scorer(best_pipeline),
                    "pipeline": best_pipeline,
                    "seed": seed,
                    "run": rep,
                    "classifier_distribution": classifier_counts,
                    "num_evaluated_individuals": len(evaluated_individuals),
                }
                with open(os.path.join(save_folder, "scores.pkl"), "wb") as file:
                    pickle.dump(result, file)
                return

            except Exception as error:
                failure = {
                    "error": str(error),
                    "trace": traceback.format_exc(),
                    "seed": seed,
                    "run": rep,
                }
                print(f"Failed on {save_folder}")
                print(failure["trace"])
                with open(os.path.join(save_folder, "failed.pkl"), "wb") as file:
                    pickle.dump(failure, file)
                return

    print("all finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TPOT CASH comparison")
    parser.add_argument("--num_cpus", type=int, default=4, help="Number of CPUs to use")
    parser.add_argument("--machine", type=str, default="local", help="Machine identifier")
    parser.add_argument("--num_reps", type=int, default=10, help="Number of replicates")
    parser.add_argument("--train_size", type=float, default=0.8, help="Training proportion")
    parser.add_argument("--data_directory", type=str, default=None, help="Dataset directory")
    args = parser.parse_args()

    assert args.num_cpus > 0
    assert args.num_reps > 0
    assert 0 < args.train_size < 1

    ga_params = {
        "population_size": 100 if args.machine == "remote" else 10,
        "generations": 20 if args.machine == "remote" else 5,
        "n_jobs": args.num_cpus,
    }
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_directories = [
        os.path.join(project_root, "Data", "Raw_OpenML_Suite_271_Classification"),
        os.path.join(project_root, "Data", "Raw_OpenML_Suite_271_Binary_Classification"),
    ]
    data_directory = args.data_directory or next(
        (path for path in data_directories if os.path.exists(path)), None
    )
    base_save_folder = os.path.join(project_root, "Results", "tpot")
    assert data_directory is not None and os.path.exists(data_directory), (
        "Data directory does not exist. Pass it with --data_directory."
    )

    summary = pd.read_csv(os.path.join(data_directory, "tasks_summary.csv"))
    taskids = summary["task_id"].astype(int).tolist()
    print(
        f"Starting TPOT experiments with {args.num_cpus} CPUs, "
        f"{args.num_reps} replicates on {args.machine}"
    )
    tpot_loop_through_tasks(
        taskids,
        data_directory,
        base_save_folder,
        args.num_reps,
        ga_params,
        args.train_size,
    )
