"""Summarize TPOT result files across populations, tasks, and seeds."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TASKS = ("LOS_extended", "discharge_Home", "HOSP_READM_90")


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize TPOT CASH results")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--pop-sizes", type=parse_int_list, default=[25, 50, 100])
    parser.add_argument("--seeds", type=parse_int_list, default=list(range(21)))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def run_directory(results_root: Path, pop_size: int, task: str, seed: int) -> Path:
    return (
        results_root
        / f"Pop{pop_size}"
        / "TPOT_CASH"
        / f"Task_{task}"
        / f"Seed_{seed}"
    )


def collect_results(
    results_root: Path,
    pop_sizes: list[int],
    tasks: list[str],
    seeds: list[int],
) -> pd.DataFrame:
    rows = []
    for pop_size in pop_sizes:
        for task in tasks:
            for seed in seeds:
                directory = run_directory(results_root, pop_size, task, seed)
                result_path = directory / "best_results.json"
                failure_path = directory / "failed.json"
                row = {
                    "population_size": pop_size,
                    "task": task,
                    "seed": seed,
                    "status": "missing",
                    "train_auc": np.nan,
                    "validation_auc": np.nan,
                    "test_auc": np.nan,
                    "error": None,
                }
                if result_path.is_file():
                    with result_path.open(encoding="utf-8") as file:
                        result = json.load(file)
                    row.update(
                        status="success",
                        train_auc=result.get("train_accuracy", np.nan),
                        validation_auc=result.get("validation_accuracy", np.nan),
                        test_auc=result.get("test_accuracy", np.nan),
                    )
                elif failure_path.is_file():
                    with failure_path.open(encoding="utf-8") as file:
                        failure = json.load(file)
                    row.update(status="failed", error=failure.get("error"))
                rows.append(row)
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for (pop_size, task), group in results.groupby(
        ["population_size", "task"], sort=True
    ):
        successful = group.loc[group["status"] == "success"]
        summary_rows.append(
            {
                "population_size": pop_size,
                "task": task,
                "successes": int((group["status"] == "success").sum()),
                "failures": int((group["status"] == "failed").sum()),
                "missing": int((group["status"] == "missing").sum()),
                "validation_auc_mean": successful["validation_auc"].mean(),
                "validation_auc_std": successful["validation_auc"].std(ddof=1),
                "test_auc_mean": successful["test_auc"].mean(),
                "test_auc_std": successful["test_auc"].std(ddof=1),
                "test_auc_median": successful["test_auc"].median(),
            }
        )
    return pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    results = collect_results(
        args.results_root,
        args.pop_sizes,
        args.tasks,
        args.seeds,
    )
    summary = summarize(results)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    failed = results.loc[results["status"] == "failed"]
    if not failed.empty:
        print("\nFailures:")
        print(failed[["population_size", "task", "seed", "error"]].to_string(index=False))

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output_csv, index=False)
        print(f"\nWrote {args.output_csv}")


if __name__ == "__main__":
    main()
