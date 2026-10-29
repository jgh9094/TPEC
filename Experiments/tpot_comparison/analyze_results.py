"""Summarize TPOT results and plot the two search-space distributions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TASKS = ("LOS_extended", "discharge_Home", "HOSP_READM_90")
DEFAULT_SEARCH_SPACES = ("Constrained", "Graph")
DEFAULT_RUN_SEARCH_SPACES = ("Constrained_100", "Graph_100", *DEFAULT_SEARCH_SPACES)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = (
    REPOSITORY_ROOT / "Results_TPOT_SpinalSurgery_CustomCVObjective"
)
DEFAULT_PLOT_OUTPUT = Path(__file__).resolve().parent / "tpot_half_violin.png"
DEFAULT_RUN_OUTPUT = Path(__file__).resolve().parent / "tpot_run_summary.csv"
RUN_OUTPUT_COLUMNS = (
    "evals",
    "condition",
    "task",
    "seed",
    "train_auroc",
    "validation_auroc",
    "test_auroc",
    "complexity",
    "best_eval",
    "best_generation",
    "total_runtime_minutes",
)
PLOT_METRICS = {
    "test_score": "Test AUROC",
    "validation_score": "Cross-validation AUROC",
    "train_score": "Training AUROC",
    "test_auprc": "Test AUPRC",
    "validation_auprc": "Cross-validation AUPRC",
    "train_auprc": "Training AUPRC",
    "test_balanced_accuracy": "Test balanced accuracy",
    "validation_balanced_accuracy": "Cross-validation balanced accuracy",
    "train_balanced_accuracy": "Training balanced accuracy",
    "complexity": "TPOT complexity score",
    "fit_time_minutes": "TPOT fit time (minutes)",
    "errored_pipelines": "Errored pipelines",
}
TASK_LABELS = {
    "LOS_extended": "Extended LOS",
    "discharge_Home": "Discharge home",
    "HOSP_READM_90": "90-day readmission",
}


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize TPOT comparison results")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--pop-size", type=int, default=25)
    parser.add_argument(
        "--search-spaces", nargs="+", default=list(DEFAULT_SEARCH_SPACES)
    )
    parser.add_argument(
        "--run-search-spaces",
        nargs="+",
        default=list(DEFAULT_RUN_SEARCH_SPACES),
        help=(
            "Result directories included in the per-run CSV. By default this includes "
            "the 100-evaluation *_100 directories and the 500-evaluation directories."
        ),
    )
    parser.add_argument("--seeds", type=parse_int_list, default=list(range(1, 22)))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--run-output-csv",
        type=Path,
        default=DEFAULT_RUN_OUTPUT,
        help=(
            "Output path for the one-row-per-completed-replicate CSV "
            f"(default: {DEFAULT_RUN_OUTPUT})."
        ),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=DEFAULT_PLOT_OUTPUT,
        help="Output path for the single split half-violin figure.",
    )
    parser.add_argument(
        "--plot-metric",
        choices=sorted(PLOT_METRICS),
        default="test_score",
        help="Quantity shown in the half-violin plot (default: test_score).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Print summaries without producing an image.",
    )
    return parser.parse_args()


def run_directory(
    results_root: Path, pop_size: int, search_space: str, task: str, seed: int
) -> Path:
    return (
        results_root / f"Pop{pop_size}" / search_space / f"Task_{task}" / f"Seed_{seed}"
    )


def _numeric(value, default=np.nan):
    """Convert a JSON/CSV scalar to a finite float without raising."""
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if np.isfinite(converted) else default


def _same_number(left, right) -> bool:
    left_number = _numeric(left)
    right_number = _numeric(right)
    return bool(
        np.isfinite(left_number)
        and np.isfinite(right_number)
        and np.isclose(left_number, right_number, rtol=1e-9, atol=1e-12)
    )


def _evaluation_number(record: dict, position: int) -> int:
    """Return TPOT's one-based candidate number, falling back to archive order."""
    try:
        return int(record.get("individual_id")) + 1
    except (TypeError, ValueError):
        return position + 1


def find_best_location(directory: Path, result: dict) -> tuple[float, float]:
    """Locate the final selected candidate in TPOT's archive.

    TPOT numbers its initial population as generation 0. The requested CSV numbers it as
    generation 1, so archive generations are shifted up by one.
    """
    archive_path = directory / "archive.json"
    configuration_path = directory / "configuration.json"
    performance_column = "roc_auc_score"
    complexity_column = "complexity_scorer"
    if configuration_path.is_file():
        with configuration_path.open(encoding="utf-8") as file:
            configuration = json.load(file)
        objectives = configuration.get("objectives", {})
        performance_column = objectives.get(
            "performance_history_column", performance_column
        )
        complexity_column = objectives.get("complexity", complexity_column)

    selected_performance = result.get(
        "validation_auroc", result.get("validation_accuracy")
    )
    selected_complexity = result.get("complexity")
    if archive_path.is_file():
        with archive_path.open(encoding="utf-8") as file:
            archive = json.load(file)

        successful = [
            (position, record)
            for position, record in enumerate(archive)
            if np.isfinite(_numeric(record.get(performance_column)))
            and np.isfinite(_numeric(record.get(complexity_column)))
        ]
        matching = [
            item
            for item in successful
            if _same_number(item[1].get(performance_column), selected_performance)
            and _same_number(item[1].get(complexity_column), selected_complexity)
        ]
        if matching:
            position, best = min(
                matching,
                key=lambda item: _evaluation_number(item[1], item[0]),
            )
        elif successful:
            position, best = min(
                successful,
                key=lambda item: (
                    -_numeric(item[1].get(performance_column)),
                    _numeric(item[1].get(complexity_column)),
                    _evaluation_number(item[1], item[0]),
                ),
            )
        else:
            best = None

        if best is not None:
            raw_generation = _numeric(best.get("Generation"))
            generation = raw_generation + 1 if np.isfinite(raw_generation) else np.nan
            return _evaluation_number(best, position), generation

    # Older result folders may have checkpoints but no archive. In that case the
    # checkpoint boundary is the best available approximation of the exact evaluation.
    checkpoints_path = directory / "checkpoints.csv"
    if checkpoints_path.is_file():
        checkpoints = pd.read_csv(checkpoints_path)
        matches = checkpoints.loc[
            np.isclose(
                pd.to_numeric(checkpoints.get("val_auc"), errors="coerce"),
                _numeric(selected_performance),
                rtol=1e-9,
                atol=1e-12,
            )
            & np.isclose(
                pd.to_numeric(checkpoints.get("complexity"), errors="coerce"),
                _numeric(selected_complexity),
                rtol=1e-9,
                atol=1e-12,
            )
        ]
        if not matches.empty:
            first = matches.iloc[0]
            return (
                _numeric(first.get("hard_evals", first.get("candidates_considered"))),
                _numeric(first.get("generation")) + 2,
            )

    return np.nan, np.nan


def resolve_evaluation_count(directory: Path, result: dict) -> float:
    """Resolve the actual number of evaluated candidates with artifact fallbacks."""
    for key in ("num_evaluated_individuals", "actual_evaluations"):
        value = _numeric(result.get(key))
        if np.isfinite(value):
            return value

    budget_path = directory / "evaluation_budget.json"
    if budget_path.is_file():
        with budget_path.open(encoding="utf-8") as file:
            budget = json.load(file)
        value = _numeric(budget.get("actual_evaluations"))
        if np.isfinite(value):
            return value

    return _numeric(result.get("requested_evaluations"))


def resolve_runtime_minutes(directory: Path, result: dict) -> float:
    """Use the TPOT fit wall time recorded by the runner."""
    runtime = _numeric(result.get("fit_time_minutes"))
    if np.isfinite(runtime):
        return runtime
    timing_path = directory / "fit_timing.json"
    if timing_path.is_file():
        with timing_path.open(encoding="utf-8") as file:
            timing = json.load(file)
        return _numeric(timing.get("fit_time_minutes"))
    return np.nan


def collect_results(
    results_root: Path,
    pop_size: int,
    search_spaces: list[str],
    tasks: list[str],
    seeds: list[int],
) -> pd.DataFrame:
    rows = []
    for search_space in search_spaces:
        for task in tasks:
            for seed in seeds:
                directory = run_directory(
                    results_root, pop_size, search_space, task, seed
                )
                result_path = directory / "best_results.json"
                failure_path = directory / "failed.json"
                row = {
                    "population_size": pop_size,
                    "search_space": search_space,
                    "task": task,
                    "seed": seed,
                    "status": "missing",
                    "metric": None,
                    "train_score": np.nan,
                    "validation_score": np.nan,
                    "test_score": np.nan,
                    "train_auprc": np.nan,
                    "validation_auprc": np.nan,
                    "test_auprc": np.nan,
                    "train_balanced_accuracy": np.nan,
                    "validation_balanced_accuracy": np.nan,
                    "test_balanced_accuracy": np.nan,
                    "complexity": np.nan,
                    "fit_time_minutes": np.nan,
                    "errored_pipelines": np.nan,
                    "evals": np.nan,
                    "best_eval": np.nan,
                    "best_generation": np.nan,
                    "total_runtime_minutes": np.nan,
                    "error": None,
                }
                if result_path.is_file():
                    with result_path.open(encoding="utf-8") as file:
                        result = json.load(file)
                    best_eval, best_generation = find_best_location(directory, result)
                    row.update(
                        status="success",
                        metric=result.get("metric"),
                        train_score=result.get(
                            "train_auroc", result.get("train_accuracy", np.nan)
                        ),
                        validation_score=result.get(
                            "validation_auroc",
                            result.get("validation_accuracy", np.nan),
                        ),
                        test_score=result.get(
                            "test_auroc", result.get("test_accuracy", np.nan)
                        ),
                        train_auprc=result.get("train_auprc", np.nan),
                        validation_auprc=result.get("validation_auprc", np.nan),
                        test_auprc=result.get("test_auprc", np.nan),
                        train_balanced_accuracy=result.get(
                            "train_balanced_accuracy", np.nan
                        ),
                        validation_balanced_accuracy=result.get(
                            "validation_balanced_accuracy", np.nan
                        ),
                        test_balanced_accuracy=result.get(
                            "test_balanced_accuracy", np.nan
                        ),
                        complexity=result.get("complexity", np.nan),
                        fit_time_minutes=result.get("fit_time_minutes", np.nan),
                        errored_pipelines=result.get("evaluation_errors", {}).get(
                            "errored", np.nan
                        ),
                        evals=resolve_evaluation_count(directory, result),
                        best_eval=best_eval,
                        best_generation=best_generation,
                        total_runtime_minutes=resolve_runtime_minutes(directory, result),
                    )
                elif failure_path.is_file():
                    with failure_path.open(encoding="utf-8") as file:
                        failure = json.load(file)
                    row.update(status="failed", error=failure.get("error"))
                rows.append(row)
    return pd.DataFrame(rows)


def build_run_output(results: pd.DataFrame) -> pd.DataFrame:
    """Build the requested one-row-per-successful-replicate export."""
    completed = results.loc[results["status"].eq("success")].copy()
    output = pd.DataFrame(
        {
            "evals": completed["evals"],
            "condition": completed["search_space"].map(condition_label),
            "task": completed["task"],
            "seed": completed["seed"],
            "train_auroc": completed["train_score"],
            "validation_auroc": completed["validation_score"],
            "test_auroc": completed["test_score"],
            "complexity": completed["complexity"],
            "best_eval": completed["best_eval"],
            "best_generation": completed["best_generation"],
            "total_runtime_minutes": completed["total_runtime_minutes"],
        }
    )
    for column in ("evals", "seed", "best_eval", "best_generation"):
        output[column] = pd.to_numeric(output[column], errors="coerce").astype("Int64")
    return output.loc[:, RUN_OUTPUT_COLUMNS].sort_values(
        ["evals", "condition", "task", "seed"], kind="stable"
    )


def condition_label(search_space: str) -> str:
    """Collapse budget-specific directories into the two requested condition labels."""
    base_name = str(search_space).split("_", maxsplit=1)[0].casefold()
    labels = {"constrained": "tpot_c", "graph": "tpot_g"}
    return labels.get(base_name, base_name)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for (search_space, task), group in results.groupby(
        ["search_space", "task"], sort=True
    ):
        successful = group.loc[group["status"] == "success"]
        summary_rows.append(
            {
                "search_space": search_space,
                "population_size": int(group["population_size"].iloc[0]),
                "task": task,
                "metric": (
                    successful["metric"].dropna().iloc[0]
                    if successful["metric"].notna().any()
                    else None
                ),
                "successes": int((group["status"] == "success").sum()),
                "failures": int((group["status"] == "failed").sum()),
                "missing": int((group["status"] == "missing").sum()),
                "validation_mean": successful["validation_score"].mean(),
                "validation_std": successful["validation_score"].std(ddof=1),
                "test_mean": successful["test_score"].mean(),
                "test_std": successful["test_score"].std(ddof=1),
                "test_median": successful["test_score"].median(),
                "validation_auprc_mean": successful["validation_auprc"].mean(),
                "test_auprc_mean": successful["test_auprc"].mean(),
                "validation_balanced_accuracy_mean": successful[
                    "validation_balanced_accuracy"
                ].mean(),
                "test_balanced_accuracy_mean": successful[
                    "test_balanced_accuracy"
                ].mean(),
                "complexity_mean": successful["complexity"].mean(),
                "fit_time_minutes_mean": successful["fit_time_minutes"].mean(),
                "errored_pipelines_mean": successful["errored_pipelines"].mean(),
            }
        )
    return pd.DataFrame(summary_rows)


def plot_half_violins(
    results: pd.DataFrame,
    *,
    metric: str,
    tasks: list[str],
    search_spaces: list[str],
    output_path: Path,
) -> None:
    """Create one split violin comparing both TPOT conditions for all tasks."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires matplotlib and seaborn; install requirements.txt "
            "in the TPOT environment."
        ) from error

    if len(search_spaces) != 2:
        raise ValueError(
            "A split half-violin requires exactly two --search-spaces values"
        )

    plot_data = results.loc[
        results["status"].eq("success"),
        ["task", "search_space", "seed", metric],
    ].copy()
    plot_data[metric] = pd.to_numeric(plot_data[metric], errors="coerce")
    plot_data = plot_data.dropna(subset=[metric])
    present_conditions = set(plot_data["search_space"])
    missing_conditions = [
        condition for condition in search_spaces if condition not in present_conditions
    ]
    if missing_conditions:
        raise ValueError(
            "Cannot create a split violin because no completed results were found for: "
            + ", ".join(missing_conditions)
        )

    task_order = [task for task in tasks if task in set(plot_data["task"])]
    if not task_order:
        raise ValueError("No completed results are available for the requested tasks")

    plot_data["task_label"] = (
        plot_data["task"].map(TASK_LABELS).fillna(plot_data["task"])
    )
    task_label_order = [TASK_LABELS.get(task, task) for task in task_order]
    palette = dict(
        zip(search_spaces, sns.color_palette("colorblind", n_colors=2), strict=True)
    )

    sns.set_theme(style="whitegrid", context="notebook")
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    sns.violinplot(
        data=plot_data,
        x="task_label",
        y=metric,
        hue="search_space",
        order=task_label_order,
        hue_order=search_spaces,
        split=True,
        inner="quart",
        cut=0,
        gap=0.08,
        density_norm="width",
        common_norm=False,
        linewidth=1.0,
        palette=palette,
        ax=axis,
    )
    sns.stripplot(
        data=plot_data,
        x="task_label",
        y=metric,
        hue="search_space",
        order=task_label_order,
        hue_order=search_spaces,
        dodge=True,
        jitter=0.10,
        size=4,
        alpha=0.65,
        linewidth=0.35,
        edgecolor="white",
        palette=palette,
        legend=False,
        ax=axis,
    )

    offsets = (-0.055, 0.055)
    count_labels = []
    for task_index, task in enumerate(task_order):
        counts = []
        for condition_index, condition in enumerate(search_spaces):
            values = plot_data.loc[
                plot_data["task"].eq(task) & plot_data["search_space"].eq(condition),
                metric,
            ]
            counts.append(len(values))
            if not values.empty:
                axis.scatter(
                    task_index + offsets[condition_index],
                    values.median(),
                    marker="D",
                    s=42,
                    facecolor="white",
                    edgecolor=palette[condition],
                    linewidth=1.3,
                    zorder=5,
                )
        count_labels.append(
            f"{TASK_LABELS.get(task, task)}\n(n={counts[0]} / {counts[1]})"
        )
    axis.set_xticks(range(len(task_order)), labels=count_labels)

    bounded_metrics = {
        "train_score",
        "validation_score",
        "test_score",
        "train_auprc",
        "validation_auprc",
        "test_auprc",
        "train_balanced_accuracy",
        "validation_balanced_accuracy",
        "test_balanced_accuracy",
    }
    if metric in bounded_metrics:
        data_min = float(plot_data[metric].min())
        data_max = float(plot_data[metric].max())
        padding = max(0.025, (data_max - data_min) * 0.18)
        lower_limit = max(0.0, data_min - padding)
        upper_limit = min(1.0, data_max + padding)
        axis.set_ylim(lower_limit, upper_limit)
        reference_metrics = {
            "train_score",
            "validation_score",
            "test_score",
            "train_balanced_accuracy",
            "validation_balanced_accuracy",
            "test_balanced_accuracy",
        }
        if metric in reference_metrics and lower_limit <= 0.5 <= upper_limit:
            axis.axhline(
                0.5,
                color="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                zorder=0,
            )

    handles, labels = axis.get_legend_handles_labels()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label in search_spaces and label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)
    axis.legend(
        unique_handles,
        unique_labels,
        title="TPOT search space",
        frameon=False,
        loc="best",
    )
    axis.set_title("TPOT performance by task and search space", pad=12)
    axis.set_xlabel("")
    axis.set_ylabel(PLOT_METRICS[metric])
    axis.grid(axis="x", visible=False)
    figure.subplots_adjust(bottom=0.18, left=0.09, right=0.98, top=0.91)
    figure.text(
        0.5,
        0.025,
        "Completed replicates only; points are seeds, diamonds are medians, "
        "and n values are Constrained / Graph.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="dimgray",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    results = collect_results(
        args.results_root,
        args.pop_size,
        args.search_spaces,
        args.tasks,
        args.seeds,
    )
    summary = summarize(results)
    run_results = collect_results(
        args.results_root,
        args.pop_size,
        args.run_search_spaces,
        args.tasks,
        args.seeds,
    )
    run_output = build_run_output(run_results)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    failed = results.loc[results["status"] == "failed"]
    if not failed.empty:
        print("\nFailures:")
        print(
            failed[["population_size", "task", "seed", "error"]].to_string(index=False)
        )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output_csv, index=False)
        print(f"\nWrote {args.output_csv}")

    args.run_output_csv.parent.mkdir(parents=True, exist_ok=True)
    run_output.to_csv(args.run_output_csv, index=False)
    print(f"\nWrote {args.run_output_csv}")

    if not args.no_plot:
        plot_half_violins(
            results,
            metric=args.plot_metric,
            tasks=args.tasks,
            search_spaces=args.search_spaces,
            output_path=args.plot_output,
        )
        print(f"\nWrote {args.plot_output}")


if __name__ == "__main__":
    main()
