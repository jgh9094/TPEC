#!/usr/bin/env python3
"""Plot random-search test performance at several evaluation budgets.

The default invocation reads ``Results/random_evals`` and writes a single-page
PDF containing one panel per dataset (OpenML task). Each panel shows right-side
half violins for the 10, 100, and 1,000 independent searches, together with the
individual split-seed scores and their medians.

Example
-------
python Source/random_evals/analyze_random_evals.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIRECTORY = PROJECT_ROOT / "Results" / "random_evals"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "pdf" / "random_evals_test_performance.pdf"
DEFAULT_TASK_SUMMARY = (
    PROJECT_ROOT
    / "Data"
    / "Raw_OpenML_Suite_271_Binary_Classification"
    / "tasks_summary.csv"
)
DEFAULT_BUDGETS = (10, 100, 1000)
EXPECTED_REPLICATES = 20

TASK_PATTERN = re.compile(r"Task_(\d+)$")
REPLICATE_PATTERN = re.compile(r"Replicate_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=DEFAULT_RESULTS_DIRECTORY,
        help="Directory containing Task_<id>/<model>/Replicate_<n> results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination PDF path.",
    )
    parser.add_argument(
        "--task-summary",
        type=Path,
        default=DEFAULT_TASK_SUMMARY,
        help="Optional task-summary CSV used for names and dataset-size ordering.",
    )
    parser.add_argument("--model", default="RF", help="Model results subdirectory to analyze.")
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_BUDGETS),
        help="Evaluation budgets to plot (default: 10 100 1000).",
    )
    parser.add_argument(
        "--expected-replicates",
        type=int,
        default=EXPECTED_REPLICATES,
        help="Required number of split seeds per task and budget (default: 20).",
    )
    return parser.parse_args()


def load_task_metadata(summary_path: Path) -> Dict[int, Dict[str, str]]:
    """Return task-summary rows keyed by integer task ID when available."""
    if not summary_path.is_file():
        return {}

    with summary_path.open(newline="", encoding="utf-8") as summary_file:
        rows = csv.DictReader(summary_file)
        if not rows.fieldnames or "task_id" not in rows.fieldnames:
            raise ValueError(f"Task summary has no 'task_id' column: {summary_path}")
        return {int(row["task_id"]): row for row in rows}


def dataset_label(task_id: int, metadata: Mapping[int, Mapping[str, str]]) -> str:
    """Prefer an optional human-readable name, while always retaining the task ID."""
    task_metadata = metadata.get(task_id, {})
    for column in ("dataset_name", "task_name", "name"):
        name = task_metadata.get(column, "").strip()
        if name:
            return f"{name}\nTask {task_id}"
    return f"Task {task_id}"


def task_sort_key(task_id: int, metadata: Mapping[int, Mapping[str, str]]) -> Tuple[float, int]:
    """Use summary-file dataset size when present, then task ID for stability."""
    try:
        rows = float(metadata.get(task_id, {}).get("rows", "nan"))
    except ValueError:
        rows = math.nan
    return (rows if math.isfinite(rows) else math.inf, task_id)


def load_scores(
    results_directory: Path,
    model: str,
    budgets: Sequence[int],
    expected_replicates: int,
) -> Dict[int, Dict[int, List[Tuple[int, float]]]]:
    """Load and validate ``(split_seed, test_score)`` pairs for every task/budget."""
    if not results_directory.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_directory}")

    requested_budgets = set(budgets)
    scores: Dict[int, Dict[int, List[Tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for task_directory in sorted(results_directory.glob("Task_*")):
        task_match = TASK_PATTERN.fullmatch(task_directory.name)
        if not task_directory.is_dir() or task_match is None:
            continue
        task_id = int(task_match.group(1))
        model_directory = task_directory / model
        if not model_directory.is_dir():
            continue

        for replicate_directory in sorted(model_directory.glob("Replicate_*")):
            replicate_match = REPLICATE_PATTERN.fullmatch(replicate_directory.name)
            if not replicate_directory.is_dir() or replicate_match is None:
                continue
            replicate = int(replicate_match.group(1))

            for budget in budgets:
                result_path = replicate_directory / f"budget_{budget}_best_results.json"
                if not result_path.is_file():
                    raise FileNotFoundError(f"Missing budget result: {result_path}")
                with result_path.open(encoding="utf-8") as result_file:
                    result = json.load(result_file)

                file_budget = int(result.get("evaluation_budget", -1))
                file_task = int(result.get("task_id", -1))
                file_replicate = int(result.get("replicate", -1))
                if file_budget not in requested_budgets or file_budget != budget:
                    raise ValueError(f"Budget mismatch in {result_path}: {file_budget}")
                if file_task != task_id or file_replicate != replicate:
                    raise ValueError(
                        f"Task/replicate metadata mismatch in {result_path}: "
                        f"task={file_task}, replicate={file_replicate}"
                    )
                score = float(result["test_score"])
                if not math.isfinite(score):
                    raise ValueError(f"Non-finite test score in {result_path}: {score}")
                split_seed = int(result["split_seed"])
                scores[task_id][budget].append((split_seed, score))

    if not scores:
        raise ValueError(f"No '{model}' result files found below {results_directory}")

    for task_id, task_scores in scores.items():
        for budget in budgets:
            observations = task_scores.get(budget, [])
            seeds = [seed for seed, _ in observations]
            if len(observations) != expected_replicates:
                raise ValueError(
                    f"Task {task_id}, budget {budget}: expected {expected_replicates} "
                    f"replicates, found {len(observations)}"
                )
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"Task {task_id}, budget {budget}: duplicate split seeds")
            observations.sort(key=lambda item: item[0])

    return {task_id: dict(task_scores) for task_id, task_scores in scores.items()}


def _style_half_violin(body: object, center: float, color: str) -> None:
    """Clip a Matplotlib violin body to its right half and apply common styling."""
    path = body.get_paths()[0]
    vertices = path.vertices
    vertices[:, 0] = np.maximum(vertices[:, 0], center)
    body.set_facecolor(color)
    body.set_edgecolor("#243447")
    body.set_linewidth(0.8)
    body.set_alpha(0.72)


def plot_task(
    axis: plt.Axes,
    task_id: int,
    task_scores: Mapping[int, Sequence[Tuple[int, float]]],
    budgets: Sequence[int],
    metadata: Mapping[int, Mapping[str, str]],
    colors: Sequence[str],
) -> None:
    positions = np.arange(1, len(budgets) + 1, dtype=float)
    score_arrays = [np.asarray([score for _, score in task_scores[budget]]) for budget in budgets]
    violins = axis.violinplot(
        score_arrays,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method="scott",
        points=200,
    )

    for position, color, body, values in zip(positions, colors, violins["bodies"], score_arrays):
        _style_half_violin(body, position, color)

        # Fixed offsets preserve the seed ordering and avoid stochastic jitter.
        offsets = np.linspace(-0.29, -0.07, len(values))
        axis.scatter(
            position + offsets,
            values,
            s=18,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.95,
            zorder=4,
        )
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        axis.vlines(position + 0.025, q1, q3, color="#17202A", linewidth=2.1, zorder=5)
        axis.scatter(
            [position + 0.025],
            [median],
            marker="_",
            s=110,
            linewidth=2,
            color="#17202A",
            zorder=6,
        )

    all_values = np.concatenate(score_arrays)
    data_range = max(float(np.ptp(all_values)), 0.02)
    padding = max(data_range * 0.18, 0.008)
    axis.set_ylim(max(0.0, float(all_values.min()) - padding), min(1.0, float(all_values.max()) + padding))
    axis.set_xlim(0.55, len(budgets) + 0.52)
    axis.set_xticks(positions, [f"{budget:,}" for budget in budgets])
    axis.set_title(dataset_label(task_id, metadata), fontsize=9.5, weight="semibold", pad=6)
    axis.grid(axis="y", color="#D8DEE6", linewidth=0.65, alpha=0.8)
    axis.tick_params(axis="both", labelsize=7.5, length=2.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#AAB2BD")


def create_figure(
    scores: Mapping[int, Mapping[int, Sequence[Tuple[int, float]]]],
    budgets: Sequence[int],
    metadata: Mapping[int, Mapping[str, str]],
    model: str,
) -> plt.Figure:
    task_ids = sorted(scores, key=lambda task_id: task_sort_key(task_id, metadata))
    replicate_count = len(scores[task_ids[0]][budgets[0]])
    column_count = min(4, len(task_ids))
    row_count = math.ceil(len(task_ids) / column_count)
    colors = ("#3B82C4", "#E59F3A", "#49A078", "#9B72CF", "#D65F5F")
    if len(budgets) > len(colors):
        colors = tuple(plt.get_cmap("tab10")(index) for index in range(len(budgets)))

    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(15.5, 3.0 * row_count + 1.25),
        squeeze=False,
        constrained_layout=False,
    )
    figure.patch.set_facecolor("white")

    for axis, task_id in zip(axes.flat, task_ids):
        plot_task(axis, task_id, scores[task_id], budgets, metadata, colors)
    for axis in list(axes.flat)[len(task_ids) :]:
        axis.set_visible(False)

    figure.suptitle(
        f"Random-search test-set performance ({model})",
        fontsize=17,
        weight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.954,
        f"Half violins show independent searches across {replicate_count} split seeds; "
        "points are individual seeds",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#4A5568",
    )
    figure.supxlabel("Number of random evaluations", fontsize=11, y=0.023)
    figure.supylabel("Test ROC AUC", fontsize=11, x=0.018)

    legend_handles: List[object] = [
        Patch(facecolor=colors[index], edgecolor="#243447", alpha=0.72, label=f"{budget:,} evals")
        for index, budget in enumerate(budgets)
    ]
    legend_handles.extend(
        [
            Line2D([], [], marker="o", linestyle="None", color="#536273", markersize=4, label="Split seed"),
            Line2D([], [], marker="_", linestyle="None", color="#17202A", markersize=9, label="Median"),
        ]
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.052),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=8.5,
        handlelength=1.4,
    )
    figure.subplots_adjust(left=0.06, right=0.985, top=0.91, bottom=0.105, hspace=0.42, wspace=0.25)
    return figure


def main() -> None:
    args = parse_args()
    if len(args.budgets) != len(set(args.budgets)) or any(budget <= 0 for budget in args.budgets):
        raise ValueError("Budgets must be distinct positive integers")
    if args.expected_replicates <= 0:
        raise ValueError("--expected-replicates must be positive")

    metadata = load_task_metadata(args.task_summary)
    scores = load_scores(
        args.results_directory,
        args.model,
        args.budgets,
        args.expected_replicates,
    )
    figure = create_figure(scores, args.budgets, metadata, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, format="pdf", dpi=300, bbox_inches="tight", metadata={
        "Title": f"Random-search test-set performance ({args.model})",
        "Subject": "Half-violin plots across split seeds and evaluation budgets",
        "Creator": Path(__file__).name,
    })
    plt.close(figure)
    print(
        f"Wrote {args.output} with {len(scores)} datasets, "
        f"{len(args.budgets)} budgets, and {args.expected_replicates} split seeds each."
    )


if __name__ == "__main__":
    main()
