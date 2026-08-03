#!/usr/bin/env python3
"""
Checker script to find missing/incomplete replicates across HPO results.

Directory structure:
    Results/{condition}/{model}/Task_{task_id}/Seed_{seed}/
        - archive.csv
        - best_results.json
"""

import pandas as pd
from pathlib import Path

# Configuration
RESULTS_DIR = Path(__file__).resolve().parents[2] / "Results"
TASKS_CSV = Path(__file__).resolve().parents[2] / "Experiments" / "Data-Preprocessing" / "Raw_OpenML_Suite_271_Classification" / "tasks_summary.csv"

CONDITIONS = ["EA", "25", "50", "75", "100"]
MODELS = ["GB", "KNN", "KSVC", "MLP", "RF"]
SEEDS = list(range(21))  # 0 to 20


def load_task_info():
    """Load task information from the summary CSV."""
    df = pd.read_csv(TASKS_CSV)
    task_info = {}
    for _, row in df.iterrows():
        task_info[int(row['task_id'])] = {
            'rows': int(row['rows']),
            'columns': int(row['columns']),
            'num_classes': int(row['num_classes'])
        }
    return task_info, sorted(task_info.keys())


def check_replicate_complete(seed_dir):
    """Check if a replicate directory has all required files."""
    required_files = ['archive.csv', 'best_results.json']
    if not seed_dir.exists():
        return False
    for f in required_files:
        if not (seed_dir / f).exists():
            return False
        # Check file is not empty
        if (seed_dir / f).stat().st_size == 0:
            return False
    return True


def main():
    task_info, all_tasks = load_task_info()

    print("=" * 80)
    print("HPO Results Checker")
    print("=" * 80)
    print(f"\nResults directory: {RESULTS_DIR}")
    print(f"Expected conditions: {CONDITIONS}")
    print(f"Expected models: {MODELS}")
    print(f"Expected tasks: {len(all_tasks)}")
    print(f"Expected seeds per task: {len(SEEDS)} (0-20)")
    print()

    # Track missing replicates
    for condition in CONDITIONS:
        condition_dir = RESULTS_DIR / condition

        if not condition_dir.exists():
            print(f"\n{'='*80}")
            print(f"CONDITION: {condition} - DIRECTORY NOT FOUND")
            print(f"{'='*80}")
            continue

        print(f"\n{'='*80}")
        print(f"CONDITION: {condition}")
        print(f"{'='*80}")

        condition_missing = {}

        for model in MODELS:
            model_dir = condition_dir / model

            if not model_dir.exists():
                print(f"\n  MODEL: {model} - DIRECTORY NOT FOUND")
                continue

            model_missing = []

            for task_id in all_tasks:
                task_dir = model_dir / f"Task_{task_id}"
                info = task_info[task_id]

                missing_seeds = []
                incomplete_seeds = []

                for seed in SEEDS:
                    seed_dir = task_dir / f"Seed_{seed}"

                    if not seed_dir.exists():
                        missing_seeds.append(seed)
                    elif not check_replicate_complete(seed_dir):
                        incomplete_seeds.append(seed)

                if missing_seeds or incomplete_seeds:
                    model_missing.append({
                        'task_id': task_id,
                        'rows': info['rows'],
                        'columns': info['columns'],
                        'num_classes': info['num_classes'],
                        'missing_seeds': missing_seeds,
                        'incomplete_seeds': incomplete_seeds
                    })

            if model_missing:
                condition_missing[model] = model_missing

        # Print summary for this condition
        for model, missing_list in condition_missing.items():
            print(f"\n  MODEL: {model}")
            print(f"  {'-'*60}")

            total_missing = 0
            total_incomplete = 0

            for item in missing_list:
                n_missing = len(item['missing_seeds'])
                n_incomplete = len(item['incomplete_seeds'])
                total_missing += n_missing
                total_incomplete += n_incomplete

                dims = f"[{item['rows']} x {item['columns']}, {item['num_classes']} classes]"

                if n_missing > 0:
                    if n_missing == 21:
                        print(f"    Task {item['task_id']} {dims}: ALL SEEDS MISSING")
                    else:
                        print(f"    Task {item['task_id']} {dims}: Missing seeds {item['missing_seeds']}")

                if n_incomplete > 0:
                    print(f"    Task {item['task_id']} {dims}: Incomplete seeds {item['incomplete_seeds']}")

            print(f"\n    Summary: {total_missing} missing, {total_incomplete} incomplete out of {len(all_tasks) * len(SEEDS)} total")

        # Print models with no missing
        complete_models = [m for m in MODELS if m not in condition_missing]
        if complete_models:
            print(f"\n  COMPLETE MODELS: {', '.join(complete_models)}")

    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")

    for condition in CONDITIONS:
        condition_dir = RESULTS_DIR / condition
        if not condition_dir.exists():
            print(f"  {condition}: NOT STARTED")
        else:
            total = 0
            complete = 0
            for model in MODELS:
                model_dir = condition_dir / model
                if model_dir.exists():
                    for task_id in all_tasks:
                        task_dir = model_dir / f"Task_{task_id}"
                        for seed in SEEDS:
                            total += 1
                            seed_dir = task_dir / f"Seed_{seed}"
                            if check_replicate_complete(seed_dir):
                                complete += 1

            expected = len(MODELS) * len(all_tasks) * len(SEEDS)
            pct = (complete / expected * 100) if expected > 0 else 0
            print(f"  {condition}: {complete}/{expected} ({pct:.1f}%) complete")


if __name__ == "__main__":
    main()
