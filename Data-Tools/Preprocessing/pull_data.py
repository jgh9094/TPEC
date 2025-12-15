# Python script to pull binary classification datasets from OpenML benchmark suite 271
# The script filters for binary classification tasks, downloads their datasets, and saves them as CSV files.
# Datasets with missing values are excluded.
# It also generates a summary CSV file listing the task IDs, number of rows, and number of columns for each dataset.
# link: https://www.openml.org/search?type=benchmark&study_type=task&sort=tasks_included&id=271
# paper: https://www.jmlr.org/papers/volume25/22-0493/22-0493.pdf

import os
import csv
import argparse
import sys
from openml import tasks

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Source'))
from Source.data_utils import load_task_dataset, is_binary_classification_task, get_suite_task_ids

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Pull binary classification datasets from OpenML suite 271')
    parser.add_argument('--save-dir', type=str, default='',
                        help='Directory path to append to the default output directory name')
    args = parser.parse_args()

    suite_id = 271
    base_output_dir = "Raw_OpenML_Suite_271_Binary_Classification"

    # Append the save-dir argument to the base output directory if provided
    if args.save_dir:
        output_dir = os.path.join(args.save_dir, base_output_dir)
    else:
        output_dir = base_output_dir

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # 1. Get all task IDs from the suite
    task_ids = get_suite_task_ids(suite_id)
    print(f"Found {len(task_ids)} tasks in suite {suite_id}.")

    summary_rows = []

    for task_id in task_ids:
        try:
            print(f"Processing task {task_id}...")
            task = tasks.get_task(task_id)

            # 2. Filter to binary classification tasks
            if not is_binary_classification_task(task):
                print(f"  -> Task {task_id} is not binary classification. Skipping.")
                continue

            # 3. Load the dataset for this task
            df, target_name, has_missing, minority_pct, majority_pct = load_task_dataset(task)

            # Skip datasets with missing values
            if has_missing:
                print(f"  -> Task {task_id} has missing values. Skipping.")
                continue

            n_rows, n_cols = df.shape

            # 4. Save the entire dataset to CSV
            dataset_csv_path = os.path.join(output_dir, f"task_{task_id}.csv")
            df.to_csv(dataset_csv_path, index=False)

            print(f"  -> Saved dataset to {dataset_csv_path}")
            print(f"  -> Rows: {n_rows}, Columns: {n_cols}, Minority: {minority_pct:.2f}%, Majority: {majority_pct:.2f}%")

            # 5. Append summary info
            summary_rows.append({
                "task_id": task_id,
                "rows": n_rows,
                "columns": n_cols,
                "minority_class_pct": round(minority_pct, 2),
                "majority_class_pct": round(majority_pct, 2)
            })

        except Exception as e:
            # If anything goes wrong for this task, print and continue
            print(f"Error processing task {task_id}: {e}")

    # 6. Sort summary rows by number of rows
    summary_rows.sort(key=lambda x: x["rows"])

    # 7. Save the summary CSV for all retained binary classification tasks
    summary_csv_path = os.path.join(output_dir, "tasks_summary.csv")
    with open(summary_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "rows", "columns", "minority_class_pct", "majority_class_pct"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"\nSummary written to: {summary_csv_path}")
    print(f"Total binary classification tasks retained: {len(summary_rows)}")

if __name__ == "__main__":
    main()