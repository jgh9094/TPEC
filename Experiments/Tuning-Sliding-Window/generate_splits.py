"""
Script to generate train/test splits for all tasks in the OpenML benchmark suite.
Creates multiple random splits for each task to enable cross-validation experiments.

Will also be used in preprocessing and main experiments in the paper.

Usage:
    python generate_splits.py <output_dir> [--data_dir DIR] [--num_splits N] [--test_size RATIO] [--seed_offset OFFSET] [--num_folds K]

Arguments:
    output_dir: Directory where splits will be saved
    --data_dir: Directory containing the OpenML CSV data folder (default: repo_root/OpenML_Suite_271_Binary_Classification_CSV)
    --num_splits: Number of splits to generate per task (default: 10)
    --test_size: Proportion of data for test set (default: 0.25)
    --seed_offset: Integer offset added to replicate index for seed (default: 0)
    --num_folds: Number of CV folds to create from training set (default: 5)
"""

import os
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import sys
from openml import tasks as openml_tasks

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Source.Base.data_utils import create_train_test_stratified_splits

def main():
    parser = argparse.ArgumentParser(
        description='Generate train/test splits for OpenML benchmark tasks'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory where splits will be saved (organized by task_id/replicate_N/)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Directory containing the OpenML CSV data folder (default: repo_root/OpenML_Suite_271_Binary_Classification_CSV)'
    )
    parser.add_argument(
        '--num_splits',
        type=int,
        default=21,
        help='Number of splits to generate per task (default: 10)'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.25,
        help='Proportion of data for test set (default: 0.25)'
    )
    parser.add_argument(
        '--seed_offset',
        type=int,
        default=0,
        help='Integer offset added to replicate index for seed (default: 0)'
    )
    parser.add_argument(
        '--num_folds',
        type=int,
        default=5,
        help='Number of CV folds to create from training set (default: 5)'
    )

    args = parser.parse_args()

    # Paths
    repo_root = os.path.join(os.path.dirname(__file__), '..', '..')

    # Use provided data_dir or default to repo_root location
    if args.data_dir is not None:
        data_dir = os.path.abspath(args.data_dir)
    else:
        data_dir = os.path.join(repo_root, 'Raw_OpenML_Suite_271_Binary_Classification')

    summary_csv = os.path.join(data_dir, 'tasks_summary.csv')
    output_dir = os.path.abspath(args.output_dir)

    # Read task IDs from summary and filter by EXP column
    summary_df = pd.read_csv(summary_csv)

    # Filter for tasks where EXP == 1
    filtered_df = summary_df[summary_df['EXP'] == 2]
    task_ids = filtered_df['task_id'].tolist()

    print(f"Filtered tasks with EXP=1:")
    print(f"Task IDs: {task_ids}")
    print(f"Number of tasks: {len(task_ids)}\n")

    # Make sure output directory already exists or else assert error
    assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist. Please create it before running the script."

    # Write parameters to file
    params_file = os.path.join(output_dir, 'generation_parameters.txt')
    with open(params_file, 'w') as f:
        f.write("Data Split Generation Parameters\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Data directory: {data_dir}\n")
        f.write(f"Summary CSV: {summary_csv}\n\n")
        f.write(f"Number of tasks: {len(task_ids)}\n")
        f.write(f"Task IDs: {task_ids}\n\n")
        f.write(f"Number of splits per task: {args.num_splits}\n")
        f.write(f"Test size: {args.test_size}\n")
        f.write(f"Seed offset: {args.seed_offset}\n")
        f.write(f"Seed range: {args.seed_offset} to {args.seed_offset + args.num_splits - 1}\n")
        f.write(f"Number of CV folds: {args.num_folds}\n\n")
        f.write(f"Total splits created: {len(task_ids) * args.num_splits}\n")
        f.write(f"Directory structure: task_{{task_id}}/Replicate_{{N}}/\n")
        f.write(f"Files per replicate: test.pkl, fold_train_{{i}}.pkl, fold_validate_{{i}}.pkl (for i=0..{args.num_folds-1})\n")

    print(f"Found {len(task_ids)} tasks to process")
    print(f"Generating {args.num_splits} splits for each task")
    print(f"Test size: {args.test_size}")
    print(f"Seed offset: {args.seed_offset}")
    print(f"CV folds: {args.num_folds}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Parameters saved to: {params_file}\n")

    # Generate splits for each task
    for task_id in task_ids:
        print(f"\n{'='*60}")
        print(f"Processing Task ID: {task_id}")
        print(f"{'='*60}")

        for replicate_idx in range(args.num_splits):
            seed = replicate_idx + args.seed_offset  # Use replicate index + offset as seed
            splits_dir = os.path.join(output_dir, f"task_{task_id}", f"Replicate_{replicate_idx}")

            # Create initial train/test split
            train_indices, test_indices = create_train_test_stratified_splits(
                task_id=task_id,
                data_dir=data_dir,
                splits_dir=splits_dir,
                test_size=args.test_size,
                seed=seed
            )

            # Load the CSV to get target labels for stratification
            csv_path = os.path.join(data_dir, f"task_{task_id}.csv")
            df = pd.read_csv(csv_path)

            # Get task to find target column
            task = openml_tasks.get_task(task_id)
            target_name = task.target_name

            # Get training labels for stratified CV
            y_train = df[target_name].iloc[train_indices].values

            # Create stratified K-fold splits from training set
            skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=seed)

            fold_idx = 0
            for fold_train_idx, fold_val_idx in skf.split(np.zeros(len(train_indices)), y_train):
                # Map back to original dataset indices
                fold_train_indices = train_indices[fold_train_idx]
                fold_val_indices = train_indices[fold_val_idx]

                # Get class labels for this fold's training and validation sets
                fold_train_labels = df[target_name].iloc[fold_train_indices].values
                fold_val_labels = df[target_name].iloc[fold_val_indices].values

                # Assert both classes are present in fold training set
                unique_train_classes = np.unique(fold_train_labels)
                assert len(unique_train_classes) >= 2, \
                    f"Fold {fold_idx} training set has only {len(unique_train_classes)} class(es): {unique_train_classes}"

                # Assert both classes are present in fold validation set
                unique_val_classes = np.unique(fold_val_labels)
                assert len(unique_val_classes) >= 2, \
                    f"Fold {fold_idx} validation set has only {len(unique_val_classes)} class(es): {unique_val_classes}"

                # Assert no overlap between fold train and fold validation
                assert len(set(fold_train_indices) & set(fold_val_indices)) == 0, \
                    f"Overlap detected between fold_train_{fold_idx} and fold_validate_{fold_idx}"

                # Assert no overlap between fold train and test
                assert len(set(fold_train_indices) & set(test_indices)) == 0, \
                    f"Overlap detected between fold_train_{fold_idx} and test set"

                # Assert no overlap between fold validation and test
                assert len(set(fold_val_indices) & set(test_indices)) == 0, \
                    f"Overlap detected between fold_validate_{fold_idx} and test set"

                # Assert fold train + fold validate = original train indices
                combined_fold_indices = np.concatenate([fold_train_indices, fold_val_indices])
                assert set(combined_fold_indices) == set(train_indices), \
                    f"Fold {fold_idx} train+validate doesn't match original training set"

                # Save fold indices
                fold_train_path = os.path.join(splits_dir, f"fold_train_{fold_idx}.pkl")
                fold_val_path = os.path.join(splits_dir, f"fold_validate_{fold_idx}.pkl")

                with open(fold_train_path, 'wb') as f:
                    pickle.dump(fold_train_indices, f)

                with open(fold_val_path, 'wb') as f:
                    pickle.dump(fold_val_indices, f)

                fold_idx += 1

            print(f"  -> Created {args.num_folds} CV folds for replicate {replicate_idx}")

    print(f"\n{'='*60}")
    print("All splits generated successfully!")
    print(f"Total tasks processed: {len(task_ids)}")
    print(f"Splits per task: {args.num_splits}")
    print(f"Total splits created: {len(task_ids) * args.num_splits}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
