import os
import json
import csv
import argparse
from pathlib import Path


def collect_results(base_dir):
    """
    Traverse the directory structure and collect results from best_results.json
    and diversity_metric.json files.

    Args:
        base_dir: Path to the Tournament_Size directory

    Returns:
        List of dictionaries containing combined results
    """
    results = []

    # Traverse the directory structure: task_id/tournament_size/replicate/
    base_path = Path(base_dir)

    for task_dir in sorted(base_path.iterdir()):
        if not task_dir.is_dir():
            continue

        # Extract task_id from directory name (e.g., "task_146818" -> 146818)
        if not task_dir.name.startswith('task_'):
            continue
        task_id = task_dir.name.split('_')[1]

        # Iterate through tournament sizes (T5, T10, T25, etc.)
        for tournament_dir in sorted(task_dir.iterdir()):
            if not tournament_dir.is_dir():
                continue

            tournament_size = tournament_dir.name[1:]  # Extract number only: "T5" -> "5"

            # Iterate through replicates
            for replicate_dir in sorted(tournament_dir.iterdir()):
                if not replicate_dir.is_dir():
                    continue

                # Extract replicate number (e.g., "Replicate_0" -> 0)
                if not replicate_dir.name.startswith('Replicate_'):
                    continue
                replicate_num = replicate_dir.name.split('_')[1]

                # Paths to JSON files
                best_results_path = replicate_dir / 'best_results.json'
                diversity_metric_path = replicate_dir / 'diversity_metric.json'

                # Check if both files exist
                if not best_results_path.exists() or not diversity_metric_path.exists():
                    print(f"Warning: Missing files in {replicate_dir}")
                    continue

                try:
                    # Load best_results.json
                    with open(best_results_path, 'r') as f:
                        best_results = json.load(f)

                    # Load diversity_metric.json
                    with open(diversity_metric_path, 'r') as f:
                        diversity_data = json.load(f)

                    # Combine the data
                    combined = {
                        'task_id': best_results.get('task_id', task_id),
                        'tournament_size': tournament_size,
                        'replicate': best_results.get('replicate', replicate_num),
                        'seed': best_results.get('seed'),
                        'train_accuracy': best_results.get('train_accuracy'),
                        'test_accuracy': best_results.get('test_accuracy'),
                        'validation_accuracy': best_results.get('validation_accuracy'),
                        'diversity_metric': diversity_data.get('diversity_metric'),
                        'model_type': best_results.get('model_type')
                    }

                    results.append(combined)

                except json.JSONDecodeError as e:
                    print(f"Error reading JSON in {replicate_dir}: {e}")
                except Exception as e:
                    print(f"Error processing {replicate_dir}: {e}")

    return results


def save_to_csv(results, output_path):
    """
    Save the collected results to a CSV file.

    Args:
        results: List of dictionaries containing the results
        output_path: Path to the output CSV file
    """
    if not results:
        print("No results to save!")
        return

    # Define the field names
    fieldnames = [
        'task_id',
        'tournament_size',
        'replicate',
        'seed',
        'train_accuracy',
        'test_accuracy',
        'validation_accuracy',
        'diversity_metric',
        'model_type'
    ]

    # Write to CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Successfully saved {len(results)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate tournament size experiment results into a CSV file.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Path to the Tournament_Size directory containing the results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data.csv',
        help='Output CSV file name (default: data.csv)'
    )

    args = parser.parse_args()

    # Check if directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} does not exist!")
        return

    print(f"Collecting results from {args.data_dir}...")
    results = collect_results(args.data_dir)

    print(f"Found {len(results)} result records")

    # Save to CSV
    save_to_csv(results, args.output)


if __name__ == '__main__':
    main()
