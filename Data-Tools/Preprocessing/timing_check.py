#!/usr/bin/env python3
"""
Script to generate and evaluate ML models in batches.
Supports all ModelParams classes from model_param_space.py.
"""

import os
import sys
import argparse
import numpy as np
import ray
from time import time
import json

# Add Source directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Source.data_utils import load_data, get_ray_cv_splits, preprocess_train_test
from Source.eval_utils import MODEL_CONFIG


def main():
    parser = argparse.ArgumentParser(
        description='Generate and evaluate ML models in batches'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        required=True,
        choices=list(MODEL_CONFIG.keys()),
        help=f'Type of model to train. Options: {", ".join(MODEL_CONFIG.keys())}'
    )
    parser.add_argument(
        '--seed',
        type=int,
        required=True,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--task_id',
        type=int,
        required=True,
        help='OpenML task ID'
    )
    parser.add_argument(
        '--rep',
        type=int,
        required=True,
        choices=range(10),
        help='Replicate number (0-9)'
    )
    parser.add_argument(
        '--data_directory',
        type=str,
        required=True,
        help='Directory containing raw data CSV files'
    )
    parser.add_argument(
        '--split_directory',
        type=str,
        required=True,
        help='Directory containing replicate-specific data splits'
    )
    parser.add_argument(
        '--output_directory',
        type=str,
        required=True,
        help='Directory to save results'
    )
    parser.add_argument(
        '--num_models',
        type=int,
        default=5000,
        help='Total number of models to generate (default: 5000)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=500,
        help='Number of models per batch (default: 500)'
    )

    args = parser.parse_args()

    # Get model configuration
    model_config = MODEL_CONFIG[args.model_type]
    model_name = model_config['display_name']

    # Set random seed
    rng = np.random.default_rng(args.seed)

    # Default number of CV folds
    NUM_FOLDS = 5

    # Initialize Ray
    ray.init(ignore_reinit_error=True, num_cpus=12, log_to_driver=True, include_dashboard=False)

    print(f"{'='*70}", flush=True)
    print(f"{model_name} Timing Check", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Model Type: {args.model_type}", flush=True)
    print(f"Task ID: {args.task_id}", flush=True)
    print(f"Replicate: {args.rep}", flush=True)
    print(f"Number of CV Folds: {NUM_FOLDS}", flush=True)
    print(f"Random Seed: {args.seed}", flush=True)
    print(f"Total Models: {args.num_models}", flush=True)
    print(f"Batch Size: {args.batch_size}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Construct paths - include task-specific folder
    rep_dir = os.path.join(args.split_directory, f"task_{args.task_id}", f"Replicate_{args.rep}")

    # Load data
    print("Loading data...", flush=True)
    X_train, X_test, y_train, y_test = load_data(
        task_id=args.task_id,
        data_dir=args.data_directory,
        splits_dir=rep_dir
    )
    print(f"Training data shape: {X_train.shape}", flush=True)
    print(f"Test data shape: {X_test.shape}\n", flush=True)

    # Initialize model parameter class
    # GradientBoostParams requires number of classes
    if args.model_type == 'gradient_boost':
        num_classes = len(np.unique(y_train))
        model_params_class = model_config['param_class'](classes=num_classes)
    else:
        model_params_class = model_config['param_class']()

    # Generate all random parameters upfront
    print(f"Generating {args.num_models} random {model_name} parameter configurations...", flush=True)
    all_params = []
    for _ in range(args.num_models):
        params = model_params_class.generate_random_parameters(rng)
        all_params.append(params)
    print(f"Generated {len(all_params)} parameter configurations.\n", flush=True)

    # Process all folds
    X_train_f0, X_val_f0, y_train_f0, y_val_f0, \
    X_train_f1, X_val_f1, y_train_f1, y_val_f1, \
    X_train_f2, X_val_f2, y_train_f2, y_val_f2, \
    X_train_f3, X_val_f3, y_train_f3, y_val_f3, \
    X_train_f4, X_val_f4, y_train_f4, y_val_f4 = get_ray_cv_splits(rep_dir=rep_dir,
                                                                    X_train=X_train,
                                                                    y_train=y_train,
                                                                    task_id=args.task_id,
                                                                    data_dir=args.data_directory)

    # Get the appropriate Ray training function
    train_func = model_config['ray_train_func']

    # iterate through all_params in batches of args.batch_size
    num_batches = args.num_models // args.batch_size
    all_config_results = {}
    start_time_all = time()
    time_limit_broken = False

    for batch_idx in range(num_batches):
        batch_start_id = batch_idx * args.batch_size
        batch_end_id = batch_start_id + args.batch_size
        batch_params = all_params[batch_start_id:batch_end_id]
        time_limit_broken = False

        print(f"Processing batch {batch_idx + 1}/{num_batches} (Models {batch_start_id} to {batch_end_id - 1})...", flush=True)

        start_time = time()

        # iteaterate through each batch_params and train on each fold
        futures = []
        for i, params in enumerate(batch_params):

            # Submit Ray tasks for each fold
            futures.append(train_func.remote(X_train_f0, y_train_f0, X_val_f0, y_val_f0, params, args.seed, i + batch_start_id))
            futures.append(train_func.remote(X_train_f1, y_train_f1, X_val_f1, y_val_f1, params, args.seed, i + batch_start_id))
            futures.append(train_func.remote(X_train_f2, y_train_f2, X_val_f2, y_val_f2, params, args.seed, i + batch_start_id))
            futures.append(train_func.remote(X_train_f3, y_train_f3, X_val_f3, y_val_f3, params, args.seed, i + batch_start_id))
            futures.append(train_func.remote(X_train_f4, y_train_f4, X_val_f4, y_val_f4, params, args.seed, i + batch_start_id))

        # Gather results with ray.wait
        while len(futures) > 0:
            done, futures = ray.wait(futures, num_returns=1)
            result = ray.get(done[0])
            model_id, train_acc, val_acc, error = result
            assert error >= 0.0, f"Error occurred during training for model ID {model_id}"

            # check if time limit exceeded an hour
            if time() - start_time_all > 3600:
                print("Time limit exceeded for this batch. Moving to next batch.", flush=True)
                time_limit_broken = True
                break

            if model_id not in all_config_results:
                all_config_results[model_id] = {
                    'train_accuracies': [],
                    'val_accuracies': [],
                    'parameters': batch_params[model_id - batch_start_id]
                }
            all_config_results[model_id]['train_accuracies'].append(train_acc)
            all_config_results[model_id]['val_accuracies'].append(val_acc)

        if time_limit_broken:
            break

        end_time = time()
        batch_duration = end_time - start_time
        print(f"Completed batch {batch_idx + 1}/{num_batches} in {batch_duration:.2f} seconds.\n", flush=True)

    # total time in minutes
    total_time = (time() - start_time_all) / 60
    global_results_path = os.path.join(args.output_directory, f"global_accuracy_results.json")
    best_model_path = os.path.join(args.output_directory, f"best_model_results.json")

    if time_limit_broken:
        print("Time limit was exceeded during processing. Partial results will be saved.", flush=True)
        # save summary_stats to json
        with open(global_results_path, 'w') as f:
            json.dump({'time_exceeded': True, 'total_time': total_time}, f, indent=4)
        with open(best_model_path, 'w') as f:
            json.dump({}, f, indent=4)

        print(f"Saved partial global results to {global_results_path}", flush=True)
        print(f"Saved empty best model results to {best_model_path}\n", flush=True)
        ray.shutdown()
        return

    # go though all all_config_results and compute mean and variance accuracies for each model
    for model_id, results in all_config_results.items():
        assert len(results['train_accuracies']) == NUM_FOLDS, f"Expected {NUM_FOLDS} train accuracies for model ID {model_id}, got {len(results['train_accuracies'])}"
        assert len(results['val_accuracies']) == NUM_FOLDS, f"Expected {NUM_FOLDS} val accuracies for model ID {model_id}, got {len(results['val_accuracies'])}"
        results['mean_train_accuracy'] = np.mean(results['train_accuracies'])
        results['var_train_accuracy'] = np.var(results['train_accuracies'])
        results['mean_val_accuracy'] = np.mean(results['val_accuracies'])
        results['var_val_accuracy'] = np.var(results['val_accuracies'])

    # Save results to output directory in a file called global_results.json
    # go through all_config_results and compute the find the min, 25th, 50th, mean, 75th, and max mean_val_accuracy
    all_val_accuracies = [results['mean_val_accuracy'] for results in all_config_results.values()]
    all_train_accuracies = [results['mean_train_accuracy'] for results in all_config_results.values()]
    summary_stats = {
        'time_exceeded': False,
        'total_time': total_time,
        'min_cv_mean_val_accuracy': np.min(all_val_accuracies),
        '25th_percentile_cv_mean_val_accuracy': np.percentile(all_val_accuracies, 25),
        '50th_percentile_cv_mean_val_accuracy': np.percentile(all_val_accuracies, 50),
        'mean_of_mean_cv_val_accuracies': np.mean(all_val_accuracies),
        '75th_percentile_cv_mean_val_accuracy': np.percentile(all_val_accuracies, 75),
        'max_cv_mean_val_accuracy': np.max(all_val_accuracies),
        'min_cv_mean_train_accuracy': np.min(all_train_accuracies),
        '25th_percentile_cv_mean_train_accuracy': np.percentile(all_train_accuracies, 25),
        '50th_percentile_cv_mean_train_accuracy': np.percentile(all_train_accuracies, 50),
        'mean_of_mean_cv_train_accuracies': np.mean(all_train_accuracies),
        '75th_percentile_cv_mean_train_accuracy': np.percentile(all_train_accuracies, 75),
        'max_cv_mean_train_accuracy': np.max(all_train_accuracies)
    }
    # save summary_stats to json
    with open(global_results_path, 'w') as f:
        json.dump(summary_stats, f, indent=4)
    print(f"Saved global results to {global_results_path}", flush=True)

    # find the best model based on greatest mean_val_accuracy
    best_model_id = max(all_config_results, key=lambda mid: all_config_results[mid]['mean_val_accuracy'])

    # Preprocess train and test data before final evaluation
    print(f"\nPreprocessing training and testing data...", flush=True)
    X_train_preprocessed, y_train_preprocessed, X_test_preprocessed, y_test_preprocessed = preprocess_train_test(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task_id=args.task_id,
        data_dir=args.data_directory
    )

    # train the best_model on the full training data and evaluate on test data
    print(f"Training best model ID {best_model_id} on full training data...", flush=True)
    best_model_params = all_config_results[best_model_id]['parameters']
    train, test, error = model_config['eval_func'](
        X_train=X_train_preprocessed,
        y_train=y_train_preprocessed,
        X_test=X_test_preprocessed,
        y_test=y_test_preprocessed,
        model_params=best_model_params,
        random_state=args.seed
    )

    if error < 0.0:
        print(f"Error occurred while training best model on full data. Skipping test evaluation.", flush=True)
    else:
        print(f"Best Model Test Accuracy: {test:.4f}", flush=True)

    best_model_results = {
        'cv_mean_train_accuracy': float(all_config_results[best_model_id]['mean_train_accuracy']),
        'cv_var_train_accuracy': float(all_config_results[best_model_id]['var_train_accuracy']),
        'cv_mean_val_accuracy': float(all_config_results[best_model_id]['mean_val_accuracy']),
        'cv_var_val_accuracy': float(all_config_results[best_model_id]['var_val_accuracy']),
        'train_accuracy': float(train),
        'test_accuracy': float(test)
    }
    # unroll all best model paramters into best_model_results
    # Convert numpy types to native Python types for JSON serialization
    for key, value in all_config_results[best_model_id]['parameters'].items():
        if isinstance(value, np.bool_):
            best_model_results[key] = bool(value)
        elif isinstance(value, (np.integer, np.floating)):
            best_model_results[key] = value.item()
        else:
            best_model_results[key] = value
    with open(best_model_path, 'w') as f:
        json.dump(best_model_results, f, indent=4)
    print(f"Saved best model results to {best_model_path}\n", flush=True)

    # Shutdown Ray
    ray.shutdown()

    # print total time taken
    print(f"Total time taken: {total_time:.2f} minutes", flush=True)


if __name__ == "__main__":
    main()
