# run the CASH evolutionary algorithm (evolves full scikit-learn pipelines)

import sys
import os
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')))

import Source.CASH.ea as optimizer
import argparse
import ray


def load_spine_opioid_data(data_path: str, y_label: str):
    """Load the spine-opioid dataset, keeping only the requested y label."""
    data_set = pd.read_csv(data_path)

    # all possible y labels in the dataset
    possible_y_labels = ['LOS_extended', 'discharge_Home', 'HOSP_READM_90']
    if y_label not in possible_y_labels:
        raise ValueError(f"Invalid y_label '{y_label}'. Must be one of {possible_y_labels}.")

    # drop all other y labels from the dataset
    for label in possible_y_labels:
        if label != y_label:
            data_set = data_set.drop(columns=[label])

    one_hot_cols = []
    scalar_cols = ['AGE', 'BMI', 'SBP', 'PAIN_SCORE', 'WBC_COUNT',
                   'HEMOGLOBIN', 'POTASSIUM', 'SODIUM', 'PLATELET_COUNT',
                   'RBC_COUNT', 'CALCIUM', 'CHLORIDE', 'BUN', 'CREATININE']
    for col in scalar_cols:
        if col not in data_set.columns:
            raise ValueError(f"Column '{col}' not found in the dataset.")

    return data_set, one_hot_cols, scalar_cols


def load_generic_csv(data_path: str, y_label: str):
    """
    Load an arbitrary CSV for debugging: the ``y_label`` column is the target and EVERY other
    column is treated as a scalar (numeric) feature to be StandardScaler-transformed. Intended
    for the synthetic debug dataset (see Experiments/CASH/debug_data.csv).
    """
    data_set = pd.read_csv(data_path)
    if y_label not in data_set.columns:
        raise ValueError(f"Target label '{y_label}' not found in the dataset columns.")

    one_hot_cols = []
    scalar_cols = [col for col in data_set.columns if col != y_label]
    return data_set, one_hot_cols, scalar_cols


def is_run_complete(output_directory: str) -> bool:
    """A complete run has {output_directory}/best_results.json."""
    results_file = os.path.join(output_directory, "best_results.json")
    return os.path.isfile(results_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EA CASH (full-pipeline optimization)")
    # random seed for reproducibility
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility.')
    # task ('so' = spine opioid; 'debug' = generic CSV where every non-target column is scalar)
    parser.add_argument('--task', type=str, required=True, choices=['so', 'debug'], help='Which dataset/loader to use.')
    # y label
    parser.add_argument('--y_label', type=str, required=True, help='Which y label (target column) to use.')
    # data directory
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data CSV.')
    # train proportion
    parser.add_argument('--train_p', type=float, required=True, help='Proportion of dataset to use for training.')
    # output directory
    parser.add_argument('--output_directory', type=str, required=True, help='Directory for output files.')
    # number of generations
    parser.add_argument('--gens', type=int, required=True, help='Number of generations.')
    # population size
    parser.add_argument('--pop_size', type=int, required=True, help='Population size.')
    # number of cores
    parser.add_argument('--cores', type=int, required=True, help='Number of CPU cores to use.')
    # mutation probability
    parser.add_argument('--mut_prob', type=float, required=True, help='Probability of mutating each hyperparameter (gene).')
    # mutation variance
    parser.add_argument('--mut_var', type=float, required=True, help='Variance for the local Gaussian shift-mutation.')
    # crossover probability
    parser.add_argument('--crossover_prob', type=float, required=True, help='Probability an offspring is produced by crossover vs. mutation.')
    # component (structural) mutation probability
    parser.add_argument('--component_mut_prob', type=float, required=True, help='Probability a node\'s component is resampled (structural mutation).')
    # TPE probability
    parser.add_argument('--tpe_prob', type=float, required=True, help='Probability of using TPE-guided variation.')
    # tournament size
    parser.add_argument('--tournament_size', type=int, required=True, help='Tournament size for parent selection.')
    # number of offspring for TPE
    parser.add_argument('--num_offspring', type=int, required=True, help='Number of pseudo-offspring generated per TPE-guided step.')
    # gamma for TPE
    parser.add_argument('--gamma', type=float, required=True, help='Gamma parameter for TPE (fraction of samples considered good).')
    # task type: classification (ROC-AUC) or regression (R^2)
    parser.add_argument('--classification', type=lambda x: str(x).lower() in ('true', '1', 'yes'), default=True,
                        help='Whether the task is classification (True, ROC-AUC) or regression (False, R^2). Default True.')
    # number of cross-validation folds
    parser.add_argument('--n_folds', type=int, default=5, help='Number of cross-validation folds. Default 5.')

    args = parser.parse_args()

    # check if run is already complete
    if is_run_complete(args.output_directory):
        print(f"Run already complete. Results exist at: {args.output_directory}/best_results.json")
        print("Skipping execution.")
        sys.exit(0)

    # print all argument values
    print("=" * 60, flush=True)
    print("CASH Experiment Configuration", flush=True)
    print("=" * 60, flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print(f"Task: {args.task}", flush=True)
    print(f"Y Label: {args.y_label}", flush=True)
    print(f"Data Path: {args.data_path}", flush=True)
    print(f"Train Proportion: {args.train_p}", flush=True)
    print(f"Output Directory: {args.output_directory}", flush=True)
    print(f"Generations: {args.gens}", flush=True)
    print(f"Population Size: {args.pop_size}", flush=True)
    print(f"Cores: {args.cores}", flush=True)
    print(f"Mutation Probability: {args.mut_prob}", flush=True)
    print(f"Mutation Variance: {args.mut_var}", flush=True)
    print(f"Crossover Probability: {args.crossover_prob}", flush=True)
    print(f"Component Mutation Probability: {args.component_mut_prob}", flush=True)
    print(f"TPE Probability: {args.tpe_prob}", flush=True)
    print(f"Tournament Size: {args.tournament_size}", flush=True)
    print(f"Num Offspring: {args.num_offspring}", flush=True)
    print(f"Gamma: {args.gamma}", flush=True)
    print(f"Classification: {args.classification}", flush=True)
    print(f"CV Folds: {args.n_folds}", flush=True)
    print("=" * 60, flush=True)
    print('', flush=True)

    # initialize ray
    if not ray.is_initialized():
        ray.init(num_cpus=args.cores, include_dashboard=True, ignore_reinit_error=True)
    print(f"Ray initialized with {args.cores} cores.", flush=True)

    # create the CASH EA
    ea = optimizer.EA(
        seed=args.seed,
        pop_size=args.pop_size,
        cores=args.cores,
        mut_prob=args.mut_prob,
        mut_var=args.mut_var,
        tpe_prob=args.tpe_prob,
        tournament_size=args.tournament_size,
        num_offspring=args.num_offspring,
        crossover_prob=args.crossover_prob,
        component_mut_prob=args.component_mut_prob,
        gamma=args.gamma,
        classification=args.classification,
    )
    print(f"CASH EA initialized", flush=True)

    # load the dataset for the requested task
    if args.task == 'so':
        data_set, one_hot_cols, scalar_cols = load_spine_opioid_data(data_path=args.data_path, y_label=args.y_label)
        print(f"Spine Opioid data loaded", flush=True)
    elif args.task == 'debug':
        data_set, one_hot_cols, scalar_cols = load_generic_csv(data_path=args.data_path, y_label=args.y_label)
        print(f"Debug data loaded", flush=True)
    else:
        raise ValueError(f"Unknown task '{args.task}'. Supported tasks: 'so', 'debug'.")

    # load data into the EA
    ea.load_data_pd(
        data=data_set,
        target_label=args.y_label,
        train_p=args.train_p,
        one_hot_cols=one_hot_cols,
        scalar_cols=scalar_cols,
        n_folds=args.n_folds,
    )
    print(f"CASH EA data loaded", flush=True)

    # run evolution (checkpoint best-so-far test performance each generation)
    ea.evolve(gens=args.gens, checkpoint_dir=args.output_directory)

    # save results
    ea.save_results(save_dir=args.output_directory)

    # shutdown ray
    ray.shutdown()
