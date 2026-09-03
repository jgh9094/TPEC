# run the evolutionary algorithm

import sys
import os
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')))

import Source.HPO.ea as optimizer
import argparse
import ray

def load_spine_opioid_data(data_path: str, y_label: str):

    # load the dataset as a pandas dataframe
    data_set = pd.read_csv(data_path)

    # all possible y labels in the dataset
    possible_y_labels = ['LOS_extended', 'discharge_Home', 'HOSP_READM_90']

    # check if the provided y_label is valid
    if y_label not in possible_y_labels:
        raise ValueError(f"Invalid y_label '{y_label}'. Must be one of {possible_y_labels}.")

    # drop all other y labels from the dataset
    for label in possible_y_labels:
        if label != y_label:
            data_set = data_set.drop(columns=[label])

    # cols needed to be one-hot encoded
    one_hot_cols = []

    # cols that need to be scalar transformed
    scalar_cols = ['AGE','BMI','SBP','PAIN_SCORE','WBC_COUNT',
                   'HEMOGLOBIN','POTASSIUM', 'SODIUM','PLATELET_COUNT',
                   'RBC_COUNT','CALCIUM','CHLORIDE','BUN','CREATININE']

    # make sure that all cols in scalar_cols exist in the dataset
    for col in scalar_cols:
        if col not in data_set.columns:
            raise ValueError(f"Column '{col}' not found in the dataset.")

    return data_set, one_hot_cols, scalar_cols

def is_run_complete(output_directory: str) -> bool:
    """
    Check if a run is complete by verifying the existence of best_results.json.
    A complete run will have: {output_directory}/best_results.json
    """
    results_file = os.path.join(output_directory, "best_results.json")
    return os.path.isfile(results_file)

if __name__ == "__main__":
    # get configs for running EA
    parser = argparse.ArgumentParser(description="Run EA HPO")
    # random seed for reproducibility
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility.')
    # task
    parser.add_argument('--task', type=str, required=True, help='what dataset are we using.')
    # y label
    parser.add_argument('--y_label', type=str, required=True, help='Which y label to use for the dataset.')
    # data directory
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data.')
    # train proportion
    parser.add_argument('--train_p', type=float, required=True, help='Proportion of dataset to use for training.')
    # output directory
    parser.add_argument('--output_directory', type=str, required=True, help='Directory for output files.')
    # model type
    parser.add_argument('--model', type=str, required=True, choices=['RF', 'KSVC', 'GB', 'KNN', 'MLP'], help='Model type to optimize.')
    # number of generations
    parser.add_argument('--gens', type=int, required=True, help='Number of generations.')
    # population size
    parser.add_argument('--pop_size', type=int, required=True, help='Population size.')
    # number of cores
    parser.add_argument('--cores', type=int, required=True, help='Number of CPU cores to use.')
    # mutation probability
    parser.add_argument('--mut_prob', type=float, required=True, help='Probability of mutating each hyperparameter.')
    # mutation variance
    parser.add_argument('--mut_var', type=float, required=True, help='Variance for Gaussian mutation.')
    # TPE probability
    parser.add_argument('--tpe_prob', type=float, required=True, help='Probability of using TPE-based selection.')
    # tournament size
    parser.add_argument('--tournament_size', type=int, required=True, help='Tournament size for parent selection.')
    # number of offspring for TPE
    parser.add_argument('--num_offspring', type=int, required=True, help='Number of pseudo-offspring to generate for TPE.')
    # gamma for TPE
    parser.add_argument('--gamma', type=float, required=True, help='Gamma parameter for TPE (fraction of samples considered good).')
    # TPE mutation scale
    parser.add_argument('--tpe_mut_scale', type=float, required=True, help='Scaling factor for TPE mutation variance.')
    # explore mutation scale
    parser.add_argument('--explore_mut_scale', type=float, required=True, help='Scaling factor for exploration mutation variance.')

    args = parser.parse_args()

    # check if run is already complete
    if is_run_complete(args.output_directory):
        print(f"Run already complete. Results exist at: {args.output_directory}/best_results.json")
        print("Skipping execution.")
        sys.exit(0)

    # print all argument values
    print("=" * 60, flush=True)
    print("Experiment Configuration", flush=True)
    print("=" * 60, flush=True)
    print(f"Seed: {args.seed}")
    print(f"Task: {args.task}", flush=True)
    print(f"Data Path: {args.data_path}", flush=True)
    print(f"Train Proportion: {args.train_p}", flush=True)
    print(f"Output Directory: {args.output_directory}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Generations: {args.gens}", flush=True)
    print(f"Population Size: {args.pop_size}", flush=True)
    print(f"Cores: {args.cores}", flush=True)
    print(f"Mutation Probability: {args.mut_prob}", flush=True)
    print(f"Mutation Variance: {args.mut_var}", flush=True)
    print(f"TPE Probability: {args.tpe_prob}", flush=True)
    print(f"Tournament Size: {args.tournament_size}", flush=True)
    print(f"Num Offspring: {args.num_offspring}", flush=True)
    print(f"Gamma: {args.gamma}", flush=True)
    print(f"TPE Mutation Scale: {args.tpe_mut_scale}", flush=True)
    print(f"Explore Mutation Scale: {args.explore_mut_scale}", flush=True)
    print("=" * 60, flush=True)
    print('', flush=True)

    # initialize ray
    if not ray.is_initialized():
        ray.init(num_cpus=args.cores, include_dashboard=True, ignore_reinit_error=True)
    print(f"Ray initialized with {args.cores} cores.", flush=True)

    # create EA
    ea = optimizer.EA(
        seed=args.seed,
        pop_size=args.pop_size,
        cores=args.cores,
        mut_prob=args.mut_prob,
        mut_var=args.mut_var,
        model=args.model,
        tpe_prob=args.tpe_prob,
        tournament_size=args.tournament_size,
        num_offspring=args.num_offspring,
        gamma=args.gamma,
        tpe_mut_scale=args.tpe_mut_scale,
        explore_mut_scale=args.explore_mut_scale
    )
    print(f"EA initalized", flush=True)

    if args.task == 'so':
        # load spine opioid data
        data_set, one_hot_cols, scalar_cols = load_spine_opioid_data(data_path=args.data_path, y_label=args.y_label)
        print(f"Spine Opioid data loaded", flush=True)
    else:
        raise ValueError(f"Unknown task '{args.task}'. Supported tasks: 'so' (spine opioid).")

    # load data
    ea.load_data_pd(
        data=data_set,
        target_label=args.y_label,
        train_p=args.train_p,
        one_hot_cols=one_hot_cols,
        scalar_cols=scalar_cols
    )
    print(f"EA data loaded", flush=True)

    # run evolution
    ea.evolve(gens=args.gens)

    # save results
    ea.save_results(save_dir=args.output_directory)

    # shutdown ray
    ray.shutdown()
