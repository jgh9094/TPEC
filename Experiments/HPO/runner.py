# run the evolutionary algorithm

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')))

import Source.HPO.ea as optimizer
import argparse
import ray

if __name__ == "__main__":
    # get configs for running EA
    parser = argparse.ArgumentParser(description="Run EA HPO")
    # random seed for reproducibility
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility.')
    # openml task id
    parser.add_argument('--task_id', type=int, required=True, help='OpenML task ID to load dataset from.')
    # data directory
    parser.add_argument('--data_directory', type=str, required=True, help='Directory where datasets are stored.')
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

    # print all argument values
    print("=" * 60)
    print("Experiment Configuration")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Task ID: {args.task_id}")
    print(f"Data Directory: {args.data_directory}")
    print(f"Train Proportion: {args.train_p}")
    print(f"Output Directory: {args.output_directory}")
    print(f"Model: {args.model}")
    print(f"Generations: {args.gens}")
    print(f"Population Size: {args.pop_size}")
    print(f"Cores: {args.cores}")
    print(f"Mutation Probability: {args.mut_prob}")
    print(f"Mutation Variance: {args.mut_var}")
    print(f"TPE Probability: {args.tpe_prob}")
    print(f"Tournament Size: {args.tournament_size}")
    print(f"Num Offspring: {args.num_offspring}")
    print(f"Gamma: {args.gamma}")
    print(f"TPE Mutation Scale: {args.tpe_mut_scale}")
    print(f"Explore Mutation Scale: {args.explore_mut_scale}")
    print("=" * 60)
    print()

    # initialize ray
    if not ray.is_initialized():
        ray.init(num_cpus=args.cores, include_dashboard=True, ignore_reinit_error=True)

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

    # load data
    ea.load_data(
        task_id=args.task_id,
        data_dir=args.data_directory,
        train_p=args.train_p
    )

    # run evolution
    ea.evolve(gens=args.gens)

    # save results
    ea.save_results(save_dir=args.output_directory)

    # shutdown ray
    ray.shutdown()
