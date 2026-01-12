# run the evolutionary algorithm

from Source.Base.tpe import TPE
from Source.Base.eval_utils import MODEL_CONFIG
from Source.HPO.ea import EA
import argparse
import ray

if __name__ == "__main__":
    # get configs for running EA
    parser = argparse.ArgumentParser(description="Run EA HPO")
    # which of the 7 model configs are we using?
    parser.add_argument('--model_config', type=str, default='RF', help='Model configuration to use.')
    # which optimizer are we using [TPEC, EA]?
    parser.add_argument('--optimizer', type=str, default='TPEC', help='Optimizer to use: TPEC or EA')
    # random seed for reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    # openml task id
    parser.add_argument('--task_id', type=int, required=True, help='OpenML task ID to load dataset from.')
    # data directory
    parser.add_argument('--data_directory', type=str, required=True, help='Directory where datasets are stored.')
    # split directory
    parser.add_argument('--split_directory', type=str, required=True, help='Directory where splits are stored.')
    # replicate number
    parser.add_argument('--rep', type=int, default=0, help='Replicate number.')
    # tournament size
    parser.add_argument('--tournament_size', type=int, default=2, help='Tournament size for parent selection.')
    # variance for mutation
    parser.add_argument('--mutation_var', type=float, default=0.5, help='Variance for Gaussian mutation of parameters.')
    # mutation rate
    parser.add_argument('--mutation_rate', type=float, default=1.0, help='Mutation rate for offspring generation.')
    # number of generations
    parser.add_argument('--gens', type=int, default=9, help='Number of generations to evolve.')
    # population size
    parser.add_argument('--pop_size', type=int, default=50, help='Population size.')
    # output directory
    parser.add_argument('--output_directory', type=str, required=True, help='Directory for output files.')
    # gamma for TPE
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma parameter for TPE.')
    # probability for TPE event
    parser.add_argument('--tpe_prob', type=float, default=0.5, help='Probability of event for TPE.')

    args = parser.parse_args()

    # print all argument values
    print("=" * 60)
    print("Experiment Configuration")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Model Configuration: {args.model_config}")
    print(f"Task ID: {args.task_id}")
    print(f"Data Directory: {args.data_directory}")
    print(f"Split Directory: {args.split_directory}")
    print(f"Replicate: {args.rep}")
    print(f"Tournament Size: {args.tournament_size}")
    print(f"Mutation Variance: {args.mutation_var}")
    print(f"Mutation Rate: {args.mutation_rate}")
    print(f"Generations: {args.gens}")
    print(f"Population Size: {args.pop_size}")
    print(f"Output Directory: {args.output_directory}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Gamma (for TPE): {args.gamma}")
    print("=" * 60)
    print()

    # initialize ray
    if not ray.is_initialized():
        ray.init(num_cpus=12, include_dashboard=False, ignore_reinit_error=True)

    # initialize EA
    ea = EA(model_config=MODEL_CONFIG[args.model_config],
            seed=args.seed,
            gens=args.gens,
            pop_size=args.pop_size,
            tournament_size=args.tournament_size,
            mutation_rate=args.mutation_rate,
            mutation_var=args.mutation_var,
            num_offspring=10,
            task_id=args.task_id,
            rep=args.rep,
            data_dir=args.data_directory,
            split_dir=args.split_directory,
            output_dir=args.output_directory,
            gamma=args.gamma,
            tpe_prob=args.tpe_prob
            )

    # load dataset from openml
    ea.load_openml_dataset()
    # let it rip
    ea.evolve()
    # save results
    ea.save_results()

    # shutdown ray
    ray.shutdown()