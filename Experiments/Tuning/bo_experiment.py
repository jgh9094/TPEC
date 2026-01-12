# run the bayesian optimization algorithm

from Source.Base.eval_utils import MODEL_CONFIG
from Source.HPO.bo import BO
import argparse
import ray

if __name__ == "__main__":
    # get configs for running BO
    parser = argparse.ArgumentParser(description="Run BO HPO")
    # which of the 7 model configs are we using?
    parser.add_argument('--model_config', type=str, default='RF', help='Model configuration to use.')
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
    # total evaluations
    parser.add_argument('--total_evals', type=int, default=500, help='Total number of evaluations to perform.')
    # top candidates
    parser.add_argument('--top_candidates', type=int, default=50, help='Number of top candidates to retain from sampled TPE set.')
    # number of offspring
    parser.add_argument('--num_offspring', type=int, default=10, help='Number of pseudo offspring to generate per cycle.')
    # output directory
    parser.add_argument('--output_directory', type=str, required=True, help='Directory for output files.')
    # gamma for TPE
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma parameter for TPE.')

    args = parser.parse_args()

    # print all argument values
    print("=" * 60)
    print("Experiment Configuration")
    print("=" * 60)
    print(f"Model Configuration: {args.model_config}")
    print(f"Seed: {args.seed}")
    print(f"Task ID: {args.task_id}")
    print(f"Data Directory: {args.data_directory}")
    print(f"Split Directory: {args.split_directory}")
    print(f"Replicate: {args.rep}")
    print(f"Total Evaluations: {args.total_evals}")
    print(f"Top Candidates: {args.top_candidates}")
    print(f"Number of Offspring: {args.num_offspring}")
    print(f"Output Directory: {args.output_directory}")
    print(f"Gamma (for TPE): {args.gamma}")
    print("=" * 60)
    print()

    # initialize ray
    if not ray.is_initialized():
        ray.init(num_cpus=12, include_dashboard=False, ignore_reinit_error=True)

    # initialize BO
    bo = BO(model_config=MODEL_CONFIG[args.model_config],
            seed=args.seed,
            total_evals=args.total_evals,
            top_candidates=args.top_candidates,
            num_offspring=args.num_offspring,
            task_id=args.task_id,
            rep=args.rep,
            data_dir=args.data_directory,
            split_dir=args.split_directory,
            output_dir=args.output_directory,
            gamma=args.gamma)

    # load dataset from openml
    bo.load_openml_dataset()
    # let it rip
    bo.run()
    # save results
    bo.save_results()

    # shutdown ray
    ray.shutdown()