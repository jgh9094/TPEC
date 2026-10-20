# run the CASH evolutionary algorithm (evolves full scikit-learn pipelines)

import sys
import os
import pandas as pd
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../..'))
sys.path.insert(0, REPO_ROOT)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../Data-Preprocessing')))

import Source.CASH.ea as optimizer
import argparse
import ray
from pathlib import Path

# components (get_model_type() names) the evolution may use at each pipeline node: every
# component in Source/ML except PolynomialFeatures and MLP (too slow). "passthrough" is always
# available where the node allows it, so it is not listed.
SCALERS = ['MinMaxScaler', 'RobustScaler', 'StandardScaler', 'MaxAbsScaler', 'Normalizer']
TRANSFORMERS = ['KBinsDiscretizer', 'Binarizer', 'PCA', 'FastICA', 'FeatureAgglomeration',
                'Nystroem', 'RBFSampler', 'QuantileTransformer', 'PowerTransformer']
SELECTORS = ['SelectFwe', 'SelectPercentile', 'VarianceThreshold']
CLASSIFIERS = ['RF', 'ET', 'KSVC', 'GB', 'KNN']


def load_hiv_data(data_directory: Path, y_label: str):

    """Load the HIV data from a directory."""
    if not data_directory.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {data_directory}")

    data = pd.read_csv(data_directory)

    # check if the provided y_label is valid
    if y_label not in data.columns:
        raise ValueError(f"Invalid y_label '{y_label}'.")

    # the target cannot be dropped with the null columns, so fail clearly if it has nulls
    if data[y_label].isnull().any():
        raise ValueError(f"Target '{y_label}' has null values.")

    # drop any column that has null values
    null_columns = data.columns[data.isnull().any()].tolist()
    data = data.drop(columns=null_columns)
    print(f"Dropped {len(null_columns)} column(s) with null values: {null_columns}", flush=True)

    # cols needed to be one-hot encoded
    one_hot_cols = []

    # cols that need to be scalar transformed
    scalar_cols = ['CURRENT_AGE']

    # make sure that all cols in scalar_cols exist in the dataset
    for col in scalar_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in the dataset.")

    print(f"Loaded HIV data from {data_directory} with shape {data.shape}.", flush=True)

    return data, one_hot_cols, scalar_cols

def is_run_complete(output_directory: str) -> bool:
    """
    Check if a run is complete by verifying the existence of best_results.json.
    A complete run will have: {output_directory}/best_results.json
    """
    results_file = os.path.join(output_directory, "best_results.json")
    return os.path.isfile(results_file)

if __name__ == "__main__":
    # get configs for running EA
    parser = argparse.ArgumentParser(description="Run EA CASH (full-pipeline optimization)")
    # random seed for reproducibility
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility.')
    # y label
    parser.add_argument('--y_label', type=str, required=True, help='Which y label to use for the dataset.')
    # data directory
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data.')
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
    parser.add_argument('--mut_prob', type=float, required=True, help='Probability of mutating each hyperparameter.')
    # mutation variance
    parser.add_argument('--mut_var', type=float, required=True, help='Variance for Gaussian mutation.')
    # crossover probability
    parser.add_argument('--crossover_prob', type=float, required=True, help='Probability of applying crossover.')
    # component (structural) mutation probability
    parser.add_argument('--component_mut_prob', type=float, required=True, help="Probability a node's component is resampled (structural mutation).")
    # TPE probability
    parser.add_argument('--tpe_prob', type=float, required=True, help='Probability of using TPE-based selection.')
    # tournament size
    parser.add_argument('--tournament_size', type=int, required=True, help='Tournament size for parent selection.')
    # number of offspring for TPE
    parser.add_argument('--num_offspring', type=int, required=True, help='Number of pseudo-offspring to generate for TPE.')
    # gamma for TPE
    parser.add_argument('--gamma', type=float, required=True, help='Gamma parameter for TPE (fraction of samples considered good).')
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
    print(f"CV Folds: {args.n_folds}", flush=True)
    print(f"Scalers: {SCALERS}", flush=True)
    print(f"Transformers: {TRANSFORMERS}", flush=True)
    print(f"Selectors: {SELECTORS}", flush=True)
    print(f"Predictors: {CLASSIFIERS}", flush=True)
    print("=" * 60, flush=True)
    print('', flush=True)

    # initialize ray
    if not ray.is_initialized():
        # Ray workers do not inherit this script's sys.path, so hand them the repo root via
        # PYTHONPATH; otherwise they cannot import Source when launched outside the repo root
        ray.init(num_cpus=args.cores, include_dashboard=False, ignore_reinit_error=True,
                 runtime_env={'env_vars': {'PYTHONPATH': REPO_ROOT}})
    print(f"Ray initialized with {args.cores} cores.", flush=True)

    # convert data_path to Path object
    args.data_path = Path(args.data_path).expanduser().resolve()
    data_set, one_hot_cols, scalar_cols = load_hiv_data(args.data_path, args.y_label)

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
        scalers=SCALERS,
        transformers=TRANSFORMERS,
        selectors=SELECTORS,
        predictors=CLASSIFIERS,
        component_mut_prob=args.component_mut_prob,
        gamma=args.gamma,
    )
    print(f"CASH EA initialized", flush=True)

    # load data
    ea.load_data_pd(
        data=data_set,
        target_label=args.y_label,
        train_p=args.train_p,
        one_hot_cols=one_hot_cols,
        scalar_cols=scalar_cols,
        n_folds=args.n_folds
    )
    print(f"CASH EA data loaded", flush=True)

    # run evolution
    ea.evolve(gens=args.gens)

    # save results
    ea.save_results(save_dir=args.output_directory)

    # shutdown ray
    ray.shutdown()
