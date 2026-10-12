##########################################################################################
#
# Optuna-based hyperparameter optimization runner for single-model HPO.
#
# This mirrors Source/HPO/ea.py (the TPE-guided EA) but swaps the evolutionary search for
# an Optuna study. The following samplers are supported:
#
#   random  -> optuna.samplers.RandomSampler
#   tpe     -> optuna.samplers.TPESampler
#   gp      -> optuna.samplers.GPSampler                 (requires `torch`)
#   cmaes   -> CatCmawmSampler (CatCMA-with-margin)      (requires `cmaes` + `optunahub`)
#
# The `cmaes` option uses CatCmawmSampler from OptunaHub rather than Optuna's core CmaEsSampler:
# the core sampler does not support categorical parameters, and every model's space here mixes
# categorical and numerical hyperparameters, so CatCMA-with-margin (the mixed-variable variant
# Optuna's docs recommend for this case) is the correct choice.
#
# The parameter space for each ML model comes from Source/Base/model_param_space.py (the
# exact same specs the EA optimizes over), and every trial is scored with the same 5-fold
# Ray cross-validation and test-set evaluation path used by the EA. Because data loading is
# delegated to BaseEA.load_data_pd, the train/test split and CV folds for a given --seed are
# identical to what ea.py sees, so results are directly comparable.
#
# SEQUENTIAL (ONE-AT-A-TIME) EVALUATION
# -------------------------------------
# This runner drives a manual ask/tell loop that evaluates exactly one candidate configuration at a
# time: each trial asks a single configuration from the sampler, evaluates it, and tells the result
# back before the next ask. The sampler is therefore re-fit on every completed trial, so each new
# suggestion sees the full archive of prior evaluations. Ray is still used to run a single
# configuration's 5 CV folds in parallel, but configurations themselves are never batched.
# Total candidates considered = --n_trials. A configuration the sampler has already suggested reuses
# its stored validation score instead of being re-fitted, matching how the EA handles duplicate
# genomes, so --n_trials and the EA's POP_SIZE * (GENS + 1) count the same thing.
#
# --pop_size no longer controls a batch size; it only sets the size of the initial random search
# phase (n_startup_trials for TPE/GP, so the first pop_size trials are fully random, matching the
# EA's random initial pop) and CatCMA's internal population size (popsize). CatCMA still accumulates
# its population internally across the one-at-a-time tells and updates once popsize results arrive.
#
# The one remaining user-defined control (overlapping the EA's TPE machinery):
#   --n_ei_candidates  : number of pseudo-offspring the TPE sampler considers per suggestion
#                        (analogous to num_offspring in ea.py when TPE is enabled) (TPE only).
#
# RESULTS
# -------
# After the final evaluation, the best configuration (highest mean validation AUC) is evaluated on
# the held-out test set and saved to {output_directory}/best_results.json, which also records
# best_eval (the 1-based evaluation number at which that configuration was found). Every trial is
# saved to {output_directory}/trials.csv (user_attrs_reused marks reused duplicates). The total
# optimization runtime in minutes is saved to {output_directory}/total_runtime_minutes.txt.
#
##########################################################################################

import sys
import os
import json
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../Data-Preprocessing')))

import argparse
import ray
import optuna
from process_data import load_modeling_data

# Reuse the HPO EA purely for its data/evaluation infrastructure: load_data_pd (which builds
# the per-seed train/test split + 5-fold CV folds), the model parameter space, the per-model
# Ray CV training function, and the final test-set evaluation. None of the EA's evolutionary
# machinery is exercised here.
from Source.HPO.ea import EA
from Source.HPO.individual import HPOIndividual


def load_spine_opioid_data(X: pd.DataFrame, y: pd.DataFrame, y_label: str):

    # all possible y labels in the dataset
    possible_y_labels = ['LOS_extended', 'discharge_Home', 'HOSP_READM_90']

    # check if the provided y_label is valid
    if y_label not in possible_y_labels:
        raise ValueError(f"Invalid y_label '{y_label}'. Must be one of {possible_y_labels}.")

    # combine features with only the selected y label
    data_set = pd.concat([X, y[[y_label]]], axis=1)

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


def build_distributions(param_space: dict) -> dict:
    """
    Translate a model's parameter-space spec (from model_param_space.py) into a dict of Optuna
    distributions. This mirrors ``suggest_parameters`` exactly (same keys, bounds, and log/choice
    semantics) and is used to hand a fixed search space to samplers that require one up front
    (e.g. CatCmawm).
    """
    distributions = {}
    for name, spec in param_space.items():
        ptype = spec['type']
        if ptype == 'int':
            lo, hi = spec['bounds']
            distributions[name] = optuna.distributions.IntDistribution(int(lo), int(hi))
        elif ptype == 'float':
            lo, hi = spec['bounds']
            distributions[name] = optuna.distributions.FloatDistribution(
                float(lo), float(hi), log=bool(spec.get('log', False)))
        elif ptype in ('cat', 'bool'):
            distributions[name] = optuna.distributions.CategoricalDistribution(list(spec['bounds']))
        else:
            raise ValueError(f"Unsupported parameter type: {ptype}")
    return distributions


def build_sampler(sampler_name: str, seed: int, pop_size: int, n_ei_candidates: int,
                  distributions: dict):
    """
    Construct an Optuna sampler for sequential (one-at-a-time) ask/tell evaluation.

    ``pop_size`` no longer sets a batch size. It sets the initial random-search phase
    (``n_startup_trials`` for TPE/GP, so the first ``pop_size`` trials are fully random, matching the
    EA's random initial pop) and CatCMA's internal ``popsize``. ``n_ei_candidates``
    ("pseudo-offspring") is applied to TPE only. The RandomSampler ignores these. All samplers are
    seeded with ``seed`` so a given seed reproduces both the datasplits and the search trajectory.
    ``distributions`` is the model's search space (from ``build_distributions``), required by
    CatCmawm which takes an explicit search space up front.
    """
    # Size of the fully-random initial search phase before the model-based sampler engages. Derived
    # from pop_size (not a separate flag) so it stays consistent across all experimental conditions.
    n_startup_trials = pop_size

    if sampler_name == 'random':
        return optuna.samplers.RandomSampler(seed=seed)
    elif sampler_name == 'tpe':
        # constant_liar=False: evaluation is sequential (every trial is told before the next ask),
        # so there are never un-told running trials for the liar to account for. Each suggestion is
        # drawn from a TPE fit on the full archive of completed trials.
        return optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=n_startup_trials,
            n_ei_candidates=n_ei_candidates,
            constant_liar=False,
        )
    elif sampler_name == 'gp':
        # GPSampler is backed by PyTorch. On macOS torch bundles its own OpenMP runtime, which
        # clashes with the copy already loaded by numpy/scikit-learn ("OMP: Error #15") and
        # deadlocks the sampler. Pinning to a single-threaded OpenMP runtime before torch is
        # imported avoids the conflict; torch is imported lazily here so it stays an optional
        # dependency needed only for the GP sampler.
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        import torch
        torch.set_num_threads(1)
        return optuna.samplers.GPSampler(
            seed=seed,
            n_startup_trials=n_startup_trials,
        )
    elif sampler_name == 'cmaes':
        # Optuna's core CmaEsSampler does not support categorical parameters (its docs recommend
        # CatCmawm for mixed spaces), and every model here mixes categorical and numerical
        # parameters. We therefore use CatCmawmSampler (CatCMA-with-margin) from OptunaHub, giving
        # it the model's full search space up front so the CMA algorithm engages from trial 0
        # rather than spending a trial estimating the space. popsize=pop_size sets CatCMA's internal
        # population size; it accumulates that many one-at-a-time tells before updating its
        # distribution. The first load downloads the module from OptunaHub (needs network access)
        # and caches it locally.
        import optunahub
        catcmawm = optunahub.load_module("samplers/catcmawm")
        return catcmawm.CatCmawmSampler(
            search_space=distributions,
            seed=seed,
            popsize=pop_size,
            independent_sampler=optuna.samplers.RandomSampler(seed=seed),
        )
    else:
        raise ValueError(f"Unknown sampler '{sampler_name}'. Must be one of "
                         "['random', 'tpe', 'gp', 'cmaes'].")


def evaluate_candidate(ea: EA, params: dict) -> float:
    """
    Score a single hyperparameter configuration with the same 5-fold Ray cross-validation the EA
    uses, returning its mean validation AUC.

    The configuration's 5 CV folds are each launched as an independent Ray task (their preprocessed
    arrays already live in the Ray object store) and gathered in a single ``ray.get``, so the folds
    run in parallel; configurations themselves are evaluated one at a time. Parameters are run
    through the model's ``eval_parameters`` before training, exactly as in ea.py's evaluation. If a
    fold errors out the underlying CV function returns a 0.0 AUC for that fold, which drags the mean
    down and lets the study continue rather than crashing.
    """
    eval_params = ea.param_space.eval_parameters(params, random_state=ea.seed)
    cv_splits = ea.get_cv_splits()

    jobs = []
    for X_train, y_train, X_validate, y_validate in cv_splits:
        jobs.append(ea.ray_train_func.remote(
            X_train=X_train,
            y_train=y_train,
            X_validate=X_validate,
            y_validate=y_validate,
            model_params=eval_params,
            id=0,
            binary_class=ea.binary_classification,
            labels=ea.labels,
        ))

    results = ray.get(jobs)
    val_accs = [val_acc for (_id, _train_acc, val_acc, _error) in results]
    return float(np.mean(val_accs))


if __name__ == "__main__":
    # get configs for running the Optuna study
    parser = argparse.ArgumentParser(description="Run Optuna HPO")
    # random seed for reproducibility (drives datasplits AND the sampler)
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
    parser.add_argument('--model', type=str, required=True, choices=['RF', 'ET', 'KSVC', 'GB', 'KNN', 'MLP'], help='Model type to optimize.')
    # number of cores
    parser.add_argument('--cores', type=int, required=True, help='Number of CPU cores to use.')
    # which Optuna sampler to use
    parser.add_argument('--sampler', type=str, required=True, choices=['random', 'tpe', 'gp', 'cmaes'], help='Optuna sampler to use.')
    # total number of evaluations (trials)
    parser.add_argument('--n_trials', type=int, required=True, help='Total number of evaluations (Optuna trials).')
    # initial random-search size (evaluation is one-at-a-time; this is NOT a batch size)
    parser.add_argument('--pop_size', type=int, required=True, help='Size of the initial random-search phase: sets n_startup_trials (TPE/GP) and CatCMA popsize. Evaluation is sequential (one candidate at a time), so this is not a batch size.')
    # number of pseudo-offspring considered per TPE suggestion
    parser.add_argument('--n_ei_candidates', type=int, required=True, help='Number of pseudo-offspring (EI candidates) for the TPE sampler.')

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
    print(f"Seed: {args.seed}", flush=True)
    print(f"Task: {args.task}", flush=True)
    print(f"Y Label: {args.y_label}", flush=True)
    print(f"Data Path: {args.data_path}", flush=True)
    print(f"Train Proportion: {args.train_p}", flush=True)
    print(f"Output Directory: {args.output_directory}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Cores: {args.cores}", flush=True)
    print(f"Sampler: {args.sampler}", flush=True)
    print(f"Num Trials (evaluations): {args.n_trials}", flush=True)
    print(f"Pop Size (initial random-search size / n_startup_trials): {args.pop_size}", flush=True)
    print(f"Num EI Candidates (pseudo-offspring): {args.n_ei_candidates}", flush=True)
    print("=" * 60, flush=True)
    print('', flush=True)

    # initialize ray
    if not ray.is_initialized():
        ray.init(num_cpus=args.cores, include_dashboard=True, ignore_reinit_error=True)
    print(f"Ray initialized with {args.cores} cores.", flush=True)

    # Instantiate the HPO EA purely for its data/evaluation infrastructure. The TPE-specific
    # constructor arguments are placeholders (the EA's evolutionary loop is never run here).
    ea = EA(
        seed=args.seed,
        pop_size=1,
        cores=args.cores,
        mut_prob=0.0,
        mut_var=0.0,
        model=args.model,
        tpe_prob=0.0,
        tournament_size=1,
        num_offspring=1,
        crossover_prob=0.0,
        classification=True,
        gamma=0.5,
    )
    print("EA (data/eval infrastructure) initialized", flush=True)

    X, y = load_modeling_data(dataset_directory=args.data_path)

    print("Modeling data loaded", flush=True)
    print(f"Features shape: {X.shape}", flush=True)
    print(X.head(5), flush=True)
    print(f"Outcomes shape: {y.shape}", flush=True)
    print(y.head(5), flush=True)

    if args.task == 'so':
        # load spine opioid data
        data_set, one_hot_cols, scalar_cols = load_spine_opioid_data(X, y, y_label=args.y_label)
        print("Spine Opioid data loaded", flush=True)
    else:
        raise ValueError(f"Unknown task '{args.task}'. Supported tasks: 'so' (spine opioid).")

    # load data (builds the identical per-seed train/test split + 5-fold CV folds used by ea.py)
    ea.load_data_pd(
        data=data_set,
        target_label=args.y_label,
        train_p=args.train_p,
        one_hot_cols=one_hot_cols,
        scalar_cols=scalar_cols,
    )
    print("EA data loaded", flush=True)

    # the parameter space (from model_param_space.py) that the study will optimize over
    param_space = ea.param_space.param_space
    model_type = ea.param_space.get_model_type()
    distributions = build_distributions(param_space)

    # build the Optuna study (maximize mean validation AUC)
    sampler = build_sampler(
        sampler_name=args.sampler,
        seed=args.seed,
        pop_size=args.pop_size,
        n_ei_candidates=args.n_ei_candidates,
        distributions=distributions,
    )
    study = optuna.create_study(direction='maximize', sampler=sampler)
    print(f"Optuna study created with sampler '{args.sampler}'", flush=True)

    # Sequential ask/tell loop (see module docstring). Each iteration asks a single configuration
    # from the current sampler state -- passing the full search space so relative/joint samplers
    # (GP, CatCMA) sample the whole config at once -- evaluates it, and tells the result back before
    # the next ask, so the sampler is re-fit on the full archive of completed trials every step.
    n_trials = args.n_trials

    # validation score of every configuration fitted so far; a repeated configuration reuses its
    # stored score instead of being re-fitted, as the EA does, so both count candidates considered
    scored_configs = {}

    start_time = time.time()
    best_val_so_far = float('-inf')
    for trial_idx in range(n_trials):
        trial = study.ask(distributions)
        params = dict(trial.params)

        config_key = json.dumps(params, sort_keys=True)
        reused = config_key in scored_configs
        if reused:
            value = scored_configs[config_key]
        else:
            # evaluate this single candidate (its 5 CV folds still run in parallel via Ray)
            value = evaluate_candidate(ea, params)
            scored_configs[config_key] = value
        trial.set_user_attr("reused", reused)

        # tell the result back -> the sampler updates before the next ask
        study.tell(trial, value)
        hard_evals = trial_idx + 1

        if value > best_val_so_far:
            best_val_so_far = value
            print(f"Eval {hard_evals}/{n_trials} - NEW BEST Val AUC: {value:.4f}", flush=True)
        else:
            print(f"Eval {hard_evals}/{n_trials} - Val AUC: {value:.4f} "
                  f"(best {best_val_so_far:.4f})", flush=True)

    # save total runtime for the run (minutes)
    total_time_minutes = (time.time() - start_time) / 60
    print(f"Hard evaluations: {len(scored_configs)}", flush=True)
    print(f"Total optimization time (mins): {total_time_minutes}", flush=True)

    # reconstruct the best configuration and evaluate it on the held-out test set (same path
    # as ea.py: fit on the full training set, score train/test AUC)
    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_val = float(best_trial.value)
    # 1-based evaluation number at which the best configuration was found (trial.number is 0-based)
    best_eval = best_trial.number + 1
    print(f"Best configuration found at evaluation {best_eval}/{n_trials}", flush=True)

    train_score, test_score, complexity = ea.model_test_evaluation(
        HPOIndividual(best_params, model_type)
    )
    print(f"Final test evaluation - Train: {train_score}, Test: {test_score}, "
          f"Complexity: {complexity}", flush=True)

    # save results (same schema as ea.py's best_results.json, plus the sampler used)
    os.makedirs(args.output_directory, exist_ok=True)
    best_results = {
        "task_id": ea.task_id,
        "model_type": model_type,
        "seed": args.seed,
        "sampler": args.sampler,
        "train_accuracy": float(train_score),
        "validation_accuracy": best_val,
        "test_accuracy": float(test_score),
        "complexity": complexity,
        "best_eval": best_eval,
        "best_params": best_params,
    }
    json_path = os.path.join(args.output_directory, "best_results.json")
    with open(json_path, 'w') as f:
        json.dump(best_results, f, indent=4)
    print(f"Best results saved to: {json_path}", flush=True)

    # save every trial (params, validation AUC, whether its score was reused from a duplicate)
    trials_path = os.path.join(args.output_directory, "trials.csv")
    study.trials_dataframe().to_csv(trials_path, index=False)
    print(f"Trial history saved to: {trials_path}", flush=True)

    # save total runtime in .txt file
    runtime_path = os.path.join(args.output_directory, "total_runtime_minutes.txt")
    with open(runtime_path, 'w') as f:
        f.write(f"{total_time_minutes}")
    print(f"Total optimization time (mins) saved to: {runtime_path}", flush=True)

    # shutdown ray
    ray.shutdown()
