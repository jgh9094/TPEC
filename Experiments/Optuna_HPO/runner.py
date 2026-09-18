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
# GENERATIONAL (BATCHED) EVALUATION
# ---------------------------------
# To keep the comparison with ea.py fair, this runner does NOT use Optuna's default sequential
# study.optimize (which re-fits the sampler after every single evaluation). Instead it drives a
# manual ask/tell loop that mirrors the EA's generation structure: each generation asks
# --pop_size trials from the sampler *before telling any results back*, so every member of a
# generation is drawn from the sampler fit on only the *previously-completed* generations (the
# sampler state is frozen within a generation, exactly like ea.py's TPE which is fit once on the
# full archive of prior generations). The whole generation's pop_size configurations are then
# evaluated in parallel via Ray and told back together, updating the sampler once per generation.
# Total evaluations = --n_trials; number of generations = ceil(n_trials / pop_size).
#
# Per-sampler mapping to ea.py's generational behaviour:
#   tpe   : constant_liar=False so a generation's asks come from one frozen TPE fit (with
#           constant_liar=True Optuna would diversify within the batch, which the EA does not do).
#   cmaes : CatCMA is natively generational; popsize = pop_size makes one CMA population per
#           generation.
#   gp/random : batched purely for the generation structure + Ray parallelism.
# n_startup_trials is auto-set to pop_size for TPE/GP, so generation 0 is a fully-random
# population (matching the EA's random initial pop) and every later generation is model-driven.
# This is derived from pop_size (not a separate flag) so it stays consistent across conditions.
#
# The one remaining user-defined control (overlapping the EA's TPE machinery):
#   --n_ei_candidates  : number of pseudo-offspring the TPE sampler considers per suggestion
#                        (analogous to num_offspring in ea.py when TPE is enabled) (TPE only).
#
# CHECKPOINTING
# -------------
# Like ea.py, after every generation the best-so-far configuration is evaluated on the held-out
# test set and a diagnostic trail is written to --output_directory: a row is appended to
# checkpoints.csv and a best_results.json-shaped snapshot results_eval_{hard_evals}.json is saved.
# Generation 0 (the fully-random startup population) is logged as -1 to match ea.py's initial pop.
# Test evaluation is only refit when the best-so-far validation AUC improves.
#
##########################################################################################

import sys
import os
import json
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')))

import argparse
import ray
import optuna

# Reuse the HPO EA purely for its data/evaluation infrastructure: load_data_pd (which builds
# the per-seed train/test split + 5-fold CV folds), the model parameter space, the per-model
# Ray CV training function, and the final test-set evaluation. None of the EA's evolutionary
# machinery is exercised here.
from Source.HPO.ea import EA


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
    Construct an Optuna sampler configured for generational (batched) evaluation.

    ``pop_size`` is the generation/batch size and drives every sampler's "evals per update":
    ``n_startup_trials`` is auto-set to ``pop_size`` for TPE/GP (so generation 0 is a fully-random
    population, matching the EA's random initial pop), and CatCMA's ``popsize`` is set to
    ``pop_size`` (one CMA population per generation). ``n_ei_candidates`` ("pseudo-offspring") is
    applied to TPE only. The RandomSampler ignores these. All samplers are seeded with ``seed`` so
    a given seed reproduces both the datasplits and the search trajectory. ``distributions`` is the
    model's search space (from ``build_distributions``), required by CatCmawm which takes an
    explicit search space up front.
    """
    # One fully-random initial generation before the model-based sampler engages. Derived from
    # pop_size (not a separate flag) so it stays consistent across all experimental conditions.
    n_startup_trials = pop_size

    if sampler_name == 'random':
        return optuna.samplers.RandomSampler(seed=seed)
    elif sampler_name == 'tpe':
        # constant_liar=False so all pop_size asks in a generation are drawn from one frozen TPE
        # fit on the completed (prior-generation) trials -- the faithful analog of ea.py, where the
        # whole population is generated from a single TPE fit on the full prior archive. (With
        # constant_liar=True Optuna would treat the not-yet-told siblings as running trials and
        # diversify within the generation, which the EA does not do.)
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
        # rather than spending a trial estimating the space. popsize=pop_size makes CatCMA sample
        # exactly one population per generation, matching the EA's batch structure. The first load
        # downloads the module from OptunaHub (needs network access) and caches it locally.
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


def evaluate_batch(ea: EA, params_list: list) -> list:
    """
    Score a whole generation of hyperparameter configurations with the same 5-fold Ray
    cross-validation the EA uses, returning the mean validation AUC for each configuration
    (in the same order as ``params_list``).

    This launches every fold of every configuration as an independent Ray task up front (one task
    per (config, fold), each fold's preprocessed arrays already live in the Ray object store) and
    gathers them in a single ``ray.get``, so the entire generation's ``pop_size * 5`` folds are
    evaluated in parallel -- mirroring how ea.py evaluates an entire population per generation.
    Parameters are run through the model's ``eval_parameters`` before training, exactly as in
    ea.py's evaluation. If a fold errors out the underlying CV function returns a 0.0 AUC for that
    fold, which drags that config's mean down and lets the study continue rather than crashing.
    """
    all_jobs = []
    group_sizes = []
    for params in params_list:
        eval_params = ea.param_space.eval_parameters(params)
        cv_splits = ea.get_cv_splits()
        n = 0
        for X_train, y_train, X_validate, y_validate in cv_splits:
            all_jobs.append(ea.ray_train_func.remote(
                X_train=X_train,
                y_train=y_train,
                X_validate=X_validate,
                y_validate=y_validate,
                model_params=eval_params,
                random_state=ea.seed,
                id=0,
                binary_class=ea.binary_classification,
                labels=ea.labels,
            ))
            n += 1
        group_sizes.append(n)

    results = ray.get(all_jobs)

    means = []
    idx = 0
    for gsz in group_sizes:
        group = results[idx:idx + gsz]
        idx += gsz
        val_accs = [val_acc for (_id, _train_acc, val_acc, _error) in group]
        means.append(float(np.mean(val_accs)))
    return means


def checkpoint_best_seen(ea: EA, model_type: str, sampler_name: str, output_directory: str,
                         checkpoints: list, generation: int, hard_evals: int,
                         best_val: float, best_params: dict) -> None:
    """
    Record the test-set performance of the best-so-far configuration for one generation.

    This mirrors ea.py's ``EA.checkpoint_best_seen`` exactly (same files, same schema) so the
    Optuna baseline produces the same per-generation diagnostic trail as the TPE-EA: after every
    generation the best-validation configuration seen so far is evaluated on the held-out test set,
    a row is appended to ``{output_directory}/checkpoints.csv``, and a best_results.json-shaped
    snapshot is written to ``{output_directory}/results_eval_{hard_evals}.json``.

    To avoid redundant full-model refits, the best configuration is only re-evaluated on the test
    set when its validation performance improved since the previous checkpoint; otherwise the
    previous generation's train/test scores are carried forward (correct, since the best-so-far only
    changes when validation performance strictly improves).

    Args:
        generation (int): Generation index. Generation 0 of the runner is the fully-random startup
            population (the analog of ea.py's initial population), so it is logged as -1 to match
            ea.py; every later generation g is logged as g-1.
        hard_evals (int): Total number of evaluations (told trials) completed so far.
        best_val (float): Best mean validation AUC seen so far (``study.best_value``).
        best_params (dict): The configuration achieving ``best_val`` (``study.best_trial.params``).
    """
    # only refit/evaluate on the test set when the best-so-far actually changed
    if checkpoints and checkpoints[-1]["val_auc"] == best_val:
        train_score = checkpoints[-1]["train_auc"]
        test_score = checkpoints[-1]["test_auc"]
    else:
        train_score, test_score = ea.model_test_evaluation(
            model_type=model_type.upper(),
            model_params=best_params,
        )

    checkpoints.append({
        "generation": generation,
        "hard_evals": hard_evals,
        "val_auc": float(best_val),
        "train_auc": float(train_score),
        "test_auc": float(test_score),
    })

    os.makedirs(output_directory, exist_ok=True)
    csv_path = os.path.join(output_directory, "checkpoints.csv")
    pd.DataFrame(checkpoints).to_csv(csv_path, index=False)

    # snapshot of the best-so-far result at this many total evaluations, mirroring the
    # best_results.json schema (plus generation/hard_evals/sampler) for generation-by-generation
    # progress tracking
    result_snapshot = {
        "generation": generation,
        "hard_evals": hard_evals,
        "task_id": ea.task_id,
        "model_type": model_type,
        "seed": ea.seed,
        "sampler": sampler_name,
        "train_accuracy": float(train_score),
        "validation_accuracy": float(best_val),
        "test_accuracy": float(test_score),
        "best_params": best_params,
    }
    json_path = os.path.join(output_directory, f"results_eval_{hard_evals}.json")
    with open(json_path, 'w') as f:
        json.dump(result_snapshot, f, indent=4)

    print(f"Checkpoint (Gen {generation}) - Val AUC: {best_val:.4f}, "
          f"Train AUC: {train_score:.4f}, Test AUC: {test_score:.4f} "
          f"-> {json_path}", flush=True)

    return


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
    # generation / batch size (number of evaluations per sampler update, analogous to ea.py's pop_size)
    parser.add_argument('--pop_size', type=int, required=True, help='Generation (batch) size: number of trials asked and evaluated in parallel before the sampler is updated. Also auto-sets n_startup_trials (TPE/GP) and CatCMA popsize, so it is consistent across conditions.')
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
    print(f"Pop Size (generation/batch size, also n_startup_trials): {args.pop_size}", flush=True)
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
    )
    print("EA (data/eval infrastructure) initialized", flush=True)

    if args.task == 'so':
        # load spine opioid data
        data_set, one_hot_cols, scalar_cols = load_spine_opioid_data(data_path=args.data_path, y_label=args.y_label)
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

    # Generational ask/tell loop (see module docstring). Each generation asks pop_size trials
    # from the current sampler state -- passing the full search space so relative/joint samplers
    # (GP, CatCMA) sample the whole config at once -- BEFORE telling any results back, so every
    # member of a generation is drawn from the sampler fit on only the prior generations. The whole
    # generation is then evaluated in parallel via Ray and told back together, updating the sampler
    # once per generation, matching ea.py.
    n_trials = args.n_trials
    pop_size = args.pop_size
    n_generations = (n_trials + pop_size - 1) // pop_size  # ceil; last gen may be partial

    # per-generation checkpoints (mirrors ea.py): after each generation the best-so-far
    # configuration is evaluated on the test set and written to output_directory. Generation 0 of
    # the runner (the fully-random startup population) is logged as -1 to match ea.py's initial pop.
    checkpoints = []

    start_time = time.time()
    trial_counter = 0
    for gen in range(n_generations):
        batch = min(pop_size, n_trials - trial_counter)

        # ask a whole generation before telling anything back (frozen sampler state per generation)
        gen_trials = [study.ask(distributions) for _ in range(batch)]
        gen_params = [dict(t.params) for t in gen_trials]

        # evaluate the entire generation in parallel via Ray
        gen_values = evaluate_batch(ea, gen_params)

        # tell all results back together -> the sampler updates once for the next generation
        for t, v in zip(gen_trials, gen_values):
            study.tell(t, v)
            trial_counter += 1

        best_val_so_far = study.best_value
        gen_best = max(gen_values)
        print(f"Gen {gen} ({batch} evals, {trial_counter}/{n_trials} total) - "
              f"Gen best Val AUC: {gen_best:.4f} | Best Val AUC: {best_val_so_far:.4f}", flush=True)

        # checkpoint the best-so-far configuration's test performance for this generation
        checkpoint_best_seen(
            ea=ea,
            model_type=model_type,
            sampler_name=args.sampler,
            output_directory=args.output_directory,
            checkpoints=checkpoints,
            generation=gen - 1,  # runner gen 0 == ea.py's initial-pop Gen -1
            hard_evals=trial_counter,
            best_val=best_val_so_far,
            best_params=dict(study.best_trial.params),
        )

    print(f"Total optimization time (mins): {(time.time() - start_time) / 60}", flush=True)

    # reconstruct the best configuration and evaluate it on the held-out test set (same path
    # as ea.py: fit on the full training set, score train/test AUC)
    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_val = float(best_trial.value)

    train_score, test_score = ea.model_test_evaluation(
        model_type=model_type.upper(),
        model_params=best_params,
    )
    print(f"Final test evaluation - Train: {train_score}, Test: {test_score}", flush=True)

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
        "best_params": best_params,
    }
    json_path = os.path.join(args.output_directory, "best_results.json")
    with open(json_path, 'w') as f:
        json.dump(best_results, f, indent=4)
    print(f"Best results saved to: {json_path}", flush=True)

    # shutdown ray
    ray.shutdown()
