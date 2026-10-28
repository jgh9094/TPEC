##########################################################################################
#
# HPO (Hyperparameter Optimization) EA for single-model optimization.
# Inherits from BaseEA and provides TPE-guided selection of offspring for a single model type.
#
##########################################################################################

import numpy as np
import ray
import os
import time
import json
import pandas as pd

from typeguard import typechecked
from typing import Any, Dict, List, Optional, Tuple

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

from Source.Base.base_ea import BaseEA
from Source.Base.individual import Individual
from Source.HPO.individual import HPOIndividual
from Source.HPO.tpe import HPO_TPE
from Source.HPO.archive import HPOArchive
from Source.ML.complexity import calculate_model_complexity
from Source.Base.archive import (
    ArchiveEntry,
    RANDOM_CONSTRUCTION, TPE_CONSTRUCTION,
    MUTATION_OPERATION, CROSSOVER_OPERATION, CROSSOVER_MUTATION_OPERATION,
)

from Source.Base.model_param_space import DataContext
from Source.ML.classifiers import (
    RandomForestParams, ExtraTreesParams, KernelSVCParams, GradientBoostParams,
    KNeighborsClassifierParams, MLPClassifierParams, XGBClassifierParams, AnyLabelXGBClassifier,
)
from Source.ML.regressor import (
    RandomForestRegressorParams, ExtraTreesRegressorParams, KernelSVRParams,
    GradientBoostRegressorParams, KNeighborsRegressorParams, MLPRegressorParams, XGBRegressorParams,
)
from Source.HPO.cv_evaluation import (
    cv_random_forest, cv_extra_trees, cv_kernel_svc, cv_gradient_boost, cv_knn, cv_mlp, cv_xgboost,
    cv_random_forest_reg, cv_extra_trees_reg, cv_kernel_svr, cv_gradient_boost_reg,
    cv_knn_reg, cv_mlp_reg, cv_xgboost_reg,
)

# Validation performance forced onto an individual when any CV fold fails (a "complete penalty":
# see EA.evaluation). It must rank strictly worse than any real score so errored configs land in
# the TPE "bad" split and never seed the population. Classification maximizes ROC AUC in [0, 1],
# so 0.0 is already the worst; regression maximizes R^2 in (-inf, 1], where even 0.0 is a valid
# (mean-predicting) score and a genuinely bad regressor can score arbitrarily negative -- so we
# use -inf to guarantee errored configs rank below every real R^2. TPE's good/bad split is purely
# rank-based (Source/Base/tpe.py: split_samples sorts, never does arithmetic on the objective),
# so -inf is safe there; the archive JSON dump guards non-finite values to null (Archive.to_record).
CLASSIFICATION_ERROR_PENALTY = 0.0
REGRESSION_ERROR_PENALTY = -np.inf


@typechecked
class EA(BaseEA):
    """
    HPO (Hyperparameter Optimization) EA for single-model optimization.
    Extends BaseEA with TPE-guided selection of offspring for optimizing hyperparameters of a single model type.
    """

    def __init__(self,
                 seed: int,
                 pop_size: int,
                 cores: int,
                 mut_prob: float,
                 model: str,  # classification: ['RF','ET','KSVC','GB','KNN','MLP','XGB']; regression: ['RF','ET','SVR','GB','KNN','MLP','XGB']
                 tpe_prob: float,
                 tournament_size: int,
                 num_offspring: int,
                 crossover_prob: float,
                 classification: bool,
                 gamma: float = 0.0) -> None:
        """
        Initializes the HPO EA class with the provided parameters.

        Args:
            model (str): The type of model to optimize. For classification (``classification=True``)
                one of ['RF', 'ET', 'KSVC', 'GB', 'KNN', 'MLP', 'XGB']; for regression
                (``classification=False``) one of ['RF', 'ET', 'SVR', 'GB', 'KNN', 'MLP', 'XGB']. The
                kernel-SVM token differs by mode ('KSVC' for classification, 'SVR' for regression)
                and each is only valid in its own mode.
            tpe_prob (float): Probability of using TPE-based selection.
            seed (int): Random seed for reproducibility.
            gens (int): Number of generations to evolve.
            pop_size (int): Population size for the evolutionary algorithm.
            cores (int): Number of CPU cores to use for parallel processing.
            tournament_size (int): Tournament size for parent selection.
            mut_prob (float): Probability of mutating each hyperparameter.
            num_offspring (int): Number of pseudo-offspring to generate for TPE.
            output_dir (str): Directory for output files.
            classification (bool): True optimizes a classifier scored by ROC AUC; False optimizes
                a regressor scored by R^2. Selects the classifier vs. regressor parameter space and
                CV evaluation function, and (via BaseEA) how the data is split. For regression the
                target is used as-is -- pre-scale/transform it as needed.
            gamma (float): Gamma parameter for TPE.
        """
        # initialize the base class (classification vs. regression drives the data split there)
        super().__init__(
            seed=seed,
            pop_size=pop_size,
            cores=cores,
            mut_prob=mut_prob,
            crossover_prob=crossover_prob,
            classification=classification,
        )


        # store model type for deferred param_space creation (after load_data sets binary_classification).
        # The kernel-SVM token is mode-specific: 'KSVC' (classifier) vs 'SVR' (regressor); each is
        # only legal in its own mode, so the model must match the classification flag.
        valid_models = ['RF', 'ET', 'KSVC', 'GB', 'KNN', 'MLP', 'XGB'] if classification \
            else ['RF', 'ET', 'SVR', 'GB', 'KNN', 'MLP', 'XGB']
        if model not in valid_models:
            raise ValueError(
                f"Unknown model type '{model}' for {'classification' if classification else 'regression'}. "
                f"Must be one of {valid_models}."
            )
        # guard the two mode-specific kernel-SVM tokens against a mismatched classification flag
        assert not (model == 'SVR' and classification), \
            "'SVR' is a regression model; set classification=False to use it (use 'KSVC' for classification)."
        assert not (model == 'KSVC' and not classification), \
            "'KSVC' is a classification model; set classification=True to use it (use 'SVR' for regression)."
        self.model = model
        self.param_space = None
        self.ray_train_func = None

        # validation penalty applied to any individual with a failed CV fold, chosen so errored
        # configs rank strictly worst under the mode's objective (see the module-level constants)
        self.error_penalty = CLASSIFICATION_ERROR_PENALTY if classification else REGRESSION_ERROR_PENALTY

        # HPO-specific EA parameters
        self.tournament_size = tournament_size
        self.num_offspring = num_offspring

        # TPE-related parameters
        self.tpe_prob = tpe_prob
        self.tpe = HPO_TPE(gamma=gamma)

        # full provenance archive of every evaluated individual (generation, construction, ei,
        # error, genome, performances); built in load_data_pd once the model type is known,
        # saved by save_results, and fed (deduplicated by genome key) to the TPE surrogate. This
        # is the single source of history.
        self.eval_archive: Optional[HPOArchive] = None
        self.hard_eval_count = 0
        self.best_perf = float("-inf")
        # archive entry currently held as the best-so-far; re-drawn (with the archive's tie-break)
        # only when a generation strictly improves ``best_perf``, so ties found later never replace it
        self.best_entry: Optional[ArchiveEntry] = None
        # final selected result (genotype + train/val/test + complexity), populated at the end of
        # evolve() from the held ``best_entry`` and consumed by save_results()
        self.best_result: Optional[Dict[str, Any]] = None

        return

    def load_data_pd(self,
                    data: pd.DataFrame,
                    target_label: str,
                    train_p: float,
                    one_hot_cols: Optional[List[str]] = None,
                    scalar_cols: Optional[List[str]] = None,
                    n_folds: int = 5) -> None:
        """
        Loads data via parent class, then initializes param_space with correct binary_classification.

        Args:
            n_folds (int): Number of cross-validation folds to build (see BaseEA.load_data_pd).
                Defaults to 5.
        """
        super().load_data_pd(data, target_label, train_p, one_hot_cols, scalar_cols, n_folds=n_folds)

        # data is loaded; build the dataset context and create the param_space
        assert self.X_train is not None, "Data must be loaded before building the param space."
        # cap n_samples at the smallest CV training fold: bounds that scale with training rows
        # must be realizable on every fold, since each model is fit on k-1 folds during CV.
        # n_classes only shapes a classifier bound (GradientBoost's legal loss set); regression
        # has no classes (labels is None), so it is reported as 0.
        self.data_ctx = DataContext(
            n_samples=self.smallest_cv_train_size(),
            n_features=self.X_train.shape[1],
            n_classes=len(self.labels) if self.labels is not None else 0,
        )
        # (param_space, ray CV function) per model token. The classification and regression maps
        # share the RF/ET/GB/KNN/MLP tokens but bind different param spaces + CV funcs; the kernel
        # SVM differs by token too ('KSVC' -> SVC/AUC, 'SVR' -> SVR/R^2).
        if self.classification:
            model_configs = {
                'RF': (RandomForestParams(self.data_ctx), cv_random_forest),
                'ET': (ExtraTreesParams(self.data_ctx), cv_extra_trees),
                'KSVC': (KernelSVCParams(self.data_ctx), cv_kernel_svc),
                'GB': (GradientBoostParams(self.data_ctx), cv_gradient_boost),
                'KNN': (KNeighborsClassifierParams(self.data_ctx), cv_knn),
                'MLP': (MLPClassifierParams(self.data_ctx), cv_mlp),
                'XGB': (XGBClassifierParams(self.data_ctx), cv_xgboost),
            }
        else:
            model_configs = {
                'RF': (RandomForestRegressorParams(self.data_ctx), cv_random_forest_reg),
                'ET': (ExtraTreesRegressorParams(self.data_ctx), cv_extra_trees_reg),
                'SVR': (KernelSVRParams(self.data_ctx), cv_kernel_svr),
                'GB': (GradientBoostRegressorParams(self.data_ctx), cv_gradient_boost_reg),
                'KNN': (KNeighborsRegressorParams(self.data_ctx), cv_knn_reg),
                'MLP': (MLPRegressorParams(self.data_ctx), cv_mlp_reg),
                'XGB': (XGBRegressorParams(self.data_ctx), cv_xgboost_reg),
            }
        self.param_space, self.ray_train_func = model_configs[self.model]

        # the provenance archive rebuilds individuals from stored genomes, so it needs the model type
        self.eval_archive = HPOArchive(self.param_space.get_model_type())

    def evolve(self, gens: int) -> None:
        """
        Run the EA with parallelized fitness evaluations using Ray.

        Args:
            gens (int): Number of generations to evolve.
        """

        start_time = time.time()

        # Initialize population with random individuals
        self.initialize_population()

        # Evaluate initial population with Ray (evaluation counts hard evaluations and dedups
        # genomes already recorded in the archive)
        evaluated = self.evaluation(self.population)

        # record full provenance for the initial (random) population as generation -1 (this is the
        # sole history store; the TPE surrogate is fit from it, deduplicated by genome key)
        self.record_history(evaluated, generation=-1)

        # errored individuals are archived above but must not seed the next generation
        self.population = self._drop_errored(evaluated)

        # keep track of best
        self.update_best_seen(self.population)
        print(f"Best performance so far (Gen -1): {self.best_perf}", flush=True)

        # Start evolution
        for g in range(gens):
            # Decide, per offspring, whether it comes from mutation or crossover. This is done
            # before parent selection because it dictates how many parents must be picked
            # (mutation consumes 1 parent, crossover consumes 2).
            var_order, num_parents = self.variation_order(self.pop_size, self.crossover_prob)

            # Parent selection with tournament selection
            parent_ids = self.parent_selection(self.population, num_parents, self.rng)

            # Generate offspring via mutation and crossover according to the variation order
            offspring = self.generate_offspring(self.population, parent_ids, var_order)

            # Evaluate offspring with Ray (dedups previously-seen genomes, counts hard evaluations)
            evaluated = self.evaluation(offspring)

            # record full provenance for this generation's offspring
            self.record_history(evaluated, generation=g)

            # errored offspring are archived above but must not seed the next generation
            self.population = self._drop_errored(evaluated)

            # keep track of best
            self.update_best_seen(self.population)
            print(f"Best performance so far (Gen {g}): {self.best_perf}", flush=True)

        # save total runtime for the run (minutes)
        self.total_time_minutes = (time.time() - start_time) / 60

        # make sure that the archive is the correct size
        print(f"Hard evaluations: {self.hard_eval_count}", flush=True)
        print(f"Total evolution time (mins): {self.total_time_minutes}", flush=True)

        # the best validation performance found across the whole run
        print(f"Best validation performance found: {self.best_perf}", flush=True)

        # The returned individual is the held best entry (``_select_best``), fixed whenever the best
        # score last strictly improved. Fit it on the full training set and score it on the test set.
        best_individual, best_val = self._select_best()
        train_score, test_score, complexity = self.model_test_evaluation(best_individual)
        # 1-based position of the best entry among all candidates considered (duplicates included),
        # matching the Optuna runner's best_eval (trial.number + 1)
        best_eval = self.best_entry.id + 1
        # generation that produced the best entry, as recorded in archive.json (-1 = initial population)
        best_generation = self.best_entry.generation
        self.best_result = {
            "genotype": best_individual.get_genotype(),
            "train_performance": train_score,
            "val_performance": best_val,
            "test_performance": test_score,
            "complexity": complexity,
            "best_eval": best_eval,
            "best_generation": best_generation,
        }

        print(f"Final test evaluation - Train: {train_score}, Val: {self.best_perf}, "
              f"Test: {test_score}, Complexity: {complexity}", flush=True)
        print(f"Best individual found at evaluation {best_eval}/{len(self.eval_archive)} "
              f"(generation {best_generation})", flush=True)

        return

    def initialize_population(self) -> None:
        """
        Initializes the starting population for the HPO EA.
        Creates random individuals using the single model's parameter space.
        """
        assert len(self.population) == 0, "Population has already been initialized."

        self.population = []
        for _ in range(self.pop_size):
            ind = HPOIndividual(
                self.param_space.generate_random_parameters(self.rng),
                self.param_space.get_model_type()
            )
            # the initial population is drawn at random (no TPE guidance)
            ind.construction = RANDOM_CONSTRUCTION
            self.population.append(ind)
        return

    def parent_selection(self, population: List[Individual], num_parents: int, rng: np.random.Generator) -> List[int]:
        """
        Select parents from the population using tournament selection.

        Parameters:
            population (List[Individual]): The population of individuals.
            num_parents (int): The number of parents to select.
            rng (np.random.Generator): Random number generator for reproducibility.

        Returns:
            List[int]: Indices of the selected parent individuals.
        """
        assert len(population) > 0, "Population must not be empty."

        # the pool can shrink below the tournament size once errored individuals are removed, so
        # clamp the tournament to the available pool (sampling without replacement needs k <= n)
        k = min(self.tournament_size, len(population))
        parent_ids = []
        for _ in range(num_parents):
            indices = rng.choice(len(population), k, replace=False)
            extracted_performances = np.array([population[i].get_val_performance() for i in indices])
            best_tour_idx = np.argmax(extracted_performances)
            winner = int(rng.choice([i for i, perf in zip(indices, extracted_performances) if perf == extracted_performances[best_tour_idx]]))
            parent_ids.append(winner)

        return parent_ids

    def evaluation(self, candidates: List[Individual]) -> List[Individual]:
        """
        Evaluate a collection of individuals using Ray across 5-fold cross-validation.
        This method will update each individual's train and validation performance.
        Uses the same per-fold CV data structure as CASH EA via get_cv_splits() from BaseEA.

        One Ray task is launched per (individual, fold) pair so that every fold of every
        pipeline is evaluated in parallel. Each fold's preprocessed data already lives in
        the Ray object store (see BaseEA._prepare_cv_folds), so a task loads only the single
        fold it needs. As results stream back, per-fold performances are tracked for each
        pipeline and its mean CV performance is finalized as soon as all of its folds arrive.

        Args:
            candidates (List[Individual]): List of individuals to evaluate.

        Returns:
            List[Individual]: The evaluated individuals with updated performance metrics.
        """
        # get per-fold CV data (each fold's arrays live in the Ray object store)
        cv_splits = self.get_cv_splits()
        num_folds = len(cv_splits)

        # Deduplicate before evaluating: a genome is deterministic under a fixed seed + CV, so any
        # individual whose genome was already evaluated (recorded in the provenance archive, or seen
        # earlier in THIS batch) reuses those performances instead of being re-evaluated. Only the
        # first sighting of each distinct genome is dispatched to Ray as a hard evaluation.
        pending = self._resolve_duplicates(candidates)

        # launch one Ray task per (pending individual, fold) so every fold evaluates in parallel
        ray_jobs = []
        for model_id, ind in enumerate(pending):
            # Seed is folded into eval_params (for stochastic estimators) so the Ray worker can
            # build the estimator straight from model_params without a separate random_state arg.
            eval_params = self.param_space.eval_parameters(ind.get_genotype(), random_state=self.seed)
            for X_train, y_train, X_validate, y_validate in cv_splits:
                ray_jobs.append(self.ray_train_func.remote(
                    X_train=X_train,
                    y_train=y_train,
                    X_validate=X_validate,
                    y_validate=y_validate,
                    model_params=eval_params,
                    id=model_id,
                    binary_class=self.binary_classification,
                    labels=self.labels
                ))

        # accumulate fold performances per pending individual as results arrive; track whether
        # any fold errored so the individual can be flagged (error < 0.0 -> the fold returns 0.0/0.0)
        pop_results = [{'train_acc': [], 'val_acc': [], 'error': False} for _ in pending]

        while len(ray_jobs) > 0:
            finished, ray_jobs = ray.wait(ray_jobs, num_returns=min(len(ray_jobs), self.cores))
            for done_id in finished:
                model_id, train_acc, val_acc, error = ray.get(done_id)

                # a fold that failed to fit is not fatal here: it returns 0.0/0.0 (status -1.0) and
                # flags the individual as errored. Any fold failure penalizes the whole individual
                # completely (see finalize below), so it cannot survive into the population.
                if error < 0.0:
                    pop_results[model_id]['error'] = True
                    print(f"Model {model_id} failed on a fold (fitness penalized to 0.0).", flush=True)

                # track this fold's performance for the corresponding pipeline as it comes in
                pop_results[model_id]['train_acc'].append(train_acc)
                pop_results[model_id]['val_acc'].append(val_acc)

                # once all folds for this individual are in, finalize its CV performance. Any fold
                # failure is a complete penalty (0.0/0.0), not an average that could still look
                # competitive; only an all-fold success keeps its true mean CV performance.
                if len(pop_results[model_id]['val_acc']) == num_folds:
                    errored = pop_results[model_id]['error']
                    # a failed fold is a complete penalty to the mode's worst score (self.error_penalty:
                    # 0.0 for classification AUC, a large negative for regression R^2); otherwise the
                    # individual keeps its true mean CV performance.
                    mean_train = self.error_penalty if errored else float(np.mean(pop_results[model_id]['train_acc']))
                    mean_val = self.error_penalty if errored else float(np.mean(pop_results[model_id]['val_acc']))
                    pending[model_id].set_train_performance(mean_train)
                    pending[model_id].set_val_performance(mean_val)
                    pending[model_id].eval_error = errored
                    print(f"Model {model_id} evaluated - Train {self.metric_name}: {mean_train:.4f}, Val {self.metric_name}: {mean_val:.4f}", flush=True)

        # in-batch duplicates: individuals that shared a (not-previously-seen) genome with a
        # pending sibling were not dispatched, so copy the freshly evaluated result onto them.
        if self.eval_archive is not None:
            evaluated_by_key = {self.eval_archive.compute_key(ind): ind for ind in pending}
            for ind in candidates:
                if ind.val_performance is None:
                    src = evaluated_by_key[self.eval_archive.compute_key(ind)]
                    ind.set_train_performance(src.get_train_performance())
                    ind.set_val_performance(src.get_val_performance())
                    ind.eval_error = src.eval_error

        # every distinct genome that reached Ray is one hard (real model-fitting) evaluation
        self.hard_eval_count += len(pending)
        return candidates

    def _reuse_archived_performance(self, ind: Individual) -> bool:
        """
        If ``ind``'s genome was already evaluated (recorded in the provenance archive), copy the
        stored train/validation/error onto it and return True; otherwise return False.

        Performances are genome-deterministic (fixed seed + CV folds), so a previously evaluated
        genome need not be re-fit -- this is what lets the EA skip redundant hard evaluations.

        Args:
            ind (Individual): The individual to (possibly) populate from the archive.

        Returns:
            bool: True if archived results were reused (``ind`` is now evaluated), else False.
        """
        if self.eval_archive is None:
            return False
        for entry in self.eval_archive.entries_for(ind):
            if entry.train_performance is not None and entry.val_performance is not None:
                ind.set_train_performance(entry.train_performance)
                ind.set_val_performance(entry.val_performance)
                ind.eval_error = entry.error
                return True
        return False

    def _drop_errored(self, evaluated: List[Individual]) -> List[Individual]:
        """
        Return the breeding population: the evaluated individuals with any that errored removed.

        Errored individuals are still archived (for provenance and dedup) but must not seed the
        next generation. If every individual errored there is nothing to breed from, so the full
        set is kept (with a warning) rather than returning an empty population.

        Args:
            evaluated (List[Individual]): The just-evaluated individuals.

        Returns:
            List[Individual]: The survivors (or all of ``evaluated`` if none survived).
        """
        survivors = [ind for ind in evaluated if not ind.eval_error]
        if not survivors:
            print("All individuals errored this generation; keeping them so the run can continue.", flush=True)
            return evaluated
        if len(survivors) < len(evaluated):
            print(f"Removed {len(evaluated) - len(survivors)} errored individual(s) from the population.", flush=True)
        return survivors

    def _resolve_duplicates(self, candidates: List[Individual]) -> List[Individual]:
        """
        Partition ``candidates`` for evaluation: reuse archived results for genomes already seen in
        a previous generation, and collapse genomes repeated within this batch to a single sighting.

        Individuals whose genome is in the archive are populated in place (via
        :meth:`_reuse_archived_performance`) and excluded from the returned list. The returned list
        holds the first occurrence of each remaining distinct genome -- the ones that must be hard
        evaluated; later in-batch duplicates are filled in after evaluation by the copy pass in
        :meth:`evaluation`.

        Args:
            candidates (List[Individual]): The individuals about to be evaluated.

        Returns:
            List[Individual]: The individuals that still require a hard evaluation.
        """
        pending: List[Individual] = []
        seen_keys = set()
        for ind in candidates:
            if self._reuse_archived_performance(ind):
                continue
            if self.eval_archive is None:
                pending.append(ind)
                continue
            key = self.eval_archive.compute_key(ind)
            if key not in seen_keys:
                seen_keys.add(key)
                pending.append(ind)
            # else: duplicate of a pending genome in this same batch -> filled in post-evaluation
        return pending

    def generate_offspring(self, candidates: List[Individual], parent_ids: List[int], variation_order: List[str]) -> List[Individual]:
        """
        Generate offspring from selected parents according to a precomputed variation order.

        Each entry in ``variation_order`` is either 'm' (mutation, consumes 1 parent) or 'c'
        (crossover, consumes 2 parents); ``parent_ids`` is consumed left-to-right in that order.
        Every offspring rolls for TPE-guided selection (probability ``self.tpe_prob``):

          - Mutation: on a TPE roll, ``num_offspring`` candidates are generated by randomly
            resampling the parent's parameters (``mutate_parameters_random``) and the best is chosen
            via TPE; otherwise a single random candidate is produced.
          - Crossover: each candidate is produced by recombining the two parents (uniform crossover)
            and then, with probability ``self.mut_prob``, randomly mutating the recombined child. On a
            TPE roll the best of ``num_offspring`` candidates is chosen via TPE; otherwise a single
            candidate is produced.

        Args:
            candidates (List[Individual]): The current population (indexed by ``parent_ids``).
            parent_ids (List[int]): Indices of selected parents, ordered to match consumption.
            variation_order (List[str]): Per-offspring operators ('m' or 'c') from variation_order.

        Returns:
            List[Individual]: List of offspring individuals (one per entry in variation_order).
        """
        expected_parents = sum(1 if op == 'm' else 2 for op in variation_order)
        assert len(parent_ids) == expected_parents, "Number of parent IDs must match the parents required by the variation order."
        assert len(variation_order) > 0, "At least one offspring must be generated."
        assert self.eval_archive is not None and len(self.eval_archive) > 0, \
            "The provenance archive must hold at least one evaluation for TPE-based mutation."

        # store offspring here
        offspring = []

        # fit tpe model if self.tpe_prob > 0
        if self.tpe_prob > 0.0:
            self.tpe.fit(self._tpe_samples(), self.param_space, self.rng)

        # consume parents left-to-right as dictated by the variation order. Each variation
        # operator (mutate / crossover) owns its own TPE decision and offspring generation.
        parent_cursor = 0
        for op in variation_order:
            if op == 'm':
                parent = candidates[parent_ids[parent_cursor]]
                parent_cursor += 1
                offspring.append(self.mutate(parent))
            else:  # 'c' -> crossover of two parents
                parent_a = candidates[parent_ids[parent_cursor]]
                parent_b = candidates[parent_ids[parent_cursor + 1]]
                parent_cursor += 2
                offspring.append(self.crossover(parent_a, parent_b))

        assert parent_cursor == len(parent_ids), "All selected parents must be consumed."
        assert len(offspring) == len(variation_order), "Number of offspring must match the variation order length."
        return offspring

    def _tpe_samples(self) -> List[Individual]:
        """
        Build the TPE fitting set from the provenance archive, one sample per UNIQUE genotype.

        The provenance archive (``self.eval_archive``) logs every production, including duplicates
        and errored individuals. Feeding duplicates to the surrogate would over-weight repeatedly
        produced genomes in the good/bad split, so we keep only the first entry seen per genome
        key; because evaluation is genome-deterministic, all entries sharing a key carry identical
        performances, so the choice of representative is immaterial.

        Each kept entry becomes one HPOIndividual whose genome is passed through
        ``param_space.tpe_parameters`` (the model-specific TPE encoding) and which carries the
        NEGATED validation performance, because HPO_TPE treats the objective as a minimization (its
        good/bad split takes the lowest values as "good") while the EA maximizes validation score.
        Errored individuals were hard-penalized to ``self.error_penalty`` (0.0 for classification
        AUC, -inf for regression R^2), so they negate to the largest objective value and reliably
        land in the "bad" group, steering the surrogate away from them.
        """
        assert self.eval_archive is not None, "Data must be loaded before fitting TPE."

        model_type = self.param_space.get_model_type()
        samples: List[Individual] = []
        seen_keys = set()
        for entry in self.eval_archive:
            if entry.key in seen_keys:
                continue
            seen_keys.add(entry.key)
            assert entry.val_performance is not None, \
                "Archived evaluation is missing a validation score (errored individuals are penalized to self.error_penalty, not None)."
            tpe_ind = HPOIndividual(self.param_space.tpe_parameters(entry.genome), model_type)
            tpe_ind.set_val_performance(entry.val_performance * -1.0)  # TPE minimizes, so invert
            samples.append(tpe_ind)
        return samples

    def record_history(self, individuals: List[Individual], generation: int) -> None:
        """
        Record every evaluated individual in the provenance archive (``self.eval_archive``).

        Each individual carries a ``construction`` tag (random vs TPE, set when it was created)
        and an ``eval_error`` flag (set during evaluation); TPE-constructed individuals also carry
        their expected-improvement score in ``ind.ei``. Random individuals are archived with the
        canonical ``-inf`` ei (the archive enforces this), so only TPE offspring pass an ei.
        Offspring additionally carry a variation ``operation`` tag and their parents'
        ``parent_ids`` (both None for the initial population). Each individual's assigned archive
        id is written back to ``ind.archive_id`` so its own offspring can reference it as a parent.

        Args:
            individuals (List[Individual]): The just-evaluated individuals to record.
            generation (int): Generation that produced them (-1 for the initial population).
        """
        assert self.eval_archive is not None, "Data must be loaded before recording history."
        for ind in individuals:
            assert ind.construction is not None, "Individual is missing its construction tag."
            assert ind.eval_error is not None, "Individual was not evaluated (eval_error unset)."
            ei = ind.ei if ind.construction == TPE_CONSTRUCTION else None
            entry = self.eval_archive.add(
                ind,
                generation=generation,
                construction=ind.construction,
                error=ind.eval_error,
                ei=ei,
                operation=ind.operation,
                parent_ids=ind.parent_ids,
            )
            # remember the id this individual was stored under so its offspring can cite it
            ind.archive_id = entry.id
        return

    def update_best_seen(self, individuals: List[Individual]) -> None:
        """
        Update the best validation performance seen across all generations, and the held best entry.

        The whole batch is scanned first; only if some individual strictly beats ``best_perf`` is the
        held best re-drawn from the provenance archive (``Archive.best``: top validation, earliest
        generation, then a uniform random tie-break with ``self.rng``). A later individual that merely ties
        the current best never replaces it, so the held best is the first one to reach that score.
        Call this after ``record_history`` so the batch is already archived.

        Args:
            individuals (List[Individual]): Individuals to check.
        """
        improved = False
        for ind in individuals:
            perf = ind.get_val_performance()
            if perf > self.best_perf:
                self.best_perf = perf
                improved = True

        if improved:
            assert self.eval_archive is not None, "No archive to select the best individual from."
            self.best_entry = self.eval_archive.best(self.rng)
            # consistency guard: the archive's best validation must match the tracked best-so-far
            assert self.best_entry is not None and self.best_entry.val_performance == self.best_perf, (
                f"Archive best validation "
                f"({None if self.best_entry is None else self.best_entry.val_performance}) does not "
                f"match the best validation performance tracked during the run ({self.best_perf})."
            )
            # alignment guard: a strict improvement can only come from this batch, so the archive's
            # pick must be an entry recorded for this batch. ``tied`` holds every individual in the
            # batch that reached the new best; several may tie, which is expected.
            tied = [ind for ind in individuals if ind.get_val_performance() == self.best_perf]
            # id check: record_history gives every individual in the batch its own fresh archive id,
            # so exactly one tied individual must carry the drawn entry's id (whatever the number of
            # ties). 0 matches -> the archive drew an entry outside this batch (e.g. an earlier
            # genome-identical entry); 2+ matches -> two individuals share an id (e.g. the same
            # object appears twice in the batch, so its id was overwritten).
            matched = [ind for ind in tied if ind.archive_id == self.best_entry.id]
            assert len(matched) == 1, (
                f"Archive best (id {self.best_entry.id}, generation {self.best_entry.generation}) "
                f"matches {len(matched)} of the {len(tied)} individual(s) in this batch that tied the "
                f"new best validation performance ({self.best_perf}); expected exactly one "
                f"(tied archive ids: {[ind.archive_id for ind in tied]})."
            )
            # genome check: the matched individual's current genome must still equal the genome
            # the archive deep-copied at record time; a mismatch means it was modified afterwards.
            assert self.eval_archive.compute_key(matched[0]) == self.best_entry.key, (
                f"Individual with archive id {self.best_entry.id} no longer carries the genome "
                f"recorded in the archive."
            )
        return

    def _select_best(self) -> Tuple[HPOIndividual, float]:
        """
        Return the held best-so-far individual (rebuilt from ``self.best_entry``, with its validation
        performance set) alongside its validation score.

        No draw happens here: the tie-break is made once, in ``update_best_seen``, when the best
        score strictly improves, so selecting the final result does not consume ``self.rng``.
        """
        assert self.eval_archive is not None, "No archive to select the best individual from."
        assert self.best_entry is not None and self.best_entry.val_performance is not None, \
            "No best entry held yet; update_best_seen must run on a scored batch first."
        best_val = float(self.best_entry.val_performance)
        best_individual = self.eval_archive.build_individual(self.best_entry)
        best_individual.set_val_performance(best_val)
        return best_individual, best_val

    def mutate(self, parent: Individual) -> HPOIndividual:
        """
        Produce one offspring from a single parent by random mutation.

        Rolls for TPE-guided selection (probability ``self.tpe_prob``):
          - non-TPE: a single offspring produced by a random resample of the parent's parameters
            (``mutate_parameters_random``).
          - TPE: generate ``self.num_offspring`` random candidates and keep the one the TPE surrogate
            ranks best.

        Args:
            parent (Individual): The parent whose hyperparameters are mutated.

        Returns:
            HPOIndividual: A new offspring individual.
        """
        child_params, operation, construction, ei = self._tpe_or_explore(
            tpe_candidate=lambda: (self._mutate_params(parent.get_genotype()), MUTATION_OPERATION),
            explore_candidate=lambda: (self._mutate_params(parent.get_genotype()), MUTATION_OPERATION),
        )
        # a mutation-only offspring has a single parent, so both parent slots reference it
        return self._build_offspring(child_params, construction, ei, operation, (parent, parent))

    def crossover(self, parent_a: Individual, parent_b: Individual) -> HPOIndividual:
        """
        Produce one offspring from two parents by crossover.

        Rolls for TPE-guided selection (probability ``self.tpe_prob``):
          - non-TPE: recombine the parents (uniform crossover) and, with probability
            ``self.mut_prob``, apply a random mutation to the child; keep that single candidate.
          - TPE: generate ``self.num_offspring`` candidates, each a recombined child that (with
            probability ``self.mut_prob``) receives a random mutation, then keep the one the TPE
            surrogate ranks best.

        Both parents share the same model type in HPO (a single model is optimized), so the
        offspring keeps that model type.

        Args:
            parent_a (Individual): The first parent.
            parent_b (Individual): The second parent.

        Returns:
            HPOIndividual: A new offspring individual.
        """
        assert parent_a.model_type == parent_b.model_type, "Parents must share the same model type for crossover."
        child_params, operation, construction, ei = self._tpe_or_explore(
            tpe_candidate=lambda: self._crossover_child(parent_a, parent_b),
            explore_candidate=lambda: self._crossover_child(parent_a, parent_b),
        )
        return self._build_offspring(child_params, construction, ei, operation, (parent_a, parent_b))

    def _build_offspring(self, params: dict, construction: str, ei: float,
                         operation: str, parents: Tuple[Individual, Individual]) -> HPOIndividual:
        """
        Wrap a child genotype in an HPOIndividual, tagging it with its construction provenance
        (random vs TPE), its variation operator, and its parents' archive ids -- and, for TPE
        offspring, its expected-improvement score. All are consumed by :meth:`record_history` when
        the individual is later archived.

        Args:
            params (dict): The child's hyperparameters.
            construction (str): RANDOM_CONSTRUCTION or TPE_CONSTRUCTION.
            ei (float): Expected improvement of the chosen candidate (-inf for random offspring).
            operation (str): The variation operator that produced the child (MUTATION_OPERATION,
                CROSSOVER_OPERATION, or CROSSOVER_MUTATION_OPERATION).
            parents (Tuple[Individual, Individual]): The two parents (identical objects for a
                mutation-only offspring); their archive ids are recorded as the child's lineage.

        Returns:
            HPOIndividual: The tagged offspring.
        """
        assert parents[0].archive_id is not None and parents[1].archive_id is not None, \
            "Parents must already be archived (carry an archive_id) before producing offspring."
        child = HPOIndividual(params, self.param_space.get_model_type())
        child.construction = construction
        if construction == TPE_CONSTRUCTION:
            child.set_ei(ei)
        child.operation = operation
        child.parent_ids = (parents[0].archive_id, parents[1].archive_id)
        return child

    def _tpe_or_explore(self, tpe_candidate, explore_candidate) -> Tuple[dict, str, str, float]:
        """
        Shared TPE decision used by both variation operators. Rolls for TPE-guided variation
        (probability ``self.tpe_prob``): on a TPE roll, generate ``self.num_offspring`` candidates
        via ``tpe_candidate`` and return the one the TPE surrogate ranks best; otherwise return a
        single candidate from ``explore_candidate``.

        Each candidate factory returns a ``(params, operation)`` pair so the chosen candidate's
        variation operator (which, for crossover, records whether its per-candidate mutation gate
        fired) is reported alongside the winning genotype.

        Args:
            tpe_candidate (Callable[[], Tuple[dict, str]]): Factory for a TPE-mode candidate
                ``(hyperparameters, operation)`` (called ``self.num_offspring`` times on a TPE roll).
            explore_candidate (Callable[[], Tuple[dict, str]]): Factory for a single
                unbiased-exploration candidate ``(hyperparameters, operation)`` (called once otherwise).

        Returns:
            Tuple[dict, str, str, float]: The chosen candidate's hyperparameters, its variation
            operator, its construction tag (``TPE_CONSTRUCTION`` or ``RANDOM_CONSTRUCTION``), and its
            expected-improvement score (the surrogate's acquisition value for a TPE pick, ``-inf``
            for a random pick).
        """
        if self.rng.random() < self.tpe_prob:
            candidate_offspring = [tpe_candidate() for _ in range(self.num_offspring)]
            encoded = [self.param_space.tpe_parameters(params) for params, _ in candidate_offspring]
            candidate_index = self.tpe.suggest_one(encoded, self.rng)
            # the acquisition score of the chosen candidate is its expected improvement
            ei = float(self.tpe.score_candidates([encoded[candidate_index]])[0])
            params, operation = candidate_offspring[candidate_index]
            return params, operation, TPE_CONSTRUCTION, ei
        params, operation = explore_candidate()
        return params, operation, RANDOM_CONSTRUCTION, float("-inf")

    def _mutate_params(self, params: dict) -> dict:
        """
        Mutate a hyperparameter dict once by resampling each gene uniformly across the parameter
        space (``mutate_parameters_random``, per-gene rate ``self.mut_prob``).

        Args:
            params (dict): The hyperparameters to mutate ({parameter_name: value}).

        Returns:
            dict: The mutated hyperparameters.
        """
        return self.param_space.mutate_parameters_random(params, self.mut_prob, self.rng)

    def _crossover_child(self, parent_a: Individual, parent_b: Individual) -> Tuple[dict, str]:
        """
        Produce a single recombined child genotype: uniform crossover of the two parents, then
        (with probability ``self.mut_prob``) a random mutation.

        Also reports the variation operator for THIS candidate: ``CROSSOVER_MUTATION_OPERATION`` if
        the mutation gate fired, else ``CROSSOVER_OPERATION``. This is per-candidate because a TPE
        roll generates several candidates (each rolling the gate independently) and only the chosen
        one's operator is recorded.

        Args:
            parent_a (Individual): The first parent.
            parent_b (Individual): The second parent.

        Returns:
            Tuple[dict, str]: The recombined child's hyperparameters ({parameter_name: value}) and
            its variation operator.
        """
        child_params = self._uniform_crossover(parent_a, parent_b)

        # offspring-level mutation gate: with probability mut_prob, mutate the crossover child
        if self.rng.random() < self.mut_prob:
            child_params = self._mutate_params(child_params)
            return child_params, CROSSOVER_MUTATION_OPERATION
        return child_params, CROSSOVER_OPERATION

    def _uniform_crossover(self, parent_a: Individual, parent_b: Individual) -> dict:
        """
        Uniform per-hyperparameter recombination: for each hyperparameter, inherit the value from
        either parent with equal probability. Returns the recombined parameter dict (no mutation).

        Args:
            parent_a (Individual): The first parent.
            parent_b (Individual): The second parent.

        Returns:
            dict: The recombined hyperparameters ({parameter_name: value}).
        """
        params_a = parent_a.get_genotype()
        params_b = parent_b.get_genotype()
        assert params_a.keys() == params_b.keys(), "Parents must share the same hyperparameter set for crossover."

        return {
            name: (params_a[name] if self.rng.random() < 0.5 else params_b[name])
            for name in params_a
        }

    def _build_preprocessor(self) -> ColumnTransformer:
        """
        Build the HPO base preprocessor: since a single bare estimator is fit directly on this
        output, the data must be both numericized and scaled. Numeric columns are StandardScaled
        and categorical columns are one-hot-encoded; transformations whose column list is empty are
        omitted, and any remaining columns pass through unchanged.

        Returns:
            ColumnTransformer: The configured preprocessor.
        """
        transformers = []
        if self.numerical_cols:
            transformers.append(('num', StandardScaler(), self.numerical_cols))
        if self.categorical_cols:
            transformers.append(('cat', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'), self.categorical_cols))

        return ColumnTransformer(transformers=transformers, remainder='passthrough')

    def model_test_evaluation(self, individual: Individual) -> Tuple[float, float, int]:
        """
        Evaluates an individual's model on the test dataset after fitting on the full training set.

        Args:
            individual (Individual): The individual to evaluate. Its genotype provides the
                hyperparameters and its model_type selects the estimator.

        Returns:
            Tuple[float, float, int]: (train_score, test_score, complexity). Complexity scores the
            fitted model only; the fixed base preprocessor is not part of the search.
        """
        assert self.X_train is not None and self.X_test is not None, "Data must be loaded before evaluation."
        assert self.y_train is not None and self.y_test is not None, "Data must be loaded before evaluation."

        model_type = individual.model_type.upper()
        model_params = individual.get_genotype()

        preprocessor = self._build_preprocessor()

        X_train_preprocessed = preprocessor.fit_transform(self.X_train)
        X_test_preprocessed = preprocessor.transform(self.X_test)

        if self.classification:
            model = self._build_classifier(model_type, model_params)
        else:
            model = self._build_regressor(model_type, model_params)

        model.fit(X_train_preprocessed, self.y_train)

        if self.classification:
            assert self.binary_classification is not None, "Classification data must be loaded before evaluation."
            train_pred_proba = model.predict_proba(X_train_preprocessed)
            test_pred_proba = model.predict_proba(X_test_preprocessed)

            if self.binary_classification:
                train_score = float(roc_auc_score(self.y_train, train_pred_proba[:, 1]))
                test_score = float(roc_auc_score(self.y_test, test_pred_proba[:, 1]))
            else:
                train_score = float(roc_auc_score(self.y_train, train_pred_proba, multi_class='ovo', labels=self.labels))
                test_score = float(roc_auc_score(self.y_test, test_pred_proba, multi_class='ovo', labels=self.labels))
        else:
            # R^2 on the raw target -- the target is assumed already scaled/processed upstream.
            train_score = float(r2_score(self.y_train, model.predict(X_train_preprocessed)))
            test_score = float(r2_score(self.y_test, model.predict(X_test_preprocessed)))

        return train_score, test_score, calculate_model_complexity(model)

    def _build_classifier(self, model_type: str, model_params: dict):
        """Builds a fitted-ready sklearn classifier for the given model type and genotype.

        The seed is folded into eval_params by eval_parameters (for stochastic estimators),
        so it is never passed separately to the constructors below.
        """
        if model_type == 'RF':
            eval_params = RandomForestParams(self.data_ctx).eval_parameters(model_params, random_state=self.seed)
            return RandomForestClassifier(**eval_params, n_jobs=self.cores)
        elif model_type == 'ET':
            eval_params = ExtraTreesParams(self.data_ctx).eval_parameters(model_params, random_state=self.seed)
            return ExtraTreesClassifier(**eval_params, n_jobs=self.cores)
        elif model_type == 'KSVC':
            eval_params = KernelSVCParams(self.data_ctx).eval_parameters(model_params, random_state=self.seed)
            return SVC(**eval_params, probability=True)
        elif model_type == 'GB':
            eval_params = GradientBoostParams(self.data_ctx).eval_parameters(model_params, random_state=self.seed)
            return GradientBoostingClassifier(**eval_params)
        elif model_type == 'KNN':
            eval_params = KNeighborsClassifierParams(self.data_ctx).eval_parameters(model_params, random_state=self.seed)
            return KNeighborsClassifier(**eval_params, n_jobs=self.cores)
        elif model_type == 'MLP':
            eval_params = MLPClassifierParams(self.data_ctx).eval_parameters(model_params, random_state=self.seed)
            layers = (eval_params['layer_1'],
                      eval_params['layer_2'],
                      eval_params['layer_3'],
                      eval_params['layer_4'],
                      eval_params['layer_5'])
            return MLPClassifier(hidden_layer_sizes=layers,
                                 activation=eval_params['activation'],
                                 solver=eval_params['solver'],
                                 alpha=eval_params['alpha'],
                                 max_iter=eval_params['max_iter'],
                                 random_state=eval_params['random_state'])
        elif model_type == 'XGB':
            eval_params = XGBClassifierParams(self.data_ctx).eval_parameters(model_params, random_state=self.seed)
            return AnyLabelXGBClassifier(**eval_params, n_jobs=self.cores)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Must be one of ['RF', 'ET', 'KSVC', 'GB', 'KNN', 'MLP', 'XGB']")

    def _build_regressor(self, model_type: str, model_params: dict):
        """Builds a fitted-ready sklearn regressor for the given model type and genotype.

        The seed is folded into eval_params by eval_parameters (for stochastic estimators);
        deterministic regressors (SVR, KNN) take no random_state.
        """
        if model_type == 'RF':
            eval_params = RandomForestRegressorParams(self.data_ctx).eval_parameters(model_params, random_state=self.seed)
            return RandomForestRegressor(**eval_params, n_jobs=self.cores)
        elif model_type == 'ET':
            eval_params = ExtraTreesRegressorParams(self.data_ctx).eval_parameters(model_params, random_state=self.seed)
            return ExtraTreesRegressor(**eval_params, n_jobs=self.cores)
        elif model_type == 'SVR':
            eval_params = KernelSVRParams(self.data_ctx).eval_parameters(model_params, random_state=self.seed)
            return SVR(**eval_params)
        elif model_type == 'GB':
            eval_params = GradientBoostRegressorParams(self.data_ctx).eval_parameters(model_params, random_state=self.seed)
            return GradientBoostingRegressor(**eval_params)
        elif model_type == 'KNN':
            eval_params = KNeighborsRegressorParams(self.data_ctx).eval_parameters(model_params, random_state=self.seed)
            return KNeighborsRegressor(**eval_params, n_jobs=self.cores)
        elif model_type == 'MLP':
            eval_params = MLPRegressorParams(self.data_ctx).eval_parameters(model_params, random_state=self.seed)
            layers = (eval_params['layer_1'],
                      eval_params['layer_2'],
                      eval_params['layer_3'],
                      eval_params['layer_4'],
                      eval_params['layer_5'])
            return MLPRegressor(hidden_layer_sizes=layers,
                                activation=eval_params['activation'],
                                solver=eval_params['solver'],
                                alpha=eval_params['alpha'],
                                max_iter=eval_params['max_iter'],
                                random_state=eval_params['random_state'])
        elif model_type == 'XGB':
            eval_params = XGBRegressorParams(self.data_ctx).eval_parameters(model_params, random_state=self.seed)
            return XGBRegressor(**eval_params, n_jobs=self.cores)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Must be one of ['RF', 'ET', 'SVR', 'GB', 'KNN', 'MLP', 'XGB']")

    def save_results(self, save_dir: str) -> None:
        """
        Save final results using the best individual evaluated at the end of evolve().
        The JSON will contain train, validation, and test accuracy as well as the hyperparameter settings.
        """
        assert self.best_result is not None, "No best result found. Run evolve() first."

        print(f"Best individual params: {self.best_result['genotype']}", flush=True)
        print(f"Best validation performance: {self.best_perf}", flush=True)

        # Create output directory structure if it doesn't exist
        task_output_dir = os.path.join(save_dir)
        os.makedirs(task_output_dir, exist_ok=True)

        # Save best individual results as JSON
        best_results = {
            "model_type": self.param_space.get_model_type(),
            "seed": self.seed,
            "train_performance": self.best_result["train_performance"],
            "validation_performance": float(self.best_perf),
            "test_performance": self.best_result["test_performance"],
            "complexity": self.best_result["complexity"],
            "best_eval": self.best_result["best_eval"],
            "best_generation": self.best_result["best_generation"],
            "best_params": self.best_result["genotype"],
        }

        json_path = os.path.join(task_output_dir, "best_results.json")
        with open(json_path, 'w') as f:
            json.dump(best_results, f, indent=4)
        print(f"Best results saved to: {json_path}", flush=True)

        # save the full provenance archive of every evaluated individual
        self.save_archive(task_output_dir)

        # save total runtime in .txt file
        runtime_path = os.path.join(save_dir, "total_runtime_minutes.txt")
        with open(runtime_path, 'w') as f:
            f.write(f"{self.total_time_minutes}")
        print(f"Total evolution time (mins) saved to: {runtime_path}", flush=True)

        return

    def save_archive(self, save_dir: str) -> None:
        """
        Save the full provenance archive (every evaluated individual, with generation,
        construction, ei, error, genome, and performances) as ``archive.json``.

        Args:
            save_dir (str): Directory to write ``archive.json`` into.
        """
        assert self.eval_archive is not None, "No archive to save. Run evolve() first."

        os.makedirs(save_dir, exist_ok=True)
        archive_path = os.path.join(save_dir, "archive.json")
        with open(archive_path, 'w') as f:
            json.dump(self.eval_archive.to_records(), f, indent=4, default=str)
        print(f"Archive ({len(self.eval_archive)} entries) saved to: {archive_path}", flush=True)

        return