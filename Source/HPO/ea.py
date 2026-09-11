##########################################################################################
#
# HPO (Hyperparameter Optimization) EA for single-model optimization.
# Inherits from BaseEA and provides TPE-guided mutation for a single model type.
#
##########################################################################################

import numpy as np
import ray
import copy as cp
import os
import time
import json
import pandas as pd

from typeguard import typechecked
from typing import List, Optional

from Source.Base.base_ea import BaseEA
from Source.Base.individual import Individual
from Source.Base.tpe import TPE

from Source.Base.model_param_space import (
    RandomForestParams, ExtraTreesParams, KernelSVCParams, GradientBoostParams,
    KNeighborsClassifierParams, MLPClassifierParams
)
from Source.Base.ray_utils import (
    cv_random_forest, cv_extra_trees, cv_kernel_svc, cv_gradient_boost, cv_knn, cv_mlp
)


@typechecked
class EA(BaseEA):
    """
    HPO (Hyperparameter Optimization) EA for single-model optimization.
    Extends BaseEA with TPE-guided mutation for optimizing hyperparameters of a single model type.
    """

    def __init__(self,
                 seed: int,
                 pop_size: int,
                 cores: int,
                 mut_prob: float,
                 mut_var: float,
                 model: str,  # must be one of ['RF', 'ET', 'KSVC', 'GB', 'KNN', 'MLP']
                 tpe_prob: float,
                 tournament_size: int,
                 num_offspring: int,
                 gamma: float = 0.0,
                 tpe_mut_scale: float = 0.5,
                 explore_mut_scale: float = 2.0) -> None:
        """
        Initializes the HPO EA class with the provided parameters.

        Args:
            model (str): The type of model to optimize. Must be one of ['RF', 'ET', 'KSVC', 'GB', 'KNN', 'MLP'].
            tpe_prob (float): Probability of using TPE-based selection.
            seed (int): Random seed for reproducibility.
            gens (int): Number of generations to evolve.
            pop_size (int): Population size for the evolutionary algorithm.
            cores (int): Number of CPU cores to use for parallel processing.
            tournament_size (int): Tournament size for parent selection.
            mut_prob (float): Probability of mutating each hyperparameter.
            mut_var (float): Variance for Gaussian mutation.
            num_offspring (int): Number of pseudo-offspring to generate for TPE.
            output_dir (str): Directory for output files.
            gamma (float): Gamma parameter for TPE.
        """
        # initialize the base class
        super().__init__(
            seed=seed,
            pop_size=pop_size,
            cores=cores,
            mut_prob=mut_prob,
            mut_var=mut_var,
        )

        # store model type for deferred param_space creation (after load_data sets binary_classification)
        valid_models = ['RF', 'ET', 'KSVC', 'GB', 'KNN', 'MLP']
        if model not in valid_models:
            raise ValueError(f"Unknown model type: {model}. Must be one of {valid_models}")
        self.model = model
        self.param_space = None
        self.ray_train_func = None

        # HPO-specific EA parameters
        self.tournament_size = tournament_size
        self.num_offspring = num_offspring

        # TPE-related parameters
        self.tpe_prob = tpe_prob
        self.tpe = TPE(gamma=gamma)
        self.tpe_mut_var = self.mut_var * tpe_mut_scale
        self.explore_mut_var = self.mut_var * explore_mut_scale

        # archive tracking
        self.archive: List[Individual] = []
        self.tpe_archive: List[Individual] = []
        self.hard_eval_count = 0
        self.best_perf = float("-inf")
        self.best_ind: Optional[Individual] = None

        # per-generation checkpoints: test-set performance of the best-so-far individual,
        # recorded across generations for diagnostic tracking (not used by the final result)
        self.checkpoints: List[dict] = []

        return

    def load_data_pd(self,
                    data: pd.DataFrame,
                    target_label: str,
                    train_p: float,
                    one_hot_cols: Optional[List[str]] = None,
                    scalar_cols: Optional[List[str]] = None) -> None:
        """
        Loads data via parent class, then initializes param_space with correct binary_classification.
        """
        super().load_data_pd(data, target_label, train_p, one_hot_cols, scalar_cols)

        # now binary_classification is set, create the param_space
        model_configs = {
            'RF': (RandomForestParams(), cv_random_forest),
            'ET': (ExtraTreesParams(), cv_extra_trees),
            'KSVC': (KernelSVCParams(), cv_kernel_svc),
            'GB': (GradientBoostParams(binary_class=self.binary_classification), cv_gradient_boost),
            'KNN': (KNeighborsClassifierParams(), cv_knn),
            'MLP': (MLPClassifierParams(), cv_mlp),
        }
        self.param_space, self.ray_train_func = model_configs[self.model]

    def evolve(self, gens: int, ucb: bool = False, pi: bool = False, ei: bool = False,
               checkpoint_dir: Optional[str] = None) -> None:
        """
        Run the EA with parallelized fitness evaluations using Ray.

        Args:
            gens (int): Number of generations to evolve.
            ucb (bool): Not used in HPO EA (included for interface compatibility).
            pi (bool): Not used in HPO EA (included for interface compatibility).
            ei (bool): Not used in HPO EA (included for interface compatibility).
            checkpoint_dir (Optional[str]): If provided, after every generation (starting with
                the initial population, logged as Gen -1) the best-so-far individual is evaluated
                on the test set and a row is appended to ``{checkpoint_dir}/checkpoints.csv``.
                This is a diagnostic aid for watching how held-out performance evolves; it does
                not affect the final result.
        """

        start_time = time.time()

        # Initialize population with random individuals
        self.initialize_population()

        # Evaluate initial population with Ray
        self.population = self.evaluation(self.population)

        # keep track of hard evaluations for debugging
        self.hard_eval_count += len(self.population)

        # update archive
        self.update_archive(self.population)

        # keep track of best
        self.update_best_seen(self.population)
        print(f"Best performance so far (Gen 0): {self.best_perf}", flush=True)

        # checkpoint the best-so-far individual's test performance for the initial
        # population (Gen -1, since evolution proper begins at Gen 0)
        self.checkpoint_best_seen(generation=-1, checkpoint_dir=checkpoint_dir)

        # Start evolution
        for g in range(gens):
            # Parent selection with tournament selection
            parent_ids = self.parent_selection(self.population, self.pop_size, self.rng)

            # Generate offspring through mutation
            offspring = self.generate_offspring(self.population, parent_ids)

            # Evaluate offspring with Ray and update population
            self.population = self.evaluation(offspring)
            self.hard_eval_count += len(offspring)

            # update archive
            self.update_archive(offspring)

            # keep track of best
            self.update_best_seen(self.population)
            print(f"Best performance so far (Gen {g+1}): {self.best_perf}", flush=True)

            # checkpoint the best-so-far individual's test performance for this generation
            self.checkpoint_best_seen(generation=g, checkpoint_dir=checkpoint_dir)

        # make sure that the archive is the correct size
        print(f"Hard evaluations: {self.hard_eval_count}", flush=True)
        print(f"Total evolution time (mins): {(time.time() - start_time) / 60}", flush=True)

        # use the best-so-far individual (first config to reach the best validation AUC) for the
        # final test evaluation. Among validation ties this is a deterministic choice; it must not
        # be broken using test performance (that would leak the test set), and first-found does not.
        # This also keeps the final reported result identical to the last checkpoint.
        assert self.best_ind is not None, "No stored global best individual."
        # rebuild from params so the train/test performance setters start clean (best_ind already
        # carries the CV train/val performance from evaluation)
        best_individual = Individual(self.best_ind.get_params(), self.best_ind.model_type)
        best_individual.set_val_performance(self.best_perf)

        # evaluate best individual on test set
        train_score, test_score = self.model_test_evaluation(
            model_type=self.param_space.get_model_type().upper(),
            model_params=best_individual.get_params()
        )
        best_individual.set_train_performance(train_score)
        best_individual.set_test_performance(test_score)
        self.best_ind = best_individual

        print(f"Final test evaluation - Train: {train_score}, Test: {test_score}", flush=True)

        return

    def initialize_population(self) -> None:
        """
        Initializes the starting population for the HPO EA.
        Creates random individuals using the single model's parameter space.
        """
        assert len(self.population) == 0, "Population has already been initialized."

        self.population = [
            Individual(
                self.param_space.generate_random_parameters(self.rng),
                self.param_space.get_model_type()
            )
            for _ in range(self.pop_size)
        ]
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

        parent_ids = []
        for _ in range(num_parents):
            indices = rng.choice(len(population), self.tournament_size, replace=False)
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

        # launch one Ray task per (individual, fold) so every fold evaluates in parallel
        ray_jobs = []
        for model_id, ind in enumerate(candidates):
            eval_params = self.param_space.eval_parameters(ind.get_params())
            for X_train, y_train, X_validate, y_validate in cv_splits:
                ray_jobs.append(self.ray_train_func.remote(
                    X_train=X_train,
                    y_train=y_train,
                    X_validate=X_validate,
                    y_validate=y_validate,
                    model_params=eval_params,
                    random_state=self.seed,
                    id=model_id,
                    binary_class=self.binary_classification,
                    labels=self.labels
                ))

        # accumulate fold performances per pipeline (individual) as results arrive
        pop_results = [{'train_acc': [], 'val_acc': []} for _ in candidates]

        while len(ray_jobs) > 0:
            finished, ray_jobs = ray.wait(ray_jobs, num_returns=min(len(ray_jobs), self.cores))
            for done_id in finished:
                model_id, train_acc, val_acc, error = ray.get(done_id)
                assert error > 0.0, f"Error during model training/evaluation for model_id {model_id}."

                # track this fold's performance for the corresponding pipeline as it comes in
                pop_results[model_id]['train_acc'].append(train_acc)
                pop_results[model_id]['val_acc'].append(val_acc)

                # once all folds for this pipeline are in, finalize its mean CV performance
                if len(pop_results[model_id]['val_acc']) == num_folds:
                    mean_train = float(np.mean(pop_results[model_id]['train_acc']))
                    mean_val = float(np.mean(pop_results[model_id]['val_acc']))
                    candidates[model_id].set_train_performance(mean_train)
                    candidates[model_id].set_val_performance(mean_val)
                    print(f"Pipeline {model_id} evaluated - Train AUC: {mean_train:.4f}, Val AUC: {mean_val:.4f}", flush=True)

        return candidates

    def generate_offspring(self, candidates: List[Individual], parent_ids: List[int]) -> List[Individual]:
        """
        Generate offspring through mutation from selected parents.

        Args:
            candidates (List[Individual]): Candidate set of individuals.
            parent_ids (List[int]): List of indices of selected parents.
            mutation_rate (float): Probability of mutating each hyperparameter.
            mutation_var (float): Variance for Gaussian mutation.
            num_offspring (int): Number of pseudo-offspring to generate for TPE.

        Returns:
            List[Individual]: List of offspring individuals.
        """
        assert len(parent_ids) == len(candidates), "Number of parent IDs must match number of candidates."
        assert len(parent_ids) > 0, "At least one parent must be selected."
        assert len(self.tpe_archive) > 0, "TPE archive must have at least one individual for TPE-based mutation"
        assert len(self.tpe_archive) == len(self.archive), "TPE archive size must match main archive size."

        # store offspring here
        offspring = []

        # fit tpe model if self.tpe_prob > 0
        if self.tpe_prob > 0.0:
            self.tpe.fit(self.tpe_archive, self.param_space, self.rng)

        # go through each parent and generate offspring, roll for tpe or random mutation
        for pid in parent_ids:
            # tpe-based mutation (small variance for local exploitation)
            if self.rng.random() < self.tpe_prob:
                candidate_offspring = []
                for _ in range(self.num_offspring):
                    # mutate parent_params with smaller variance
                    candidate_offspring.append(self.param_space.mutate_parameters(
                        candidates[pid].get_params(),
                        self.tpe_mut_var,
                        self.mut_prob,
                        self.rng
                    ))
                # get best offspring according to tpe
                candidate_index = self.tpe.suggest_one(
                    self.param_space,
                    [self.param_space.tpe_parameters(params) for params in candidate_offspring],
                    self.rng
                )

                # append offspring recommended by tpe
                offspring.append(Individual(candidate_offspring[candidate_index], self.param_space.get_model_type()))

            # random mutation (large variance for global exploration)
            else:
                child_params = self.param_space.mutate_parameters(
                    candidates[pid].get_params(),
                    self.explore_mut_var,
                    self.mut_prob,
                    self.rng
                )
                offspring.append(Individual(child_params, self.param_space.get_model_type()))

        assert len(offspring) == len(parent_ids), "Number of offspring must match number of parents."
        return offspring

    def update_archive(self, evaluated_individuals: List[Individual]) -> None:
        """
        Update the archive with newly evaluated individuals.
        This archive is used to find the best performing individuals for final test set evaluation.
        Can also be used for TPE fitting.

        Args:
            evaluated_individuals (List[Individual]): List of newly evaluated individuals.
        """
        for ind in evaluated_individuals:
            arch_ind = Individual(ind.get_params(), ind.model_type)
            arch_ind.set_val_performance(ind.get_val_performance())
            self.archive.append(arch_ind)

            tpe_ind = Individual(self.param_space.tpe_parameters(ind.get_params()), ind.model_type)
            tpe_ind.set_val_performance(ind.get_val_performance() * -1.0)  # TPE minimizes, so invert performance
            self.tpe_archive.append(tpe_ind)

        return

    def update_best_seen(self, individuals: List[Individual]) -> None:
        """
        Update the best seen individual across all generations.

        Args:
            individuals (List[Individual]): List of individuals to check.
        """
        for ind in individuals:
            perf = ind.get_val_performance()
            if perf > self.best_perf:
                self.best_perf = perf
                self.best_ind = cp.deepcopy(ind)

    def checkpoint_best_seen(self, generation: int, checkpoint_dir: Optional[str]) -> None:
        """
        Record the test-set performance of the best-so-far individual for one generation.

        This is a diagnostic aid: it lets us watch how the held-out (test) performance of the
        best-validation individual evolves generation by generation. The tracked individual is
        ``self.best_ind`` (the first individual to reach the current best validation AUC), which
        is the same individual ``evolve`` reports at the end, so the final generation's checkpoint
        matches the final saved result.

        To avoid redundant full-model refits, the best individual is only re-evaluated on the
        test set when its validation performance improved since the previous checkpoint;
        otherwise the previous generation's train/test scores are carried forward (correct,
        since the best-so-far only changes when validation performance strictly improves).

        Args:
            generation (int): Generation index (-1 for the initial population; evolution
                proper starts at 0).
            checkpoint_dir (Optional[str]): Directory to write ``checkpoints.csv`` into.
                If None, checkpointing is disabled and this is a no-op.
        """
        if checkpoint_dir is None:
            return

        assert self.best_ind is not None, "No best individual to checkpoint."

        # only refit/evaluate on the test set when the best-so-far actually changed
        if self.checkpoints and self.checkpoints[-1]["val_auc"] == self.best_perf:
            train_score = self.checkpoints[-1]["train_auc"]
            test_score = self.checkpoints[-1]["test_auc"]
        else:
            train_score, test_score = self.model_test_evaluation(
                model_type=self.param_space.get_model_type().upper(),
                model_params=self.best_ind.get_params()
            )

        self.checkpoints.append({
            "generation": generation,
            "hard_evals": self.hard_eval_count,
            "val_auc": float(self.best_perf),
            "train_auc": float(train_score),
            "test_auc": float(test_score),
        })

        os.makedirs(checkpoint_dir, exist_ok=True)
        csv_path = os.path.join(checkpoint_dir, "checkpoints.csv")
        pd.DataFrame(self.checkpoints).to_csv(csv_path, index=False)

        # snapshot of the best-so-far result at this many total hard evaluations, mirroring
        # the best_results.json schema so progress can be tracked generation by generation
        result_snapshot = {
            "generation": generation,
            "hard_evals": self.hard_eval_count,
            "task_id": self.task_id,
            "model_type": self.param_space.get_model_type(),
            "seed": self.seed,
            "train_accuracy": float(train_score),
            "validation_accuracy": float(self.best_perf),
            "test_accuracy": float(test_score),
            "best_params": self.best_ind.get_params(),
        }
        json_path = os.path.join(checkpoint_dir, f"results_eval_{self.hard_eval_count}.json")
        with open(json_path, 'w') as f:
            json.dump(result_snapshot, f, indent=4)

        print(f"Checkpoint (Gen {generation}) - Val AUC: {self.best_perf:.4f}, "
              f"Train AUC: {train_score:.4f}, Test AUC: {test_score:.4f} "
              f"-> {json_path}", flush=True)

        return

    def save_results(self, save_dir: str) -> None:
        """
        Save final results using the best individual evaluated at the end of evolve().
        The JSON will contain train, validation, and test accuracy as well as the hyperparameter settings.
        """
        assert self.best_ind is not None, "No best individual found. Run evolve() first."

        print(f"Best individual params: {self.best_ind.get_params()}", flush=True)
        print(f"Best validation performance: {self.best_perf}", flush=True)

        # Create output directory structure if it doesn't exist
        task_output_dir = os.path.join(save_dir)
        os.makedirs(task_output_dir, exist_ok=True)

        # Save best individual results as JSON
        best_results = {
            "task_id": self.task_id,
            "model_type": self.param_space.get_model_type(),
            "seed": self.seed,
            "train_accuracy": self.best_ind.get_train_performance(),
            "validation_accuracy": float(self.best_perf),
            "test_accuracy": self.best_ind.get_test_performance(),
            "best_params": self.best_ind.get_params(),
        }

        json_path = os.path.join(task_output_dir, "best_results.json")
        with open(json_path, 'w') as f:
            json.dump(best_results, f, indent=4)
        print(f"Best results saved to: {json_path}", flush=True)

        return