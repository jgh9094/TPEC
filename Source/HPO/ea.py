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
    RandomForestParams, KernelSVCParams, GradientBoostParams,
    KNeighborsClassifierParams, MLPClassifierParams
)
from Source.Base.ray_utils import (
    cv_random_forest, cv_kernel_svc, cv_gradient_boost, cv_knn, cv_mlp
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
                 model: str,  # must be one of ['RF', 'KSVC', 'GB', 'KNN', 'MLP']
                 tpe_prob: float,
                 tournament_size: int,
                 num_offspring: int,
                 gamma: float = 0.0) -> None:
        """
        Initializes the HPO EA class with the provided parameters.

        Args:
            model (str): The type of model to optimize. Must be one of ['RF', 'KSVC', 'GB', 'KNN', 'MLP'].
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

        # map model string to parameter space class and ray training function
        model_configs = {
            'RF': (RandomForestParams(), cv_random_forest),
            'KSVC': (KernelSVCParams(), cv_kernel_svc),
            'GB': (GradientBoostParams(binary_class=True), cv_gradient_boost),
            'KNN': (KNeighborsClassifierParams(), cv_knn),
            'MLP': (MLPClassifierParams(), cv_mlp),
        }
        if model not in model_configs:
            raise ValueError(f"Unknown model type: {model}. Must be one of {list(model_configs.keys())}")
        self.param_space, self.ray_train_func = model_configs[model]

        # HPO-specific EA parameters
        self.tournament_size = tournament_size
        self.num_offspring = num_offspring

        # TPE-related parameters
        self.tpe_prob = tpe_prob
        self.tpe = TPE(gamma=gamma)

        # archive tracking
        self.archive: List[Individual] = []
        self.tpe_archive: List[Individual] = []
        self.hard_eval_count = 0
        self.best_perf = float("-inf")
        self.best_ind: Optional[Individual] = None

        return

    def evolve(self, gens: int, ucb: bool = False, pi: bool = False, ei: bool = False) -> None:
        """
        Run the EA with parallelized fitness evaluations using Ray.

        Args:
            gens (int): Number of generations to evolve.
            ucb (bool): Not used in HPO EA (included for interface compatibility).
            pi (bool): Not used in HPO EA (included for interface compatibility).
            ei (bool): Not used in HPO EA (included for interface compatibility).
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
        print(f"Initial population size: {len(self.population)}", flush=True)
        print(f"Best performance so far (Gen 0): {self.best_perf}", flush=True)

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

        # make sure that the archive is the correct size
        print(f"Hard evaluations: {self.hard_eval_count}", flush=True)
        print(f"Total evolution time (mins): {(time.time() - start_time) / 60}", flush=True)

        # find all best performers in the archive, and randomly select one for final test evaluation
        tol = 1e-12
        best_performers = [
            pos for pos, ind in enumerate(self.archive)
            if abs(ind.get_val_performance() - self.best_perf) <= tol
        ]

        if len(best_performers) > 0:
            best_individual = cp.deepcopy(self.archive[self.rng.choice(best_performers)])
        else:
            assert self.best_ind is not None, "No stored global best individual."
            best_individual = cp.deepcopy(self.best_ind)

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
        Uses the same CV split data structure as CASH EA via get_cv_splits() from BaseEA.

        Args:
            candidates (List[Individual]): List of individuals to evaluate.

        Returns:
            List[Individual]: The evaluated individuals with updated performance metrics.
        """
        # get CV splits from base class
        cv_splits = self.get_cv_splits()

        ray_jobs = []
        # Create Ray tasks for each individual across all 5 folds
        for model_id, ind in enumerate(candidates):
            for X_train, y_train, X_validate, y_validate in cv_splits:
                ray_jobs.append(self.ray_train_func.remote(
                    X_train=X_train,
                    y_train=y_train,
                    X_validate=X_validate,
                    y_validate=y_validate,
                    model_params=self.param_space.eval_parameters(ind.get_params()),
                    random_state=self.seed,
                    id=model_id,
                    binary_class=self.binary_classification,
                    labels=self.labels
                ))

        # Initialize results storage
        pop_results = {i: {'train_acc': [], 'val_acc': []} for i in range(len(candidates))}

        # Gather results as they complete
        while len(ray_jobs) > 0:
            finished, ray_jobs = ray.wait(ray_jobs, num_returns=min(len(ray_jobs), self.cores))
            for done_id in finished:
                model_id, train_acc, val_acc, error = ray.get(done_id)
                assert error > 0.0, f"Error during model training/evaluation for model_id {model_id}."
                pop_results[model_id]['train_acc'].append(train_acc)
                pop_results[model_id]['val_acc'].append(val_acc)

        # Assign performances to individuals
        for i, ind in enumerate(candidates):
            ind.set_train_performance(float(np.mean(pop_results[i]['train_acc'])))
            ind.set_val_performance(float(np.mean(pop_results[i]['val_acc'])))

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
            # tpe-based mutation
            if self.rng.random() < self.tpe_prob:
                candidate_offspring = []
                for _ in range(self.num_offspring):
                    # mutate parent_params
                    candidate_offspring.append(self.param_space.mutate_parameters(
                        candidates[pid].get_params(),
                        self.mut_var,
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

            # random mutation
            else:
                child_params = self.param_space.mutate_parameters(
                    candidates[pid].get_params(),
                    self.mut_var,
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

    def save_results(self, save_dir: str) -> None:
        """
        Save final results using the best individual evaluated at the end of evolve().
        The JSON will contain train, validation, and test accuracy as well as the hyperparameter settings.
        Save the entire archive to output directory as well as a csv file with headers reflecting the model parameters type.
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

        # Save entire archive as CSV with only hyperparameters
        archive_data = []
        for ind in self.tpe_archive:
            # Only store hyperparameters, convert None to 'None' string
            params = ind.get_params()
            params_cleaned = {k: ('None' if v is None else v) for k, v in params.items()}
            archive_data.append(params_cleaned)

        archive_df = pd.DataFrame(archive_data)
        csv_path = os.path.join(task_output_dir, "archive.csv")
        archive_df.to_csv(csv_path, index=False)
        print(f"Archive saved to: {csv_path}", flush=True)

        return