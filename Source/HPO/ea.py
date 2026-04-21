import numpy as np
import ray
from Source.Base.individual import Individual
from typeguard import typechecked
from typing import List, Dict, Optional
import copy as cp
from Source.Base.tpe import TPE
from Source.Base.data_utils import load_data, get_ray_cv_splits, preprocess_train_test
from Source.Base.ea_utils import parent_selection
import os
import time
import pandas as pd
import json

@typechecked
class EA:
    """
    EA Solver for hyperparameter optimization only.
    """
    def __init__(self,
                 model_config: Dict,
                 tpe_prob: float,
                 seed: int,
                 gens: int,
                 pop_size: int,
                 tournament_size: int,
                 mutation_rate: float,
                 mutation_var: float,
                 num_offspring: int,
                 task_id: int,
                 rep: int,
                 data_dir: str,
                 split_dir: str,
                 output_dir: str,
                 window: int,
                 gamma: float = 0.0,) -> None:
        """
        Parameters:
            param_space (ModelParams): Model parameter space object that
        """

        # extract model parameter space and functions from model_config

        self.param_space = model_config['param_class'] # model parameter space
        self.test_eval_func = model_config['test_eval_func'] # model test evaluation function
        self.ray_train_func = model_config['ray_train_func'] # ray remote training function (deprecated)
        self.ray_train_func_cv = model_config['ray_train_func_cv'] # ray remote training function for consolidated CV

        # ea parameters

        self.gens = gens # number of generations
        self.pop_size = pop_size # population size
        self.tournament_size = tournament_size # tournament size for parent selection
        self.mutation_rate = mutation_rate # mutation rate
        self.mutation_var = mutation_var # mutation variance
        self.num_offspring = num_offspring # number of offspring per parent for tpe
        self.seed = seed # random seed
        self.rng = np.random.default_rng(seed) # random number generator
        self.population: List[Individual] = [] # current population

        # history data for analysis

        self.archive: List[Individual] = [] # archive of evaluated individuals
        self.tpe_archive: List[Individual] = [] # archive for tpe individuals
        self.hard_eval_count = 0 # evaluations on the true objective
        self.tpe = TPE(gamma=gamma) # tpe object for tpe-based mutation, can be None
        self.best_perf = float("-inf") # best performance seen so far
        self.best_ind: Optional[Individual] = None # GLOBAL best, regardless of sliding window size
        self.tpe_prob = tpe_prob # probability of using tpe-based selection
        self.window = window # sliding window size for archive

        # openml dataset loading

        self.task_id = task_id # openml task id
        self.rep = rep # replicate number
        self.data_directory = data_dir # directory where datasets are stored
        self.split_directory = split_dir # directory where splits are stored
        self.output_directory = output_dir # directory for output files that are generated

        return

    def load_openml_dataset(self) -> None:
        """
        Load dataset from OpenML given task ID and replicate number.
        Will use the load_data, get_ray_cv_splits, preprocess_train_test functions.
        Located in ..Source/Base/data_utils.py

        Parameters:
            task_id (int): OpenML task ID.
            rep (int): Replicate number.
            data_directory (str): Directory where datasets are stored.
            split_directory (str): Directory where splits are stored.
        """

        # indicies for train/test split for a given task and replicate
        rep_dir = os.path.join(self.split_directory, f"task_{self.task_id}", f"Replicate_{self.rep}")
        # load all train/test data
        self.X_train, self.X_test, self.y_train, self.y_test = load_data(self.task_id, self.data_directory, rep_dir)
        # get all 5-fold cv splits as list of (train_idx, val_idx)
        print(f"Preparing cross-validation splits...", flush=True)
        self.X_train_f0, self.X_val_f0, self.y_train_f0, self.y_val_f0, \
        self.X_train_f1, self.X_val_f1, self.y_train_f1, self.y_val_f1, \
        self.X_train_f2, self.X_val_f2, self.y_train_f2, self.y_val_f2, \
        self.X_train_f3, self.X_val_f3, self.y_train_f3, self.y_val_f3, \
        self.X_train_f4, self.X_val_f4, self.y_train_f4, self.y_val_f4 = get_ray_cv_splits(rep_dir=rep_dir,
                                                                                           X_train=self.X_train,
                                                                                           y_train=self.y_train,
                                                                                           task_id=self.task_id,
                                                                                           data_dir=self.data_directory)

        # Put CV data into Ray's object store (optimization to avoid repeated serialization)
        print(f"Storing CV data in Ray object store...", flush=True)
        cv_data_tuple = (
            self.X_train_f0, self.y_train_f0, self.X_val_f0, self.y_val_f0,
            self.X_train_f1, self.y_train_f1, self.X_val_f1, self.y_val_f1,
            self.X_train_f2, self.y_train_f2, self.X_val_f2, self.y_val_f2,
            self.X_train_f3, self.y_train_f3, self.X_val_f3, self.y_val_f3,
            self.X_train_f4, self.y_train_f4, self.X_val_f4, self.y_val_f4
        )
        self.cv_data_ref = ray.put(cv_data_tuple)
        return

    def evolve(self) -> None:
        """
        Run the default EA with parallelized fitness evaluations using Ray.

        Parameters:
            gens (int): Number of generations to evolve.
            pop_size (int): Population size.
        """

        start_time = time.time()

        # Initialize population with random individuals
        self.population = [Individual(self.param_space.generate_random_parameters(self.rng),self.param_space.get_model_type()) \
            for _ in range(self.pop_size)]

        # Evaluate initial population with Ray
        self.population = self.evaluation(self.population)

        # keep track of hard evaluations for debugging
        self.hard_eval_count += len(self.population)

        # update archive
        self.update_archive(self.population, self.window)

        # keep track of best
        self.update_best_seen(self.population)
        print(f"Initial population size: {len(self.population)}", flush=True)
        print(f"Best performance so far (Gen 0): {self.best_perf}", flush=True)

        # Start evolution
        for g in range(self.gens):
            # Parent selection with tournament selection
            parent_ids = parent_selection(self.population, self.pop_size, self.rng, self.tournament_size)

            # Generate offspring through mutation
            offspring = self.generate_offspring(self.population,
                                                parent_ids,
                                                self.mutation_rate,
                                                self.mutation_var,
                                                self.num_offspring)

            # Evaluate offspring with Ray and update population
            self.population = self.evaluation(offspring)
            self.hard_eval_count += len(offspring)

            # update archive
            self.update_archive(offspring, self.window)

            # Get best performance in current population
            # current_best = max([ind.get_val_performance() for ind in self.population])
            # if current_best > self.best_perf:
            #     self.best_perf = current_best

            # keep track of best
            self.update_best_seen(self.population)
            print(f"Best performance so far (Gen {g+1}): {self.best_perf}", flush=True)

        # make sure that the archive is the correct size
        assert len(self.archive) <= self.window
        # assert len(self.archive) == self.gens * self.pop_size + self.pop_size
        print(f"Hard evaluations: {self.hard_eval_count}", flush=True)
        print(f"Total evolution time (mins): {(time.time() - start_time) / 60}", flush=True)
        return

    def save_results(self) -> None:
        """
        Save final results by evaluating the best individual on the test set in a JSON format.
        The JSON will contain train and test accuracy as well as the hyperparameter settings.

        Save the entire archive to output directory as well as a csv file with headers reflecting the model parameters type.
        """

        # Iterate through archive to get all set of best performers
        # best_performers = [pos for pos, ind in enumerate(self.archive) if ind.get_val_performance() == self.best_perf]

        # Get rid of exact float equality
        tol = 1e-12
        best_performers = [
            pos for pos, ind in enumerate(self.archive)
            if abs(ind.get_val_performance() - self.best_perf) <= tol
        ]
        print(f"Number of best performers in archive: {len(best_performers)}", flush=True)

        if len(best_performers) > 0:
            # randomly select one of the best performers for final test evaluation
            best_individual = cp.deepcopy(self.archive[self.rng.choice(best_performers)])
        else:
            assert self.best_ind is not None, "No stored global best individual."
            best_individual = cp.deepcopy(self.best_ind)

        # fit best individual on full training data and evaluate on test set
        X_train_transformed, y_train, X_test_transformed, y_test = preprocess_train_test(self.X_train,
                                                                                         self.y_train,
                                                                                         self.X_test,
                                                                                         self.y_test,
                                                                                         self.task_id,
                                                                                         self.data_directory)
        # train final model with evaluation map
        train, test, error = self.test_eval_func(X_train_transformed, y_train, X_test_transformed,
                            y_test, best_individual.get_params(),self.seed)
        assert error > 0.0, "Error during final model train/test evaluation."

        print(f"Final evaluation on test set:", flush=True)
        print(f"Train Accuracy: {train}", flush=True)
        print(f"Test Accuracy: {test}", flush=True)

        # Create output directory structure if it doesn't exist
        task_output_dir = os.path.join(self.output_directory)
        os.makedirs(task_output_dir, exist_ok=True)

        # Save best individual results as JSON
        best_results = {
            "task_id": self.task_id,
            "replicate": self.rep,
            "model_type": self.param_space.get_model_type(),
            "seed": self.seed,
            "train_accuracy": float(train),
            "test_accuracy": float(test),
            "validation_accuracy": float(self.best_perf),
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

    def evaluation(self, candidates: List[Individual]) -> List[Individual]:
        """
        Evaluate a collection of individuals using Ray across 5-fold cross-validation.
        This method will update each individual's train and validation performance.
        Can be used for offspring evaluation during evolution or initial population evaluation.

        Optimized version: Uses consolidated CV functions that process all 5 folds per
        individual in a single Ray task, and leverages Ray's object store for CV data.

        Parameters:
            candidates (List[Individual]): List of individuals to evaluate.

        Returns:
            List[Individual]: The evaluated individuals with updated performance metrics.
        """

        ray_jobs = []
        # Create one Ray task per individual (instead of 5 per individual)
        for i, ind in enumerate(candidates):
            ray_jobs.append(self.ray_train_func_cv.remote(self.cv_data_ref,
                                                          ind.get_params(),
                                                          self.seed,
                                                          i))

        # Gather results as they complete
        pop_results = {}
        while len(ray_jobs) > 0:
            finished, ray_jobs = ray.wait(ray_jobs)
            id, train_accs, val_accs, err = ray.get(finished[0])
            assert err > 0.0, "Error during model training/evaluation."
            assert len(train_accs) == 5, f"Expected 5 train accuracies, got {len(train_accs)}"
            assert len(val_accs) == 5, f"Expected 5 val accuracies, got {len(val_accs)}"
            pop_results[id] = {'train_acc': train_accs, 'val_acc': val_accs}

        # Assign performances to individuals
        for i, ind in enumerate(candidates):
            ind.set_train_performance(np.mean(pop_results[i]['train_acc']))
            ind.set_val_performance(np.mean(pop_results[i]['val_acc']))

        return candidates

    def generate_offspring(self,
                           candidates: List[Individual],
                           parent_ids: List[int],
                           mutation_rate: float,
                           mutation_var: float,
                           num_offspring: int) -> List[Individual]:
        """
        Generate offspring through mutation from selected parents.

        Parameters:
            candidates (List[Individual]): Candidate set of individuals.
            parent_ids (List[int]): List of indices of selected parents.
            mutation_rate (float): Probability of mutating each hyperparameter.
            mutation_var (float): Variance for Gaussian mutation.
            num_offspring (int): Number of pseudo-offspring to generate for tpe.

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
                for _ in range(num_offspring):
                    # mutate parent_params
                    candidate_offspring.append(self.param_space.mutate_parameters(candidates[pid].get_params(),
                                                                        mutation_var,
                                                                        mutation_rate,
                                                                        self.rng))
                # get best offspring according to tpe
                candidate_index = self.tpe.suggest_one(self.param_space,
                                                    [self.param_space.tpe_parameters(params) for params in candidate_offspring],
                                                    self.rng)

                # append offspring recommended by tpe
                offspring.append(Individual(candidate_offspring[candidate_index], self.param_space.get_model_type()))

            # random mutation
            else:
                child_params = self.param_space.mutate_parameters(candidates[pid].get_params(),
                                                                  mutation_var,
                                                                  mutation_rate,
                                                                  self.rng)
                offspring.append(Individual(child_params, self.param_space.get_model_type()))

        assert len(offspring) == len(parent_ids), "Number of offspring must match number of parents."
        return offspring

    def update_archive(self, evaluated_individuals: List[Individual], window: int) -> None:
        """
        Update the archive with newly evaluated individuals.
        This archive is used to find the best performing individuals for final test set evaluation.
        Can also be used for TPE fitting.

        Parameters:
            evaluated_individuals (List[Individual]): List of newly evaluated individuals.
        """

        for ind in evaluated_individuals:
            arch_ind = Individual(ind.get_params(), ind.model_type)
            arch_ind.set_val_performance(ind.get_val_performance())  # TPE minimizes, so invert performance
            self.archive.append(arch_ind)

            tpe_ind = Individual(self.param_space.tpe_parameters(ind.get_params()), ind.model_type)
            tpe_ind.set_val_performance(ind.get_val_performance() * -1.0)  # TPE minimizes, so invert performance
            self.tpe_archive.append(tpe_ind)

        # sliding window: keep only the most recent `window` individuals
        if len(self.archive) > window:
            self.archive = self.archive[-window:]

        if len(self.tpe_archive) > window:
            self.tpe_archive = self.tpe_archive[-window:]
        return
    
    def update_best_seen(self, individuals: List[Individual]) -> None:
        for ind in individuals:
            perf = ind.get_val_performance()
            if perf > self.best_perf:
                self.best_perf = perf
                self.best_ind = cp.deepcopy(ind)