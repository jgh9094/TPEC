##########################################################################################
#
# Class dedicated to the evolutionary algorithm (EA) optimization of hyperparameters for machine learning models.
# This CASH (Combined Algorithm Selection and Hyperparameter optimization) EA inherits from BaseEA.
#
##########################################################################################

import numpy as np
import ray
import numpy.typing as npt
import copy as cp

from typeguard import typechecked
from typing import List, Dict, Tuple, Optional

from Source.Base import model_param_space
from Source.Base.base_ea import BaseEA
from Source.Base.ray_utils import cv_random_forest, cv_kernel_svc, cv_gradient_boost, cv_knn, cv_mlp
from Source.Base.model_param_space import RandomForestParams, KernelSVCParams, GradientBoostParams, KNeighborsClassifierParams, MLPClassifierParams
from Source.Base.individual import Individual
import Source.CASH.nsag_toolset as nsga

from .bo_ray_utils import bo_rf_optimizer, bo_ksvc_optimizer, bo_gb_optimizer, bo_knn_optimizer, bo_mlp_optimizer


@typechecked
class EA(BaseEA):
    """
    CASH (Combined Algorithm Selection and Hyperparameter optimization) EA.
    Extends BaseEA with multi-model support and Bayesian optimization-guided acquisition functions.
    """

    def __init__(self,
                 seed: int,
                 pop_size: int,
                 cores: int,
                 mut_prob: float,
                 mut_var: float,
                 initial_history_size: int) -> None:
        """
        Initializes the CASH EA class with the provided parameters.

        Args:
            seed (int): Random seed for reproducibility.
            pop_size (int): Population size for the evolutionary algorithm.
            cores (int): Number of CPU cores to use for parallel processing.
            mut_prob (float): Mutation probability for the evolutionary algorithm.
            mut_var (float): Mutation variance for the evolutionary algorithm.
            initial_history_size (int): Size of the initial history per model for the Bayesian optimizer.
        """
        # initialize the base class
        super().__init__(
            seed=seed,
            pop_size=pop_size,
            cores=cores,
            mut_prob=mut_prob,
            mut_var=mut_var
        )

        # CASH-specific validation
        assert initial_history_size > 0, "Initial history size must be a positive integer."
        self.initial_history_size = initial_history_size

        # CASH-specific history tracking for Bayesian optimization (per model type)
        self.rf_history = {'n_estimators': [], 'criterion': [], 'max_depth': [], 'max_features': [], 'max_samples': [], 'class_weight': [], self.SOLUTION_VALIDATION_SCORE: [], self.SOLUTION_TRAIN_SCORE: []}
        self.ksvc_history = {'C': [], 'kernel': [], 'max_iter': [], 'class_weight': [], 'decision_function_shape': [], self.SOLUTION_VALIDATION_SCORE: [], self.SOLUTION_TRAIN_SCORE: []}
        self.gb_history = {'loss': [], 'learning_rate': [], 'n_estimators': [], 'subsample': [], 'criterion': [], 'max_depth': [], 'max_features': [], self.SOLUTION_VALIDATION_SCORE: [], self.SOLUTION_TRAIN_SCORE: []}
        self.knn_history = {'n_neighbors': [], 'weights': [], 'algorithm': [], 'leaf_size': [], 'p': [], self.SOLUTION_VALIDATION_SCORE: [], self.SOLUTION_TRAIN_SCORE: []}
        self.mlp_history = {'layer_1': [], 'layer_2': [], 'layer_3': [], 'layer_4': [], 'layer_5': [], 'activation': [], 'solver': [], 'max_iter': [], self.SOLUTION_VALIDATION_SCORE: [], self.SOLUTION_TRAIN_SCORE: []}

        # best individual tracking
        self.best_perf = float("-inf")
        self.best_ind: Optional[Individual] = None

        return

    def evolve(self, gens: int, ucb: bool, pi: bool, ei: bool) -> None:
        """
        Evolves the hyperparameters for all model types over a given number of generations.
        This function optimizes all 3 acquisition functions (UCB, EI, PI) simultaneously using NSGA-II.

        Args:
            gens (int): Number of generations to evolve.
            ucb (bool): Whether to use Upper Confidence Bound for selection.
            pi (bool): Whether to use Probability of Improvement for selection.
            ei (bool): Whether to use Expected Improvement for selection.
        """
        if ucb and pi and ei:
            print("Evolving with all acquisition functions (UCB, PI, EI) using NSGA-II.")
            self.evolve_3d(gens=gens)
        return

    def evolve_3d(self, gens: int) -> None:
        """
        Evolves the hyperparameters for all model types over a given number of generations.
        This function optimizes all 3 acquisition functions (UCB, EI, PI) simultaneously using NSGA-II.

        Args:
            gens (int): Number of generations to evolve.
        """
        # quick sanity checks
        assert gens > 0, "Number of generations must be a positive integer."

        # step 1: generate a random set of hyperparameters and evaluate them to populate the initial history for the Bayesian optimizer
        self.initialize_bo_history()
        print(f"Initial history for Bayesian optimizer populated with {self.initial_history_size} random hyperparameter evaluations per model type.")

        # step 2: initialize the starting population for the ea
        self.initialize_population()
        print(f"Initial population for evolutionary algorithm generated with {self.pop_size} random hyperparameter configurations.")

        # step 3: evolve the population over the specified number of generations
        for g in range(gens):
            print(f"Starting generation {g + 1}/{gens} of the evolutionary algorithm.")
            print(f"  rf best validation: {max(self.rf_history[self.SOLUTION_VALIDATION_SCORE]) if self.rf_history[self.SOLUTION_VALIDATION_SCORE] else 'N/A'} size {len(self.rf_history[self.SOLUTION_VALIDATION_SCORE])}")
            print(f"  gb best validation: {max(self.gb_history[self.SOLUTION_VALIDATION_SCORE]) if self.gb_history[self.SOLUTION_VALIDATION_SCORE] else 'N/A'} size {len(self.gb_history[self.SOLUTION_VALIDATION_SCORE])}")
            print(f" knn best validation: {max(self.knn_history[self.SOLUTION_VALIDATION_SCORE]) if self.knn_history[self.SOLUTION_VALIDATION_SCORE] else 'N/A'} size {len(self.knn_history[self.SOLUTION_VALIDATION_SCORE])}")
            print(f"ksvc best validation: {max(self.ksvc_history[self.SOLUTION_VALIDATION_SCORE]) if self.ksvc_history[self.SOLUTION_VALIDATION_SCORE] else 'N/A'} size {len(self.ksvc_history[self.SOLUTION_VALIDATION_SCORE])}")
            print(f" mlp best validation: {max(self.mlp_history[self.SOLUTION_VALIDATION_SCORE]) if self.mlp_history[self.SOLUTION_VALIDATION_SCORE] else 'N/A'} size {len(self.mlp_history[self.SOLUTION_VALIDATION_SCORE])}")
            print("*"*50)

            # get fronts and ranks based on the current population's acquisition scores
            fronts, ranks = nsga.non_dominated_sorting(self.generate_acquisition_scores_for_nsga(population=self.population))

            # compute crowding distances for each front
            crowding_distances = nsga.crowding_distance(self.generate_acquisition_scores_for_nsga(population=self.population), fronts, count=3)

            # select parents based on ranks and crowding distances
            parent_ids = [nsga.non_dominated_binary_tournament(ranks, crowding_distances, self.rng) for _ in range(self.pop_size * 2)]  # select twice the population size for mating pool

            # generate offspring through mutation of the selected parents
            offspring = self.generate_offspring([self.population[i] for i in parent_ids])

            # compute acquisition scores for the offspring
            self.compute_acquisition_scores(offspring)

            # compute acquisition scores for the offspring
            offspring_acquisition_scores = self.generate_acquisition_scores_for_nsga(population=offspring)

            # truncate the offspring to the population size based on non-dominated sorting and crowding distance
            off_fronts, off_ranks = nsga.non_dominated_sorting(offspring_acquisition_scores)
            off_crowding_distances = nsga.crowding_distance(offspring_acquisition_scores, off_fronts, count=3)

            # get survivor ids based on ranks and crowding distances
            survivor_ids = nsga.non_dominated_truncate(off_fronts, off_crowding_distances, self.pop_size)

            # update the population with the selected survivors
            self.population = [offspring[i] for i in survivor_ids]

            # compute performance metrics for the current generation
            rf_models, ksvc_models, gb_models, knn_models, mlp_models = [ind.get_params() for ind in self.population if ind.model_type == 'rf'], \
                [ind.get_params() for ind in self.population if ind.model_type == 'ksvc'], \
                [ind.get_params() for ind in self.population if ind.model_type == 'gb'], \
                [ind.get_params() for ind in self.population if ind.model_type == 'knn'], \
                [ind.get_params() for ind in self.population if ind.model_type == 'mlp']

            rf_models, ksvc_models, gb_models, knn_models, mlp_models = self.evaluation(rf_models, ksvc_models, gb_models, knn_models, mlp_models)

            # update the history with the evaluated offspring
            self.update_history(rf_models, ksvc_models, gb_models, knn_models, mlp_models)

            # make sure population size is maintained
            assert len(self.population) == self.pop_size, f"Population size mismatch after generation {g + 1}: expected {self.pop_size}, got {len(self.population)}."

        self.final_model_evaluation()

        return

    def initialize_bo_history(self) -> None:
        """
        Initializes the history of the Bayesian optimizer with the current hyperparameter evaluations.
        For each model type, it generates a set of random hyperparameter configurations and evaluates them to populate the history.
        The total number of random configurations generated is determined by the `self.initial_history_size` parameter.

        Returns:
            None
        """
        # generate random hyperparameters for random forest
        rf_initial_history = []
        for _ in range(self.initial_history_size):
            rf_params = model_param_space.RandomForestParams().generate_random_parameters(self.rng)
            rf_initial_history.append(rf_params)

        # generate random hyperparameters for kernel SVC
        ksvc_initial_history = []
        for _ in range(self.initial_history_size):
            ksvc_params = model_param_space.KernelSVCParams().generate_random_parameters(self.rng)
            ksvc_initial_history.append(ksvc_params)

        # generate random hyperparameters for gradient boosting
        gb_initial_history = []
        for _ in range(self.initial_history_size):
            gb_params = model_param_space.GradientBoostParams(binary_class=self.binary_classification).generate_random_parameters(self.rng)
            gb_initial_history.append(gb_params)

        # generate random hyperparameters for k-nearest neighbors
        knn_initial_history = []
        for _ in range(self.initial_history_size):
            knn_params = model_param_space.KNeighborsClassifierParams().generate_random_parameters(self.rng)
            knn_initial_history.append(knn_params)

        # generate random hyperparameters for multi-layer perceptron
        mlp_initial_history = []
        for _ in range(self.initial_history_size):
            mlp_params = model_param_space.MLPClassifierParams().generate_random_parameters(self.rng)
            mlp_initial_history.append(mlp_params)

        # evaluate the initial random hyperparameter configurations and populate the history for each model type
        rf_initial_history, ksvc_initial_history, gb_initial_history, knn_initial_history, mlp_initial_history = self.evaluation(rf_initial_history,
                                                                                                                                 ksvc_initial_history,
                                                                                                                                 gb_initial_history,
                                                                                                                                 knn_initial_history,
                                                                                                                                 mlp_initial_history)

        # update the history with the evaluated hyperparameter configurations
        self.update_history(rf_initial_history, ksvc_initial_history, gb_initial_history, knn_initial_history, mlp_initial_history)

        return

    def update_history(self, rf_models: List[Dict],
                       ksvc_models: List[Dict],
                       gb_models: List[Dict],
                       knn_models: List[Dict],
                       mlp_models: List[Dict]) -> None:
        """
        Updates the history of evaluated hyperparameter configurations for each model type.
        """

        # quick sanity checks
        assert len(rf_models) > 0 or len(ksvc_models) > 0 or len(gb_models) > 0 or len(knn_models) > 0 or len(mlp_models) > 0, "At least one model type must be provided for history update."

        # update rf history with the new evaluations
        for params in rf_models:
            self.rf_history['n_estimators'].append(params['n_estimators'])
            self.rf_history['criterion'].append(params['criterion'])
            self.rf_history['max_depth'].append(params['max_depth'])
            self.rf_history['max_features'].append(params['max_features'])
            self.rf_history['max_samples'].append(params['max_samples'])
            self.rf_history['class_weight'].append(params['class_weight'])
            self.rf_history[self.SOLUTION_VALIDATION_SCORE].append(params[self.SOLUTION_VALIDATION_SCORE])
            self.rf_history[self.SOLUTION_TRAIN_SCORE].append(params[self.SOLUTION_TRAIN_SCORE])
        # make sure that all history lists are of the same length after the update
        history_lengths = [len(self.rf_history[key]) for key in self.rf_history]
        assert len(set(history_lengths)) == 1, "Mismatch in lengths of RF history lists after update."

        # update ksvc history with the new evaluations
        for params in ksvc_models:
            self.ksvc_history['C'].append(params['C'])
            self.ksvc_history['kernel'].append(params['kernel'])
            self.ksvc_history['max_iter'].append(params['max_iter'])
            self.ksvc_history['class_weight'].append(params['class_weight'])
            self.ksvc_history['decision_function_shape'].append(params['decision_function_shape'])
            self.ksvc_history[self.SOLUTION_VALIDATION_SCORE].append(params[self.SOLUTION_VALIDATION_SCORE])
            self.ksvc_history[self.SOLUTION_TRAIN_SCORE].append(params[self.SOLUTION_TRAIN_SCORE])
        # make sure that all history lists are of the same length after the update
        history_lengths = [len(self.ksvc_history[key]) for key in self.ksvc_history]
        assert len(set(history_lengths)) == 1, "Mismatch in lengths of KSVC history lists after update."

        # update gb history with the new evaluations
        for params in gb_models:
            self.gb_history['loss'].append(params['loss'])
            self.gb_history['learning_rate'].append(params['learning_rate'])
            self.gb_history['n_estimators'].append(params['n_estimators'])
            self.gb_history['subsample'].append(params['subsample'])
            self.gb_history['criterion'].append(params['criterion'])
            self.gb_history['max_depth'].append(params['max_depth'])
            self.gb_history['max_features'].append(params['max_features'])
            self.gb_history[self.SOLUTION_VALIDATION_SCORE].append(params[self.SOLUTION_VALIDATION_SCORE])
            self.gb_history[self.SOLUTION_TRAIN_SCORE].append(params[self.SOLUTION_TRAIN_SCORE])
        # make sure that all history lists are of the same length after the update
        history_lengths = [len(self.gb_history[key]) for key in self.gb_history]
        assert len(set(history_lengths)) == 1, "Mismatch in lengths of GB history lists after update."

        # update knn history with the new evaluations
        for params in knn_models:
            self.knn_history['n_neighbors'].append(params['n_neighbors'])
            self.knn_history['weights'].append(params['weights'])
            self.knn_history['algorithm'].append(params['algorithm'])
            self.knn_history['leaf_size'].append(params['leaf_size'])
            self.knn_history['p'].append(params['p'])
            self.knn_history[self.SOLUTION_VALIDATION_SCORE].append(params[self.SOLUTION_VALIDATION_SCORE])
            self.knn_history[self.SOLUTION_TRAIN_SCORE].append(params[self.SOLUTION_TRAIN_SCORE])
        # make sure that all history lists are of the same length after the update
        history_lengths = [len(self.knn_history[key]) for key in self.knn_history]
        assert len(set(history_lengths)) == 1, "Mismatch in lengths of KNN history lists after update."

        # update mlp history with the new evaluations
        for params in mlp_models:
            self.mlp_history['layer_1'].append(params['layer_1'])
            self.mlp_history['layer_2'].append(params['layer_2'])
            self.mlp_history['layer_3'].append(params['layer_3'])
            self.mlp_history['layer_4'].append(params['layer_4'])
            self.mlp_history['layer_5'].append(params['layer_5'])
            self.mlp_history['activation'].append(params['activation'])
            self.mlp_history['solver'].append(params['solver'])
            self.mlp_history['max_iter'].append(params['max_iter'])
            self.mlp_history[self.SOLUTION_VALIDATION_SCORE].append(params[self.SOLUTION_VALIDATION_SCORE])
            self.mlp_history[self.SOLUTION_TRAIN_SCORE].append(params[self.SOLUTION_TRAIN_SCORE])
        # make sure that all history lists are of the same length after the update
        history_lengths = [len(self.mlp_history[key]) for key in self.mlp_history]
        assert len(set(history_lengths)) == 1, "Mismatch in lengths of MLP history lists after update."

        return

    def initialize_population(self) -> None:
        """
        Initializes the starting population for the evolutionary algorithm (EA) by generating a set of random hyperparameter configurations for each model type.
        Each model type gets an equal number of individuals, with the remainder distributed evenly across model types.
        """

        # quick sanity checks
        assert self.pop_size > 0, "Population size must be a positive integer."
        assert len(self.population) == 0, "Population has already been initialized."

        model_types = ['rf', 'ksvc', 'gb', 'knn', 'mlp']
        num_models = len(model_types)
        base_count = self.pop_size // num_models
        remainder = self.pop_size % num_models

        # shuffle model types to randomize which ones get the extra individuals from remainder
        shuffled_types = list(model_types)
        self.rng.shuffle(shuffled_types)

        for i, model_type in enumerate(shuffled_types):
            # each model type gets base_count individuals, plus 1 extra if within the remainder
            count = base_count + (1 if i < remainder else 0)

            for _ in range(count):
                # generate random hyperparameters for the selected model type
                if model_type == 'rf':
                    params = model_param_space.RandomForestParams().generate_random_parameters(self.rng)
                elif model_type == 'ksvc':
                    params = model_param_space.KernelSVCParams().generate_random_parameters(self.rng)
                elif model_type == 'gb':
                    params = model_param_space.GradientBoostParams(binary_class=self.binary_classification).generate_random_parameters(self.rng)
                elif model_type == 'knn':
                    params = model_param_space.KNeighborsClassifierParams().generate_random_parameters(self.rng)
                elif model_type == 'mlp':
                    params = model_param_space.MLPClassifierParams().generate_random_parameters(self.rng)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                # add the individual to the population
                self.population.append(Individual(params=params, model_type=model_type))

        assert len(self.population) == self.pop_size, "Population size mismatch after initialization."

        # compute the acquisition scores for the initial population based on the current history of evaluated hyperparameter configurations
        self.compute_acquisition_scores(self.population)

        return

    def compute_acquisition_scores(self, population: List[Individual]) -> None:
        """
        Computes the acquisition scores (UCB, EI, PI) for each individual in the population based on the current history of evaluated hyperparameter configurations.
        The computed scores are stored in the corresponding fields of each individual in the provided population.

        Args:
            population (List[Individual]): List of individuals for which to compute acquisition scores.
        """

        # quick sanity checks
        assert len(population) > 0, "Population is empty. Cannot compute acquisition scores."

        # will hold the specific index and aquistion score for each individual in the population
        rf_list, ksvc_list, gb_list, knn_list, mlp_list = [], [], [], [], []

        # hold all the ray remote jobs for acquisition score computation
        ray_jobs = []

        # iterate through the population and group individuals by model type for acquisition score computation
        for idx, individual in enumerate(population):
            model_type = individual.model_type
            if model_type == 'rf':
                rf_list.append((idx, individual.get_params()))
            elif model_type == 'ksvc':
                ksvc_list.append((idx, individual.get_params()))
            elif model_type == 'gb':
                gb_list.append((idx, individual.get_params()))
            elif model_type == 'knn':
                knn_list.append((idx, individual.get_params()))
            elif model_type == 'mlp':
                mlp_list.append((idx, individual.get_params()))
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        # load all rf models into ray remote jobs for acquisition score computation
        rf_candidates = {'n_estimators': [], 'criterion': [], 'max_depth': [], 'max_features': [], 'max_samples': [], 'class_weight': []}
        for params in rf_list:
            rf_candidates['n_estimators'].append(params[1]['n_estimators'])
            rf_candidates['criterion'].append(params[1]['criterion'])
            rf_candidates['max_depth'].append(params[1]['max_depth'])
            rf_candidates['max_features'].append(params[1]['max_features'])
            rf_candidates['max_samples'].append(params[1]['max_samples'])
            rf_candidates['class_weight'].append(params[1]['class_weight'])

        if len(rf_list) > 0:
            ray_jobs.append(bo_rf_optimizer.remote(param_space=RandomForestParams().get_parameter_space(),
                                                   history=self.rf_history,
                                                   n_initial_points=len(self.rf_history[self.SOLUTION_VALIDATION_SCORE]),
                                                   candidates=rf_candidates,
                                                   seed=self.seed,
                                                   xi=0.01,
                                                   kappa=1.0))

        # load all ksvc models into ray remote jobs for acquisition score computation
        ksvc_candidates = {'C': [], 'kernel': [], 'max_iter': [], 'class_weight': [], 'decision_function_shape': []}
        for params in ksvc_list:
            ksvc_candidates['C'].append(params[1]['C'])
            ksvc_candidates['kernel'].append(params[1]['kernel'])
            ksvc_candidates['max_iter'].append(params[1]['max_iter'])
            ksvc_candidates['class_weight'].append(params[1]['class_weight'])
            ksvc_candidates['decision_function_shape'].append(params[1]['decision_function_shape'])

        if len(ksvc_list) > 0:
            ray_jobs.append(bo_ksvc_optimizer.remote(param_space=KernelSVCParams().get_parameter_space(),
                                                    history=self.ksvc_history,
                                                    n_initial_points=len(self.ksvc_history[self.SOLUTION_VALIDATION_SCORE]),
                                                    candidates=ksvc_candidates,
                                                    seed=self.seed,
                                                    xi=0.01,
                                                    kappa=1.0))

        # load all gb models into ray remote jobs for acquisition score computation
        gb_candidates = {'loss': [], 'learning_rate': [], 'n_estimators': [], 'subsample': [], 'criterion': [], 'max_depth': [], 'max_features': []}
        for params in gb_list:
            gb_candidates['loss'].append(params[1]['loss'])
            gb_candidates['learning_rate'].append(params[1]['learning_rate'])
            gb_candidates['n_estimators'].append(params[1]['n_estimators'])
            gb_candidates['subsample'].append(params[1]['subsample'])
            gb_candidates['criterion'].append(params[1]['criterion'])
            gb_candidates['max_depth'].append(params[1]['max_depth'])
            gb_candidates['max_features'].append(params[1]['max_features'])

        if len(gb_list) > 0:
            ray_jobs.append(bo_gb_optimizer.remote(param_space=GradientBoostParams(binary_class=self.binary_classification).get_parameter_space(),
                                                history=self.gb_history,
                                                n_initial_points=len(self.gb_history[self.SOLUTION_VALIDATION_SCORE]),
                                                candidates=gb_candidates,
                                                seed=self.seed,
                                                xi=0.01,
                                                kappa=1.0))

        # load all knn models into ray remote jobs for acquisition score computation
        knn_candidates = {'n_neighbors': [], 'weights': [], 'algorithm': [], 'leaf_size': [], 'p': []}
        for params in knn_list:
            knn_candidates['n_neighbors'].append(params[1]['n_neighbors'])
            knn_candidates['weights'].append(params[1]['weights'])
            knn_candidates['algorithm'].append(params[1]['algorithm'])
            knn_candidates['leaf_size'].append(params[1]['leaf_size'])
            knn_candidates['p'].append(params[1]['p'])

        if len(knn_list) > 0:
            ray_jobs.append(bo_knn_optimizer.remote(param_space=KNeighborsClassifierParams().get_parameter_space(),
                                                    history=self.knn_history,
                                                    n_initial_points=len(self.knn_history[self.SOLUTION_VALIDATION_SCORE]),
                                                    candidates=knn_candidates,
                                                    seed=self.seed,
                                                    xi=0.01,
                                                    kappa=1.0))

        # load all mlp models into ray remote jobs for acquisition score computation
        mlp_candidates = {'layer_1': [], 'layer_2': [], 'layer_3': [], 'layer_4': [], 'layer_5': [], 'activation': [], 'solver': [], 'max_iter': []}
        for params in mlp_list:
            mlp_candidates['layer_1'].append(params[1]['layer_1'])
            mlp_candidates['layer_2'].append(params[1]['layer_2'])
            mlp_candidates['layer_3'].append(params[1]['layer_3'])
            mlp_candidates['layer_4'].append(params[1]['layer_4'])
            mlp_candidates['layer_5'].append(params[1]['layer_5'])
            mlp_candidates['activation'].append(params[1]['activation'])
            mlp_candidates['solver'].append(params[1]['solver'])
            mlp_candidates['max_iter'].append(params[1]['max_iter'])

        if len(mlp_list) > 0:
            ray_jobs.append(bo_mlp_optimizer.remote(param_space=MLPClassifierParams().get_parameter_space(),
                                                    history=self.mlp_history,
                                                    n_initial_points=len(self.mlp_history[self.SOLUTION_VALIDATION_SCORE]),
                                                    candidates=mlp_candidates,
                                                    seed=self.seed,
                                                    xi=0.01,
                                                    kappa=1.0))

        # process the results of the ray jobs for acquisition score computation (in parallel and asynchronously)
        while len(ray_jobs) > 0:
            done_ids, ray_jobs = ray.wait(ray_jobs, num_returns=min(len(ray_jobs), self.cores))
            for done_id in done_ids:
                acquisition_scores, model_type = ray.get(done_id)
                if model_type == 'rf':
                    for idx, score in zip([idx for idx, _ in rf_list], acquisition_scores):
                        population[idx].set_ucb(score['ucb'])
                        population[idx].set_ei(score['ei'])
                        population[idx].set_pi(score['pi'])
                elif model_type == 'ksvc':
                    for idx, score in zip([idx for idx, _ in ksvc_list], acquisition_scores):
                        population[idx].set_ucb(score['ucb'])
                        population[idx].set_ei(score['ei'])
                        population[idx].set_pi(score['pi'])
                elif model_type == 'gb':
                    for idx, score in zip([idx for idx, _ in gb_list], acquisition_scores):
                        population[idx].set_ucb(score['ucb'])
                        population[idx].set_ei(score['ei'])
                        population[idx].set_pi(score['pi'])
                elif model_type == 'knn':
                    for idx, score in zip([idx for idx, _ in knn_list], acquisition_scores):
                        population[idx].set_ucb(score['ucb'])
                        population[idx].set_ei(score['ei'])
                        population[idx].set_pi(score['pi'])
                elif model_type == 'mlp':
                    for idx, score in zip([idx for idx, _ in mlp_list], acquisition_scores):
                        population[idx].set_ucb(score['ucb'])
                        population[idx].set_ei(score['ei'])
                        population[idx].set_pi(score['pi'])
                else:
                    assert False, f"Unknown model type: {model_type} during acquisition score computation."

        return

    def evaluation(self,
                   rf_models: List[Dict],
                   ksvc_models: List[Dict],
                   gb_models: List[Dict],
                   knn_models: List[Dict],
                   mlp_models: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict]]:
        """
        Evaluates the performance of a given model type with specified hyperparameters using cross-validation.

        Args:
            rf_models (List[Dict]): A list of dictionaries containing hyperparameters for Random Forest models.
            ksvc_models (List[Dict]): A list of dictionaries containing hyperparameters for KernelSVC models.
            gb_models (List[Dict]): A list of dictionaries containing hyperparameters for Gradient Boost models.
            knn_models (List[Dict]): A list of dictionaries containing hyperparameters for K-Nearest Neighbors models.
            mlp_models (List[Dict]): A list of dictionaries containing hyperparameters for Multi-Layer Perceptron models.

        Returns:
            Tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict]]: A tuple containing the evaluated models for each type.
        """
        # quick sanity checks
        assert len(rf_models) > 0 or len(ksvc_models) > 0 or len(gb_models) > 0 or len(knn_models) > 0 or len(mlp_models) > 0, "At least one model type must be provided for evaluation."

        # get CV splits from base class
        cv_splits = self.get_cv_splits()

        # load all rf models into ray remote jobs for all 5-fold cross-validation splits (compute them in parallel and asynchronously)
        ray_jobs = []
        rf_results = [] # rf_models and rf_results are parallel lists, where rf_results[i] contains the results for rf_models[i]
        for model_id, rf_params in enumerate(rf_models):
            for X_train, y_train, X_validate, y_validate in cv_splits:
                ray_jobs.append(cv_random_forest.remote(X_train=X_train,
                                                        y_train=y_train,
                                                        X_validate=X_validate,
                                                        y_validate=y_validate,
                                                        model_params=RandomForestParams().eval_parameters(rf_params),
                                                        random_state=self.seed,
                                                        id=model_id,
                                                        binary_class=self.binary_classification,
                                                        labels=self.labels)
                                )
            rf_results.append({'train_auc': [], 'val_auc': [], 'error': False}) # initialize results for this model_id
        assert len(rf_results) == len(rf_models), "Mismatch between number of RF models and results."

        # process the results of the ray jobs for random forest models
        while len(ray_jobs) > 0:
            done_ids, ray_jobs = ray.wait(ray_jobs, num_returns=min(len(ray_jobs), self.cores))
            for done_id in done_ids:
                model_id, train_auc, val_auc, error = ray.get(done_id)
                if error == 1.0:
                    rf_results[model_id]['train_auc'].append(train_auc)
                    rf_results[model_id]['val_auc'].append(val_auc)
                else:
                    rf_results[model_id]['error'] = True
                    assert False, f"Error occurred during RF evaluation for model_id {model_id}. Params: {rf_models[model_id]}"

        # compute the average train and validation AUC for each model_id in rf_results
        for model_id in range(len(rf_results)):
            rf_models[model_id][self.SOLUTION_TRAIN_SCORE] = np.mean(rf_results[model_id]['train_auc'])
            rf_models[model_id][self.SOLUTION_VALIDATION_SCORE] = np.mean(rf_results[model_id]['val_auc'])

        # load all ksvc models into ray remote jobs for all 5-fold cross-validation splits (compute them in parallel and asynchronously)
        ray_jobs = []
        ksvc_results = [] # ksvc_models and ksvc_results are parallel lists, where ksvc_results[i] contains the results for ksvc_models[i]
        for model_id, ksvc_params in enumerate(ksvc_models):
            for X_train, y_train, X_validate, y_validate in cv_splits:
                ray_jobs.append(cv_kernel_svc.remote(X_train=X_train,
                                                     y_train=y_train,
                                                     X_validate=X_validate,
                                                     y_validate=y_validate,
                                                     model_params=KernelSVCParams().eval_parameters(ksvc_params),
                                                     random_state=self.seed,
                                                     id=model_id,
                                                     binary_class=self.binary_classification,
                                                     labels=self.labels)
                                )
            ksvc_results.append({'train_auc': [], 'val_auc': [], 'error': False}) # initialize results for this model_id

        # process the results of the ray jobs for kernel SVC models
        while len(ray_jobs) > 0:
            done_ids, ray_jobs = ray.wait(ray_jobs, num_returns=min(len(ray_jobs), self.cores))
            for done_id in done_ids:
                model_id, train_auc, val_auc, error = ray.get(done_id)
                if error == 1.0:
                    ksvc_results[model_id]['train_auc'].append(train_auc)
                    ksvc_results[model_id]['val_auc'].append(val_auc)
                else:
                    ksvc_results[model_id]['error'] = True
                    assert False, f"Error occurred during Kernel SVC evaluation for model_id {model_id}. Params: {ksvc_models[model_id]}"

        # compute the average train and validation AUC for each model_id in ksvc_results
        for model_id in range(len(ksvc_results)):
            ksvc_models[model_id][self.SOLUTION_TRAIN_SCORE] = np.mean(ksvc_results[model_id]['train_auc'])
            ksvc_models[model_id][self.SOLUTION_VALIDATION_SCORE] = np.mean(ksvc_results[model_id]['val_auc'])

        # load all gb models into ray remote jobs for all 5-fold cross-validation splits (compute them in parallel and asynchronously)
        ray_jobs = []
        gb_results = [] # gb_models and gb_results are parallel lists, where gb_results[i] contains the results for gb_models[i]
        for model_id, gb_params in enumerate(gb_models):
            for X_train, y_train, X_validate, y_validate in cv_splits:
                ray_jobs.append(cv_gradient_boost.remote(X_train=X_train,
                                                         y_train=y_train,
                                                         X_validate=X_validate,
                                                         y_validate=y_validate,
                                                         model_params=GradientBoostParams(binary_class=self.binary_classification).eval_parameters(gb_params),
                                                         random_state=self.seed,
                                                         id=model_id,
                                                         binary_class=self.binary_classification,
                                                         labels=self.labels)
                                )
            gb_results.append({'train_auc': [], 'val_auc': [], 'error': False})

        # process the results of the ray jobs for gradient boosting models
        while len(ray_jobs) > 0:
            done_ids, ray_jobs = ray.wait(ray_jobs, num_returns=min(len(ray_jobs), self.cores))
            for done_id in done_ids:
                model_id, train_auc, val_auc, error = ray.get(done_id)
                if error == 1.0:
                    gb_results[model_id]['train_auc'].append(train_auc)
                    gb_results[model_id]['val_auc'].append(val_auc)
                else:
                    gb_results[model_id]['error'] = True
                    assert False, f"Error occurred during Gradient Boost evaluation for model_id {model_id}. Params: {gb_models[model_id]}"

        # compute the average train and validation AUC for each model_id in gb_results
        for model_id in range(len(gb_results)):
            gb_models[model_id][self.SOLUTION_TRAIN_SCORE] = np.mean(gb_results[model_id]['train_auc'])
            gb_models[model_id][self.SOLUTION_VALIDATION_SCORE] = np.mean(gb_results[model_id]['val_auc'])

        # load all knn models into ray remote jobs for all 5-fold cross-validation splits (compute them in parallel and asynchronously)
        ray_jobs = []
        knn_results = [] # knn_models and knn_results are parallel lists, where knn_results[i] contains the results for knn_models[i]
        for model_id, knn_params in enumerate(knn_models):
            for X_train, y_train, X_validate, y_validate in cv_splits:
                ray_jobs.append(cv_knn.remote(X_train=X_train,
                                              y_train=y_train,
                                              X_validate=X_validate,
                                              y_validate=y_validate,
                                              model_params=KNeighborsClassifierParams().eval_parameters(knn_params),
                                              id=model_id,
                                              binary_class=self.binary_classification,
                                              labels=self.labels)
                                )
            knn_results.append({'train_auc': [], 'val_auc': [], 'error': False})

        # process the results of the ray jobs for k-nearest neighbors models
        while len(ray_jobs) > 0:
            done_ids, ray_jobs = ray.wait(ray_jobs, num_returns=min(len(ray_jobs), self.cores))
            for done_id in done_ids:
                model_id, train_auc, val_auc, error = ray.get(done_id)
                if error == 1.0:
                    knn_results[model_id]['train_auc'].append(train_auc)
                    knn_results[model_id]['val_auc'].append(val_auc)
                else:
                    knn_results[model_id]['error'] = True
                    assert False, f"Error occurred during KNN evaluation for model_id {model_id}. Params: {knn_models[model_id]}"

        # compute the average train and validation AUC for each model_id in knn_results
        for model_id in range(len(knn_results)):
            knn_models[model_id][self.SOLUTION_TRAIN_SCORE] = np.mean(knn_results[model_id]['train_auc'])
            knn_models[model_id][self.SOLUTION_VALIDATION_SCORE] = np.mean(knn_results[model_id]['val_auc'])

        # load all mlp models into ray remote jobs for all 5-fold cross-validation splits (compute them in parallel and asynchronously)
        ray_jobs = []
        mlp_results = [] # mlp_models and mlp_results are parallel lists, where mlp_results[i] contains the results for mlp_models[i]
        for model_id, mlp_params in enumerate(mlp_models):
            for X_train, y_train, X_validate, y_validate in cv_splits:
                ray_jobs.append(cv_mlp.remote(X_train=X_train,
                                              y_train=y_train,
                                              X_validate=X_validate,
                                              y_validate=y_validate,
                                              model_params=MLPClassifierParams().eval_parameters(mlp_params),
                                              random_state=self.seed,
                                              id=model_id,
                                              binary_class=self.binary_classification,
                                              labels=self.labels)
                                )
            mlp_results.append({'train_auc': [], 'val_auc': [], 'error': False})

        # process the results of the ray jobs for multi-layer perceptron models
        while len(ray_jobs) > 0:
            done_ids, ray_jobs = ray.wait(ray_jobs, num_returns=min(len(ray_jobs), self.cores))
            for done_id in done_ids:
                model_id, train_auc, val_auc, error = ray.get(done_id)
                if error == 1.0:
                    mlp_results[model_id]['train_auc'].append(train_auc)
                    mlp_results[model_id]['val_auc'].append(val_auc)
                else:
                    mlp_results[model_id]['error'] = True
                    assert False, f"Error occurred during MLP evaluation for model_id {model_id}. Params: {mlp_models[model_id]}"

        # compute the average train and validation AUC for each model_id in mlp_results
        for model_id in range(len(mlp_results)):
            mlp_models[model_id][self.SOLUTION_TRAIN_SCORE] = np.mean(mlp_results[model_id]['train_auc'])
            mlp_models[model_id][self.SOLUTION_VALIDATION_SCORE] = np.mean(mlp_results[model_id]['val_auc'])

        return rf_models, ksvc_models, gb_models, knn_models, mlp_models

    def generate_acquisition_scores_for_nsga(self, population: List[Individual]) -> npt.NDArray:
        """
        Generates a set of acquisition scores from the population formatted for the non_dominated_sorting function.
        The acquisition scores are extracted as tuples of floats representing the objective values for NSGA-II.

        Since non_dominated_sorting expects maximization, acquisition scores are used directly:
        - UCB: Higher values indicate better exploration/exploitation trade-off
        - EI: Higher values indicate greater expected improvement
        - PI: Higher values indicate greater probability of improvement

        Note: UCB values can be negative when the predicted performance is poor. The non_dominated_sorting
        function handles this correctly as it only compares relative values for dominance relationships.

        Args:
            population (List[Individual]): The list of individuals for which to generate acquisition scores.

        Returns:
            npt.NDArray: A numpy array where each element is a tuple of floats representing the acquisition scores
                        for each individual in the population. The tuple size is 2 or 3 depending on which
                        acquisition functions are enabled.
        """
        # quick sanity checks
        assert len(population) > 0, "Population is empty. Cannot generate acquisition scores."

        # extract acquisition scores from the population and format as tuples
        acquisition_scores = []
        for individual in population:
            scores = []
            scores.append(float(individual.get_ucb()))
            scores.append(float(individual.get_ei()))
            scores.append(float(individual.get_pi()))

            acquisition_scores.append(tuple(scores))

        # final sanity checks
        assert len(acquisition_scores) == len(population), "Mismatch between acquisition scores and population size."
        assert all(isinstance(score, tuple) for score in acquisition_scores), "All acquisition scores must be tuples."
        assert all(all(isinstance(val, float) for val in score) for score in acquisition_scores), "All acquisition scores must be floats."

        # Return as numpy array (non_dominated_sorting expects npt.NDArray)
        # Use dtype=object and np.empty to preserve tuple structure - each element is a tuple
        result = np.empty(len(acquisition_scores), dtype=object)
        for i, score in enumerate(acquisition_scores):
            result[i] = score

        return result

    def generate_offspring(self, parents: List[Individual]) -> List[Individual]:
        """
        Generates offspring from the given parents using mutation operations only.
        Crossover cannot be used because the hyperparameter spaces for each model type are not compatible with each other.

        Args:
            parents (List[Individual]): A list of parent individuals from the population.

        Returns:
            List[Individual]: A list of offspring individuals generated from the parents.
        """
        # quick sanity checks
        assert len(parents) > 0, "Parents list is empty. Cannot generate offspring"

        offspring = []

        for parent in parents:
            # generate a mutated offspring from the parent
            child = self.mutate(parent)
            offspring.append(child)

        return offspring

    def final_model_evaluation(self) -> None:
        """
        Evaluates the final model after the optimization process is complete.
        This function collects all model/hyperparameter combinations that tie for best validation performance,
        randomly selects one, fits it on the entire training set, and evaluates it on the test set.
        """

        # find best performances across all history data structures
        best_rf = max(self.rf_history[self.SOLUTION_VALIDATION_SCORE], default=-np.inf)
        best_ksvc = max(self.ksvc_history[self.SOLUTION_VALIDATION_SCORE], default=-np.inf)
        best_gb = max(self.gb_history[self.SOLUTION_VALIDATION_SCORE], default=-np.inf)
        best_knn = max(self.knn_history[self.SOLUTION_VALIDATION_SCORE], default=-np.inf)
        best_mlp = max(self.mlp_history[self.SOLUTION_VALIDATION_SCORE], default=-np.inf)

        # determine the overall best validation score
        best_overall_score = max([best_rf, best_ksvc, best_gb, best_knn, best_mlp])
        self.best_perf = best_overall_score

        # collect all model/hyperparameter combinations that achieve the best validation score
        best_candidates = []

        # collect all RF models that achieve best performance
        if best_rf == best_overall_score:
            for idx, score in enumerate(self.rf_history[self.SOLUTION_VALIDATION_SCORE]):
                if score == best_overall_score:
                    best_candidates.append({
                        'model_type': 'RF',
                        'params': {
                            'n_estimators': self.rf_history['n_estimators'][idx],
                            'criterion': self.rf_history['criterion'][idx],
                            'max_depth': self.rf_history['max_depth'][idx],
                            'max_features': self.rf_history['max_features'][idx],
                            'max_samples': self.rf_history['max_samples'][idx],
                            'class_weight': self.rf_history['class_weight'][idx]
                        }
                    })

        # collect all KSVC models that achieve best performance
        if best_ksvc == best_overall_score:
            for idx, score in enumerate(self.ksvc_history[self.SOLUTION_VALIDATION_SCORE]):
                if score == best_overall_score:
                    best_candidates.append({
                        'model_type': 'KSVC',
                        'params': {
                            'C': self.ksvc_history['C'][idx],
                            'kernel': self.ksvc_history['kernel'][idx],
                            'max_iter': self.ksvc_history['max_iter'][idx],
                            'class_weight': self.ksvc_history['class_weight'][idx],
                            'decision_function_shape': self.ksvc_history['decision_function_shape'][idx]
                        }
                    })

        # collect all GB models that achieve best performance
        if best_gb == best_overall_score:
            for idx, score in enumerate(self.gb_history[self.SOLUTION_VALIDATION_SCORE]):
                if score == best_overall_score:
                    best_candidates.append({
                        'model_type': 'GB',
                        'params': {
                            'loss': self.gb_history['loss'][idx],
                            'learning_rate': self.gb_history['learning_rate'][idx],
                            'n_estimators': self.gb_history['n_estimators'][idx],
                            'subsample': self.gb_history['subsample'][idx],
                            'criterion': self.gb_history['criterion'][idx],
                            'max_depth': self.gb_history['max_depth'][idx],
                            'max_features': self.gb_history['max_features'][idx]
                        }
                    })

        # collect all KNN models that achieve best performance
        if best_knn == best_overall_score:
            for idx, score in enumerate(self.knn_history[self.SOLUTION_VALIDATION_SCORE]):
                if score == best_overall_score:
                    best_candidates.append({
                        'model_type': 'KNN',
                        'params': {
                            'n_neighbors': self.knn_history['n_neighbors'][idx],
                            'weights': self.knn_history['weights'][idx],
                            'algorithm': self.knn_history['algorithm'][idx],
                            'leaf_size': self.knn_history['leaf_size'][idx],
                            'p': self.knn_history['p'][idx]
                        }
                    })

        # collect all MLP models that achieve best performance
        if best_mlp == best_overall_score:
            for idx, score in enumerate(self.mlp_history[self.SOLUTION_VALIDATION_SCORE]):
                if score == best_overall_score:
                    best_candidates.append({
                        'model_type': 'MLP',
                        'params': {
                            'layer_1': self.mlp_history['layer_1'][idx],
                            'layer_2': self.mlp_history['layer_2'][idx],
                            'layer_3': self.mlp_history['layer_3'][idx],
                            'layer_4': self.mlp_history['layer_4'][idx],
                            'layer_5': self.mlp_history['layer_5'][idx],
                            'activation': self.mlp_history['activation'][idx],
                            'solver': self.mlp_history['solver'][idx],
                            'max_iter': self.mlp_history['max_iter'][idx]
                        }
                    })

        # randomly select one model/hyperparameter combination from the best candidates
        assert len(best_candidates) > 0, "No best candidates found for final model evaluation."
        selected_candidate = best_candidates[self.rng.integers(0, len(best_candidates))]

        print(f"Selected final model: {selected_candidate['model_type']} with validation score: {best_overall_score}")
        print(f"Total candidates with best performance: {len(best_candidates)}")

        # evaluate best individual on test set using BaseEA method
        train_score, test_score = self.model_test_evaluation(
            model_type=selected_candidate['model_type'],
            model_params=selected_candidate['params']
        )

        print(f"Final model train score: {train_score}")
        print(f"Final model test score: {test_score}")

        # store the best individual with all performance metrics
        best_individual = Individual(selected_candidate['params'], selected_candidate['model_type'].lower())
        best_individual.set_val_performance(best_overall_score)
        best_individual.set_train_performance(train_score)
        best_individual.set_test_performance(test_score)
        self.best_ind = best_individual

        return

    def save_results(self, save_dir: str) -> None:
        """
        Save final results using the best individual evaluated at the end of evolve().
        The JSON will contain train, validation, and test accuracy as well as the hyperparameter settings.
        """
        import os
        import json

        assert self.best_ind is not None, "No best individual found. Run evolve() first."

        print(f"Best individual params: {self.best_ind.get_params()}", flush=True)
        print(f"Best validation performance: {self.best_perf}", flush=True)

        # Create output directory structure if it doesn't exist
        task_output_dir = os.path.join(save_dir)
        os.makedirs(task_output_dir, exist_ok=True)

        # Save best individual results as JSON
        best_results = {
            "task_id": self.task_id,
            "model_type": self.best_ind.model_type,
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