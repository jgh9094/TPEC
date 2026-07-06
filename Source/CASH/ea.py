##########################################################################################
#
# Class dedicated to the evolutionary algorithm (EA) optimization of hyperparameters for machine learning models.
#
##########################################################################################

import numpy as np
import pandas as pd
import os
import ray
import sklearn as skl
import numpy.typing as npt
import copy as cp

from typeguard import typechecked
from typing import List, Dict, Tuple
from Source.Base import model_param_space
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from openml import tasks
from sklearn.metrics import roc_auc_score, log_loss


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from Source.Base.ray_utils import cv_random_forest, cv_kernel_svc, cv_gradient_boost, cv_knn, cv_mlp
from Source.Base.model_param_space import RandomForestParams, KernelSVCParams, GradientBoostParams, KNeighborsClassifierParams, MLPClassifierParams
from .bo_ray_utils import bo_rf_optimizer, bo_ksvc_optimizer, bo_gb_optimizer, bo_knn_optimizer, bo_mlp_optimizer
import Source.CASH.nsag_toolset as nsga

@typechecked
class EA:
    def __init__(self,
                 seed: int,
                 pop_size: int,
                 cores: int,
                 mut_prob: float,
                 mut_var: float,
                 save_directory: str,
                 initial_history_size: int) -> None:
        """
        Initializes the EA class with the provided parameters.

        Args:
            seed (int): Random seed for reproducibility.
            pop_size (int): Population size for the evolutionary algorithm.
            cores (int): Number of CPU cores to use for parallel processing.
            mut_var (float): Mutation variance for the evolutionary algorithm.
            save_directory (str): Directory to save results.
            initial_history_size (int): Size of the initial history per model for the Bayesian optimizer.
        """
        # quick sanity checks
        assert seed >= 0, "Seed must be a non-negative integer."
        assert pop_size > 0, "Population size must be a positive integer."
        assert cores > 0, "Number of cores must be a positive integer."
        assert 0.0 <= mut_var, "Mutation variance must be non-negative."
        assert 0.0 <= mut_prob <= 1.0, "Mutation probability must be between 0 and 1    ."
        assert initial_history_size > 0, "Initial history size must be a positive integer."

        # global paramters for the EA and BO optimization process
        self.SOLUTION_TYPE = 'SOLUTION_TYPE'
        self.SOLUTION_PARAMS = 'SOLUTION_PARAMS'
        self.SOLUTION_VALIDATION_SCORE = 'SOLUTION_VALIDATION_SCORE'
        self.SOLUTION_TRAIN_SCORE = 'SOLUTION_TRAIN_SCORE'
        self.SOLUTION_UCB = 'SOLUTION_UCB'
        self.SOLUTION_EI = 'SOLUTION_EI'
        self.SOLUTION_PI = 'SOLUTION_PI'

        # save the parameters
        self.seed = seed
        self.pop_size = pop_size
        self.cores = cores
        self.save_directory = save_directory
        self.initial_history_size = initial_history_size
        self.rng = np.random.default_rng(seed)

        # variables tracked during the optimization process
        self.total_evaluations = 0
        self.rf_history = {'n_estimators': [], 'criterion': [], 'max_depth': [], 'max_features': [], 'max_samples': [], 'class_weight': [], self.SOLUTION_VALIDATION_SCORE: [], self.SOLUTION_TRAIN_SCORE: []}
        self.ksvc_history = {'C': [], 'kernel': [], 'max_iter': [], 'class_weight': [], 'decision_function_shape': [], self.SOLUTION_VALIDATION_SCORE: [], self.SOLUTION_TRAIN_SCORE: []}
        self.gb_history = {'loss': [], 'learning_rate': [], 'n_estimators': [], 'subsample': [], 'criterion': [], 'max_depth': [], 'max_features': [], self.SOLUTION_VALIDATION_SCORE: [], self.SOLUTION_TRAIN_SCORE: []}
        self.knn_history = {'n_neighbors': [], 'weights': [], 'algorithm': [], 'leaf_size': [], 'p': [], self.SOLUTION_VALIDATION_SCORE: [], self.SOLUTION_TRAIN_SCORE: []}
        self.mlp_history = {'layer_1': [], 'layer_2': [], 'layer_3': [], 'layer_4': [], 'layer_5': [], 'activation': [], 'solver': [], 'max_iter': [], self.SOLUTION_VALIDATION_SCORE: [], self.SOLUTION_TRAIN_SCORE: []}

        # ea specific variables
        self.population = []
        self.mut_prob = mut_prob
        self.mut_var = mut_var
        return

    def load_data(self,
                  task_id: int,
                  data_dir: str,
                  train_p: float) -> None:
        """
        Loads the dataset for the specified OpenML task ID and applies preprocessing.
        data_dir must contain task_{task_id}.csv and tasks_summary.csv files.

        Args:
            task_id (int): OpenML task ID to load dataset from.
            data_dir (str): Directory where datasets are stored.
            train_p (float): Proportion of the dataset to use for training.
        """
        # quick sanity checks
        assert task_id > 0, "Task ID must be a positive integer."
        assert os.path.exists(data_dir), f"Data directory '{data_dir}' does not exist."
        assert os.path.exists(os.path.join(data_dir, f"task_{task_id}.csv")), f"Dataset file for task {task_id} not found."
        assert os.path.exists(os.path.join(data_dir, f"tasks_summary.csv")), f"Summary file for task {task_id} not found."

        # store task_id for later use
        self.task_id = task_id

        # load the summary CSV
        summary_csv = pd.read_csv(os.path.join(data_dir, f"tasks_summary.csv"))

        # find the row corresponding to the specified task_id
        tasks_summary_row = summary_csv[summary_csv['task_id'] == task_id]
        assert not tasks_summary_row.empty, f"No summary information found for task {task_id}."

        # extract target_name to know which column is the target variable
        target_name = tasks_summary_row['target_name'].values[0]
        assert target_name is not None, f"Target name for task {task_id} is missing in the summary."

        # extract the number of classes to determine if it's a binary or multi-class classification problem
        num_classes = tasks_summary_row['num_classes'].values[0]
        assert num_classes is not None, f"Number of classes for task {task_id} is missing in the summary."
        self.binary_classification = bool(num_classes == 2)

        # load the specified dataset
        self.data_set = pd.read_csv(os.path.join(data_dir, f"task_{task_id}.csv"))

        # split the dataset into features and target
        X = self.data_set.drop(columns=[target_name])
        y = self.data_set[target_name].values

        # make a list of all unique classes in the target variable
        self.labels = np.unique(y)

        # generate an initial train-test split for the evolutionary algorithm (train_p is the proportion of the dataset to use for training)
        self.X_train, self.X_test, self.y_train, self.y_test = skl.model_selection.train_test_split(
            X, y, train_size=train_p, random_state=self.seed, shuffle=True, stratify=y
        )

        # generate a 5-fold cross-validation split for the evolutionary algorithm based on the training set and store the indices for each fold
        self.cv_splits = list(skl.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed).split(self.X_train, self.y_train))

        # get the categorical indicator from OpenML
        task = tasks.get_task(task_id)
        dataset = task.get_dataset()
        categorical_indicator = dataset.get_data()[2]  # (X, y, categorical_indicator, attribute_names)

        # identify categorical and numerical columns based on the categorical indicator and store for later use
        self.categorical_cols = [col for col, is_cat in zip(self.X_train.columns, categorical_indicator) if is_cat]
        self.numerical_cols = [col for col, is_cat in zip(self.X_train.columns, categorical_indicator) if not is_cat]

        # use local references for the rest of this method
        categorical_cols = self.categorical_cols
        numerical_cols = self.numerical_cols

        # generate each fold's train and validation sets with preprocessing
        # fold 1
        X_train_f1_raw = self.X_train.iloc[self.cv_splits[0][0]].reset_index(drop=True)
        X_val_f1_raw = self.X_train.iloc[self.cv_splits[0][1]].reset_index(drop=True)

        preprocessor_f1 = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        self.X_train_f1_rf = ray.put(preprocessor_f1.fit_transform(X_train_f1_raw))
        self.X_val_f1_rf = ray.put(preprocessor_f1.transform(X_val_f1_raw))
        self.y_train_f1_rf = ray.put(self.y_train[self.cv_splits[0][0]])
        self.y_val_f1_rf = ray.put(self.y_train[self.cv_splits[0][1]])

        # fold 2
        X_train_f2_raw = self.X_train.iloc[self.cv_splits[1][0]].reset_index(drop=True)
        X_val_f2_raw = self.X_train.iloc[self.cv_splits[1][1]].reset_index(drop=True)

        preprocessor_f2 = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        self.X_train_f2_rf = ray.put(preprocessor_f2.fit_transform(X_train_f2_raw))
        self.X_val_f2_rf = ray.put(preprocessor_f2.transform(X_val_f2_raw))
        self.y_train_f2_rf = ray.put(self.y_train[self.cv_splits[1][0]])
        self.y_val_f2_rf = ray.put(self.y_train[self.cv_splits[1][1]])

        # fold 3
        X_train_f3_raw = self.X_train.iloc[self.cv_splits[2][0]].reset_index(drop=True)
        X_val_f3_raw = self.X_train.iloc[self.cv_splits[2][1]].reset_index(drop=True)

        preprocessor_f3 = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        self.X_train_f3_rf = ray.put(preprocessor_f3.fit_transform(X_train_f3_raw))
        self.X_val_f3_rf = ray.put(preprocessor_f3.transform(X_val_f3_raw))
        self.y_train_f3_rf = ray.put(self.y_train[self.cv_splits[2][0]])
        self.y_val_f3_rf = ray.put(self.y_train[self.cv_splits[2][1]])

        # fold 4
        X_train_f4_raw = self.X_train.iloc[self.cv_splits[3][0]].reset_index(drop=True)
        X_val_f4_raw = self.X_train.iloc[self.cv_splits[3][1]].reset_index(drop=True)

        preprocessor_f4 = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        self.X_train_f4_rf = ray.put(preprocessor_f4.fit_transform(X_train_f4_raw))
        self.X_val_f4_rf = ray.put(preprocessor_f4.transform(X_val_f4_raw))
        self.y_train_f4_rf = ray.put(self.y_train[self.cv_splits[3][0]])
        self.y_val_f4_rf = ray.put(self.y_train[self.cv_splits[3][1]])

        # fold 5
        X_train_f5_raw = self.X_train.iloc[self.cv_splits[4][0]].reset_index(drop=True)
        X_val_f5_raw = self.X_train.iloc[self.cv_splits[4][1]].reset_index(drop=True)

        preprocessor_f5 = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        self.X_train_f5_rf = ray.put(preprocessor_f5.fit_transform(X_train_f5_raw))
        self.X_val_f5_rf = ray.put(preprocessor_f5.transform(X_val_f5_raw))
        self.y_train_f5_rf = ray.put(self.y_train[self.cv_splits[4][0]])
        self.y_val_f5_rf = ray.put(self.y_train[self.cv_splits[4][1]])

        return

    def evolve_3d(self, gens: int) -> None:
        """
        Evolves the hyperparameters for the specified model type over a given number of generations.
        This function assumes that we are optimizing all 3 acquisition functions (UCB, EI, PI) simultaneously for the NSGA-II algorithm.

        Args:
            gens (int): Number of generations to evolve.
            ucb (bool): Whether to use Upper Confidence Bound acquisition function.
            ei (bool): Whether to use Expected Improvement acquisition function.
            pi (bool): Whether to use Probability of Improvement acquisition function.
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
            rf_models, ksvc_models, gb_models, knn_models, mlp_models = [ind[self.SOLUTION_PARAMS] for ind in self.population if ind[self.SOLUTION_TYPE] == 'rf'], \
                [ind[self.SOLUTION_PARAMS] for ind in self.population if ind[self.SOLUTION_TYPE] == 'ksvc'], \
                [ind[self.SOLUTION_PARAMS] for ind in self.population if ind[self.SOLUTION_TYPE] == 'gb'], \
                [ind[self.SOLUTION_PARAMS] for ind in self.population if ind[self.SOLUTION_TYPE] == 'knn'], \
                [ind[self.SOLUTION_PARAMS] for ind in self.population if ind[self.SOLUTION_TYPE] == 'mlp']

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

    def update_history(self, rf_models: List[Dict], ksvc_models: List[Dict], gb_models: List[Dict], knn_models: List[Dict], mlp_models: List[Dict]) -> None:
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
        Models types and parameters are randomly generated based on self.rng, the defined parameter spaces in model_param_space.py, and self.pop_size.
        """

        # quick sanity checks
        assert self.pop_size > 0, "Population size must be a positive integer."
        assert len(self.population) == 0, "Population has already been initialized."

        for _ in range(self.pop_size):
            # randomly select a model type for this individual in the population
            model_type = self.rng.choice(['rf', 'ksvc', 'gb', 'knn', 'mlp'])

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
            self.population.append({self.SOLUTION_TYPE: model_type, # what type of model this individual represents
                                    self.SOLUTION_PARAMS: params, # what are the hyperparameters for this individual
                                    self.SOLUTION_UCB: None, # what is the UCB score for this individual (to be computed later)
                                    self.SOLUTION_EI: None, # what is the EI score for this individual (to be computed later)
                                    self.SOLUTION_PI: None, # what is the PI score for this individual (to be computed later)
                                    self.SOLUTION_VALIDATION_SCORE: None, # what is the validation score for this individual (to be computed later)
                                    })
        assert len(self.population) == self.pop_size, "Population size mismatch after initialization."

        # compute the acquisition scores for the initial population based on the current history of evaluated hyperparameter configurations
        self.compute_acquisition_scores(self.population)

        return

    def compute_acquisition_scores(self, population) -> None:
        """
        Computes the acquisition scores (UCB, EI, PI) for each individual in the population based on the current history of evaluated hyperparameter configurations.
        The computed scores are stored in the corresponding fields of each individual in the provided population.

        Args:
            population (list): List of individuals for which to compute acquisition scores.
        """

        # quick sanity checks
        assert len(population) > 0, "Population is empty. Cannot compute acquisition scores."

        # will hold the specific index and aquistion score for each individual in the population
        rf_list, ksvc_list, gb_list, knn_list, mlp_list = [], [], [], [], []

        # hold all the ray remote jobs for acquisition score computation
        ray_jobs = []

        # iterate through the population and group individuals by model type for acquisition score computation
        for idx, individual in enumerate(population):
            model_type = individual[self.SOLUTION_TYPE]
            if model_type == 'rf':
                rf_list.append((idx, individual[self.SOLUTION_PARAMS]))
            elif model_type == 'ksvc':
                ksvc_list.append((idx, individual[self.SOLUTION_PARAMS]))
            elif model_type == 'gb':
                gb_list.append((idx, individual[self.SOLUTION_PARAMS]))
            elif model_type == 'knn':
                knn_list.append((idx, individual[self.SOLUTION_PARAMS]))
            elif model_type == 'mlp':
                mlp_list.append((idx, individual[self.SOLUTION_PARAMS]))
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
                        population[idx][self.SOLUTION_UCB] = score['ucb']
                        population[idx][self.SOLUTION_EI] = score['ei']
                        population[idx][self.SOLUTION_PI] = score['pi']
                elif model_type == 'ksvc':
                    for idx, score in zip([idx for idx, _ in ksvc_list], acquisition_scores):
                        population[idx][self.SOLUTION_UCB] = score['ucb']
                        population[idx][self.SOLUTION_EI] = score['ei']
                        population[idx][self.SOLUTION_PI] = score['pi']
                elif model_type == 'gb':
                    for idx, score in zip([idx for idx, _ in gb_list], acquisition_scores):
                        population[idx][self.SOLUTION_UCB] = score['ucb']
                        population[idx][self.SOLUTION_EI] = score['ei']
                        population[idx][self.SOLUTION_PI] = score['pi']
                elif model_type == 'knn':
                    for idx, score in zip([idx for idx, _ in knn_list], acquisition_scores):
                        population[idx][self.SOLUTION_UCB] = score['ucb']
                        population[idx][self.SOLUTION_EI] = score['ei']
                        population[idx][self.SOLUTION_PI] = score['pi']
                elif model_type == 'mlp':
                    for idx, score in zip([idx for idx, _ in mlp_list], acquisition_scores):
                        population[idx][self.SOLUTION_UCB] = score['ucb']
                        population[idx][self.SOLUTION_EI] = score['ei']
                        population[idx][self.SOLUTION_PI] = score['pi']
                else:
                    assert False, f"Unknown model type: {model_type} during acquisition score computation."

        # quick sanity check to ensure all acquisition scores have been computed for the population
        for individual in population:
            assert individual[self.SOLUTION_UCB] is not None, "UCB score not computed for an individual in the population."
            assert individual[self.SOLUTION_EI] is not None, "EI score not computed for an individual in the population."
            assert individual[self.SOLUTION_PI] is not None, "PI score not computed for an individual in the population."

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

        # make list of tuples with each train/validation split for each fold of the 5-fold cross-validation
        cv_splits = [(self.X_train_f1_rf, self.y_train_f1_rf, self.X_val_f1_rf, self.y_val_f1_rf),
                     (self.X_train_f2_rf, self.y_train_f2_rf, self.X_val_f2_rf, self.y_val_f2_rf),
                     (self.X_train_f3_rf, self.y_train_f3_rf, self.X_val_f3_rf, self.y_val_f3_rf),
                     (self.X_train_f4_rf, self.y_train_f4_rf, self.X_val_f4_rf, self.y_val_f4_rf),
                     (self.X_train_f5_rf, self.y_train_f5_rf, self.X_val_f5_rf, self.y_val_f5_rf)
                     ]

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

    def generate_acquisition_scores_for_nsga(self, population: list) -> npt.NDArray:
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
            population (list): The list of individuals for which to generate acquisition scores.

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
            assert individual[self.SOLUTION_UCB] is not None, "UCB score not found for an individual."
            scores.append(float(individual[self.SOLUTION_UCB]))
            assert individual[self.SOLUTION_EI] is not None, "EI score not found for an individual."
            scores.append(float(individual[self.SOLUTION_EI]))
            assert individual[self.SOLUTION_PI] is not None, "PI score not found for an individual."
            scores.append(float(individual[self.SOLUTION_PI]))

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

    def generate_offspring(self, parents: List[Dict]) -> List[Dict]:
        """
        Generates offspring from the given parents using mutation operations only.
        Crossover cannot be used because the hyperparameter spaces for each model type are not compatible with each other.

        Args:
            parents (List[Dict]): A list of parent individuals from the population.

        Returns:
            List[Dict]: A list of offspring individuals generated from the parents.
        """
        # quick sanity checks
        assert len(parents) > 0, "Parents list is empty. Cannot generate offspring"

        offspring = []

        for parent in parents:
            # quick sanity check to ensure the parent has the required keys
            assert self.SOLUTION_TYPE in parent, "Parent individual missing 'type' key."
            assert self.SOLUTION_PARAMS in parent, "Parent individual missing 'params' key."

            # generate a mutated offspring from the parent
            child = self.mutate(parent)

            # quick sanity check to ensure the child has the required keys
            assert self.SOLUTION_TYPE in child, "Child individual missing 'type' key."
            assert self.SOLUTION_PARAMS in child, "Child individual missing 'params' key."

            offspring.append(child)

        return offspring

    def mutate(self, individual: Dict) -> Dict:
        """
        Mutates the given individual by randomly altering its hyperparameters based on the mutation probability.

        Args:
            individual (Dict): The individual to be mutated.

        Returns:
            Dict: A new individual with mutated hyperparameters.
        """
        # quick sanity checks
        assert self.SOLUTION_TYPE in individual, "Individual missing 'type' key."
        assert self.SOLUTION_PARAMS in individual, "Individual missing 'params' key."

        model_type = individual[self.SOLUTION_TYPE]
        params = individual[self.SOLUTION_PARAMS]

        # mutate the hyperparameters based on the model type
        if model_type == 'rf':
            mutated_params = model_param_space.RandomForestParams().mutate_parameters(model_params=params, var=self.mut_var, mut_rate=self.mut_prob, rng=self.rng)
        elif model_type == 'ksvc':
            mutated_params = model_param_space.KernelSVCParams().mutate_parameters(model_params=params, var=self.mut_var, mut_rate=self.mut_prob, rng=self.rng)
        elif model_type == 'gb':
            mutated_params = model_param_space.GradientBoostParams(binary_class=self.binary_classification).mutate_parameters(model_params=params, var=self.mut_var, mut_rate=self.mut_prob, rng=self.rng)
        elif model_type == 'knn':
            mutated_params = model_param_space.KNeighborsClassifierParams().mutate_parameters(model_params=params, var=self.mut_var, mut_rate=self.mut_prob, rng=self.rng)
        elif model_type == 'mlp':
            mutated_params = model_param_space.MLPClassifierParams().mutate_parameters(model_params=params, var=self.mut_var, mut_rate=self.mut_prob, rng=self.rng)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # return a new individual with the same type but mutated parameters
        return {self.SOLUTION_TYPE: cp.deepcopy(model_type),
                self.SOLUTION_PARAMS: cp.deepcopy(mutated_params),
                self.SOLUTION_UCB: None,
                self.SOLUTION_EI: None,
                self.SOLUTION_PI: None,
                self.SOLUTION_VALIDATION_SCORE: None,
                self.SOLUTION_TRAIN_SCORE: None}

    def final_model_evaluation(self,) -> None:
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

        # collect all model/hyperparameter combinations that achieve the best validation score
        best_candidates = []

        # collect all RF models that achieve best performance
        if best_rf == best_overall_score:
            for idx, score in enumerate(self.rf_history[self.SOLUTION_VALIDATION_SCORE]):
                if score == best_overall_score:
                    best_candidates.append({
                        'model_type': 'rf',
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
                        'model_type': 'ksvc',
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
                        'model_type': 'gb',
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
                        'model_type': 'knn',
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
                        'model_type': 'mlp',
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


        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'), self.categorical_cols)
            ],
            remainder='passthrough'
        )

        X_train_preprocessed = preprocessor.fit_transform(self.X_train)
        X_test_preprocessed = preprocessor.transform(self.X_test)

        # fit the selected model on the entire training set and evaluate on test set
        model_type = selected_candidate['model_type']
        params = selected_candidate['params']

        if model_type == 'rf':
            eval_params = RandomForestParams().eval_parameters(params)
            model = RandomForestClassifier(**eval_params, random_state=self.seed)
        elif model_type == 'ksvc':
            eval_params = KernelSVCParams().eval_parameters(params)
            model = SVC(**eval_params, random_state=self.seed, probability=True)
        elif model_type == 'gb':
            eval_params = GradientBoostParams(binary_class=self.binary_classification).eval_parameters(params)
            model = GradientBoostingClassifier(**eval_params, random_state=self.seed)
        elif model_type == 'knn':
            eval_params = KNeighborsClassifierParams().eval_parameters(params)
            model = KNeighborsClassifier(**eval_params)
        elif model_type == 'mlp':
            eval_params = MLPClassifierParams().eval_parameters(params)
            model = MLPClassifier(**eval_params, random_state=self.seed)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # fit the model on the entire training set
        model.fit(X_train_preprocessed, self.y_train)

        # evaluate on train and test sets (matching the evaluation method in ray_utils.py)
        train_pred_proba = model.predict_proba(X_train_preprocessed)
        test_pred_proba = model.predict_proba(X_test_preprocessed)

        if self.binary_classification:
            train_score = float(roc_auc_score(self.y_train, train_pred_proba[:, 1]))
            test_score = float(roc_auc_score(self.y_test, test_pred_proba[:, 1]))
        else:
            train_score = -float(log_loss(self.y_train, train_pred_proba, labels=self.labels))
            test_score = -float(log_loss(self.y_test, test_pred_proba, labels=self.labels))

        print(f"Final model train score: {train_score}")
        print(f"Final model test score: {test_score}")

        # store the final model and results
        self.final_model = model
        self.final_model_type = model_type
        self.final_model_params = params
        self.final_model_train_score = train_score
        self.final_model_test_score = test_score
        self.final_model_validation_score = best_overall_score

        return

    def save_results(self) -> None:
        return