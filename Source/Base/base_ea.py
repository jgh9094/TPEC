##########################################################################################
#
# Abstract base class for evolutionary algorithm (EA) optimization of hyperparameters.
# All EA implementations (CASH, HPO) should derive from this class.
#
##########################################################################################

import numpy as np
import pandas as pd
import os
import ray
import sklearn as skl
import copy as cp
from abc import ABC, abstractmethod

from typeguard import typechecked
from typing import List, Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from openml import tasks

from Source.Base.individual import Individual
from Source.Base import model_param_space


@typechecked
class BaseEA(ABC):
    """
    Abstract base class for evolutionary algorithm optimization.
    Provides common functionality for data loading, population management, and evaluation.
    """

    def __init__(self,
                 seed: int,
                 pop_size: int,
                 cores: int,
                 mut_prob: float,
                 mut_var: float) -> None:
        """
        Initializes the base EA class with common parameters.

        Args:
            seed (int): Random seed for reproducibility.
            pop_size (int): Population size for the evolutionary algorithm.
            cores (int): Number of CPU cores to use for parallel processing.
            mut_prob (float): Mutation probability for the evolutionary algorithm.
            mut_var (float): Mutation variance for the evolutionary algorithm.
            save_directory (str): Directory to save results.
        """
        # quick sanity checks
        assert seed >= 0, "Seed must be a non-negative integer."
        assert pop_size > 0, "Population size must be a positive integer."
        assert cores > 0, "Number of cores must be a positive integer."
        assert 0.0 <= mut_var, "Mutation variance must be non-negative."
        assert 0.0 <= mut_prob <= 1.0, "Mutation probability must be between 0 and 1."

        # global parameters for history tracking
        self.SOLUTION_VALIDATION_SCORE = 'SOLUTION_VALIDATION_SCORE'
        self.SOLUTION_TRAIN_SCORE = 'SOLUTION_TRAIN_SCORE'

        # save the parameters
        self.seed = seed
        self.pop_size = pop_size
        self.cores = cores
        self.rng = np.random.default_rng(seed)

        # ea specific variables
        self.population: List[Individual] = []
        self.mut_prob = mut_prob
        self.mut_var = mut_var

        # variables tracked during the optimization process
        self.total_evaluations = 0

        # data-related attributes (set by load_data)
        self.task_id: Optional[int] = None
        self.binary_classification: Optional[bool] = None
        self.labels: Optional[np.ndarray] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.cv_splits: Optional[List[Tuple]] = None
        self.categorical_cols: Optional[List[str]] = None
        self.numerical_cols: Optional[List[str]] = None

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

        # prepare CV fold data
        self._prepare_cv_folds()

        return

    def _prepare_cv_folds(self) -> None:
        """
        Prepares cross-validation fold data with preprocessing and stores in Ray object store.
        """
        categorical_cols = self.categorical_cols
        numerical_cols = self.numerical_cols

        # generate each fold's train and validation sets with preprocessing
        for fold_idx in range(5):
            X_train_fold_raw = self.X_train.iloc[self.cv_splits[fold_idx][0]].reset_index(drop=True)
            X_val_fold_raw = self.X_train.iloc[self.cv_splits[fold_idx][1]].reset_index(drop=True)

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'), categorical_cols)
                ],
                remainder='passthrough'
            )

            X_train_transformed = ray.put(preprocessor.fit_transform(X_train_fold_raw))
            X_val_transformed = ray.put(preprocessor.transform(X_val_fold_raw))
            y_train_fold = ray.put(self.y_train[self.cv_splits[fold_idx][0]])
            y_val_fold = ray.put(self.y_train[self.cv_splits[fold_idx][1]])

            # store as instance attributes dynamically
            setattr(self, f'X_train_f{fold_idx+1}_rf', X_train_transformed)
            setattr(self, f'X_val_f{fold_idx+1}_rf', X_val_transformed)
            setattr(self, f'y_train_f{fold_idx+1}_rf', y_train_fold)
            setattr(self, f'y_val_f{fold_idx+1}_rf', y_val_fold)

        return

    def get_cv_splits(self) -> List[Tuple]:
        """
        Returns the list of CV split data tuples for evaluation.

        Returns:
            List[Tuple]: List of tuples containing (X_train, y_train, X_val, y_val) Ray references.
        """
        return [
            (getattr(self, f'X_train_f{i}_rf'), getattr(self, f'y_train_f{i}_rf'),
             getattr(self, f'X_val_f{i}_rf'), getattr(self, f'y_val_f{i}_rf'))
            for i in range(1, 6)
        ]

    def mutate(self, individual: Individual) -> Individual:
            """
            Mutates the given individual by randomly altering its hyperparameters based on the mutation probability.

            Args:
                individual (Individual): The individual to be mutated.

            Returns:
                Individual: A new individual with mutated hyperparameters.
            """
            model_type = individual.model_type
            params = individual.get_params()

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
            return Individual(params=cp.deepcopy(mutated_params), model_type=cp.deepcopy(model_type))

    def model_test_evaluation(self, model_type: str, model_params: Dict[str, Any]) -> Tuple[float, float]:
        """
        Evaluates a model on the test dataset after fitting on the full training set.

        Args:
            model_type (str): The type of model to evaluate. Must be one of ['RF', 'KSVC', 'GB', 'KNN', 'MLP'].
            model_params (Dict[str, Any]): Dictionary of hyperparameters for the model.

        Returns:
            Tuple[float, float]: A tuple containing (train_score, test_score).
        """
        assert self.X_train is not None and self.X_test is not None, "Data must be loaded before evaluation."
        assert self.y_train is not None and self.y_test is not None, "Data must be loaded before evaluation."
        assert self.binary_classification is not None, "Data must be loaded before evaluation."

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'), self.categorical_cols)
            ],
            remainder='passthrough'
        )

        X_train_preprocessed = preprocessor.fit_transform(self.X_train)
        X_test_preprocessed = preprocessor.transform(self.X_test)

        if model_type == 'RF':
            eval_params = model_param_space.RandomForestParams().eval_parameters(model_params)
            model = RandomForestClassifier(**eval_params, random_state=self.seed)
        elif model_type == 'KSVC':
            eval_params = model_param_space.KernelSVCParams().eval_parameters(model_params)
            model = SVC(**eval_params, random_state=self.seed, probability=True)
        elif model_type == 'GB':
            eval_params = model_param_space.GradientBoostParams(binary_class=self.binary_classification).eval_parameters(model_params)
            model = GradientBoostingClassifier(**eval_params, random_state=self.seed)
        elif model_type == 'KNN':
            eval_params = model_param_space.KNeighborsClassifierParams().eval_parameters(model_params)
            model = KNeighborsClassifier(**eval_params)
        elif model_type == 'MLP':
            eval_params = model_param_space.MLPClassifierParams().eval_parameters(model_params)
            layers = (eval_params['layer_1'],
                      eval_params['layer_2'],
                      eval_params['layer_3'],
                      eval_params['layer_4'],
                      eval_params['layer_5'])
            model = MLPClassifier(hidden_layer_sizes=layers,
                                  activation=eval_params['activation'],
                                  solver=eval_params['solver'],
                                  max_iter=eval_params['max_iter'],
                                  random_state=self.seed)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Must be one of ['RF', 'KSVC', 'GB', 'KNN', 'MLP']")

        model.fit(X_train_preprocessed, self.y_train)

        train_pred_proba = model.predict_proba(X_train_preprocessed)
        test_pred_proba = model.predict_proba(X_test_preprocessed)

        if self.binary_classification:
            train_score = float(roc_auc_score(self.y_train, train_pred_proba[:, 1]))
            test_score = float(roc_auc_score(self.y_test, test_pred_proba[:, 1]))
        else:
            train_score = float(roc_auc_score(self.y_train, train_pred_proba, multi_class='ovo', labels=self.labels))
            test_score = float(roc_auc_score(self.y_test, test_pred_proba, multi_class='ovo', labels=self.labels))

        return train_score, test_score

    @abstractmethod
    def evolve(self, gens: int, ucb: bool, pi: bool, ei: bool) -> None:
        """
        Evolves the population over a given number of generations.
        Must be implemented by derived classes.

        Args:
            gens (int): Number of generations to evolve.
            ucb (bool): Whether to use Upper Confidence Bound for selection.
            pi (bool): Whether to use Probability of Improvement for selection.
            ei (bool): Whether to use Expected Improvement for selection.
        """
        pass

    @abstractmethod
    def initialize_population(self) -> None:
        """
        Initializes the starting population for the evolutionary algorithm.
        Must be implemented by derived classes.
        """
        pass

    @abstractmethod
    def evaluation(self, *args, **kwargs) -> Any:
        """
        Evaluates the performance of individuals.
        Must be implemented by derived classes.
        """
        pass

    @abstractmethod
    def save_results(self, save_dir: str) -> None:
        """
        Saves the results of the optimization process.
        Must be implemented by derived classes.
        """
        pass
