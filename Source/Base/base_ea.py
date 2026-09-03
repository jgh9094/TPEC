##########################################################################################
#
# Abstract base class for evolutionary algorithm (EA) optimization of hyperparameters.
# All EA implementations (CASH, HPO) should derive from this class.
#
##########################################################################################

import numpy as np
import pandas as pd
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

    def load_data_pd(self,
                     data: pd.DataFrame,
                     target_label: str,
                     train_p: float,
                     one_hot_cols: Optional[List[str]] = None,
                     scalar_cols: Optional[List[str]] = None) -> None:
        """
        Loads a dataset from an in-memory pandas DataFrame and applies preprocessing.

        The DataFrame must contain all the X features and the Y label to predict.
        Columns listed in `one_hot_cols` are one-hot-encoded and columns listed in
        `scalar_cols` are transformed via StandardScaler. If a list is empty (or None)
        the corresponding transformation is skipped; any columns not listed in either
        list are passed through unchanged.

        Args:
            data (pd.DataFrame): DataFrame containing all X features and the Y label.
            target_label (str): Name of the column to predict (the Y label).
            train_p (float): Proportion of the dataset to use for training.
            one_hot_cols (Optional[List[str]]): Feature columns to one-hot-encode.
            scalar_cols (Optional[List[str]]): Feature columns to scale via StandardScaler.
        """
        # normalize the column lists so empty/None are treated the same
        one_hot_cols = list(one_hot_cols) if one_hot_cols else []
        scalar_cols = list(scalar_cols) if scalar_cols else []

        # quick sanity checks
        assert 0.0 < train_p < 1.0, "train_p must be between 0 and 1 (exclusive)."
        assert target_label in data.columns, f"Target label '{target_label}' not found in DataFrame columns."

        feature_cols = [col for col in data.columns if col != target_label]
        for col in one_hot_cols:
            assert col in feature_cols, f"One-hot column '{col}' is not a feature column in the DataFrame."
        for col in scalar_cols:
            assert col in feature_cols, f"Scale column '{col}' is not a feature column in the DataFrame."
        overlap = set(one_hot_cols) & set(scalar_cols)
        assert not overlap, f"Columns cannot be both one-hot-encoded and scaled: {sorted(overlap)}."

        # store the dataset and the columns to transform for later use
        self.data_set = data.reset_index(drop=True)
        self.categorical_cols = one_hot_cols
        self.numerical_cols = scalar_cols

        # split the dataset into features and target
        X = self.data_set.drop(columns=[target_label])
        y = self.data_set[target_label].values

        # print dimesionality of X
        print(f"Loaded dataset with {X.shape[0]} rows and {X.shape[1]} features.")

        # make a list of all unique classes in the target variable and determine problem type
        self.labels = np.unique(y)
        self.binary_classification = bool(len(self.labels) == 2)

        # print the unique classes
        print(f"Unique classes in target variable '{target_label}': {self.labels}")

        # generate an initial train-test split for the evolutionary algorithm (train_p is the proportion of the dataset to use for training)
        self.X_train, self.X_test, self.y_train, self.y_test = skl.model_selection.train_test_split(
            X, y, train_size=train_p, random_state=self.seed, shuffle=True, stratify=y
        )

        # generate a 5-fold cross-validation split for the evolutionary algorithm based on the training set and store the indices for each fold
        self.cv_splits = list(skl.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed).split(self.X_train, self.y_train))

        # prepare CV fold data
        self._prepare_cv_folds()

        return

    def _prepare_cv_folds(self) -> None:
        """
        Prepares cross-validation fold data with preprocessing and stores each fold
        independently in the Ray object store.

        Every fold is preprocessed on its own: the preprocessor is fit on the fold's
        training partition and then used to transform both the training and validation
        partitions, which prevents data leakage. The four resulting arrays for each fold
        (X_train, y_train, X_val, y_val) are placed in the Ray object store individually
        so that a single fold can be loaded by a dedicated per-fold evaluation task.
        """
        # generate each fold's train and validation sets with preprocessing
        cv_fold_refs = []
        for fold_idx in range(5):
            X_train_fold_raw = self.X_train.iloc[self.cv_splits[fold_idx][0]].reset_index(drop=True)
            X_val_fold_raw = self.X_train.iloc[self.cv_splits[fold_idx][1]].reset_index(drop=True)

            preprocessor = self._build_preprocessor()

            # fit on the training partition only, then transform both (no leakage)
            X_train_transformed = preprocessor.fit_transform(X_train_fold_raw)
            X_val_transformed = preprocessor.transform(X_val_fold_raw)
            y_train_fold = self.y_train[self.cv_splits[fold_idx][0]]
            y_val_fold = self.y_train[self.cv_splits[fold_idx][1]]

            # place each fold's data in the Ray object store as separate references
            cv_fold_refs.append((
                ray.put(X_train_transformed),
                ray.put(y_train_fold),
                ray.put(X_val_transformed),
                ray.put(y_val_fold),
            ))

        # store the per-fold (X_train, y_train, X_val, y_val) Ray object references
        self.cv_splits_ref = cv_fold_refs

        return

    def _build_preprocessor(self) -> ColumnTransformer:
            """
            Builds a ColumnTransformer that scales the numerical columns and one-hot-encodes
            the categorical columns. Transformations whose column list is empty are omitted so
            that only the requested operations are applied; all remaining columns pass through.

            Returns:
                ColumnTransformer: The configured preprocessor.
            """
            transformers = []
            if self.numerical_cols:
                transformers.append(('num', StandardScaler(), self.numerical_cols))
            if self.categorical_cols:
                transformers.append(('cat', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'), self.categorical_cols))

            return ColumnTransformer(transformers=transformers, remainder='passthrough')

    def get_cv_splits(self) -> List[Tuple]:
        """
        Returns the per-fold CV data as a list of Ray object reference tuples.

        Returns:
            List[Tuple]: One (X_train, y_train, X_val, y_val) tuple of Ray ObjectRefs
            per fold, so that each fold can be dispatched to its own evaluation task.
        """
        return self.cv_splits_ref

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

        preprocessor = self._build_preprocessor()

        X_train_preprocessed = preprocessor.fit_transform(self.X_train)
        X_test_preprocessed = preprocessor.transform(self.X_test)

        if model_type == 'RF':
            eval_params = model_param_space.RandomForestParams().eval_parameters(model_params)
            model = RandomForestClassifier(**eval_params, random_state=self.seed, n_jobs=self.cores)
        elif model_type == 'KSVC':
            eval_params = model_param_space.KernelSVCParams().eval_parameters(model_params)
            model = SVC(**eval_params, random_state=self.seed, probability=True)
        elif model_type == 'GB':
            eval_params = model_param_space.GradientBoostParams(binary_class=self.binary_classification).eval_parameters(model_params)
            model = GradientBoostingClassifier(**eval_params, random_state=self.seed)
        elif model_type == 'KNN':
            eval_params = model_param_space.KNeighborsClassifierParams().eval_parameters(model_params)
            model = KNeighborsClassifier(**eval_params, n_jobs=self.cores)
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
