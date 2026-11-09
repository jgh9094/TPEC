##########################################################################################
#
# Abstract base class for evolutionary algorithm (EA) optimization of hyperparameters.
# All EA implementations (CASH, HPO) are derived from this class.
#
##########################################################################################

import csv
import os
import time
import numpy as np
import pandas as pd
import ray
import sklearn as skl
from abc import ABC, abstractmethod

from typeguard import typechecked
from typing import Dict, List, Any, Optional, Tuple
from sklearn.compose import ColumnTransformer

from Source.Base.individual import Individual

# Columns of checkpoints.csv that every EA writes (see BaseEA._checkpoint). In this order.
# CASH appends its TPE-selection columns (see CASH EA.checkpoint_columns).
CHECKPOINT_COLUMNS = [
    "generation", "evaluations", "hard_evaluations", "best_id", "best_eval", "best_generation",
    "train_performance", "validation_performance", "test_performance", "complexity",
    "refit", "refit_error", "search_minutes",
]


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
                 crossover_prob: float,
                 classification: bool,
                 checkpoint_every_generation: bool = True) -> None:
        """
        Initializes the base EA class with common parameters.

        Args:
            seed (int): Random seed for reproducibility.
            pop_size (int): Population size for the evolutionary algorithm.
            cores (int): Number of CPU cores to use for parallel processing.
            mut_prob (float): Per-parameter mutation probability for the mutation operator.
            crossover_prob (float): Probability that an offspring is generated via crossover;
                the remaining ``1.0 - crossover_prob`` are generated via mutation. Defaults to
                0.0 (all mutation, no crossover).
            classification (bool): Whether the task is classification (True) or regression
                (False). Controls how ``load_data_pd`` splits the data: classification derives
                class ``labels`` and uses a stratified train/test split + StratifiedKFold;
                regression has no classes, so it uses a plain shuffled split + KFold and leaves
                ``labels``/``binary_classification`` as None. For regression the target ``y`` is
                used as-is -- callers must pre-scale/transform it as needed (some estimators, e.g.
                an MLPRegressor with solver='sgd', diverge on large-magnitude unscaled targets).
            checkpoint_every_generation (bool): If True, after the initial population and after
                every generation, refit the best-so-far model and record its train/validation/test
                performance in ``checkpoints.csv`` (see ``_checkpoint``). The search never reads
                these scores, so turning checkpoints on or off does not change the search.
        """
        # quick sanity checks
        assert seed >= 0, "Seed must be a non-negative integer."
        assert pop_size > 0, "Population size must be a positive integer."
        assert cores > 0, "Number of cores must be a positive integer."
        assert 0.0 <= mut_prob <= 1.0, "Mutation probability must be between 0 and 1."
        assert 0.0 <= crossover_prob <= 1.0, "Crossover probability must be between 0 and 1."

        # save the parameters
        self.seed = seed
        self.pop_size = pop_size
        self.cores = cores
        self.classification = classification
        # human-readable name of the maximized metric, for task-aware logging (AUC vs R^2)
        self.metric_name = "AUC" if classification else "R2"
        self.rng = np.random.default_rng(seed)

        # ea specific variables
        self.population: List[Individual] = []
        self.mut_prob = mut_prob
        # probability an offspring is produced by crossover vs. mutation
        self.crossover_prob = crossover_prob

        # variables tracked during the optimization process
        self.total_evaluations = 0

        # data-related attributes (set by load_data)
        self.binary_classification: Optional[bool] = None
        self.labels: Optional[np.ndarray] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.cv_splits: Optional[List[Tuple]] = None
        self.categorical_cols: Optional[List[str]] = None
        self.numerical_cols: Optional[List[str]] = None
        # number of cross-validation folds; unset until load_data_pd assigns it (defaults to 5
        # there). Kept None here so load_data_pd can assert it has not already been set.
        self.n_folds: Optional[int] = None

        # per-generation test checkpoints (see _checkpoint). Reporting only: the search never reads them.
        self.checkpoint_every_generation = checkpoint_every_generation
        self.checkpoints: List[Dict[str, Any]] = []
        # wall-clock minutes spent inside checkpoints; excluded from total_time_minutes
        self.checkpoint_minutes = 0.0
        # when evolve() started, so search_minutes can be measured net of checkpoint time
        self._search_clock_start: Optional[float] = None
        # best entry id last refit, and its cached (train, test, complexity, refit_error)
        self._last_checkpoint_entry_id: Optional[int] = None
        self._last_checkpoint_scores: Optional[Tuple[float, float, int, str]] = None

        return

    def load_data_pd(self,
                     data: pd.DataFrame,
                     target_label: str,
                     train_p: float,
                     one_hot_cols: Optional[List[str]] = None,
                     scalar_cols: Optional[List[str]] = None,
                     n_folds: int = 5) -> None:
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
            n_folds (int): Number of cross-validation folds to build over the training set
                (StratifiedKFold for classification, KFold for regression). Defaults to 5.
        """
        # normalize the column lists so empty/None are treated the same
        one_hot_cols = list(one_hot_cols) if one_hot_cols else []
        scalar_cols = list(scalar_cols) if scalar_cols else []

        # quick sanity checks
        assert 0.0 < train_p < 1.0, "train_p must be between 0 and 1 (exclusive)."
        assert n_folds >= 2, "n_folds must be at least 2 for cross-validation."
        # load_data_pd wires up the (immutable) CV structure, so it must run exactly once: n_folds
        # is None until set here, and a second call would silently rebuild the folds.
        assert self.n_folds is None, "n_folds has already been set; load_data_pd must be called only once."
        self.n_folds = n_folds
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

        # Determine problem type. Classification derives its class labels (used to pick the legal
        # multi-class ROC AUC labels and to stratify the splits); regression has no classes, so
        # labels/binary_classification stay None and the target is treated as continuous.
        if self.classification:
            self.labels = np.unique(y)
            self.binary_classification = bool(len(self.labels) == 2)
            print(f"Classification task. Unique classes in target variable '{target_label}': {self.labels}")
        else:
            self.labels = None
            self.binary_classification = None
            y_numeric = np.asarray(y, dtype=float)
            print(f"Regression task. Target variable '{target_label}' treated as continuous "
                  f"(min={y_numeric.min():.4g}, max={y_numeric.max():.4g}). "
                  f"NOTE: the target is used as-is -- pre-scale/transform it if your model needs it.")

        # Generate an initial train-test split (train_p is the proportion used for training).
        # Classification stratifies on the class labels; regression cannot stratify a continuous
        # target, so it uses a plain shuffled split.
        stratify = y if self.classification else None
        self.X_train, self.X_test, self.y_train, self.y_test = skl.model_selection.train_test_split(
            X, y, train_size=train_p, random_state=self.seed, shuffle=True, stratify=stratify
        )

        # Generate an ``n_folds``-fold CV split over the training set (indices per fold).
        # StratifiedKFold for classification (balanced class proportions per fold); plain KFold
        # for regression.
        if self.classification:
            splitter = skl.model_selection.StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        else:
            splitter = skl.model_selection.KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        self.cv_splits = list(splitter.split(self.X_train, self.y_train))

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
        assert self.n_folds is not None, "n_folds must be set (via load_data_pd) before preparing folds."
        # generate each fold's train and validation sets with preprocessing
        cv_fold_refs = []
        for fold_idx in range(self.n_folds):
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

    @abstractmethod
    def _build_preprocessor(self) -> ColumnTransformer:
        """
        Builds the base ColumnTransformer applied to every fold's training partition (and the
        final train/test split) BEFORE the evolved model/pipeline sees the data. Must be
        implemented by derived classes, since the appropriate baseline preprocessing differs by
        EA setup:

          * HPO fits a single bare estimator, so the preprocessor must fully numericize AND scale
            the data (StandardScaler on the numeric columns + one-hot on the categoricals).
          * CASH evolves its own ``feature_scaling`` node, so its preprocessor only needs to make
            the data numeric (one-hot the categoricals); scaling is left to the evolved pipeline.

        Returns:
            ColumnTransformer: The configured preprocessor.
        """
        pass

    def smallest_cv_train_size(self) -> int:
        """
        Returns the number of rows in the smallest cross-validation training fold.

        Parameter bounds that scale with the number of training rows (e.g. QuantileTransformer's
        ``n_quantiles`` or Nystroem's ``n_components``) should be capped by what a model actually
        sees while being cross-validated, not by the full training-set size ``X_train``. Since each
        model is fit on ``k - 1`` folds during CV, the honest cap is the smallest such training
        partition; sizing the search space to it means proposed values are always realizable on
        every fold (no scikit-learn clip warnings, no wasted components).

        Requires ``load_data_pd`` to have been called (``cv_splits`` populated).

        Returns:
            int: The minimum training-partition size across the CV folds.
        """
        assert self.cv_splits is not None, "load_data_pd must be called before smallest_cv_train_size."
        return int(min(len(train_idx) for train_idx, _ in self.cv_splits))

    def get_cv_splits(self) -> List[Tuple]:
        """
        Returns the per-fold CV data as a list of Ray object reference tuples.

        Returns:
            List[Tuple]: One (X_train, y_train, X_val, y_val) tuple of Ray ObjectRefs
            per fold, so that each fold can be dispatched to its own evaluation task.
        """
        return self.cv_splits_ref

    def variation_order(self, offspring_cnt: int, crossover_prob: float) -> Tuple[List[str], int]:
        """Decide, per offspring, whether it comes from mutation ('m') or crossover ('c').

        Each offspring is independently assigned an operator using ``crossover_prob`` as the
        probability of crossover (and ``1.0 - crossover_prob`` for mutation). Returns the
        operator sequence and the total number of parents it consumes (mutation draws 1
        parent, crossover draws 2).

        Args:
            offspring_cnt (int): Number of offspring to generate.
            crossover_prob (float): Probability that a given offspring is generated via
                crossover; the remaining ``1.0 - crossover_prob`` are generated via mutation.

        Returns:
            Tuple[List[str], int]: The per-offspring operator sequence (each 'm' or 'c')
            and the total number of parents required to realize it.
        """
        assert 0.0 <= crossover_prob <= 1.0, "Crossover probability must be between 0 and 1."
        parent_count = {'m': 1, 'c': 2}

        order = self.rng.choice(['m', 'c'], offspring_cnt, p=[1.0 - crossover_prob, crossover_prob])
        order = [str(op) for op in order]

        assert len(order) == offspring_cnt
        return order, int(sum(parent_count[op] for op in order))

    @abstractmethod
    def crossover(self, parent_a: Individual, parent_b: Individual) -> Individual:
        """
        Recombines two parents into a single offspring individual. Must be implemented by
        derived classes, since the genotype representation (and thus how parameter values
        are combined) differs across EA setups (e.g., single-model HPO vs. CASH).

        Args:
            parent_a (Individual): The first parent.
            parent_b (Individual): The second parent.

        Returns:
            Individual: A new offspring individual derived from both parents.
        """
        pass

    @abstractmethod
    def mutate(self, parent: Individual) -> Individual:
        """
        Mutates a parent individual to produce a new offspring individual. Must be implemented
        by derived classes, since the genotype representation (and thus how parameter values
        are mutated) differs across EA setups (e.g., single-model HPO vs. CASH).

        Args:
            parent (Individual): The parent individual to mutate.

        Returns:
            Individual: A new offspring individual derived from the parent.
        """
        pass

    @abstractmethod
    def model_test_evaluation(self, individual: Individual) -> Tuple[float, float, int]:
        """
        Fits an individual's model on the full training set and evaluates it on the held-out
        test set. Must be implemented by derived classes, since the way an individual maps to a
        concrete model differs across EA setups (e.g., single-model HPO vs. CASH).

        Args:
            individual (Individual): The individual to evaluate.

        Returns:
            Tuple[float, float, int]: (train_score, test_score, complexity), where complexity is
            the fitted model's ``calculate_model_complexity`` score (Source/ML/complexity.py).
        """
        pass

    @abstractmethod
    def _select_best(self) -> Tuple[Individual, float]:
        """
        Return the held best-so-far individual (rebuilt from the archive entry ``best_entry``) and
        its validation score. Makes no random draws, so reading it never changes the search.
        Must be implemented by derived classes.
        """
        pass

    def checkpoint_columns(self) -> List[str]:
        """
        Column order of ``checkpoints.csv``. Derived classes append their own columns.

        Returns:
            List[str]: The column names, in file order.
        """
        return list(CHECKPOINT_COLUMNS)

    def _checkpoint_extra_fields(self, generation: int) -> Dict[str, Any]:
        """
        Hook for derived classes: extra fields for this generation's checkpoint row. Default: none.

        Args:
            generation (int): Generation just completed (-1 for the initial population).

        Returns:
            Dict[str, Any]: Extra column values, keyed by the names from ``checkpoint_columns``.
        """
        return {}

    def _checkpoint(self, generation: int) -> None:
        """
        Record the best-so-far model's train, validation and test performance as one checkpoint row.

        Why: a single run then yields every smaller evaluation budget. Each row is the search state at
        the end of ``generation`` (-1 = initial population), so a shorter run is a prefix of a longer one.

        Test scores are REPORTING ONLY. Nothing inside the search may read ``self.checkpoints`` or the
        cached scores below. The search reads the archive and the TPE, never test data.

        The best model is refit on the full training set only when the best entry changed since the
        last checkpoint; otherwise the cached scores are reused. A refit failure is recorded in the row
        (NaN scores plus ``refit_error``) and is never raised, so a checkpoint cannot end the run. The
        failure is cached for that best entry rather than retried every generation; the end of
        ``evolve`` retries it once.

        Checkpoint time is measured and excluded from ``total_time_minutes`` and from ``search_minutes``,
        so runtimes stay comparable with runs made before checkpointing existed. Makes no random draws.

        Args:
            generation (int): Generation just completed (-1 for the initial population).
        """
        assert self._search_clock_start is not None, "evolve() must start the search clock before checkpointing."
        checkpoint_start = time.time()

        # Nothing to report until some pipeline has a finite validation score. Only possible for
        # regression, when every initial individual errored (R^2 = -inf). Skip the row rather than fail.
        if self.best_entry is None:
            print(f"Checkpoint (Gen {generation}) skipped: no best entry yet.", flush=True)
            self.checkpoint_minutes += (time.time() - checkpoint_start) / 60
            return

        best_individual, best_val = self._select_best()
        best_id = self.best_entry.id
        search_minutes = (checkpoint_start - self._search_clock_start) / 60 - self.checkpoint_minutes

        refit = best_id != self._last_checkpoint_entry_id
        if refit:
            try:
                train_score, test_score, complexity = self.model_test_evaluation(best_individual)
                error = ""
            except Exception as exc:  # a checkpoint must never end the search
                error = f"{type(exc).__name__}: {exc}"
                print(f"WARNING: checkpoint refit failed at generation {generation}: {error}", flush=True)
                train_score, test_score, complexity = float("nan"), float("nan"), -1
            self._last_checkpoint_entry_id = best_id
            self._last_checkpoint_scores = (train_score, test_score, complexity, error)
        else:
            train_score, test_score, complexity, error = self._last_checkpoint_scores

        row: Dict[str, Any] = {
            "generation": generation,
            "evaluations": len(self.eval_archive),
            "hard_evaluations": self.hard_eval_count,
            "best_id": best_id,
            "best_eval": best_id + 1,  # same convention as best_results.json
            "best_generation": self.best_entry.generation,
            "train_performance": train_score,
            "validation_performance": best_val,
            "test_performance": test_score,
            "complexity": complexity,
            "refit": refit,
            "refit_error": error,
            "search_minutes": search_minutes,
        }
        row.update(self._checkpoint_extra_fields(generation))
        self.checkpoints.append(row)

        # charge this checkpoint's own time to checkpoint_minutes, so total runtime excludes it
        self.checkpoint_minutes += (time.time() - checkpoint_start) / 60

    def _final_scores_from_checkpoint(self) -> Optional[Tuple[float, float, int]]:
        """
        Train, test and complexity of the held best, taken from the last checkpoint row, so the end
        of ``evolve`` does not refit the same model a second time.

        Returns None when the caller must refit (the old path): checkpointing is off, there is no row,
        the best entry has moved on since the row, or the row's refit failed (the end of the run retries it,
        so a real failure still raises here as it always has).

        Returns:
            Optional[Tuple[float, float, int]]: (train, test, complexity), or None.
        """
        if not self.checkpoint_every_generation or not self.checkpoints or self.best_entry is None:
            return None
        last = self.checkpoints[-1]
        if last["refit_error"] or last["best_id"] != self.best_entry.id:
            return None
        return float(last["train_performance"]), float(last["test_performance"]), int(last["complexity"])

    def save_checkpoints(self, save_dir: str) -> None:
        """
        Write ``checkpoints.csv`` (one row per checkpoint, columns in ``checkpoint_columns()`` order)
        and ``checkpoint_runtime_minutes.txt``. Writes nothing if no checkpoint was recorded.

        Args:
            save_dir (str): Directory to write the files into.
        """
        if not self.checkpoints:
            return

        os.makedirs(save_dir, exist_ok=True)
        columns = self.checkpoint_columns()
        csv_path = os.path.join(save_dir, "checkpoints.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.checkpoints)
        print(f"Checkpoints ({len(self.checkpoints)} rows) saved to: {csv_path}", flush=True)

        runtime_path = os.path.join(save_dir, "checkpoint_runtime_minutes.txt")
        with open(runtime_path, "w") as f:
            f.write(f"{self.checkpoint_minutes}")

        return

    @abstractmethod
    def evolve(self, gens: int) -> None:
        """
        Evolves the population over a given number of generations.
        Must be implemented by derived classes.

        Args:
            gens (int): Number of generations to evolve.
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
