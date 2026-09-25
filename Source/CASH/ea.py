##########################################################################################
#
# CASH (Combined Algorithm Selection and Hyperparameter optimization) EA 4 AutoML.
#
# Evolves a full scikit-learn pipeline:
# feature scaling -> feature engineering -> feature selection -> predictor.
#
# Each of the four nodes is an evolved decision (which component, plus its hyperparameters).
# EA executes the following workflow:
# - tournament selection
# - per-offspring mutation/crossover with/out TPE-guided variation
# - TPE-guided candidate ranking against an evaluated archive.
#
# Things to note:
#   * the genotype is a nested pipeline candidate (see Source/CASH/individual.py)
#   * variation acts on both the component CHOICE at a node (structural) and that
#     component's parameters (parametric);
#   * TPE is the pipeline-aware CASH_TPE (Source/CASH/tpe.py), which models the joint
#     architecture plus architecture-conditioned parameter evidence;
#   * evaluation builds a real scikit-learn pipeline, skipping any node whose component is
#     "passthrough".
#
##########################################################################################

import numpy as np
import copy as cp
import os
import time
import json
import pandas as pd
import ray

from typeguard import typechecked
from typing import Any, Dict, List, Optional, Tuple

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import (
    MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler, Normalizer,
    KBinsDiscretizer, Binarizer, PolynomialFeatures, QuantileTransformer, PowerTransformer,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import FeatureAgglomeration
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.feature_selection import SelectFwe, SelectPercentile, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

from Source.Base.base_ea import BaseEA
from Source.Base.individual import Individual
from Source.Base.model_param_space import DataContext
from Source.CASH.individual import CASHIndividual
from Source.CASH.tpe import CASH_TPE, PipelineSpace, Candidate
from Source.CASH.archive import CASHArchive
from Source.Base.archive import (
    RANDOM_CONSTRUCTION, TPE_CONSTRUCTION,
    MUTATION_OPERATION, CROSSOVER_OPERATION, CROSSOVER_MUTATION_OPERATION,
)

from Source.ML.scaler import SCALERS
from Source.ML.transformer import TRANSFORMERS
from Source.ML.selector import SELECTORS
from Source.ML.classifiers import CLASSIFIERS
from Source.ML.regressor import REGRESSORS

# Component available at a node when its step is skipped. Represented as a literal string
# with an empty parameter space (matches the Source/ML registries' convention).
PASSTHROUGH = "passthrough"
# Ordered pipeline nodes: (node_name, ModelParams-class registry, whether "passthrough" is
# an allowed component). Order defines the pipeline execution order and the architecture-key
# order used by CASH_TPE. The predictor node is mandatory, so it does not allow passthrough.
PIPELINE_NODES = (
    ("feature_scaling", SCALERS, True),
    ("feature_engineering", TRANSFORMERS, True),
    ("feature_selection", SELECTORS, True),
    ("predictor", CLASSIFIERS, False),
)


# Feature-preprocessing components (scaling / engineering / selection) shared by both task
# types. Maps each component's identifier (its ModelParams.get_model_type()) to the scikit-learn
# estimator class it builds. Keys must stay in sync with the get_model_type() strings of the
# Source/ML registries.
PREPROCESSING_CLASSES = {
    # feature scaling
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
    "StandardScaler": StandardScaler,
    "MaxAbsScaler": MaxAbsScaler,
    "Normalizer": Normalizer,
    # feature engineering
    "KBinsDiscretizer": KBinsDiscretizer,
    "Binarizer": Binarizer,
    "PolynomialFeatures": PolynomialFeatures,
    "PCA": PCA,
    "FastICA": FastICA,
    "FeatureAgglomeration": FeatureAgglomeration,
    "Nystroem": Nystroem,
    "RBFSampler": RBFSampler,
    "QuantileTransformer": QuantileTransformer,
    "PowerTransformer": PowerTransformer,
    # feature selection
    "SelectFwe": SelectFwe,
    "SelectPercentile": SelectPercentile,
    "VarianceThreshold": VarianceThreshold,
}

# Predictor-node component -> scikit-learn estimator, one table per task type. The preprocessing
# components are shared; only the predictor family differs (classifiers vs. regressors). Note the
# kernel-SVM identifier differs by task -- "KSVC" (classification) vs. "SVR" (regression) -- which
# is why the two share no predictor keys beyond the tree/boosting/KNN/MLP families.
COMPONENT_CLASSES = {
    **PREPROCESSING_CLASSES,
    "RF": RandomForestClassifier,
    "ET": ExtraTreesClassifier,
    "KSVC": SVC,
    "GB": GradientBoostingClassifier,
    "KNN": KNeighborsClassifier,
    "MLP": MLPClassifier,
}
COMPONENT_CLASSES_REGRESSION = {
    **PREPROCESSING_CLASSES,
    "RF": RandomForestRegressor,
    "ET": ExtraTreesRegressor,
    "SVR": SVR,
    "GB": GradientBoostingRegressor,
    "KNN": KNeighborsRegressor,
    "MLP": MLPRegressor,
}


def build_estimator(component_name: str, eval_kwargs: Dict[str, Any], classification: bool):
    """
    Instantiate the scikit-learn estimator for a single pipeline node from its component name
    and the kwargs produced by that component's ``eval_parameters``.

    The predictor family is chosen by ``classification`` (classifier vs. regressor table); the
    shared preprocessing components resolve identically in either table. A few components need
    light post-processing that their ``eval_parameters`` leaves for the construction site
    (mirroring how Source/HPO/cv_evaluation.py builds the HPO estimators):
      * ``MLP`` -- the per-layer genes ``layer_1..layer_5`` are recombined into the single
        ``hidden_layer_sizes`` tuple scikit-learn expects (MLPClassifier or MLPRegressor).
      * ``KSVC`` -- ``probability=True`` is required so the classification pipeline predictor can
        expose ``predict_proba`` for ROC-AUC scoring.
      * ``SVR`` -- the regression kernel-SVM predictor, built directly from its kwargs.

    Args:
        component_name (str): The component identifier (its ModelParams.get_model_type()).
        eval_kwargs (Dict[str, Any]): The scikit-learn kwargs from ``eval_parameters``.
        classification (bool): Whether this pipeline is a classification (True) or regression
            (False) task; selects the predictor estimator table.

    Returns:
        A ready-to-fit scikit-learn estimator instance.
    """
    kwargs = dict(eval_kwargs)
    if component_name == "MLP":
        layers = tuple(kwargs.pop(f"layer_{i}") for i in range(1, 6))
        mlp_cls = MLPClassifier if classification else MLPRegressor
        return mlp_cls(hidden_layer_sizes=layers, **kwargs)
    if component_name == "KSVC":
        return SVC(**kwargs, probability=True)
    if component_name == "SVR":
        return SVR(**kwargs)
    table = COMPONENT_CLASSES if classification else COMPONENT_CLASSES_REGRESSION
    return table[component_name](**kwargs)


# Node whose evolved scaler must respect the numeric/categorical split (School-B scaling).
SCALING_NODE = PIPELINE_NODES[0][0]  # "feature_scaling"
# Predictor node -- the mandatory final step. Its component registry is task-dependent
# (CLASSIFIERS for classification, REGRESSORS for regression); see _build_pipeline_space.
PREDICTOR_NODE = PIPELINE_NODES[-1][0]  # "predictor"


def assemble_steps(
    pipeline_steps: List[Tuple[str, str, Dict[str, Any]]],
    scale_cols: Optional[List[int]],
    classification: bool,
) -> List[Tuple[str, Any]]:
    """
    Turn a candidate's resolved ``(node_name, component_name, eval_kwargs)`` triples into concrete
    ``(name, estimator)`` scikit-learn Pipeline steps, dropping any node whose component is
    ``PASSTHROUGH`` (no identity step inserted).

    The ``feature_scaling`` node gets School-B handling so a scaler never distorts one-hot dummies:
      * ``scale_cols is None`` -- the base-preprocessed matrix is entirely numeric, so the evolved
        scaler is applied bare to the whole matrix.
      * ``scale_cols`` is a non-empty index list -- the matrix also holds columns that must NOT be
        scaled (one-hot dummies / unlisted passthrough columns), so the scaler is wrapped in a
        ColumnTransformer that scales ONLY those numeric indices and passes the rest through
        untouched, preserving their information.
      * ``scale_cols`` is empty -- there are no numeric columns to scale, so the scaling step is
        skipped entirely.

    Args:
        pipeline_steps: Ordered (node_name, component_name, eval_kwargs) per node.
        scale_cols: Numeric column indices the scaler may touch (see above), or None for an
            all-numeric matrix.
        classification: Whether the pipeline is a classification (True) or regression (False)
            task; forwarded to ``build_estimator`` to pick the predictor estimator table.

    Returns:
        List[Tuple[str, Any]]: The concrete (name, estimator) Pipeline steps.
    """
    steps: List[Tuple[str, Any]] = []
    for node_name, component_name, eval_kwargs in pipeline_steps:
        if component_name == PASSTHROUGH:
            continue
        estimator = build_estimator(component_name, eval_kwargs, classification)
        if node_name == SCALING_NODE and scale_cols is not None:
            if len(scale_cols) == 0:
                continue  # no numeric columns exist -> nothing for the scaler to do
            # scale only the numeric columns; leave one-hot dummies (and any other columns) as-is
            estimator = ColumnTransformer(
                [("scale", estimator, scale_cols)], remainder="passthrough"
            )
        steps.append((node_name, estimator))
    return steps


@ray.remote
def cv_pipeline_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    pipeline_steps: List[Tuple[str, str, Dict[str, Any]]],
    scale_cols: Optional[List[int]],
    id: int,
    binary_class: bool,
    labels: np.ndarray,
) -> Tuple[int, float, float, float]:
    """
    Build a scikit-learn Pipeline from a candidate's resolved genotype and evaluate one CV fold
    for a CLASSIFICATION task (scored by ROC-AUC).

    ``pipeline_steps`` is the candidate processed into an ordered list of
    ``(node_name, component_name, eval_kwargs)`` -- one entry per pipeline node, in execution
    order. As the steps are processed, any node whose component is ``"passthrough"`` is simply
    OMITTED (no identity step is inserted); every other node contributes a real estimator built
    via ``build_estimator``. The predictor node is mandatory, so the assembled pipeline always
    ends in a classifier exposing ``predict_proba``.

    Called once per (candidate, fold) pair -- i.e. 5 times per candidate under 5-fold CV -- with
    that fold's preprocessed train/validation arrays.

    Args:
        X_train, y_train: The fold's training partition (preprocessed upstream).
        X_validate, y_validate: The fold's validation partition (preprocessed upstream).
        pipeline_steps: Ordered (node_name, component_name, eval_kwargs) per node.
        scale_cols: Numeric column indices the feature_scaling scaler may touch (School-B
            scaling), or None when the matrix is entirely numeric. See ``assemble_steps``.
        id: Identifier of the candidate this fold belongs to (echoed back for accumulation).
        binary_class: True for binary classification (ROC-AUC on the positive column),
            False for multi-class (one-vs-one ROC-AUC over ``labels``).
        labels: All possible class labels (used for the multi-class ROC-AUC).

    Returns:
        Tuple[int, float, float, float]: (id, training_auc, validation_auc, status) where
        status is 1.0 on success and -1.0 if pipeline construction/fit/scoring raised.
    """
    try:
        # process the genotype into pipeline steps (omitting passthrough nodes; the scaler is
        # made numeric-column-aware when categorical dummies are present -- see assemble_steps)
        steps = assemble_steps(pipeline_steps, scale_cols, classification=True)
        model = Pipeline(steps)
        model.fit(X_train, y_train)

        if binary_class:
            train_acc = float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
            val_acc = float(roc_auc_score(y_validate, model.predict_proba(X_validate)[:, 1]))
        else:
            train_acc = float(roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovo', labels=labels))
            val_acc = float(roc_auc_score(y_validate, model.predict_proba(X_validate), multi_class='ovo', labels=labels))
        return id, train_acc, val_acc, 1.0

    except Exception as e:
        print(f"Error in cv_pipeline_classification: {e}")
        return id, 0.0, 0.0, -1.0


@ray.remote
def cv_pipeline_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    pipeline_steps: List[Tuple[str, str, Dict[str, Any]]],
    scale_cols: Optional[List[int]],
    id: int,
    binary_class: Optional[bool] = None,
    labels: Optional[np.ndarray] = None,
) -> Tuple[int, float, float, float]:
    """
    Build a scikit-learn Pipeline from a candidate's resolved genotype and evaluate one CV fold
    for a REGRESSION task (scored by R^2).

    Structurally identical to ``cv_pipeline_classification`` -- the same ``assemble_steps`` builds
    the pipeline (omitting passthrough nodes) -- but the assembled pipeline ends in a regressor and
    is scored with ``model.predict`` + ``r2_score`` instead of ``predict_proba`` + ROC-AUC. The
    target is used as-is: pre-scale/transform ``y`` upstream if the chosen predictor needs it.

    ``binary_class`` and ``labels`` are accepted only so the Ray call site can stay uniform with the
    classification task; they are unused here.

    Args:
        X_train, y_train: The fold's training partition (preprocessed upstream).
        X_validate, y_validate: The fold's validation partition (preprocessed upstream).
        pipeline_steps: Ordered (node_name, component_name, eval_kwargs) per node.
        scale_cols: Numeric column indices the feature_scaling scaler may touch (School-B
            scaling), or None when the matrix is entirely numeric. See ``assemble_steps``.
        id: Identifier of the candidate this fold belongs to (echoed back for accumulation).
        binary_class: Ignored (present for a uniform call site with the classification task).
        labels: Ignored (present for a uniform call site with the classification task).

    Returns:
        Tuple[int, float, float, float]: (id, training_r2, validation_r2, status) where
        status is 1.0 on success and -1.0 if pipeline construction/fit/scoring raised.
    """
    try:
        # process the genotype into pipeline steps (omitting passthrough nodes; the scaler is
        # made numeric-column-aware when categorical dummies are present -- see assemble_steps)
        steps = assemble_steps(pipeline_steps, scale_cols, classification=False)
        model = Pipeline(steps)
        model.fit(X_train, y_train)

        train_r2 = float(r2_score(y_train, model.predict(X_train)))
        val_r2 = float(r2_score(y_validate, model.predict(X_validate)))
        return id, train_r2, val_r2, 1.0

    except Exception as e:
        print(f"Error in cv_pipeline_regression: {e}")
        return id, 0.0, 0.0, -1.0


@typechecked
class EA(BaseEA):
    """
    CASH EA: evolves full scikit-learn pipelines with TPE-guided variation.
    Extends BaseEA with pipeline-aware (structural + parametric) mutation and crossover.
    """

    def __init__(self,
                 seed: int,
                 pop_size: int,
                 cores: int,
                 mut_prob: float,
                 mut_var: float,
                 tpe_prob: float,
                 tournament_size: int,
                 num_offspring: int,
                 crossover_prob: float,
                 scalers: Optional[List[str]] = None,
                 transformers: Optional[List[str]] = None,
                 selectors: Optional[List[str]] = None,
                 predictors: Optional[List[str]] = None,
                 component_mut_prob: float = 0.1,
                 gamma: float = 0.0,
                 classification: bool = True) -> None:
        """
        Initializes the CASH EA.

        Args:
            seed (int): Random seed for reproducibility.
            pop_size (int): Population size for the evolutionary algorithm.
            cores (int): Number of CPU cores to use for parallel evaluation.
            mut_prob (float): Per-parameter (gene-level) mutation probability.
            mut_var (float): Variance of the small local shift-mutation applied to TPE-guided
                offspring (non-TPE offspring instead resample parameters uniformly).
            tpe_prob (float): Probability an offspring is produced via TPE-guided variation.
            tournament_size (int): Tournament size for parent selection.
            num_offspring (int): Number of pseudo-offspring generated per TPE-guided step
                (the most promising is chosen by TPE).
            crossover_prob (float): Probability an offspring is produced by crossover vs. mutation.
            scalers (Optional[List[str]]): Component identifiers (``get_model_type()`` names) the
                evolution may use at the feature_scaling node. An empty list or None means "use the
                full registered set" for that step. Lets a user restrict which models each pipeline
                stage may draw from. Unknown names raise at pipeline-space construction.
            transformers (Optional[List[str]]): Allowed components at the feature_engineering node
                (empty/None -> full set). See ``scalers``.
            selectors (Optional[List[str]]): Allowed components at the feature_selection node
                (empty/None -> full set). See ``scalers``.
            predictors (Optional[List[str]]): Allowed components at the predictor node
                (empty/None -> full set). See ``scalers``.
            component_mut_prob (float): Probability a node's COMPONENT is resampled (structural
                mutation) during a mutation step; otherwise only that component's parameters are
                shift-mutated.
            gamma (float): Fraction of history treated as "good" by TPE.
            classification (bool): Whether the pipeline optimizes a classification task (True,
                scored by ROC-AUC) or a regression task (False, scored by R^2). Selects the
                predictor node's component registry (CLASSIFIERS vs. REGRESSORS) and the
                CV/test scoring path. Defaults to True.

        Note:
            The per-step allow-lists restrict only the concrete estimators; the parameter-free
            ``passthrough`` option (where a node permits it) is always available regardless.
        """
        # initialize the base class
        super().__init__(
            seed=seed,
            pop_size=pop_size,
            cores=cores,
            mut_prob=mut_prob,
            mut_var=mut_var,
            crossover_prob=crossover_prob,
            classification=classification,  # classification (ROC-AUC) or regression (R^2)
        )

        # CASH-specific validation
        assert 0.0 <= component_mut_prob <= 1.0, "Component mutation probability must be between 0 and 1."
        self.component_mut_prob = component_mut_prob

        # per-step allow-lists of component identifiers the evolution may use (empty -> full set for
        # that step). Keyed by node name so _build_pipeline_space can filter each registry. None is
        # normalized to [] so an empty list and an omitted argument both mean "use every component".
        self.allowed_components: Dict[str, List[str]] = {
            "feature_scaling": scalers or [],
            "feature_engineering": transformers or [],
            "feature_selection": selectors or [],
            "predictor": predictors or [],
        }

        # EA parameters
        self.tournament_size = tournament_size
        self.num_offspring = num_offspring

        # TPE-related parameters
        self.tpe_prob = tpe_prob
        self.gamma = gamma
        # whether the most recent TPE fit succeeded; refreshed once per generation by
        # generate_offspring and read by the variation operators to gate TPE-guided variation
        self.tpe_ready = False

        # numeric column indices (in the base-preprocessed matrix) the evolved scaler may touch;
        # None when every column is numeric (scaler applies to the whole matrix). Set in load_data_pd.
        self.scale_cols: Optional[List[int]] = None

        # deferred until load_data_pd (all require the dataset-derived DataContext)
        self.data_ctx: Optional[DataContext] = None
        self.pipeline_space: Optional[PipelineSpace] = None      # {node: {component: param_space}}
        self.operators: Optional[dict] = None                    # {node: {component: ModelParams or None}}
        self.node_names: Optional[Tuple[str, ...]] = None
        self.tpe: Optional[CASH_TPE] = None

        # full provenance archive of every evaluated individual (generation, construction, ei,
        # error, genome, performances); built in load_data_pd, saved by save_results, and fed
        # (deduplicated by genome key) to the TPE surrogate. This is the single source of history.
        self.eval_archive: Optional[CASHArchive] = None
        self.hard_eval_count = 0
        self.best_perf = float("-inf")
        # final selected result (architecture + genotype + train/val/test), populated at the end of
        # evolve() from the provenance archive and consumed by save_results(); no running
        # best-individual is kept
        self.best_result: Optional[Dict[str, Any]] = None

        # per-generation diagnostic checkpoints (test performance of the best-so-far individual),
        # recorded across generations so a checkpoint can stand in for a shorter run
        self.checkpoints: List[dict] = []
        # genome key of the most recent checkpoint's selected best, so a full-model test refit is
        # skipped when the drawn best is genome-identical to the previous checkpoint's
        self._last_ckpt_key: Optional[str] = None

        return

    def load_data_pd(self,
                     data: pd.DataFrame,
                     target_label: str,
                     train_p: float,
                     one_hot_cols: Optional[List[str]] = None,
                     scalar_cols: Optional[List[str]] = None,
                     n_folds: int = 5) -> None:
        """
        Loads data via the base class, then builds the (data-dependent) pipeline space and TPE.

        Several operators size their parameter bounds from the dataset (n_samples / n_features /
        n_classes via DataContext), so the pipeline space can only be constructed after the data
        is loaded.

        Args:
            n_folds (int): Number of cross-validation folds to build (see BaseEA.load_data_pd).
                Defaults to 5.
        """
        super().load_data_pd(data, target_label, train_p, one_hot_cols, scalar_cols, n_folds=n_folds)

        assert self.X_train is not None, "Data must be loaded before building the pipeline space."
        # n_classes only shapes classifier spaces (e.g. GB's exponential loss); regression targets
        # have no classes, so labels is None there and n_classes is set to 0 (unused by regressors).
        n_classes = len(self.labels) if self.labels is not None else 0
        # cap n_samples at the smallest CV training fold: bounds that scale with training rows
        # must be realizable on every fold, since each model is fit on k-1 folds during CV.
        self.data_ctx = DataContext(
            n_samples=self.smallest_cv_train_size(),
            n_features=self.X_train.shape[1],
            n_classes=n_classes,
        )
        # Determine which columns the evolved scaler is allowed to touch. The CASH base preprocessor
        # lays the matrix out numeric-first, so the numeric block is the contiguous index range
        # [0, n_numeric). If the matrix also contains one-hot dummies or unlisted passthrough columns
        # (i.e. numeric columns don't cover every original feature), the scaler must scale ONLY that
        # numeric range and leave the rest untouched; otherwise the whole matrix is numeric and the
        # scaler applies bare (scale_cols = None).
        n_numeric = len(self.numerical_cols) if self.numerical_cols else 0
        has_protected_cols = n_numeric < self.X_train.shape[1]
        self.scale_cols = list(range(n_numeric)) if has_protected_cols else None

        self._build_pipeline_space()
        self.tpe = CASH_TPE(gamma=self.gamma, pipeline_space=self.pipeline_space)

        # provenance archive of every evaluated pipeline, saved by save_results
        self.eval_archive = CASHArchive()

        return

    def _build_pipeline_space(self) -> None:
        """
        Build the operator table and the TPE pipeline space from the Source/ML registries.

        For each node: instantiate every registered ModelParams class with the DataContext,
        keyed by its ``get_model_type()`` name, restrict it to the user's per-step allow-list
        (``self.allowed_components``; an empty list keeps the full registry), and (where allowed)
        add the parameter-free "passthrough" component. ``self.operators`` holds the live
        ModelParams objects (used for sampling/mutating that component's parameters; ``None`` for
        passthrough), while ``self.pipeline_space`` holds the lightweight
        ``{node: {component: param_space}}`` metadata that CASH_TPE consumes.

        Raises:
            ValueError: if a node's allow-list names a component the node's registry does not offer.
        """
        assert self.data_ctx is not None, "DataContext must be built before the pipeline space."

        operators: dict = {}
        pipeline_space: PipelineSpace = {}
        for node, registry, allow_passthrough in PIPELINE_NODES:
            # the predictor node's registry is task-dependent: classifiers for classification,
            # regressors for regression (both share the same six model-family identifiers, except
            # the kernel-SVM, which is "KSVC" vs. "SVR"). Preprocessing nodes are task-agnostic.
            if node == PREDICTOR_NODE and not self.classification:
                registry = REGRESSORS
            # empty allow-list means "use every registered component" for this step
            allowed = self.allowed_components.get(node, [])
            node_ops: dict = {}
            node_space: dict = {}
            for cls in registry:
                instance = cls(self.data_ctx)
                name = instance.get_model_type()
                if allowed and name not in allowed:
                    continue  # user restricted this step to a subset that excludes this component
                node_ops[name] = instance
                node_space[name] = instance.param_space
            # fail loudly if the user asked for a component this node cannot provide (typo / wrong step)
            if allowed:
                available = [cls(self.data_ctx).get_model_type() for cls in registry]
                unknown = [m for m in allowed if m not in available]
                if unknown:
                    raise ValueError(
                        f"Unknown component(s) {unknown} for node '{node}'. "
                        f"Available components: {available}."
                    )
            if allow_passthrough:
                node_ops[PASSTHROUGH] = None
                node_space[PASSTHROUGH] = {}
            operators[node] = node_ops
            pipeline_space[node] = node_space

        self.operators = operators
        self.pipeline_space = pipeline_space
        self.node_names = tuple(pipeline_space.keys())

        return

    # ------------------------------------------------------------------ candidate construction

    def _random_node_entry(self, node: str, rng: np.random.Generator) -> dict:
        """Pick a random component for ``node`` and sample its parameters (``{}`` for passthrough)."""
        components = list(self.operators[node].keys())
        name = str(rng.choice(components))
        ops = self.operators[node][name]
        params = {} if ops is None else ops.generate_random_parameters(rng)
        return {"name": name, "params": params}

    def _random_candidate(self, rng: np.random.Generator) -> Candidate:
        """Build a full random pipeline candidate (one component + params per node)."""
        return {node: self._random_node_entry(node, rng) for node in self.node_names}

    def _mutate_genotype(self, genotype: Candidate, use_tpe: bool) -> Candidate:
        """
        Produce a new pipeline candidate by mutating ``genotype``.

        Each node is mutated independently. With probability ``component_mut_prob`` the node
        undergoes STRUCTURAL mutation -- its component is resampled and fresh parameters drawn
        (a passthrough choice is possible where allowed). Otherwise the component is kept and its
        parameters are mutated PARAMETRICALLY, in a way that matches the offspring's exploration
        mode: a TPE offspring takes a small local Gaussian shift (variance ``self.mut_var``, per-gene
        rate ``self.mut_prob``) so it explores near the parent, while a non-TPE offspring resamples
        each gene uniformly from its full range (``mutate_parameters_random``) so it jumps around
        unbiased. A parameter-free component (passthrough or default-only operator) simply keeps its
        empty parameter dict.
        """
        child: Candidate = {}
        for node in self.node_names:
            entry = genotype[node]
            if self.rng.random() < self.component_mut_prob:
                # structural mutation: resample the component and its parameters
                child[node] = self._random_node_entry(node, self.rng)
            else:
                # parametric mutation: keep the component, mutate its parameters
                name = entry["name"]
                ops = self.operators[node][name]
                if ops is None:
                    params: dict = {}
                elif use_tpe:
                    params = ops.mutate_parameters_shift(cp.deepcopy(entry["params"]), self.mut_var, self.mut_prob, self.rng)
                else:
                    params = ops.mutate_parameters_random(cp.deepcopy(entry["params"]), self.mut_prob, self.rng)
                child[node] = {"name": name, "params": params}
        return child

    def _crossover_child(self, parent_a: Individual, parent_b: Individual, use_tpe: bool) -> Tuple[Candidate, str]:
        """
        Produce a single recombined child genotype: uniform (per-node) crossover of the two
        parents, then (with probability ``self.mut_prob``) a mutation (structural + parametric via
        ``_mutate_genotype``) matching the exploration mode -- a small local shift on the TPE path,
        an unbiased random resample otherwise.

        Also reports the variation operator for THIS candidate: ``CROSSOVER_MUTATION_OPERATION`` if
        the mutation gate fired, else ``CROSSOVER_OPERATION``. This is per-candidate because a TPE
        roll generates several candidates (each rolling the gate independently) and only the chosen
        one's operator is recorded.

        Args:
            parent_a (Individual): The first parent.
            parent_b (Individual): The second parent.
            use_tpe (bool): Which mutation mode to apply if the child is mutated.

        Returns:
            Tuple[Candidate, str]: The recombined child's pipeline genotype and its variation operator.
        """
        child = self._uniform_crossover(parent_a, parent_b)

        # offspring-level mutation gate: with probability mut_prob, mutate the crossover child
        if self.rng.random() < self.mut_prob:
            child = self._mutate_genotype(child, use_tpe)
            return child, CROSSOVER_MUTATION_OPERATION
        return child, CROSSOVER_OPERATION

    def _uniform_crossover(self, parent_a: Individual, parent_b: Individual) -> Candidate:
        """
        Uniform per-node recombination: for each pipeline node, the offspring inherits that node's
        component AND its parameters wholesale from either parent with equal probability. Nodes are
        recombined as a unit (component + params kept together) so the child never pairs one
        component's name with another's parameters. Returns the recombined genotype (no mutation).

        Args:
            parent_a (Individual): The first parent.
            parent_b (Individual): The second parent.

        Returns:
            Candidate: The recombined pipeline genotype.
        """
        genotype_a = parent_a.get_genotype()
        genotype_b = parent_b.get_genotype()
        assert genotype_a.keys() == genotype_b.keys(), "Parents must share the same pipeline nodes for crossover."

        return {
            node: cp.deepcopy(genotype_a[node] if self.rng.random() < 0.5 else genotype_b[node])
            for node in self.node_names
        }

    def _tpe_or_explore(self, tpe_candidate, explore_candidate) -> Tuple[Candidate, str, str, float]:
        """
        Shared TPE decision used by both variation operators. Rolls for TPE-guided variation
        (probability ``self.tpe_prob``, gated on a successful TPE fit ``self.tpe_ready``): on a TPE
        roll, generate ``self.num_offspring`` candidates via ``tpe_candidate`` and return the one the
        TPE surrogate ranks best; otherwise return a single candidate from ``explore_candidate``.
        CASH_TPE scores the nested candidate dicts directly (no per-parameter re-encoding needed).

        Each candidate factory returns a ``(genotype, operation)`` pair so the chosen candidate's
        variation operator (which, for crossover, records whether its per-candidate mutation gate
        fired) is reported alongside the winning genotype.

        Args:
            tpe_candidate (Callable[[], Tuple[Candidate, str]]): Factory for a TPE-mode candidate
                ``(genotype, operation)`` (called ``self.num_offspring`` times on a TPE roll).
            explore_candidate (Callable[[], Tuple[Candidate, str]]): Factory for a single
                unbiased-exploration candidate ``(genotype, operation)`` (called once otherwise).

        Returns:
            Tuple[Candidate, str, str, float]: The chosen candidate's pipeline genotype, its
            variation operator, its construction tag (``TPE_CONSTRUCTION`` or
            ``RANDOM_CONSTRUCTION``), and its expected-improvement score (the surrogate's acquisition
            value for a TPE pick, ``-inf`` for a random pick).
        """
        if self.tpe_ready and (self.rng.random() < self.tpe_prob):
            candidate_offspring = [tpe_candidate() for _ in range(self.num_offspring)]
            genotypes = [genotype for genotype, _ in candidate_offspring]
            candidate_index = self.tpe.suggest_one(genotypes, self.rng)
            # the acquisition score of the chosen candidate is its expected improvement
            ei = float(self.tpe.score_candidates([genotypes[candidate_index]])[0])
            genotype, operation = candidate_offspring[candidate_index]
            return genotype, operation, TPE_CONSTRUCTION, ei
        genotype, operation = explore_candidate()
        return genotype, operation, RANDOM_CONSTRUCTION, float("-inf")

    # ------------------------------------------------------------------ core EA loop

    def evolve(self, gens: int, checkpoint_dir: Optional[str] = None) -> None:
        """
        Run the CASH EA. Mirrors the HPO EA's generational loop: evaluate, archive, select
        parents, generate offspring (mutation/crossover with optional TPE guidance), evaluate,
        and track the best-so-far individual.

        NOTE: evaluation of pipelines is not yet implemented (see evaluation()), so this method
        cannot be run end-to-end until that is added.

        Args:
            gens (int): Number of generations to evolve.
            checkpoint_dir (Optional[str]): If provided, the best-so-far individual's test
                performance is recorded after every generation (diagnostic only).
        """
        assert gens > 0, "Number of generations must be a positive integer."
        assert self.operators is not None, "Data must be loaded before evolving."

        start_time = time.time()

        # initialize and evaluate the starting population
        self.initialize_population()
        evaluated = self.evaluation(self.population)

        # record full provenance for the initial (random) population as generation -1 (this is the
        # sole history store; the TPE surrogate is fit from it, deduplicated by genome key)
        self.record_history(evaluated, generation=-1)
        # errored pipelines are archived above but must not seed the next generation
        self.population = self._drop_errored(evaluated)
        self.update_best_seen(self.population)
        print(f"Best performance so far (Gen -1): {self.best_perf}", flush=True)
        self.checkpoint_best_seen(generation=-1, checkpoint_dir=checkpoint_dir)

        for g in range(gens):
            # decide per-offspring operators and how many parents they consume
            var_order, num_parents = self.variation_order(self.pop_size, self.crossover_prob)

            # tournament parent selection
            parent_ids = self.parent_selection(self.population, num_parents, self.rng)

            # generate offspring via mutation/crossover (TPE-guided where rolled)
            offspring = self.generate_offspring(self.population, parent_ids, var_order)

            # evaluate offspring (reusing archived/duplicate results where possible)
            evaluated = self.evaluation(offspring)

            # record full provenance for this generation's offspring
            self.record_history(evaluated, generation=g)
            # errored pipelines are archived above but must not seed the next generation
            self.population = self._drop_errored(evaluated)
            self.update_best_seen(self.population)
            print(f"Best performance so far (Gen {g}): {self.best_perf}", flush=True)
            self.checkpoint_best_seen(generation=g, checkpoint_dir=checkpoint_dir)

        print(f"Hard evaluations: {self.hard_eval_count}", flush=True)
        print(f"Total evolution time (mins): {(time.time() - start_time) / 60}", flush=True)

        # the best validation performance found across the whole run
        print(f"Best validation performance found: {self.best_perf}", flush=True)

        # The returned pipeline is drawn from the archive with a random tie-break (``_select_best``).
        # When checkpointing ran, the final checkpoint already made this draw and stashed it in
        # ``self.best_result``; reuse it so the saved result matches the last checkpoint exactly.
        # Otherwise (no checkpointing) draw and evaluate once here.
        if self.best_result is None:
            best_individual, best_val = self._select_best()
            train_score, test_score = self.model_test_evaluation(best_individual)
            self.best_result = {
                "architecture": best_individual.get_architecture(),
                "genotype": best_individual.get_genotype(),
                "train_performance": train_score,
                "val_performance": best_val,
                "test_performance": test_score,
            }

        print(f"Final test evaluation - Train: {self.best_result['train_performance']}, "
              f"Val: {self.best_perf}, Test: {self.best_result['test_performance']}", flush=True)

        return

    def initialize_population(self) -> None:
        """Initialize the population with random pipeline candidates."""
        assert self.operators is not None, "Data must be loaded before initializing the population."
        assert len(self.population) == 0, "Population has already been initialized."

        self.population = []
        for _ in range(self.pop_size):
            ind = CASHIndividual(self._random_candidate(self.rng))
            # the initial population is drawn at random (no TPE guidance)
            ind.construction = RANDOM_CONSTRUCTION
            self.population.append(ind)
        return

    def parent_selection(self, population: List[Individual], num_parents: int, rng: np.random.Generator) -> List[int]:
        """
        Select parents via tournament selection (maximize validation performance).

        Parameters:
            population (List[Individual]): The population of individuals.
            num_parents (int): The number of parents to select.
            rng (np.random.Generator): Random number generator for reproducibility.

        Returns:
            List[int]: Indices of the selected parent individuals.
        """
        assert len(population) > 0, "Population must not be empty."

        # a tournament cannot draw more distinct competitors than the population holds; dropping
        # errored pipelines can shrink the population below tournament_size, so clamp accordingly.
        k = min(self.tournament_size, len(population))

        parent_ids = []
        for _ in range(num_parents):
            indices = rng.choice(len(population), k, replace=False)
            extracted_performances = np.array([population[i].get_val_performance() for i in indices])
            best_tour_idx = np.argmax(extracted_performances)
            winner = int(rng.choice([i for i, perf in zip(indices, extracted_performances) if perf == extracted_performances[best_tour_idx]]))
            parent_ids.append(winner)

        return parent_ids

    def generate_offspring(self, candidates: List[Individual], parent_ids: List[int], variation_order: List[str]) -> List[Individual]:
        """
        Generate offspring from selected parents according to a precomputed variation order.

        Each entry in ``variation_order`` is 'm' (mutation, 1 parent) or 'c' (crossover, 2 parents),
        and ``parent_ids`` is consumed left-to-right. TPE is fit once here (gated on ``self.tpe_prob``
        and a successful fit) and the result stashed in ``self.tpe_ready``; each variation operator
        (``mutate`` / ``crossover``) then owns its own per-offspring TPE decision and offspring generation.

        Args:
            candidates (List[Individual]): The current population (indexed by ``parent_ids``).
            parent_ids (List[int]): Indices of selected parents, ordered to match consumption.
            variation_order (List[str]): Per-offspring operators ('m' or 'c').

        Returns:
            List[Individual]: One offspring per entry in ``variation_order``.
        """
        expected_parents = sum(1 if op == 'm' else 2 for op in variation_order)
        assert len(parent_ids) == expected_parents, "Number of parent IDs must match the parents required by the variation order."
        assert len(variation_order) > 0, "At least one offspring must be generated."
        assert self.eval_archive is not None and len(self.eval_archive) > 0, \
            "The provenance archive must hold at least one evaluation for TPE-based variation."

        offspring = []

        # fit TPE on the current history; gate TPE-guided variation on a successful fit (the
        # pipeline-aware fit returns False when history cannot be split into good/bad groups).
        # tpe_ready is stashed so the variation operators (mutate / crossover) can read it.
        self.tpe_ready = False
        if self.tpe_prob > 0.0:
            self.tpe_ready = self.tpe.fit(self._tpe_samples(), self.rng)

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

    def mutate(self, parent: Individual) -> CASHIndividual:
        """
        Produce one offspring from a single parent, owning the full TPE decision.

        Rolls for TPE-guided variation (probability ``self.tpe_prob``, gated on a successful TPE
        fit):
          - non-TPE: a single offspring whose parametric genes are resampled uniformly at random
            (``mutate_parameters_random``), so it jumps around the space unbiased.
          - TPE: generate ``self.num_offspring`` candidates whose parametric genes take a small
            local shift (variance ``self.mut_var``) and keep the one the TPE surrogate ranks best.
        In both modes each node independently undergoes structural mutation with probability
        ``component_mut_prob`` (its component is resampled and fresh parameters drawn).

        Args:
            parent (Individual): The parent whose pipeline is mutated.

        Returns:
            CASHIndividual: A new offspring pipeline individual.
        """
        child, operation, construction, ei = self._tpe_or_explore(
            tpe_candidate=lambda: (self._mutate_genotype(parent.get_genotype(), use_tpe=True), MUTATION_OPERATION),
            explore_candidate=lambda: (self._mutate_genotype(parent.get_genotype(), use_tpe=False), MUTATION_OPERATION),
        )
        # a mutation-only offspring has a single parent, so both parent slots reference it
        return self._build_offspring(child, construction, ei, operation, (parent, parent))

    def crossover(self, parent_a: Individual, parent_b: Individual) -> CASHIndividual:
        """
        Produce one offspring from two parents, owning the full TPE decision.

        Rolls for TPE-guided variation (probability ``self.tpe_prob``, gated on a successful TPE
        fit):
          - non-TPE: recombine the parents (uniform per-node crossover) and, with probability
            ``self.mut_prob``, apply an unbiased random mutation to the child; keep that single
            candidate.
          - TPE: generate ``self.num_offspring`` recombined candidates, each of which (with
            probability ``self.mut_prob``) receives a small local shift-mutation, then keep the one
            the TPE surrogate ranks best.

        Nodes are recombined as a unit (component + params kept together) so the child never pairs
        one component's name with another's parameters.

        Args:
            parent_a (Individual): The first parent.
            parent_b (Individual): The second parent.

        Returns:
            CASHIndividual: A new offspring pipeline individual.
        """
        child, operation, construction, ei = self._tpe_or_explore(
            tpe_candidate=lambda: self._crossover_child(parent_a, parent_b, use_tpe=True),
            explore_candidate=lambda: self._crossover_child(parent_a, parent_b, use_tpe=False),
        )
        return self._build_offspring(child, construction, ei, operation, (parent_a, parent_b))

    def _build_offspring(self, genotype: Candidate, construction: str, ei: float,
                         operation: str, parents: Tuple[Individual, Individual]) -> CASHIndividual:
        """
        Wrap a child genotype in a CASHIndividual, tagging it with its construction provenance
        (random vs TPE), its variation operator, and its parents' archive ids -- and, for TPE
        offspring, its expected-improvement score. All are consumed by :meth:`record_history` when
        the individual is later archived.

        Args:
            genotype (Candidate): The child's nested pipeline genotype.
            construction (str): RANDOM_CONSTRUCTION or TPE_CONSTRUCTION.
            ei (float): Expected improvement of the chosen candidate (-inf for random offspring).
            operation (str): The variation operator that produced the child (MUTATION_OPERATION,
                CROSSOVER_OPERATION, or CROSSOVER_MUTATION_OPERATION).
            parents (Tuple[Individual, Individual]): The two parents (identical objects for a
                mutation-only offspring); their archive ids are recorded as the child's lineage.

        Returns:
            CASHIndividual: The tagged offspring.
        """
        assert parents[0].archive_id is not None and parents[1].archive_id is not None, \
            "Parents must already be archived (carry an archive_id) before producing offspring."
        child = CASHIndividual(genotype)
        child.construction = construction
        if construction == TPE_CONSTRUCTION:
            child.set_ei(ei)
        child.operation = operation
        child.parent_ids = (parents[0].archive_id, parents[1].archive_id)
        return child

    def _tpe_samples(self) -> List[Individual]:
        """
        Build the TPE fitting set from the archive, one sample per UNIQUE pipeline.

        The  archive (``self.eval_archive``) logs every production, including duplicates
        and errored pipelines. Feeding duplicates to the surrogate would over-weight repeatedly
        produced genomes in the good/bad split, so we keep only the first entry seen per genome
        key; because evaluation is genome-deterministic, all entries sharing a key carry identical
        performances, so the choice of representative is immaterial.

        Each kept entry becomes one CASHIndividual carrying the NEGATED validation performance,
        because CASH_TPE treats the objective as a minimization (its good/bad split takes the
        lowest values as "good") while the EA maximizes validation AUC. Errored pipelines were
        hard-penalized to validation 0.0, so they negate to 0.0 and reliably land in the "bad"
        group, steering the surrogate away from them.
        """
        assert self.eval_archive is not None, "Data must be loaded before fitting TPE."

        samples: List[Individual] = []
        seen_keys = set()
        for entry in self.eval_archive:
            if entry.key in seen_keys:
                continue
            seen_keys.add(entry.key)
            assert entry.val_performance is not None, \
                "Archived evaluation is missing a validation score (errored pipelines are penalized to 0.0, not None)."
            ind = self.eval_archive.build_individual(entry)
            ind.set_val_performance(entry.val_performance * -1.0)  # TPE minimizes, so invert
            samples.append(ind)
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
        Update the best validation performance seen across all generations.

        Only the scalar best-so-far performance is tracked; the winning pipeline itself is
        recovered from the provenance archive when needed (checkpoints, final selection), so no
        running copy is kept here.

        Args:
            individuals (List[Individual]): Individuals to check.
        """
        for ind in individuals:
            perf = ind.get_val_performance()
            if perf > self.best_perf:
                self.best_perf = perf
        return

    def _select_best(self) -> Tuple[CASHIndividual, float]:
        """
        Draw the current best pipeline from the provenance archive, breaking validation ties
        uniformly at random with ``self.rng``, and return it (with its validation performance set)
        alongside that validation score.

        Both the per-generation checkpoint and the final evaluation select through here, so a
        checkpoint recorded at a given evaluation budget is drawn by exactly the same rule as a full
        run that stops at that budget -- which is what lets a checkpoint stand in for a shorter run
        without re-running. Because the draw consumes ``self.rng``, enabling checkpointing shifts the
        search trajectory relative to a run with checkpointing off.
        """
        assert self.eval_archive is not None, "No archive to select the best individual from."
        best_entry = self.eval_archive.best(self.rng)
        assert best_entry is not None and best_entry.val_performance is not None, \
            "Archive holds no scoreable individual to select."
        best_val = float(best_entry.val_performance)
        # consistency guard: the archive's best validation must match the tracked best-so-far
        assert best_val == self.best_perf, (
            f"Archive best validation ({best_val}) does not match the best "
            f"validation performance tracked during the run ({self.best_perf})."
        )
        best_individual = self.eval_archive.build_individual(best_entry)
        best_individual.set_val_performance(best_val)
        return best_individual, best_val

    def checkpoint_best_seen(self, generation: int, checkpoint_dir: Optional[str]) -> None:
        """
        Record the test-set performance of the best-so-far individual for one generation
        (diagnostic aid, and a stand-in for a run that stops at that evaluation budget). The best
        pipeline is drawn from the archive via ``_select_best`` -- the SAME random tie-break used
        for the final result -- so the final checkpoint's selection is exactly what ``evolve``
        returns. It is only re-evaluated on the test set when its genome differs from the previous
        checkpoint's; a genome-identical draw reuses the previous scores.

        Args:
            generation (int): Generation index (-1 for the initial population).
            checkpoint_dir (Optional[str]): Directory to write ``checkpoints.csv`` into; None
                disables checkpointing.
        """
        if checkpoint_dir is None:
            return

        # draw the current best with the same random tie-break used for the final result
        assert self.eval_archive is not None, "No archive to checkpoint from."
        best_individual, best_val = self._select_best()

        # refit on the test set only when the drawn genome changed since the previous checkpoint
        ckpt_key = self.eval_archive.compute_key(best_individual)
        if self.checkpoints and self._last_ckpt_key == ckpt_key:
            train_score = self.checkpoints[-1]["train_auc"]
            test_score = self.checkpoints[-1]["test_auc"]
        else:
            train_score, test_score = self.model_test_evaluation(best_individual)
        self._last_ckpt_key = ckpt_key

        self.checkpoints.append({
            "generation": generation,
            "hard_evals": self.hard_eval_count,
            "val_auc": float(best_val),
            "train_auc": float(train_score),
            "test_auc": float(test_score),
        })

        os.makedirs(checkpoint_dir, exist_ok=True)
        csv_path = os.path.join(checkpoint_dir, "checkpoints.csv")
        pd.DataFrame(self.checkpoints).to_csv(csv_path, index=False)

        # snapshot of the best-so-far result at this many candidates considered, mirroring the
        # best_results.json schema so progress can be tracked generation by generation. The budget
        # index N in the filename is the total number of candidates considered so far -- the running
        # sum of population size over generations -- which is exactly the archive length (every
        # evaluated pipeline, including redundant genomes, is recorded there). This is NOT
        # ``hard_eval_count`` (distinct genomes actually fitted), which is <= candidates considered.
        candidates_considered = len(self.eval_archive)
        result_snapshot = {
            "generation": generation,
            "candidates_considered": candidates_considered,
            "hard_evals": self.hard_eval_count,
            "task_id": self.task_id,
            "seed": self.seed,
            "architecture": best_individual.get_architecture(),
            "train_accuracy": float(train_score),
            "validation_accuracy": float(best_val),
            "test_accuracy": float(test_score),
            "best_pipeline": best_individual.get_genotype(),
        }
        json_path = os.path.join(checkpoint_dir, f"results_eval_{candidates_considered}.json")
        with open(json_path, 'w') as f:
            json.dump(result_snapshot, f, indent=4)

        # record this selection so the final evaluation returns exactly the last checkpoint
        self.best_result = {
            "architecture": best_individual.get_architecture(),
            "genotype": best_individual.get_genotype(),
            "train_performance": train_score,
            "val_performance": best_val,
            "test_performance": test_score,
        }

        print(f"Checkpoint (Gen {generation}) - Val {self.metric_name}: {self.best_perf:.4f}, "
              f"Train {self.metric_name}: {train_score:.4f}, Test {self.metric_name}: {test_score:.4f} "
              f"-> {json_path}", flush=True)

        return

    # ------------------------------------------------------------------ evaluation

    def _build_preprocessor(self) -> ColumnTransformer:
        """
        Build the CASH base preprocessor: only make the data numeric so the evolved pipeline can
        take over. Categorical columns are one-hot-encoded; numeric columns are passed through
        UNSCALED because scaling is an evolved decision (the pipeline's ``feature_scaling`` node),
        not a fixed baseline step. Doing otherwise would both double-scale (base scaler + evolved
        scaler) and rob a passthrough scaling node of its meaning.

        The numeric columns are listed FIRST (as an explicit passthrough) so the output matrix has
        a stable, fit-independent layout: the numeric block always occupies indices
        ``[0, len(numerical_cols))``, followed by the one-hot dummies and any remaining columns.
        This is what lets the evolved scaler target the numeric columns by index.

        Returns:
            ColumnTransformer: The configured preprocessor (numeric passthrough first, then
            one-hot; everything else passes through untouched).
        """
        transformers = []
        if self.numerical_cols:
            transformers.append(('num', 'passthrough', self.numerical_cols))
        if self.categorical_cols:
            transformers.append(('cat', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'), self.categorical_cols))

        return ColumnTransformer(transformers=transformers, remainder='passthrough')

    def _resolve_pipeline_steps(self, genotype: Candidate) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Turn a candidate's nested genotype into an ordered, serializable list of pipeline steps
        for the Ray evaluation task: one ``(node_name, component_name, eval_kwargs)`` per node, in
        pipeline execution order. Each component's own ``eval_parameters`` translates its evolved
        genotype into scikit-learn kwargs (the seed is folded in for stochastic estimators);
        passthrough nodes carry the ``PASSTHROUGH`` marker with empty kwargs and are dropped later,
        when the pipeline is assembled inside the CV task (``cv_pipeline_classification`` /
        ``cv_pipeline_regression``).

        Args:
            genotype (Candidate): The nested pipeline genotype ({node: {"name", "params"}}).

        Returns:
            List[Tuple[str, str, Dict[str, Any]]]: Ordered per-node (node, component, kwargs).
        """
        steps: List[Tuple[str, str, Dict[str, Any]]] = []
        for node in self.node_names:
            entry = genotype[node]
            name = entry["name"]
            ops = self.operators[node][name]
            if ops is None:  # passthrough: no estimator, marked for omission at build time
                steps.append((node, PASSTHROUGH, {}))
            else:
                steps.append((node, name, ops.eval_parameters(entry["params"], random_state=self.seed)))
        return steps

    def evaluation(self, candidates: List[Individual]) -> List[Individual]:
        """
        Evaluate pipeline individuals using Ray across 5-fold cross-validation, updating each
        individual's train and validation performance.

        One Ray task is launched per (candidate, fold) pair so every fold of every pipeline evaluates
        in parallel. Each fold's preprocessed data already lives in the Ray object store
        (see BaseEA._prepare_cv_folds), so a task loads only the single fold it needs. The candidate's
        resolved pipeline steps (see ``_resolve_pipeline_steps``) are passed to each fold task, which
        builds the scikit-learn Pipeline (omitting passthrough nodes) and scores it. As results stream
        back, per-fold performances are accumulated per pipeline and its mean CV performance is finalized
        once all of its folds arrive.

        Args:
            candidates (List[Individual]): List of individuals to evaluate.

        Returns:
            List[Individual]: The evaluated individuals with updated performance metrics.
        """
        # get per-fold CV data (each fold's arrays live in the Ray object store)
        cv_splits = self.get_cv_splits()
        num_folds = len(cv_splits)

        # Deduplicate before evaluating: a pipeline is deterministic under a fixed seed + CV, so any
        # candidate whose genome was already evaluated (recorded in the provenance archive, or seen
        # earlier in THIS batch) reuses those performances instead of being re-evaluated. Only the
        # first sighting of each distinct genome is dispatched to Ray as a hard evaluation.
        pending = self._resolve_duplicates(candidates)

        # classification pipelines are scored by ROC-AUC, regression pipelines by R^2 (the
        # regression task ignores binary_class/labels, which are passed for a uniform call site)
        cv_pipeline = cv_pipeline_classification if self.classification else cv_pipeline_regression

        # launch one Ray task per (pending candidate, fold) so every fold evaluates in parallel
        ray_jobs = []
        for model_id, ind in enumerate(pending):
            pipeline_steps = self._resolve_pipeline_steps(ind.get_genotype())
            for X_train, y_train, X_validate, y_validate in cv_splits:
                ray_jobs.append(cv_pipeline.remote(
                    X_train=X_train,
                    y_train=y_train,
                    X_validate=X_validate,
                    y_validate=y_validate,
                    pipeline_steps=pipeline_steps,
                    scale_cols=self.scale_cols,
                    id=model_id,
                    binary_class=self.binary_classification,
                    labels=self.labels,
                ))

        # accumulate fold performances per pending pipeline as results arrive; track whether any
        # fold errored so the individual can be flagged when archived
        pop_results = [{'train_acc': [], 'val_acc': [], 'error': False} for _ in pending]

        while len(ray_jobs) > 0:
            finished, ray_jobs = ray.wait(ray_jobs, num_returns=min(len(ray_jobs), self.cores))
            for done_id in finished:
                model_id, train_acc, val_acc, error = ray.get(done_id)

                # A CASH pipeline can legitimately fail to fit on a fold when its evolved components
                # are mutually incompatible (e.g. a VarianceThreshold that removes every feature, or
                # a chi2 kernel on signed input). Such a fold is not an error to abort on -- but a
                # pipeline that fails on ANY fold is penalized completely (fitness forced to 0.0, see
                # finalize below) so that only pipelines succeeding across all folds survive into the population.
                if error < 0.0:
                    pop_results[model_id]['error'] = True
                    print(f"Pipeline {model_id} failed on a fold (fitness penalized to 0.0).", flush=True)

                # track this fold's performance for the corresponding pipeline as it comes in
                pop_results[model_id]['train_acc'].append(train_acc)
                pop_results[model_id]['val_acc'].append(val_acc)

                # once all folds for this pipeline are in, finalize its CV performance. Any fold
                # failure is a complete penalty (0.0/0.0), not an average that could still look
                # competitive; only an all-fold success keeps its true mean CV performance.
                if len(pop_results[model_id]['val_acc']) == num_folds:
                    errored = pop_results[model_id]['error']
                    # Errored pipelines are penalized to the reliably-worst score so they never
                    # outrank a real evaluation. For classification (ROC-AUC in [0, 1]) that floor
                    # is 0.0; for regression (R^2 in (-inf, 1]) 0.0 is NOT the worst -- it is
                    # "predicts the mean" -- so we use -inf to keep errored pipelines out of the
                    # good group when the TPE surrogate is fit.
                    penalty = 0.0 if self.classification else float("-inf")
                    mean_train = penalty if errored else float(np.mean(pop_results[model_id]['train_acc']))
                    mean_val = penalty if errored else float(np.mean(pop_results[model_id]['val_acc']))
                    pending[model_id].set_train_performance(mean_train)
                    pending[model_id].set_val_performance(mean_val)
                    pending[model_id].eval_error = errored
                    print(f"Pipeline {model_id} evaluated - Train {self.metric_name}: {mean_train:.4f}, Val {self.metric_name}: {mean_val:.4f}", flush=True)

        # in-batch duplicates: candidates that shared a (not-previously-seen) genome with a pending
        # sibling were not dispatched, so copy the freshly evaluated result onto them.
        if self.eval_archive is not None:
            evaluated_by_key = {self.eval_archive.compute_key(ind): ind for ind in pending}
            for ind in candidates:
                if ind.val_performance is None:
                    src = evaluated_by_key[self.eval_archive.compute_key(ind)]
                    ind.set_train_performance(src.get_train_performance())
                    ind.set_val_performance(src.get_val_performance())
                    ind.eval_error = src.eval_error

        # every distinct genome that reached Ray is one hard (real pipeline-fitting) evaluation
        self.hard_eval_count += len(pending)
        return candidates

    def _reuse_archived_performance(self, ind: Individual) -> bool:
        """
        If ``ind``'s genome was already evaluated (recorded in the provenance archive), copy the
        stored train/validation/error onto it and return True; otherwise return False.

        Pipeline performances are genome-deterministic (fixed seed + CV folds), so a previously
        evaluated pipeline need not be re-fit -- this is what lets the EA skip redundant hard
        evaluations.

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

        Errored pipelines are still archived (for provenance and dedup) but must not seed the next
        generation. If every individual errored there is nothing to breed from, so the full set is
        kept (with a warning) rather than returning an empty population.

        Args:
            evaluated (List[Individual]): The just-evaluated individuals.

        Returns:
            List[Individual]: The survivors (or all of ``evaluated`` if none survived).
        """
        survivors = [ind for ind in evaluated if not ind.eval_error]
        if not survivors:
            print("All pipelines errored this generation; keeping them so the run can continue.", flush=True)
            return evaluated
        if len(survivors) < len(evaluated):
            print(f"Removed {len(evaluated) - len(survivors)} errored pipeline(s) from the population.", flush=True)
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

    def model_test_evaluation(self, individual: Individual) -> Tuple[float, float]:
        """
        Fit an individual's pipeline on the full training set and evaluate on the held-out test
        set, returning (train_score, test_score).

        The CASH base preprocessor (one-hot only; numerics pass through unscaled) is fit on the
        full training split and applied to both train and test, then the candidate's scikit-learn
        Pipeline is assembled from its genotype -- the same steps as ``evaluation`` (omitting
        "passthrough" nodes, numeric-column-aware scaler) via ``_resolve_pipeline_steps``/``assemble_steps``
        -- fit on the preprocessed training data. Classification pipelines are scored by train/test
        ROC-AUC (via ``predict_proba``); regression pipelines by train/test R^2 (via ``predict``).

        This is only ever called on the CV-validated best-so-far individual (recovered from the
        provenance archive), which fit successfully on all folds, so no per-fold failure handling
        is needed here.

        Args:
            individual (Individual): The individual to evaluate.

        Returns:
            Tuple[float, float]: (train_score, test_score).
        """
        assert self.X_train is not None and self.X_test is not None, "Data must be loaded before evaluation."
        assert self.y_train is not None and self.y_test is not None, "Data must be loaded before evaluation."

        # preprocess (one-hot only; numerics pass through unscaled) with the base preprocessor,
        # fit on train only
        preprocessor = self._build_preprocessor()
        X_train_preprocessed = preprocessor.fit_transform(self.X_train)
        X_test_preprocessed = preprocessor.transform(self.X_test)

        # assemble the candidate's pipeline (omitting passthrough nodes; scaler is numeric-column-
        # aware when dummies are present) and fit on the full train -- same steps as evaluation
        pipeline_steps = self._resolve_pipeline_steps(individual.get_genotype())
        steps = assemble_steps(pipeline_steps, self.scale_cols, classification=self.classification)
        model = Pipeline(steps)
        model.fit(X_train_preprocessed, self.y_train)

        # regression pipelines are scored by R^2 (predict); classification by ROC-AUC (predict_proba)
        if not self.classification:
            train_score = float(r2_score(self.y_train, model.predict(X_train_preprocessed)))
            test_score = float(r2_score(self.y_test, model.predict(X_test_preprocessed)))
            return train_score, test_score

        assert self.binary_classification is not None, "Data must be loaded before evaluation."
        train_pred_proba = model.predict_proba(X_train_preprocessed)
        test_pred_proba = model.predict_proba(X_test_preprocessed)

        if self.binary_classification:
            train_score = float(roc_auc_score(self.y_train, train_pred_proba[:, 1]))
            test_score = float(roc_auc_score(self.y_test, test_pred_proba[:, 1]))
        else:
            train_score = float(roc_auc_score(self.y_train, train_pred_proba, multi_class='ovo', labels=self.labels))
            test_score = float(roc_auc_score(self.y_test, test_pred_proba, multi_class='ovo', labels=self.labels))

        return train_score, test_score

    # ------------------------------------------------------------------ results

    def save_results(self, save_dir: str) -> None:
        """
        Save final results (train/validation/test AUC and the winning pipeline genotype) as JSON.
        """
        assert self.best_result is not None, "No best result found. Run evolve() first."

        print(f"Best pipeline: {self.best_result['genotype']}", flush=True)
        print(f"Best validation performance: {self.best_perf}", flush=True)

        os.makedirs(save_dir, exist_ok=True)

        best_results = {
            "task_id": self.task_id,
            "seed": self.seed,
            "architecture": self.best_result["architecture"],
            "train_accuracy": self.best_result["train_performance"],
            "validation_accuracy": float(self.best_perf),
            "test_accuracy": self.best_result["test_performance"],
            "best_pipeline": self.best_result["genotype"],
        }

        json_path = os.path.join(save_dir, "best_results.json")
        with open(json_path, 'w') as f:
            json.dump(best_results, f, indent=4)
        print(f"Best results saved to: {json_path}", flush=True)

        # save the full provenance archive of every evaluated pipeline
        self.save_archive(save_dir)

        return

    def save_archive(self, save_dir: str) -> None:
        """
        Save the full provenance archive (every evaluated pipeline, with generation,
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
