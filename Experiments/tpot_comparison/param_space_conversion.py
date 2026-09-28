"""Translate the full Source/ML CASH space into TPOT search-space nodes."""

from functools import partial
from typing import Dict, Optional, Type, Union

import tpot
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from sklearn.cluster import FeatureAgglomeration
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import FastICA, PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectFwe, SelectPercentile, VarianceThreshold
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import (
    Binarizer,
    KBinsDiscretizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC, SVR
from tpot.builtin_modules import Passthrough

from Source.Base.model_param_space import (
    BoolParam,
    CatParam,
    DataContext,
    FloatParam,
    IntParam,
    ModelParams,
)
from Source.ML.classifiers import CLASSIFIERS
from Source.ML.regressor import REGRESSORS
from Source.ML.scaler import SCALERS
from Source.ML.selector import SELECTORS
from Source.ML.transformer import TRANSFORMERS


ParamSpec = Union[IntParam, FloatParam, CatParam, BoolParam]
ParamSpace = Dict[str, ParamSpec]
ModelParamsType = Type[ModelParams]


PREPROCESSING_CLASSES = {
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
    "SelectFwe": SelectFwe,
    "SelectPercentile": SelectPercentile,
    "VarianceThreshold": VarianceThreshold,
}

CLASSIFIER_CLASSES = {
    "RF": RandomForestClassifier,
    "ET": ExtraTreesClassifier,
    "KSVC": SVC,
    "GB": GradientBoostingClassifier,
    "KNN": KNeighborsClassifier,
    "MLP": MLPClassifier,
}

REGRESSOR_CLASSES = {
    "RF": RandomForestRegressor,
    "ET": ExtraTreesRegressor,
    "SVR": SVR,
    "GB": GradientBoostingRegressor,
    "KNN": KNeighborsRegressor,
    "MLP": MLPRegressor,
}


def _column_scaler(estimator, scale_columns: Optional[tuple[int, ...]]):
    if scale_columns is None:
        return estimator
    return ColumnTransformer(
        [("scale", estimator, list(scale_columns))],
        remainder="passthrough",
    )


# Separate top-level factories deliberately give TPOT distinct operator identities. If every
# scaler used ColumnTransformer directly, TPOT could cross hyperparameters between incompatible
# scaler spaces because EstimatorNode identifies compatibility by its method object.
def min_max_column_scaler(scale_columns, **kwargs):
    return _column_scaler(MinMaxScaler(**kwargs), scale_columns)


def robust_column_scaler(scale_columns, **kwargs):
    return _column_scaler(RobustScaler(**kwargs), scale_columns)


def standard_column_scaler(scale_columns, **kwargs):
    return _column_scaler(StandardScaler(**kwargs), scale_columns)


def max_abs_column_scaler(scale_columns, **kwargs):
    return _column_scaler(MaxAbsScaler(**kwargs), scale_columns)


def normalizer_column_scaler(scale_columns, **kwargs):
    return _column_scaler(Normalizer(**kwargs), scale_columns)


SCALER_FACTORIES = {
    "MinMaxScaler": min_max_column_scaler,
    "RobustScaler": robust_column_scaler,
    "StandardScaler": standard_column_scaler,
    "MaxAbsScaler": max_abs_column_scaler,
    "Normalizer": normalizer_column_scaler,
}


def convert_param_space(param_space: ParamSpace) -> ConfigurationSpace:
    """Convert a project parameter space while preserving log distributions."""
    config_space = ConfigurationSpace()
    for param_name, param_spec in param_space.items():
        param_type = param_spec["type"]
        bounds = param_spec["bounds"]

        if param_type == "int":
            hyperparameter = Integer(param_name, bounds)
        elif param_type == "float":
            hyperparameter = Float(
                param_name,
                bounds,
                log=bool(param_spec.get("log", False)),
            )
        elif param_type == "cat":
            hyperparameter = Categorical(param_name, list(bounds))
        elif param_type == "bool":
            hyperparameter = Categorical(param_name, [True, False])
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")

        config_space.add(hyperparameter)

    return config_space


def parse_operator_parameters(
    params,
    *,
    model_param_class: ModelParamsType,
    data_context: DataContext,
    seed: int,
):
    """Apply the same Source/ML eval-parameter conversion used by CASH."""
    model_params = model_param_class(data_context)
    component_name = model_params.get_model_type()
    parsed = model_params.eval_parameters(dict(params), random_state=seed)

    if component_name == "KSVC":
        parsed["probability"] = True
    elif component_name == "MLP":
        parsed["hidden_layer_sizes"] = tuple(
            parsed.pop(f"layer_{index}") for index in range(1, 6)
        )

    # CASH's sklearn estimators use their default single-process behavior. TPOT handles
    # candidate-level parallelism, so make that behavior explicit for estimators that expose it.
    if component_name in {"RF", "ET", "KNN"}:
        parsed["n_jobs"] = 1

    return parsed


def parse_scaler_parameters(
    params,
    *,
    model_param_class: ModelParamsType,
    data_context: DataContext,
    seed: int,
    scale_columns: Optional[tuple[int, ...]],
):
    parsed = parse_operator_parameters(
        params,
        model_param_class=model_param_class,
        data_context=data_context,
        seed=seed,
    )
    parsed["scale_columns"] = scale_columns
    return parsed


def make_choice_space(
    registry,
    estimator_classes,
    data_context: DataContext,
    seed: int,
    allow_passthrough: bool,
):
    nodes = []
    for model_param_class in registry:
        model_params = model_param_class(data_context)
        component_name = model_params.get_model_type()
        nodes.append(
            tpot.search_spaces.nodes.EstimatorNode(
                method=estimator_classes[component_name],
                space=convert_param_space(model_params.param_space),
                hyperparameter_parser=partial(
                    parse_operator_parameters,
                    model_param_class=model_param_class,
                    data_context=data_context,
                    seed=seed,
                ),
            )
        )
    if allow_passthrough:
        nodes.append(tpot.search_spaces.nodes.EstimatorNode(Passthrough, {}))
    return tpot.search_spaces.pipelines.ChoicePipeline(search_spaces=nodes)


def make_scaler_choice_space(
    data_context: DataContext,
    seed: int,
    scale_columns: Optional[tuple[int, ...]],
):
    nodes = []
    for model_param_class in SCALERS:
        model_params = model_param_class(data_context)
        component_name = model_params.get_model_type()
        nodes.append(
            tpot.search_spaces.nodes.EstimatorNode(
                method=SCALER_FACTORIES[component_name],
                space=convert_param_space(model_params.param_space),
                hyperparameter_parser=partial(
                    parse_scaler_parameters,
                    model_param_class=model_param_class,
                    data_context=data_context,
                    seed=seed,
                    scale_columns=scale_columns,
                ),
            )
        )
    nodes.append(tpot.search_spaces.nodes.EstimatorNode(Passthrough, {}))
    return tpot.search_spaces.pipelines.ChoicePipeline(search_spaces=nodes)


def generate_tpot_search_space(
    data_context: DataContext,
    seed: int,
    classification: bool,
    scale_columns: Optional[tuple[int, ...]],
):
    """Build CASH's scaler -> transformer -> selector -> predictor pipeline."""
    predictor_registry = CLASSIFIERS if classification else REGRESSORS
    predictor_classes = CLASSIFIER_CLASSES if classification else REGRESSOR_CLASSES

    stages = [
        make_scaler_choice_space(data_context, seed, scale_columns),
        make_choice_space(
            TRANSFORMERS,
            PREPROCESSING_CLASSES,
            data_context,
            seed,
            allow_passthrough=True,
        ),
        make_choice_space(
            SELECTORS,
            PREPROCESSING_CLASSES,
            data_context,
            seed,
            allow_passthrough=True,
        ),
        make_choice_space(
            predictor_registry,
            predictor_classes,
            data_context,
            seed,
            allow_passthrough=False,
        ),
    ]
    return tpot.search_spaces.pipelines.SequentialPipeline(search_spaces=stages)
