"""Translate the project's classifier spaces into TPOT/ConfigSpace nodes."""

from functools import partial
from typing import Dict, Type, Union

import tpot
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from Source.Base.model_param_space import (
    BoolParam,
    CatParam,
    DataContext,
    FloatParam,
    IntParam,
    ModelParams,
)
from Source.ML.classifiers import (
    ExtraTreesParams,
    GradientBoostParams,
    KNeighborsClassifierParams,
    KernelSVCParams,
    MLPClassifierParams,
    RandomForestParams,
)


ParamSpec = Union[IntParam, FloatParam, CatParam, BoolParam]
ParamSpace = Dict[str, ParamSpec]
ModelParamsType = Type[ModelParams]

MODEL_PARAM_CLASSES = (
    RandomForestParams,
    ExtraTreesParams,
    KernelSVCParams,
    GradientBoostParams,
    KNeighborsClassifierParams,
    MLPClassifierParams,
)

PARAM_CLASS_TO_ESTIMATOR = {
    RandomForestParams: RandomForestClassifier,
    ExtraTreesParams: ExtraTreesClassifier,
    KernelSVCParams: SVC,
    GradientBoostParams: GradientBoostingClassifier,
    KNeighborsClassifierParams: KNeighborsClassifier,
    MLPClassifierParams: MLPClassifier,
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


def parse_estimator_parameters(
    params,
    *,
    model_param_class: ModelParamsType,
    data_context: DataContext,
    seed: int,
):
    """Map a TPOT configuration to the sklearn arguments used by this project."""
    model_params = model_param_class(data_context)
    parsed = model_params.eval_parameters(dict(params), random_state=seed)

    if model_param_class is KernelSVCParams:
        parsed["probability"] = True
    elif model_param_class is MLPClassifierParams:
        parsed["hidden_layer_sizes"] = tuple(
            parsed.pop(f"layer_{index}") for index in range(1, 6)
        )

    # TPOT parallelizes candidate evaluation. Keeping estimator-level parallelism at
    # one avoids multiplying the requested SLURM core count for tree/KNN estimators.
    if model_param_class in {
        RandomForestParams,
        ExtraTreesParams,
        KNeighborsClassifierParams,
    }:
        parsed["n_jobs"] = 1

    return parsed


def generate_tpot_search_space(data_context: DataContext, seed: int):
    """Build a classifier-choice space matching the six current CASH families."""
    nodes = []
    for model_param_class in MODEL_PARAM_CLASSES:
        model_params = model_param_class(data_context)
        parser = partial(
            parse_estimator_parameters,
            model_param_class=model_param_class,
            data_context=data_context,
            seed=seed,
        )
        nodes.append(
            tpot.search_spaces.nodes.EstimatorNode(
                method=PARAM_CLASS_TO_ESTIMATOR[model_param_class],
                space=convert_param_space(model_params.param_space),
                hyperparameter_parser=parser,
            )
        )

    return tpot.search_spaces.pipelines.ChoicePipeline(search_spaces=nodes)
