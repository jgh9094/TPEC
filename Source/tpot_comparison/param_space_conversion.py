from typing import Dict, Union

import tpot
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from Source.Base.model_param_space import (
    BoolParam,
    CatParam,
    FloatParam,
    GradientBoostParams,
    IntParam,
    KNeighborsClassifierParams,
    KernelSVCParams,
    MLPClassifierParams,
    RandomForestParams,
)


map_param_space_to_ml = {
    RandomForestParams: RandomForestClassifier,
    KernelSVCParams: SVC,
    GradientBoostParams: GradientBoostingClassifier,
    KNeighborsClassifierParams: KNeighborsClassifier,
    MLPClassifierParams: MLPClassifier,
}


ParamSpec = Union[IntParam, FloatParam, CatParam, BoolParam]
ParamSpace = Dict[str, ParamSpec]


def convert_param_space(param_space: ParamSpace) -> ConfigurationSpace:
    """Convert a project parameter space to a ConfigSpace object."""
    cs = ConfigurationSpace()
    for param_name, param_spec in param_space.items():
        param_type = param_spec["type"]
        bounds = param_spec["bounds"]

        if param_type == "int":
            cs.add(Integer(param_name, bounds))
        elif param_type == "float":
            cs.add(Float(param_name, bounds))
        elif param_type == "cat":
            cs.add(Categorical(param_name, list(bounds)))
        elif param_type == "bool":
            cs.add(Categorical(param_name, [True, False]))
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")

    return cs


def preprocess_class_weight_params(params):
    """Convert the project's categorical sentinel to sklearn's None value."""
    params = dict(params)
    if params.get("class_weight") == "None":
        params["class_weight"] = None
    return params


def preprocess_svc_params(params):
    """Enable probabilities for the project's AUC/log-loss objective."""
    params = preprocess_class_weight_params(params)
    params["probability"] = True
    return params


def preprocess_mlp_params(params):
    """Convert the five layer genes to sklearn's hidden-layer tuple."""
    params = dict(params)
    params["hidden_layer_sizes"] = tuple(params.pop(f"layer_{i}") for i in range(1, 6))
    return params


def generate_tpot_search_space(classes: int, num_cpus: int):
    """Build TPOT's choice space from the model families used by current CASH."""
    del num_cpus  # TPOT controls parallelism at the estimator level.
    model_param_classes = [
        RandomForestParams,
        KernelSVCParams,
        GradientBoostParams,
        KNeighborsClassifierParams,
        MLPClassifierParams,
    ]
    parsers = {
        RandomForestParams: preprocess_class_weight_params,
        KernelSVCParams: preprocess_svc_params,
        MLPClassifierParams: preprocess_mlp_params,
    }
    nodes = []

    for model_param_class in model_param_classes:
        if model_param_class is GradientBoostParams:
            model_param_space = model_param_class(binary_class=classes == 2)
        else:
            model_param_space = model_param_class()

        node_kwargs = {
            "method": map_param_space_to_ml[model_param_class],
            "space": convert_param_space(model_param_space.param_space),
        }
        if model_param_class in parsers:
            node_kwargs["hyperparameter_parser"] = parsers[model_param_class]
        nodes.append(tpot.search_spaces.nodes.EstimatorNode(**node_kwargs))

    return tpot.search_spaces.pipelines.ChoicePipeline(search_spaces=nodes)
