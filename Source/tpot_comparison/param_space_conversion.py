import numpy as np
import tpot
import sklearn
from typing import Dict, Union
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical
from Source.Base.model_param_space import IntParam, FloatParam, CatParam, BoolParam
from Source.Base.model_param_space import RandomForestParams, LinearSVCParams, DecisionTreeParams, KernelSVCParams, ExtraTreesParams, GradientBoostParams, LinearSGDParams 

map_param_space_to_ml = {
    RandomForestParams: sklearn.ensemble.RandomForestClassifier,
    LinearSVCParams: sklearn.svm.LinearSVC,
    DecisionTreeParams: sklearn.tree.DecisionTreeClassifier,
    KernelSVCParams: sklearn.svm.SVC,
    ExtraTreesParams: sklearn.ensemble.ExtraTreesClassifier,
    GradientBoostParams: sklearn.ensemble.GradientBoostingClassifier,
    LinearSGDParams: sklearn.linear_model.SGDClassifier,
}


# ParamSpec can be one of IntParam, FloatParam, CatParam, and BoolParam
ParamSpec = Union[IntParam, FloatParam, CatParam, BoolParam]
# Dictionary where each key is a parameter name, and each value is exactly one of the 3 kinds of ParamSpecs
ParamSpace = Dict[str, ParamSpec] # {parameter_name: {"bounds": Tuple, "type": Literal["int", "float", "cat", "bool"]}}

def convert_param_space(param_space: ParamSpace) -> ConfigurationSpace:
    """Convert param space to ConfigurationSpace."""
    
    cs = ConfigurationSpace()
    for param_name, param_spec in param_space.items():            
        param_type = param_spec["type"]
        bounds = param_spec["bounds"]
        
        if param_type == "int":
            cs.add_hyperparameter(Integer(param_name, bounds))
        elif param_type == "float":
            cs.add_hyperparameter(Float(param_name, bounds))
        elif param_type == "cat":
            cs.add_hyperparameter(Categorical(param_name, list(bounds)))
        elif param_type == "bool":
            cs.add_hyperparameter(Categorical(param_name, [True, False]))
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")
        
    return cs

def preprocess_rf_params(params):
    """Preprocess RandomForest parameters to ensure max_samples respects bootstrap."""
    max_samples_low, max_samples_high = 0.001, 1.0  # bounds from RandomForestParams

    if params.get('bootstrap') is True:
        # If bootstrap=True and max_samples missing/None, draw a valid value in bounds
        if params.get('max_samples') is None:
            params['max_samples'] = np.random.uniform(max_samples_low, max_samples_high)
        # If already a float, leave as-is
    else:
        # bootstrap=False: set max_samples to None
        params['max_samples']= None
    
    return params

def preprocess_linear_svc_params(params):
    """Preprocess LinearSVC parameters to enforce valid penalty-loss-dual combinations."""
    # penalty='l1' only works with: loss='squared_hinge', dual=False
    if params.get('penalty') == 'l1':
        params['loss'] = 'squared_hinge'
        params['dual'] = False
    # loss='hinge' only works with: penalty='l2', dual=True
    elif params.get('loss') == 'hinge':
        params['penalty'] = 'l2'
        params['dual'] = True
    return params

def generate_tpot_search_space(classes: int, num_cpus: int) -> None:
    ml_param_space_classes = [RandomForestParams, LinearSVCParams, DecisionTreeParams, KernelSVCParams, ExtraTreesParams, GradientBoostParams, LinearSGDParams]
    nodes = []

    for ml_param_space_class in ml_param_space_classes:
        if ml_param_space_class == GradientBoostParams:
            ml_param_space = ml_param_space_class(classes=classes)   
        else:
            ml_param_space = ml_param_space_class()
        cs = convert_param_space(ml_param_space.param_space)
        
        # Special handling of hyperparameters with RandomForestClassifier and ExtraTreesClassifier
        if ml_param_space_class == RandomForestParams or ml_param_space_class == ExtraTreesParams:
            cs = convert_param_space(ml_param_space.param_space)
            
            node = tpot.search_spaces.nodes.EstimatorNode(
                method=map_param_space_to_ml[ml_param_space_class],
                space=cs,
                hyperparameter_parser=preprocess_rf_params,
            )
        # Special handling of hyperparameters with LinearSVC
        elif ml_param_space_class == LinearSVCParams:
            node = tpot.search_spaces.nodes.EstimatorNode(
                method=map_param_space_to_ml[ml_param_space_class],
                space=cs,
                hyperparameter_parser=preprocess_linear_svc_params,
            )
        else:
            node = tpot.search_spaces.nodes.EstimatorNode(
                method=map_param_space_to_ml[ml_param_space_class],
                space=cs,
            )
        nodes.append(node)

    assert len(nodes) == len(ml_param_space_classes), "No nodes were created for the TPOT search space."
    return tpot.search_spaces.pipelines.ChoicePipeline(search_spaces=nodes)

