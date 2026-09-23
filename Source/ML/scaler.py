##########################################################################################
#
# Parameter-space classes for the feature-scaling stage of the scikit-learn pipeline
# (see Source/ML/sklearn_pipeline_structure.md, stage 1).
#
# Each scaler operator is its own class derived from ModelParams, mirroring how each
# classifier is defined in Source/ML/classifiers.py. Random sampling, mutation, and TPE
# encoding are inherited from ModelParams; every operator only defines its own parameter
# space (__init__), how a genotype maps to scikit-learn kwargs (eval_parameters), and its
# operator identifier (get_model_type).
#
# The "passthrough" option required by the pipeline spec is NOT modeled as a class here:
# it carries no hyperparameters and no estimator kwargs, so it is represented at the stage
# level by the literal string "passthrough" (see the SCALERS registry at the bottom).
#
# Boolean-sentinel convention:
#   The inherited sampler only supports the 'int', 'float', and 'cat' parameter types, so
#   boolean hyperparameters are encoded as categorical ('True', 'False') strings and
#   converted back to real bools in eval_parameters -- the same pattern classifiers.py uses
#   for the 'None' -> None class_weight sentinel.
#
##########################################################################################

from typeguard import typechecked
from typing import Dict, Any, Optional

from Source.Base.model_param_space import ModelParams, DataContext, CatParam, FloatParam


@typechecked
class MinMaxScalerParams(ModelParams):
    """
    sklearn.preprocessing.MinMaxScaler -- scales each feature to the [0, 1] range.

    No hyperparameters are searched. Its only tunables (``feature_range`` and ``clip``)
    are low-impact and, in the case of ``feature_range``, a tuple that the flat parameter
    space cannot represent.
    """
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={})

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Maps the genotype to scikit-learn MinMaxScaler kwargs. """
        return {}

    def get_model_type(self) -> str:
        return "MinMaxScaler"


@typechecked
class RobustScalerParams(ModelParams):
    """
    sklearn.preprocessing.RobustScaler -- scales features using median and IQR.

    ``quantile_range`` (default (25.0, 75.0)) is searched by evolving its two endpoints as
    separate float genes (``q_min``, ``q_max``) and recombining them into the (q_min, q_max)
    tuple in eval_parameters -- the "split into two float genes" approach, since the flat
    parameter space cannot represent a tuple directly. The genes live directly on the 0-100
    percentile scale scikit-learn expects; the disjoint bounds keep q_min < q_max with a
    meaningful IQR between them. Sampling is linear (log=False): percentile endpoints are
    additive positions on a bounded axis, so linear sampling gives even coverage around the
    default (25, 75), whereas log-scaling would crowd q_min toward the extreme lower tail.
    """
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={
            'with_centering': CatParam(bounds=('True', 'False'), type='cat'),
            'with_scaling': CatParam(bounds=('True', 'False'), type='cat'),
            'q_min': FloatParam(bounds=(0.0 + offset, 30.0 - offset), type='float', log=False),
            'q_max': FloatParam(bounds=(70.0, 100.0 - offset), type='float', log=False),
        })

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Maps the genotype to scikit-learn RobustScaler kwargs. """
        return {'with_centering': model_params['with_centering'] == 'True',
                'with_scaling': model_params['with_scaling'] == 'True',
                'quantile_range': (model_params['q_min'], model_params['q_max'])}

    def get_model_type(self) -> str:
        return "RobustScaler"


@typechecked
class StandardScalerParams(ModelParams):
    """
    sklearn.preprocessing.StandardScaler -- removes the mean and scales to unit variance.
    """
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={
            'with_mean': CatParam(bounds=('True', 'False'), type='cat'),
            'with_std': CatParam(bounds=('True', 'False'), type='cat'),
        })

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Maps the genotype to scikit-learn StandardScaler kwargs. """
        return {'with_mean': model_params['with_mean'] == 'True',
                'with_std': model_params['with_std'] == 'True'}

    def get_model_type(self) -> str:
        return "StandardScaler"


@typechecked
class MaxAbsScalerParams(ModelParams):
    """
    sklearn.preprocessing.MaxAbsScaler -- scales each feature by its maximum absolute value.

    No hyperparameters are searched (MaxAbsScaler exposes none worth tuning).
    """
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={})

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Maps the genotype to scikit-learn MaxAbsScaler kwargs. """
        return {}

    def get_model_type(self) -> str:
        return "MaxAbsScaler"


@typechecked
class NormalizerParams(ModelParams):
    """
    sklearn.preprocessing.Normalizer -- normalizes each sample (row) to unit norm.

    NOTE: unlike the other scalers, Normalizer operates per-sample rather than per-feature
    (see the pipeline spec).
    """
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={
            'norm': CatParam(bounds=('l1', 'l2', 'max'), type='cat'),
        })

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Maps the genotype to scikit-learn Normalizer kwargs. """
        return {'norm': model_params['norm']}

    def get_model_type(self) -> str:
        return "Normalizer"


# Operators available at the feature-scaling stage. "passthrough" is represented by the
# literal string (no ModelParams class) since it has no hyperparameters or estimator.
SCALERS = (
    MinMaxScalerParams,
    RobustScalerParams,
    StandardScalerParams,
    MaxAbsScalerParams,
    NormalizerParams,
)
