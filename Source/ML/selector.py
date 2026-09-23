##########################################################################################
#
# Parameter-space classes for the feature-selection stage of the scikit-learn pipeline
# (see Source/ML/sklearn_pipeline_structure.md, stage 3).
#
# Each selector operator is its own class derived from ModelParams, mirroring how each
# classifier is defined in Source/ML/classifiers.py. Random sampling, mutation, and TPE
# encoding are inherited from ModelParams; every operator only defines its own parameter
# space (__init__), how a genotype maps to scikit-learn kwargs (eval_parameters), and its
# operator identifier (get_model_type).
#
# The selector operates on the CURRENT feature representation (i.e. after the transformer
# stage), so it may be selecting among transformed features (PCA components, polynomial
# terms, ...) rather than the original inputs -- see the pipeline spec.
#
# "passthrough" is represented at the stage level by the literal string (see the SELECTORS
# registry at the bottom), not as a class.
#
##########################################################################################

from typeguard import typechecked
from typing import Dict, Any, Optional

from Source.Base.model_param_space import ModelParams, DataContext, IntParam, FloatParam


@typechecked
class SelectFweParams(ModelParams):
    """
    sklearn.feature_selection.SelectFwe -- keeps features passing a family-wise error-rate
    threshold on a univariate statistical test.
    """
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={
            'alpha': FloatParam(bounds=(1e-4, 0.05), type='float', log=True),
        })

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Maps the genotype to scikit-learn SelectFwe kwargs. """
        return {'alpha': model_params['alpha']}

    def get_model_type(self) -> str:
        return "SelectFwe"


@typechecked
class SelectPercentileParams(ModelParams):
    """
    sklearn.feature_selection.SelectPercentile -- keeps the top ``percentile`` percent of
    features by univariate score.

    The upper bound is 100 (SelectPercentile's hard maximum). Random sampling uses
    rng.integers with an EXCLUSIVE upper bound, so it reaches at most 99; the exact value
    100 is still attainable via the shift mutation, which clips to bounds[1] inclusively.
    """
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={
            'percentile': IntParam(bounds=(1, 100), type='int'),
        })

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Maps the genotype to scikit-learn SelectPercentile kwargs. """
        return {'percentile': model_params['percentile']}

    def get_model_type(self) -> str:
        return "SelectPercentile"


@typechecked
class VarianceThresholdParams(ModelParams):
    """
    sklearn.feature_selection.VarianceThreshold -- removes features with variance below a
    threshold.

    The meaningful range of ``threshold`` depends on the upstream scaler: at 0.0 only
    constant features are dropped, and the (0.0, 0.5) range is well-matched to roughly
    unit-scale features (e.g. following StandardScaler/MinMaxScaler). This coupling is left
    intentional -- since we evolve full pipelines, a threshold paired with a scaler that
    leaves features at a mismatched scale (e.g. wiping out all features, or dropping none)
    simply produces a poor classifier and is selected against on fitness. No constraint or
    data-dependent rescaling is imposed here.
    """
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={
            'threshold': FloatParam(bounds=(0.0 + offset, 0.5), type='float', log=False),
        })

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Maps the genotype to scikit-learn VarianceThreshold kwargs. """
        return {'threshold': model_params['threshold']}

    def get_model_type(self) -> str:
        return "VarianceThreshold"


# Operators available at the feature-selection stage. "passthrough" is represented by the
# literal string (no ModelParams class) since it has no hyperparameters or estimator.
SELECTORS = (
    SelectFweParams,
    SelectPercentileParams,
    VarianceThresholdParams,
)
