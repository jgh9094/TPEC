##########################################################################################
#
# Parameter-space classes for the feature-transformation stage of the scikit-learn
# pipeline (see Source/ML/sklearn_pipeline_structure.md, stage 2).
#
# Each transformer operator is its own class derived from ModelParams, mirroring how each
# classifier is defined in Source/ML/classifiers.py. Random sampling, mutation, and TPE
# encoding are inherited from ModelParams; every operator only defines its own parameter
# space (__init__), how a genotype maps to scikit-learn kwargs (eval_parameters), and its
# operator identifier (get_model_type).
#
# "passthrough" is represented at the stage level by the literal string (see the
# TRANSFORMERS registry at the bottom), not as a class.
#
# Boolean-sentinel convention:
#   The inherited sampler supports only 'int', 'float', and 'cat' parameter types, so
#   boolean hyperparameters are encoded as categorical ('True', 'False') strings and
#   converted back to real bools in eval_parameters (same pattern as the 'None' -> None
#   class_weight sentinel in classifiers.py).
#
# Dimensionality/data coupling:
#   Several transformers below have parameters whose valid upper bound depends on the data
#   (n_features or n_samples). That dimension is supplied via the DataContext passed to
#   __init__, so each such parameter sets its upper bound directly from ctx.n_features or
#   ctx.n_samples at construction time. Each is flagged individually below.
#
##########################################################################################

import numpy as np
from typeguard import typechecked
from typing import Dict, Any, Optional

from Source.Base.model_param_space import ModelParams, DataContext, IntParam, FloatParam, CatParam


@typechecked
class KBinsDiscretizerParams(ModelParams):
    """
    sklearn.preprocessing.KBinsDiscretizer -- discretizes continuous features into bins.
    """
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={
            'n_bins': IntParam(bounds=(2, 100), type='int'),
            'encode': CatParam(bounds=('onehot-dense', 'ordinal'), type='cat'),
            'strategy': CatParam(bounds=('uniform', 'quantile', 'kmeans'), type='cat'),
        })

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Maps the genotype to scikit-learn KBinsDiscretizer kwargs. """
        return {'n_bins': model_params['n_bins'],
                'encode': model_params['encode'],
                'strategy': model_params['strategy'],
                'random_state': random_state}

    def get_model_type(self) -> str:
        return "KBinsDiscretizer"


@typechecked
class BinarizerParams(ModelParams):
    """
    sklearn.preprocessing.Binarizer -- thresholds features into binary 0/1 values.

    ``threshold`` is compared directly against feature values, so its meaningful range
    depends on the upstream scaler; the (0.0, 1.0) range is well-matched to roughly
    [0, 1]-scaled input (e.g. following MinMaxScaler). This coupling is left intentional --
    since we evolve full pipelines, a threshold paired with a scaler that puts features on a
    mismatched scale (so the Binarizer maps everything to all-0 or all-1) simply yields a
    poor classifier and is selected against on fitness. No constraint or data-dependent
    rescaling is imposed here.

    Sampling is linear (log=False): the threshold is an additive position on a bounded axis,
    so linear sampling gives even coverage of the meaningful mid-range, whereas log-scaling
    would crowd it toward ~0 where the Binarizer maps nearly everything to 1.
    """
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={
            'threshold': FloatParam(bounds=(0.0 + offset, 1.0 - offset), type='float', log=False),
        })

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Maps the genotype to scikit-learn Binarizer kwargs. """
        return {'threshold': model_params['threshold']}

    def get_model_type(self) -> str:
        return "Binarizer"


@typechecked
class PolynomialFeaturesParams(ModelParams):
    """
    sklearn.preprocessing.PolynomialFeatures -- generates polynomial and interaction terms.
    """
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={
            'degree': IntParam(bounds=(2, 3), type='int'),
            'interaction_only': CatParam(bounds=('True', 'False'), type='cat'),
            'include_bias': CatParam(bounds=('True', 'False'), type='cat'),
        })

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Maps the genotype to scikit-learn PolynomialFeatures kwargs. """
        return {'degree': model_params['degree'],
                'interaction_only': model_params['interaction_only'] == 'True',
                'include_bias': model_params['include_bias'] == 'True'}

    def get_model_type(self) -> str:
        return "PolynomialFeatures"


@typechecked
class PCAParams(ModelParams):
    """
    sklearn.decomposition.PCA -- projects data onto orthogonal principal components.

    ``n_components`` is expressed as a float in (0, 1): the fraction of variance to retain.
    This form is dimensionality-independent (unlike an integer component count), so it is
    safe to sample without knowing n_features.
    """
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={
            'n_components': FloatParam(bounds=(0.5, 1.0 - offset), type='float', log=False),
            'whiten': CatParam(bounds=('True', 'False'), type='cat'),
        })

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Maps the genotype to scikit-learn PCA kwargs. """
        return {'n_components': model_params['n_components'],
                'whiten': model_params['whiten'] == 'True',
                'random_state': random_state}

    def get_model_type(self) -> str:
        return "PCA"


@typechecked
class FastICAParams(ModelParams):
    """
    sklearn.decomposition.FastICA -- decomposes data into independent components.

    ``n_components`` must be <= n_features, so its upper bound is set directly to
    ctx.n_features at construction.

    FOLLOW-UP:
      - FastICA frequently emits convergence warnings; ``max_iter`` is searched to mitigate
        this but a hard cap on runtime may still be desirable.
      - ``whiten`` is left at its scikit-learn default ('unit-variance' in recent versions),
        which is required when ``n_components`` is set.
    """
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={
            'n_components': IntParam(bounds=(1, ctx.n_features), type='int'),
            'algorithm': CatParam(bounds=('parallel', 'deflation'), type='cat'),
            'fun': CatParam(bounds=('logcosh', 'exp', 'cube'), type='cat'),
            'max_iter': IntParam(bounds=(200, 1001), type='int'),
        })

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Maps the genotype to scikit-learn FastICA kwargs. """
        return {'n_components': model_params['n_components'],
                'algorithm': model_params['algorithm'],
                'fun': model_params['fun'],
                'max_iter': model_params['max_iter'],
                'random_state': random_state}

    def get_model_type(self) -> str:
        return "FastICA"


@typechecked
class FeatureAgglomerationParams(ModelParams):
    """
    sklearn.cluster.FeatureAgglomeration -- hierarchically clusters and merges features.

    ``linkage`` and ``metric`` are both searched, but scikit-learn imposes a conditional
    constraint the flat parameter space cannot express on its own: 'ward' linkage is only
    valid with the euclidean metric. The two genes are therefore sampled independently and
    then repaired -- see ``_repair_ward_conflict`` -- so that an invalid (ward, non-euclidean)
    pairing is never emitted by random generation or either mutation operator. The random/
    mutation methods are overridden solely to append this repair step; the underlying
    sampling and shifting logic is still inherited from ModelParams.

    The repair is symmetric: when the conflict arises, a coin flip decides whether to keep
    'ward' and snap the metric to euclidean, or keep the metric and resample linkage from the
    non-ward options. This avoids systematically biasing either gene toward one value.

    ``n_clusters`` must be <= n_features (features are merged into clusters), so its upper
    bound is set directly to ctx.n_features at construction.
    """

    # linkage/metric options and the subset of each that is valid under the 'ward' constraint.
    _NON_WARD_LINKAGES = ('complete', 'average', 'single')
    _WARD_METRIC = 'euclidean'

    # ``pooling_func`` is a callable, which the flat/categorical parameter space cannot hold
    # directly, so it is encoded as a categorical name and mapped back to the numpy reduction
    # in eval_parameters (same sentinel pattern as the boolean/None genes elsewhere).
    _POOLING_FUNCS = {'mean': np.mean, 'median': np.median, 'max': np.max, 'min': np.min}

    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={
            'n_clusters': IntParam(bounds=(2, ctx.n_features), type='int'),
            'linkage': CatParam(bounds=('ward', 'complete', 'average', 'single'), type='cat'),
            'metric': CatParam(bounds=('euclidean', 'l1', 'l2', 'manhattan', 'cosine'), type='cat'),
            'pooling_func': CatParam(bounds=('mean', 'median', 'max', 'min'), type='cat'),
        })

    def _repair_ward_conflict(self, model_params: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        """
        Enforces scikit-learn's "ward linkage requires the euclidean metric" constraint in
        place. Only the invalid (ward, non-euclidean) pairing is touched; every other
        combination is already valid and left unchanged. When the conflict is present, a coin
        flip picks which gene to correct so neither is systematically biased.
        """
        if model_params['linkage'] == 'ward' and model_params['metric'] != self._WARD_METRIC:
            if rng.uniform() < 0.5:
                model_params['metric'] = self._WARD_METRIC
            else:
                model_params['linkage'] = self.pick_categorical_parameter(self._NON_WARD_LINKAGES, rng)
        return model_params

    def generate_random_parameters(self, rng: np.random.Generator) -> Dict[str, Any]:
        """ Samples each gene independently (inherited), then repairs the ward/metric pair. """
        return self._repair_ward_conflict(super().generate_random_parameters(rng), rng)

    def mutate_parameters_shift(self, model_params: Dict[str, Any], var: float, mut_rate: float, rng: np.random.Generator) -> Dict[str, Any]:
        """ Shift-mutates (inherited), then repairs any ward/metric conflict the shift created. """
        return self._repair_ward_conflict(super().mutate_parameters_shift(model_params, var, mut_rate, rng), rng)

    def mutate_parameters_random(self, model_params: Dict[str, Any], mut_rate: float, rng: np.random.Generator) -> Dict[str, Any]:
        """ Resample-mutates (inherited), then repairs any ward/metric conflict the resample created. """
        return self._repair_ward_conflict(super().mutate_parameters_random(model_params, mut_rate, rng), rng)

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Maps the genotype to scikit-learn FeatureAgglomeration kwargs. """
        return {'n_clusters': model_params['n_clusters'],
                'linkage': model_params['linkage'],
                'metric': model_params['metric'],
                'pooling_func': self._POOLING_FUNCS[model_params['pooling_func']]}

    def get_model_type(self) -> str:
        return "FeatureAgglomeration"


@typechecked
class NystroemParams(ModelParams):
    """
    sklearn.kernel_approximation.Nystroem -- approximate kernel feature map.

    ``n_components`` controls the output dimensionality; scikit-learn internally clips it to
    n_samples (the landmark points are drawn from the training rows), so its upper bound is
    set directly to ctx.n_samples here to avoid that warning and wasted components. ``gamma`` and
    ``degree`` are passed for every kernel; scikit-learn ignores them for kernels that do not
    use them (e.g. linear, cosine), so this is harmless.

    The ``chi2`` and ``additive_chi2`` kernels require strictly non-negative input and raise
    ValueError at fit time on signed data (i.e. anything other than a non-negative upstream
    scaler such as MinMaxScaler). This coupling is left intentional -- since we evolve full
    pipelines, a chi2 kernel paired with a scaler that produces negative features simply fails
    to fit; CASH catches the exception and assigns a fitness of 0.0, so the combination is
    selected against rather than being forbidden by a constraint here.

    FOLLOW-UP:
      - ``coef0`` (used by poly/sigmoid) is not searched; add it if those kernels matter.
    """
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={
            'kernel': CatParam(bounds=('rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'), type='cat'),
            'gamma': FloatParam(bounds=(0.0 + offset, 1.0 - offset), type='float', log=True),
            'degree': IntParam(bounds=(1, 3), type='int'),
            'n_components': IntParam(bounds=(1, ctx.n_samples), type='int'),
        })

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Maps the genotype to scikit-learn Nystroem kwargs. """
        return {'kernel': model_params['kernel'],
                'gamma': model_params['gamma'],
                'degree': model_params['degree'],
                'n_components': model_params['n_components'],
                'random_state': random_state}

    def get_model_type(self) -> str:
        return "Nystroem"


@typechecked
class RBFSamplerParams(ModelParams):
    """
    sklearn.kernel_approximation.RBFSampler -- approximates an RBF kernel via random
    Fourier features. ``n_components`` controls the output dimensionality directly.
    """
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={
            'gamma': FloatParam(bounds=(0.0 + offset, 1.0 - offset), type='float', log=True),
            'n_components': IntParam(bounds=(1, ctx.n_features), type='int'),
        })

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Maps the genotype to scikit-learn RBFSampler kwargs. """
        return {'gamma': model_params['gamma'],
                'n_components': model_params['n_components'],
                'random_state': random_state}

    def get_model_type(self) -> str:
        return "RBFSampler"


@typechecked
class QuantileTransformerParams(ModelParams):
    """
    sklearn.preprocessing.QuantileTransformer -- maps features to a uniform or normal
    distribution using estimated quantiles.

    ``n_quantiles`` must be <= n_samples (scikit-learn clips it with a warning otherwise), so
    its upper bound is set directly to ctx.n_samples here. Note ``ctx.n_samples`` is the full
    training-set size; within cross-validation each fold sees slightly fewer rows, so a value
    equal to the cap can still trigger scikit-learn's internal clip on the smaller fold -- that
    only costs a warning, not a failure.
    """
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={
            'n_quantiles': IntParam(bounds=(10, ctx.n_samples), type='int'),
            'output_distribution': CatParam(bounds=('uniform', 'normal'), type='cat'),
        })

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Maps the genotype to scikit-learn QuantileTransformer kwargs. """
        return {'n_quantiles': model_params['n_quantiles'],
                'output_distribution': model_params['output_distribution'],
                'random_state': random_state}

    def get_model_type(self) -> str:
        return "QuantileTransformer"


@typechecked
class PowerTransformerParams(ModelParams):
    """
    sklearn.preprocessing.PowerTransformer -- makes features more Gaussian-like.

    ``method`` is fixed to 'yeo-johnson' because 'box-cox' requires strictly positive input,
    which cannot be guaranteed after arbitrary upstream scaling/transformation.

    FOLLOW-UP: add 'box-cox' to ``method`` only if the pipeline guarantees strictly positive
    features reaching this stage (the flat parameter space cannot express that conditional
    constraint).
    """
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={
            'method': CatParam(bounds=('yeo-johnson',), type='cat'),
            'standardize': CatParam(bounds=('True', 'False'), type='cat'),
        })

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Maps the genotype to scikit-learn PowerTransformer kwargs. """
        return {'method': model_params['method'],
                'standardize': model_params['standardize'] == 'True'}

    def get_model_type(self) -> str:
        return "PowerTransformer"


# Operators available at the feature-transformation stage. "passthrough" is represented by
# the literal string (no ModelParams class) since it has no hyperparameters or estimator.
TRANSFORMERS = (
    KBinsDiscretizerParams,
    BinarizerParams,
    PolynomialFeaturesParams,
    PCAParams,
    FastICAParams,
    FeatureAgglomerationParams,
    NystroemParams,
    RBFSamplerParams,
    QuantileTransformerParams,
    PowerTransformerParams,
)
