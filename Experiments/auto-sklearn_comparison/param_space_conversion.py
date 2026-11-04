##########################################################################################
#
# TPEC TO AUTO-SKLEARN: PARAMETER SPACE CONVERSION AND COMPONENT FACTORIES
# Factories for custom classifiers, regressors, and feature preprocessors.
# Keep auto-sklearn's built-in data preprocessing (imputation, encoding, scaling), 
# ignore TPEC's scaler classes. 
# Custom transformers and selectors are alternatives in ONE feature-preprocessing stage;
# they are not applied as a transformer -> selector sequence.
# Unlike TPEC, auto-sklearn uses a fixed 3-node pipeline:
# data preprocessing -> feature preprocessing -> estimator
#
# 
# Usage:
#
# from Source.ML.classifiers import RandomForestParams
# CustomRF = make_classifier_class(
#     RandomForestClassifier, RandomForestParams, name="CustomRF",
# )
# add_classifier(CustomRF)
# 
# Create an instance of CustomRF:
# classifier = CustomRF(n_estimators=200, random_state=42)
# classifier.fit(X, y)
#
# Pass the corresponding ModelParams class from Source.ML.regressor:
# CustomRegressor = make_regressor_class(
#     RandomForestRegressor, RandomForestRegressorParams, name="CustomRFRegressor",
# )
# add_regressor(CustomRegressor)
#
# Pass a training context for dimension-dependent preprocessing spaces:
# CustomPCA = make_preprocessor_class(
#     PCA, YourPCAParams, name="CustomPCA", data_context=training_context,
#     classification=True, deterministic=False,
# )
# register_preprocessor(CustomPCA)
#
# Configure run by including custom components: 
# include={"classifier": ["CustomRF"], "feature_preprocessor": ["CustomPCA"]}
# Leave "data_preprocessor" out of include to retain the built-in data preprocessing.
#
# Or generate/register all custom transformers and selectors (not scalers):
# feature_names = register_feature_preprocessors(training_context, classification=True)
# include={"classifier": ["CustomRF"], "feature_preprocessor": feature_names}
# Add "no_preprocessing" to feature_names if skipping feature preprocessing is allowed.
##########################################################################################

import inspect

from typing import Dict, Optional, Type, Union

import numpy as np

from ConfigSpace import ConfigurationSpace
from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenEqualsClause
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from autosklearn.pipeline.components.base import (
    AutoSklearnPreprocessingAlgorithm, 
    AutoSklearnRegressionAlgorithm, 
    AutoSklearnClassificationAlgorithm
)
from autosklearn.pipeline.components.classification import add_classifier
from autosklearn.pipeline.components.regression import add_regressor
from autosklearn.pipeline.components.feature_preprocessing import add_preprocessor as add_feature_preprocessor
from autosklearn.pipeline.constants import DENSE, SIGNED_DATA, UNSIGNED_DATA, PREDICTIONS

from sklearn.base import clone, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.exceptions import NotFittedError
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import FastICA, PCA
from sklearn.feature_selection import SelectFwe, SelectPercentile, VarianceThreshold, f_regression
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.preprocessing import (
    Binarizer,
    KBinsDiscretizer,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
)
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

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
from Source.ML.selector import SELECTORS
from Source.ML.transformer import TRANSFORMERS

ParamSpec = Union[IntParam, FloatParam, CatParam, BoolParam]
ParamSpace = Dict[str, ParamSpec]
ModelParamsType = Type[ModelParams]

# copied this over from the TPOT script :) 
PREPROCESSING_CLASSES = {
    "KBinsDiscretizer": KBinsDiscretizer,
    "Binarizer": Binarizer,
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
    "PolynomialFeatures": PolynomialFeatures,
}


CLASSIFIER_CLASSES = {
    "RF": RandomForestClassifier,
    "ET": ExtraTreesClassifier,
    "KSVC": SVC,
    "GB": GradientBoostingClassifier,
    "KNN": KNeighborsClassifier,
    "MLP": MLPClassifier,
    "XGB": XGBClassifier
}

REGRESSOR_CLASSES = {
    "RF": RandomForestRegressor,
    "ET": ExtraTreesRegressor,
    "SVR": SVR,
    "GB": GradientBoostingRegressor,
    "KNN": KNeighborsRegressor,
    "MLP": MLPRegressor,
    "XGB": XGBRegressor
}

def convert_param_space(param_space: ParamSpace) -> ConfigurationSpace:
    """Translate the project's parameter definitions into older ConfigSpace API for auto-sklearn."""
    config_space = ConfigurationSpace()

    for param_name, param_spec in param_space.items():
        param_type = param_spec["type"]
        bounds = param_spec["bounds"]
        log = bool(param_spec.get("log", False))

        if param_type in {"int", "float"}:
            lower, upper = bounds

            # ConfigSpace represents fixed values as constants.
            if lower == upper:
                hyperparameter = Constant(param_name, lower)
            elif param_type == "int":
                hyperparameter = UniformIntegerHyperparameter(
                    name=param_name,
                    lower=lower,
                    upper=upper,
                    log=log,
                )
            else:
                hyperparameter = UniformFloatHyperparameter(
                    name=param_name,
                    lower=lower,
                    upper=upper,
                    log=log,
                )

        elif param_type == "cat":
            choices = list(bounds)
            if not choices:
                raise ValueError(
                    f"{param_name!r} has no categorical choices"
                )
            if len(choices) == 1:
                hyperparameter = Constant(param_name, choices[0])
            else:
                hyperparameter = CategoricalHyperparameter(
                    name=param_name,
                    choices=choices,
                )

        elif param_type == "bool":
            hyperparameter = CategoricalHyperparameter(
                name=param_name,
                choices=[True, False],
            )

        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")

        config_space.add_hyperparameter(hyperparameter)

    return config_space

def search_space(params_class: ModelParamsType, dataset_properties: dict | None,
                 data_context: Optional[DataContext] = None) -> ConfigurationSpace:
    """
    Convert the project's parameter-definition class, use 'dataset_properties' to
    distinguish binary from multiclass classification for Gradient Boost. 
    'params_class' refers to the class itself (inheriting from ModelParams).
    """
    properties = dataset_properties or {}
    # supply a context for parameter bounds that depend on dataset dimensions
    ctx = data_context if data_context is not None else DataContext(
        n_samples=1,
        n_features=1,
        # 3 means 'more than 2 classses', not actual class count
        n_classes=3 if properties.get("multiclass", False) else 2,
    )
    # create an instance of ModelParams, then convert to ConfigSpace
    model_params = params_class(ctx)
    config_space = convert_param_space(model_params.param_space)

    # auto-sklearn does not call the EA sampler's ward/metric repair,
    # so forbid ward linkage with any metric other than euclidean in ConfigSpace
    if model_params.get_model_type() == "FeatureAgglomeration":
        for metric in model_params.param_space["metric"]["bounds"]:
            if metric != "euclidean":
                config_space.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(config_space["linkage"], "ward"),
                    ForbiddenEqualsClause(config_space["metric"], metric),
                ))
    return config_space

##########################################################################################
#
# CLASSIFIER NODE FACTORY
#
##########################################################################################

class ClassifierProperties:
    """Class to encapsulate shared classifier properties and methods"""

    @classmethod
    def get_properties(cls, dataset_properties=None):
        """All classifiers share these properties"""
        return {
            "shortname": cls.__name__, "name": cls.__name__,
            "handles_regression": False, "handles_classification": True,
            "handles_multiclass": True, "handles_multilabel": False,
            "handles_multioutput": False, "is_deterministic": cls.deterministic,
            # both input and output must be tuple(iterable)
            "input": (DENSE, SIGNED_DATA, UNSIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim != 1:
            raise ValueError("These wrappers support single-output classification only")
        
        ctx = DataContext(n_samples=X.shape[0], n_features=X.shape[1],
                          n_classes=len(np.unique(y)))
        
        model_params = self.params_class(ctx)
        configured_params = {name: getattr(self, name) for name in model_params.param_space}

        # convert self.random_state into an integer seed or None for ModelParam's eval_parameters()
        # auto-sklearn supplies its own NumPy RandomState
        seed = self.random_state
        if seed is not None:
            if isinstance(seed, (int, np.integer)):
                seed = int(seed)
            else: # copy generator and draw a numeric seed
                rng_copy = np.random.RandomState()
                rng_copy.set_state(check_random_state(seed).get_state())
                seed = int(rng_copy.randint(0, 2**31 - 1))

        kwargs = model_params.eval_parameters(configured_params, random_state=seed)

        # adapt parameters to the arguments sklearn expects
        if self.estimator_class is SVC:
            kwargs["probability"] = True 
        if self.estimator_class is MLPClassifier: 
            kwargs["hidden_layer_sizes"] = tuple(
                kwargs.pop(f"layer_{i}") for i in range(1, 6)
            )
        if "n_jobs" in inspect.signature(self.estimator_class).parameters:
            kwargs["n_jobs"] = 1

        # create the actual sklearn model using prepared parameters
        estimator = self.estimator_class(**kwargs)

        if sample_weight is None:
            estimator.fit(X, y)
        elif "sample_weight" in inspect.signature(estimator.fit).parameters:
            estimator.fit(X, y, sample_weight=sample_weight)
        else:
            raise ValueError(f"{type(estimator).__name__} does not support sample_weight")
        self.estimator = estimator
        self.classes_ = np.unique(y) # store distinct class labels from training targets
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotFittedError("Call fit before predict")
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotFittedError("Call fit before predict_proba")
        return self.estimator.predict_proba(X)


def make_classifier_class(model_class: Type[ClassifierMixin], model_param_class: ModelParamsType, *, name,
                          data_context: Optional[DataContext] = None):
    """
    Return a named auto-sklearn subclass, built from one estimator class and one ModelParams class.
    'name' is keyword-only.
    """
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError("name must be a valid Python identifier")
    if name in globals():
        raise ValueError(f"A module object named {name!r} already exists")

    default_space = search_space(model_param_class, {"multiclass": False}, data_context)
    defaults = dict(default_space.get_default_configuration())
    parameter_names = tuple(defaults)
    if "random_state" in parameter_names:
        raise ValueError("random_state is an evaluation seed, not a search parameter")

    # custom classifier component
    class CustomClassifier(ClassifierProperties, AutoSklearnClassificationAlgorithm):
        params_class = model_param_class
        estimator_class = model_class
        deterministic = model_class is KNeighborsClassifier

        def __init__(self, random_state=None, **params):
            # reject unrecognized parameter names
            unknown = set(params) - set(parameter_names)
            if unknown:
                raise TypeError(f"Unknown parameters for {name}: {sorted(unknown)}")

            # assign attributes (e.g, setattr(self, "n_estimators", 200) is self.n_estimators = 200)
            # use values in 'params' if present, otherwise use ConfigSpace's defaults
            for param_name in parameter_names:
                setattr(self, param_name, params.get(param_name, defaults[param_name]))
            self.random_state = random_state
            # defer constructing the sklearn estimator until fit(), after parsing
            self.estimator = None

        def get_params(self, deep=True):
            # sklearn cannot infer all hyperparameters from **params. Expose
            # the named attributes explicitly for clone() and set_params().
            params = {key: getattr(self, key) for key in parameter_names}
            params["random_state"] = self.random_state
            return params

        @staticmethod
        def get_hyperparameter_search_space(feat_type=None, dataset_properties=None):
            return search_space(model_param_class, dataset_properties, data_context)

    # a fresh class is produced for every call; unique names prevent auto-sklearn
    # registration collisions. qualname/module/global binding support pickling.
    CustomClassifier.__name__ = name
    CustomClassifier.__qualname__ = name
    CustomClassifier.__module__ = __name__
    CustomClassifier.__doc__ = f"Factory wrapper for {model_class.__name__}."
    globals()[name] = CustomClassifier
    return CustomClassifier


##########################################################################################
#
# REGRESSOR NODE FACTORY
#
##########################################################################################

class RegressorProperties:
    """Class to encapsulate shared regressor properties and methods"""

    @classmethod
    def get_properties(cls, dataset_properties=None):
        """All regressors share these properties"""
        return {
            "shortname": cls.__name__, "name": cls.__name__,
            "handles_regression": True, "handles_classification": False,
            "handles_multiclass": False, "handles_multilabel": False,
            "handles_multioutput": False, "is_deterministic": cls.deterministic,
            # both input and output must be tuple(iterable)
            "input": (DENSE, SIGNED_DATA, UNSIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim != 1:
            raise ValueError("These wrappers support single-output regression only")
        
        ctx = DataContext(n_samples=X.shape[0], n_features=X.shape[1],
                          n_classes=0)
        
        model_params = self.params_class(ctx)
        configured_params = {name: getattr(self, name) for name in model_params.param_space}

        # convert self.random_state into an integer seed or None for ModelParam's eval_parameters()
        # auto-sklearn supplies its own NumPy RandomState
        seed = self.random_state
        if seed is not None:
            if isinstance(seed, (int, np.integer)):
                seed = int(seed)
            else: # copy generator and draw a numeric seed
                rng_copy = np.random.RandomState()
                rng_copy.set_state(check_random_state(seed).get_state())
                seed = int(rng_copy.randint(0, 2**31 - 1))

        kwargs = model_params.eval_parameters(configured_params, random_state=seed)

        # adapt parameters to the arguments sklearn expects
        if self.estimator_class is MLPRegressor: 
            kwargs["hidden_layer_sizes"] = tuple(
                kwargs.pop(f"layer_{i}") for i in range(1, 6)
            )
        if "n_jobs" in inspect.signature(self.estimator_class).parameters:
            kwargs["n_jobs"] = 1

        # create the actual sklearn model using prepared parameters
        estimator = self.estimator_class(**kwargs)

        if sample_weight is None:
            estimator.fit(X, y)
        elif "sample_weight" in inspect.signature(estimator.fit).parameters:
            estimator.fit(X, y, sample_weight=sample_weight)
        else:
            raise ValueError(f"{type(estimator).__name__} does not support sample_weight")
        self.estimator = estimator
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotFittedError("Call fit before predict")
        return self.estimator.predict(X)


def make_regressor_class(model_class: Type[RegressorMixin], model_param_class: ModelParamsType, *, name,
                          data_context: Optional[DataContext] = None):
    """
    Return a named auto-sklearn subclass, built from one estimator class and one ModelParams class.
    'name' is keyword-only.
    """
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError("name must be a valid Python identifier")
    if name in globals():
        raise ValueError(f"A module object named {name!r} already exists")

    default_space = search_space(model_param_class, {"multiclass": False}, data_context)
    defaults = dict(default_space.get_default_configuration())
    parameter_names = tuple(defaults)
    if "random_state" in parameter_names:
        raise ValueError("random_state is an evaluation seed, not a search parameter")

    # custom regressor component
    class CustomRegressor(RegressorProperties, AutoSklearnRegressionAlgorithm):
        params_class = model_param_class
        estimator_class = model_class
        deterministic = model_class is KNeighborsRegressor

        def __init__(self, random_state=None, **params):
            # reject unrecognized parameter names
            unknown = set(params) - set(parameter_names)
            if unknown:
                raise TypeError(f"Unknown parameters for {name}: {sorted(unknown)}")

            # assign attributes (e.g, setattr(self, "n_estimators", 200) is self.n_estimators = 200)
            # use values in 'params' if present, otherwise use ConfigSpace's defaults
            for param_name in parameter_names:
                setattr(self, param_name, params.get(param_name, defaults[param_name]))
            self.random_state = random_state
            # defer constructing the sklearn estimator until fit(), after parsing.
            self.estimator = None

        def get_params(self, deep=True):
            # sklearn cannot infer all hyperparameters from **params. Expose
            # the named attributes explicitly for clone() and set_params().
            params = {key: getattr(self, key) for key in parameter_names}
            params["random_state"] = self.random_state
            return params

        @staticmethod
        def get_hyperparameter_search_space(feat_type=None, dataset_properties=None):
            return search_space(model_param_class, dataset_properties, data_context)

    # a fresh class is produced for every call; unique names prevent auto-sklearn
    # registration collisions. qualname/module/global binding support pickling.
    CustomRegressor.__name__ = name
    CustomRegressor.__qualname__ = name
    CustomRegressor.__module__ = __name__
    CustomRegressor.__doc__ = f"Factory wrapper for {model_class.__name__}."
    globals()[name] = CustomRegressor
    return CustomRegressor


##########################################################################################
#
# FEATURE PREPROCESSING NODE FACTORY
#
##########################################################################################

class PreprocessorProperties:
    """Class to encapsulate shared preprocessing properties and methods"""

    @classmethod
    def get_properties(cls, dataset_properties=None):
        """These wrappers support dense, single-output classification or regression"""
        return {
            "shortname": cls.__name__, "name": cls.__name__,
            "handles_regression": not cls.classification_task,
            "handles_classification": cls.classification_task,
            "handles_multiclass": cls.classification_task,
            "handles_multilabel": False, "handles_multioutput": False,
            "is_deterministic": cls.deterministic,
            # override these when an operator requires unsigned input or produces sparse data
            "input": cls.component_input,
            "output": cls.component_output,
        }

    def fit(self, X, y=None):
        ctx = DataContext(
            n_samples=X.shape[0], n_features=X.shape[1],
            n_classes=(len(np.unique(y)) if self.classification_task and y is not None
                       else self.context.n_classes),
        )
        model_params = self.params_class(ctx)
        configured_params = {name: getattr(self, name) for name in model_params.param_space}

        # convert self.random_state into an integer seed or None for ModelParam's eval_parameters()
        seed = self.random_state
        if seed is not None:
            if isinstance(seed, (int, np.integer)):
                seed = int(seed)
            else: # copy generator and draw a numeric seed
                rng_copy = np.random.RandomState()
                rng_copy.set_state(check_random_state(seed).get_state())
                seed = int(rng_copy.randint(0, 2**31 - 1))

        kwargs = model_params.eval_parameters(configured_params, random_state=seed)

        # sklearn's default selector score is for classification, not regression.
        if not self.classification_task and self.estimator_class in {SelectFwe, SelectPercentile}:
            kwargs["score_func"] = f_regression

        # some version(s) of sklearn refers to 'metric' as 'affinity', replace
        if self.estimator_class is FeatureAgglomeration:
            if "metric" not in inspect.signature(FeatureAgglomeration).parameters:
                kwargs["affinity"] = kwargs.pop("metric")

        self.estimator = self.estimator_class(**kwargs)
        self.estimator.fit(X, y)
        return self

    def transform(self, X):
        if self.estimator is None:
            raise NotFittedError("Call fit before transform")
        return self.estimator.transform(X)


def make_preprocessor_class(model_class: Type[TransformerMixin], model_param_class: ModelParamsType, *, name,
                            data_context: DataContext,
                            classification: bool = True,
                            deterministic: bool = False,
                            input_types=(DENSE, SIGNED_DATA, UNSIGNED_DATA),
                            output_types=(DENSE, SIGNED_DATA, UNSIGNED_DATA)):
    """
    Return a named auto-sklearn subclass, built from one estimator class and one ModelParams class.
    'name' is keyword-only.
    """
    component_deterministic = deterministic
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError("name must be a valid Python identifier")
    if name in globals():
        raise ValueError(f"A module object named {name!r} already exists")

    default_space = search_space(model_param_class, {"multiclass": False}, data_context)
    defaults = dict(default_space.get_default_configuration())
    parameter_names = tuple(defaults)
    if "random_state" in parameter_names:
        raise ValueError("random_state is an evaluation seed, not a search parameter")

    # custom preprocessing component
    class CustomPreprocessor(PreprocessorProperties, AutoSklearnPreprocessingAlgorithm):
        params_class = model_param_class
        estimator_class = model_class
        deterministic = component_deterministic
        context = data_context
        classification_task = classification
        component_input = input_types
        component_output = output_types

        def __init__(self, random_state=None, feat_type=None, **params):
            # reject unrecognized parameter names
            unknown = set(params) - set(parameter_names)
            if unknown:
                raise TypeError(f"Unknown parameters for {name}: {sorted(unknown)}")

            # assign attributes (e.g, setattr(self, "n_estimators", 200) is self.n_estimators = 200)
            # use values in 'params' if present, otherwise use ConfigSpace's defaults
            for param_name in parameter_names:
                setattr(self, param_name, params.get(param_name, defaults[param_name]))
            self.feat_type = feat_type
            self.random_state = random_state
            # Defer constructing the sklearn estimator until fit(), after parsing
            self.estimator = None

        def get_params(self, deep=True):
            # sklearn cannot infer all hyperparameters from **params. Expose
            # the named attributes explicitly for clone() and set_params().
            params = {key: getattr(self, key) for key in parameter_names}
            params["feat_type"] = self.feat_type
            params["random_state"] = self.random_state
            return params

        @staticmethod
        def get_hyperparameter_search_space(feat_type=None, dataset_properties=None):
            return search_space(model_param_class, dataset_properties, data_context)

    # A fresh class is produced for every call; unique names prevent auto-sklearn
    # registration collisions. qualname/module/global binding support pickling.
    CustomPreprocessor.__name__ = name
    CustomPreprocessor.__qualname__ = name
    CustomPreprocessor.__module__ = __name__
    CustomPreprocessor.__doc__ = f"Factory wrapper for {model_class.__name__}."
    globals()[name] = CustomPreprocessor
    return CustomPreprocessor


##########################################################################################
#
# COMPONENT REGISTRATION
#
##########################################################################################


def register_preprocessor(component_class):
    """Register a transformer or selector in the feature-preprocessing stage only."""
    add_feature_preprocessor(component_class)
    return component_class.__name__

def register_classifier(component_class):
    """Register a classifier for the estimator stage."""
    add_classifier(component_class)
    return component_class.__name__

def register_regressor(component_class):
    """Register a regressor for the estimator stage."""
    add_regressor(component_class)
    return component_class.__name__

def register_feature_preprocessors(data_context: DataContext, *, classification: bool = True):
    """
    Generate/register all transformer and selector alternatives; 
    keep built-in data preprocessing (ignore TPEC's scaler).
    Return names to include in the runner.
    """
    names = []
    for model_param_class in (*TRANSFORMERS, *SELECTORS):
        model_params = model_param_class(data_context)
        component_name = model_params.get_model_type()
        component_class = make_preprocessor_class(
            PREPROCESSING_CLASSES[component_name], model_param_class,
            name=f"Custom{component_name}", data_context=data_context,
            classification=classification,
        )
        names.append(register_preprocessor(component_class))
    return names


def register_classifiers(data_context: DataContext):
    """
    Generate/register classifier alternatives from the project's parameter spaces.
    Return names to include in the runner.
    """
    names = []
    for model_param_class in CLASSIFIERS:
        model_params = model_param_class(data_context)
        component_name = model_params.get_model_type()
        component_class = make_classifier_class(
            CLASSIFIER_CLASSES[component_name], model_param_class,
            name=f"Custom{component_name}Classifier", data_context=data_context,
        )
        names.append(register_classifier(component_class))
    return names


def register_regressors(data_context: DataContext):
    """
    Generate/register regressor alternatives from the project's parameter spaces.
    Return names to include in the runner.
    """
    names = []
    for model_param_class in REGRESSORS:
        model_params = model_param_class(data_context)
        component_name = model_params.get_model_type()
        component_class = make_regressor_class(
            REGRESSOR_CLASSES[component_name], model_param_class,
            name=f"Custom{component_name}Regressor", data_context=data_context,
        )
        names.append(register_regressor(component_class))
    return names