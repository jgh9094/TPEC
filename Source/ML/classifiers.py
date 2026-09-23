##########################################################################################
#
# Concrete parameter-space classes for the scikit-learn classifiers used by the EAs.
# Each class derives from Source.Base.model_param_space.ModelParams and only needs to
# define its own hyperparameter space (__init__), how a genotype is mapped to scikit-learn
# kwargs (eval_parameters), and its model-type identifier (get_model_type). Random
# sampling, mutation, and TPE encoding are inherited from ModelParams.
#
##########################################################################################

from typeguard import typechecked
from typing import Dict, Any, Optional

from Source.Base.model_param_space import ModelParams, DataContext, IntParam, FloatParam, CatParam


@typechecked
class RandomForestParams(ModelParams):
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={
            'n_estimators': IntParam(bounds=(100, 1000), type='int'),
            'criterion': CatParam(bounds=('gini', 'entropy', 'log_loss'), type='cat'),
            'max_depth': IntParam(bounds=(1, 30), type='int'),
            'max_features': FloatParam(bounds=(0.0 + offset, 1.0 - offset), type='float', log=False),
            'max_samples': FloatParam(bounds=(0.0 + offset, 1.0 - offset), type='float', log=False),
            'class_weight': CatParam(bounds=('None', 'balanced', 'balanced_subsample'), type='cat')
        })

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Fixes a set of parameters for hard evaluation with scikit-learn. """
        return {'n_estimators': model_params['n_estimators'],
                'criterion': model_params['criterion'],
                'max_depth': model_params['max_depth'],
                'max_features': model_params['max_features'],
                'max_samples': model_params['max_samples'],
                'class_weight': model_params['class_weight'] if model_params['class_weight'] != 'None' else None,
                'random_state': random_state}

    def get_model_type(self) -> str:
        return "RF"


@typechecked
class ExtraTreesParams(ModelParams):
    """
    Parameter space for scikit-learn's ExtraTreesClassifier (extremely randomized trees).

    Mirrors RandomForestParams but intentionally omits ``max_samples``: ExtraTrees is used
    with its default ``bootstrap=False`` (the whole sample is drawn for every tree and split
    thresholds are chosen at random), which is what distinguishes it from Random Forest and
    is the source of its additional stochasticity. ``max_samples`` only applies when
    ``bootstrap=True`` in scikit-learn, so it is not part of this space.
    """
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={
            'n_estimators': IntParam(bounds=(100, 1000), type='int'),
            'criterion': CatParam(bounds=('gini', 'entropy', 'log_loss'), type='cat'),
            'max_depth': IntParam(bounds=(1, 30), type='int'),
            'max_features': FloatParam(bounds=(0.0 + offset, 1.0 - offset), type='float', log=False),
            'class_weight': CatParam(bounds=('None', 'balanced', 'balanced_subsample'), type='cat')
        })

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Fixes a set of parameters for hard evaluation with scikit-learn. """
        return {'n_estimators': model_params['n_estimators'],
                'criterion': model_params['criterion'],
                'max_depth': model_params['max_depth'],
                'max_features': model_params['max_features'],
                'class_weight': model_params['class_weight'] if model_params['class_weight'] != 'None' else None,
                'random_state': random_state}

    def get_model_type(self) -> str:
        return "ET"


@typechecked
class KernelSVCParams(ModelParams):
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={
            'C': FloatParam(bounds=(1e-3, 1e2), type='float', log=True),
            'kernel': CatParam(bounds=('linear', 'poly', 'rbf', 'sigmoid'), type='cat'),
            'gamma': FloatParam(bounds=(1e-4, 1e1), type='float', log=True),
            'degree': IntParam(bounds=(2, 5), type='int'),
            'max_iter': IntParam(bounds=(10000, 100000), type='int'),
            'class_weight': CatParam(bounds=('None', 'balanced'), type='cat'),
            'decision_function_shape': CatParam(bounds=('ovo', 'ovr'), type='cat')
        })

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Fixes a set of parameters for hard evaluation with scikit-learn. """
        return {'C': model_params['C'],
                'kernel': model_params['kernel'],
                'gamma': model_params['gamma'],
                'degree': model_params['degree'],
                'max_iter': model_params['max_iter'],
                'class_weight': model_params['class_weight'] if model_params['class_weight'] != 'None' else None,
                'decision_function_shape': model_params['decision_function_shape'],
                'random_state': random_state}

    def get_model_type(self) -> str:
        return "KSVC"


@typechecked
class GradientBoostParams(ModelParams):
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        # 'exponential' loss is only valid for binary targets; only offer it when n_classes == 2.
        binary_class = ctx.n_classes == 2
        super().__init__(param_space={
            'loss': CatParam(bounds=('log_loss', 'exponential'), type='cat') if binary_class else CatParam(bounds=('log_loss',), type='cat'),
            'learning_rate': FloatParam(bounds=(1e-3, 0.5), type='float', log=True),
            'n_estimators': IntParam(bounds=(100, 1000), type='int'),
            'subsample': FloatParam(bounds=(0.0 + offset, 1.0 - offset), type='float', log=False),
            'criterion': CatParam(bounds=('friedman_mse', 'squared_error'), type='cat'),
            'max_depth': IntParam(bounds=(1, 8), type='int'),
            'max_features': FloatParam(bounds=(0.0 + offset, 1.0 - offset), type='float', log=False)
        })

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Fixes a set of parameters for hard evaluation with scikit-learn. """
        return {'n_estimators': model_params['n_estimators'],
                'learning_rate': model_params['learning_rate'],
                'subsample': model_params['subsample'],
                'criterion': model_params['criterion'],
                'max_depth': model_params['max_depth'],
                'max_features': model_params['max_features'],
                'loss': model_params['loss'],
                'random_state': random_state}

    def get_model_type(self) -> str:
        return "GB"


@typechecked
class KNeighborsClassifierParams(ModelParams):
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={
            'n_neighbors': IntParam(bounds=(1, 300), type='int'),
            'weights': CatParam(bounds=('uniform', 'distance'), type='cat'),
            'algorithm': CatParam(bounds=('ball_tree', 'kd_tree', 'brute'), type='cat'),
            'leaf_size': IntParam(bounds=(1, 100), type='int'),
            'p': IntParam(bounds=(1, 5), type='int')
        })

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Fixes a set of parameters for hard evaluation with scikit-learn.

        KNeighborsClassifier is deterministic and accepts no ``random_state``, so the argument
        is ignored here (kept in the signature for a uniform eval_parameters interface).
        """
        return {'n_neighbors': model_params['n_neighbors'],
                'weights': model_params['weights'],
                'algorithm': model_params['algorithm'],
                'leaf_size': model_params['leaf_size'],
                'p': model_params['p']}

    def get_model_type(self) -> str:
        return "KNN"


@typechecked
class MLPClassifierParams(ModelParams):
    def __init__(self, ctx: DataContext, offset: float = 1.0e-4):
        super().__init__(param_space={
            'layer_1': IntParam(bounds=(10, 100), type='int'),
            'layer_2': IntParam(bounds=(10, 100), type='int'),
            'layer_3': IntParam(bounds=(10, 100), type='int'),
            'layer_4': IntParam(bounds=(10, 100), type='int'),
            'layer_5': IntParam(bounds=(10, 100), type='int'),
            'activation': CatParam(bounds=('identity', 'logistic', 'tanh', 'relu'), type='cat'),
            'solver': CatParam(bounds=('lbfgs', 'sgd', 'adam'), type='cat'),
            'alpha': FloatParam(bounds=(1e-6, 1e-1), type='float', log=True),
            'max_iter': IntParam(bounds=(10000, 100000), type='int')
        })

    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """ Fixes a set of parameters for hard evaluation with scikit-learn. """
        return {'layer_1': model_params['layer_1'],
                'layer_2': model_params['layer_2'],
                'layer_3': model_params['layer_3'],
                'layer_4': model_params['layer_4'],
                'layer_5': model_params['layer_5'],
                'activation': model_params['activation'],
                'solver': model_params['solver'],
                'alpha': model_params['alpha'],
                'max_iter': model_params['max_iter'],
                'random_state': random_state}

    def get_model_type(self) -> str:
        return "MLP"
