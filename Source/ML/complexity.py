##########################################################################################
#
# Model complexity score: an estimate of the number of learned parameters in a fitted
# estimator or pipeline.
#
# Ported from TPOT's complexity objective (LGPL-3.0):
# https://github.com/EpistasisLab/tpot/blob/1bca6c6a51a79dc7370fbd2a8864a561fc5d7529/tpot/objectives/complexity.py
#
# Only the estimators this repo builds are covered. Differences from TPOT:
#   * MLPs count every weight and bias (fan_in * fan_out + fan_out per layer); TPOT counts
#     fan_in + fan_out per layer, which is not a parameter count.
#   * Scores are returned as plain ints so they serialize to JSON.
#
##########################################################################################

import warnings

import numpy as np
from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
)
from sklearn.feature_selection import SelectorMixin
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC, SVR
from xgboost import XGBRegressor

from Source.ML.classifiers import AnyLabelXGBClassifier


def tree_complexity(tree) -> int:
    # each node has 5 parameters (feature, threshold, left child, right child, value)
    return int(tree.tree_.node_count * 5)


def forest_complexity(forest) -> int:
    # GradientBoosting stores a 2-D (n_estimators, K) array of trees, so flatten it
    all_trees = np.array(forest.estimators_)
    if len(all_trees.shape) > 1:
        all_trees = all_trees.ravel()
    return int(sum(tree_complexity(tree) for tree in all_trees))


def xgboost_complexity(model) -> int:
    # one row per node across all stored trees (including every class's trees in multiclass); each
    # node is counted with the same 5 parameters as tree_complexity
    return int(len(model.get_booster().trees_to_dataframe()) * 5)


def knn_complexity(knn) -> int:
    return int(knn.n_neighbors)


def support_vector_machine_complexity(svm) -> int:
    count = int(sum(svm.n_support_))
    if svm.kernel == 'linear':
        count += int(np.count_nonzero(svm.coef_))
    return count


def mlp_complexity(mlp) -> int:
    return int(sum(coef.size + intercept.size for coef, intercept in zip(mlp.coefs_, mlp.intercepts_)))


COMPLEXITY_PER_ESTIMATOR = {
    RandomForestClassifier: forest_complexity,
    RandomForestRegressor: forest_complexity,
    ExtraTreesClassifier: forest_complexity,
    ExtraTreesRegressor: forest_complexity,
    GradientBoostingClassifier: forest_complexity,
    GradientBoostingRegressor: forest_complexity,
    SVC: support_vector_machine_complexity,
    SVR: support_vector_machine_complexity,
    KNeighborsClassifier: knn_complexity,
    KNeighborsRegressor: knn_complexity,
    MLPClassifier: mlp_complexity,
    MLPRegressor: mlp_complexity,
    AnyLabelXGBClassifier: xgboost_complexity,
    XGBRegressor: xgboost_complexity,
}


def calculate_model_complexity(est) -> int:
    """
    Estimate the number of learned parameters in a fitted estimator or pipeline.

    Pipelines and FeatureUnions sum their steps. Predictors use their per-model formula;
    feature selectors add 0 and every other transformer (scalers, ColumnTransformer, PCA, ...)
    adds 1, as in TPOT.
    """
    if isinstance(est, Pipeline):
        return sum(calculate_model_complexity(step) for _, step in est.steps)
    if isinstance(est, FeatureUnion):
        return sum(calculate_model_complexity(step) for _, step in est.transformer_list)

    model_type = type(est)
    if model_type in COMPLEXITY_PER_ESTIMATOR:
        return COMPLEXITY_PER_ESTIMATOR[model_type](est)
    if is_classifier(est) or is_regressor(est):
        warnings.warn(f"Complexity objective not defined for this classifier/regressor: {model_type}")
    if isinstance(est, SelectorMixin):
        return 0
    return 1
