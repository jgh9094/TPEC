##########################################################################################
#
# Per-fold cross-validation training/evaluation tasks for the HPO EA, run as Ray remote
# functions (one task per (individual, fold) pair; see Source/HPO/ea.py::evaluation).
#
# Two families of functions with an identical signature so the EA can dispatch either one
# uniformly:
#   * Classification (``cv_*``): fit a scikit-learn *Classifier* and score with ROC AUC
#     (``predict_proba``); binary vs. multi-class is selected by ``binary_class``/``labels``.
#   * Regression (``cv_*_reg``): fit the sibling *Regressor* and score with R^2
#     (``predict``); ``binary_class``/``labels`` are accepted for signature uniformity and
#     ignored.
#
# Every function returns ``(id, train_score, val_score, status)`` where ``status`` is 1.0 on
# success and -1.0 if the fit/score raised. On failure the scores are returned as 0.0/0.0, but
# the caller treats any failed fold as a complete penalty and overwrites the individual's
# performance with its mode-appropriate error penalty (see EA.evaluation), so the 0.0 sentinel
# here is not what ultimately ranks an errored individual.
#
##########################################################################################

import ray
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor


# =========================================================================================
# Classification CV tasks (ROC AUC)
# =========================================================================================
@ray.remote
def cv_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    model_params: Dict[str, Any],
    id: int,
    binary_class: bool,
    labels: np.ndarray
) -> Tuple[int, float, float, float]:
    """
    Train and evaluate a RandomForestClassifier using Ray.

    Parameters:
        X_train: Training features
        y_train: Training labels
        X_validate: Validation features
        y_validate: Validation labels
        model_params: Dictionary of hyperparameters for RandomForestClassifier
            (already includes ``random_state`` when the caller seeded eval_parameters)
        id: Identifier for this model instance
        binary_class: True for binary classification, False for multi-class
        labels: Array of all possible class labels

    Returns:
        Tuple of (id, training_auc, validation_auc, status)
        status: 1.0 if successful, -1.0 if error occurred
    """
    try:
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        if binary_class:
            train_acc = float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
            val_acc = float(roc_auc_score(y_validate, model.predict_proba(X_validate)[:, 1]))
        else:
            train_acc = float(roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovo', labels=labels))
            val_acc = float(roc_auc_score(y_validate, model.predict_proba(X_validate), multi_class='ovo', labels=labels))
        return id, train_acc, val_acc, 1.0

    except Exception as e:
        print(f"Error in cv_random_forest: {e}")
        return id, 0.0, 0.0, -1.0

@ray.remote
def cv_kernel_svc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    model_params: Dict[str, Any],
    id: int,
    binary_class: bool,
    labels: np.ndarray
) -> Tuple[int, float, float, float]:
    """
    Train and evaluate a SVC (Kernel SVM) using Ray.

    Parameters:
        X_train: Training features
        y_train: Training labels
        X_validate: Validation features
        y_validate: Validation labels
        model_params: Dictionary of hyperparameters for SVC
            (already includes ``random_state`` when the caller seeded eval_parameters)
        id: Identifier for this model instance
        binary_class: True for binary classification, False for multi-class
        labels: Array of all possible class labels

    Returns:
        Tuple of (id, training_auc, validation_auc, status)
        status: 1.0 if successful, -1.0 if error occurred
    """
    try:
        model = SVC(**model_params, probability=True)
        model.fit(X_train, y_train)
        if binary_class:
            train_acc = float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
            val_acc = float(roc_auc_score(y_validate, model.predict_proba(X_validate)[:, 1]))
        else:
            train_acc = float(roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovo', labels=labels))
            val_acc = float(roc_auc_score(y_validate, model.predict_proba(X_validate), multi_class='ovo', labels=labels))
        return id, train_acc, val_acc, 1.0

    except Exception as e:
        print(f"Error in cv_kernel_svc: {e}")
        return id, 0.0, 0.0, -1.0

@ray.remote
def cv_gradient_boost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    model_params: Dict[str, Any],
    id: int,
    binary_class: bool,
    labels: np.ndarray
) -> Tuple[int, float, float, float]:
    """
    Train and evaluate a GradientBoostingClassifier using Ray.

    Parameters:
        X_train: Training features
        y_train: Training labels
        X_validate: Validation features
        y_validate: Validation labels
        model_params: Dictionary of hyperparameters for GradientBoostingClassifier
            (already includes ``random_state`` when the caller seeded eval_parameters)
        id: Identifier for this model instance
        binary_class: True for binary classification, False for multi-class
        labels: Array of all possible class labels

    Returns:
        Tuple of (id, training_auc, validation_auc, status)
        status: 1.0 if successful, -1.0 if error occurred
    """
    try:
        model = GradientBoostingClassifier(**model_params)
        model.fit(X_train, y_train)
        if binary_class:
            train_acc = float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
            val_acc = float(roc_auc_score(y_validate, model.predict_proba(X_validate)[:, 1]))
        else:
            train_acc = float(roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovo', labels=labels))
            val_acc = float(roc_auc_score(y_validate, model.predict_proba(X_validate), multi_class='ovo', labels=labels))
        return id, train_acc, val_acc, 1.0

    except Exception as e:
        print(f"Error in cv_gradient_boost: {e}")
        return id, 0.0, 0.0, -1.0

@ray.remote
def cv_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    model_params: Dict[str, Any],
    id: int,
    binary_class: bool,
    labels: np.ndarray
) -> Tuple[int, float, float, float]:
    """
    Train and evaluate a KNeighborsClassifier using Ray.

    Parameters:
        X_train: Training features
        y_train: Training labels
        X_validate: Validation features
        y_validate: Validation labels
        model_params: Dictionary of hyperparameters for KNeighborsClassifier
        id: Identifier for this model instance
        binary_class: True for binary classification, False for multi-class
        labels: Array of all possible class labels

    Returns:
        Tuple of (id, training_auc, validation_auc, status)
        status: 1.0 if successful, -1.0 if error occurred
    """
    try:
        model = KNeighborsClassifier(**model_params)
        model.fit(X_train, y_train)
        if binary_class:
            train_acc = float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
            val_acc = float(roc_auc_score(y_validate, model.predict_proba(X_validate)[:, 1]))
        else:
            train_acc = float(roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovo', labels=labels))
            val_acc = float(roc_auc_score(y_validate, model.predict_proba(X_validate), multi_class='ovo', labels=labels))
        return id, train_acc, val_acc, 1.0

    except Exception as e:
        print(f"Error in cv_knn: {e}")
        return id, 0.0, 0.0, -1.0

@ray.remote
def cv_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    model_params: Dict[str, Any],
    id: int,
    binary_class: bool,
    labels: np.ndarray
) -> Tuple[int, float, float, float]:
    """
    Train and evaluate a MLPClassifier using Ray.

    Parameters:
        X_train: Training features
        y_train: Training labels
        X_validate: Validation features
        y_validate: Validation labels
        model_params: Dictionary of hyperparameters for MLPClassifier
            (the per-layer sizes are recombined into hidden_layer_sizes here; already
            includes ``random_state`` when the caller seeded eval_parameters)
        id: Identifier for this model instance
        binary_class: True for binary classification, False for multi-class
        labels: Array of all possible class labels

    Returns:
        Tuple of (id, training_auc, validation_auc, status)
        status: 1.0 if successful, -1.0 if error occurred
    """
    try:
        layers = (model_params.get('layer_1'),
                  model_params.get('layer_2'),
                  model_params.get('layer_3'),
                  model_params.get('layer_4'),
                  model_params.get('layer_5'))

        model = MLPClassifier(hidden_layer_sizes=layers,
                              activation=model_params.get('activation'),
                              solver=model_params.get('solver'),
                              alpha=model_params.get('alpha'),
                              max_iter=model_params.get('max_iter'),
                              random_state=model_params.get('random_state'))
        model.fit(X_train, y_train)

        if binary_class:
            train_acc = float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
            val_acc = float(roc_auc_score(y_validate, model.predict_proba(X_validate)[:, 1]))
        else:
            train_acc = float(roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovo', labels=labels))
            val_acc = float(roc_auc_score(y_validate, model.predict_proba(X_validate), multi_class='ovo', labels=labels))

        return id, train_acc, val_acc, 1.0

    except Exception as e:
        print(f"Error in cv_mlp: {e}")
        return id, 0.0, 0.0, -1.0

@ray.remote
def cv_extra_trees(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    model_params: Dict[str, Any],
    id: int,
    binary_class: bool,
    labels: np.ndarray
) -> Tuple[int, float, float, float]:
    """
    Train and evaluate an ExtraTreesClassifier (extremely randomized trees) using Ray.

    Parameters:
        X_train: Training features
        y_train: Training labels
        X_validate: Validation features
        y_validate: Validation labels
        model_params: Dictionary of hyperparameters for ExtraTreesClassifier
            (already includes ``random_state`` when the caller seeded eval_parameters)
        id: Identifier for this model instance
        binary_class: True for binary classification, False for multi-class
        labels: Array of all possible class labels

    Returns:
        Tuple of (id, training_auc, validation_auc, status)
        status: 1.0 if successful, -1.0 if error occurred
    """
    try:
        model = ExtraTreesClassifier(**model_params)
        model.fit(X_train, y_train)
        if binary_class:
            train_acc = float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
            val_acc = float(roc_auc_score(y_validate, model.predict_proba(X_validate)[:, 1]))
        else:
            train_acc = float(roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovo', labels=labels))
            val_acc = float(roc_auc_score(y_validate, model.predict_proba(X_validate), multi_class='ovo', labels=labels))
        return id, train_acc, val_acc, 1.0

    except Exception as e:
        print(f"Error in cv_extra_trees: {e}")
        return id, 0.0, 0.0, -1.0


# =========================================================================================
# Regression CV tasks (R^2)
#
# Same signature as the classification tasks so EA.evaluation can dispatch either family
# uniformly. ``binary_class`` and ``labels`` are accepted but unused (regression has no
# classes). The target ``y`` is used as-is: callers must pre-scale/transform the regression
# target as needed (e.g. an MLPRegressor with solver='sgd' diverges on large-magnitude
# unscaled targets) -- such divergence is caught here and reported as an error (status -1.0).
# =========================================================================================
@ray.remote
def cv_random_forest_reg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    model_params: Dict[str, Any],
    id: int,
    binary_class: Optional[bool] = None,
    labels: Optional[np.ndarray] = None
) -> Tuple[int, float, float, float]:
    """
    Train and evaluate a RandomForestRegressor using Ray, scored with R^2.

    ``binary_class`` and ``labels`` are accepted for signature parity with the classification
    tasks and ignored. Returns (id, train_r2, val_r2, status); status is 1.0 on success and
    -1.0 on failure (scores 0.0/0.0, overwritten by the caller's error penalty).
    """
    try:
        model = RandomForestRegressor(**model_params)
        model.fit(X_train, y_train)
        train_r2 = float(r2_score(y_train, model.predict(X_train)))
        val_r2 = float(r2_score(y_validate, model.predict(X_validate)))
        return id, train_r2, val_r2, 1.0

    except Exception as e:
        print(f"Error in cv_random_forest_reg: {e}")
        return id, 0.0, 0.0, -1.0

@ray.remote
def cv_kernel_svr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    model_params: Dict[str, Any],
    id: int,
    binary_class: Optional[bool] = None,
    labels: Optional[np.ndarray] = None
) -> Tuple[int, float, float, float]:
    """
    Train and evaluate an SVR (Kernel SVM regression) using Ray, scored with R^2.

    ``binary_class`` and ``labels`` are accepted for signature parity and ignored. SVR is
    deterministic and takes no ``random_state``. Returns (id, train_r2, val_r2, status).
    """
    try:
        model = SVR(**model_params)
        model.fit(X_train, y_train)
        train_r2 = float(r2_score(y_train, model.predict(X_train)))
        val_r2 = float(r2_score(y_validate, model.predict(X_validate)))
        return id, train_r2, val_r2, 1.0

    except Exception as e:
        print(f"Error in cv_kernel_svr: {e}")
        return id, 0.0, 0.0, -1.0

@ray.remote
def cv_gradient_boost_reg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    model_params: Dict[str, Any],
    id: int,
    binary_class: Optional[bool] = None,
    labels: Optional[np.ndarray] = None
) -> Tuple[int, float, float, float]:
    """
    Train and evaluate a GradientBoostingRegressor using Ray, scored with R^2.

    ``binary_class`` and ``labels`` are accepted for signature parity and ignored. Returns
    (id, train_r2, val_r2, status).
    """
    try:
        model = GradientBoostingRegressor(**model_params)
        model.fit(X_train, y_train)
        train_r2 = float(r2_score(y_train, model.predict(X_train)))
        val_r2 = float(r2_score(y_validate, model.predict(X_validate)))
        return id, train_r2, val_r2, 1.0

    except Exception as e:
        print(f"Error in cv_gradient_boost_reg: {e}")
        return id, 0.0, 0.0, -1.0

@ray.remote
def cv_knn_reg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    model_params: Dict[str, Any],
    id: int,
    binary_class: Optional[bool] = None,
    labels: Optional[np.ndarray] = None
) -> Tuple[int, float, float, float]:
    """
    Train and evaluate a KNeighborsRegressor using Ray, scored with R^2.

    ``binary_class`` and ``labels`` are accepted for signature parity and ignored. Returns
    (id, train_r2, val_r2, status).
    """
    try:
        model = KNeighborsRegressor(**model_params)
        model.fit(X_train, y_train)
        train_r2 = float(r2_score(y_train, model.predict(X_train)))
        val_r2 = float(r2_score(y_validate, model.predict(X_validate)))
        return id, train_r2, val_r2, 1.0

    except Exception as e:
        print(f"Error in cv_knn_reg: {e}")
        return id, 0.0, 0.0, -1.0

@ray.remote
def cv_mlp_reg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    model_params: Dict[str, Any],
    id: int,
    binary_class: Optional[bool] = None,
    labels: Optional[np.ndarray] = None
) -> Tuple[int, float, float, float]:
    """
    Train and evaluate an MLPRegressor using Ray, scored with R^2.

    The per-layer sizes are recombined into hidden_layer_sizes here; ``model_params`` already
    includes ``random_state`` when the caller seeded eval_parameters. ``binary_class`` and
    ``labels`` are accepted for signature parity and ignored.

    NOTE: an MLPRegressor with solver='sgd' can diverge to non-finite weights on large-magnitude
    unscaled targets; the target must be pre-scaled by the caller. Divergence is caught here and
    reported as an error (status -1.0). Returns (id, train_r2, val_r2, status).
    """
    try:
        layers = (model_params.get('layer_1'),
                  model_params.get('layer_2'),
                  model_params.get('layer_3'),
                  model_params.get('layer_4'),
                  model_params.get('layer_5'))

        model = MLPRegressor(hidden_layer_sizes=layers,
                             activation=model_params.get('activation'),
                             solver=model_params.get('solver'),
                             alpha=model_params.get('alpha'),
                             max_iter=model_params.get('max_iter'),
                             random_state=model_params.get('random_state'))
        model.fit(X_train, y_train)
        train_r2 = float(r2_score(y_train, model.predict(X_train)))
        val_r2 = float(r2_score(y_validate, model.predict(X_validate)))
        return id, train_r2, val_r2, 1.0

    except Exception as e:
        print(f"Error in cv_mlp_reg: {e}")
        return id, 0.0, 0.0, -1.0

@ray.remote
def cv_extra_trees_reg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    model_params: Dict[str, Any],
    id: int,
    binary_class: Optional[bool] = None,
    labels: Optional[np.ndarray] = None
) -> Tuple[int, float, float, float]:
    """
    Train and evaluate an ExtraTreesRegressor (extremely randomized trees) using Ray, scored
    with R^2.

    ``binary_class`` and ``labels`` are accepted for signature parity and ignored. Returns
    (id, train_r2, val_r2, status).
    """
    try:
        model = ExtraTreesRegressor(**model_params)
        model.fit(X_train, y_train)
        train_r2 = float(r2_score(y_train, model.predict(X_train)))
        val_r2 = float(r2_score(y_validate, model.predict(X_validate)))
        return id, train_r2, val_r2, 1.0

    except Exception as e:
        print(f"Error in cv_extra_trees_reg: {e}")
        return id, 0.0, 0.0, -1.0
