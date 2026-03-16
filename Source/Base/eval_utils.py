# Import scikit-learn models
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from typing import Dict, Any, Tuple
from typeguard import typechecked
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Source.Base.model_param_space import (
    RandomForestParams, LinearSVCParams, DecisionTreeParams,
    KernelSVCParams, ExtraTreesParams, GradientBoostParams, LinearSGDParams
)
from Source.Base.ray_utils import (
    train_random_forest, train_linear_svc, train_decision_tree,
    train_kernel_svc, train_extra_trees, train_gradient_boost, train_linear_sgd,
    train_random_forest_cv, train_linear_svc_cv, train_decision_tree_cv,
    train_kernel_svc_cv, train_extra_trees_cv, train_gradient_boost_cv, train_linear_sgd_cv
)

@typechecked
def train_test_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_params: Dict[str, Any],
    random_state: int
) -> Tuple[float, float, float]:
    """
    Train and evaluate a Random Forest classifier.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : np.ndarray
        Training labels
    X_test : pd.DataFrame
        Testing features
    y_test : np.ndarray
        Testing labels
    model_params : Dict[str, Any]
        Model hyperparameters
    random_state : int
        Random seed

    Returns:
    --------
    train_accuracy : float
        Accuracy on training set
    test_accuracy : float
        Accuracy on test set
    error : float
        1.0 if successful, -1.0 if error occurred
    """
    try:
        model = RandomForestClassifier(**model_params, random_state=random_state)
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        return float(train_acc), float(test_acc), 1.0
    except Exception as e:
        print(f"Error in train_test_random_forest: {e}", flush=True)
        return 0.0, 0.0, -1.0

@typechecked
def train_test_linear_svc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_params: Dict[str, Any],
    random_state: int
) -> Tuple[float, float, float]:
    """
    Train and evaluate a Linear SVC classifier.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : np.ndarray
        Training labels
    X_test : pd.DataFrame
        Testing features
    y_test : np.ndarray
        Testing labels
    model_params : Dict[str, Any]
        Model hyperparameters
    random_state : int
        Random seed

    Returns:
    --------
    train_accuracy : float
        Accuracy on training set
    test_accuracy : float
        Accuracy on test set
    error : float
        1.0 if successful, -1.0 if error occurred
    """
    try:
        model = LinearSVC(**model_params, random_state=random_state)
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        return float(train_acc), float(test_acc), 1.0
    except Exception as e:
        print(f"Error in train_test_linear_svc: {e}", flush=True)
        return 0.0, 0.0, -1.0

@typechecked
def train_test_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_params: Dict[str, Any],
    random_state: int
) -> Tuple[float, float, float]:
    """
    Train and evaluate a Decision Tree classifier.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : np.ndarray
        Training labels
    X_test : pd.DataFrame
        Testing features
    y_test : np.ndarray
        Testing labels
    model_params : Dict[str, Any]
        Model hyperparameters
    random_state : int
        Random seed

    Returns:
    --------
    train_accuracy : float
        Accuracy on training set
    test_accuracy : float
        Accuracy on test set
    error : float
        1.0 if successful, -1.0 if error occurred
    """
    try:
        model = DecisionTreeClassifier(**model_params, random_state=random_state)
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        return float(train_acc), float(test_acc), 1.0
    except Exception as e:
        print(f"Error in train_test_decision_tree: {e}", flush=True)
        return 0.0, 0.0, -1.0

@typechecked
def train_test_kernel_svc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_params: Dict[str, Any],
    random_state: int
) -> Tuple[float, float, float]:
    """
    Train and evaluate a Kernel SVC classifier.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : np.ndarray
        Training labels
    X_test : pd.DataFrame
        Testing features
    y_test : np.ndarray
        Testing labels
    model_params : Dict[str, Any]
        Model hyperparameters
    random_state : int
        Random seed

    Returns:
    --------
    train_accuracy : float
        Accuracy on training set
    test_accuracy : float
        Accuracy on test set
    error : float
        1.0 if successful, -1.0 if error occurred
    """
    try:
        model = SVC(**model_params, random_state=random_state)
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        return float(train_acc), float(test_acc), 1.0
    except Exception as e:
        print(f"Error in train_test_kernel_svc: {e}", flush=True)
        return 0.0, 0.0, -1.0

@typechecked
def train_test_extra_trees(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_params: Dict[str, Any],
    random_state: int
) -> Tuple[float, float, float]:
    """
    Train and evaluate an Extra Trees classifier.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : np.ndarray
        Training labels
    X_test : pd.DataFrame
        Testing features
    y_test : np.ndarray
        Testing labels
    model_params : Dict[str, Any]
        Model hyperparameters
    random_state : int
        Random seed

    Returns:
    --------
    train_accuracy : float
        Accuracy on training set
    test_accuracy : float
        Accuracy on test set
    error : float
        1.0 if successful, -1.0 if error occurred
    """
    try:
        model = ExtraTreesClassifier(**model_params, random_state=random_state)
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        return float(train_acc), float(test_acc), 1.0
    except Exception as e:
        print(f"Error in train_test_extra_trees: {e}", flush=True)
        return 0.0, 0.0, -1.0

@typechecked
def train_test_gradient_boost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_params: Dict[str, Any],
    random_state: int
) -> Tuple[float, float, float]:
    """
    Train and evaluate a Gradient Boosting classifier.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : np.ndarray
        Training labels
    X_test : pd.DataFrame
        Testing features
    y_test : np.ndarray
        Testing labels
    model_params : Dict[str, Any]
        Model hyperparameters
    random_state : int
        Random seed

    Returns:
    --------
    train_accuracy : float
        Accuracy on training set
    test_accuracy : float
        Accuracy on test set
    error : float
        1.0 if successful, -1.0 if error occurred
    """
    try:
        model = GradientBoostingClassifier(**model_params, random_state=random_state)
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        return float(train_acc), float(test_acc), 1.0
    except Exception as e:
        print(f"Error in train_test_gradient_boost: {e}", flush=True)
        return 0.0, 0.0, -1.0

@typechecked
def train_test_linear_sgd(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_params: Dict[str, Any],
    random_state: int
) -> Tuple[float, float, float]:
    """
    Train and evaluate a Linear SGD classifier.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : np.ndarray
        Training labels
    X_test : pd.DataFrame
        Testing features
    y_test : np.ndarray
        Testing labels
    model_params : Dict[str, Any]
        Model hyperparameters
    random_state : int
        Random seed

    Returns:
    --------
    train_accuracy : float
        Accuracy on training set
    test_accuracy : float
        Accuracy on test set
    error : float
        1.0 if successful, -1.0 if error occurred
    """
    try:
        model = SGDClassifier(**model_params, random_state=random_state)
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        return float(train_acc), float(test_acc), 1.0
    except Exception as e:
        print(f"Error in train_test_linear_sgd: {e}", flush=True)
        return 0.0, 0.0, -1.0

# Model configuration mapping
MODEL_CONFIG = {
    'RF': {
        'param_class': RandomForestParams(),
        'ray_train_func': train_random_forest,
        'ray_train_func_cv': train_random_forest_cv,
        'test_eval_func': train_test_random_forest,
        'display_name': 'Random Forest'
    },
    'LSVC': {
        'param_class': LinearSVCParams(),
        'ray_train_func': train_linear_svc,
        'ray_train_func_cv': train_linear_svc_cv,
        'test_eval_func': train_test_linear_svc,
        'display_name': 'Linear SVC'
    },
    'DT': {
        'param_class': DecisionTreeParams(),
        'ray_train_func': train_decision_tree,
        'ray_train_func_cv': train_decision_tree_cv,
        'test_eval_func': train_test_decision_tree,
        'display_name': 'Decision Tree'
    },
    'KSVC': {
        'param_class': KernelSVCParams(),
        'ray_train_func': train_kernel_svc,
        'ray_train_func_cv': train_kernel_svc_cv,
        'test_eval_func': train_test_kernel_svc,
        'display_name': 'Kernel SVC'
    },
    'ET': {
        'param_class': ExtraTreesParams(),
        'ray_train_func': train_extra_trees,
        'ray_train_func_cv': train_extra_trees_cv,
        'test_eval_func': train_test_extra_trees,
        'display_name': 'Extra Trees'
    },
    'GB': {
        'param_class': GradientBoostParams(classes=2),
        'ray_train_func': train_gradient_boost,
        'ray_train_func_cv': train_gradient_boost_cv,
        'test_eval_func': train_test_gradient_boost,
        'display_name': 'Gradient Boosting'
    },
    'LSGD': {
        'param_class': LinearSGDParams(),
        'ray_train_func': train_linear_sgd,
        'ray_train_func_cv': train_linear_sgd_cv,
        'test_eval_func': train_test_linear_sgd,
        'display_name': 'Linear SGD'
    }
}