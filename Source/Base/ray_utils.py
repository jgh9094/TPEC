import ray
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

@ray.remote
def cv_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    model_params: Dict[str, Any],
    random_state: int,
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
        random_state: Random seed for reproducibility
        id: Identifier for this model instance
        binary_class: True for binary classification, False for multi-class
        labels: Array of all possible class labels

    Returns:
        Tuple of (id, training_auc, validation_auc, error)
        error: 1.0 if successful, -1.0 if error occurred
    """
    try:
        model = RandomForestClassifier(**model_params, random_state=random_state)
        model.fit(X_train, y_train)
        if binary_class:
            train_acc = float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
            val_acc = float(roc_auc_score(y_validate, model.predict_proba(X_validate)[:, 1]))
        else:
            train_acc = -float(log_loss(y_train, model.predict_proba(X_train), labels=labels))
            val_acc = -float(log_loss(y_validate, model.predict_proba(X_validate), labels=labels))
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
    random_state: int,
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
        random_state: Random seed for reproducibility
        id: Identifier for this model instance
        binary_class: True for binary classification, False for multi-class
        labels: Array of all possible class labels

    Returns:
        Tuple of (id, training_auc, validation_auc, error)
        error: 1.0 if successful, -1.0 if error occurred
    """
    try:
        model = SVC(**model_params, random_state=random_state, probability=True)
        model.fit(X_train, y_train)
        if binary_class:
            train_acc = float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
            val_acc = float(roc_auc_score(y_validate, model.predict_proba(X_validate)[:, 1]))
        else:
            train_acc = -float(log_loss(y_train, model.predict_proba(X_train), labels=labels))
            val_acc = -float(log_loss(y_validate, model.predict_proba(X_validate), labels=labels))
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
    random_state: int,
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
        random_state: Random seed for reproducibility
        id: Identifier for this model instance
        binary_class: True for binary classification, False for multi-class
        labels: Array of all possible class labels

    Returns:
        Tuple of (id, training_auc, validation_auc, error)
        error: 1.0 if successful, -1.0 if error occurred
    """
    try:
        model = GradientBoostingClassifier(**model_params, random_state=random_state)
        model.fit(X_train, y_train)
        if binary_class:
            train_acc = float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
            val_acc = float(roc_auc_score(y_validate, model.predict_proba(X_validate)[:, 1]))
        else:
            train_acc = -float(log_loss(y_train, model.predict_proba(X_train), labels=labels))
            val_acc = -float(log_loss(y_validate, model.predict_proba(X_validate), labels=labels))
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
    random_state: int,
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
        Tuple of (id, training_auc, validation_auc, error)
        error: 1.0 if successful, -1.0 if error occurred
    """
    try:
        model = KNeighborsClassifier(**model_params)
        model.fit(X_train, y_train)
        if binary_class:
            train_acc = float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
            val_acc = float(roc_auc_score(y_validate, model.predict_proba(X_validate)[:, 1]))
        else:
            train_acc = -float(log_loss(y_train, model.predict_proba(X_train), labels=labels))
            val_acc = -float(log_loss(y_validate, model.predict_proba(X_validate), labels=labels))
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
    random_state: int,
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
        random_state: Random seed for reproducibility
        id: Identifier for this model instance
        binary_class: True for binary classification, False for multi-class
        labels: Array of all possible class labels

    Returns:
        Tuple of (id, training_auc, validation_auc, error)
        error: 1.0 if successful, -1.0 if error occurred
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
                              max_iter=model_params.get('max_iter'),
                              random_state=random_state)
        model.fit(X_train, y_train)

        if binary_class:
            train_acc = float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
            val_acc = float(roc_auc_score(y_validate, model.predict_proba(X_validate)[:, 1]))
        else:
            train_acc = -float(log_loss(y_train, model.predict_proba(X_train), labels=labels))
            val_acc = -float(log_loss(y_validate, model.predict_proba(X_validate), labels=labels))

        return id, train_acc, val_acc, 1.0

    except Exception as e:
        print(f"Error in cv_mlp: {e}")
        return id, 0.0, 0.0, -1.0