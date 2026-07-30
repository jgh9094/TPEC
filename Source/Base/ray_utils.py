import ray
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

@ray.remote
def cv_random_forest(
    cv_splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    model_params: Dict[str, Any],
    random_state: int,
    id: int,
    binary_class: bool,
    labels: np.ndarray
) -> Tuple[int, float, float, float]:
    """
    Train and evaluate a RandomForestClassifier across all CV folds using Ray.

    Parameters:
        cv_splits: List of (X_train, y_train, X_validate, y_validate) tuples for each fold
        model_params: Dictionary of hyperparameters for RandomForestClassifier
        random_state: Random seed for reproducibility
        id: Identifier for this model instance
        binary_class: True for binary classification, False for multi-class
        labels: Array of all possible class labels

    Returns:
        Tuple of (id, mean_training_auc, mean_validation_auc, error)
        error: 1.0 if successful, -1.0 if error occurred
    """
    try:
        train_accs, val_accs = [], []
        for X_train, y_train, X_validate, y_validate in cv_splits:
            model = RandomForestClassifier(**model_params, random_state=random_state)
            model.fit(X_train, y_train)
            if binary_class:
                train_accs.append(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
                val_accs.append(roc_auc_score(y_validate, model.predict_proba(X_validate)[:, 1]))
            else:
                train_accs.append(roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovo', labels=labels))
                val_accs.append(roc_auc_score(y_validate, model.predict_proba(X_validate), multi_class='ovo', labels=labels))
        return id, float(np.mean(train_accs)), float(np.mean(val_accs)), 1.0

    except Exception as e:
        print(f"Error in cv_random_forest: {e}")
        return id, 0.0, 0.0, -1.0

@ray.remote
def cv_kernel_svc(
    cv_splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    model_params: Dict[str, Any],
    random_state: int,
    id: int,
    binary_class: bool,
    labels: np.ndarray
) -> Tuple[int, float, float, float]:
    """
    Train and evaluate a SVC (Kernel SVM) across all CV folds using Ray.

    Parameters:
        cv_splits: List of (X_train, y_train, X_validate, y_validate) tuples for each fold
        model_params: Dictionary of hyperparameters for SVC
        random_state: Random seed for reproducibility
        id: Identifier for this model instance
        binary_class: True for binary classification, False for multi-class
        labels: Array of all possible class labels

    Returns:
        Tuple of (id, mean_training_auc, mean_validation_auc, error)
        error: 1.0 if successful, -1.0 if error occurred
    """
    try:
        train_accs, val_accs = [], []
        for X_train, y_train, X_validate, y_validate in cv_splits:
            model = SVC(**model_params, random_state=random_state, probability=True)
            model.fit(X_train, y_train)
            if binary_class:
                train_accs.append(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
                val_accs.append(roc_auc_score(y_validate, model.predict_proba(X_validate)[:, 1]))
            else:
                train_accs.append(roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovo', labels=labels))
                val_accs.append(roc_auc_score(y_validate, model.predict_proba(X_validate), multi_class='ovo', labels=labels))
        return id, float(np.mean(train_accs)), float(np.mean(val_accs)), 1.0

    except Exception as e:
        print(f"Error in cv_kernel_svc: {e}")
        return id, 0.0, 0.0, -1.0

@ray.remote
def cv_gradient_boost(
    cv_splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    model_params: Dict[str, Any],
    random_state: int,
    id: int,
    binary_class: bool,
    labels: np.ndarray
) -> Tuple[int, float, float, float]:
    """
    Train and evaluate a GradientBoostingClassifier across all CV folds using Ray.

    Parameters:
        cv_splits: List of (X_train, y_train, X_validate, y_validate) tuples for each fold
        model_params: Dictionary of hyperparameters for GradientBoostingClassifier
        random_state: Random seed for reproducibility
        id: Identifier for this model instance
        binary_class: True for binary classification, False for multi-class
        labels: Array of all possible class labels

    Returns:
        Tuple of (id, mean_training_auc, mean_validation_auc, error)
        error: 1.0 if successful, -1.0 if error occurred
    """
    try:
        train_accs, val_accs = [], []
        for X_train, y_train, X_validate, y_validate in cv_splits:
            model = GradientBoostingClassifier(**model_params, random_state=random_state)
            model.fit(X_train, y_train)
            if binary_class:
                train_accs.append(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
                val_accs.append(roc_auc_score(y_validate, model.predict_proba(X_validate)[:, 1]))
            else:
                train_accs.append(roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovo', labels=labels))
                val_accs.append(roc_auc_score(y_validate, model.predict_proba(X_validate), multi_class='ovo', labels=labels))
        return id, float(np.mean(train_accs)), float(np.mean(val_accs)), 1.0

    except Exception as e:
        print(f"Error in cv_gradient_boost: {e}")
        return id, 0.0, 0.0, -1.0

@ray.remote
def cv_knn(
    cv_splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    model_params: Dict[str, Any],
    random_state: int,
    id: int,
    binary_class: bool,
    labels: np.ndarray
) -> Tuple[int, float, float, float]:
    """
    Train and evaluate a KNeighborsClassifier across all CV folds using Ray.

    Parameters:
        cv_splits: List of (X_train, y_train, X_validate, y_validate) tuples for each fold
        model_params: Dictionary of hyperparameters for KNeighborsClassifier
        random_state: Random seed (unused for KNN but kept for consistent interface)
        id: Identifier for this model instance
        binary_class: True for binary classification, False for multi-class
        labels: Array of all possible class labels

    Returns:
        Tuple of (id, mean_training_auc, mean_validation_auc, error)
        error: 1.0 if successful, -1.0 if error occurred
    """
    try:
        train_accs, val_accs = [], []
        for X_train, y_train, X_validate, y_validate in cv_splits:
            model = KNeighborsClassifier(**model_params)
            model.fit(X_train, y_train)
            if binary_class:
                train_accs.append(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
                val_accs.append(roc_auc_score(y_validate, model.predict_proba(X_validate)[:, 1]))
            else:
                train_accs.append(roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovo', labels=labels))
                val_accs.append(roc_auc_score(y_validate, model.predict_proba(X_validate), multi_class='ovo', labels=labels))
        return id, float(np.mean(train_accs)), float(np.mean(val_accs)), 1.0

    except Exception as e:
        print(f"Error in cv_knn: {e}")
        return id, 0.0, 0.0, -1.0

@ray.remote
def cv_mlp(
    cv_splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    model_params: Dict[str, Any],
    random_state: int,
    id: int,
    binary_class: bool,
    labels: np.ndarray
) -> Tuple[int, float, float, float]:
    """
    Train and evaluate a MLPClassifier across all CV folds using Ray.

    Parameters:
        cv_splits: List of (X_train, y_train, X_validate, y_validate) tuples for each fold
        model_params: Dictionary of hyperparameters for MLPClassifier
        random_state: Random seed for reproducibility
        id: Identifier for this model instance
        binary_class: True for binary classification, False for multi-class
        labels: Array of all possible class labels

    Returns:
        Tuple of (id, mean_training_auc, mean_validation_auc, error)
        error: 1.0 if successful, -1.0 if error occurred
    """
    try:
        layers = (model_params.get('layer_1'),
                  model_params.get('layer_2'),
                  model_params.get('layer_3'),
                  model_params.get('layer_4'),
                  model_params.get('layer_5'))

        train_accs, val_accs = [], []
        for X_train, y_train, X_validate, y_validate in cv_splits:
            model = MLPClassifier(hidden_layer_sizes=layers,
                                  activation=model_params.get('activation'),
                                  solver=model_params.get('solver'),
                                  max_iter=model_params.get('max_iter'),
                                  random_state=random_state)
            model.fit(X_train, y_train)

            if binary_class:
                train_accs.append(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
                val_accs.append(roc_auc_score(y_validate, model.predict_proba(X_validate)[:, 1]))
            else:
                train_accs.append(roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovo', labels=labels))
                val_accs.append(roc_auc_score(y_validate, model.predict_proba(X_validate), multi_class='ovo', labels=labels))

        return id, float(np.mean(train_accs)), float(np.mean(val_accs)), 1.0

    except Exception as e:
        print(f"Error in cv_mlp: {e}")
        return id, 0.0, 0.0, -1.0