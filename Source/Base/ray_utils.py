import ray
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

@ray.remote
def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    model_params: Dict[str, Any],
    random_state: int,
    id: int
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

    Returns:
        Tuple of (id, training_accuracy, validation_accuracy, error)
        error: 1.0 if successful, -1.0 if error occurred
    """
    try:
        model = RandomForestClassifier(**model_params, random_state=random_state)
        model.fit(X_train, y_train)
        train_acc = float(accuracy_score(y_train, model.predict(X_train)))
        val_acc = float(accuracy_score(y_validate, model.predict(X_validate)))
        return id, train_acc, val_acc, 1.0

    except Exception as e:
        print(f"Error in train_decision_tree: {e}")
        return id, 0.0, 0.0, -1.0

@ray.remote
def train_linear_svc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    model_params: Dict[str, Any],
    random_state: int,
    id: int
) -> Tuple[int, float, float, float]:
    """
    Train and evaluate a LinearSVC using Ray.

    Parameters:
        X_train: Training features
        y_train: Training labels
        X_validate: Validation features
        y_validate: Validation labels
        model_params: Dictionary of hyperparameters for LinearSVC
        random_state: Random seed for reproducibility
        id: Identifier for this model instance

    Returns:
        Tuple of (id, training_accuracy, validation_accuracy, error)
        error: 1.0 if successful, -1.0 if error occurred
    """
    try:
        model = LinearSVC(**model_params, random_state=random_state)
        model.fit(X_train, y_train)
        train_acc = float(accuracy_score(y_train, model.predict(X_train)))
        val_acc = float(accuracy_score(y_validate, model.predict(X_validate)))
        return id, train_acc, val_acc, 1.0

    except Exception as e:
        print(f"Error in train_linear_svc: {e}")
        return id, 0.0, 0.0, -1.0

@ray.remote
def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    model_params: Dict[str, Any],
    random_state: int,
    id: int
) -> Tuple[int, float, float, float]:
    """
    Train and evaluate a DecisionTreeClassifier using Ray.

    Parameters:
        X_train: Training features
        y_train: Training labels
        X_validate: Validation features
        y_validate: Validation labels
        model_params: Dictionary of hyperparameters for DecisionTreeClassifier
        random_state: Random seed for reproducibility
        id: Identifier for this model instance

    Returns:
        Tuple of (id, training_accuracy, validation_accuracy, error)
        error: 1.0 if successful, -1.0 if error occurred
    """
    try:
        model = DecisionTreeClassifier(**model_params, random_state=random_state)
        model.fit(X_train, y_train)
        train_acc = float(accuracy_score(y_train, model.predict(X_train)))
        val_acc = float(accuracy_score(y_validate, model.predict(X_validate)))
        return id, train_acc, val_acc, 1.0

    except Exception as e:
        print(f"Error in train_decision_tree: {e}")
        return id, 0.0, 0.0, -1.0

@ray.remote
def train_kernel_svc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    model_params: Dict[str, Any],
    random_state: int,
    id: int
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

    Returns:
        Tuple of (id, training_accuracy, validation_accuracy, error)
        error: 1.0 if successful, -1.0 if error occurred
    """
    try:
        model = SVC(**model_params, random_state=random_state)
        model.fit(X_train, y_train)
        train_acc = float(accuracy_score(y_train, model.predict(X_train)))
        val_acc = float(accuracy_score(y_validate, model.predict(X_validate)))
        return id, train_acc, val_acc, 1.0

    except Exception as e:
        print(f"Error in train_kernel_svc: {e}")
        return id, 0.0, 0.0, -1.0

@ray.remote
def train_extra_trees(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    model_params: Dict[str, Any],
    random_state: int,
    id: int
) -> Tuple[int, float, float, float]:
    """
    Train and evaluate an ExtraTreesClassifier using Ray.

    Parameters:
        X_train: Training features
        y_train: Training labels
        X_validate: Validation features
        y_validate: Validation labels
        model_params: Dictionary of hyperparameters for ExtraTreesClassifier
        random_state: Random seed for reproducibility
        id: Identifier for this model instance

    Returns:
        Tuple of (id, training_accuracy, validation_accuracy, error)
        error: 1.0 if successful, -1.0 if error occurred
    """
    try:
        model = ExtraTreesClassifier(**model_params, random_state=random_state)
        model.fit(X_train, y_train)
        train_acc = float(accuracy_score(y_train, model.predict(X_train)))
        val_acc = float(accuracy_score(y_validate, model.predict(X_validate)))
        return id, train_acc, val_acc, 1.0

    except Exception as e:
        print(f"Error in train_decision_tree: {e}")
        return id, 0.0, 0.0, -1.0

@ray.remote
def train_gradient_boost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    model_params: Dict[str, Any],
    random_state: int,
    id: int
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

    Returns:
        Tuple of (id, training_accuracy, validation_accuracy, error)
        error: 1.0 if successful, -1.0 if error occurred
    """
    try:
        model = GradientBoostingClassifier(**model_params, random_state=random_state)
        model.fit(X_train, y_train)
        train_acc = float(accuracy_score(y_train, model.predict(X_train)))
        val_acc = float(accuracy_score(y_validate, model.predict(X_validate)))
        return id, train_acc, val_acc, 1.0

    except Exception as e:
        print(f"Error in train_gradient_boost: {e}")
        return id, 0.0, 0.0, -1.0

@ray.remote
def train_linear_sgd(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
    model_params: Dict[str, Any],
    random_state: int,
    id: int
) -> Tuple[int, float, float, float]:
    """
    Train and evaluate a SGDClassifier using Ray.

    Parameters:
        X_train: Training features
        y_train: Training labels
        X_validate: Validation features
        y_validate: Validation labels
        model_params: Dictionary of hyperparameters for SGDClassifier
        random_state: Random seed for reproducibility
        id: Identifier for this model instance

    Returns:
        Tuple of (id, training_accuracy, validation_accuracy, error)
        error: 1.0 if successful, -1.0 if error occurred
    """
    try:
        model = SGDClassifier(**model_params, random_state=random_state)
        model.fit(X_train, y_train)
        train_acc = float(accuracy_score(y_train, model.predict(X_train)))
        val_acc = float(accuracy_score(y_validate, model.predict(X_validate)))
        return id, train_acc, val_acc, 1.0

    except Exception as e:
        print(f"Error in train_linear_sgd: {e}")
        return id, 0.0, 0.0, -1.0


# ============================================================================
# Consolidated CV Functions - Process all 5 folds in a single task
# ============================================================================

@ray.remote
def train_random_forest_cv(
    cv_data: Tuple,
    model_params: Dict[str, Any],
    random_state: int,
    id: int
) -> Tuple[int, list, list, float]:
    """
    Train and evaluate RandomForestClassifier across all 5 CV folds in one task.

    Parameters:
        cv_data: Tuple of (X_train_f0, y_train_f0, X_val_f0, y_val_f0, ..., X_train_f4, y_train_f4, X_val_f4, y_val_f4)
        model_params: Dictionary of hyperparameters
        random_state: Random seed
        id: Identifier for this model instance

    Returns:
        Tuple of (id, list of train accuracies, list of val accuracies, error)
    """
    try:
        # Dereference Ray ObjectRefs if needed
        if isinstance(cv_data[0], ray.ObjectRef):
            cv_data = tuple(ray.get(list(cv_data)))

        train_accs = []
        val_accs = []

        # Process all 5 folds
        for fold in range(5):
            idx = fold * 4
            X_train = cv_data[idx]
            y_train = cv_data[idx + 1]
            X_val = cv_data[idx + 2]
            y_val = cv_data[idx + 3]

            model = RandomForestClassifier(**model_params, random_state=random_state)
            model.fit(X_train, y_train)
            train_acc = float(accuracy_score(y_train, model.predict(X_train)))
            val_acc = float(accuracy_score(y_val, model.predict(X_val)))

            train_accs.append(train_acc)
            val_accs.append(val_acc)

        return id, train_accs, val_accs, 1.0

    except Exception as e:
        print(f"Error in train_random_forest_cv: {e}")
        return id, [], [], -1.0


@ray.remote
def train_linear_svc_cv(
    cv_data: Tuple,
    model_params: Dict[str, Any],
    random_state: int,
    id: int
) -> Tuple[int, list, list, float]:
    """
    Train and evaluate LinearSVC across all 5 CV folds in one task.
    """
    try:
        # Dereference Ray ObjectRefs if needed
        if isinstance(cv_data[0], ray.ObjectRef):
            cv_data = tuple(ray.get(list(cv_data)))

        train_accs = []
        val_accs = []

        for fold in range(5):
            idx = fold * 4
            X_train = cv_data[idx]
            y_train = cv_data[idx + 1]
            X_val = cv_data[idx + 2]
            y_val = cv_data[idx + 3]

            model = LinearSVC(**model_params, random_state=random_state)
            model.fit(X_train, y_train)
            train_acc = float(accuracy_score(y_train, model.predict(X_train)))
            val_acc = float(accuracy_score(y_val, model.predict(X_val)))

            train_accs.append(train_acc)
            val_accs.append(val_acc)

        return id, train_accs, val_accs, 1.0

    except Exception as e:
        print(f"Error in train_linear_svc_cv: {e}")
        return id, [], [], -1.0


@ray.remote
def train_decision_tree_cv(
    cv_data: Tuple,
    model_params: Dict[str, Any],
    random_state: int,
    id: int
) -> Tuple[int, list, list, float]:
    """
    Train and evaluate DecisionTreeClassifier across all 5 CV folds in one task.
    """
    try:
        # Dereference Ray ObjectRefs if needed
        if isinstance(cv_data[0], ray.ObjectRef):
            cv_data = tuple(ray.get(list(cv_data)))

        train_accs = []
        val_accs = []

        for fold in range(5):
            idx = fold * 4
            X_train = cv_data[idx]
            y_train = cv_data[idx + 1]
            X_val = cv_data[idx + 2]
            y_val = cv_data[idx + 3]

            model = DecisionTreeClassifier(**model_params, random_state=random_state)
            model.fit(X_train, y_train)
            train_acc = float(accuracy_score(y_train, model.predict(X_train)))
            val_acc = float(accuracy_score(y_val, model.predict(X_val)))

            train_accs.append(train_acc)
            val_accs.append(val_acc)

        return id, train_accs, val_accs, 1.0

    except Exception as e:
        print(f"Error in train_decision_tree_cv: {e}")
        return id, [], [], -1.0


@ray.remote
def train_kernel_svc_cv(
    cv_data: Tuple,
    model_params: Dict[str, Any],
    random_state: int,
    id: int
) -> Tuple[int, list, list, float]:
    """
    Train and evaluate SVC (Kernel SVM) across all 5 CV folds in one task.
    """
    try:
        # Dereference Ray ObjectRefs if needed
        if isinstance(cv_data[0], ray.ObjectRef):
            cv_data = tuple(ray.get(list(cv_data)))

        train_accs = []
        val_accs = []

        for fold in range(5):
            idx = fold * 4
            X_train = cv_data[idx]
            y_train = cv_data[idx + 1]
            X_val = cv_data[idx + 2]
            y_val = cv_data[idx + 3]

            model = SVC(**model_params, random_state=random_state)
            model.fit(X_train, y_train)
            train_acc = float(accuracy_score(y_train, model.predict(X_train)))
            val_acc = float(accuracy_score(y_val, model.predict(X_val)))

            train_accs.append(train_acc)
            val_accs.append(val_acc)

        return id, train_accs, val_accs, 1.0

    except Exception as e:
        print(f"Error in train_kernel_svc_cv: {e}")
        return id, [], [], -1.0


@ray.remote
def train_extra_trees_cv(
    cv_data: Tuple,
    model_params: Dict[str, Any],
    random_state: int,
    id: int
) -> Tuple[int, list, list, float]:
    """
    Train and evaluate ExtraTreesClassifier across all 5 CV folds in one task.
    """
    try:
        # Dereference Ray ObjectRefs if needed
        if isinstance(cv_data[0], ray.ObjectRef):
            cv_data = tuple(ray.get(list(cv_data)))

        train_accs = []
        val_accs = []

        for fold in range(5):
            idx = fold * 4
            X_train = cv_data[idx]
            y_train = cv_data[idx + 1]
            X_val = cv_data[idx + 2]
            y_val = cv_data[idx + 3]

            model = ExtraTreesClassifier(**model_params, random_state=random_state)
            model.fit(X_train, y_train)
            train_acc = float(accuracy_score(y_train, model.predict(X_train)))
            val_acc = float(accuracy_score(y_val, model.predict(X_val)))

            train_accs.append(train_acc)
            val_accs.append(val_acc)

        return id, train_accs, val_accs, 1.0

    except Exception as e:
        print(f"Error in train_extra_trees_cv: {e}")
        return id, [], [], -1.0


@ray.remote
def train_gradient_boost_cv(
    cv_data: Tuple,
    model_params: Dict[str, Any],
    random_state: int,
    id: int
) -> Tuple[int, list, list, float]:
    """
    Train and evaluate GradientBoostingClassifier across all 5 CV folds in one task.
    """
    try:
        # Dereference Ray ObjectRefs if needed
        if isinstance(cv_data[0], ray.ObjectRef):
            cv_data = tuple(ray.get(list(cv_data)))

        train_accs = []
        val_accs = []

        for fold in range(5):
            idx = fold * 4
            X_train = cv_data[idx]
            y_train = cv_data[idx + 1]
            X_val = cv_data[idx + 2]
            y_val = cv_data[idx + 3]

            model = GradientBoostingClassifier(**model_params, random_state=random_state)
            model.fit(X_train, y_train)
            train_acc = float(accuracy_score(y_train, model.predict(X_train)))
            val_acc = float(accuracy_score(y_val, model.predict(X_val)))

            train_accs.append(train_acc)
            val_accs.append(val_acc)

        return id, train_accs, val_accs, 1.0

    except Exception as e:
        print(f"Error in train_gradient_boost_cv: {e}")
        return id, [], [], -1.0


@ray.remote
def train_linear_sgd_cv(
    cv_data: Tuple,
    model_params: Dict[str, Any],
    random_state: int,
    id: int
) -> Tuple[int, list, list, float]:
    """
    Train and evaluate SGDClassifier across all 5 CV folds in one task.
    """
    try:
        # Dereference Ray ObjectRefs if needed
        if isinstance(cv_data[0], ray.ObjectRef):
            cv_data = tuple(ray.get(list(cv_data)))

        train_accs = []
        val_accs = []

        for fold in range(5):
            idx = fold * 4
            X_train = cv_data[idx]
            y_train = cv_data[idx + 1]
            X_val = cv_data[idx + 2]
            y_val = cv_data[idx + 3]

            model = SGDClassifier(**model_params, random_state=random_state)
            model.fit(X_train, y_train)
            train_acc = float(accuracy_score(y_train, model.predict(X_train)))
            val_acc = float(accuracy_score(y_val, model.predict(X_val)))

            train_accs.append(train_acc)
            val_accs.append(val_acc)

        return id, train_accs, val_accs, 1.0

    except Exception as e:
        print(f"Error in train_linear_sgd_cv: {e}")
        return id, [], [], -1.0