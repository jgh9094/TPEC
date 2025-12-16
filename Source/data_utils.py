import os
import pickle
import pandas as pd
import numpy as np
import ray
from openml import tasks
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typeguard import typechecked
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from openml import tasks
from openml.study import get_suite

@typechecked
def get_ray_cv_splits(rep_dir: str, X_train: pd.DataFrame, y_train: np.ndarray, task_id: int):
    """
    Generate CV split file paths for a given replicate directory.
    Load them into ray and return the object references.

    Parameters:
    rep_dir : str
        Path to the replicate directory containing fold files

    X_train : np.ndarray
        Full training feature data from load_data function

    y_train : np.ndarray
        Full training labels from load_data function

    task_id : int
        The OpenML task ID (used to determine categorical features)

    Returns:
    Tuple of ray.ObjectRef:
        References to X_train, X_validate, y_train, y_validate for each fold
    """

    # CV fold 0 stuff
    X_train_f0, X_val_f0, y_train_f0, y_val_f0 = cv_data_splitter(
        X_train=X_train,
        y_train=y_train,
        fold_train_path=os.path.join(rep_dir, f"fold_train_{0}.pkl"),
        fold_validate_path=os.path.join(rep_dir, f"fold_validate_{0}.pkl"),
        task_id=task_id)

    # CV fold 1 stuff
    X_train_f1, X_val_f1, y_train_f1, y_val_f1 = cv_data_splitter(
        X_train=X_train,
        y_train=y_train,
        fold_train_path=os.path.join(rep_dir, f"fold_train_{1}.pkl"),
        fold_validate_path=os.path.join(rep_dir, f"fold_validate_{1}.pkl"),
        task_id=task_id)

    # CV fold 2 stuff
    X_train_f2, X_val_f2, y_train_f2, y_val_f2 = cv_data_splitter(
        X_train=X_train,
        y_train=y_train,
        fold_train_path=os.path.join(rep_dir, f"fold_train_{2}.pkl"),
        fold_validate_path=os.path.join(rep_dir, f"fold_validate_{2}.pkl"),
        task_id=task_id)

    # CV fold 3 stuff
    X_train_f3, X_val_f3, y_train_f3, y_val_f3 = cv_data_splitter(
        X_train=X_train,
        y_train=y_train,
        fold_train_path=os.path.join(rep_dir, f"fold_train_{3}.pkl"),
        fold_validate_path=os.path.join(rep_dir, f"fold_validate_{3}.pkl"),
        task_id=task_id)

    # CV fold 4 stuff
    X_train_f4, X_val_f4, y_train_f4, y_val_f4 = cv_data_splitter(
        X_train=X_train,
        y_train=y_train,
        fold_train_path=os.path.join(rep_dir, f"fold_train_{4}.pkl"),
        fold_validate_path=os.path.join(rep_dir, f"fold_validate_{4}.pkl"),
        task_id=task_id)

    return X_train_f0, X_val_f0, y_train_f0, y_val_f0, \
           X_train_f1, X_val_f1, y_train_f1, y_val_f1, \
           X_train_f2, X_val_f2, y_train_f2, y_val_f2, \
           X_train_f3, X_val_f3, y_train_f3, y_val_f3, \
           X_train_f4, X_val_f4, y_train_f4, y_val_f4

@typechecked
def get_suite_task_ids(suite_id: int = 271):
    """Return the list of task IDs in the given OpenML benchmark suite."""
    suite = get_suite(suite_id)  # OpenMLBenchmarkSuite object
    return suite.tasks  # list of task IDs (ints)

@typechecked
def is_binary_classification_task(task):
    """
    Check whether a given OpenML task is a binary classification task.
    Assumes the task is a supervised classification task.
    """
    # For classification tasks, there is a target feature with a finite set of class labels
    dataset = task.get_dataset()
    target_name = task.target_name

    # Load only the target column to inspect #classes without loading full data
    _, y, _, _ = dataset.get_data(
        target=target_name,
        dataset_format="dataframe"
    )

    # Drop missing labels if any to count actual classes
    unique_classes = pd.Series(y).dropna().unique()
    return len(unique_classes) == 2

@typechecked
def load_task_dataset(task):
    """
    Load the full dataset for a task into a pandas DataFrame.
    Returns (df, target_name, has_missing, minority_pct, majority_pct).
    """
    dataset = task.get_dataset()
    target_name = task.target_name

    X, y, _, _ = dataset.get_data(
        target=target_name,
        dataset_format="dataframe"
    )

    # Combine X and y into a single DataFrame
    df = X.copy()
    df[target_name] = y
    has_missing = df.isna().any().any()

    # Calculate class percentages
    class_counts = pd.Series(y).value_counts()
    total_count = len(y)
    minority_count = class_counts.min()
    majority_count = class_counts.max()
    minority_pct = (minority_count / total_count) * 100
    majority_pct = (majority_count / total_count) * 100

    return df, target_name, has_missing, minority_pct, majority_pct

@typechecked
def create_train_test_stratified_splits(
    task_id: int,
    data_dir: str,
    splits_dir: str,
    test_size: float,
    seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create stratified train/test splits for a given OpenML task and save indices to pickle files.

    Parameters:
    -----------
    task_id : int
        The OpenML task ID
    data_dir : str
        Directory containing task CSV files in format 'task_{task_id}.csv'
    splits_dir : str
        Directory where train.pkl and test.pkl will be saved
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    seed : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    train_indices : np.ndarray
        Indices for the training set
    test_indices : np.ndarray
        Indices for the test set
    """
    # Create splits directory if it doesn't exist
    os.makedirs(splits_dir, exist_ok=True)

    # Load the CSV data
    csv_path = os.path.join(data_dir, f"task_{task_id}.csv")
    df = pd.read_csv(csv_path)

    # Get task information to find the target column
    task = tasks.get_task(task_id)
    target_name = task.target_name

    # Get target values for stratification
    y = df[target_name]

    # Create indices array
    indices = np.arange(len(df))

    # Perform stratified split
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    # Save indices to pickle files
    train_pkl_path = os.path.join(splits_dir, "train.pkl")
    test_pkl_path = os.path.join(splits_dir, "test.pkl")

    with open(train_pkl_path, 'wb') as f:
        pickle.dump(train_indices, f)

    with open(test_pkl_path, 'wb') as f:
        pickle.dump(test_indices, f)

    print(f"Created splits for task {task_id}:")
    print(f"  Train size: {len(train_indices)} ({len(train_indices)/len(df)*100:.1f}%)")
    print(f"  Test size: {len(test_indices)} ({len(test_indices)/len(df)*100:.1f}%)")
    print(f"  Saved to: {splits_dir}")

    return train_indices, test_indices

@typechecked
def train_test_random_forrest(X_train: pd.DataFrame,
                              y_train: np.ndarray,
                              X_test: pd.DataFrame,
                              y_test: np.ndarray,
                              model_params: Dict[str, Any],
                              random_state: int) -> Tuple[float, float]:
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
    n_estimators : int
        Number of trees in the forest
    max_depth : int
        Maximum depth of the tree
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    train_accuracy : float
        Accuracy on the training set
    test_accuracy : float
        Accuracy on the testing set
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    clf = RandomForestClassifier(**model_params,random_state=random_state)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    train_accuracy = float(accuracy_score(y_train, y_train_pred))
    test_accuracy = float(accuracy_score(y_test, y_test_pred))

    return train_accuracy, test_accuracy

@typechecked
def load_data(
    task_id: int,
    data_dir: str,
    splits_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load data for a given OpenML task without preprocessing.

    Parameters:
    -----------
    task_id : int
        The OpenML task ID
    data_dir : str
        Directory containing task CSV files in format 'task_{task_id}.csv'
    splits_dir : str
        Directory path to the replicate folder containing train.pkl and test.pkl
        (e.g., '/path/to/Timing_Splits/task_146818/Replicate_0')

    Returns:
    --------
    X_train : pd.DataFrame
        Raw training features (not preprocessed)
    X_test : pd.DataFrame
        Raw testing features (not preprocessed)
    y_train : np.ndarray
        Training labels
    y_test : np.ndarray
        Testing labels
    """
    # Load the CSV data
    csv_path = os.path.join(data_dir, f"task_{task_id}.csv")
    df = pd.read_csv(csv_path)

    # Load train/test splits from the replicate-specific directory
    train_pkl_path = os.path.join(splits_dir, "train.pkl")
    test_pkl_path = os.path.join(splits_dir, "test.pkl")

    with open(train_pkl_path, 'rb') as f:
        train_indices = pickle.load(f)

    with open(test_pkl_path, 'rb') as f:
        test_indices = pickle.load(f)

    # Get task information from OpenML to determine categorical features
    task = tasks.get_task(task_id)
    target_name = task.target_name

    # Separate features and target
    X = df.drop(columns=[target_name])
    y = df[target_name]

    # Split into train and test
    X_train = X.iloc[train_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)
    y_train = y.iloc[train_indices].reset_index(drop=True).values
    y_test = y.iloc[test_indices].reset_index(drop=True).values

    return X_train, X_test, y_train, y_test

@typechecked
def cv_data_splitter(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    fold_train_path: str,
    fold_validate_path: str,
    task_id: int
) -> Tuple[ray.ObjectRef, ray.ObjectRef, ray.ObjectRef, ray.ObjectRef]:
    """
    Create cross-validation training and validation splits with preprocessing.

    This function takes the full training data and CV fold indices, creates
    train/validation subsets, applies preprocessing (OneHotEncoder + StandardScaler),
    and loads the transformed data into Ray for distributed computation.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Full training feature data from load_data function
    y_train : np.ndarray
        Full training labels from load_data function
    fold_train_path : str
        Path to pickle file containing training indices for this CV fold
        (e.g., '/path/to/Replicate_0/fold_train_0.pkl')
    fold_validate_path : str
        Path to pickle file containing validation indices for this CV fold
        (e.g., '/path/to/Replicate_0/fold_validate_0.pkl')
    task_id : int
        The OpenML task ID (used to determine categorical features)

    Returns:
    --------
    X_train_cv_ref : ray.ObjectRef
        Ray object reference to preprocessed CV training features
    X_val_cv_ref : ray.ObjectRef
        Ray object reference to preprocessed CV validation features
    y_train_cv_ref : ray.ObjectRef
        Ray object reference to CV training labels
    y_val_cv_ref : ray.ObjectRef
        Ray object reference to CV validation labels
    """
    # Load fold indices (these are indices in the original full dataset)
    with open(fold_train_path, 'rb') as f:
        fold_train_indices_original = pickle.load(f)

    with open(fold_validate_path, 'rb') as f:
        fold_validate_indices_original = pickle.load(f)

    # Get task information to determine categorical features
    task = tasks.get_task(task_id)
    dataset = task.get_dataset()
    target_name = task.target_name

    # Get feature type information
    _, _, categorical_indicator, _ = dataset.get_data(
        target=target_name,
        dataset_format="dataframe"
    )

    # Since X_train was created from the original dataset using train_indices,
    # we need to map the fold indices from original dataset space to X_train index space.
    # X_train has its own 0-indexed positions, but the fold files contain indices
    # referring to the original dataset.

    # We need to load the original train indices to create the mapping
    # The fold indices should be a subset of the train indices
    splits_dir = os.path.dirname(fold_train_path)
    train_pkl_path = os.path.join(splits_dir, "train.pkl")

    with open(train_pkl_path, 'rb') as f:
        original_train_indices = pickle.load(f)

    # Create a mapping from original dataset indices to X_train indices
    original_to_train_map = {orig_idx: train_idx for train_idx, orig_idx in enumerate(original_train_indices)}

    # Map fold indices from original space to X_train space
    fold_train_indices = np.array([original_to_train_map[idx] for idx in fold_train_indices_original])
    fold_validate_indices = np.array([original_to_train_map[idx] for idx in fold_validate_indices_original])

    # Create train/validation subsets using the mapped fold indices
    X_train_cv = X_train.iloc[fold_train_indices].reset_index(drop=True)
    X_val_cv = X_train.iloc[fold_validate_indices].reset_index(drop=True)
    y_train_cv = y_train[fold_train_indices]
    y_val_cv = y_train[fold_validate_indices]

    # Identify categorical and numerical columns
    categorical_cols = [col for col, is_cat in zip(X_train_cv.columns, categorical_indicator) if is_cat]
    numerical_cols = [col for col, is_cat in zip(X_train_cv.columns, categorical_indicator) if not is_cat]

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    # Fit on training fold and transform both train and validation
    X_train_cv_transformed = preprocessor.fit_transform(X_train_cv)
    X_val_cv_transformed = preprocessor.transform(X_val_cv)

    # Load transformed data into Ray
    X_train_cv_ref = ray.put(X_train_cv_transformed)
    X_val_cv_ref = ray.put(X_val_cv_transformed)
    y_train_cv_ref = ray.put(y_train_cv)
    y_val_cv_ref = ray.put(y_val_cv)

    return X_train_cv_ref, X_val_cv_ref, y_train_cv_ref, y_val_cv_ref