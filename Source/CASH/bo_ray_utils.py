##########################################################################################
#
# Class dedicarted to ray utils involving the Bayesian Optimization of hyperparameters for
# model paramters within Source/Base/model_param_space.py
#
##########################################################################################

import ray
from skopt import Optimizer
from skopt.space import Real, Categorical, Integer
from skopt.utils import point_asdict
from typing import Tuple, Dict, List, TypedDict, Any, Literal, Union
from scipy.stats import norm


# function that initializes the Bayesian Optimizer for RandomForestParams
# convert to a ray call

@ray.remote
def bo_rf_optimizer(param_space:Dict,
                    history:Dict,
                    n_initial_points:int,
                    candidates:Dict,
                    seed:int,
                    xi:float,
                    kappa:float) -> Tuple[List[Tuple[float, float, float]], str]:
    """
    Initialize the Bayesian Optimizer for RandomForestParams.
    Make predictions on the hyperparameters in evaluations.
    Return each hyperparameter's corresponding evaluation score.

    Args:
        param_space (Dict): The hyperparameter space for Random Forest.
        history (Dict): The history of evaluated hyperparameters and their scores.
        n_initial_points (int): The number of initial points to evaluate.
        candidates (Dict): The candidates for evaluation.
        seed (int): The random seed for reproducibility.
        xi (float): The parameter for the acquisition function.
        kappa (float): The parameter for the acquisition function.

    Returns:
        Tuple[List[Dict[str, float]], str]: A tuple containing:
            - List of dictionaries with keys 'ucb', 'ei', 'pi' for each candidate.
            - String indicating the model type ('rf').
    """


    # quick checks for param_space
    assert len(param_space) == 6, "param_space must match the expected number of hyperparameters for Random Forest being optimized."
    assert 'n_estimators' in param_space, "param_space must contain 'n_estimators'."
    assert 'criterion' in param_space, "param_space must contain 'criterion'."
    assert 'max_depth' in param_space, "param_space must contain 'max_depth'."
    assert 'max_features' in param_space, "param_space must contain 'max_features'."
    assert 'max_samples' in param_space, "param_space must contain 'max_samples'."
    assert 'class_weight' in param_space, "param_space must contain 'class_weight'."

    # same quick checks for history
    assert len(history) == 8, "history must match the expected number of hyperparameters for Random Forest being optimized."
    assert 'n_estimators' in history, "history must contain 'n_estimators'."
    assert 'criterion' in history, "history must contain 'criterion'."
    assert 'max_depth' in history, "history must contain 'max_depth'."
    assert 'max_features' in history, "history must contain 'max_features'."
    assert 'max_samples' in history, "history must contain 'max_samples'."
    assert 'class_weight' in history, "history must contain 'class_weight'."
    assert 'SOLUTION_VALIDATION_SCORE' in history, "history must contain 'SOLUTION_VALIDATION_SCORE'."
    assert 'SOLUTION_TRAIN_SCORE' in history, "history must contain 'SOLUTION_TRAIN_SCORE'."


    # Define the Hyperparameter Space for Random Forest
    search_space = {
        'n_estimators': Integer(param_space['n_estimators']['bounds'][0], param_space['n_estimators']['bounds'][1], prior='uniform'),
        'criterion': Categorical(param_space['criterion']['bounds']),
        'max_depth': Integer(param_space['max_depth']['bounds'][0], param_space['max_depth']['bounds'][1], prior='uniform'),
        'max_features': Real(param_space['max_features']['bounds'][0], param_space['max_features']['bounds'][1], prior='uniform'),
        'max_samples': Real(param_space['max_samples']['bounds'][0], param_space['max_samples']['bounds'][1], prior='uniform'),
        'class_weight': Categorical(param_space['class_weight']['bounds'])
    }
    # Convert to list format for Optimizer
    search_space_list = [search_space[k] for k in search_space.keys()]

    # Initialize the Bayesian Optimizer with GP and the specified acquisition function
    optimizer = Optimizer(
        dimensions=search_space_list,
        base_estimator="GP",
        # acq_func=acquisition, # not need this bc we are manually calculating the acquisition score below
        acq_optimizer="auto", # Minimizing -AUC is mathematically identical
        # acq_func_kwargs={"xi": 1.0, "kappa": 2.0},  # xi and kappa passed via acq_func_kwargs for EI, PI, and LCB
        n_initial_points=n_initial_points,   # Number of random points before GP fitting
        random_state=seed
    )

    # iterate through the history and tell the optimizer the results
    for n_estimators, criterion, max_depth, max_features, max_samples, class_weight, score in zip(
        history['n_estimators'],
        history['criterion'],
        history['max_depth'],
        history['max_features'],
        history['max_samples'],
        history['class_weight'],
        history['SOLUTION_VALIDATION_SCORE']):

        # Create a point in the same order as search_space_list
        point = [n_estimators, criterion, max_depth, max_features, max_samples, class_weight]
        # Store the actual score (we're maximizing, so negate for skopt which minimizes)
        optimizer.tell(point, -score)

    # iterate through the candidates and ask the optimizer for predictions
    performance_predictions = []
    for n_estimators, criterion, max_depth, max_features, max_samples, class_weight in zip(
        candidates['n_estimators'],
        candidates['criterion'],
        candidates['max_depth'],
        candidates['max_features'],
        candidates['max_samples'],
        candidates['class_weight']):

        # Create a point in the same order as search_space_list
        point = [n_estimators, criterion, max_depth, max_features, max_samples, class_weight]

        # fetch the lastest fitted GP model from the optimizer
        gp_model = optimizer.models[-1]

        # transform the point to the internal representation used by the GP model
        transformer = optimizer.space
        transformed_x = transformer.transform([point])

        # Ask the gp_model for the predicted mean and uncertainty (sigma)
        # Note: mu and sigma are in the negated space (since we store -score)
        mu, sigma = gp_model.predict(transformed_x, return_std=True)

        # Get the best observed value so far (maximization: min of negated scores)
        y_best = min(optimizer.yi)

        # Calculate the acquisition score based on the specified acquisition function
        # Upper Confidence Bound (converted back to maximization space: higher is better)
        # UCB = -mu + kappa*sigma (negate mu to get back to original space)
        ucb_score = -mu[0] + (kappa * sigma[0])

        # Expected Improvement (for maximization via minimization of negated scores)
        if sigma[0] > 0:
            z = (y_best - mu[0] - xi) / sigma[0]
            ei_score = (y_best - mu[0] - xi) * norm.cdf(z) + sigma[0] * norm.pdf(z)
            # EI should be non-negative; clamp to 0 if negative due to numerical issues
            ei_score = max(0.0, ei_score)
        else:
            ei_score = 0.0

        # Probability of Improvement (for maximization via minimization of negated scores)
        if sigma[0] > 0:
            z = (y_best - mu[0] - xi) / sigma[0]
            pi_score = norm.cdf(z)
            # PI is a probability (CDF), should be in [0,1]; clamp to 0 if negative due to numerical issues
            pi_score = max(0.0, pi_score)
        else:
            pi_score = 0.0

        performance_predictions.append({'ucb': ucb_score, 'ei': ei_score, 'pi': pi_score})

    return performance_predictions, "rf"

@ray.remote
def bo_ksvc_optimizer(param_space:Dict,
                      history:Dict,
                      n_initial_points:int,
                      candidates:Dict,
                      seed:int,
                      xi:float,
                      kappa:float) -> Tuple[List[Dict[str, float]], str]:
    """
    Initialize the Bayesian Optimizer for KernelSVCParams.
    Make predictions on the hyperparameters in evaluations.
    Return each hyperparameter's corresponding evaluation score.

    Args:
        param_space (Dict): The hyperparameter space for Kernel SVC.
        history (Dict): The history of evaluated hyperparameters and their scores.
        n_initial_points (int): The number of initial points to evaluate.
        candidates (Dict): The candidates for evaluation.
        seed (int): The random seed for reproducibility.
        xi (float): The parameter for the acquisition function.
        kappa (float): The parameter for the acquisition function.

    Returns:
        Tuple[List[Dict[str, float]], str]: A tuple containing:
            - List of dictionaries with keys 'ucb', 'ei', 'pi' for each candidate.
            - String indicating the model type ('ksvc').
    """

    # quick checks for param_space
    assert len(param_space) == 5, "param_space must match the expected number of hyperparameters for Kernel SVC being optimized."
    assert 'C' in param_space, "param_space must contain 'C'."
    assert 'kernel' in param_space, "param_space must contain 'kernel'."
    assert 'max_iter' in param_space, "param_space must contain 'max_iter'."
    assert 'class_weight' in param_space, "param_space must contain 'class_weight'."
    assert 'decision_function_shape' in param_space, "param_space must contain 'decision_function_shape'."

    # same quick checks for history
    assert len(history) == 7, "history must match the expected number of hyperparameters for Kernel SVC being optimized."
    assert 'C' in history, "history must contain 'C'."
    assert 'kernel' in history, "history must contain 'kernel'."
    assert 'max_iter' in history, "history must contain 'max_iter'."
    assert 'class_weight' in history, "history must contain 'class_weight'."
    assert 'decision_function_shape' in history, "history must contain 'decision_function_shape'."
    assert 'SOLUTION_VALIDATION_SCORE' in history, "history must contain 'SOLUTION_VALIDATION_SCORE'."
    assert 'SOLUTION_TRAIN_SCORE' in history, "history must contain 'SOLUTION_TRAIN_SCORE'."

    # Define the Hyperparameter Space for Kernel SVC
    search_space = {
        'C': Real(param_space['C']['bounds'][0], param_space['C']['bounds'][1], prior='uniform'),
        'kernel': Categorical(param_space['kernel']['bounds']),
        'max_iter': Integer(param_space['max_iter']['bounds'][0], param_space['max_iter']['bounds'][1], prior='uniform'),
        'class_weight': Categorical(param_space['class_weight']['bounds']),
        'decision_function_shape': Categorical(param_space['decision_function_shape']['bounds'])
    }
    # Convert to list format for Optimizer
    search_space_list = [search_space[k] for k in search_space.keys()]

    # Initialize the Bayesian Optimizer with GP
    optimizer = Optimizer(
        dimensions=search_space_list,
        base_estimator="GP",
        n_initial_points=n_initial_points,
        random_state=seed
    )

    # iterate through the history and tell the optimizer the results
    for C, kernel, max_iter, class_weight, decision_function_shape, score in zip(
        history['C'],
        history['kernel'],
        history['max_iter'],
        history['class_weight'],
        history['decision_function_shape'],
        history['SOLUTION_VALIDATION_SCORE']):

        # Create a point in the same order as search_space_list
        point = [C, kernel, max_iter, class_weight, decision_function_shape]
        # Store the actual score (we're maximizing, so negate for skopt which minimizes)
        optimizer.tell(point, -score)

    # iterate through the candidates and ask the optimizer for predictions
    performance_predictions = []
    for C, kernel, max_iter, class_weight, decision_function_shape in zip(
        candidates['C'],
        candidates['kernel'],
        candidates['max_iter'],
        candidates['class_weight'],
        candidates['decision_function_shape']):

        # Create a point in the same order as search_space_list
        point = [C, kernel, max_iter, class_weight, decision_function_shape]

        # fetch the lastest fitted GP model from the optimizer
        gp_model = optimizer.models[-1]

        # transform the point to the internal representation used by the GP model
        transformer = optimizer.space
        transformed_x = transformer.transform([point])

        # Ask the gp_model for the predicted mean and uncertainty (sigma)
        # Note: mu and sigma are in the negated space (since we store -score)
        mu, sigma = gp_model.predict(transformed_x, return_std=True)

        # Get the best observed value so far (maximization: min of negated scores)
        y_best = min(optimizer.yi)

        # Calculate the acquisition score based on the specified acquisition function
        # Upper Confidence Bound (converted back to maximization space: higher is better)
        # UCB = -mu + kappa*sigma (negate mu to get back to original space)
        ucb_score = -mu[0] + (kappa * sigma[0])

        # Expected Improvement (for maximization via minimization of negated scores)
        if sigma[0] > 0:
            z = (y_best - mu[0] - xi) / sigma[0]
            ei_score = (y_best - mu[0] - xi) * norm.cdf(z) + sigma[0] * norm.pdf(z)
            # EI should be non-negative; clamp to 0 if negative due to numerical issues
            ei_score = max(0.0, ei_score)
        else:
            ei_score = 0.0

        # Probability of Improvement (for maximization via minimization of negated scores)
        if sigma[0] > 0:
            z = (y_best - mu[0] - xi) / sigma[0]
            pi_score = norm.cdf(z)
            # PI is a probability (CDF), should be in [0,1]; clamp to 0 if negative due to numerical issues
            pi_score = max(0.0, pi_score)
        else:
            pi_score = 0.0

        performance_predictions.append({'ucb': ucb_score, 'ei': ei_score, 'pi': pi_score})

    return performance_predictions, "ksvc"

@ray.remote
def bo_gb_optimizer(param_space:Dict,
                    history:Dict,
                    n_initial_points:int,
                    candidates:Dict,
                    seed:int,
                    xi:float,
                    kappa:float) -> Tuple[List[Dict[str, float]], str]:
    """
    Initialize the Bayesian Optimizer for GradientBoostParams.
    Make predictions on the hyperparameters in evaluations.
    Return each hyperparameter's corresponding evaluation score.

    Args:
        param_space (Dict): The hyperparameter space for Gradient Boosting.
        history (Dict): The history of evaluated hyperparameters and their scores.
        n_initial_points (int): The number of initial points to evaluate.
        candidates (Dict): The candidates for evaluation.
        seed (int): The random seed for reproducibility.
        xi (float): The parameter for the acquisition function.
        kappa (float): The parameter for the acquisition function.

    Returns:
        Tuple[List[Dict[str, float]], str]: A tuple containing:
            - List of dictionaries with keys 'ucb', 'ei', 'pi' for each candidate.
            - String indicating the model type ('gb').
    """

    # quick checks for param_space
    assert len(param_space) == 7, "param_space must match the expected number of hyperparameters for Gradient Boosting being optimized."
    assert 'loss' in param_space, "param_space must contain 'loss'."
    assert 'learning_rate' in param_space, "param_space must contain 'learning_rate'."
    assert 'n_estimators' in param_space, "param_space must contain 'n_estimators'."
    assert 'subsample' in param_space, "param_space must contain 'subsample'."
    assert 'criterion' in param_space, "param_space must contain 'criterion'."
    assert 'max_depth' in param_space, "param_space must contain 'max_depth'."
    assert 'max_features' in param_space, "param_space must contain 'max_features'."

    # same quick checks for history
    assert len(history) == 9, "history must match the expected number of hyperparameters for Gradient Boosting being optimized."
    assert 'loss' in history, "history must contain 'loss'."
    assert 'learning_rate' in history, "history must contain 'learning_rate'."
    assert 'n_estimators' in history, "history must contain 'n_estimators'."
    assert 'subsample' in history, "history must contain 'subsample'."
    assert 'criterion' in history, "history must contain 'criterion'."
    assert 'max_depth' in history, "history must contain 'max_depth'."
    assert 'max_features' in history, "history must contain 'max_features'."
    assert 'SOLUTION_VALIDATION_SCORE' in history, "history must contain 'SOLUTION_VALIDATION_SCORE'."
    assert 'SOLUTION_TRAIN_SCORE' in history, "history must contain 'SOLUTION_TRAIN_SCORE'."

    # Define the Hyperparameter Space for Gradient Boosting
    search_space = {
        'loss': Categorical(param_space['loss']['bounds']),
        'learning_rate': Real(param_space['learning_rate']['bounds'][0], param_space['learning_rate']['bounds'][1], prior='uniform'),
        'n_estimators': Integer(param_space['n_estimators']['bounds'][0], param_space['n_estimators']['bounds'][1], prior='uniform'),
        'subsample': Real(param_space['subsample']['bounds'][0], param_space['subsample']['bounds'][1], prior='uniform'),
        'criterion': Categorical(param_space['criterion']['bounds']),
        'max_depth': Integer(param_space['max_depth']['bounds'][0], param_space['max_depth']['bounds'][1], prior='uniform'),
        'max_features': Real(param_space['max_features']['bounds'][0], param_space['max_features']['bounds'][1], prior='uniform')
    }
    # Convert to list format for Optimizer
    search_space_list = [search_space[k] for k in search_space.keys()]

    # Initialize the Bayesian Optimizer with GP
    optimizer = Optimizer(
        dimensions=search_space_list,
        base_estimator="GP",
        n_initial_points=n_initial_points,
        random_state=seed
    )

    # iterate through the history and tell the optimizer the results
    for loss, learning_rate, n_estimators, subsample, criterion, max_depth, max_features, score in zip(
        history['loss'],
        history['learning_rate'],
        history['n_estimators'],
        history['subsample'],
        history['criterion'],
        history['max_depth'],
        history['max_features'],
        history['SOLUTION_VALIDATION_SCORE']):

        # Create a point in the same order as search_space_list
        point = [loss, learning_rate, n_estimators, subsample, criterion, max_depth, max_features]
        # Store the actual score (we're maximizing, so negate for skopt which minimizes)
        optimizer.tell(point, -score)

    # iterate through the candidates and ask the optimizer for predictions
    performance_predictions = []
    for loss, learning_rate, n_estimators, subsample, criterion, max_depth, max_features in zip(
        candidates['loss'],
        candidates['learning_rate'],
        candidates['n_estimators'],
        candidates['subsample'],
        candidates['criterion'],
        candidates['max_depth'],
        candidates['max_features']):

        # Create a point in the same order as search_space_list
        point = [loss, learning_rate, n_estimators, subsample, criterion, max_depth, max_features]

        # fetch the lastest fitted GP model from the optimizer
        gp_model = optimizer.models[-1]

        # transform the point to the internal representation used by the GP model
        transformer = optimizer.space
        transformed_x = transformer.transform([point])

        # Ask the gp_model for the predicted mean and uncertainty (sigma)
        # Note: mu and sigma are in the negated space (since we store -score)
        mu, sigma = gp_model.predict(transformed_x, return_std=True)

        # Get the best observed value so far (maximization: min of negated scores)
        y_best = min(optimizer.yi)

        # Calculate the acquisition score based on the specified acquisition function
        # Upper Confidence Bound (converted back to maximization space: higher is better)
        # UCB = -mu + kappa*sigma (negate mu to get back to original space)
        ucb_score = -mu[0] + (kappa * sigma[0])

        # Expected Improvement (for maximization via minimization of negated scores)
        if sigma[0] > 0:
            z = (y_best - mu[0] - xi) / sigma[0]
            ei_score = (y_best - mu[0] - xi) * norm.cdf(z) + sigma[0] * norm.pdf(z)
            # EI should be non-negative; clamp to 0 if negative due to numerical issues
            ei_score = max(0.0, ei_score)
        else:
            ei_score = 0.0

        # Probability of Improvement (for maximization via minimization of negated scores)
        if sigma[0] > 0:
            z = (y_best - mu[0] - xi) / sigma[0]
            pi_score = norm.cdf(z)
            # PI is a probability (CDF), should be in [0,1]; clamp to 0 if negative due to numerical issues
            pi_score = max(0.0, pi_score)
        else:
            pi_score = 0.0

        performance_predictions.append({'ucb': ucb_score, 'ei': ei_score, 'pi': pi_score})

    return performance_predictions, "gb"

@ray.remote
def bo_knn_optimizer(param_space:Dict,
                     history:Dict,
                     n_initial_points:int,
                     candidates:Dict,
                     seed:int,
                     xi:float,
                     kappa:float) -> Tuple[List[Dict[str, float]], str]:
    """
    Initialize the Bayesian Optimizer for KNeighborsClassifierParams.
    Make predictions on the hyperparameters in evaluations.
    Return each hyperparameter's corresponding evaluation score.

    Args:
        param_space (Dict): The hyperparameter space for K-Nearest Neighbors.
        history (Dict): The history of evaluated hyperparameters and their scores.
        n_initial_points (int): The number of initial points to evaluate.
        candidates (Dict): The candidates for evaluation.
        seed (int): The random seed for reproducibility.
        xi (float): The parameter for the acquisition function.
        kappa (float): The parameter for the acquisition function.

    Returns:
        Tuple[List[Dict[str, float]], str]: A tuple containing:
            - List of dictionaries with keys 'ucb', 'ei', 'pi' for each candidate.
            - String indicating the model type ('knn').
    """

    # quick checks for param_space
    assert len(param_space) == 5, "param_space must match the expected number of hyperparameters for KNN being optimized."
    assert 'n_neighbors' in param_space, "param_space must contain 'n_neighbors'."
    assert 'weights' in param_space, "param_space must contain 'weights'."
    assert 'algorithm' in param_space, "param_space must contain 'algorithm'."
    assert 'leaf_size' in param_space, "param_space must contain 'leaf_size'."
    assert 'p' in param_space, "param_space must contain 'p'."

    # same quick checks for history
    assert len(history) == 7, "history must match the expected number of hyperparameters for KNN being optimized."
    assert 'n_neighbors' in history, "history must contain 'n_neighbors'."
    assert 'weights' in history, "history must contain 'weights'."
    assert 'algorithm' in history, "history must contain 'algorithm'."
    assert 'leaf_size' in history, "history must contain 'leaf_size'."
    assert 'p' in history, "history must contain 'p'."
    assert 'SOLUTION_TRAIN_SCORE' in history, "history must contain 'SOLUTION_TRAIN_SCORE'."
    assert 'SOLUTION_VALIDATION_SCORE' in history, "history must contain 'SOLUTION_VALIDATION_SCORE'."

    # Define the Hyperparameter Space for KNN
    search_space = {
        'n_neighbors': Integer(param_space['n_neighbors']['bounds'][0], param_space['n_neighbors']['bounds'][1], prior='uniform'),
        'weights': Categorical(param_space['weights']['bounds']),
        'algorithm': Categorical(param_space['algorithm']['bounds']),
        'leaf_size': Integer(param_space['leaf_size']['bounds'][0], param_space['leaf_size']['bounds'][1], prior='uniform'),
        'p': Integer(param_space['p']['bounds'][0], param_space['p']['bounds'][1], prior='uniform')
    }
    # Convert to list format for Optimizer
    search_space_list = [search_space[k] for k in search_space.keys()]

    # Initialize the Bayesian Optimizer with GP
    optimizer = Optimizer(
        dimensions=search_space_list,
        base_estimator="GP",
        n_initial_points=n_initial_points,
        random_state=seed
    )

    # iterate through the history and tell the optimizer the results
    for n_neighbors, weights, algorithm, leaf_size, p, score in zip(
        history['n_neighbors'],
        history['weights'],
        history['algorithm'],
        history['leaf_size'],
        history['p'],
        history['SOLUTION_VALIDATION_SCORE']):

        # Create a point in the same order as search_space_list
        point = [n_neighbors, weights, algorithm, leaf_size, p]
        # Store the actual score (we're maximizing, so negate for skopt which minimizes)
        optimizer.tell(point, -score)

    # iterate through the candidates and ask the optimizer for predictions
    performance_predictions = []
    for n_neighbors, weights, algorithm, leaf_size, p in zip(
        candidates['n_neighbors'],
        candidates['weights'],
        candidates['algorithm'],
        candidates['leaf_size'],
        candidates['p']):

        # Create a point in the same order as search_space_list
        point = [n_neighbors, weights, algorithm, leaf_size, p]

        # fetch the lastest fitted GP model from the optimizer
        gp_model = optimizer.models[-1]

        # transform the point to the internal representation used by the GP model
        transformer = optimizer.space
        transformed_x = transformer.transform([point])

        # Ask the gp_model for the predicted mean and uncertainty (sigma)
        # Note: mu and sigma are in the negated space (since we store -score)
        mu, sigma = gp_model.predict(transformed_x, return_std=True)

        # Get the best observed value so far (maximization: min of negated scores)
        y_best = min(optimizer.yi)

        # Calculate the acquisition score based on the specified acquisition function
        # Upper Confidence Bound (converted back to maximization space: higher is better)
        # UCB = -mu + kappa*sigma (negate mu to get back to original space)
        ucb_score = -mu[0] + (kappa * sigma[0])

        # Expected Improvement (for maximization via minimization of negated scores)
        if sigma[0] > 0:
            z = (y_best - mu[0] - xi) / sigma[0]
            ei_score = (y_best - mu[0] - xi) * norm.cdf(z) + sigma[0] * norm.pdf(z)
            # EI should be non-negative; clamp to 0 if negative due to numerical issues
            ei_score = max(0.0, ei_score)
        else:
            ei_score = 0.0

        # Probability of Improvement (for maximization via minimization of negated scores)
        if sigma[0] > 0:
            z = (y_best - mu[0] - xi) / sigma[0]
            pi_score = norm.cdf(z)
            # PI is a probability (CDF), should be in [0,1]; clamp to 0 if negative due to numerical issues
            pi_score = max(0.0, pi_score)
        else:
            pi_score = 0.0

        performance_predictions.append({'ucb': ucb_score, 'ei': ei_score, 'pi': pi_score})

    return performance_predictions, "knn"

@ray.remote
def bo_mlp_optimizer(param_space:Dict,
                     history:Dict,
                     n_initial_points:int,
                     candidates:Dict,
                     seed:int,
                     xi:float,
                     kappa:float) -> Tuple[List[Dict[str, float]], str]:
    """
    Initialize the Bayesian Optimizer for MLPClassifierParams.
    Make predictions on the hyperparameters in evaluations.
    Return each hyperparameter's corresponding evaluation score.

    Args:
        param_space (Dict): The hyperparameter space for MLP Classifier.
        history (Dict): The history of evaluated hyperparameters and their scores.
        n_initial_points (int): The number of initial points to evaluate.
        candidates (Dict): The candidates for evaluation.
        seed (int): The random seed for reproducibility.
        xi (float): The parameter for the acquisition function.
        kappa (float): The parameter for the acquisition function.

    Returns:
        Tuple[List[Dict[str, float]], str]: A tuple containing:
            - List of dictionaries with keys 'ucb', 'ei', 'pi' for each candidate.
            - String indicating the model type ('mlp').
    """

    # quick checks for param_space
    assert len(param_space) == 8, "param_space must match the expected number of hyperparameters for MLP being optimized."
    assert 'layer_1' in param_space, "param_space must contain 'layer_1'."
    assert 'layer_2' in param_space, "param_space must contain 'layer_2'."
    assert 'layer_3' in param_space, "param_space must contain 'layer_3'."
    assert 'layer_4' in param_space, "param_space must contain 'layer_4'."
    assert 'layer_5' in param_space, "param_space must contain 'layer_5'."
    assert 'activation' in param_space, "param_space must contain 'activation'."
    assert 'solver' in param_space, "param_space must contain 'solver'."
    assert 'max_iter' in param_space, "param_space must contain 'max_iter'."

    # same quick checks for history
    assert len(history) == 10, "history must match the expected number of hyperparameters for MLP being optimized."
    assert 'layer_1' in history, "history must contain 'layer_1'."
    assert 'layer_2' in history, "history must contain 'layer_2'."
    assert 'layer_3' in history, "history must contain 'layer_3'."
    assert 'layer_4' in history, "history must contain 'layer_4'."
    assert 'layer_5' in history, "history must contain 'layer_5'."
    assert 'activation' in history, "history must contain 'activation'."
    assert 'solver' in history, "history must contain 'solver'."
    assert 'max_iter' in history, "history must contain 'max_iter'."
    assert 'SOLUTION_VALIDATION_SCORE' in history, "history must contain 'SOLUTION_VALIDATION_SCORE'."
    assert 'SOLUTION_TRAIN_SCORE' in history, "history must contain 'SOLUTION_TRAIN_SCORE'."

    # Define the Hyperparameter Space for MLP
    search_space = {
        'layer_1': Integer(param_space['layer_1']['bounds'][0], param_space['layer_1']['bounds'][1], prior='uniform'),
        'layer_2': Integer(param_space['layer_2']['bounds'][0], param_space['layer_2']['bounds'][1], prior='uniform'),
        'layer_3': Integer(param_space['layer_3']['bounds'][0], param_space['layer_3']['bounds'][1], prior='uniform'),
        'layer_4': Integer(param_space['layer_4']['bounds'][0], param_space['layer_4']['bounds'][1], prior='uniform'),
        'layer_5': Integer(param_space['layer_5']['bounds'][0], param_space['layer_5']['bounds'][1], prior='uniform'),
        'activation': Categorical(param_space['activation']['bounds']),
        'solver': Categorical(param_space['solver']['bounds']),
        'max_iter': Integer(param_space['max_iter']['bounds'][0], param_space['max_iter']['bounds'][1], prior='uniform')
    }
    # Convert to list format for Optimizer
    search_space_list = [search_space[k] for k in search_space.keys()]

    # Initialize the Bayesian Optimizer with GP
    optimizer = Optimizer(
        dimensions=search_space_list,
        base_estimator="GP",
        n_initial_points=n_initial_points,
        random_state=seed
    )

    # iterate through the history and tell the optimizer the results
    for layer_1, layer_2, layer_3, layer_4, layer_5, activation, solver, max_iter, score in zip(
        history['layer_1'],
        history['layer_2'],
        history['layer_3'],
        history['layer_4'],
        history['layer_5'],
        history['activation'],
        history['solver'],
        history['max_iter'],
        history['SOLUTION_VALIDATION_SCORE']):

        # Create a point in the same order as search_space_list
        point = [layer_1, layer_2, layer_3, layer_4, layer_5, activation, solver, max_iter]
        # Store the actual score (we're maximizing, so negate for skopt which minimizes)
        optimizer.tell(point, -score)

    # iterate through the candidates and ask the optimizer for predictions
    performance_predictions = []
    for layer_1, layer_2, layer_3, layer_4, layer_5, activation, solver, max_iter in zip(
        candidates['layer_1'],
        candidates['layer_2'],
        candidates['layer_3'],
        candidates['layer_4'],
        candidates['layer_5'],
        candidates['activation'],
        candidates['solver'],
        candidates['max_iter']):

        # Create a point in the same order as search_space_list
        point = [layer_1, layer_2, layer_3, layer_4, layer_5, activation, solver, max_iter]

        # fetch the lastest fitted GP model from the optimizer
        gp_model = optimizer.models[-1]

        # transform the point to the internal representation used by the GP model
        transformer = optimizer.space
        transformed_x = transformer.transform([point])

        # Ask the gp_model for the predicted mean and uncertainty (sigma)
        # Note: mu and sigma are in the negated space (since we store -score)
        mu, sigma = gp_model.predict(transformed_x, return_std=True)

        # Get the best observed value so far (maximization: min of negated scores)
        y_best = min(optimizer.yi)

        # Calculate the acquisition score based on the specified acquisition function
        # Upper Confidence Bound (converted back to maximization space: higher is better)
        # UCB = -mu + kappa*sigma (negate mu to get back to original space)
        ucb_score = -mu[0] + (kappa * sigma[0])

        # Expected Improvement (for maximization via minimization of negated scores)
        if sigma[0] > 0:
            z = (y_best - mu[0] - xi) / sigma[0]
            ei_score = (y_best - mu[0] - xi) * norm.cdf(z) + sigma[0] * norm.pdf(z)
            # EI should be non-negative; clamp to 0 if negative due to numerical issues
            ei_score = max(0.0, ei_score)
        else:
            ei_score = 0.0

        # Probability of Improvement (for maximization via minimization of negated scores)
        if sigma[0] > 0:
            z = (y_best - mu[0] - xi) / sigma[0]
            pi_score = norm.cdf(z)
            # PI is a probability (CDF), should be in [0,1]; clamp to 0 if negative due to numerical issues
            pi_score = max(0.0, pi_score)
        else:
            pi_score = 0.0

        performance_predictions.append({'ucb': ucb_score, 'ei': ei_score, 'pi': pi_score})

    return performance_predictions, "mlp"