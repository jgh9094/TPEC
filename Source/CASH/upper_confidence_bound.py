"""
Upper Confidence Bound (UCB) Acquisition Function for Hyperparameter Optimization.

This implementation is based on the UCB1 algorithm for multi-armed bandits, adapted
for hyperparameter optimization in the context of Combined Algorithm Selection and
Hyperparameter Optimization (CASH).

References:
    Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002).
    Finite-time analysis of the multiarmed bandit problem.
    Machine Learning, 47(2-3), 235-256.
    https://doi.org/10.1023/A:1013689704352

    Brochu, E., Cora, V. M., & De Freitas, N. (2010).
    A tutorial on Bayesian optimization of expensive cost functions, with application
    to active user modeling and hierarchical reinforcement learning.
    arXiv preprint arXiv:1012.2599.
"""
import numpy as np
from Source.Base.individual import Individual
from Source.Base.model_param_space import ModelParams
from typing import List, Dict, Optional, Any
from typeguard import typechecked

@typechecked
class UCB:
    """
    Upper Confidence Bound (UCB) acquisition function for hyperparameter optimization.
    This class supports both categorical and numeric parameters, computing UCB scores
    based on observed performances to balance exploration and exploitation.

    The UCB score is computed as: UCB = mean + kappa * sqrt(log(N) / n)
    where N is the total number of evaluations and n is the number of times
    a particular parameter value has been evaluated.
    """
    def __init__(self, kappa: float = 2.0):
        """
        Parameters:
            kappa (float): Exploration parameter controlling the trade-off between
                          exploration and exploitation. Higher values favor exploration.
        """
        self.kappa = kappa

        # Statistics for computing UCB scores
        self.param_means: Dict[str, Dict] = {}  # Mean performance per parameter value
        self.param_counts: Dict[str, Dict] = {}  # Visit counts per parameter value
        self.total_evaluations = 0
        self.best_performance = float('-inf')

    def fit(self, samples: List[Individual], param_space: ModelParams) -> None:
        """
        Fit UCB statistics based on observed samples.
        Computes mean performance and visit counts for each parameter value.

        Parameters:
            samples (List[Individual]): List of evaluated individuals.
            param_space (ModelParams): The parameter space definition.
        """
        if len(samples) < 1:
            raise RuntimeError("Need at least 1 sample before UCB can fit.")

        # Reset statistics
        self.param_means.clear()
        self.param_counts.clear()
        self.total_evaluations = len(samples)
        self.best_performance = max(ind.get_val_performance() for ind in samples)

        # Initialize statistics dictionaries for each parameter
        for param_name in param_space.param_space.keys():
            self.param_means[param_name] = {}
            self.param_counts[param_name] = {}

        # Accumulate performance statistics for each parameter value
        for ind in samples:
            params = ind.get_params()
            performance = ind.get_val_performance()

            for param_name, param_value in params.items():
                # Convert value to hashable key
                key = self._make_hashable(param_value)

                # Initialize if first occurrence
                if key not in self.param_counts[param_name]:
                    self.param_counts[param_name][key] = 0
                    self.param_means[param_name][key] = 0.0

                # Update statistics using incremental mean formula
                count = self.param_counts[param_name][key]
                old_mean = self.param_means[param_name][key]
                self.param_counts[param_name][key] = count + 1
                self.param_means[param_name][key] = old_mean + (performance - old_mean) / (count + 1)

        return

    def _make_hashable(self, value) -> str:
        """
        Convert parameter value to hashable string representation.

        Parameters:
            value: Parameter value (int, float, str, bool, or None)

        Returns:
            str: Hashable string representation
        """
        if value is None:
            return "None"
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float)):
            return str(value)
        return str(value)

    def _compute_ucb_score(self, param_name: str, param_value: Any, numeric_info: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute UCB score for a single parameter value.
        UCB = mean + kappa * sqrt(log(N) / n)
        where N is total evaluations and n is visits to this parameter value.

        For unseen values:
        - Categorical/boolean: return optimistic score (best + exploration bonus)
        - Numeric: interpolate from nearby values or return optimistic score

        Parameters:
            param_name (str): Name of the parameter
            param_value: Value of the parameter
            numeric_info (Dict): Optional dict with 'type' and 'bounds' for numeric interpolation

        Returns:
            float: UCB score for this parameter value
        """
        key = self._make_hashable(param_value)

        # If we've seen this value before, use standard UCB formula
        if key in self.param_counts[param_name]:
            count = self.param_counts[param_name][key]
            mean = self.param_means[param_name][key]
            exploration_bonus = self.kappa * np.sqrt(np.log(self.total_evaluations + 1) / (count + 1e-8))
            return mean + exploration_bonus

        # For unseen values, use optimistic initialization
        # For numeric parameters, try to interpolate from nearby values
        if numeric_info is not None and numeric_info['type'] in ['int', 'float']:
            # Try to find nearby values for interpolation
            nearby_scores = []
            for seen_key, mean in self.param_means[param_name].items():
                try:
                    seen_value = float(seen_key)
                    distance = abs(float(param_value) - seen_value)
                    # Weight by inverse distance
                    if distance < 1e-8:  # essentially the same value
                        return mean + self.kappa * np.sqrt(np.log(self.total_evaluations + 1))
                    nearby_scores.append((mean, 1.0 / distance))
                except (ValueError, TypeError):
                    continue

            # If we have nearby values, use weighted average
            if nearby_scores:
                total_weight = sum(w for _, w in nearby_scores)
                interpolated_mean = sum(m * w for m, w in nearby_scores) / total_weight
                # Add large exploration bonus for unseen values
                return interpolated_mean + 2.0 * self.kappa * np.sqrt(np.log(self.total_evaluations + 1))

        # Default: optimistic initialization (best performance + large exploration bonus)
        if self.param_means[param_name]:  # If we have any data for this parameter
            max_mean = max(self.param_means[param_name].values())
            return max_mean + 2.0 * self.kappa * np.sqrt(np.log(self.total_evaluations + 1))
        else:
            # No data at all for this parameter, return very optimistic score
            return self.best_performance + 3.0 * self.kappa * np.sqrt(np.log(self.total_evaluations + 1))

    def score_candidates(self, param_space: ModelParams, candidates: List[Dict]) -> np.ndarray:
        """
        Compute UCB scores for a list of candidate parameter configurations.
        The score for a configuration is the product of individual parameter UCB scores.

        Parameters:
            param_space (ModelParams): The parameter space definition
            candidates (List[Dict]): List of candidate parameter dictionaries

        Returns:
            np.ndarray: Array of UCB scores, one per candidate
        """
        scores = []

        for params in candidates:
            # Compute UCB score as product of per-parameter scores
            # Using product encourages configurations where all parameters have high UCB
            config_score = 1.0

            for param_name, param_value in params.items():
                # Get parameter info for numeric interpolation
                param_info = param_space.param_space.get(param_name)

                param_score = self._compute_ucb_score(param_name, param_value, param_info)  # type: ignore
                # Shift scores to be positive before multiplication
                # Using log-sum instead of product to avoid numerical underflow
                config_score += param_score

            scores.append(config_score)

        return np.asarray(scores)

    def suggest_one(self, param_space: ModelParams, candidates: List[Dict], rng: np.random.Generator) -> int:
        """
        Suggest the best candidate based on UCB scores.

        Parameters:
            param_space (ModelParams): The parameter space definition
            candidates (List[Dict]): Candidate parameter dictionaries to rank
            rng (np.random.Generator): Random number generator for tie-breaking

        Returns:
            int: Index of the best candidate in the original candidates list
        """
        scores = self.score_candidates(param_space, candidates)

        # Find max score
        best_score = np.max(scores)

        # Collect all indices with the best score (for tie-breaking)
        best_indices = [i for i, score in enumerate(scores) if np.abs(score - best_score) < 1e-10]

        # Randomly select one of the best indices
        return int(rng.choice(best_indices))

    def suggest_top_k(self, param_space: ModelParams, candidates: List[Dict], k: int, rng: np.random.Generator) -> List[int]:
        """
        Suggest the top k candidates based on UCB scores.
        Handles ties by randomly sampling among candidates with equal scores.

        Parameters:
            param_space (ModelParams): The parameter space definition
            candidates (List[Dict]): Candidate parameter dictionaries to rank
            k (int): Number of top candidates to return
            rng (np.random.Generator): Random number generator for tie-breaking

        Returns:
            List[int]: Indices of the top k candidates in the original candidates list
        """
        if k > len(candidates):
            k = len(candidates)

        scores = self.score_candidates(param_space, candidates)

        # Create list of (index, score) tuples
        indexed_scores = list(enumerate(scores))

        # Sort by score (descending)
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # If requesting all candidates, return them all
        if k == len(candidates):
            return [idx for idx, _ in indexed_scores]

        # Find the k-th highest score (or tie boundary)
        kth_score = indexed_scores[k-1][1]

        # Collect all candidates with scores >= k-th score
        candidates_above_threshold = [idx for idx, score in indexed_scores if score > kth_score]
        candidates_at_threshold = [idx for idx, score in indexed_scores if np.abs(score - kth_score) < 1e-10]

        # If we have exactly k candidates above threshold, return them
        if len(candidates_above_threshold) == k:
            return candidates_above_threshold

        # If we have fewer than k above threshold, we need to sample from the tie
        remaining_slots = k - len(candidates_above_threshold)

        # Randomly sample from candidates at the threshold score
        sampled_at_threshold = rng.choice(
            candidates_at_threshold,
            size=remaining_slots,
            replace=False
        ).tolist()

        return candidates_above_threshold + sampled_at_threshold
