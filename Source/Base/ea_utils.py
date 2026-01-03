from .individual import Individual
from .ray_utils import (train_random_forest, train_linear_svc, train_decision_tree,
                        train_kernel_svc, train_extra_trees, train_gradient_boost,
                        train_linear_sgd)
from typeguard import typechecked
from typing import List, Dict
import numpy as np
import ray

evaluation_map = {
    'RF': train_random_forest,
    'LSVC': train_linear_svc,
    'DT': train_decision_tree,
    'KSVC': train_kernel_svc,
    'ET': train_extra_trees,
    'GB': train_gradient_boost,
    'LSGD': train_linear_sgd
}


@typechecked
def parent_selection(population: List[Individual],
                     num_parents: int,
                     rng: np.random.Generator,
                     tournament_size: int = 2) -> List[int]:
    """
    Select parents from the population based on their performance.

    Parameters:
        population (List[Individual]): The population of individuals.
        num_parents (int): The number of parents to select.
        rng (np.random.Generator): Random number generator for reproducibility.
        tournament_size (int): The size of the tournament for selection.

    Returns:
        List[Individual]: The selected parent individuals.
    """
    return [tournament_selection(population, tournament_size, rng) for _ in range(num_parents)]

@typechecked
def tournament_selection(population: List[Individual], size: int, rng: np.random.Generator) -> int:
        """
        Selects a single parent using tournament selection.

        Parameters:
            population (List[Individual]): A population of individuals to select from.
            size (int): The tournament size.
            rng (np.random.Generator): Random number generator for reproducibility.
        Returns:
            Individual: a single parent individual.

        """
        assert len(population) >= 0

        # Randomly choose a size number of population indices (determined by config)
        indices = rng.choice(len(population), size, replace=False)
        # Extract performances at the chosen indices
        extracted_performances = np.array([population[i].get_val_performance() for i in indices])
        # Get the position of the best (highest) performance in the tournament (not population-based index)
        best_tour_idx = np.argmax(extracted_performances)
        # Randomly select one of the tied best individuals (population-based index)
        return int(rng.choice([i for i, perf in zip(indices, extracted_performances) if perf == extracted_performances[best_tour_idx]]))