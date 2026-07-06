#####################################################################################################
#
# NSGA-II tool box for the selection and evolutionary process.
# Note that we assume objectives are to be maximized, must convert to negative if needed.
#
#####################################################################################################

import numpy as np
from typeguard import typechecked
from typing import List, Tuple
import numpy.typing as npt

# Type alias for numpy random generator
rng_t = np.random.Generator

@typechecked
def non_dominated_sorting(obj_scores: npt.NDArray) -> Tuple[List[npt.NDArray[int]],npt.NDArray[int]]:
    """
    Perform non-dominated sorting for a maximization problem using NumPy arrays of type float32.

    Parameters:
    obj_scores (np.ndarray): A 2D array where each row represents the objective values for a solution.
                             Supports 2D and 3D tuples (all non-negative floats).

    Returns:
    Tuple(fronts, rank):
    fronts (list of numpy array): Each sublist contains the indices of solutions in the corresponding Pareto front.
    rank (numpy array of int): The front rank of each solution in the population.
    """

    # quick check to make sure that elements in scores are tuples
    assert all(isinstance(x, tuple) for x in obj_scores)
    # make sure dimensionality is either 2 or 3
    assert all(len(x) in [2, 3] for x in obj_scores)
    # make sure all tuples have the same dimensionality
    assert len(set(len(x) for x in obj_scores)) == 1
    # make sure all elements are floats
    assert all(all(isinstance(val, float) for val in x) for x in obj_scores)

    pop_size = obj_scores.shape[0]
    # final fronts returned
    fronts = [[]]
    # what front is solutions 'p' in
    rank = np.zeros(pop_size, dtype=int)
    # what 'q' solutions dominate 'p' solution
    domination_count = np.zeros(pop_size, dtype=int)
    # what 'q' solutions are dominated by 'p' solution
    dominated_solutions = [[] for _ in range(pop_size)]

    for p in range(pop_size):
        for q in range(pop_size):
            if dominates(obj_scores[p], obj_scores[q]):
                dominated_solutions[p].append(q)
            elif dominates(obj_scores[q], obj_scores[p]):
                domination_count[p] += 1

        if domination_count[p] == 0:
            rank[p] = int(0)
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                assert domination_count[q] >= 0 #check that it's always positive
                if domination_count[q] == 0:
                    rank[q] = int(i + 1)
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()

    fronts = [np.array(front, dtype=int) for front in fronts]
    return fronts, rank

@typechecked
def crowding_distance(obj_scores: npt.NDArray, front_map, count = None) -> npt.NDArray[np.floating]:
    """
    Calculate the crowding distance for each individual in the population.

    Parameters:
    - obj_scores: List of performances on obj_scores for each individual. We are assuming that the
                position of scores are the same as the position of the individuals in the population.
                Supports 2D and 3D tuples (all non-negative floats).
    - count: Number of obj_scores. If None, inferred from the first tuple in obj_scores.

    Returns:
    - crowding_distances: List of crowding distances corresponding to each individual.
    """

    # quick check to make sure that elements in scores are tuples
    assert all(isinstance(x, tuple) for x in obj_scores)
    # make sure dimensionality is either 2 or 3
    assert all(len(x) in [2, 3] for x in obj_scores)
    # make sure all tuples have the same dimensionality
    assert len(set(len(x) for x in obj_scores)) == 1
    # make sure all elements are floats
    assert all(all(isinstance(val, float) for val in x) for x in obj_scores)
    # make sure that count matches the dimensionality of the obj_scores if provided
    if count is not None:
        assert count == len(obj_scores[0])

    # infer count from the dimensionality of the tuples if not provided
    if count is None:
        count = len(obj_scores[0])

    # initialize the crowding distances to negative for guards
    crowding_distances = np.full(len(obj_scores), float(-1.0), dtype=float)

    for front in front_map:
        # set inital front crowding distances to zero for addition
        crowding_distances[front] = float(0.0)

        for m in range(count):
            # Sort the front scores based on the m-th objective
            sorted_indices = np.argsort([ind[m] for ind in obj_scores[front]], kind='mergesort')
            sorted_front = obj_scores[front[sorted_indices]]

            # calculate the range of the m-th objective
            min_obj = sorted_front[0][m]
            max_obj = sorted_front[-1][m]

            # skip if both max and min are the same
            if max_obj == min_obj:
                continue

            # set the crowding distance of boundary points to infinity
            crowding_distances[front[sorted_indices[0]]] = np.inf
            crowding_distances[front[sorted_indices[-1]]] = np.inf

            # calculate crowding distances for intermediate points
            for i in range(1, len(front) - 1):
                next_obj = sorted_front[i + 1][m]
                prev_obj = sorted_front[i - 1][m]
                crowding_distances[front[sorted_indices[i]]] += float(next_obj - prev_obj) / float(max_obj - min_obj)

    # make sure all crowding distances are non-negative
    assert np.all(crowding_distances >= 0.0)

    return crowding_distances

@typechecked
def dominates(solution1: Tuple[float, ...], solution2: Tuple[float, ...]) -> bool:
    """
    Check if solution1 dominates solution2.

    Parameters:
    solution1 (Tuple[float, ...]): The first solution's objective values (2D or 3D).
    solution2 (Tuple[float, ...]): The second solution's objective values (2D or 3D).

    Returns:
    bool: True if solution1 dominates solution2, False otherwise.
    """

    # check that solutions scores are of the same dimension
    assert len(solution1) == len(solution2)
    # make sure dimensionality is either 2 or 3
    assert len(solution1) in [2, 3]
    # make sure all elements are floats
    assert all(isinstance(val, float) for val in solution1)
    assert all(isinstance(val, float) for val in solution2)

    # solution1 dominates solution2 if it's >= in all objectives and > in at least one
    greater_or_equal = all(s1 >= s2 for s1, s2 in zip(solution1, solution2))
    better_in_at_least_one = any(s1 > s2 for s1, s2 in zip(solution1, solution2))

    return bool(greater_or_equal and better_in_at_least_one)

@typechecked
def non_dominated_binary_tournament(ranks: npt.NDArray[int], distances: npt.NDArray[float], rng: rng_t) -> int:
    """
    Perform a binary tournament selection based on non-dominated sorting and crowding distance.
    First, two individuals are randomly selected from the population.
    Winners are determined based on their ranks (fronts) and crowding distances.
    Lower rank individuals are preferred, followed by higher crowding distances in case of ties.

    Args:
        ranks (npt.NDArray[int]): The front rank of each solution in the population.
        distances (npt.NDArray[float]): The crowding distance of each solution in the population.
        rng (rng_t): Random number generator.

    Returns:
        int: The index of the winning individual.
    """

    # make sure that ranks and distances are the same size
    assert ranks.shape == distances.shape

    # get two random number between 0 and the population size
    t1,t2 = rng.choice(len(ranks), size=2, replace=False)
    t1, t2 = int(t1), int(t2)

    assert t1 != t2
    assert 0 <= t1 < len(ranks)
    assert 0 <= t2 < len(ranks)

    # check if the two solutions are in the same front
    if ranks[t1] == ranks[t2]:
        # the one with the greatest crowding distance wins
        return t1 if distances[t1] > distances[t2] else t2

    # if they are in different fronts, the lower rank one wins
    else:
        return t1 if ranks[t1] < ranks[t2] else t2

@typechecked
def non_dominated_truncate(fronts: List[npt.NDArray[int]], distances: npt.NDArray[float], N: int) -> npt.NDArray[int]:
    """
    Truncate the population to the N best individuals based on non-dominated sorting and crowding distance.
    First, individuals are added front by front until adding another front would exceed N.
    If the last front cannot be fully added, individuals from that front are selected based on their
    crowding distances in descending order.

    Args:
        fronts (List[npt.NDArray[int]]): List of Pareto fronts, each containing indices of individuals.
        distances (npt.NDArray[float]): The crowding distance of each solution in the population.
        N (int): The desired population size after truncation.

    Returns:
        npt.NDArray[int]: Indices of the selected individuals after truncation.
    """

    # make sure that fronts and distances are the same size
    assert sum([len(x) for x in fronts]) == len(distances)
    # make sure each front in the list are within the correct range
    assert all(all(0 <= ind < len(distances) for ind in front) for front in fronts)
    # make sure that distances is non-empty
    assert len(distances) > 0
    # make sure that distances are non-negative
    assert np.all(distances >= 0.0)
    # check that first object in fronts is a numpy array
    assert isinstance(fronts[0], np.ndarray)

    # go through each front and add the solutions to the survivors
    survivors = []
    for front in fronts:
        # add solutions without ordering based on distance (as is)
        if len(survivors) + len(front) <= N:
            survivors.extend(front)
        else:
            # sort the front by crowding distance in decending order
            sorted_distance = np.flip(np.argsort(distances[front], kind='mergesort'))
            sorted_front = front[sorted_distance]
            survivors.extend(sorted_front[:N-len(survivors)])
            break

    # make sure all survivor ids are within the correct range
    assert all(0 <= s < len(distances) for s in survivors)
    return np.array(survivors, dtype=int)