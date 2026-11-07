"""Bees Algorithm for the symmetric TSP with swap or segment-reversal moves.

Tours are permutations of 0-based city indices with city 1 (index 0) fixed in
position 0. Both moves only touch positions 1..n-1, so city 1 never moves and
every neighbour is still a valid permutation.
"""
import numpy as np

from .bees import BAParams, bees_algorithm
from .tour import distance_matrix, tour_length, validate_tour


def random_tour(n, rng):
    """Uniformly shuffled tour with city 1 first."""
    return np.concatenate(([0], rng.permutation(np.arange(1, n))))


def _two_positions(n, rng):
    i, j = rng.choice(np.arange(1, n), size=2, replace=False)
    return min(i, j), max(i, j)


def swap_move(tour, rng):
    """Swap the cities at two random positions."""
    i, j = _two_positions(len(tour), rng)
    new = tour.copy()
    new[i], new[j] = new[j], new[i]
    return new


def reversal_move(tour, rng):
    """Reverse the segment between two random positions (inclusive)."""
    i, j = _two_positions(len(tour), rng)
    new = tour.copy()
    new[i : j + 1] = new[i : j + 1][::-1]
    return new


MOVES = {"swap": swap_move, "reversal": reversal_move}

# Chosen by pilot tuning (experiments/tuning, setting E); used for both moves.
TSP_PARAMS = BAParams(n=15, m=3, e=1, nep=50, nsp=25)


def solve_tsp(instance, move, seed, params=TSP_PARAMS, budget=30_000):
    """Run BA on a TSPInstance with the named move and return a BAResult."""
    dist = distance_matrix(instance.coords)
    n = instance.dimension
    result = bees_algorithm(
        objective=lambda t: tour_length(t, dist),
        random_solution=lambda rng: random_tour(n, rng),
        neighbour=MOVES[move],
        params=params,
        rng=np.random.default_rng(seed),
        budget=budget,
    )
    validate_tour(result.best_x, n)
    return result
