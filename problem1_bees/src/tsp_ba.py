"""Bees Algorithm for the symmetric TSP with swap or segment-reversal moves.

Tours are permutations of 0-based city indices with city 1 (index 0) fixed in
position 0. Both moves only touch positions 1..n-1, so city 1 never moves and
every neighbour is still a valid permutation.

The move names, the BA settings and the budget come from config_ba.yaml
(`tsp.moves`, `tsp.ba`, `budget`). The moves themselves are implemented here.
"""
import numpy as np

from prob_1_util.ba_config_loader import BUDGET, MOVE_NAMES, TSP_BA

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


# YAML move name -> Python move function.
MOVE_FUNCTIONS = {"swap": swap_move, "reversal": reversal_move}

# The moves listed in config_ba.yaml, in the same order.
MOVES = {name: MOVE_FUNCTIONS[name] for name in MOVE_NAMES}

# Chosen by pilot tuning (experiments/tuning, setting E); used for both moves.
TSP_PARAMS = BAParams(**TSP_BA)


def solve_tsp(instance, move, seed, params=TSP_PARAMS, budget=BUDGET):
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