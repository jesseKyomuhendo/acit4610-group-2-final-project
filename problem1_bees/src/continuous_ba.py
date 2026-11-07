"""Bees Algorithm for bounded continuous minimisation.

Scouts are drawn uniformly within the bounds. A recruit changes every
coordinate independently by a uniform amount in [-10%, +10%] of the allowed
range (upper - lower) and is then clipped to the bounds.
"""
import numpy as np

from .bees import BAParams, bees_algorithm
from .functions import BENCHMARKS

PATCH_FRACTION = 0.10

# Chosen by pilot tuning (experiments/tuning, setting B).
CONTINUOUS_PARAMS = BAParams(n=30, m=8, e=3, nep=30, nsp=10)


def make_neighbour(lower, upper, fraction=PATCH_FRACTION):
    half_width = fraction * (upper - lower)

    def neighbour(x, rng):
        return np.clip(x + rng.uniform(-half_width, half_width, size=x.size), lower, upper)

    return neighbour


def solve_continuous(function, dim, seed, params=CONTINUOUS_PARAMS, budget=30_000):
    """Run BA on a named benchmark ('sphere', 'rosenbrock', ...) in dim dimensions."""
    b = BENCHMARKS[function]
    return bees_algorithm(
        objective=b.f,
        random_solution=lambda rng: rng.uniform(b.lower, b.upper, size=dim),
        neighbour=make_neighbour(b.lower, b.upper),
        params=params,
        rng=np.random.default_rng(seed),
        budget=budget,
    )
