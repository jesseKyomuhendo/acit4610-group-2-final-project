"""Continuous benchmark functions for Part B (Surjanovic & Bingham, SFU).

All four are unshifted and noiseless, with global minimum 0. Each takes a 1-D
numpy array of length D and returns a float.

The formulas live here. The bounds, the minimisers (used only in tests) and the
dimensions come from config_ba.yaml (`continuous.functions`,
`continuous.dimensions`).
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from prob_1_util.ba_config_loader import DIMENSIONS, FUNCTION_BOUNDS


def sphere(x):
    return float(np.sum(x**2))


def rosenbrock(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rastrigin(x):
    return float(10.0 * x.size + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x, a=20.0, b=0.2, c=2.0 * np.pi):
    # a, b, c are part of the SFU formula, not tunable settings.
    d = x.size
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    term2 = -np.exp(np.sum(np.cos(c * x)) / d)
    return float(term1 + term2 + a + np.e)


# YAML function name -> Python formula.
FORMULAS = {
    "sphere": sphere,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
    "ackley": ackley,
}


@dataclass(frozen=True)
class Benchmark:
    name: str
    f: Callable[[np.ndarray], float]
    lower: float
    upper: float
    minimiser: float  # every coordinate of the global minimiser has this value


# One Benchmark per function listed in config_ba.yaml, in the same order.
BENCHMARKS = {
    key: Benchmark(key.capitalize(), FORMULAS[key], lower, upper, minimiser)
    for key, (lower, upper, minimiser) in FUNCTION_BOUNDS.items()
}