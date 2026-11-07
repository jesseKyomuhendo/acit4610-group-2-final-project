# AI-assisted: Claude Code, continuous benchmark functions, reviewed by <member>
"""Continuous benchmark functions for Part B (Surjanovic & Bingham, SFU).

All four are unshifted and noiseless, with global minimum 0. Each takes a 1-D
numpy array of length D and returns a float.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np


def sphere(x):
    return float(np.sum(x**2))


def rosenbrock(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rastrigin(x):
    return float(10.0 * x.size + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x, a=20.0, b=0.2, c=2.0 * np.pi):
    d = x.size
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    term2 = -np.exp(np.sum(np.cos(c * x)) / d)
    return float(term1 + term2 + a + np.e)


@dataclass(frozen=True)
class Benchmark:
    name: str
    f: Callable[[np.ndarray], float]
    lower: float
    upper: float
    minimiser: float  # every coordinate of the global minimiser has this value


BENCHMARKS = {
    "sphere": Benchmark("Sphere", sphere, -5.12, 5.12, 0.0),
    "rosenbrock": Benchmark("Rosenbrock", rosenbrock, -5.0, 10.0, 1.0),
    "rastrigin": Benchmark("Rastrigin", rastrigin, -5.12, 5.12, 0.0),
    "ackley": Benchmark("Ackley", ackley, -32.768, 32.768, 0.0),
}

DIMENSIONS = (10, 30)
