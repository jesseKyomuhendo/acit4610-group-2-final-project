"""Each function is checked at its known minimiser and at one other point.

Expected values at the other point follow directly from the SFU definitions:
  Sphere(1,...,1)     = D
  Rosenbrock(0,...,0) = D - 1
  Rastrigin(1,...,1)  = 10D + D(1 - 10) = D
  Ackley(1,...,1)     = 20 - 20 exp(-0.2)
plus one hand-computed non-uniform point per function at D = 2.
"""
import numpy as np
import pytest

from src.functions import BENCHMARKS, DIMENSIONS, ackley, rastrigin, rosenbrock, sphere

TOL = 1e-12


@pytest.mark.parametrize("key", BENCHMARKS)
@pytest.mark.parametrize("d", DIMENSIONS)
def test_zero_at_minimiser(key, d):
    b = BENCHMARKS[key]
    assert abs(b.f(np.full(d, b.minimiser))) < TOL


@pytest.mark.parametrize("d", DIMENSIONS)
def test_other_point_uniform(d):
    ones, zeros = np.ones(d), np.zeros(d)
    assert sphere(ones) == pytest.approx(d, abs=TOL)
    assert rosenbrock(zeros) == pytest.approx(d - 1, abs=TOL)
    assert rastrigin(ones) == pytest.approx(d, abs=1e-9)
    assert ackley(ones) == pytest.approx(20 - 20 * np.exp(-0.2), abs=TOL)


def test_other_point_non_uniform():
    x = np.array([1.0, 2.0])
    assert sphere(x) == pytest.approx(5.0)
    # 100 (2 - 1)^2 + (1 - 1)^2
    assert rosenbrock(x) == pytest.approx(100.0)
    # cos(2 pi k) = 1 for integers: 20 + (1 - 10) + (4 - 10)
    assert rastrigin(x) == pytest.approx(5.0)
    # -20 exp(-0.2 sqrt(5/2)) - exp(1) + 20 + e
    assert ackley(x) == pytest.approx(20 - 20 * np.exp(-0.2 * np.sqrt(2.5)))


def test_bounds_match_spec():
    spec = {
        "sphere": (-5.12, 5.12),
        "rosenbrock": (-5.0, 10.0),
        "rastrigin": (-5.12, 5.12),
        "ackley": (-32.768, 32.768),
    }
    for key, (lo, hi) in spec.items():
        assert (BENCHMARKS[key].lower, BENCHMARKS[key].upper) == (lo, hi)
