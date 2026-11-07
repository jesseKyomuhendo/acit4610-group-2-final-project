import numpy as np
import pytest

from src.continuous_ba import make_neighbour, solve_continuous
from src.functions import BENCHMARKS


def test_neighbour_step_within_ten_percent_of_range():
    lower, upper = -5.0, 10.0
    neighbour = make_neighbour(lower, upper)
    rng = np.random.default_rng(0)
    x = np.full(1000, 2.5)
    y = neighbour(x, rng)
    step = np.abs(y - x)
    assert step.max() <= 0.1 * (upper - lower)
    assert step.max() > 0.9 * 0.1 * (upper - lower)
    assert len(np.unique(y)) > 900


def test_neighbour_clips_to_bounds():
    neighbour = make_neighbour(-1.0, 1.0)
    rng = np.random.default_rng(1)
    y = neighbour(np.array([1.0, -1.0] * 500), rng)
    assert y.min() >= -1.0 and y.max() <= 1.0
    assert np.any(y == 1.0) and np.any(y == -1.0)


def test_neighbour_does_not_modify_input():
    x = np.zeros(5)
    make_neighbour(-1.0, 1.0)(x, np.random.default_rng(2))
    assert np.all(x == 0)


@pytest.mark.parametrize("function", BENCHMARKS)
@pytest.mark.parametrize("dim", [10, 30])
def test_full_run(function, dim):
    b = BENCHMARKS[function]
    r = solve_continuous(function, dim, seed=0)
    assert r.evaluations == 30_000
    assert r.best_x.shape == (dim,)
    assert np.all(r.best_x >= b.lower) and np.all(r.best_x <= b.upper)
    assert r.best == pytest.approx(b.f(r.best_x))
    assert 0 <= r.best < r.initial_best


def test_same_seed_same_result():
    a = solve_continuous("rastrigin", 10, seed=4, budget=1_000)
    b = solve_continuous("rastrigin", 10, seed=4, budget=1_000)
    assert a.best == b.best and np.array_equal(a.best_x, b.best_x)
