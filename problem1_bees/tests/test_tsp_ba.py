import numpy as np
import pytest

from src.bees import BAParams, bees_algorithm
from src.tour import distance_matrix, tour_length, validate_tour
from src.tsp_ba import MOVES, random_tour, solve_tsp
from src.tsplib import load_instance


@pytest.fixture(scope="module")
def berlin52():
    return load_instance("berlin52")


@pytest.mark.parametrize("move", MOVES)
def test_moves_keep_city1_fixed_and_valid(move):
    rng = np.random.default_rng(0)
    tour = random_tour(20, rng)
    for _ in range(500):
        new = MOVES[move](tour, rng)
        validate_tour(new, 20)
        assert not np.array_equal(new, tour)
        tour = new


def test_moves_do_not_modify_input():
    rng = np.random.default_rng(1)
    tour = random_tour(10, rng)
    before = tour.copy()
    for move in MOVES.values():
        move(tour, rng)
    assert np.array_equal(tour, before)


def test_reversal_on_known_positions():
    class FixedRng:
        def choice(self, a, size, replace):
            return np.array([4, 2])

    tour = np.arange(7)
    assert MOVES["reversal"](tour, FixedRng()).tolist() == [0, 1, 4, 3, 2, 5, 6]
    assert MOVES["swap"](tour, FixedRng()).tolist() == [0, 1, 4, 3, 2, 5, 6]


@pytest.mark.parametrize("move", MOVES)
def test_full_run_uses_exact_budget(berlin52, move):
    r = solve_tsp(berlin52, move, seed=0)
    assert r.evaluations == 30_000
    assert r.history_evals[-1] == 30_000
    validate_tour(r.best_x, 52)
    assert r.best == tour_length(r.best_x, distance_matrix(berlin52.coords))
    assert r.best < r.initial_best
    assert r.best >= berlin52.optimum
    assert all(a >= b for a, b in zip(r.history_best[1:], r.history_best[2:]))


def test_stops_mid_iteration():
    # 50 initial + 170 per iteration: budget 300 ends partway through iteration 2.
    calls = []

    def objective(x):
        calls.append(1)
        return float(x)

    r = bees_algorithm(
        objective,
        random_solution=lambda rng: rng.random(),
        neighbour=lambda x, rng: x + rng.normal() * 0.1,
        params=BAParams(),
        rng=np.random.default_rng(0),
        budget=300,
    )
    assert len(calls) == 300 == r.evaluations
    assert r.iterations == 2


def test_same_seed_same_result(berlin52):
    a = solve_tsp(berlin52, "swap", seed=3, budget=2_000)
    b = solve_tsp(berlin52, "swap", seed=3, budget=2_000)
    assert a.best == b.best and np.array_equal(a.best_x, b.best_x)


def test_same_initial_population_for_both_moves(berlin52):
    a = solve_tsp(berlin52, "swap", seed=5, budget=50)
    b = solve_tsp(berlin52, "reversal", seed=5, budget=50)
    assert a.initial_best == b.initial_best
