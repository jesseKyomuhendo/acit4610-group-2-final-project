import numpy as np
import pytest

from src.data import EXPECTED, all_cases, load_portfolio
from src.es import population_es, run
from src.repair import is_feasible, project_simplex, repair


@pytest.fixture(scope="module")
def port1():
    return load_portfolio("port1")


def test_port1_checkpoint(port1):
    assert port1.r_eq == pytest.approx(0.003504064516129, abs=1e-12)
    assert port1.v_eq == pytest.approx(0.001130937943724, abs=1e-12)
    assert port1.r_min(0.5) == pytest.approx(0.007184532258065, abs=1e-12)


@pytest.mark.parametrize("name", list(EXPECTED))
def test_all_datasets_load(name):
    p = load_portfolio(name)
    assert p.n == EXPECTED[name][0]
    assert np.allclose(p.C, p.C.T) and np.allclose(np.diag(p.C), 1)
    assert np.allclose(p.cov, p.cov.T)


def test_fifteen_cases():
    cases = all_cases()
    assert len(cases) == 15 and cases[0] == "port1_alpha025" and cases[-1] == "port5_alpha075"


def test_projection_basic():
    rng = np.random.default_rng(1)
    for _ in range(200):
        z = rng.normal(0, 3, rng.integers(1, 50))
        q = project_simplex(z)
        assert (q >= 0).all() and abs(q.sum() - 1) < 1e-12
        # already on the simplex -> unchanged
        assert np.allclose(project_simplex(q), q)
    assert np.allclose(project_simplex([0.2, 0.3, 0.5]), [0.2, 0.3, 0.5])
    assert np.allclose(project_simplex([2.0, 0.0]), [1.0, 0.0])
    assert np.allclose(project_simplex([1.0, 1.0]), [0.5, 0.5])


def test_projection_is_closest_point():
    """KKT check: q = max(z - theta, 0) for a single theta."""
    rng = np.random.default_rng(2)
    z = rng.normal(size=20)
    q = project_simplex(z)
    pos = q > 0
    theta = (z[pos] - q[pos]).mean()
    assert np.allclose(z[pos] - q[pos], theta)
    assert (z[~pos] <= theta + 1e-12).all()


def test_repair_feasible(port1):
    rng = np.random.default_rng(3)
    for alpha in (0.25, 0.5, 0.75, 1.0):
        rmin = port1.r_min(alpha)
        for _ in range(200):
            w = repair(rng.normal(size=port1.n), port1.r, rmin, port1.j_max)
            assert is_feasible(w, port1.r, rmin)


def test_spec_numerical_example():
    r = np.array([0.02, 0.06, 0.10]); s = np.array([0.1, 0.2, 0.3])
    cov = np.diag(s ** 2)
    A = np.array([0.5, 0.3, 0.2]); B = np.array([0.6, 0.3, 0.1])
    assert r @ A == pytest.approx(0.048) and A @ cov @ A == pytest.approx(0.0097)
    assert r @ B == pytest.approx(0.040) and B @ cov @ B == pytest.approx(0.0081)


@pytest.mark.parametrize("variant", ["1+1", "plus", "comma"])
def test_budget_and_feasibility(port1, variant):
    rmin = port1.r_min(0.5)
    res = run(variant, port1, rmin, seed=0, budget=3000)
    assert res.evals == 3000
    assert is_feasible(res.best_w, port1.r, rmin)
    assert res.best_var == pytest.approx(res.best_w @ port1.cov @ res.best_w)
    assert res.history_evals[-1] == 3000
    assert np.all(np.diff(res.history_best) <= 0)        # archive never worsens


def test_comma_pop_best_can_worsen_archive_cannot(port1):
    res = population_es(port1, port1.r_min(0.5), seed=0, plus=False, budget=30_000)
    assert np.any(np.diff(res.pop_best) > 0)              # comma population best worsens
    assert np.all(np.diff(res.history_best) <= 0)
    res_p = population_es(port1, port1.r_min(0.5), seed=0, plus=True, budget=3000)
    assert np.all(np.diff(res_p.pop_best) <= 0)           # plus selection is elitist


def test_reproducible(port1):
    a = run("comma", port1, port1.r_min(0.25), seed=5, budget=1500)
    b = run("comma", port1, port1.r_min(0.25), seed=5, budget=1500)
    assert a.best_var == b.best_var and np.array_equal(a.best_w, b.best_w)
