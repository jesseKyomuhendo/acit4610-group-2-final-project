"""Deterministic reference optimum via SciPy SLSQP.

Used ONLY to judge ES solution quality in the tables/discussion (relative gap);
it never initialises or guides any ES run.
"""
import numpy as np
from scipy.optimize import minimize

from .repair import is_feasible, repair


def reference_min_variance(port, r_min):
    n, r, cov = port.n, port.r, port.cov
    x0 = repair(np.full(n, 1.0 / n), r, r_min, port.j_max)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)},
            {"type": "ineq", "fun": lambda w: r @ w - r_min, "jac": lambda w: r}]
    res = minimize(lambda w: w @ cov @ w, x0, jac=lambda w: 2 * cov @ w,
                   bounds=[(0.0, 1.0)] * n, constraints=cons, method="SLSQP",
                   options={"ftol": 1e-15, "maxiter": 2000})
    w = np.clip(res.x, 0.0, None)
    w = w / w.sum()
    w = repair(w, r, r_min, port.j_max)       # guard tiny constraint violations
    assert is_feasible(w, r, r_min)
    return w, float(w @ cov @ w)
