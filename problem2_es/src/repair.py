# AI-assisted: Claude Code, simplex projection / return repair / feasibility check, reviewed by <member>
"""Constraint handling shared by all three ES variants.

Repair is NOT an objective evaluation; it is applied to every candidate z
before the variance is computed, so every evaluated point is feasible.
"""
import numpy as np

TOL = 1e-8


def project_simplex(z):
    """Euclidean projection of z onto {q : q >= 0, sum q = 1}.

    Duchi, Shalev-Shwartz, Singer & Chandra (2008), Fig. 1:
    sort z descending, find the largest rho with u_rho - (cumsum_rho - 1)/rho > 0,
    theta = (cumsum_rho - 1)/rho, q_i = max(z_i - theta, 0).
    """
    z = np.asarray(z, dtype=float)
    u = np.sort(z)[::-1]
    css = np.cumsum(u) - 1.0
    k = np.arange(1, len(z) + 1)
    rho = np.nonzero(u - css / k > 0)[0][-1]
    theta = css[rho] / (rho + 1)
    return np.maximum(z - theta, 0.0)


def repair(z, r, r_min, j_max):
    """Step 1: simplex projection. Step 2: move towards e_j if R(q) < Rmin."""
    q = project_simplex(z)
    rq = float(r @ q)
    if rq >= r_min:
        return q
    r_max = r[j_max]
    beta = (r_min - rq) / (r_max - rq)
    w = (1.0 - beta) * q
    w[j_max] += beta
    return w


def is_feasible(w, r, r_min, tol=TOL):
    """Numerical feasibility check required for every reported solution."""
    w = np.asarray(w, dtype=float)
    return bool(
        (w >= -tol).all()
        and abs(w.sum() - 1.0) <= tol
        and float(r @ w) >= r_min - tol
    )


def variance(w, cov):
    return float(w @ cov @ w)
