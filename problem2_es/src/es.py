"""Evolution Strategies for long-only minimum-variance portfolios.

Genotype = repaired weight vector w (plus step size(s)). Every candidate z is
repaired before its variance is evaluated, so the archive (best-so-far kept by
Evaluator) only ever holds feasible allocations.

Budget accounting (30,000 evaluations):
  (1+1)-ES        : 1 initial parent + 29,999 children
  (mu+/,lambda)-ES: 10 initial parents + 428 full generations x 70 + final batch of 30
"""
import time
from dataclasses import dataclass, field

import numpy as np

from .budget import Evaluator
from .repair import repair, variance

SIGMA_MIN, SIGMA_MAX = 1e-8, 1.0
VARIANTS = ("1+1", "plus", "comma")
LABELS = {"1+1": "(1+1)-ES", "plus": "(μ+λ)-ES", "comma": "(μ,λ)-ES"}


@dataclass
class RunResult:
    variant: str
    seed: int
    best_w: np.ndarray
    best_var: float
    initial_best: float
    evals: int
    runtime: float
    history_evals: np.ndarray
    history_best: np.ndarray
    # population best per generation (for (mu,lambda) worsening analysis)
    pop_evals: np.ndarray = field(default_factory=lambda: np.array([]))
    pop_best: np.ndarray = field(default_factory=lambda: np.array([]))
    final_sigma_mean: float = np.nan


def _make_problem(port, r_min):
    r, cov, j = port.r, port.cov, port.j_max
    rep = lambda z: repair(z, r, r_min, j)
    obj = lambda w: variance(w, cov)
    return rep, obj


def one_plus_one(port, r_min, seed, budget=30_000):
    """(1+1)-ES with scalar step size and the 1/5th success rule."""
    rng = np.random.default_rng(seed)
    rep, obj = _make_problem(port, r_min)
    n = port.n
    t0 = time.perf_counter()
    ev = Evaluator(obj, budget=budget)

    parent = rep(rng.uniform(0.0, 1.0, n))
    f_parent = ev(parent)
    ev.mark_initial()
    sigma = 0.1 / np.sqrt(n)
    up, down = np.exp(1.0 / 3.0), np.exp(-1.0 / 12.0)

    while ev.remaining() > 0:
        child = rep(parent + sigma * rng.standard_normal(n))
        f_child = ev(child)
        if f_child < f_parent:          # strictly lower -> success
            parent, f_parent = child, f_child
            sigma *= up
        else:
            sigma *= down
        sigma = min(max(sigma, SIGMA_MIN), SIGMA_MAX)

    rt = time.perf_counter() - t0
    return RunResult("1+1", seed, ev.best_x, ev.best, ev.initial_best, ev.n, rt,
                     np.array(ev.history_evals), np.array(ev.history_best),
                     final_sigma_mean=sigma)


def population_es(port, r_min, seed, plus, mu=10, lam=70, budget=30_000):
    """(mu+lambda)-ES if plus else (mu,lambda)-ES, with self-adaptive
    per-coordinate step sizes, intermediate weight recombination and
    geometric step-size recombination."""
    rng = np.random.default_rng(seed)
    rep, obj = _make_problem(port, r_min)
    n = port.n
    tau_p = 1.0 / np.sqrt(2.0 * n)            # global learning rate tau'
    tau = 1.0 / np.sqrt(2.0 * np.sqrt(n))     # coordinate-wise learning rate tau
    t0 = time.perf_counter()
    ev = Evaluator(obj, budget=budget)

    # --- initialisation: mu repaired uniform vectors, sigma = 0.1/sqrt(n) ---
    P_w = np.array([rep(rng.uniform(0.0, 1.0, n)) for _ in range(mu)])
    P_s = np.full((mu, n), 0.1 / np.sqrt(n))
    P_f = np.array([ev(w) for w in P_w])
    ev.mark_initial()
    pop_evals, pop_best = [ev.n], [P_f.min()]

    while ev.remaining() > 0:
        k = min(lam, ev.remaining())          # final batch may be < lambda
        O_w = np.empty((k, n))
        O_s = np.empty((k, n))
        O_f = np.empty(k)
        for c in range(k):
            # two distinct parents, uniformly at random (cheaper than rng.choice)
            a = int(rng.integers(mu))
            b = int(rng.integers(mu - 1))
            b += b >= a
            w_r = 0.5 * (P_w[a] + P_w[b])
            s_r = np.sqrt(P_s[a] * P_s[b])
            z0 = rng.standard_normal()                     # shared across coordinates
            s_new = s_r * np.exp(tau_p * z0 + tau * rng.standard_normal(n))
            s_new = np.clip(s_new, SIGMA_MIN, SIGMA_MAX)
            w_new = rep(w_r + s_new * rng.standard_normal(n))
            O_w[c], O_s[c], O_f[c] = w_new, s_new, ev(w_new)

        if plus:
            all_w = np.vstack([P_w, O_w]); all_s = np.vstack([P_s, O_s])
            all_f = np.concatenate([P_f, O_f])
        else:
            all_w, all_s, all_f = O_w, O_s, O_f
        if len(all_f) >= mu:                  # always true here (final batch = 30)
            idx = np.argsort(all_f, kind="stable")[:mu]
            P_w, P_s, P_f = all_w[idx], all_s[idx], all_f[idx]
        pop_evals.append(ev.n)
        pop_best.append(P_f.min())

    rt = time.perf_counter() - t0
    return RunResult("plus" if plus else "comma", seed, ev.best_x, ev.best, ev.initial_best,
                     ev.n, rt, np.array(ev.history_evals), np.array(ev.history_best),
                     np.array(pop_evals), np.array(pop_best), float(P_s.mean()))


def run(variant, port, r_min, seed, budget=30_000):
    if variant == "1+1":
        return one_plus_one(port, r_min, seed, budget)
    if variant == "plus":
        return population_es(port, r_min, seed, plus=True, budget=budget)
    if variant == "comma":
        return population_es(port, r_min, seed, plus=False, budget=budget)
    raise ValueError(f"unknown variant {variant!r}")
