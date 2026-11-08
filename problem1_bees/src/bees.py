"""Basic Bees Algorithm (Pham et al., 2006), independent of the representation.

The caller supplies how to create a random solution and how to create one
neighbour of a site, so the same loop serves the TSP and continuous problems.

Each iteration:
  1. Rank the n sites by objective value.
  2. The best m sites are selected; the first e of them are elite.
  3. Recruit nep bees around each elite site and nsp around each other
     selected site. Each recruit is one neighbourhood move from the site.
     A site moves to its best recruit only if that recruit is strictly better.
  4. The remaining n - m sites are replaced by fresh random scouts.
Every candidate (initial scouts, recruits, fresh scouts) costs one evaluation,
whether it is kept or not. The run stops as soon as the budget is spent, which
may be in the middle of an iteration.
"""
from dataclasses import dataclass

from .budget import BudgetExhausted, Evaluator


@dataclass(frozen=True)
class BAParams:
    n: int = 50  # scout bees / sites
    m: int = 10  # selected sites
    e: int = 3  # elite sites (among the m)
    nep: int = 20  # recruits per elite site
    nsp: int = 10  # recruits per other selected site

    def __post_init__(self):
        if not (0 < self.e <= self.m <= self.n):
            raise ValueError("need 0 < e <= m <= n")
        if self.nep < 1 or self.nsp < 1:
            raise ValueError("nep and nsp must be at least 1")


@dataclass
class BAResult:
    best_x: object
    best: float
    initial_best: float
    evaluations: int
    iterations: int
    history_evals: list
    history_best: list


def bees_algorithm(objective, random_solution, neighbour, params, rng, budget=30_000):
    """Minimise objective within the evaluation budget.

    random_solution(rng) -> new solution
    neighbour(x, rng)    -> new solution one move away from x (x is not modified)
    """
    ev = Evaluator(objective, budget=budget)
    sites, values = [], []
    iterations = 0
    try:
        for _ in range(params.n):
            x = random_solution(rng)
            sites.append(x)
            values.append(ev(x))
        ev.mark_initial()

        while ev.remaining() > 0:
            iterations += 1
            order = sorted(range(params.n), key=values.__getitem__)
            sites = [sites[i] for i in order]
            values = [values[i] for i in order]

            for rank in range(params.m):
                recruits = params.nep if rank < params.e else params.nsp
                best_x, best_v = None, values[rank]
                for _ in range(recruits):
                    y = neighbour(sites[rank], rng)
                    v = ev(y)
                    if v < best_v:
                        best_x, best_v = y, v
                if best_x is not None:
                    sites[rank], values[rank] = best_x, best_v

            for k in range(params.m, params.n):
                x = random_solution(rng)
                v = ev(x)
                sites[k], values[k] = x, v
    except BudgetExhausted:
        pass

    if ev.initial_best is None:
        ev.mark_initial()
    return BAResult(
        best_x=ev.best_x,
        best=ev.best,
        initial_best=ev.initial_best,
        evaluations=ev.n,
        iterations=iterations,
        history_evals=ev.history_evals,
        history_best=ev.history_best,
    )
