# AI-assisted: Claude Code, Evaluator scaffold, reviewed by <member>
"""Evaluation budget shared by every algorithm in this problem.

Every objective call goes through Evaluator, so initial population, recruits,
scouts / offspring and rejected candidates are all counted. Best-so-far is
tracked here, which also gives the archive required by the spec.
"""
import numpy as np


class BudgetExhausted(Exception):
    """Raised when an evaluation is requested after the budget is spent."""


class Evaluator:
    def __init__(self, f, budget=30_000, log_every=10):
        self.f = f
        self.budget = budget
        self.log_every = log_every
        self.n = 0
        self.best = np.inf
        self.best_x = None
        self.initial_best = None
        self.history_evals = [0]
        self.history_best = [np.inf]

    def remaining(self):
        return self.budget - self.n

    def mark_initial(self):
        """Call once the initial population has been evaluated."""
        self.initial_best = self.best

    def __call__(self, x):
        if self.n >= self.budget:
            raise BudgetExhausted
        v = self.f(x)
        self.n += 1
        if v < self.best:
            self.best = v
            self.best_x = np.array(x, copy=True)
        if self.n % self.log_every == 0 or self.n == self.budget:
            self.history_evals.append(self.n)
            self.history_best.append(self.best)
        return v
