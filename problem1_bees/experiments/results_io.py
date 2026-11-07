"""Loading saved runs and shared labels for the tables and plots."""
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
RAW = RESULTS / "raw"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

MOVE_LABELS = {"swap": "Swap", "reversal": "Segment reversal"}
FUNCTION_LABELS = {"sphere": "Sphere", "rosenbrock": "Rosenbrock", "rastrigin": "Rastrigin", "ackley": "Ackley"}


def load_runs(kind, **match):
    """All saved runs of kind ('tsp' or 'continuous') whose fields equal match."""
    runs = []
    for path in sorted((RAW / kind).glob("*.npz")):
        d = dict(np.load(path))
        if all(d[k].item() == v for k, v in match.items()):
            runs.append(d)
    if not runs:
        raise FileNotFoundError(f"no {kind} runs matching {match}; run the experiments first")
    return sorted(runs, key=lambda d: int(d["seed"]))


def histories(runs):
    """Evaluation axis and an (n_runs, n_points) array of best-so-far values.

    The first logged point (evaluation 0, value inf) is dropped.
    """
    evals = runs[0]["history_evals"][1:]
    for d in runs:
        if not np.array_equal(d["history_evals"][1:], evals):
            raise ValueError("runs were logged at different evaluation counts")
    return evals, np.stack([d["history_best"][1:] for d in runs])
