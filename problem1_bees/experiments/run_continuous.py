"""Run the Bees Algorithm on the continuous benchmark functions.

One run per (function, dimension, seed). Each run is saved to
results/raw/continuous/<function>_d<dim>_s<seed>.npz with the best-so-far
history and final vector; results/raw/continuous_runs.csv is rebuilt from those
files after every invocation. Existing runs are skipped.

Usage (from problem1_bees/):
    python -m experiments.run_continuous --function rastrigin --dim 10 --seed 0
    python -m experiments.run_continuous --all
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from src.continuous_ba import CONTINUOUS_PARAMS, solve_continuous
from src.functions import BENCHMARKS, DIMENSIONS

SEEDS = tuple(range(10))
RAW = Path(__file__).resolve().parents[1] / "results" / "raw"
RUN_DIR = RAW / "continuous"
CSV_PATH = RAW / "continuous_runs.csv"
COLUMNS = [
    "function", "dim", "seed", "initial_best", "best", "evaluations",
    "iterations", "runtime_s", "x",
]


def run_path(function, dim, seed):
    return RUN_DIR / f"{function}_d{dim}_s{seed}.npz"


def run_one(function, dim, seed, force=False):
    path = run_path(function, dim, seed)
    if path.exists() and not force:
        print(f"skip {path.name} (exists)")
        return
    start = time.perf_counter()
    r = solve_continuous(function, dim, seed)
    runtime = time.perf_counter() - start

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        function=function,
        dim=dim,
        seed=seed,
        initial_best=r.initial_best,
        best=r.best,
        evaluations=r.evaluations,
        iterations=r.iterations,
        runtime_s=runtime,
        x=r.best_x,
        history_evals=np.array(r.history_evals),
        history_best=np.array(r.history_best),
        params=str(CONTINUOUS_PARAMS),
    )
    print(f"{function:10s} D={dim:2d} seed {seed}: {r.best:.6g} in {runtime:.2f}s")


def write_csv():
    rows = []
    for path in sorted(RUN_DIR.glob("*.npz")):
        d = np.load(path)
        row = {c: d[c].item() for c in COLUMNS if c != "x"}
        row["runtime_s"] = round(row["runtime_s"], 4)
        row["x"] = " ".join(f"{v:.10g}" for v in d["x"])
        rows.append(row)
    order = {name: i for i, name in enumerate(BENCHMARKS)}
    rows.sort(key=lambda r: (order[r["function"]], r["dim"], r["seed"]))
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {CSV_PATH} ({len(rows)} runs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="run every function, dimension and seed")
    ap.add_argument("--function", choices=BENCHMARKS)
    ap.add_argument("--dim", type=int, choices=DIMENSIONS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--force", action="store_true", help="rerun even if the result exists")
    args = ap.parse_args()

    if args.all:
        for function in BENCHMARKS:
            for dim in DIMENSIONS:
                for seed in SEEDS:
                    run_one(function, dim, seed, args.force)
    elif args.function and args.dim and args.seed is not None:
        run_one(args.function, args.dim, args.seed, args.force)
    else:
        ap.error("give --all, or --function, --dim and --seed")
    write_csv()


if __name__ == "__main__":
    main()
