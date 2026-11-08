"""Run the Bees Algorithm on the TSPLIB instances.

One run per (instance, move, seed). Each run is saved to
results/raw/tsp/<instance>_<move>_s<seed>.npz with the best-so-far history and
final tour; results/raw/tsp_runs.csv is rebuilt from those files after every
invocation. Existing runs are skipped, so an interrupted --all can be resumed.

Seeds and output paths come from config_ba.yaml (`seeds`,
`paths.tsp_runs_dir`, `paths.tsp_runs_csv`).

Usage (from problem1_bees/):
    python -m experiments.run_tsp --instance berlin52 --move swap --seed 0
    python -m experiments.run_tsp --all
"""
import argparse
import csv
import time

import numpy as np

from prob_1_util.ba_config_loader import PATHS, SEEDS
from src.tour import closed_route
from src.tsp_ba import MOVES, TSP_PARAMS, solve_tsp
from src.tsplib import INSTANCES, load_instance

RUN_DIR = PATHS["tsp_runs_dir"]
CSV_PATH = PATHS["tsp_runs_csv"]
COLUMNS = [
    "instance", "cities", "optimum", "move", "seed", "initial_best", "best",
    "gap_pct", "evaluations", "iterations", "runtime_s", "tour",
]


def run_path(instance, move, seed):
    return RUN_DIR / f"{instance}_{move}_s{seed}.npz"


def run_one(instance, move, seed, force=False):
    path = run_path(instance, move, seed)
    if path.exists() and not force:
        print(f"skip {path.name} (exists)")
        return
    inst = load_instance(instance)
    start = time.perf_counter()
    r = solve_tsp(inst, move, seed)
    runtime = time.perf_counter() - start
    gap = 100.0 * (r.best - inst.optimum) / inst.optimum

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        instance=instance,
        cities=inst.dimension,
        optimum=inst.optimum,
        move=move,
        seed=seed,
        initial_best=r.initial_best,
        best=r.best,
        gap_pct=gap,
        evaluations=r.evaluations,
        iterations=r.iterations,
        runtime_s=runtime,
        tour=r.best_x,
        history_evals=np.array(r.history_evals),
        history_best=np.array(r.history_best),
        params=str(TSP_PARAMS),
    )
    print(f"{instance:9s} {move:8s} seed {seed}: {r.best} (gap {gap:.2f}%) in {runtime:.2f}s")


def write_csv():
    rows = []
    for path in sorted(RUN_DIR.glob("*.npz")):
        d = np.load(path)
        row = {c: d[c].item() for c in COLUMNS if c != "tour"}
        row["gap_pct"] = round(row["gap_pct"], 4)
        row["runtime_s"] = round(row["runtime_s"], 4)
        row["tour"] = " ".join(map(str, closed_route(d["tour"])))
        rows.append(row)
    order = {name: i for i, name in enumerate(INSTANCES)}
    rows.sort(key=lambda r: (order[r["instance"]], r["move"], r["seed"]))
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {CSV_PATH} ({len(rows)} runs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="run every instance, move and seed")
    ap.add_argument("--instance", choices=INSTANCES)
    ap.add_argument("--move", choices=MOVES)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--force", action="store_true", help="rerun even if the result exists")
    args = ap.parse_args()

    if args.all:
        for instance in INSTANCES:
            for move in MOVES:
                for seed in SEEDS:
                    run_one(instance, move, seed, args.force)
    elif args.instance and args.move and args.seed is not None:
        run_one(args.instance, args.move, args.seed, args.force)
    else:
        ap.error("give --all, or --instance, --move and --seed")
    write_csv()


if __name__ == "__main__":
    main()