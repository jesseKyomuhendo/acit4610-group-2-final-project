"""Pilot tuning of the BA settings.

Runs a few candidate settings on berlin52 (both moves) and Rastrigin D=10 with
tuning seeds 100-102, which are never used in the main experiments (seeds 0-9).
Writes results/tuning/tuning.csv and prints mean results per setting.

Usage (from problem1_bees/):
    python -m experiments.tuning.tune_ba
"""
import csv
from pathlib import Path

import numpy as np

from src.bees import BAParams
from src.continuous_ba import solve_continuous
from src.tsp_ba import solve_tsp
from src.tsplib import load_instance

TUNING_SEEDS = (100, 101, 102)

CANDIDATES = {
    "A": BAParams(n=50, m=10, e=3, nep=20, nsp=10),
    "B": BAParams(n=30, m=8, e=3, nep=30, nsp=10),
    "C": BAParams(n=20, m=6, e=2, nep=30, nsp=15),
    "D": BAParams(n=10, m=5, e=2, nep=40, nsp=20),
    "E": BAParams(n=15, m=3, e=1, nep=50, nsp=25),
    "F": BAParams(n=100, m=20, e=5, nep=15, nsp=5),
}

OUT = Path(__file__).resolve().parents[2] / "results" / "tuning" / "tuning.csv"


def main():
    berlin = load_instance("berlin52")
    rows = []
    for label, p in CANDIDATES.items():
        for seed in TUNING_SEEDS:
            for move in ("swap", "reversal"):
                r = solve_tsp(berlin, move, seed, params=p)
                gap = 100 * (r.best - berlin.optimum) / berlin.optimum
                rows.append([label, p.n, p.m, p.e, p.nep, p.nsp, f"berlin52_{move}", seed, r.best, gap])
            r = solve_continuous("rastrigin", 10, seed, params=p)
            rows.append([label, p.n, p.m, p.e, p.nep, p.nsp, "rastrigin_10", seed, r.best, ""])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "n", "m", "e", "nep", "nsp", "case", "seed", "best", "gap_pct"])
        w.writerows(rows)

    cases = ["berlin52_swap", "berlin52_reversal", "rastrigin_10"]
    print(f"{'setting':8s}" + "".join(f"{c:>20s}" for c in cases))
    for label, p in CANDIDATES.items():
        means = [np.mean([r[8] for r in rows if r[0] == label and r[6] == c]) for c in cases]
        print(f"{label:8s}" + "".join(f"{m:20.2f}" for m in means) + f"   {p}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
