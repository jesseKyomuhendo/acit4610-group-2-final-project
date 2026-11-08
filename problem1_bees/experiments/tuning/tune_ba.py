"""Pilot tuning of the BA settings.

Runs each candidate setting on every tuning case with the tuning seeds, which
are never used in the main experiments. Writes results/tuning/tuning.csv and
prints mean results per setting.

Candidates, cases, seeds and the output path come from config_ba.yaml
(`tuning.candidates`, `tuning.cases`, `tuning.seeds`, `paths.tuning_csv`).

Usage (from problem1_bees/):
    python -m experiments.tuning.tune_ba
"""
import csv

import numpy as np

from prob_1_util.ba_config_loader import PATHS, TUNING_CANDIDATES, TUNING_CASES, TUNING_SEEDS
from src.bees import BAParams
from src.continuous_ba import solve_continuous
from src.tsp_ba import solve_tsp
from src.tsplib import load_instance

CANDIDATES = {label: BAParams(**p) for label, p in TUNING_CANDIDATES.items()}

OUT = PATHS["tuning_csv"]


def case_label(case):
    """Short name used in the CSV, e.g. 'berlin52_swap' or 'rastrigin_10'."""
    if case["problem"] == "tsp":
        return f"{case['instance']}_{case['move']}"
    return f"{case['function']}_{case['dimension']}"


def main():
    # Load each TSP instance once, not once per run.
    instances = {
        c["instance"]: load_instance(c["instance"]) for c in TUNING_CASES if c["problem"] == "tsp"
    }
    rows = []
    for label, p in CANDIDATES.items():
        for seed in TUNING_SEEDS:
            for case in TUNING_CASES:
                if case["problem"] == "tsp":
                    inst = instances[case["instance"]]
                    r = solve_tsp(inst, case["move"], seed, params=p)
                    gap = 100 * (r.best - inst.optimum) / inst.optimum
                else:
                    r = solve_continuous(case["function"], case["dimension"], seed, params=p)
                    gap = ""
                rows.append([label, p.n, p.m, p.e, p.nep, p.nsp, case_label(case), seed, r.best, gap])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "n", "m", "e", "nep", "nsp", "case", "seed", "best", "gap_pct"])
        w.writerows(rows)

    cases = [case_label(c) for c in TUNING_CASES]
    print(f"{'setting':8s}" + "".join(f"{c:>20s}" for c in cases))
    for label, p in CANDIDATES.items():
        means = [np.mean([r[8] for r in rows if r[0] == label and r[6] == c]) for c in cases]
        print(f"{label:8s}" + "".join(f"{m:20.2f}" for m in means) + f"   {p}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()