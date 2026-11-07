"""Run ES experiments.

From inside problem2_es/:
    python -m experiments.run_es --case port1_alpha050 --variant plus --seed 0
    python -m experiments.run_es --all              # 15 cases x 3 variants x 10 seeds

Writes one row per run to results/raw/runs.csv and one .npz per run
(best weights + best-so-far log + population-best log). Resumable: runs whose
.npz already exists are skipped.
"""
import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np

from src.data import SOURCE_URL, all_cases, load_portfolio, parse_case
from src.es import VARIANTS, run
from src.repair import is_feasible

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
RUNS_CSV = RAW / "runs.csv"
SEEDS = list(range(10))
FIELDS = ["case", "dataset", "n", "alpha", "r_min", "variant", "seed", "evals",
          "initial_best", "best_variance", "expected_return", "feasible",
          "min_weight", "sum_weight_err", "n_assets_held", "runtime_s", "final_sigma_mean"]


def npz_path(case, variant, seed):
    return RAW / case / f"{variant.replace('+', 'p')}_seed{seed}.npz"


def run_one(case, variant, seed, cache={}):
    name, alpha = parse_case(case)
    if name not in cache:
        cache[name] = load_portfolio(name)
    port = cache[name]
    r_min = port.r_min(alpha)
    res = run(variant, port, r_min, seed)
    w = res.best_w
    feas = is_feasible(w, port.r, r_min)
    row = dict(case=case, dataset=name, n=port.n, alpha=alpha, r_min=r_min,
               variant=variant, seed=seed, evals=res.evals,
               initial_best=res.initial_best, best_variance=res.best_var,
               expected_return=float(port.r @ w), feasible=int(feas),
               min_weight=float(w.min()), sum_weight_err=float(abs(w.sum() - 1)),
               n_assets_held=int((w > 1e-6).sum()), runtime_s=res.runtime,
               final_sigma_mean=res.final_sigma_mean)
    p = npz_path(case, variant, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, w=w, history_evals=res.history_evals,
                        history_best=res.history_best, pop_evals=res.pop_evals,
                        pop_best=res.pop_best)
    new = not RUNS_CSV.exists()
    with RUNS_CSV.open("a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            wr.writeheader()
        wr.writerow(row)
    return row


def write_environment():
    import matplotlib, scipy
    env = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
               scipy=scipy.__version__, matplotlib=matplotlib.__version__,
               processor=platform.processor(), seeds=SEEDS, data_source=SOURCE_URL)
    (ROOT / "results" / "environment.json").write_text(json.dumps(env, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="e.g. port1_alpha050")
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    if a.all:
        write_environment()
        jobs = [(c, v, s) for c in all_cases() for v in VARIANTS for s in SEEDS]
    else:
        if not (a.case and a.variant and a.seed is not None):
            ap.error("give --case, --variant and --seed, or --all")
        jobs = [(a.case, a.variant, a.seed)]

    for i, (c, v, s) in enumerate(jobs, 1):
        if npz_path(c, v, s).exists():
            continue
        row = run_one(c, v, s)
        print(f"[{i}/{len(jobs)}] {c} {v:5s} seed={s} var={row['best_variance']:.6e} "
              f"feasible={row['feasible']} evals={row['evals']} t={row['runtime_s']:.2f}s",
              flush=True)


if __name__ == "__main__":
    main()
