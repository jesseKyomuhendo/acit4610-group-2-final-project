"""Summary tables for Problem 1, built from results/raw only.

Writes CSV and Markdown versions to results/tables/:
  tsp_summary_<move>   instance, cities, optimum, best/mean/std length, mean gap, mean runtime
  tsp_wilcoxon         paired Wilcoxon signed-rank test, swap vs reversal, per instance
  continuous_summary   function, dimension, best/mean/std final value, mean runtime

Usage (from problem1_bees/):
    python -m experiments.make_tables
"""
import csv

import numpy as np
from scipy.stats import wilcoxon

from src.functions import BENCHMARKS, DIMENSIONS
from src.tsp_ba import MOVES
from src.tsplib import INSTANCES

from .results_io import FUNCTION_LABELS, MOVE_LABELS, TABLES, load_runs


def write(name, header, rows):
    TABLES.mkdir(parents=True, exist_ok=True)
    with open(TABLES / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    (TABLES / f"{name}.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLES / name}.csv/.md")


def tsp_tables():
    header = ["Instance", "Cities", "Optimum", "Best", "Mean", "Std", "Mean gap (%)", "Mean runtime (s)"]
    finals = {}
    for move in MOVES:
        rows = []
        for name in INSTANCES:
            runs = load_runs("tsp", instance=name, move=move)
            best = np.array([int(d["best"]) for d in runs])
            finals[name, move] = best
            rows.append([
                name,
                int(runs[0]["cities"]),
                int(runs[0]["optimum"]),
                best.min(),
                f"{best.mean():.1f}",
                f"{best.std(ddof=1):.1f}",
                f"{np.mean([float(d['gap_pct']) for d in runs]):.2f}",
                f"{np.mean([float(d['runtime_s']) for d in runs]):.3f}",
            ])
        write(f"tsp_summary_{move}", header, rows)

    rows = []
    for name in INSTANCES:
        swap, rev = finals[name, "swap"], finals[name, "reversal"]
        p = wilcoxon(swap, rev).pvalue
        rows.append([name, f"{swap.mean():.1f}", f"{rev.mean():.1f}", f"{p:.4f}"])
    write(
        "tsp_wilcoxon",
        ["Instance", f"Mean {MOVE_LABELS['swap']}", f"Mean {MOVE_LABELS['reversal']}", "Wilcoxon p (paired by seed)"],
        rows,
    )


def continuous_table():
    header = ["Function", "D", "Best", "Mean", "Std", "Mean runtime (s)"]
    rows = []
    for function in BENCHMARKS:
        for dim in DIMENSIONS:
            runs = load_runs("continuous", function=function, dim=dim)
            best = np.array([float(d["best"]) for d in runs])
            rows.append([
                FUNCTION_LABELS[function],
                dim,
                f"{best.min():.4g}",
                f"{best.mean():.4g}",
                f"{best.std(ddof=1):.3g}",
                f"{np.mean([float(d['runtime_s']) for d in runs]):.3f}",
            ])
    write("continuous_summary", header, rows)


def main():
    tsp_tables()
    continuous_table()


if __name__ == "__main__":
    main()
