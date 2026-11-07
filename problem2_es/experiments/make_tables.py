"""Build summary tables from results/raw/runs.csv.

    python -m experiments.make_tables

Outputs in results/tables/:
  summary.csv / summary.md / summary.tex   - required table (15 cases x 3 variants)
  reference.csv                            - SLSQP reference variance per case
  wilcoxon.csv                             - pairwise rank-sum tests per case
Variance statistics use valid (feasible) runs only; runtime is averaged over all runs.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ranksums

from src.data import all_cases, load_portfolio, parse_case
from src.es import LABELS, VARIANTS
from src.reference import reference_min_variance

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "raw" / "runs.csv"
OUT = ROOT / "results" / "tables"


def load_runs():
    with RUNS.open() as f:
        rows = list(csv.DictReader(f))
    by = defaultdict(list)
    for r in rows:
        by[(r["case"], r["variant"])].append(r)
    return by


def reference_table():
    ref, ports = {}, {}
    for c in all_cases():
        name, alpha = parse_case(c)
        ports.setdefault(name, load_portfolio(name))
        _, v = reference_min_variance(ports[name], ports[name].r_min(alpha))
        ref[c] = v
    with (OUT / "reference.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "reference_variance_slsqp"])
        for c, v in ref.items():
            w.writerow([c, f"{v:.10e}"])
    return ref


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    by = load_runs()
    ref = reference_table()
    rows = []
    for c in all_cases():
        for v in VARIANTS:
            runs = by.get((c, v), [])
            if not runs:
                continue
            valid = [float(r["best_variance"]) for r in runs if r["feasible"] == "1"]
            var = np.array(valid)
            gap = 100 * (var - ref[c]) / ref[c]
            rows.append(dict(
                case=c, variant=LABELS[v], n=int(runs[0]["n"]),
                r_min=float(runs[0]["r_min"]),
                best=var.min(), mean=var.mean(), std=var.std(ddof=1) if len(var) > 1 else 0.0,
                feasible=f"{len(valid)}/{len(runs)}",
                runtime=np.mean([float(r["runtime_s"]) for r in runs]),
                ref=ref[c], mean_gap=gap.mean(), best_gap=gap.min(),
                held=np.mean([int(r["n_assets_held"]) for r in runs]),
                evals=",".join(sorted({r["evals"] for r in runs}))))

    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    hdr = ("| Case | Variant | n | R_min | Best V | Mean V | SD V | Feasible | "
           "Mean runtime (s) | Ref. V (SLSQP) | Mean gap (%) |")
    md = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    tex = [r"\begin{tabular}{llrrrrrrrr}", r"\hline",
           r"Case & Variant & $n$ & $R_{\min}$ & Best $V$ & Mean $V$ & SD & Feas. & Time (s) & Gap (\%) \\",
           r"\hline"]
    for r in rows:
        md.append(f"| {r['case']} | {r['variant']} | {r['n']} | {r['r_min']:.6f} | {r['best']:.6e} | "
                  f"{r['mean']:.6e} | {r['std']:.2e} | {r['feasible']} | {r['runtime']:.2f} | "
                  f"{r['ref']:.6e} | {r['mean_gap']:.2f} |")
        v = r["variant"].replace("μ", r"$\mu$").replace("λ", r"$\lambda$")
        tex.append(f"{r['case'].replace('_', r'\_')} & {v} & {r['n']} & {r['r_min']:.5f} & "
                   f"{r['best']:.4e} & {r['mean']:.4e} & {r['std']:.1e} & {r['feasible']} & "
                   f"{r['runtime']:.2f} & {r['mean_gap']:.2f} \\\\")
    tex += [r"\hline", r"\end{tabular}"]
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "summary.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    # Pairwise Wilcoxon rank-sum tests on final variance (two-sided)
    pairs = [("1+1", "plus"), ("1+1", "comma"), ("plus", "comma")]
    with (OUT / "wilcoxon.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["case"] + [f"p_{a}_vs_{b}" for a, b in pairs])
        for c in all_cases():
            out = [c]
            for a, b in pairs:
                xa = [float(r["best_variance"]) for r in by[(c, a)]]
                xb = [float(r["best_variance"]) for r in by[(c, b)]]
                out.append(f"{ranksums(xa, xb).pvalue:.3e}")
            wr.writerow(out)
    print("summary written to", OUT)


if __name__ == "__main__":
    main()
