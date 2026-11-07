"""Figures from results/raw/.

    python -m experiments.make_plots

  convergence_port5_alpha050.png   - REQUIRED: mean archive-best variance vs evaluations
  convergence_alpha050_all.png     - same for every dataset at alpha = 0.50 (supporting)
  comma_popbest_vs_archive.png     - (mu,lambda) population best vs archive, one seed
  gap_boxplots.png                 - relative gap to SLSQP reference, all 15 cases
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import EXPECTED, all_cases
from src.es import LABELS, VARIANTS

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
SEEDS = range(10)
COLORS = {"1+1": "#1f77b4", "plus": "#d62728", "comma": "#2ca02c"}


def npz(case, v, s):
    return np.load(RAW / case / f"{v.replace('+', 'p')}_seed{s}.npz")


def mean_history(case, v):
    H = []
    for s in SEEDS:
        d = npz(case, v, s)
        e, b = d["history_evals"][1:], d["history_best"][1:]   # drop the inf placeholder at 0
        H.append(b)
    H = np.array(H)
    return e, H.mean(0), H.std(0, ddof=1)


def plot_convergence(ax, case, ref=None):
    for v in VARIANTS:
        e, m, sd = mean_history(case, v)
        ax.plot(e, m, color=COLORS[v], label=LABELS[v], lw=1.5)
        ax.fill_between(e, np.maximum(m - sd, 1e-12), m + sd, color=COLORS[v], alpha=0.15)
    if ref is not None:
        ax.axhline(ref, color="k", ls="--", lw=1, label="SLSQP reference")
    ax.set_yscale("log")
    ax.set_xlabel("Objective evaluations")
    ax.set_ylabel("Best-so-far variance V(w) (log scale)")
    ax.grid(True, which="both", alpha=0.3)


def load_ref():
    with (TAB / "reference.csv").open() as f:
        return {r["case"]: float(r["reference_variance_slsqp"]) for r in csv.DictReader(f)}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    ref = load_ref() if (TAB / "reference.csv").exists() else {}

    # 1. Required convergence plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_convergence(ax, "port5_alpha050", ref.get("port5_alpha050"))
    ax.set_title("port5_alpha050 (n = 225): mean archive best of 10 runs, ±1 SD band")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "convergence_port5_alpha050.png", dpi=150); plt.close(fig)

    # 2. All datasets at alpha = 0.50
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))
    for ax, p in zip(axes, EXPECTED):
        c = f"{p}_alpha050"
        plot_convergence(ax, c, ref.get(c))
        ax.set_title(f"{c} (n = {EXPECTED[p][0]})")
    axes[0].legend(fontsize=8)
    fig.suptitle("Mean archive-best variance over 10 runs (±1 SD), α = 0.50")
    fig.tight_layout(); fig.savefig(FIG / "convergence_alpha050_all.png", dpi=130); plt.close(fig)

    # 3. (mu,lambda) population best vs archive, port5_alpha050 seed 0
    d = npz("port5_alpha050", "comma", 0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(d["pop_evals"], d["pop_best"], lw=0.8, color=COLORS["comma"],
            label="(μ,λ)-ES current population best")
    ax.plot(d["history_evals"][1:], d["history_best"][1:], color="k", lw=1.5,
            label="Archive best-so-far")
    worse = np.nonzero(np.diff(d["pop_best"]) > 0)[0] + 1
    ax.scatter(d["pop_evals"][worse], d["pop_best"][worse], s=6, color="orange",
               label=f"Generations where population best worsened ({len(worse)})", zorder=3)
    ax.set_yscale("log"); ax.set_xlabel("Objective evaluations")
    ax.set_ylabel("Variance V(w) (log scale)")
    ax.set_title("port5_alpha050, seed 0: comma selection vs archive")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "comma_popbest_vs_archive.png", dpi=150); plt.close(fig)

    # 4. Gap-to-reference boxplots for every case
    if ref:
        with (RAW / "runs.csv").open() as f:
            rows = list(csv.DictReader(f))
        fig, axes = plt.subplots(3, 5, figsize=(20, 10), sharex=True)
        for ax, c in zip(axes.T.ravel(), all_cases()):
            # gaps <= 1e-6 % (ES equals the reference to ~8 digits) are floored for the log axis
            data = [[max(100 * (float(r["best_variance"]) - ref[c]) / ref[c], 1e-6)
                     for r in rows if r["case"] == c and r["variant"] == v] for v in VARIANTS]
            bp = ax.boxplot(data, patch_artist=True)
            for patch, v in zip(bp["boxes"], VARIANTS):
                patch.set_facecolor(COLORS[v]); patch.set_alpha(0.5)
            ax.set_xticks([1, 2, 3], [LABELS[v] for v in VARIANTS], fontsize=8)
            ax.set_yscale("log")
            ax.set_title(c, fontsize=10); ax.grid(True, axis="y", alpha=0.3)
        for ax in axes[:, 0]:
            ax.set_ylabel("Gap to SLSQP ref. (%) (log scale)")
        fig.suptitle("Final variance gap to SLSQP reference, 10 seeds per variant "
                     "(gaps below 1e-6 % shown at 1e-6 %)")
        fig.tight_layout(); fig.savefig(FIG / "gap_boxplots.png", dpi=120); plt.close(fig)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
