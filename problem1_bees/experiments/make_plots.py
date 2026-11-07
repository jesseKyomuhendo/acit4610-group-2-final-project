"""Figures for Problem 1, built from results/raw only.

Writes PDF and PNG files to results/figures/:
  tsp_convergence_berlin52, tsp_convergence_ch150
  tsp_route_berlin52
  continuous_convergence_rastrigin_d10
  continuous_boxplots_d30

Usage (from problem1_bees/):
    python -m experiments.make_plots
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.functions import BENCHMARKS
from src.tour import closed_route
from src.tsp_ba import MOVES
from src.tsplib import load_instance

from .results_io import FIGURES, FUNCTION_LABELS, MOVE_LABELS, histories, load_runs

MOVE_COLORS = {"reversal": "#2a78d6", "swap": "#eb6834"}
SERIES_COLOR = "#2a78d6"
INK = "#3d3d3a"
MUTED = "#8a8980"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e4e3dd",
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "lines.linewidth": 2,
})


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGURES / name}.pdf/.png")


def mean_band(ax, evals, values, color, label):
    mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
    ax.fill_between(evals, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0)
    ax.plot(evals, mean, color=color, label=label)


def tsp_convergence(name):
    inst = load_instance(name)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for move in ("reversal", "swap"):
        evals, values = histories(load_runs("tsp", instance=name, move=move))
        mean_band(ax, evals, values, MOVE_COLORS[move], f"{MOVE_LABELS[move]} (mean ± 1 SD)")
    ax.axhline(inst.optimum, color=INK, linestyle="--", linewidth=1.2, label=f"Optimum ({inst.optimum})")
    ax.set_xlim(0, 30_000)
    ax.set_xlabel("Objective evaluations")
    ax.set_ylabel("Best-so-far tour length")
    ax.set_title(f"{name}: mean best-so-far tour length over 10 runs")
    ax.legend()
    save(fig, f"tsp_convergence_{name}")


def tsp_best_route():
    inst = load_instance("berlin52")
    runs = [d for move in MOVES for d in load_runs("tsp", instance="berlin52", move=move)]
    best = min(runs, key=lambda d: (int(d["best"]), str(d["move"]), int(d["seed"])))
    route = np.array(closed_route(best["tour"])) - 1
    xy = inst.coords[route]
    move = str(best["move"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xy[:-1, 0], xy[:-1, 1], color=SERIES_COLOR, linewidth=1.5, zorder=1, label="Tour")
    ax.plot(xy[-2:, 0], xy[-2:, 1], color=SERIES_COLOR, linewidth=1.5, linestyle=":", zorder=1, label="Return leg to city 1")
    ax.scatter(inst.coords[:, 0], inst.coords[:, 1], s=22, color="white", edgecolor=INK, linewidth=1, zorder=2)
    ax.scatter(*inst.coords[0], s=110, marker="*", color="#eb6834", edgecolor="white", linewidth=1, zorder=3, label="Start: city 1")
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.set_aspect("equal")
    ax.set_title(
        f"berlin52 shortest tour found: {MOVE_LABELS[move]}, seed {int(best['seed'])}\n"
        f"length {int(best['best'])}, gap {float(best['gap_pct']):.2f}% (optimum {inst.optimum})"
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "tsp_route_berlin52")


def rastrigin_convergence():
    evals, values = histories(load_runs("continuous", function="rastrigin", dim=10))
    fig, ax = plt.subplots(figsize=(6.4, 4))
    mean_band(ax, evals, values, SERIES_COLOR, "Mean ± 1 SD of 10 runs")
    ax.set_yscale("log")
    ax.set_xlim(0, 30_000)
    ax.set_xlabel("Objective evaluations")
    ax.set_ylabel("Best-so-far f(x) (log scale)")
    ax.set_title("Rastrigin, D = 10: mean best-so-far objective")
    ax.legend()
    save(fig, "continuous_convergence_rastrigin_d10")


def boxplots_d30():
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.6))
    for ax, function in zip(axes, BENCHMARKS):
        finals = [float(d["best"]) for d in load_runs("continuous", function=function, dim=30)]
        ax.boxplot(
            finals,
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor="#cde2fb", edgecolor=SERIES_COLOR),
            medianprops=dict(color=INK, linewidth=1.5),
            whiskerprops=dict(color=SERIES_COLOR),
            capprops=dict(color=SERIES_COLOR),
            flierprops=dict(markeredgecolor=SERIES_COLOR, markersize=5),
        )
        ax.set_xticks([])
        ax.set_title(FUNCTION_LABELS[function])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Final best f(x) (10 runs)")
    fig.suptitle("Final objective values at D = 30 (each panel has its own y-axis)", y=1.02)
    fig.tight_layout()
    save(fig, "continuous_boxplots_d30")


def main():
    tsp_convergence("berlin52")
    tsp_convergence("ch150")
    tsp_best_route()
    rastrigin_convergence()
    boxplots_d30()


if __name__ == "__main__":
    main()
