# Problem 1 – Bees Algorithm

The Bees Algorithm (Pham et al., 2006) applied to

- **Part A:** the symmetric Travelling Salesman Problem on six TSPLIB instances, with two neighbourhood moves (swap and segment reversal);
- **Part B:** four continuous benchmark functions (Sphere, Rosenbrock, Rastrigin, Ackley) at D = 10 and D = 30.

Every run uses exactly **30,000 objective evaluations**. The initial population, every recruit and every fresh scout are counted, including rejected candidates. A run stops as soon as the budget is spent, even partway through an iteration.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd problem1_bees
```

All commands below are run from `problem1_bees/`.

Tested with Python 3.13.13 on macOS (Apple M3 Pro) with:

| Package | Version |
|---|---|
| numpy | 2.3.4 |
| scipy | 1.16.3 |
| matplotlib | 3.10.7 |
| pytest | 8.4.2 |

## Quick check

```bash
python -m pytest -q tests
```

The tests validate the TSPLIB loader, check our distance function against the published optimal tours, check each benchmark function at its minimiser and at other points, and check that every algorithm uses exactly 30,000 evaluations.

## Running

### One run

```bash
python -m experiments.run_tsp --instance berlin52 --move reversal --seed 0
python -m experiments.run_continuous --function rastrigin --dim 10 --seed 0
```

Options: `--instance` is one of `eil51 berlin52 st70 eil76 kroA100 ch150`; `--move` is `swap` or `reversal`; `--function` is one of `sphere rosenbrock rastrigin ackley`; `--dim` is `10` or `30`.

Runs whose result file already exists are skipped. Add `--force` to rerun.

### Full experiment set

```bash
python -m experiments.run_tsp --all          # 120 runs: 6 instances x 2 moves x 10 seeds (about 1 min)
python -m experiments.run_continuous --all   # 80 runs: 4 functions x 2 dimensions x 10 seeds (about 20 s)
python -m experiments.make_tables
python -m experiments.make_plots
```

To reproduce everything from scratch, delete `results/raw/tsp/` and `results/raw/continuous/` first (or pass `--force`). Results are deterministic for a given seed; only the measured runtimes change.

### Pilot tuning

```bash
python -m experiments.tuning.tune_ba
```

Tries six BA settings on berlin52 (both moves) and Rastrigin D = 10 and writes `results/tuning/tuning.csv`.

## Settings

### Bees Algorithm

| Parameter | Meaning | TSP (both moves) | Continuous |
|---|---|---|---|
| n | scout bees / sites | 15 | 30 |
| m | selected sites | 3 | 8 |
| e | elite sites (among m) | 1 | 3 |
| nep | recruits per elite site | 50 | 30 |
| nsp | recruits per other selected site | 25 | 10 |
| | evaluations per iteration | 112 | 162 |

Each iteration ranks the sites, recruits nep or nsp bees around each selected site (one neighbourhood move each), moves a site to its best recruit only if that recruit is strictly better, and replaces the n − m unselected sites with fresh random scouts. The neighbourhood size is fixed (no shrinking or site abandonment).

The settings were chosen by the pilot tuning above (settings E and B in `results/tuning/tuning.csv`).

### Representations and moves

- **TSP:** a permutation of the cities with city 1 fixed in the first position (not repeated at the end). Scouts are uniformly shuffled permutations of cities 2..n. A recruit applies one random move to positions 2..n, so city 1 never moves:
  - *swap* exchanges the cities at two random positions;
  - *reversal* reverses the segment between two random positions.
- **Continuous:** a vector of D reals. Scouts are drawn uniformly within the bounds. A recruit adds an independent uniform change in [−10%, +10%] of (upper − lower) to every coordinate and clips the result to the bounds.

### Seeds

| Use | Seeds |
|---|---|
| Main experiments (every instance, move, function and dimension) | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| Pilot tuning only | 100, 101, 102 |

Each seed initialises `numpy.random.default_rng(seed)`. Both TSP moves use the same seeds, so they start from the same initial population.

## Data

| File | Cities | Optimal length | Source |
|---|---|---|---|
| `data/eil51.tsp.gz` | 51 | 426 | http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/eil51.tsp.gz |
| `data/berlin52.tsp.gz` | 52 | 7542 | http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/berlin52.tsp.gz |
| `data/st70.tsp.gz` | 70 | 675 | http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/st70.tsp.gz |
| `data/eil76.tsp.gz` | 76 | 538 | http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/eil76.tsp.gz |
| `data/kroA100.tsp.gz` | 100 | 21282 | http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/kroA100.tsp.gz |
| `data/ch150.tsp.gz` | 150 | 6528 | http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ch150.tsp.gz |

The `.tsp.gz` files are the TSPLIB originals and are what the code reads. The `.tsp` files beside them are decompressed copies for convenience. Optimal lengths: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html

The loader requires `TYPE = TSP`, `EDGE_WEIGHT_TYPE = EUC_2D`, the expected `DIMENSION`, a `NODE_COORD_SECTION` with that many rows, and unique city ids 1..n. Distances use TSPLIB's rounding, `d_ij = floor(sqrt((a_i − a_j)^2 + (b_i − b_j)^2) + 0.5)`, applied to each edge before summing (including the return edge).

The published optimal tours in `tests/fixtures/*.opt.tour.gz` are used only by the tests, to check the distance calculation. They are never used to initialise or guide the algorithm.

The continuous functions need no data; their bounds are:

| Function | Bounds | Global minimum |
|---|---|---|
| Sphere | [−5.12, 5.12] | f(0) = 0 |
| Rosenbrock | [−5, 10] | f(1) = 0 |
| Rastrigin | [−5.12, 5.12] | f(0) = 0 |
| Ackley | [−32.768, 32.768] | f(0) = 0 |

## Outputs

| Path | Contents |
|---|---|
| `results/raw/tsp_runs.csv` | one row per TSP run: instance, cities, optimum, move, seed, initial best, final best length, gap %, evaluations, iterations, runtime, closed route |
| `results/raw/continuous_runs.csv` | one row per continuous run: function, D, seed, initial best, final best value, evaluations, iterations, runtime, final vector |
| `results/raw/tsp/*.npz`, `results/raw/continuous/*.npz` | per-run files with the same fields plus the best-so-far history (logged every 10 evaluations) |
| `results/tables/` | summary tables (CSV and Markdown) and the swap vs reversal Wilcoxon test |
| `results/figures/` | convergence plots, best berlin52 route, Rastrigin D = 10 convergence, D = 30 boxplots (PDF and PNG) |
| `results/tuning/tuning.csv` | pilot tuning results |

Gap (%) = 100 × (L_BA − L*) / L*, where L* is the optimal length above and L_BA the final best valid tour length. Runtime is wall-clock time of the algorithm only (data loading and saving excluded), measured with `time.perf_counter`.

## Code layout

| File | Purpose |
|---|---|
| `src/tsplib.py` | TSPLIB parsing and validation |
| `src/tour.py` | distance matrix, tour length, tour validation |
| `src/functions.py` | benchmark functions and bounds |
| `src/budget.py` | evaluation counter (30,000 budget, best-so-far history) |
| `src/bees.py` | Bees Algorithm, independent of representation |
| `src/tsp_ba.py` | TSP representation, swap and reversal moves, settings |
| `src/continuous_ba.py` | continuous representation, ±10% move, settings |
| `experiments/` | run scripts, tables, plots, tuning |
| `tests/` | unit and integration tests |
