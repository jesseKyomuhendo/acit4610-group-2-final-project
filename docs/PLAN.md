# Implementation plan (agreed 2026-11-01)

Spec: `docs/Portfolio 2026.pdf`. Deadline 30 Nov 2026 12:00, Inspera.

## Layout
`problem1_bees/` and `problem2_es/` are fully self-contained, each with
`data/ src/ experiments/ results/{raw,figures,tables}/ tests/`.
`src/budget.py` is duplicated deliberately (one folder per problem, per spec).

## Evaluation budget
- All objective calls go through `Evaluator` (budget 30,000). It counts initial
  population, recruits/offspring, scouts and rejected candidates.
- Stops mid-iteration (BudgetExhausted / `remaining()` checks); ES final batch may be < 70.
- Tracks best-so-far = archive; records `initial_best`.
- History logged every 10 evals (incl. 0 and 30,000) to keep committed results small.
- Test: `ev.n == 30_000` at end of every algorithm. ES repair is not an evaluation; never re-evaluate cached fitness.

## Data
- TSPLIB: read `.gz` with `gzip.open`; also commit decompressed `.tsp`. Validate NAME, TYPE=TSP,
  DIMENSION (= table), EDGE_WEIGHT_TYPE=EUC_2D, NODE_COORD_SECTION, row count, unique ids; raise ValueError.
- Integer distance matrix `floor(sqrt(.)+0.5)`. `.opt.tour` files only in `tests/fixtures/`.
- OR-Library: symmetric C, unit diagonal, record counts as in spec; port1 checkpoint test (Req, V(u), Rmin(0.5)).
- Continuous functions tested at minimiser + one other point vs SFU values.

## BA (Problem 1)
Classic Pham, fixed neighbourhood, no shrinking/abandonment.
n=50, m=10, e=3, nep=20, nsp=10 (same for swap and reversal), confirmed by pilot
in `experiments/tuning/` on seeds 100-102. Site moves to best recruit only if strictly better.
Continuous move: per-coordinate U(-10%, +10%) of range, clipped. City 1 fixed.

## ES (Problem 2)
- Mutate repaired `w` (genotype = repaired weights).
- Own Duchi et al. (2008) simplex projection, unit-tested.
- (1+1): 1 init eval + 29,999 children. mu-ES: 10 init + 428x70 + 30.
- Numeric feasibility check on every final solution (1e-8 tolerances); report rate honestly.
- Survivors keep their sigma vector; test that (mu,lambda) pop-best can worsen while archive cannot.

## Experiments
- Seeds 0-9 for all main runs. Sequential runs; time only the algorithm (`perf_counter`).
- CLIs: `python -m experiments.run_tsp --instance berlin52 --move swap --seed 0`, `--all`;
  likewise `run_continuous`, `run_es`; then `make_tables`, `make_plots`.
- `results/raw/runs.csv` + one `.npz` per run; resumable (skip existing). `results/environment.json`.
- Results committed.

## Plots / tables
Log-y for Rastrigin and ES convergence (labelled); TSP gap% or length with optimum line;
±1 SD bands, "mean of 10 runs"; D=30 boxplots 4 panels, own y-axes; berlin52 best route with
move, seed, length, gap, start city, return leg. Tables as CSV + Markdown/LaTeX from scripts.
Shared label constants. Optional Wilcoxon rank-sum tests.

## Report (written by humans only)
~6,000 words (limit 3,000-7,500), PDF, group no., names, repo link, AI statement, no code.
Checklist in `docs/report_checklist.md` mapping every spec Report/Discussion item to a section.
Mark AI code via `# AI-assisted:` headers and `AI_USAGE.md`. Members commit under own identity.
