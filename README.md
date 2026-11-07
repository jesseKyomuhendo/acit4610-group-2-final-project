# ACIT 4610 – Group 2 final project

- `problem1_bees/` – Bees Algorithm for TSP (TSPLIB) and continuous benchmark functions
- `problem2_es/` – (1+1)-ES, (μ+λ)-ES, (μ,λ)-ES for portfolio optimisation (OR-Library)

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Tested with Python 3.13.

## Running
_TODO (Problem 1): one-run and full-set commands, settings, seed lists, data sources, runtimes._

### Problem 2 – ES portfolio optimisation
Data: `problem2_es/data/port1.txt … port5.txt`, the original OR-Library files from
<https://people.brunel.ac.uk/~mastjjb/jeb/orlib/portinfo.html>.

All commands are run from inside `problem2_es/`:
```bash
cd problem2_es
python -m pytest -q                                                    # unit tests
python -m experiments.run_es --case port1_alpha050 --variant plus --seed 0   # one run
python -m experiments.run_es --all          # 15 cases x 3 variants x 10 seeds = 450 runs (~40 min, resumable)
python -m experiments.make_tables           # results/tables/summary.{csv,md,tex}, reference.csv, wilcoxon.csv
python -m experiments.make_plots            # results/figures/*.png
```
Variants: `1+1`, `plus` = (μ+λ), `comma` = (μ,λ). Cases: `port{1..5}_alpha{025,050,075}`.
Seeds: 0–9 (`numpy.random.default_rng(seed)`). Budget: 30,000 evaluations per run.

| Setting | (1+1)-ES | (μ+λ)-ES / (μ,λ)-ES |
|---|---|---|
| Population | 1 parent, 1 child | μ = 10, λ = 70 (final batch 30) |
| Init weights | U[0,1]^n then repair | same, 10 parents |
| Initial σ | 0.1/√n (scalar) | 0.1/√n per coordinate |
| σ bounds | [1e-8, 1.0] | [1e-8, 1.0] |
| Step-size control | 1/5 rule: ×e^{1/3} on strict improvement, ×e^{-1/12} otherwise | self-adaptive, τ' = 1/√(2n), τ = 1/√(2√n) |
| Recombination | – | 2 distinct parents uniformly; w = mean, σ = geometric mean |
| Survivor selection | child replaces parent if strictly lower V | best μ of parents+offspring / offspring only |
| Repair | simplex projection (Duchi et al. 2008), then convex move toward argmax r if R < R_min | same |



See `docs/PLAN.md` for design decisions.
