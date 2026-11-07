# ACIT 4610 – Group 2 final project

| Folder | Problem |
|---|---|
| [`problem1_bees/`](problem1_bees/README.md) | Bees Algorithm for the TSP (TSPLIB) and continuous benchmark functions |
| `problem2_es/` | (1+1)-ES, (μ+λ)-ES and (μ,λ)-ES for portfolio optimisation (OR-Library) |

Each problem folder has its own README with the exact commands, settings, seeds and data sources.

## Setup

Requires Python 3.11 or newer (tested with 3.13.13).

```bash
git clone https://github.com/jesseKyomuhendo/acit4610-group-2-final-project.git
cd acit4610-group-2-final-project
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies (pinned in `requirements.txt`): numpy 2.3.4, scipy 1.16.3, matplotlib 3.10.7, pytest 8.4.2.

See `docs/PLAN.md` for design decisions.
