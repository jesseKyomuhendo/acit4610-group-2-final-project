"""Load config_ba.yaml once and expose its values as module-level constants.

Every Problem 1 module imports its settings from here, so config_ba.yaml is the
single place to look up a parameter. This module only reads the YAML; it holds
no values of its own. It does not import anything from src/, which avoids
circular imports (src modules import from here, never the other way round).

Check that the config loads (from problem1_bees/):
    python -m prob_1_util.ba_config_loader
"""
from pathlib import Path

import yaml

# Resolve the YAML relative to this file, not the working directory, so it
# loads the same way from run scripts, tuning and pytest.
PROBLEM_DIR = Path(__file__).resolve().parents[1]  # problem1_bees/
CONFIG_PATH = PROBLEM_DIR / "config_ba.yaml"

# raw_config is the whole YAML file as nested dicts. Other modules import the
# named UPPERCASE constants below, not this dict, so every setting has exactly
# one name to search for.
with open(CONFIG_PATH, encoding="utf-8") as f:
    raw_config = yaml.safe_load(f)

# Debug value, printed by the main guard below.
LOAD_MESSAGE = raw_config["load_message"]

# --- Shared by Part A and Part B ---------------------------------------------
BUDGET = raw_config["budget"]
LOG_EVERY = raw_config["log_every"]
SEEDS = tuple(raw_config["seeds"])

# Absolute paths, so scripts work from any working directory.
PATHS = {name: PROBLEM_DIR / rel for name, rel in raw_config["paths"].items()}

# --- Part A: TSP ---------------------------------------------------------------
tsp_section = raw_config["tsp"]
# name -> (cities, optimum), the same shape src/tsplib.py already uses.
INSTANCES = {name: (v["cities"], v["optimum"]) for name, v in tsp_section["instances"].items()}
REQUIRED_TYPE = tsp_section["required_type"]
REQUIRED_EDGE_WEIGHT_TYPE = tsp_section["required_edge_weight_type"]
MOVE_NAMES = tuple(tsp_section["moves"])
TSP_BA = dict(tsp_section["ba"])  # n, m, e, nep, nsp  ->  BAParams(**TSP_BA)
TSP_CONVERGENCE_INSTANCES = tuple(tsp_section["convergence_plot_instances"])
ROUTE_PLOT_INSTANCE = tsp_section["route_plot_instance"]

# --- Part B: continuous --------------------------------------------------------
continuous_section = raw_config["continuous"]
# name -> (lower, upper, minimiser)
FUNCTION_BOUNDS = {
    name: (v["lower"], v["upper"], v["minimiser"]) for name, v in continuous_section["functions"].items()
}
DIMENSIONS = tuple(continuous_section["dimensions"])
PATCH_FRACTION = continuous_section["patch_fraction"]
CONTINUOUS_BA = dict(continuous_section["ba"])  # ->  BAParams(**CONTINUOUS_BA)
RASTRIGIN_CONVERGENCE = dict(continuous_section["convergence_plot"])  # {function, dimension}
BOXPLOT_DIMENSION = continuous_section["boxplot_dimension"]

# --- Pilot tuning --------------------------------------------------------------
tuning_section = raw_config["tuning"]
TUNING_SEEDS = tuple(tuning_section["seeds"])
TUNING_CASES = [dict(c) for c in tuning_section["cases"]]
TUNING_CANDIDATES = {label: dict(p) for label, p in tuning_section["candidates"].items()}


if __name__ == "__main__":
    # Debug only: if this prints, the YAML was found and parsed.
    print(LOAD_MESSAGE)