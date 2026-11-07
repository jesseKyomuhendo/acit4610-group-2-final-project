"""Loading and validation of TSPLIB instances.

Only symmetric EUC_2D instances are accepted. Files are read straight from the
original .tsp.gz archives (decompressed in memory); plain .tsp also works.
"""
import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Instances required by the assignment: name -> (cities, optimal tour length).
INSTANCES = {
    "eil51": (51, 426),
    "berlin52": (52, 7542),
    "st70": (70, 675),
    "eil76": (76, 538),
    "kroA100": (100, 21282),
    "ch150": (150, 6528),
}


@dataclass
class TSPInstance:
    name: str
    dimension: int
    coords: np.ndarray  # shape (n, 2); row k holds city id k + 1
    optimum: int | None = None


def _open(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path)


def parse_tsplib(path, expected_dimension=None):
    """Parse a TSPLIB file and validate the fields the assignment requires.

    Raises ValueError (naming the file) if any check fails.
    """
    path = Path(path)
    header = {}
    rows = []
    in_coords = False
    with _open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line == "EOF":
                break
            if in_coords:
                rows.append(line.split())
                continue
            if line.startswith("NODE_COORD_SECTION"):
                in_coords = True
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                header[key.strip()] = value.strip()

    def fail(msg):
        raise ValueError(f"{path.name}: {msg}")

    for key in ("NAME", "TYPE", "DIMENSION", "EDGE_WEIGHT_TYPE"):
        if key not in header:
            fail(f"missing {key}")
    if not in_coords:
        fail("missing NODE_COORD_SECTION")
    if header["TYPE"] != "TSP":
        fail(f"TYPE must be TSP, got {header['TYPE']}")
    if header["EDGE_WEIGHT_TYPE"] != "EUC_2D":
        fail(f"EDGE_WEIGHT_TYPE must be EUC_2D, got {header['EDGE_WEIGHT_TYPE']}")

    try:
        dimension = int(header["DIMENSION"])
    except ValueError:
        fail(f"DIMENSION is not an integer: {header['DIMENSION']}")
    if expected_dimension is not None and dimension != expected_dimension:
        fail(f"DIMENSION {dimension} != expected {expected_dimension}")
    if len(rows) != dimension:
        fail(f"{len(rows)} coordinate rows, expected {dimension}")

    try:
        ids = [int(r[0]) for r in rows]
        xy = np.array([[float(r[1]), float(r[2])] for r in rows])
    except (ValueError, IndexError):
        fail("malformed coordinate row")
    if len(set(ids)) != dimension:
        fail("duplicate city ids")
    if sorted(ids) != list(range(1, dimension + 1)):
        fail("city ids must be 1..DIMENSION")

    # Order rows by city id so coords[k] is city k + 1.
    coords = xy[np.argsort(ids)]
    return TSPInstance(name=header["NAME"], dimension=dimension, coords=coords)


def load_instance(name, data_dir=DATA_DIR):
    """Load one of the six assignment instances from data/<name>.tsp.gz."""
    if name not in INSTANCES:
        raise ValueError(f"unknown instance {name}; choose from {list(INSTANCES)}")
    cities, optimum = INSTANCES[name]
    inst = parse_tsplib(Path(data_dir) / f"{name}.tsp.gz", expected_dimension=cities)
    if inst.name != name:
        raise ValueError(f"{name}.tsp.gz: NAME is {inst.name}")
    inst.optimum = optimum
    return inst
