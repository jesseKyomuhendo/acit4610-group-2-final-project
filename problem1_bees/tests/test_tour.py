# AI-assisted: Claude Code, distance and tour tests, reviewed by <member>
"""Published optimal tours are used here only to check the distance rule.
They are never read by the Bees Algorithm."""
import gzip
from pathlib import Path

import numpy as np
import pytest

from src.tour import closed_route, distance_matrix, tour_length, validate_tour
from src.tsplib import INSTANCES, load_instance

FIXTURES = Path(__file__).parent / "fixtures"


def read_opt_tour(name):
    """Return the optimal tour as 0-based indices, rotated to start at city 1."""
    ids = []
    with gzip.open(FIXTURES / f"{name}.opt.tour.gz", "rt") as f:
        in_tour = False
        for line in f:
            line = line.strip()
            if line == "TOUR_SECTION":
                in_tour = True
            elif in_tour:
                if line in ("-1", "EOF"):
                    break
                ids.extend(int(t) for t in line.split())
    tour = np.array(ids) - 1
    return np.roll(tour, -int(np.where(tour == 0)[0][0]))


@pytest.mark.parametrize("name", INSTANCES)
def test_optimal_tour_lengths_match_published(name):
    inst = load_instance(name)
    dist = distance_matrix(inst.coords)
    tour = read_opt_tour(name)
    validate_tour(tour, inst.dimension)
    assert tour_length(tour, dist) == INSTANCES[name][1]


def test_rounding_rule():
    # 3-4-5 triangle, plus a leg of sqrt(2.25) = 1.5 which rounds up to 2.
    coords = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 5.5]])
    dist = distance_matrix(coords)
    assert dist[0, 1] == 5
    assert dist[1, 2] == 2
    assert np.array_equal(dist, dist.T)
    assert dist.dtype.kind == "i"


def test_length_includes_return_leg():
    coords = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    dist = distance_matrix(coords)
    assert tour_length([0, 1, 2], dist) == 3 + 4 + 5


@pytest.mark.parametrize(
    "tour, msg",
    [([0, 1], "entries"), ([1, 0, 2], "start at city 1"), ([0, 1, 1], "exactly once")],
)
def test_invalid_tours_rejected(tour, msg):
    with pytest.raises(ValueError, match=msg):
        validate_tour(tour, 3)


def test_closed_route_display():
    assert closed_route([0, 2, 1]) == [1, 3, 2, 1]
