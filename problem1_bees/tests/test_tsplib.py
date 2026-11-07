import pytest

from src.tsplib import INSTANCES, load_instance, parse_tsplib

VALID = """NAME : tiny
TYPE : TSP
DIMENSION : 3
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
2 3.0 4.0
1 0.0 0.0
3 6.0 8.0
EOF
"""


def write(tmp_path, text):
    p = tmp_path / "tiny.tsp"
    p.write_text(text)
    return p


@pytest.mark.parametrize("name", INSTANCES)
def test_all_assignment_instances_load(name):
    inst = load_instance(name)
    assert inst.dimension == INSTANCES[name][0]
    assert inst.coords.shape == (inst.dimension, 2)
    assert inst.optimum == INSTANCES[name][1]


def test_rows_sorted_by_city_id(tmp_path):
    inst = parse_tsplib(write(tmp_path, VALID))
    assert inst.coords.tolist() == [[0, 0], [3, 4], [6, 8]]


@pytest.mark.parametrize(
    "old, new, msg",
    [
        ("TYPE : TSP", "TYPE : ATSP", "TYPE"),
        ("EUC_2D", "GEO", "EDGE_WEIGHT_TYPE"),
        ("DIMENSION : 3", "DIMENSION : 4", "rows"),
        ("2 3.0 4.0", "1 3.0 4.0", "duplicate"),
        ("NAME : tiny\n", "", "NAME"),
        ("NODE_COORD_SECTION", "DISPLAY_DATA_SECTION", "NODE_COORD_SECTION"),
    ],
)
def test_invalid_files_rejected(tmp_path, old, new, msg):
    with pytest.raises(ValueError, match=msg):
        parse_tsplib(write(tmp_path, VALID.replace(old, new)))


def test_expected_dimension_enforced(tmp_path):
    with pytest.raises(ValueError, match="expected 5"):
        parse_tsplib(write(tmp_path, VALID), expected_dimension=5)
