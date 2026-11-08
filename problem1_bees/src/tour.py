"""Distances, tour length and tour validation for symmetric EUC_2D TSP.

A tour is a permutation of 0-based city indices (index k is TSPLIB city k + 1).
City 1 (index 0) is always first and is not repeated at the end; the return
leg back to it is added when computing the length.
"""
import numpy as np


def distance_matrix(coords):
    """TSPLIB EUC_2D distances: d_ij = floor(sqrt(dx^2 + dy^2) + 0.5), as int."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.floor(np.sqrt((diff**2).sum(axis=2)) + 0.5).astype(np.int64)


def tour_length(tour, dist):
    """Sum of the rounded legs, including the return leg to the first city."""
    tour = np.asarray(tour)
    return int(dist[tour, np.roll(tour, -1)].sum())


def validate_tour(tour, n):
    """Raise ValueError unless tour visits every city once and starts at city 1."""
    tour = np.asarray(tour)
    if tour.shape != (n,):
        raise ValueError(f"tour has {tour.size} entries, expected {n}")
    if tour[0] != 0:
        raise ValueError(f"tour must start at city 1, starts at city {tour[0] + 1}")
    if not np.array_equal(np.sort(tour), np.arange(n)):
        raise ValueError("tour must contain every city exactly once")


def closed_route(tour):
    """1-based city ids with city 1 appended at the end, for display only."""
    ids = [int(c) + 1 for c in tour]
    return ids + [ids[0]]
