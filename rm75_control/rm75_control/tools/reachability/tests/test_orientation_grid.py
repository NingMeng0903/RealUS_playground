"""Tests for tool-axis / roll grids (icosphere, Fibonacci, nearest, neighbours)."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.tools.reachability.data_model import (
    IcosphereToolAxisGrid,
    OrientationGridConfig,
    RollGrid,
    ToolAxisGrid,
    make_tool_axis_grid,
)
from rm75_control.tools.reachability.data_model.orientation_grid import (
    fibonacci_sphere,
    subdivide_icosphere,
)


@pytest.mark.parametrize(
    "subdiv,expected_v,expected_f",
    [(0, 12, 20), (1, 42, 80), (2, 162, 320), (3, 642, 1280)],
)
def test_icosphere_counts(subdiv, expected_v, expected_f):
    v, f = subdivide_icosphere(subdiv)
    assert v.shape == (expected_v, 3)
    assert f.shape == (expected_f, 3)
    # unit-length
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-10)


def test_fibonacci_sphere_unit_length():
    v = fibonacci_sphere(200)
    assert v.shape == (200, 3)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-10)


def test_make_tool_axis_grid_icosphere():
    g = make_tool_axis_grid(OrientationGridConfig(kind="icosphere", subdiv=3))
    assert isinstance(g, IcosphereToolAxisGrid)
    assert g.n == 642
    assert g.faces.shape == (1280, 3)


def test_make_tool_axis_grid_fibonacci():
    g = make_tool_axis_grid(OrientationGridConfig(kind="fibonacci", fibonacci_n=200))
    assert isinstance(g, ToolAxisGrid)
    assert g.n == 200


def test_nearest_returns_self():
    g = IcosphereToolAxisGrid.build(subdiv=2)
    for i in (0, 17, 100, g.n - 1):
        assert g.nearest(g.vectors[i]) == i


def test_nearest_batch():
    g = IcosphereToolAxisGrid.build(subdiv=2)
    idx = g.nearest_batch(g.vectors[[0, 5, 100]])
    np.testing.assert_array_equal(idx, [0, 5, 100])


def test_neighbors_include_self_and_close_verts():
    g = IcosphereToolAxisGrid.build(subdiv=3)
    seed = 0
    # subdiv=3 spacing ~11°; half_angle=15° should include several neighbours
    nb = g.neighbors(seed, half_angle_deg=15.0)
    assert seed in nb.tolist()
    assert len(nb) >= 6  # at least the seed + a handful of ring-1 neighbours


def test_neighbors_of_dir_matches_neighbors_of_vertex():
    g = IcosphereToolAxisGrid.build(subdiv=2)
    seed = 10
    a = set(g.neighbors(seed, half_angle_deg=20.0).tolist())
    b = set(g.neighbors_of_dir(g.vectors[seed], half_angle_deg=20.0).tolist())
    assert a == b


def test_roll_grid_default_15deg():
    r = RollGrid(step_deg=15.0)
    assert r.n == 24
    a = r.angles_rad
    assert a.shape == (24,)
    np.testing.assert_allclose(a[0], 0.0)
    np.testing.assert_allclose(a[1], np.deg2rad(15.0))
    assert r.nearest(np.deg2rad(7.5)) in (0, 1)
    assert r.nearest(np.deg2rad(15.1)) == 1
    assert r.nearest(np.deg2rad(360.0 + 30.0)) == 2


def test_normalization_in_constructor():
    # deliberately non-unit input
    v = np.array([[2.0, 0, 0], [0, 3.0, 0]])
    g = ToolAxisGrid(vectors=v)
    np.testing.assert_allclose(np.linalg.norm(g.vectors, axis=1), 1.0)
