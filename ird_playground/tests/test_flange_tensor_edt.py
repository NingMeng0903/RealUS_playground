"""Tests for flange occupancy tensor + anisotropic 5-D EDT."""

from __future__ import annotations

import inspect

import numpy as np
import pytest


def test_relative_azimuth_invariant_under_base_yaw():
    """B4: relative azimuth keeps a yaw orbit in one chart voxel."""
    from ird_playground.map.build_flange_tensor import flange_pose_to_chart

    tilt = np.deg2rad(35.0)
    ct, st = np.cos(tilt), np.sin(tilt)
    R0 = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]], dtype=np.float64)
    p0 = np.array([0.3, 0.0, 0.4], dtype=np.float64)
    chart0 = flange_pose_to_chart(p0, R0)
    for deg in (0.0, 15.0, 40.0, 90.0, 150.0, 179.0):
        phi = np.deg2rad(deg)
        c, s = np.cos(phi), np.sin(phi)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        chart = flange_pose_to_chart(Rz @ p0, Rz @ R0)
        assert np.allclose(chart0, chart, atol=1e-5), deg


def test_absolute_azimuth_would_break_yaw_orbit():
    """Sanity: absolute uz azimuth tracks base yaw (the B4 pathology)."""
    p0 = np.array([0.3, 0.0, 0.4], dtype=np.float64)
    uz0 = np.array([np.sin(np.deg2rad(35.0)), 0.0, np.cos(np.deg2rad(35.0))])
    abs0 = np.arctan2(uz0[1], uz0[0])
    phi = np.deg2rad(60.0)
    c, s = np.cos(phi), np.sin(phi)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    uz1 = Rz @ uz0
    abs1 = np.arctan2(uz1[1], uz1[0])
    assert abs(abs1 - abs0) > 0.5


def test_self_collision_filter_kwargs_match_api():
    """Builder must use keyword ``security_margin`` (not ``security_margin_m``)."""
    from rm75_control.tools.reachability.build.self_collision import SelfCollisionFilter

    params = inspect.signature(SelfCollisionFilter.__init__).parameters
    assert "security_margin" in params
    assert "security_margin_m" not in params
    assert params["kin_urdf"].kind == inspect.Parameter.KEYWORD_ONLY


def test_unknown_vs_occupied_semantics():
    """Empty cells within budget are UNKNOWN (0), not marked unreachable."""
    from ird_playground.map.build_flange_tensor import (
        OCC_OCCUPIED,
        OCC_UNKNOWN,
        UNKNOWN_POLICY,
        FlangeOccupancyConfig,
        chart_coords_to_indices,
    )

    cfg = FlangeOccupancyConfig()
    axes = cfg.axis_arrays()
    shape = tuple(len(a) for a in axes)
    assert shape == (58, 41, 16, 31, 31)
    assert int(np.prod(shape)) == 36_564_128
    occupancy = np.zeros(shape, dtype=np.uint8)
    assert occupancy.dtype == np.uint8
    assert (occupancy == OCC_UNKNOWN).all()
    # Simulate one FK hit.
    chart = np.array([[0.4, 0.3, 0.5, 0.1, -0.2]], dtype=np.float32)
    idx = chart_coords_to_indices(chart, axes)
    occupancy[idx] = OCC_OCCUPIED
    assert int((occupancy == OCC_OCCUPIED).sum()) == 1
    assert int((occupancy == OCC_UNKNOWN).sum()) == occupancy.size - 1
    assert "unknown" in UNKNOWN_POLICY.lower()
    assert "not unreachable" in UNKNOWN_POLICY.lower()


def test_periodic_edt_wrap_smoke():
    """Azimuth/gamma periodic wrap: site near −π reaches cell near +π."""
    from ird_playground.map.signed_distance import (
        EDT_WARNING,
        signed_distance_from_occupancy,
    )

    # Tiny 5-D grid; occupy one voxel at azimuth index 0 (near −π).
    shape = (3, 3, 3, 8, 8)
    occ = np.zeros(shape, dtype=bool)
    occ[1, 1, 1, 0, 1] = True
    step = np.deg2rad(12.0)
    spacings = np.array([0.03, 0.03, step, step, step], dtype=np.float64)
    tilt = np.linspace(0.0, np.pi, shape[2])
    sdf = signed_distance_from_occupancy(
        occ, spacings, periodic_axes=(3, 4), tilt_axis_values=tilt
    )
    # Cell at azimuth index -1 (near +π) is one wrap-step from index 0.
    d_wrap = abs(float(sdf[1, 1, 1, -1, 1]))
    d_far = abs(float(sdf[1, 1, 1, 4, 1]))
    assert d_wrap < d_far
    # One native azimuth step under λ·sin(tilt≈π/2).
    expected = float(spacings[3] * 0.5730)  # λ ≈ 0.573; sin(π/2)=1
    assert d_wrap < 2.5 * expected
    assert "not an SE(3) geodesic" in EDT_WARNING
    assert "far-field" in EDT_WARNING.lower()


def test_edt_positive_inside_negative_outside():
    from ird_playground.map.signed_distance import signed_distance_from_occupancy

    occ = np.zeros((5, 5, 4, 4, 4), dtype=bool)
    occ[1:4, 1:4, 1:3, 1:3, 1:3] = True
    step = np.deg2rad(12.0)
    sp = np.array([0.03, 0.03, step, step, step])
    sdf = signed_distance_from_occupancy(occ, sp)
    assert float(sdf[occ].min()) >= 0.0
    assert float(sdf[~occ].max()) <= 0.0


def test_grid_shape_matches_phase0():
    from ird_playground.map.build_flange_tensor import FlangeOccupancyConfig

    axes = FlangeOccupancyConfig().axis_arrays()
    shape = tuple(len(a) for a in axes)
    assert shape == (58, 41, 16, 31, 31)
