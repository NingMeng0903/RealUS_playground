"""Pre-execution IRD gate + RegionA rail-goodness adapter scaffolding."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.ird_precheck import (
    PrecheckConfig,
    assert_trajectory_precheck,
    calibrated_ird_clearance,
    try_import_ird,
    validate_tcp_rail_trajectory,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.tasks.rail_goodness import (
    RegionARailGoodness,
    SigmaMinGoodness,
)


def test_calibrated_clearance_hook():
    assert calibrated_ird_clearance(0.05, 0.01) == pytest.approx(0.04)


def test_fail_closed_without_clearance_fn():
    kin = RobotKinematics()
    poses = np.zeros((2, 6), dtype=float)
    poses[:, 0] = 0.4
    poses[:, 2] = 0.35
    rails = np.array([0.0, 0.0])
    result = validate_tcp_rail_trajectory(
        poses,
        rails,
        kin=kin,
        clearance_fn=None,
        cfg=PrecheckConfig(fail_closed_without_ird=True, require_srs=False),
    )
    assert result.ok is False
    assert result.ird_available is False
    assert "fail-closed" in result.message
    with pytest.raises(RuntimeError, match="pre-execution"):
        assert_trajectory_precheck(result)


def test_precheck_with_mock_clearance_and_srs_seed():
    kin = RobotKinematics()
    # Use a reachable home-ish configuration's FK as the TCP target.
    q0 = np.array([0.0, 0.2, 0.4, 0.1, 1.2, 0.0, 0.5, 0.0], dtype=float)
    pose = kin.fk_pose(q0)
    poses = np.stack([pose, pose], axis=0)
    rails = np.array([0.0, 0.0])

    def clearance_fn(_pose, _rail):
        return 0.05

    result = validate_tcp_rail_trajectory(
        poses,
        rails,
        kin=kin,
        clearance_fn=clearance_fn,
        q_seed_arm=q0[1:],
        cfg=PrecheckConfig(
            fail_closed_without_ird=True,
            require_srs=True,
            clearance_margin=0.0,
            conformal_threshold=0.01,
            collision_min_distance_m=0.0,
            branch_id=None,
        ),
    )
    assert result.ird_available is True
    assert len(result.waypoints) == 2
    for wp in result.waypoints:
        assert wp.ird_clearance == pytest.approx(0.04)
        assert np.isfinite(wp.psi_rad)
        assert 0 <= wp.branch_id <= 7
        assert wp.collision_min_distance_m < float("inf")
    # May or may not be fully ok depending on SRS/collision at this seed; the
    # important contract is structured per-waypoint output without crashing.
    assert isinstance(result.ok, bool)


def test_margin_failure_recorded():
    kin = RobotKinematics()
    poses = np.zeros((1, 6), dtype=float)
    poses[0, 0] = 0.35
    poses[0, 2] = 0.4
    rails = np.array([0.0])

    def clearance_fn(_pose, _rail):
        return 0.001

    result = validate_tcp_rail_trajectory(
        poses,
        rails,
        kin=kin,
        clearance_fn=clearance_fn,
        cfg=PrecheckConfig(
            fail_closed_without_ird=True,
            require_srs=False,
            clearance_margin=0.02,
            conformal_threshold=0.0,
            collision_min_distance_m=-1.0,  # disable collision gate
        ),
    )
    assert result.ok is False
    assert any("ird_clearance" in r for r in result.waypoints[0].reasons)


def test_sigma_min_goodness_still_default():
    kin = RobotKinematics()
    g = SigmaMinGoodness(kin)
    q = np.array([0.0, 0.2, 0.4, 0.1, 1.2, 0.0, 0.5, 0.0])
    assert g.g(q) > 0.0
    assert np.isfinite(g.dg_dy_rail(q))


def test_region_a_adapter_optional_import():
    available, _ = try_import_ird()
    kin = RobotKinematics()
    if not available:
        with pytest.raises(ImportError, match="ird_playground"):
            RegionARailGoodness(kin, field=object())
        return

    class _ConstField:
        def score_world(self, tcp, axis):
            import torch

            return tcp.new_ones(tcp.shape[:-2])

    adapter = RegionARailGoodness(kin, _ConstField(), device="cpu")
    q = np.array([0.0, 0.2, 0.4, 0.1, 1.2, 0.0, 0.5, 0.0])
    val = adapter.g(q)
    assert np.isfinite(val)
    dg = adapter.dg_dy_rail(q)
    assert np.isfinite(dg)
