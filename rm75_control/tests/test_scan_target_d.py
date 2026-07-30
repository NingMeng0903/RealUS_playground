"""Offline tests for scan move->D target resolution (no RealMan TCP)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, deg2rad, full_q_from_arm
from rm75_control.control.joint_admittance_8dof.sin_tool_y_program import (
    load_slot_joints_only,
    resolve_scan_target_at_d,
)

_CFG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"


def _inner_cfg():
    with open(_CFG, "r", encoding="utf-8") as f:
        return build_joint_ik_config(yaml.safe_load(f) or {})


def test_load_slot_joints_only_d():
    q_deg, pose_id, rec = load_slot_joints_only("d")
    assert q_deg.shape == (7,)
    assert pose_id.shape == (6,)
    assert "q_deg" in rec


def test_resolve_scan_target_joints_matches_fk():
    kin = RobotKinematics()
    inner_cfg = _inner_cfg()
    rail_m = float(inner_cfg.rail.q_ref_m)

    target = resolve_scan_target_at_d(
        "d",
        kin,
        d_target="joints",
        rail_m=rail_m,
        qp_cfg=inner_cfg.qp,
        nullspace_cfg=inner_cfg.nullspace,
    )
    assert target.d_target == "joints"
    assert target.q_target_rad.shape == (8,)
    pose_fk = kin.fk_pose(target.q_target_rad)
    np.testing.assert_allclose(target.pose_d, pose_fk, atol=1e-6)


def test_resolve_scan_target_kin_fk_no_robot():
    kin = RobotKinematics()
    inner_cfg = _inner_cfg()
    rail_m = float(inner_cfg.rail.q_ref_m)

    target = resolve_scan_target_at_d(
        "d",
        kin,
        d_target="kin-fk",
        approach_dz_m=0.220,
        use_force_id_pose=False,
        euler_order=inner_cfg.euler_order,
        rail_m=rail_m,
        robot=None,
        qp_cfg=inner_cfg.qp,
        nullspace_cfg=inner_cfg.nullspace,
    )
    assert target.d_target == "kin-fk"
    assert target.pose_d.shape == (6,)
    err_mm = np.linalg.norm(kin.fk_pose(target.q_target_rad)[:3] - target.pose_d[:3]) * 1000.0
    assert err_mm < 5.0
