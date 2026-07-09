"""P0 move-guard verification (offline, no robot)."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance.loop import Phase, _reference_governor_scale
from rm75_control.control.joint_admittance.model import (
    RobotKinematics,
    auto_move_duration_s,
    deg2rad,
    max_joint_err_deg,
    pose_distance,
    wrap_joint_delta,
)
from rm75_control.control.joint_admittance.reference import JointSmoothMoveReference


class _DummyOuter:
    pass


def _gov_joint_max_deg(max_dq_deg: float) -> float:
    return float(np.clip(0.40 * max_dq_deg, 20.0, 90.0))


def test_gov_joint_max_deg_clip():
    assert _gov_joint_max_deg(40.0) == 20.0
    assert _gov_joint_max_deg(77.8) == pytest.approx(31.12)
    assert _gov_joint_max_deg(400.0) == 90.0


def test_wrap_joint_delta_shortest_path():
    q0 = deg2rad(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    q1 = deg2rad(np.array([170.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    dq = wrap_joint_delta(q0, q1)
    assert float(np.rad2deg(dq[0])) == pytest.approx(170.0)


def test_joint_smooth_move_uses_wrap_delta():
    kin = RobotKinematics()
    q0 = deg2rad(np.zeros(7))
    q1 = deg2rad(np.array([170.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    ref = JointSmoothMoveReference(kin, q0, q1, duration_s=2.0)
    q_end, _ = ref.sample_q(2.0)
    assert max_joint_err_deg(q_end, q1) < 0.01


def test_move_arrived_requires_pose_and_joint():
    kin = RobotKinematics()
    euler = "xyz"
    q_tgt = deg2rad(np.array([10.0, -40.0, 15.0, 85.0, -10.0, 50.0, 5.0]))
    pose_d = kin.fk_pose(q_tgt)
    q_near = q_tgt + deg2rad(np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    pose_near = kin.fk_pose(q_near)

    def move_arrived(pose_meas: np.ndarray, q_meas: np.ndarray) -> bool:
        d_mm, d_deg = pose_distance(pose_meas, pose_d, euler)
        if d_mm > 3.0 or d_deg > 1.5:
            return False
        return max_joint_err_deg(q_meas, q_tgt) <= 3.0

    assert move_arrived(pose_d, q_tgt)
    assert not move_arrived(pose_near, q_near)


def test_require_arrival_phase_flag():
    phase = Phase(outer=_DummyOuter(), label="move", require_arrival=True)
    assert phase.require_arrival is True


def test_auto_move_duration_scales_with_joint_travel():
    kin = RobotKinematics()
    q0 = deg2rad(np.zeros(7))
    q1 = deg2rad(np.array([80.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    pose_d = kin.fk_pose(q1)
    t_short, _ = auto_move_duration_s(
        kin, q0, q1, pose_d, v_scale=0.5, v_max_rad_s=kin.v_max, duration_min_s=2.5
    )
    q2 = deg2rad(np.array([160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    pose_d2 = kin.fk_pose(q2)
    t_long, meta = auto_move_duration_s(
        kin, q0, q2, pose_d2, v_scale=0.5, v_max_rad_s=kin.v_max, duration_min_s=2.5
    )
    assert t_long > t_short
    assert meta["joint_headroom"] > 1.0
    assert meta["joint_headroom"] <= 1.5


def test_auto_move_duration_capped_and_tcp_standoff():
    kin = RobotKinematics()
    q0 = deg2rad(np.array([-166.0, -30.0, 80.0, 5.0, -90.0, 60.0, 0.0]))
    q1 = deg2rad([4.99, -23.07, -3.95, 77.84, 2.45, 65.54, 14.41])
    pose_d = kin.fk_pose(q1)
    t, meta = auto_move_duration_s(
        kin,
        q0,
        q1,
        pose_d,
        v_scale=0.5,
        v_max_rad_s=kin.v_max,
        approach_dz_m=0.22,
        duration_max_s=5.0,
    )
    assert t <= 5.0
    assert meta["tcp_mm"] <= 0.22 * 1000.0 * 1.15 + 1.0


def test_joint_governor_only_when_cart_max_zero():
    phase = Phase(
        outer=_DummyOuter(),
        governor_err_max_mm=0.0,
        governor_joint_err_ok_deg=3.0,
        governor_joint_err_max_deg=31.12,
    )
    scale = _reference_governor_scale(phase, outer_err_mm=681.0, joint_err_deg=8.0)
    joint = (31.12 - 8.0) / (31.12 - 3.0)
    assert scale == pytest.approx(joint, abs=1e-6)
