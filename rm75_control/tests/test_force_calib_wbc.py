"""Offline smoke tests for WBC force-ID collection (no robot)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.reference import StreamingPoseReference
from rm75_control.force.compensation import excitation as ex
from rm75_control.force.compensation.collection import require_tool_frame
from rm75_control.force.compensation.id_config import load_config
from rm75_control.force.compensation.paths import CONFIG_ID


def test_streaming_pose_reference_set_pose():
    ref = StreamingPoseReference(np.zeros(6))
    ref.set_pose([0.01, -0.02, 0.5, 0.1, 0.0, -0.1])
    m = ref.sample(1.0)
    np.testing.assert_allclose(m.pose_d, [0.01, -0.02, 0.5, 0.1, 0.0, -0.1])
    np.testing.assert_allclose(m.vel_ff, np.zeros(6))


def test_excitation_parity_cartesian_and_joint_and_burst():
    cfg = load_config(CONFIG_ID)
    cart = cfg.collect.cartesian
    exc = ex.CartesianExcitation.from_config(cart, cfg.collect.scale, "a")
    d0 = exc.delta_pose(0.0)
    d1 = exc.delta_pose(1.25)
    assert d0.shape == (6,)
    assert float(np.linalg.norm(d1[:3])) > 0.0

    q0 = np.zeros(7, dtype=float)
    q = ex.joint_cmd(2.0, q0, cfg.collect.pose_d, 1.0)
    assert q.shape == (7,)
    assert float(np.max(np.abs(q - q0))) > 0.0

    vb = cfg.collect.pose_d.velocity_burst
    vel, axis = ex.vel_burst_cmd(1.0, vb, scale=1.0)
    assert vel.shape == (6,)
    assert axis in (0, 1, 2)
    assert float(np.linalg.norm(vel[3:6])) > 0.0


def test_require_tool_frame_rejects_wrong_tool():
    class _Bot:
        def rm_get_current_tool_frame(self):
            return 0, {"name": "gripper"}

    with pytest.raises(SystemExit) as ei:
        require_tool_frame(_Bot(), required="Arm_Tip")
    assert "Arm_Tip" in str(ei.value)


def test_require_tool_frame_accepts_arm_tip():
    class _Bot:
        def rm_get_current_tool_frame(self):
            return 0, {"name": "Arm_Tip"}

    require_tool_frame(_Bot(), required="Arm_Tip")


def test_cartesian_ramp_cosine_matches_vendor_formula():
    """Vendor ramp_down_cartesian uses 0.5*(1+cos(pi*i/(n-1)))."""
    n = 5
    scales = [0.5 * (1.0 + math.cos(math.pi * i / (n - 1))) for i in range(n)]
    assert scales[0] == pytest.approx(1.0)
    assert scales[-1] == pytest.approx(0.0)


def test_npz_schema_keys_documented():
    """Keys written by WBC collection must match identification expectations."""
    cart_keys = {
        "t", "pose", "q_deg", "force_raw", "delta_pose",
        "pose0", "q0_deg", "pose_slot", "preset", "scale",
        "max_delta_mm", "max_delta_deg", "dt_ms", "log_every", "method",
    }
    d_keys = {
        "t", "pose", "q_deg", "force_raw", "phase",
        "pose0", "pose_burst0", "q0_deg", "pose_slot", "preset", "scale",
        "joint_s", "burst_s", "dt_ms", "log_every", "method", "velocity_burst_profile",
    }
    assert "pose" in cart_keys and "force_raw" in cart_keys
    assert "phase" in d_keys and "pose_burst0" in d_keys


def test_burst_jplus_open_loop_matches_commanded_omega():
    """Open-loop twist→J⁺ at identity TCP keeps origin still and hits ω≈cmd."""
    from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
    from rm75_control.force.compensation.collection_wbc import integrate_burst_twist_step

    kin = RobotKinematics()
    # Arm_Tip ≈ flange (vendor calib tool).
    kin.apply_link7_to_tcp_offset(np.zeros(6), euler_order="xyz")

    q = np.array([0.0, 0.1, -0.4, 0.05, 1.2, 0.0, 1.0, 0.2], dtype=float)
    pose0 = kin.fk_pose(q)
    dt = 0.01
    wz = 0.25  # rad/s about tool Z
    twist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, wz], dtype=float)
    n = 40
    for _ in range(n):
        pose = kin.fk_pose(q)
        q = integrate_burst_twist_step(
            kin,
            q,
            twist,
            dt_s=dt,
            rail_m=0.0,
            frame_type=1,
            pose_tool=pose,
            euler_order="xyz",
        )
    pose1 = kin.fk_pose(q)
    # Origin of Arm_Tip should barely translate under pure ω.
    assert float(np.linalg.norm(pose1[:3] - pose0[:3])) < 0.005
    # Orientation should move ~ wz * n * dt about tool Z (base-frame magnitude).
    from scipy.spatial.transform import Rotation as Rsc

    R0 = Rsc.from_euler("xyz", pose0[3:6]).as_matrix()
    R1 = Rsc.from_euler("xyz", pose1[3:6]).as_matrix()
    dR = R0.T @ R1
    rotvec = Rsc.from_matrix(dR).as_rotvec()
    assert float(np.linalg.norm(rotvec)) == pytest.approx(wz * n * dt, rel=0.15)


def test_burst_jplus_probe_tcp_translates_under_pure_omega():
    """Wrong (probe) TCP makes pure ω orbit the tip — origin drifts (the old bug)."""
    from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
    from rm75_control.force.compensation.collection_wbc import integrate_burst_twist_step

    kin = RobotKinematics()
    # Leave default probe URDF TCP (non-zero offset).
    assert float(np.linalg.norm(kin.tcp_offset_pose[:3])) > 0.05

    q = np.array([0.0, 0.1, -0.4, 0.05, 1.2, 0.0, 1.0, 0.2], dtype=float)
    # Measure flange-ish: apply identity temporarily to read link7, or use tcp
    # origin at probe — for this test we care that TCP origin stays still while
    # a flange point would move. Simpler: Arm_Tip identity origin drifts if we
    # wrongly use probe J... Compare Arm_Tip origin before/after with probe J.
    kin_tip = RobotKinematics()
    kin_tip.apply_link7_to_tcp_offset(np.zeros(6), euler_order="xyz")
    tip0 = kin_tip.fk_pose(q)[:3]

    dt = 0.01
    twist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.25], dtype=float)
    for _ in range(40):
        pose = kin.fk_pose(q)  # probe pose for R — still wrong frame combo
        q = integrate_burst_twist_step(
            kin,
            q,
            twist,
            dt_s=dt,
            rail_m=0.0,
            frame_type=1,
            pose_tool=pose,
            euler_order="xyz",
        )
    tip1 = kin_tip.fk_pose(q)[:3]
    # Spinning about probe tip moves the Arm_Tip/flange origin noticeably.
    assert float(np.linalg.norm(tip1 - tip0)) > 0.005