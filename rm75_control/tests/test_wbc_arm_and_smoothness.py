"""Unit tests for Part A numerical fixes + WbcArm industrial facade."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionPairInfo
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, full_q_from_arm
from rm75_control.control.joint_admittance_8dof.reference import SrsSmoothMoveReference
from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import CbfSlotTracker
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    VelocityBoxConstraints,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits
from rm75_control.control.joint_admittance_8dof.wbc_arm import (
    ERR_PARAM,
    ERR_SEND,
    ERR_TIMEOUT,
    OK,
    WbcArm,
)

Q_HOME = full_q_from_arm(
    np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), 0.4
)


def _pair(ga: int, gb: int, dist: float) -> CollisionPairInfo:
    return CollisionPairInfo(
        pair_index=0,
        geom_a=ga,
        geom_b=gb,
        name_a=f"a{ga}",
        name_b=f"b{gb}",
        distance=dist,
        normal=np.array([1.0, 0.0, 0.0]),
        point_a=np.zeros(3),
        point_b=np.zeros(3),
    )


def test_accel_box_projects_when_infeasible():
    kin = RobotKinematics()
    limits = SafetyLimits.from_kinematics(kin, v_scale=0.5, a_max=1.0)
    box = VelocityBoxConstraints(limits, damper_band_rad=0.0)
    q = np.zeros(kin.nv)
    # Huge previous velocity so accel band cannot intersect position-safe box.
    qdot_prev = np.full(kin.nv, 50.0)
    lo, hi = box.bounds(q, dt=0.005, qdot_prev=qdot_prev)
    assert np.all(lo <= hi)
    assert np.all(np.isfinite(lo))
    assert np.all(np.isfinite(hi))


def test_cbf_slot_sticky_assignment():
    tracker = CbfSlotTracker(max_pairs=2, hyst_m=0.01)
    d_act = 0.08
    p01 = _pair(0, 1, 0.05)
    p23 = _pair(2, 3, 0.04)
    slots = tracker.update([p01, p23], d_act)
    assert slots[0] is not None and slots[1] is not None
    # Closer pair fills slot 0 first.
    assert (slots[0].geom_a, slots[0].geom_b) == (2, 3)
    assert (slots[1].geom_a, slots[1].geom_b) == (0, 1)

    # Rank order flips — same keys must keep their slots.
    slots2 = tracker.update([p01, p23], d_act)
    assert (slots2[0].geom_a, slots2[0].geom_b) == (2, 3)
    assert (slots2[1].geom_a, slots2[1].geom_b) == (0, 1)

    # Leave activate band but stay in hysteresis → still held in slot 1.
    p01_far = _pair(0, 1, d_act + 0.005)
    slots3 = tracker.update([p01_far, p23], d_act)
    assert slots3[1] is not None
    assert (slots3[1].geom_a, slots3[1].geom_b) == (0, 1)

    # Beyond hysteresis → slot 1 freed.
    p01_gone = _pair(0, 1, d_act + 0.02)
    slots4 = tracker.update([p01_gone, p23], d_act)
    assert slots4[1] is None
    assert slots4[0] is not None
    assert (slots4[0].geom_a, slots4[0].geom_b) == (2, 3)


def test_srs_psi_shortest_arc():
    kin = RobotKinematics()
    q0 = Q_HOME.copy()
    pose = kin.fk_pose(q0)
    # Force a long-way linear delta if not unwrapped: start≈+170°, target≈−170°.
    ref = SrsSmoothMoveReference(
        kin,
        q0,
        pose,
        y_rail_target_m=float(q0[0]),
        psi_target_rad=np.deg2rad(-170.0),
        duration_s=2.0,
    )
    ref.psi_start = np.deg2rad(170.0)
    ref.psi_target = np.deg2rad(-170.0)
    ref.psi_delta = float(
        (ref.psi_target - ref.psi_start + np.pi) % (2.0 * np.pi) - np.pi
    )
    assert abs(ref.psi_delta) < np.deg2rad(30.0)
    mid = ref.sample_psi(1.0)
    # Midpoint of shortest arc near ±180°, not near 0°.
    assert abs(abs(mid) - np.pi) < np.deg2rad(20.0)


def test_joint_move_phase_enables_plan_drives_rail():
    """MoveJ compile must pin rail to the joint plan (not pose_attract fight)."""
    from rm75_control.control.joint_admittance_8dof.api import (
        CompileContext,
        compile_phase,
    )
    from rm75_control.control.joint_admittance_8dof.loop import JointIkController, JointIkConfig
    from rm75_control.control.joint_admittance_8dof.wbc_arm import WbcArm

    kin = RobotKinematics()
    q0 = Q_HOME.copy()
    qt = q0.copy()
    qt[0] = 0.4
    qt[1] += np.deg2rad(30.0)
    spec = WbcArm.make_movej_phase(kin, q0, qt, duration_s=3.0)
    inner = JointIkController(kin, JointIkConfig())
    ctx = CompileContext(
        kin=kin,
        inner=inner,
        euler_order="xyz",
        control_frame="base",
        v_scale=0.5,
    )
    compiled = compile_phase(spec, ctx)
    assert compiled.phase.on_enter is not None
    inner.set_plan_drives_rail(False)
    inner.set_rail_extension_active(True)
    compiled.phase.on_enter()
    assert inner._plan_drives_rail is True
    assert inner._rail_ext_active is False
    if compiled.phase.on_exit is not None:
        compiled.phase.on_exit()
    assert inner._plan_drives_rail is False



def test_wbc_arm_bad_joint_returns_param_error():
    arm = WbcArm(phase_client=MagicMock())
    assert arm.movej([1.0, 2.0], v=10, r=0, connect=0, block=0) == ERR_PARAM


def test_wbc_arm_connect_failure():
    client = MagicMock()
    client.wait_for_hub.side_effect = TimeoutError("no hub")
    arm = WbcArm(phase_client=client)
    assert arm.connect(timeout_s=0.01) == ERR_SEND


def test_wbc_arm_block_timeout():
    from rm75_control.control.admittance_common.phase_ipc import PhaseStatus

    client = MagicMock()
    client.wait_for_hub.return_value = None
    client.start.return_value = 7
    client.read_status.return_value = {
        "status": PhaseStatus.RUNNING,
        "status_seq": 7,
        "msg": "running",
    }
    arm = WbcArm(phase_client=client, default_timeout_s=0.15)
    q_list = [400.0, 5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]
    tag = arm.movej(q_list, v=20, r=0, connect=0, block=1, timeout_s=0.15)
    assert tag == ERR_TIMEOUT
    client.stop.assert_called()


def test_wbc_arm_nonblock_ok_after_start():
    client = MagicMock()
    client.wait_for_hub.return_value = None
    client.start.return_value = 1
    arm = WbcArm(phase_client=client)
    q_list = [400.0, 5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]
    assert arm.movej(q_list, v=20, r=0, connect=0, block=0) == OK


def test_algo_fk_roundtrip_near_home():
    arm = WbcArm(phase_client=MagicMock())
    q_list = [400.0, 5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]
    code, pose = arm.algo_fk(q_list)
    assert code == OK
    assert len(pose) == 6
    assert np.isfinite(pose).all()
    # FK(q) should match kin.fk_pose
    q = arm._joint_list_to_rad(q_list)
    assert np.allclose(pose, arm.kin.fk_pose(q), atol=1e-9)

