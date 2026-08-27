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


def test_inconsistent_state_collapses_to_feasible_brake():
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
    inner.set_direct_joint_ptp(False)
    compiled.phase.on_enter()
    assert inner._plan_drives_rail is True
    assert inner._direct_joint_ptp is True
    if compiled.phase.on_exit is not None:
        compiled.phase.on_exit()
    assert inner._plan_drives_rail is False
    assert inner._direct_joint_ptp is False


def test_arrival_dwell_requires_plan_and_continuous_settled_command():
    from rm75_control.control.joint_admittance_8dof.loop import _ArrivalDwellGate

    gate = _ArrivalDwellGate(
        plan_duration_s=1.0,
        dwell_required_s=0.015,
        arm_speed_rad_s=0.02,
        rail_speed_m_s=0.003,
    )
    stopped = np.zeros(8)
    assert not gate.update(
        geometric_arrival=True, t_ref_s=0.99, qdot_applied=stopped, dt_s=0.005
    )
    assert gate.dwell_s == 0.0
    assert not gate.update(
        geometric_arrival=True, t_ref_s=1.0, qdot_applied=stopped, dt_s=0.005
    )
    moving = stopped.copy()
    moving[4] = 0.03
    assert not gate.update(
        geometric_arrival=True, t_ref_s=1.005, qdot_applied=moving, dt_s=0.005
    )
    assert gate.dwell_s == 0.0
    for index in range(3):
        arrived = gate.update(
            geometric_arrival=True,
            t_ref_s=1.01 + 0.005 * index,
            qdot_applied=stopped,
            dt_s=0.005,
        )
    assert arrived


def test_zero_arrival_dwell_still_requires_all_candidate_conditions():
    from rm75_control.control.joint_admittance_8dof.loop import _ArrivalDwellGate

    gate = _ArrivalDwellGate(
        plan_duration_s=1.0,
        dwell_required_s=0.0,
        arm_speed_rad_s=0.02,
        rail_speed_m_s=0.003,
    )
    stopped = np.zeros(8)
    assert not gate.update(
        geometric_arrival=False, t_ref_s=1.0, qdot_applied=stopped, dt_s=0.005
    )
    assert not gate.update(
        geometric_arrival=True, t_ref_s=0.9, qdot_applied=stopped, dt_s=0.005
    )
    moving = stopped.copy()
    moving[1] = 0.03
    assert not gate.update(
        geometric_arrival=True, t_ref_s=1.0, qdot_applied=moving, dt_s=0.005
    )
    assert gate.update(
        geometric_arrival=True, t_ref_s=1.0, qdot_applied=stopped, dt_s=0.005
    )


def test_arrival_dwell_requires_fresh_rail_worker_standstill():
    from types import SimpleNamespace

    from rm75_control.control.joint_admittance_8dof.loop import (
        _ArrivalDwellGate,
        _rail_settled_for_arrival,
    )

    bridge = SimpleNamespace(
        enabled=True,
        servo_sample=SimpleNamespace(
            sample_mono_s=10.0,
            v_cmd_m_s=0.004,
            v_meas_m_s=0.001,
        ),
    )
    gate = _ArrivalDwellGate(None, 0.01, 0.02, 0.003)
    settled = _rail_settled_for_arrival(
        bridge, speed_limit_m_s=0.003, now_s=10.01, freshness_s=0.05
    )
    assert settled is False
    assert not gate.update(
        geometric_arrival=True,
        t_ref_s=1.0,
        qdot_applied=np.zeros(8),
        dt_s=0.005,
        rail_settled=settled,
    )
    bridge.servo_sample = SimpleNamespace(
        sample_mono_s=9.0,
        v_cmd_m_s=0.0,
        v_meas_m_s=0.0,
    )
    assert not _rail_settled_for_arrival(
        bridge, speed_limit_m_s=0.003, now_s=10.01, freshness_s=0.05
    )


def test_public_move_factory_has_arrival_timeout():
    from rm75_control.control.joint_admittance_8dof.api import phase_cartesian_goto
    from rm75_control.control.joint_admittance_8dof.reference import (
        JointSmoothMoveReference,
    )

    kin = RobotKinematics()
    q0 = Q_HOME.copy()
    ref = JointSmoothMoveReference(kin, q0, q0, duration_s=2.0)
    spec = phase_cartesian_goto(
        ref,
        pose_target=kin.fk_pose(q0),
        q_target_rad=q0,
        max_duration_s=None,
    )
    assert spec.max_duration_s == pytest.approx(20.0)


def test_compiled_move_and_rail_arrival_use_explicit_plan_duration():
    from rm75_control.control.joint_admittance_8dof.api import (
        CompileContext,
        compile_phase,
        phase_rail_reposition,
    )
    from rm75_control.control.joint_admittance_8dof.loop import (
        JointIkConfig,
        JointIkController,
    )

    kin = RobotKinematics()
    inner = JointIkController(kin, JointIkConfig())
    ctx = CompileContext(kin=kin, inner=inner, control_frame="base")
    q0 = Q_HOME.copy()
    qt = q0.copy()
    qt[2] += np.deg2rad(5.0)
    move = WbcArm.make_movej_phase(kin, q0, qt, duration_s=1.25)
    compiled_move = compile_phase(move, ctx)
    assert compiled_move.phase.arrival_plan_duration_s == pytest.approx(1.25)
    assert compiled_move.phase.arrival_dwell_s == pytest.approx(0.10)

    rail = phase_rail_reposition(0.42, q0, kin, duration_s=0.75)
    compiled_rail = compile_phase(rail, ctx)
    assert compiled_rail.phase.arrival_plan_duration_s == pytest.approx(0.75)
    assert compiled_rail.phase.max_duration_s > 0.75


def test_cartesian_track_sample_uses_local_error_saturation():
    from rm75_control.control.joint_admittance_8dof.loop import (
        CartesianTrackConfig,
        CartesianTrackOuterLoop,
    )
    from rm75_control.control.joint_admittance_8dof.reference import HoldReference

    pose = np.array([0.2, -0.1, 0.4, 0.1, -0.2, 0.3])
    outer = CartesianTrackOuterLoop(
        HoldReference(),
        CartesianTrackConfig(control_frame="base", k_task=np.ones(6)),
    )
    outer.set_origin(pose)
    current = pose.copy()
    current[0] += 0.2
    command = outer.sample(0.0, current, np.zeros(6))
    assert np.all(np.isfinite(command))
    assert np.linalg.norm(outer.last_feedback_twist[:3]) <= 0.05 + 1.0e-12



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
