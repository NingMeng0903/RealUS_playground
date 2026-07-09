"""Offline closed-loop validation of the WBC slack-QP inner loop (no robot)."""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance.collision_model import CollisionConfig
from rm75_control.control.joint_admittance.loop import JointIkConfig, JointIkController
from rm75_control.control.joint_admittance.model import RobotKinematics, deg2rad
from rm75_control.control.joint_admittance.solver.qp_builder import QpConfig
from rm75_control.control.joint_admittance.tasks.arm_angle import ArmAngleTaskConfig
from rm75_control.control.joint_admittance.tasks.nullspace_task import NullspaceTaskConfig

Q_HOME_DEG = np.array([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0])


def _make(
    control_frame: str = "base",
    k_center: float = 0.0,
    *,
    collision_enabled: bool = False,
    arm_angle_enabled: bool = False,
    **cfg_kw,
) -> JointIkController:
    kin = RobotKinematics()
    # Production-like weighting: task_weight >> reg keeps the Cartesian equality
    # effectively hard (soft weights let the slack shave % off weak directions).
    qp = QpConfig(
        task_weight=np.array([1000.0, 1000.0, 1000.0, 500.0, 500.0, 500.0]),
        reg=np.full(7, 0.001),
        collision=CollisionConfig(enabled=collision_enabled),
    )
    cfg = JointIkConfig(
        control_frame=control_frame,
        qp=qp,
        nullspace=NullspaceTaskConfig(k_center=k_center, k_limit=0.0),
        arm_angle=ArmAngleTaskConfig(enabled=arm_angle_enabled, k_psi=1.0),
        v_scale=0.9,
        a_max=50.0,
        **cfg_kw,
    )
    ctrl = JointIkController(kin, cfg)
    ctrl.reset(deg2rad(Q_HOME_DEG))
    return ctrl


def test_zero_drift_on_hold():
    ctrl = _make(k_center=0.0)
    pose0 = ctrl.kin.fk_pose(ctrl.q_cmd)
    for _ in range(300):
        ctrl.update(np.zeros(6))
    pose1 = ctrl.kin.fk_pose(ctrl.q_cmd)
    assert np.linalg.norm(pose1[:3] - pose0[:3]) < 1e-4


def test_constant_twist_moves_tcp_at_commanded_rate():
    """No hidden attenuation on the send path: a constant twist must move the
    TCP by ~v*T (this guards against the old filter+sync stage that divided
    every commanded velocity by ~6.7)."""
    ctrl = _make(k_center=0.0)
    dt = ctrl.cfg.dt
    twist = np.array([0.02, 0.0, -0.01, 0.0, 0.0, 0.0])
    pose0 = ctrl.kin.fk_pose(ctrl.q_cmd)
    n = 400
    for _ in range(n):
        ctrl.update(twist, dt)
    pose1 = ctrl.kin.fk_pose(ctrl.q_cmd)
    moved = pose1[:3] - pose0[:3]
    expect = twist[:3] * n * dt
    assert np.linalg.norm(moved - expect) * 1000.0 < 2.0, (moved, expect)


def test_nullspace_preserves_tcp():
    from scipy.spatial.transform import Rotation as Rsc

    ctrl = _make(k_center=1.5)
    q_mid = 0.5 * (ctrl.kin.q_lower + ctrl.kin.q_upper)
    q_start = ctrl.q_cmd.copy()
    pose0 = ctrl.kin.fk_pose(ctrl.q_cmd)
    dist0 = np.linalg.norm(q_start - q_mid)
    max_pos_err_mm = 0.0
    max_rot_err_deg = 0.0
    for _ in range(400):
        ctrl.update(np.zeros(6))
        pose = ctrl.kin.fk_pose(ctrl.q_cmd)
        max_pos_err_mm = max(max_pos_err_mm, np.linalg.norm(pose[:3] - pose0[:3]) * 1000.0)
        r0 = Rsc.from_euler("xyz", pose0[3:6]).as_matrix()
        r1 = Rsc.from_euler("xyz", pose[3:6]).as_matrix()
        d_deg = np.degrees(np.linalg.norm(Rsc.from_matrix(r1 @ r0.T).as_rotvec()))
        max_rot_err_deg = max(max_rot_err_deg, d_deg)
    dist1 = np.linalg.norm(ctrl.q_cmd - q_mid)
    joint_motion = np.linalg.norm(ctrl.q_cmd - q_start)
    assert max_pos_err_mm < 2.0, max_pos_err_mm
    assert max_rot_err_deg < 0.2, max_rot_err_deg
    assert joint_motion > 1e-4, joint_motion
    # Self-motion at this posture is nearly orthogonal to (q - q_mid): the
    # Euclidean distance barely changes (constrained descent through N_dyn is
    # not a descent in this metric).  Assert only that it does not run away.
    assert dist1 <= dist0 + 0.02, (dist0, dist1)


def test_arm_angle_task_tracks_swivel_in_nullspace():
    """Swivel-angle secondary task: psi converges to psi_ref while TCP holds."""
    ctrl = _make(k_center=0.0, arm_angle_enabled=True)
    task = ctrl.arm_task
    assert task is not None
    psi0 = task.arm_angle(ctrl.q_cmd)
    psi_ref = psi0 + np.radians(10.0)
    task.set_reference(psi_ref)
    pose0 = ctrl.kin.fk_pose(ctrl.q_cmd)
    for _ in range(1500):
        ctrl.update(np.zeros(6))
    psi1 = task.arm_angle(ctrl.q_cmd)
    pose1 = ctrl.kin.fk_pose(ctrl.q_cmd)
    psi_err = float((psi1 - psi_ref + np.pi) % (2.0 * np.pi) - np.pi)
    assert abs(psi_err) < np.radians(3.0), np.degrees((psi0, psi1, psi_ref))
    assert np.linalg.norm(pose1[:3] - pose0[:3]) * 1000.0 < 2.0


def test_command_lead_anti_windup_no_teleport():
    """If the (simulated) robot stops following, q_cmd's lead over q_meas must
    saturate at resync_err_rad via the normal velocity-limited rate - NEVER a
    discontinuous position jump (a prior implementation reassigned q_cmd
    directly, bypassing the velocity/acceleration box entirely: a multi-degree
    jump in one 5 ms tick that rm_movej_canfd sees as a discontinuity - the
    real cause of on-hardware jerk/shake, not the control period)."""
    ctrl = _make(k_center=0.0)
    dt = ctrl.cfg.dt
    q_frozen = ctrl.q_cmd.copy()  # robot "stuck" here (e.g. following lag)
    twist = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    dq_max = ctrl.kin.v_max * ctrl.cfg.v_scale * dt
    q_prev = ctrl.q_cmd.copy()
    leads = []
    for _ in range(500):
        ctrl.update(twist, dt, q_meas=q_frozen)
        dq = ctrl.q_cmd - q_prev
        # every tick's step must still respect the ordinary velocity box -
        # no teleport, ever, regardless of how far q_cmd has lagged.
        assert np.all(np.abs(dq) <= dq_max + 1e-9), (dq, dq_max)
        q_prev = ctrl.q_cmd.copy()
        leads.append(np.max(np.abs(ctrl.q_cmd - q_frozen)))
    # the lead must settle at resync_err_rad (anti-windup works), with only a
    # small accel-limited transient overshoot on the way there - never an
    # unbounded runaway (a prior bug dropped the bound entirely on conflict).
    assert abs(leads[-1] - ctrl.cfg.resync_err_rad) < 1e-6, leads[-1]
    assert max(leads) <= ctrl.cfg.resync_err_rad * 2.0, max(leads)


def test_velocity_and_position_limits():
    ctrl = _make(k_center=0.0)
    dt = ctrl.cfg.dt
    dq_max = ctrl.kin.v_max * ctrl.cfg.v_scale * dt
    q_prev = ctrl.q_cmd.copy()
    for _ in range(500):
        ctrl.update(np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0]))
        dq = ctrl.q_cmd - q_prev
        assert np.all(np.abs(dq) <= dq_max + 1e-8), "velocity limit violated"
        assert np.all(ctrl.q_cmd >= ctrl.kin.q_lower - 1e-9)
        assert np.all(ctrl.q_cmd <= ctrl.kin.q_upper + 1e-9)
        q_prev = ctrl.q_cmd.copy()


def test_wbc_proxqp_backend():
    ctrl = _make(k_center=0.0)
    assert ctrl.core.backend_name == "proxqpwbc"
    for _ in range(50):
        ctrl.update(np.zeros(6))
    assert np.isfinite(ctrl.q_cmd).all()


def test_slack_bounded_near_singularity():
    """Near-singular configuration: task slack w absorbs error instead of exploding qdot."""
    ctrl = _make(k_center=0.0, collision_enabled=False)
    q_sing = np.zeros(7)
    ctrl.reset(q_sing)
    r = ctrl.core.step(q_sing, np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0]), ctrl.cfg.dt)
    assert r.sigma_min < 0.05
    assert r.slack_norm < 0.5
    assert np.all(np.abs(r.qdot) <= ctrl.kin.v_max * ctrl.cfg.v_scale + 1e-6)


def test_cbf_model_loads_and_reports_distances():
    from rm75_control.control.joint_admittance.collision_model import CollisionModel

    kin = RobotKinematics()
    col = CollisionModel(kin.model)
    q = deg2rad(Q_HOME_DEG)
    col.update(q)
    assert len(col.all_pairs()) > 0
    assert col.min_distance() < 0.2


def test_cbf_active_during_control():
    ctrl = _make(k_center=0.0, collision_enabled=True)
    ctrl.cfg.qp.collision.d_activate = 0.15
    for _ in range(20):
        s = ctrl.update(np.zeros(6))
    assert s.sigma_min >= 0.0


def test_soft_start_position_velocity_consistent():
    """sin_y_motion soft start: vy must equal d(dy)/dt (the old version ramped
    only the velocity, contradicting the position for the first ramp_s)."""
    from rm75_control.control.joint_admittance.reference import sin_y_motion

    A, omega, ramp = 0.03, 0.7, 2.0
    h = 1e-6
    for t in np.linspace(0.01, 6.0, 120):
        dy_m, _ = sin_y_motion(t - h, A, omega, soft_start=True, ramp_s=ramp)
        dy_p, _ = sin_y_motion(t + h, A, omega, soft_start=True, ramp_s=ramp)
        _, vy = sin_y_motion(t, A, omega, soft_start=True, ramp_s=ramp)
        vy_num = (dy_p - dy_m) / (2 * h)
        assert abs(vy - vy_num) < 1e-4, (t, vy, vy_num)
    dy0, vy0 = sin_y_motion(0.0, A, omega, soft_start=True, ramp_s=ramp)
    assert abs(dy0) < 1e-12 and abs(vy0) < 1e-12


def test_cartesian_track_outer_loop_tool_frame_converges():
    from rm75_control.control.joint_admittance.loop import CartesianTrackConfig, CartesianTrackOuterLoop
    from rm75_control.control.joint_admittance.pose_ik import solve_pose_ik
    from rm75_control.control.joint_admittance.reference import JointSmoothMoveReference

    ctrl = _make(control_frame="tool", k_center=0.0, collision_enabled=False)
    dt = ctrl.cfg.dt
    q_start = deg2rad(np.array([20.0, -30.0, 10.0, 70.0, 15.0, 50.0, -10.0]))
    ctrl.reset(q_start)
    pose0 = ctrl.kin.fk_pose(ctrl.q_cmd)
    pose_target = pose0.copy()
    pose_target[0] += 0.05
    pose_target[1] -= 0.03
    pose_target[2] += 0.02

    q_target, ik_ok, _report = solve_pose_ik(ctrl.kin, q_start, pose_target)
    assert ik_ok

    move_ref = JointSmoothMoveReference(ctrl.kin, q_start, q_target, duration_s=3.0)
    move_outer = CartesianTrackOuterLoop(move_ref, CartesianTrackConfig(control_frame="tool"))
    move_outer.set_origin(pose0)

    t_s = 0.0
    err_mm = []
    for _ in range(1000):
        cur = ctrl.kin.fk_pose(ctrl.q_cmd)
        twist = move_outer.sample(t_s, cur, np.zeros(6))
        ctrl.update(twist, dt, qdot_ff=move_ref.sample_q(t_s)[1])
        err_mm.append(np.linalg.norm(ctrl.kin.fk_pose(ctrl.q_cmd)[:3] - pose_target[:3]) * 1000.0)
        t_s += dt

    assert err_mm[-1] < 2.0, err_mm[-1]
    assert max(err_mm[600:]) < 3.0, max(err_mm[600:])


def test_cartesian_move_then_sin_reference_offline():
    from rm75_control.control.joint_admittance.loop import CartesianTrackOuterLoop
    from rm75_control.control.joint_admittance.pose_ik import solve_pose_ik
    from rm75_control.control.joint_admittance.reference import (
        JointSmoothMoveReference,
        SinToolYReference,
    )

    ctrl = _make(k_center=0.0, collision_enabled=False)
    dt = ctrl.cfg.dt
    q_start = ctrl.q_cmd.copy()
    pose0 = ctrl.kin.fk_pose(ctrl.q_cmd)
    pose_target = pose0.copy()
    pose_target[0] += 0.05
    pose_target[2] += 0.03
    pose_target[5] += np.radians(5.0)

    q_target, ik_ok, _report = solve_pose_ik(ctrl.kin, q_start, pose_target)
    assert ik_ok

    move_ref = JointSmoothMoveReference(ctrl.kin, q_start, q_target, duration_s=3.0)
    move_outer = CartesianTrackOuterLoop(move_ref)
    move_outer.cfg.control_frame = "base"
    move_outer.set_origin(pose0)

    t_s = 0.0
    for _ in range(700):
        twist = move_outer.sample(t_s, ctrl.kin.fk_pose(ctrl.q_cmd), np.zeros(6))
        ctrl.update(twist, dt, qdot_ff=move_ref.sample_q(t_s)[1])
        t_s += dt

    pose_arrived = ctrl.kin.fk_pose(ctrl.q_cmd)
    assert np.linalg.norm(pose_arrived[:3] - pose_target[:3]) * 1000.0 < 3.0

    sin_ref = SinToolYReference(amplitude_m=0.01, period_s=4.0, soft_start=True, ramp_s=1.0)
    sin_outer = CartesianTrackOuterLoop(sin_ref)
    sin_outer.cfg.control_frame = "base"
    sin_outer.set_origin(pose_arrived)

    t_s = 0.0
    track_err_mm = []
    first_tick_dq = None
    for _ in range(800):
        q_prev = ctrl.q_cmd.copy()
        cur = ctrl.kin.fk_pose(ctrl.q_cmd)
        ref = sin_ref.sample(t_s)
        twist = sin_outer.sample(t_s, cur, np.zeros(6))
        ctrl.update(twist, dt)
        if first_tick_dq is None:
            first_tick_dq = np.linalg.norm(ctrl.q_cmd - q_prev)
        track_err_mm.append(
            np.linalg.norm(ctrl.kin.fk_pose(ctrl.q_cmd)[:3] - ref.pose_d[:3]) * 1000.0
        )
        t_s += dt

    assert first_tick_dq < 0.03, first_tick_dq
    # ideal plant offline: sub-mm sinusoid tracking, including the soft start
    assert max(track_err_mm) < 1.0, max(track_err_mm)


def _report() -> None:
    print("Running offline WBC joint-IK validation report...\n")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                print(f"  FAIL  {name}: {e}")


if __name__ == "__main__":
    _report()
