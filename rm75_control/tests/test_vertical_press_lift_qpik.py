"""Closed-loop vertical press/lift regression for rail-arm reconfiguration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
import uuid

from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"
# Side-lying design family (yaml q_nominal) at the logged rail station.
RISK_POSE = np.deg2rad(
    np.array([0.0, -89.5, -94.5, 65.2, 96.0, 89.3, 61.0, 94.6])
)
RISK_POSE[0] = 0.361050


def _meaningful_reversals(velocity: np.ndarray, threshold: float = 2.0e-3) -> int:
    signs = np.sign(np.asarray(velocity)[np.abs(velocity) > threshold])
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def test_vertical_press_lift_reconfigures_without_rail_hunting(request) -> None:
    cfg = build_joint_ik_config(yaml.safe_load(CONFIG.read_text()))
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    cfg.qp.joint_comfort.enabled = False
    cfg.native_shm_prefix = f"press_lift_{uuid.uuid4().hex}"
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    if inner._native is not None:
        request.addfinalizer(inner._native.shutdown)
    q = RISK_POSE.copy()
    inner.reset(q)
    inner.begin_hybrid_episode(q, np.zeros(8))
    elbow_sign = np.sign(q[4])
    wrist_sign = np.sign(q[6])
    phases: list[list] = [[], []]

    for phase, protected_z in enumerate((-0.020, 0.020)):
        for _ in range(300):
            twist = np.zeros(6)
            twist[2] = protected_z
            step = inner.update(
                twist,
                dt=cfg.dt,
                q_meas=q,
                contact_active=True,
                path_twist=np.zeros(6),
                feedback_twist=np.zeros(6),
                rail_exec_vel_m_s=float(inner.core.qdot_prev[0]),
            )
            q = step.q_send.copy()
            phases[phase].append(step)

    press, lift = phases
    for phase_index, steps in enumerate(phases):
        for tick, step in enumerate(steps):
            assert step.qp_solver_call_count >= 1
            assert not step.task_paused, (phase_index, tick, step.task_pause_reason)
            assert step.fallback_level == "none"
            assert not step.solver_fault_latched
            np.testing.assert_allclose(
                step.v_tcp_estimated, step.v_cmd_feasible, atol=1.0e-5
            )
    assert min(s.task_progress for s in press[40:]) > 0.99
    assert min(s.task_progress for s in lift[40:]) > 0.99

    press_rail = np.array([step.qdot[0] for step in press])
    lift_rail = np.array([step.qdot[0] for step in lift])
    # One settle reverse is not hunting; the minus-policy split may
    # change sign once as Z couples into Y.
    assert _meaningful_reversals(press_rail) <= 1
    assert _meaningful_reversals(lift_rail) <= 1
    # Healthy σ: σ-escape is demoted and must not hunt the carriage.
    assert not any(bool(s.rail_escape_active) for s in press)
    prefs = np.array(
        [s.rail_macro_pref_v for s in press if abs(s.rail_macro_pref_v) > 5.0e-4]
    )
    assert prefs.size == 0 or float(np.max(np.abs(prefs))) < 0.02
    assert all(np.sign(step.q_send[4]) == elbow_sign for step in (*press, *lift))
    assert all(np.sign(step.q_send[6]) == wrist_sign for step in (*press, *lift))
