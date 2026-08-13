"""Closed-loop vertical press/lift regression for rail-arm reconfiguration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"
RISK_POSE = np.array(
    [
        0.361050,
        0.150901,
        -0.455461,
        -0.109834,
        2.175169,
        -0.167517,
        0.410309,
        1.714856,
    ]
)


def _meaningful_reversals(velocity: np.ndarray, threshold: float = 2.0e-3) -> int:
    signs = np.sign(np.asarray(velocity)[np.abs(velocity) > threshold])
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def test_vertical_press_lift_reconfigures_without_rail_hunting() -> None:
    cfg = build_joint_ik_config(yaml.safe_load(CONFIG.read_text()))
    cfg.collision.enabled = False
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
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
            )
            q = step.q_send.copy()
            phases[phase].append(step)

    press, lift = phases
    for step in (*press, *lift):
        assert step.qp_solver_call_count == 1
        assert step.fallback_level == "none"
        assert step.qpik_hard_residual_max <= cfg.generic_qpik.solver.feasibility_tolerance
        assert step.rail_decomposition_error <= 1.0e-6
        np.testing.assert_allclose(
            step.rail_xy_contribution + step.arm_xy_contribution,
            step.scan_achieved,
            atol=1.0e-6,
        )

    press_rail = np.array([step.qdot[0] for step in press])
    lift_rail = np.array([step.qdot[0] for step in lift])
    assert _meaningful_reversals(press_rail) == 0
    assert _meaningful_reversals(lift_rail) <= 1
    assert max(abs(press_rail)) > 0.02
    assert min(step.arm_health for step in lift[-50:]) > press[0].arm_health
    assert all(np.sign(step.q_send[4]) == elbow_sign for step in (*press, *lift))
    assert all(np.sign(step.q_send[6]) == wrist_sign for step in (*press, *lift))

    active_cosines = np.array(
        [
            step.risk_direction_cosine
            for step in press
            if step.arm_risk_pref_norm > 2.0e-3
            and np.isfinite(step.risk_direction_cosine)
        ]
    )
    assert active_cosines.size > 0
    assert np.min(active_cosines) >= 0.95
