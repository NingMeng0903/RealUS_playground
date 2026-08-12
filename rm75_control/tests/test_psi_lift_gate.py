"""Lift-gate behaviour for soft ψ@D planar recovery."""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.generic_runtime import (
    GenericQpikRuntimeConfig,
)
from rm75_control.control.joint_admittance_8dof.loop import JointIkConfig, JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, full_q_from_arm
from rm75_control.control.joint_admittance_8dof.solver.two_level_qpik import (
    TwoLevelQpikConfig,
)


Q_HOME = full_q_from_arm(
    np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), 0.4
)


def _ctrl() -> JointIkController:
    cfg = JointIkConfig(
        generic_qpik=GenericQpikRuntimeConfig(
            solver=TwoLevelQpikConfig(
                backend="scipy",
            max_solve_ms=500.0,
                max_rows=96,
                max_scalable_groups=4,
            )
        ),
        collision=CollisionConfig(enabled=False),
        psi_lift_enabled=True,
        psi_lift_lost_contact_s=0.02,
        psi_lift_fz_frac=0.5,
        psi_lift_vz_m_s=0.005,
    )
    ctrl = JointIkController(RobotKinematics(), cfg)
    ctrl.reset(Q_HOME)
    ctrl.ensure_psi_ref_from_q(Q_HOME)
    return ctrl


def test_psi_gate_opens_after_lost_contact_dwell() -> None:
    ctrl = _ctrl()
    assert ctrl._update_psi_lift_gate(  # noqa: SLF001
        dt=0.005,
        contact_active=False,
        f_ext_z=0.0,
        f_des_z=2.0,
        twist_tool_z=0.0,
    ) is False
    assert ctrl._update_psi_lift_gate(  # noqa: SLF001
        dt=0.020,
        contact_active=False,
        f_ext_z=0.0,
        f_des_z=2.0,
        twist_tool_z=0.0,
    ) is True
    soft = ctrl._build_psi_soft(Q_HOME)  # noqa: SLF001
    assert soft is not None
    assert soft["grad"].shape == (8,)


def test_psi_gate_opens_on_low_force_retract() -> None:
    ctrl = _ctrl()
    assert ctrl._update_psi_lift_gate(  # noqa: SLF001
        dt=0.005,
        contact_active=True,
        f_ext_z=0.4,
        f_des_z=2.0,
        twist_tool_z=-0.01,
    ) is True


def test_psi_gate_closed_during_firm_contact_press() -> None:
    ctrl = _ctrl()
    assert ctrl._update_psi_lift_gate(  # noqa: SLF001
        dt=0.005,
        contact_active=True,
        f_ext_z=2.0,
        f_des_z=2.0,
        twist_tool_z=0.01,
    ) is False
    # Lift gate off, but ψ@D attractor stays on once latched (anti-fold).
    soft = ctrl._build_psi_soft(Q_HOME)  # noqa: SLF001
    assert soft is not None
    assert soft["weight"] > 0.0


def test_psi_weight_boosts_near_singularity_and_large_error() -> None:
    ctrl = _ctrl()
    ctrl._update_psi_lift_gate(  # noqa: SLF001
        dt=0.005,
        contact_active=True,
        f_ext_z=2.0,
        f_des_z=2.0,
        twist_tool_z=0.0,
    )
    base = ctrl._build_psi_soft(Q_HOME)  # noqa: SLF001
    assert base is not None
    # Unified arm dexterity (d_arm / arm_rho), not full-J sigma_min.
    ctrl.last_arm_rho = 0.01
    boosted = ctrl._build_psi_soft(Q_HOME)  # noqa: SLF001
    assert boosted is not None
    assert boosted["weight"] > base["weight"]
