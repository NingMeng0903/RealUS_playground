"""Singularity twist-brake contracts.

The singularity brake is a safety layer around the complete task twist.  In
particular, tool-Z force/retract motion must not be copied around the brake:
doing so lets a force task keep driving into an ill-conditioned posture while
the Cartesian axes are being slowed.
"""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.ik_types import SrDampingConfig
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkConfig,
    JointIkController,
    twist_scale_lpf_step,
    twist_scale_target,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import RailExtensionConfig


def test_twist_scale_target_is_monotonic_continuous_and_floored():
    sigma_ref = 0.08
    floor = 0.25
    sigmas = np.linspace(0.0, 0.16, 81)
    scales = [twist_scale_target(s, sigma_ref, floor) for s in sigmas]

    assert scales[0] == floor
    assert twist_scale_target(0.010, sigma_ref, floor) == floor
    assert twist_scale_target(0.024, sigma_ref, floor) == pytest.approx(0.024 / 0.08)
    assert twist_scale_target(sigma_ref, sigma_ref, floor) == 1.0
    assert twist_scale_target(0.12, sigma_ref, floor) == 1.0

    for a, b in zip(scales, scales[1:]):
        assert b + 1e-12 >= a
        assert abs(b - a) < 0.05


def test_twist_scale_lpf_limits_single_tick_jump():
    dt = 0.005
    tau = 0.08
    target = twist_scale_target(0.09, 0.08, 0.25)
    assert target == 1.0
    filt = twist_scale_lpf_step(0.25, target, dt=dt, tau_s=tau)
    assert filt == 0.25 + (dt / tau) * (1.0 - 0.25)
    assert abs(filt - 0.25) < 0.05
    assert filt < 0.40
    assert twist_scale_lpf_step(0.25, 1.0, dt=dt, tau_s=0.0) == 1.0


def test_full_twist_brake_has_no_overforce_retract_bypass(monkeypatch):
    """Even an over-force retract is attenuated while σ is below the floor.

    This uses the controller's public ``JointIkStep.twist_base`` telemetry, so
    it does not depend on the QP's particular joint solution.  The kinematics
    are patched only to place this deterministic fixture below ``sigma_ref``.
    """
    kin = RobotKinematics()
    monkeypatch.setattr(
        kin,
        "singular_values",
        lambda _J: np.array([0.01, 0.04, 0.08, 0.15, 0.30, 0.50]),
    )
    qp = QpConfig(
        backend="proxqp",
        collision=CollisionConfig(enabled=False),
        sr_damping=SrDampingConfig(sigma_ref=0.08),
        twist_sigma_floor=0.25,
        twist_scale_lpf_tau_s=0.0,
    )
    ctrl = JointIkController(
        kin,
        JointIkConfig(
            dt=0.005,
            control_frame="base",
            qp=qp,
            rail_extension=RailExtensionConfig(enabled=False),
        ),
    )
    ctrl.reset(np.zeros(8))
    # Older development snapshots did not initialize this optional debug
    # counter in ``reset``; seeding it keeps the fixture focused on the
    # singularity-brake contract while remaining harmless once initialized by
    # the controller itself.
    if not hasattr(ctrl, "_dbg_tick"):
        ctrl._dbg_tick = 0
    requested = np.array([0.10, -0.03, -0.04, 0.20, -0.10, 0.05])
    step = ctrl.update(
        requested,
        dt=0.005,
        f_ext_z=5.0,
        f_des_z=1.0,
    )
    assert np.all(np.isfinite(step.twist_base))
    # All six components, including negative tool-Z retract, are braked.
    assert np.all(np.abs(step.twist_base) <= np.abs(requested) + 1e-12)
    assert abs(step.twist_base[2]) < abs(requested[2])
    assert abs(step.twist_base[0]) < abs(requested[0])
