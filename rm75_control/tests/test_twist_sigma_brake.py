"""4d singularity brake with the e85 over-force retract exception."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.ik_types import SrDampingConfig
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkConfig,
    JointIkController,
    twist_scale_target,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import RailExtensionConfig


def test_twist_scale_target_is_monotonic_with_frozen_4d_branches():
    sigma_ref = 0.08
    floor = 0.25
    sigmas = np.linspace(0.0, 0.16, 81)
    scales = [twist_scale_target(s, sigma_ref, floor) for s in sigmas]

    assert scales[0] == pytest.approx(max(floor * 0.5, floor * floor))
    assert twist_scale_target(0.010, sigma_ref, floor) == pytest.approx(0.125)
    assert twist_scale_target(0.024, sigma_ref, floor) == pytest.approx(0.125)
    assert twist_scale_target(sigma_ref, sigma_ref, floor) == 1.0
    assert twist_scale_target(0.12, sigma_ref, floor) == 1.0

    # The restored 4d controller deliberately switches from the squared deep
    # branch to the linear branch at sigma_ref/2.  That boundary is not a
    # continuously blended transition; check continuity within each branch
    # and permit only the known branch jump.
    boundary = 0.5 * sigma_ref
    for a_sigma, a, b_sigma, b in zip(sigmas, scales, sigmas[1:], scales[1:]):
        assert b + 1e-12 >= a
        if a_sigma < boundary <= b_sigma:
            assert a == pytest.approx((a_sigma / sigma_ref) ** 2)
            assert b == pytest.approx(b_sigma / sigma_ref)
        else:
            assert abs(b - a) < 0.05


def _low_sigma_controller(monkeypatch, control_frame: str) -> JointIkController:
    """Build a deterministic low-sigma controller for public twist telemetry."""
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
        twist_sigma_floor=0.08,
    )
    ctrl = JointIkController(
        kin,
        JointIkConfig(
            dt=0.005,
            control_frame=control_frame,
            qp=qp,
            rail_extension=RailExtensionConfig(enabled=False),
        ),
    )
    ctrl.reset(np.zeros(8))
    return ctrl


@pytest.mark.parametrize("control_frame", ["tool", "base"])
def test_overforce_retract_keeps_tool_z_but_scales_other_axes(
    monkeypatch, control_frame: str
):
    """e85 bypasses only negative tool-Z during an over-force retract.

    This uses the controller's public ``JointIkStep.twist_base`` telemetry, so
    it does not depend on the QP's particular joint solution.  The kinematics
    are patched only to place this deterministic fixture below ``sigma_ref``.
    """
    ctrl = _low_sigma_controller(monkeypatch, control_frame)
    requested_tool = np.array([0.10, -0.03, -0.04, 0.20, -0.10, 0.05])
    R_base_tool = ctrl.kin.fk_placement(ctrl.q_cmd).rotation
    requested = requested_tool.copy()
    if control_frame == "base":
        requested[:3] = R_base_tool @ requested_tool[:3]
        requested[3:] = R_base_tool @ requested_tool[3:]
    step = ctrl.update(
        requested,
        dt=0.005,
        f_ext_z=5.0,
        f_des_z=1.0,
    )
    assert np.all(np.isfinite(step.twist_base))
    observed_tool = np.concatenate(
        [
            R_base_tool.T @ step.twist_base[:3],
            R_base_tool.T @ step.twist_base[3:],
        ]
    )
    scale = twist_scale_target(0.01, 0.08, 0.08)
    assert observed_tool[2] == pytest.approx(requested_tool[2])
    expected_tool = requested_tool * scale
    expected_tool[2] = requested_tool[2]
    assert np.allclose(observed_tool, expected_tool, atol=1e-12)


@pytest.mark.parametrize("control_frame", ["tool", "base"])
def test_underforce_press_and_tangential_axes_are_scaled(
    monkeypatch, control_frame: str
):
    """Press and tangential motion retain the immediate 4d brake."""
    ctrl = _low_sigma_controller(monkeypatch, control_frame)
    requested_tool = np.array([0.10, -0.03, 0.04, 0.20, -0.10, 0.05])
    R_base_tool = ctrl.kin.fk_placement(ctrl.q_cmd).rotation
    requested = requested_tool.copy()
    if control_frame == "base":
        requested[:3] = R_base_tool @ requested_tool[:3]
        requested[3:] = R_base_tool @ requested_tool[3:]
    step = ctrl.update(requested, dt=0.005, f_ext_z=0.5, f_des_z=1.0)
    observed_tool = np.concatenate(
        [
            R_base_tool.T @ step.twist_base[:3],
            R_base_tool.T @ step.twist_base[3:],
        ]
    )
    scale = twist_scale_target(0.01, 0.08, 0.08)
    assert np.allclose(observed_tool, requested_tool * scale, atol=1e-12)
    assert abs(observed_tool[0]) < abs(requested_tool[0])
