"""Unit tests for the geometry-independent reference governor."""

import numpy as np
import pytest

from rm75_control.control.admittance_common.reference import MotionReference
from rm75_control.control.joint_admittance_8dof.generic_tasks import ScalableTask
from rm75_control.control.joint_admittance_8dof.reference_governor import (
    AcceptedTaskReferenceGovernor,
    GovernorConfig,
    ReferenceGovernor,
)
from rm75_control.control.joint_admittance_8dof.task_adapter import (
    CartesianTaskProfile,
)


def test_absolute_reference_and_group_residual_gate_progress() -> None:
    governor = ReferenceGovernor(
        ["position"], GovernorConfig(residual_ok=0.1, residual_max=1.0, tau_s=0.0)
    )
    stopped = governor.update(
        0.1,
        absolute_reference=np.array([1.0, 0.0]),
        current=np.zeros(2),
        residuals={"position": 2.0},
    )
    assert stopped.alpha == pytest.approx(0.0)
    assert np.allclose(stopped.reference, np.zeros(2))
    running = governor.update(
        0.1,
        absolute_reference=np.array([1.0, 0.0]),
        current=np.zeros(2),
        residuals={"position": 0.0},
    )
    assert running.alpha == pytest.approx(1.0)
    assert np.allclose(running.reference, [1.0, 0.0])
    assert running.progress["position"] > 0.0


def test_streaming_twist_is_scaled_without_geometry_assumptions() -> None:
    governor = ReferenceGovernor(["twist"], tau_s=0.0)
    out = governor.update(0.01, streaming_twist=[1.0, -2.0], residuals={"twist": 2.0})
    assert np.allclose(out.twist, [0.0, 0.0])
    assert not out.twist.flags.writeable


def test_scalable_tasks_use_their_own_group_alpha_and_achieved_residual() -> None:
    governor = ReferenceGovernor(tau_s=0.0, residual_ok=0.0, residual_max=1.0)
    task_a = ScalableTask([[1.0, 0.0]], [1.0], "a")
    task_b = ScalableTask([[0.0, 1.0]], [1.0], "b")
    out = governor.update_tasks(
        [task_a, task_b],
        0.01,
        achieved={"a": [1.0, 0.0], "b": [0.0, 0.0]},
    )
    assert out.alphas["a"] == pytest.approx(1.0)
    assert out.alphas["b"] == pytest.approx(0.0)
    assert out.tasks[0].b[0] == pytest.approx(1.0)
    assert out.tasks[1].b[0] == pytest.approx(0.0)


def test_governor_rejects_conflicting_reference_modes() -> None:
    governor = ReferenceGovernor()
    with pytest.raises(ValueError, match="not both"):
        governor.update(0.01, reference=[1.0], twist=[1.0])


def test_solver_authority_and_health_cap_reference_progress() -> None:
    governor = ReferenceGovernor(["motion"], tau_s=0.0)
    limited = governor.update(
        0.01,
        streaming_twist=[0.4, -0.2],
        group_id="motion",
        residuals={"motion": 0.0},
        solver_authority={"motion": 0.25},
        health_scale=0.5,
    )
    assert limited.alphas["motion"] == pytest.approx(0.25)
    np.testing.assert_allclose(limited.twist, [0.1, -0.05])


def test_achieved_residual_is_divided_by_physical_row_scale() -> None:
    governor = ReferenceGovernor(tau_s=0.0, residual_ok=0.0, residual_max=1.0)
    task = ScalableTask([[1.0, 0.0]], [0.2], "motion", row_scales=[0.2])
    output = governor.update_tasks([task], 0.01, achieved={"motion": [0.1, 0.0]})
    # 0.1 m/s error / 0.2 m/s characteristic allowance = 0.5.
    assert output.residuals["motion"] == pytest.approx(0.5)
    assert output.alphas["motion"] == pytest.approx(0.5)


def test_accepted_pose_freezes_only_scalable_rows() -> None:
    profile = CartesianTaskProfile.from_indices(
        protected=(5,), scalable=(("motion", (0, 1)),), name="generic"
    )
    governor = AcceptedTaskReferenceGovernor(profile)
    governor.reset(np.zeros(6))
    first = governor.update(
        MotionReference(
            np.array([0.01, -0.02, 0.0, 0.0, 0.0, 0.10]),
            np.zeros(6),
            t_ref=0.01,
        ),
        dt=0.01,
        rotation_base_task=np.eye(3),
        group_alphas={"motion": 0.0},
    )
    np.testing.assert_allclose(first.pose_d[:2], 0.0, atol=1e-12)
    assert abs(first.pose_d[5] - 0.10) < 1e-9

    second = governor.update(
        MotionReference(
            np.array([0.02, -0.03, 0.0, 0.0, 0.0, 0.20]),
            np.zeros(6),
            t_ref=0.02,
        ),
        dt=0.01,
        rotation_base_task=np.eye(3),
        group_alphas={"motion": 1.0},
    )
    assert 0.0100 < second.pose_d[0] < 0.0103
    assert -0.0103 < second.pose_d[1] < -0.0100
    assert abs(second.pose_d[5] - 0.20) < 1e-9


def test_accepted_pose_selection_is_expressed_in_arbitrary_task_frame() -> None:
    profile = CartesianTaskProfile.from_indices(
        protected=(), scalable=(("task_x", (0,)),), name="rotated"
    )
    governor = AcceptedTaskReferenceGovernor(profile)
    governor.reset(np.zeros(6))
    rotation_base_task = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    result = governor.update(
        MotionReference(np.array([0.0, 0.01, 0, 0, 0, 0]), np.zeros(6)),
        dt=0.01,
        rotation_base_task=rotation_base_task,
        group_alphas={"task_x": 0.5},
    )
    assert abs(result.pose_d[0]) < 1e-12
    assert abs(result.pose_d[1] - 0.005) < 1e-9
