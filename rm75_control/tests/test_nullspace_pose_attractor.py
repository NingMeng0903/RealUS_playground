"""Runtime 8-DOF posture-attractor contract."""

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.tasks.nullspace_task import (
    JointCenteringTask,
    NullspaceTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.secondary_composer import (
    SecondaryComposer,
)


def _task() -> JointCenteringTask:
    return JointCenteringTask(
        np.array([0.0] + [-2.0] * 7),
        np.array([0.8] + [2.0] * 7),
        NullspaceTaskConfig(
            k_center=1.0,
            k_limit=0.0,
            weights=np.array([0.0] + [1.0] * 7),
            q_nominal_rad=np.zeros(8),
        ),
    )


def test_runtime_target_accepts_eight_values_but_does_not_attract_rail() -> None:
    task = _task()
    target = np.linspace(0.1, 0.8, 8)
    task.set_q_target(target)
    assert np.allclose(task.q_target, target)

    velocity = task(np.zeros(8))
    assert velocity[0] == pytest.approx(0.0)
    assert np.linalg.norm(velocity[1:]) > 0.0


def test_runtime_target_rejects_partial_or_nonfinite_updates() -> None:
    task = _task()
    with pytest.raises(ValueError):
        task.set_q_target(np.zeros(7))
    with pytest.raises(ValueError):
        task.set_q_target(np.array([0.0] * 7 + [np.nan]))


def test_post_escape_recovery_stays_latched_until_arm_nears_target() -> None:
    task = _task()
    controller = JointIkController.__new__(JointIkController)
    controller.centering_task = task
    controller.cfg = SimpleNamespace(
        centering_recovery_gain=3.0,
        centering_recovery_tol=0.12,
    )
    controller._singularity_escape_seen = False
    controller._centering_recovery_active = False

    q_far = np.array([0.4] + [1.0] * 7)
    assert controller._centering_recovery_scale(q_far, 0.10, 0.16) == 1.0
    assert controller._singularity_escape_seen

    # Once sigma is healthy, recovery remains strong for as many ticks as the
    # nullspace needs; it is not a fixed-duration pulse.
    assert controller._centering_recovery_scale(q_far, 0.20, 0.16) == 3.0
    assert controller._centering_recovery_scale(q_far, 0.25, 0.16) == 3.0
    assert controller._centering_recovery_active

    q_near = np.array([0.4] + [0.1] * 7)
    assert controller._centering_recovery_scale(q_near, 0.25, 0.16) == 1.0
    assert not controller._singularity_escape_seen
    assert not controller._centering_recovery_active


def test_recovery_gain_is_arm_only_and_raises_the_soft_velocity_cap() -> None:
    task = _task()
    composer = SecondaryComposer(
        task,
        None,
        v_max=np.ones(8),
        max_qdot_frac=0.2,
    )
    q_far = np.array([0.4] + [1.0] * 7)
    normal = composer.compose(
        q_far,
        None,
        None,
        arm_suppressed=True,
    )
    recovery = composer.compose(
        q_far,
        None,
        None,
        arm_suppressed=True,
        centering_gain_scale=3.0,
        max_qdot_frac_override=0.35,
    )

    assert normal[0] == pytest.approx(0.0)
    assert recovery[0] == pytest.approx(0.0)
    assert np.max(np.abs(normal[1:])) == pytest.approx(0.2)
    assert np.max(np.abs(recovery[1:])) == pytest.approx(0.35)
