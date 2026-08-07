"""Runtime 8-DOF posture attractor + c3ba58e XOR nullspace (∇μ replaces centering)."""

from __future__ import annotations

import numpy as np
import pytest

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


class _FakeManip:
    """Unit ∇μ along +J5."""

    def __call__(self, q, *, sigma_min=1.0, exclude_rail=True):
        del sigma_min, exclude_rail
        out = np.zeros_like(q, dtype=float)
        out[5] = 0.5
        return out


def test_manip_xor_replaces_centering_when_armed() -> None:
    """c3ba58e: manipulability_active → only ∇μ (centering fully off)."""
    task = _task()
    composer = SecondaryComposer(
        task,
        None,
        manipulability=_FakeManip(),
        v_max=np.ones(8) * 10.0,
        max_qdot_frac=1.0,
    )
    q = np.zeros(8)
    q[5] = np.deg2rad(120.0)
    q[2] = np.deg2rad(-90.0)
    center_only = composer.compose(
        q, None, None, arm_suppressed=True, manipulability_active=False
    )
    manip_only = composer.compose(
        q,
        None,
        None,
        arm_suppressed=True,
        manipulability_active=True,
        sigma_min=0.02,
        sigma_ref=0.08,
        centering_sigma_fade=False,
    )
    assert center_only[5] < 0.0
    assert manip_only[5] == pytest.approx(0.5)
    # XOR: orthogonal centering joints are also dropped while manip is armed.
    assert manip_only[2] == pytest.approx(0.0)


def test_centering_returns_when_manip_inactive() -> None:
    """When escape is off, centering pulls all arm joints (incl. J6)."""
    task = _task()
    composer = SecondaryComposer(
        task,
        None,
        manipulability=_FakeManip(),
        v_max=np.ones(8) * 10.0,
        max_qdot_frac=1.0,
    )
    q = np.zeros(8)
    q[6] = np.deg2rad(90.0)
    q[2] = np.deg2rad(-90.0)
    full = composer.compose(
        q, None, None, arm_suppressed=True, manipulability_active=False
    )
    escaping = composer.compose(
        q,
        None,
        None,
        arm_suppressed=True,
        manipulability_active=True,
        sigma_min=0.01,
        centering_sigma_fade=False,
    )
    assert abs(full[6]) > 0.0
    assert abs(full[2]) > 0.0
    assert escaping[6] == pytest.approx(0.0)
    assert escaping[2] == pytest.approx(0.0)
    assert escaping[5] == pytest.approx(0.5)


def test_force_manip_when_sigma_below_ref() -> None:
    sigma_ref = 0.08
    sigma_now = 0.05
    force_manip = sigma_ref > 1e-9 and sigma_now < sigma_ref
    assert force_manip is True
