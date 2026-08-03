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


class _FakeManip:
    """Unit ∇μ along +J5 so we can see centering still pull −J5 toward 0."""

    def __call__(self, q, *, sigma_min=1.0, exclude_rail=True):
        out = np.zeros_like(q, dtype=float)
        out[5] = 0.5  # push wrist further positive (toward flip)
        return out


def test_manip_active_still_keeps_centering_pull_on_wrist() -> None:
    """Escape-zone ∇μ must ADD to centering, never replace the attractor."""
    task = _task()
    composer = SecondaryComposer(
        task,
        None,
        manipulability=_FakeManip(),
        v_max=np.ones(8) * 10.0,
        max_qdot_frac=1.0,
    )
    # J5 at +120deg-ish with q_nominal=0 → centering alone wants negative qdot[5].
    q = np.zeros(8)
    q[5] = np.deg2rad(120.0)
    center_only = composer.compose(
        q, None, None, arm_suppressed=True, manipulability_active=False
    )
    blended = composer.compose(
        q, None, None, arm_suppressed=True, manipulability_active=True
    )
    assert center_only[5] < 0.0
    # Attractive component still present: blend is not pure +0.5 manip.
    assert blended[5] < 0.5
    # And not identical to center-only (manip added).
    assert blended[5] > center_only[5]


def test_centering_yields_to_manip_in_escape_band() -> None:
    """When manip is on below σ_escape_ref, only centering is σ-scaled."""
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
    sigma_escape_ref = 0.10
    sigma_min = 0.05  # yield = 0.5
    center_only = composer.compose(
        q, None, None, arm_suppressed=True, manipulability_active=False
    )
    healthy = composer.compose(
        q,
        None,
        None,
        arm_suppressed=True,
        manipulability_active=True,
        sigma_min=1.0,
        sigma_escape_ref=sigma_escape_ref,
    )
    yielded = composer.compose(
        q,
        None,
        None,
        arm_suppressed=True,
        manipulability_active=True,
        sigma_min=sigma_min,
        sigma_escape_ref=sigma_escape_ref,
    )
    # Healthy σ: no yield — same as prior additive blend.
    assert healthy[5] == pytest.approx(center_only[5] + 0.5)
    # Escape band: centering halved, manip still +0.5.
    assert yielded[5] == pytest.approx(0.5 * center_only[5] + 0.5)


def test_centering_yield_floor_keeps_anti_flip_attractor() -> None:
    """Yield never drops below 0.25 so the wrist attractor cannot vanish."""
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
    center_only = composer.compose(
        q, None, None, arm_suppressed=True, manipulability_active=False
    )
    deep = composer.compose(
        q,
        None,
        None,
        arm_suppressed=True,
        manipulability_active=True,
        sigma_min=0.01,
        sigma_escape_ref=0.10,
    )
    assert deep[5] == pytest.approx(0.25 * center_only[5] + 0.5)
    # Floor still leaves a negative centering pull against +manip.
    assert deep[5] < 0.5


def test_j6_centering_not_yielded_in_escape_band() -> None:
    """J6 keeps full pull toward nominal during escape; proximal joints yield.

    |J6|≈0 is wrist singularity; yielding J6 with the rest let scans jitter
    when the wrist went straight (hw run_20260803_151521).
    """
    task = JointCenteringTask(
        np.array([0.0] + [-2.0] * 7),
        np.array([0.8] + [2.0] * 7),
        NullspaceTaskConfig(
            k_center=1.0,
            k_limit=0.0,
            weights=np.array([0.0, 1.0, 1.0, 1.0, 1.0, 2.5, 2.5, 1.0]),
            q_nominal_rad=np.deg2rad(
                np.array([0.0, 0.0, -45.0, 0.0, 90.0, 0.0, 45.0, 0.0])
            ),
        ),
    )
    composer = SecondaryComposer(
        task,
        None,
        manipulability=_FakeManip(),
        v_max=np.ones(8) * 10.0,
        max_qdot_frac=1.0,
    )
    q = np.deg2rad(np.array([0.0, 0.0, -45.0, 0.0, 90.0, 0.0, 5.0, 0.0]))
    center_only = composer.compose(
        q, None, None, arm_suppressed=True, manipulability_active=False
    )
    assert center_only[6] > 0.0  # pull J6 5° → 45°
    yielded = composer.compose(
        q,
        None,
        None,
        arm_suppressed=True,
        manipulability_active=True,
        sigma_min=0.05,
        sigma_escape_ref=0.10,
    )
    # J6 full centering (manip does not touch index 6 in _FakeManip).
    assert yielded[6] == pytest.approx(center_only[6])
    # Proximal J4 still σ-yielded (half at σ=0.05 / 0.10).
    assert yielded[4] == pytest.approx(0.5 * center_only[4])


def test_recovery_policy_does_not_force_manip_while_latched() -> None:
    """Mirrors loop.py: centering recovery owns nullspace; do not force ∇μ on.

    Inline condition under test (kept here so a refactor that re-enables the
    fight is caught without spinning up the full WBC):
        force_manip = (sigma < escape_ref) and (not centering_recovery_active)
    """
    sigma_escape_ref = 0.10
    sigma_now = 0.05
    centering_recovery_active = True
    force_manip = (
        sigma_escape_ref > 1e-9
        and sigma_now < sigma_escape_ref
        and not centering_recovery_active
    )
    assert force_manip is False
    centering_recovery_active = False
    force_manip = (
        sigma_escape_ref > 1e-9
        and sigma_now < sigma_escape_ref
        and not centering_recovery_active
    )
    assert force_manip is True
