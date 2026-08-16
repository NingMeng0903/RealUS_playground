"""Regression tests for the composed 8-DOF nullspace velocity cap."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rm75_control.control.joint_admittance_8dof.tasks.secondary_composer import (
    SecondaryComposer,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import (
    RailExtensionConfig,
    RailExtensionTask,
)


class _Centering:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(activation=0.8)
        self.q_mid = np.zeros(8)
        self.half = np.ones(8)

    def __call__(self, q: np.ndarray) -> np.ndarray:
        # Include a rail value deliberately: coupled mode must clear it.
        return np.r_[5.0, np.full(7, 0.3)]


class _Arm:
    last_singularity_smooth = 1.0

    def __call__(self, q: np.ndarray) -> np.ndarray:
        return np.r_[0.0, np.full(7, 0.3)]


class _Manipulability:
    def __call__(
        self, q: np.ndarray, *, sigma_min: float, exclude_rail: bool
    ) -> np.ndarray:
        out = np.full(8, 0.3)
        if exclude_rail:
            out[0] = 5.0
        return out


def test_composed_centering_psi_manipulability_share_one_arm_cap() -> None:
    v_max = np.r_[0.3, np.ones(7)]
    composer = SecondaryComposer(
        _Centering(),
        _Arm(),
        manipulability=_Manipulability(),
        v_max=v_max,
        max_qdot_frac=0.2,
    )

    qdot = composer.compose(
        np.zeros(8),
        None,
        np.zeros(8),
        arm_suppressed=False,
        manipulability_active=True,
        sigma_min=0.01,
    )

    # The individual terms add constructively before the cap.  The arm part
    # must nevertheless remain within one configured fraction of v_max.
    assert np.all(np.abs(qdot[1:]) <= 0.2 * v_max[1:] + 1e-12)
    # No secondary task is allowed to drive the rail in coupled mode.
    assert qdot[0] == 0.0


def test_joint_plan_feedforward_remains_after_secondary_cap() -> None:
    v_max = np.r_[0.3, np.ones(7)]
    composer = SecondaryComposer(
        _Centering(),
        _Arm(),
        v_max=v_max,
        max_qdot_frac=0.2,
    )
    qdot_ff = np.r_[0.12, np.zeros(7)]
    qdot = composer.compose(
        np.zeros(8),
        qdot_ff,
        np.zeros(8),
        arm_suppressed=False,
    )

    # qdot_ff is an explicit joint plan, not a nullspace soft task, and keeps
    # its rail command after the arm-secondary cap.
    assert qdot[0] == qdot_ff[0]


def test_reach_cap_is_not_inflated_by_large_extension_error() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_ext=10.0,
            k_esc=0.0,
            k_ff=0.0,
            e0_m=0.01,
            e1_m=0.04,
            v_reach_cap_m_s=0.02,
            v_max_m_s=0.08,
            v_lpf_tau_s=0.0,
        ),
    )
    q = 0.5 * (kin.q_lower + kin.q_upper)
    task.set_d_pref(task.extension(q) - 0.30)

    task(q, sigma_scale=1.0, dt_s=0.005, stroke_limiters=False)

    # The old d* drift term expanded this 20 mm/s budget by 60 mm/s.
    assert abs(task.last_v_reach) <= 0.02 + 1e-12
