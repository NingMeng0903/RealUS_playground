"""Focused Stage-1 regressions for secondary tasks and the rail contract."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkConfig,
    JointIkController,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig
from rm75_control.control.joint_admittance_8dof.tasks.nullspace_task import (
    JointCenteringTask,
    NullspaceTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import (
    RailExtensionConfig,
    RailExtensionTask,
)
from rm75_control.control.joint_admittance_8dof.tasks.secondary_composer import (
    SecondaryComposer,
)
from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge,
    parse_rail_servo_config,
)


class _FixedManip:
    def __init__(self, value: np.ndarray) -> None:
        self.value = np.asarray(value, dtype=float)
        self.calls = 0

    def __call__(self, q, *, sigma_min=1.0, exclude_rail=False, dt_s=None):
        del q, sigma_min, dt_s
        self.calls += 1
        out = self.value.copy()
        if exclude_rail:
            out[0] = 0.0
        return out


def test_manipulability_exclusively_owns_escape_nullspace_slot():
    """The proven 4d/c3 escape policy must not cancel grad-mu with centering."""
    kin = RobotKinematics()
    centering = JointCenteringTask.from_kinematics(
        kin,
        NullspaceTaskConfig(k_center=0.8, k_limit=0.0),
    )
    manip = _FixedManip(
        np.array([0.0, 0.15, -0.10, 0.08, 0.0, 0.03, -0.04, 0.02])
    )
    comp = SecondaryComposer(centering, None, manipulability=manip)
    q = centering.q_mid + np.array(
        [0.0, 0.12, -0.08, 0.06, 0.04, -0.05, 0.03, -0.02]
    )

    center_only = comp.compose(
        q,
        None,
        None,
        arm_suppressed=True,
        centering_suppressed=False,
        manipulability_active=False,
        centering_sigma_fade=False,
    )
    manip_only = comp.compose(
        q,
        None,
        None,
        arm_suppressed=True,
        centering_suppressed=True,
        manipulability_active=True,
        sigma_min=0.04,
        centering_sigma_fade=False,
    )
    both = comp.compose(
        q,
        None,
        None,
        arm_suppressed=True,
        centering_suppressed=False,
        manipulability_active=True,
        sigma_min=0.04,
        centering_sigma_fade=False,
    )
    assert manip.calls >= 2
    assert np.linalg.norm(center_only) > 1e-6
    assert np.linalg.norm(manip_only) > 1e-6
    assert np.allclose(both, manip_only, atol=1e-12)


def test_pose_guard_has_enter_exit_hysteresis():
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(sigma_guard_enter=0.45, sigma_guard_exit=0.70),
    )
    def active(scale: float) -> bool:
        task._sigma_guard_velocity(
            sigma_scale=scale,
            sigma_grad_rail=1.0,
            v_primary=0.0,
        )
        return task._guard_active

    assert not active(0.50)
    assert active(0.40)
    # Between thresholds, the latch remains active.
    assert active(0.60)
    assert not active(0.71)


def test_sigma_gradient_reversal_is_continuous_and_not_latched():
    """The restored 4d task follows the current gradient on every call."""
    kin = RobotKinematics()
    task = RailExtensionTask(kin, RailExtensionConfig(enabled=True))
    q = np.zeros(8)
    q[0] = 0.40
    task.reset(q)

    v_pos, _ = task(
        q,
        sigma_scale=0.0,
        sigma_grad_rail=0.10,
    )
    v_neg, _ = task(
        q,
        sigma_scale=0.0,
        sigma_grad_rail=-0.10,
    )
    assert v_pos > 0.0
    assert v_neg < 0.0
    assert v_neg == pytest.approx(-v_pos)


def test_phase_reentry_clears_stale_rail_filter_and_gradient():
    """A hold boundary cannot replay the previous scan direction."""
    kin = RobotKinematics()
    ctrl = JointIkController(
        kin,
        JointIkConfig(
            control_frame="base",
            qp=QpConfig(collision=CollisionConfig(enabled=False)),
        ),
    )
    q = np.array(
        [
            0.40,
            -0.905938,
            1.117987,
            0.459109,
            1.775407,
            -0.342094,
            1.06775,
            0.749873,
        ]
    )
    ctrl.reset(q)
    task = ctrl.rail_ext_task
    assert task is not None

    vel_ff = np.zeros(6)
    vel_ff[1] = 0.08
    for _ in range(40):
        v_before, _ = task(q, vel_ff=vel_ff, dt_s=0.005)
    assert v_before > 0.07

    ctrl._sigma_grad_tick = 7
    ctrl._sigma_grad_rail_cached = 0.2
    ctrl.set_rail_extension_active(False)
    assert ctrl._sigma_grad_tick == 0
    assert ctrl._sigma_grad_rail_cached == 0.0
    assert not task._v_lpf_initialized

    ctrl.set_rail_extension_active(True)
    vel_ff[1] = -0.08
    v_after, _ = task(q, vel_ff=vel_ff, dt_s=0.005)
    assert v_after < -0.07

    ctrl._sigma_grad_tick = 5
    ctrl._sigma_grad_rail_cached = -0.2
    ctrl.capture_rail_extension_ref()
    assert ctrl._sigma_grad_tick == 0
    assert ctrl._sigma_grad_rail_cached == 0.0
    assert not task._v_lpf_initialized


def test_rail_extension_weight_uses_uncapped_4d_schedule():
    kin = RobotKinematics()
    cfg = RailExtensionConfig(
        w_max=4.0,
        w_sigma_floor=2.0,
        k_sigma_boost=8.0,
        e0_m=0.01,
        e1_m=0.05,
    )
    task = RailExtensionTask(kin, cfg)
    q = np.zeros(8)
    task.reset(q)
    task.d_pref_m = float(task.d_pref_m) - 0.20
    _, weight = task(
        q,
        sigma_scale=0.0,
        sigma_grad_rail=0.0,
    )
    expected = (cfg.w_max + cfg.w_sigma_floor) * (1.0 + cfg.k_sigma_boost)
    assert weight == pytest.approx(expected)
    assert task.last_weight_raw == pytest.approx(expected)
    assert task.last_weight_capped == pytest.approx(expected)


def test_rail_soft_bounds_are_shared_and_mismatch_is_rejected():
    raw = {
        "hw": {
            "lw100": {
                "enabled": True,
                "soft_min_m": 0.03,
                "soft_max_m": 0.70,
                "sign": -1.0,
            }
        },
        "inner": {
            "rail": {
                "travel_m": 0.80,
                "soft_min_m": 0.03,
                "soft_max_m": 0.70,
            }
        },
    }
    cfg = parse_rail_servo_config(raw)
    assert cfg.soft_min_m == pytest.approx(0.03)
    assert cfg.soft_max_m == pytest.approx(0.70)
    assert cfg.sign == pytest.approx(-1.0)

    mismatched = {
        **raw,
        "hw": {
            "lw100": {
                **raw["hw"]["lw100"],
                "soft_max_m": 0.69,
            }
        },
    }
    with pytest.raises(ValueError, match="soft-limit mismatch"):
        parse_rail_servo_config(mismatched)


def test_rail_panic_latch_rejects_targets_until_explicit_rearm():
    cfg = parse_rail_servo_config(
        {
            "hw": {"lw100": {"soft_min_m": 0.03, "soft_max_m": 0.70}},
            "inner": {"rail": {"travel_m": 0.80, "soft_min_m": 0.03, "soft_max_m": 0.70}},
        }
    )
    bridge = RailServoBridge(cfg)
    with bridge._lock:
        bridge._calibrated = True
        bridge._armed = True
    bridge.set_target_m(0.35)
    assert bridge._target_m == pytest.approx(0.35)

    bridge._trip_panic(0.35, "test limit")
    bridge.set_target_m(0.40)
    assert bridge.panicked
    assert bridge._target_m == pytest.approx(0.35)
