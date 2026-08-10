"""Focused Stage-1 regressions for secondary tasks and the rail contract."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
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
        sigma_escape_ref=0.10,
        centering_sigma_fade=False,
    )
    assert manip.calls >= 2
    assert np.linalg.norm(center_only) > 1e-6
    assert np.linalg.norm(manip_only) > 1e-6
    assert np.allclose(both, manip_only, atol=1e-12)


def test_sigma_escape_guard_has_enter_exit_hysteresis():
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(sigma_guard_enter=0.45, sigma_guard_exit=0.70),
    )
    assert not task._sigma_guard_hold(0.50)
    assert task._sigma_guard_hold(0.40)
    # Between thresholds, the latch remains active.
    assert task._sigma_guard_hold(0.60)
    assert not task._sigma_guard_hold(0.71)


def test_sigma_escape_direction_is_latched_until_exit_threshold():
    """Finite-difference sign noise cannot reverse an active rail escape."""
    kin = RobotKinematics()
    task = RailExtensionTask(kin, RailExtensionConfig(enabled=True))
    q = np.zeros(8)
    task.reset(q)

    v_pos, _ = task(
        q,
        sigma_scale=0.0,
        sigma_escape_scale=0.0,
        sigma_grad_rail=0.10,
        sigma_min=0.05,
        sigma_escape_enter=0.10,
        sigma_escape_exit=0.12,
    )
    assert task.escape_active
    assert task.escape_sign > 0.0
    assert v_pos >= 0.0

    # Still below exit: a reversed gradient must retain the committed sign.
    v_latched, _ = task(
        q,
        sigma_scale=0.0,
        sigma_escape_scale=0.0,
        sigma_grad_rail=-0.10,
        sigma_min=0.08,
        sigma_escape_enter=0.10,
        sigma_escape_exit=0.12,
    )
    assert task.escape_active
    assert task.escape_sign > 0.0
    assert v_latched >= 0.0

    # At/above exit, release the sign so a new episode may choose either side.
    task(
        q,
        sigma_scale=0.0,
        sigma_escape_scale=0.0,
        sigma_grad_rail=-0.10,
        sigma_min=0.12,
        sigma_escape_enter=0.10,
        sigma_escape_exit=0.12,
    )
    assert not task.escape_active
    assert task.escape_sign == pytest.approx(0.0)


def test_rail_extension_weight_is_hard_capped_after_sigma_boost():
    kin = RobotKinematics()
    cfg = RailExtensionConfig(
        w_max=4.0,
        w_sigma_floor=2.0,
        k_sigma_boost=8.0,
        # The rail is an escape hint, not a competing Cartesian task.  Keep
        # its absolute cap at 2.0 even when the sigma boost is extreme.
        weight_hard_max=2.0,
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
        sigma_escape_scale=0.0,
        sigma_grad_rail=0.0,
    )
    assert weight <= 2.0 + 1e-12
    assert task.last_weight <= 2.0 + 1e-12


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
