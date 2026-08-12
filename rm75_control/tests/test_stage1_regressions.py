"""Focused rail ownership and hardware-contract regressions."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge,
    parse_rail_servo_config,
)
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkConfig,
    JointIkController,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics


def test_phase_reentry_restores_generic_rail_ownership() -> None:
    kin = RobotKinematics()
    ctrl = JointIkController(
        kin,
        JointIkConfig(control_frame="base", collision=CollisionConfig(enabled=False)),
    )
    q = np.array(
        [0.40, -0.905938, 1.117987, 0.459109, 1.775407, -0.342094, 1.06775, 0.749873]
    )
    ctrl.reset(q)
    ctrl.set_locked("hold", q_ref_m=float(q[0]))
    assert ctrl.is_locked_hold
    ctrl.set_coupled()
    assert ctrl.rail_mode.value == "coupled"
    assert not ctrl.is_locked_hold


def test_rail_soft_bounds_are_shared_and_mismatch_is_rejected() -> None:
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


def test_rail_panic_latch_rejects_targets_until_explicit_rearm() -> None:
    cfg = parse_rail_servo_config(
        {
            "hw": {"lw100": {"soft_min_m": 0.03, "soft_max_m": 0.70}},
            "inner": {
                "rail": {
                    "travel_m": 0.80,
                    "soft_min_m": 0.03,
                    "soft_max_m": 0.70,
                }
            },
        }
    )
    bridge = RailServoBridge(cfg)
    with bridge._lock:
        bridge._calibrated = True
        bridge._armed = True
    assert bridge.set_target_m(0.35)
    assert bridge._target_m == pytest.approx(0.35)

    bridge._trip_panic(0.35, "test limit")
    assert not bridge.set_target_m(0.40)
    assert bridge.panicked
    assert bridge._target_m == pytest.approx(0.35)
