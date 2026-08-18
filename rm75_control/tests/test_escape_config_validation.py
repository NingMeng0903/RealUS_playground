"""Configuration contract for the slack-QP 8-DOF inner loop."""

from __future__ import annotations

from copy import deepcopy
import inspect
import math
from pathlib import Path

import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.hw.rail_servo import parse_rail_servo_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpIkController


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"


def _raw() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def test_production_config_declares_slack_qp_and_canonical_soft_band():
    raw = _raw()
    cfg = build_joint_ik_config(_raw())
    assert cfg.qp.backend.lower() == "proxqp"
    assert cfg.qp.max_iter == 400
    assert cfg.qp.twist_sigma_floor == pytest.approx(0.02)
    assert cfg.qp.task_weight_min_frac == pytest.approx(0.05)
    assert cfg.qp.near_arm_margin_rad == pytest.approx(0.08)
    assert cfg.qp.sigma_setbased.enabled
    assert cfg.qp.branch_barrier.enabled
    assert cfg.nullspace.q_nominal_rad is not None
    assert cfg.rail_extension.enabled
    assert cfg.a_max_arm_rad_s2 == pytest.approx(3.0)
    assert cfg.a_max_rail_m_s2 == pytest.approx(0.60)
    assert cfg.qp.j_max_arm_rad_s3 == pytest.approx(300.0)
    assert cfg.qp.limit_damper_band_rail_m == pytest.approx(0.025)
    assert cfg.rail.soft_min_m == pytest.approx(0.030)
    assert cfg.rail.soft_max_m == pytest.approx(0.755)
    assert cfg.rail.hard_min_m == pytest.approx(0.005)
    assert cfg.rail.hard_max_m == pytest.approx(0.78)
    assert cfg.rail.soft_min_m == pytest.approx(
        raw["qpik"]["hard_limits"]["rail"]["soft_min_m"]
    )
    assert raw["hw"]["lw100"]["soft_min_m"] == pytest.approx(0.030)
    assert raw["hw"]["lw100"]["soft_min_m"] == pytest.approx(
        raw["qpik"]["hard_limits"]["rail"]["soft_min_m"]
    )
    servo = parse_rail_servo_config(raw)
    assert servo.poll_hz == pytest.approx(60.0)
    assert servo.accel_ms == 120
    assert servo.decel_ms == 120
    assert servo.vel_amax_m_s2 == pytest.approx(0.60)
    assert servo.vel_max_m_s == pytest.approx(0.12)
    assert cfg.psi_retarget.psi_attr_rad == pytest.approx(math.radians(68.0))
    assert cfg.rail_extension.escape_sign_policy == "minus"
    assert servo.soft_min_m == pytest.approx(0.030)
    assert servo.soft_max_m == pytest.approx(cfg.rail.soft_max_m)
    assert servo.hard_min_m == pytest.approx(0.005)
    assert servo.hard_max_m == pytest.approx(0.78)
    assert servo.vel_kp == pytest.approx(14.0)
    assert servo.vel_kd == pytest.approx(0.22)
    assert servo.target_stale_coast_s == pytest.approx(0.35)
    assert cfg.rail.soft_max_m == pytest.approx(
        raw["qpik"]["hard_limits"]["rail"]["soft_max_m"]
    )
    assert cfg.rail.v_max_m_s == pytest.approx(0.15)
    assert cfg.collision.d_safe == pytest.approx(0.01)
    assert cfg.collision.d_activate == pytest.approx(0.04)
    assert cfg.qmeas_filter == "raw"
    assert cfg.qp_geometry_source == "cmd"
    assert cfg.gil_switch_interval_ms == pytest.approx(0.5)
    assert not hasattr(cfg, "generic_qpik")


def test_qp_geometry_source_rejects_unknown():
    raw = deepcopy(_raw())
    raw["inner"]["qp_geometry_source"] = "jacobian"
    with pytest.raises(ValueError, match="qp_geometry_source"):
        build_joint_ik_config(raw)


def test_retired_task_group_and_solver_keys_fail_fast():
    raw = deepcopy(_raw())
    raw["qpik"]["scalable_tasks"] = [{"name": "motion"}]
    with pytest.raises(ValueError, match="retired multi-level QPIK"):
        build_joint_ik_config(raw)

    raw = deepcopy(_raw())
    raw["qpik"]["solver"] = {"max_scalable_groups": 4}
    with pytest.raises(ValueError, match="max_scalable_groups"):
        build_joint_ik_config(raw)


def test_retired_escape_and_retry_keys_fail_fast():
    raw = deepcopy(_raw())
    raw["inner"]["a_max_rail_escape_m_s2"] = 0.8
    with pytest.raises(ValueError, match="a_max_rail_escape_m_s2"):
        build_joint_ik_config(raw)

    raw = deepcopy(_raw())
    raw["qpik"]["solver"] = {"regularization_retry": 1.0e-5}
    with pytest.raises(ValueError, match="regularization_retry"):
        build_joint_ik_config(raw)


def test_generic_call_shapes_drop_episode_plumbing():
    step_params = inspect.signature(QpIkController.step).parameters
    update_params = inspect.signature(JointIkController.update).parameters
    assert not {
        "rail_escape_active",
        "rail_escape_sign",
        "rail_escape_stop",
        "rail_escape_v_min_m_s",
        "rail_escape_v_max_m_s",
        "rail_escape_accel_m_s2",
    } & (set(step_params) | set(update_params))
    assert "task_profile" not in update_params
    assert "posture_guide" not in update_params


@pytest.mark.parametrize(
    "key,value",
    [("soft_min_m", 0.0), ("soft_max_m", 0.81), ("soft_min_m", float("nan"))],
)
def test_invalid_soft_bounds_still_fail_closed(key: str, value: float):
    raw = _raw()
    raw["qpik"]["hard_limits"]["rail"][key] = value
    with pytest.raises(ValueError, match="soft|finite|rail limits"):
        build_joint_ik_config(raw)
