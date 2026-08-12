"""Configuration contract for the generic two-level QPIK path.

The old weighted-QP/rail-extension episode knobs were intentionally removed.
These tests guard the replacement schema and make sure retired keys cannot
silently become runtime state again.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path

import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.generic_runtime import GenericQpikRuntime
from rm75_control.control.joint_admittance_8dof.loop import JointIkController


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"


def _raw() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def test_production_config_declares_generic_solver_and_canonical_soft_band():
    cfg = build_joint_ik_config(_raw())
    assert cfg.generic_qpik.solver.backend.lower() == "proxqp"
    assert cfg.generic_qpik.solver.max_rows == 128
    assert cfg.rail.soft_min_m == pytest.approx(0.01)
    assert cfg.rail.soft_max_m == pytest.approx(0.78)
    assert cfg.rail.v_max_m_s == pytest.approx(0.30)
    assert not hasattr(cfg, "qp")
    assert not hasattr(cfg, "nullspace")
    assert not hasattr(cfg, "rail_extension")


def test_retired_episode_keys_are_ignored_and_do_not_create_runtime_fields():
    raw = deepcopy(_raw())
    raw["inner"]["a_max_rail_escape_m_s2"] = 0.8
    raw["qpik"]["solver"].update(
        sigma_escape_enter=0.10,
        sigma_escape_exit=0.12,
        rail_escape_v_min_m_s=0.01,
        rail_escape_v_max_m_s=0.02,
    )
    raw["qpik"]["scalable_tasks"][0].update(
        weight_hard_max=4.5,
        task_weight_max_frac=0.2,
        escape_max_travel_m=0.08,
    )
    cfg = build_joint_ik_config(raw)
    assert not hasattr(cfg, "a_max_rail_escape_m_s2")
    assert not hasattr(cfg.generic_qpik.solver, "sigma_escape_enter")
    assert not hasattr(cfg.generic_qpik.solver, "rail_escape_v_max_m_s")
    group = cfg.generic_qpik.task_profile.scalable_groups[0]
    assert not hasattr(group, "escape_max_travel_m")
    assert not hasattr(group, "weight_hard_max")


def test_generic_call_shapes_drop_episode_plumbing():
    runtime_params = inspect.signature(GenericQpikRuntime.solve).parameters
    update_params = inspect.signature(JointIkController.update).parameters
    assert not {
        "rail_escape_active",
        "rail_escape_sign",
        "rail_escape_stop",
        "rail_escape_v_min_m_s",
        "rail_escape_v_max_m_s",
        "rail_escape_accel_m_s2",
    } & (set(runtime_params) | set(update_params))
    assert {"task_profile", "posture_guide"} <= set(update_params)


@pytest.mark.parametrize(
    "key,value",
    [("soft_min_m", 0.0), ("soft_max_m", 0.81), ("soft_min_m", float("nan"))],
)
def test_invalid_soft_bounds_still_fail_closed(key: str, value: float):
    raw = _raw()
    raw["qpik"]["hard_limits"]["rail"][key] = value
    with pytest.raises(ValueError, match="soft|finite|rail limits"):
        build_joint_ik_config(raw)
