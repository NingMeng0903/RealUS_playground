"""Production configuration contract after restoring the stateless 4d path."""

from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path

import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpIkController
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import RailExtensionTask


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"


def _raw() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def test_production_config_keeps_canonical_soft_band_and_proxqp():
    cfg = build_joint_ik_config(_raw())
    assert cfg.qp.backend.lower() == "proxqp"
    assert cfg.qp.rail_soft_min_m == pytest.approx(0.01)
    assert cfg.qp.rail_soft_max_m == pytest.approx(0.78)
    assert cfg.rail.soft_min_m == pytest.approx(0.01)
    assert cfg.rail.soft_max_m == pytest.approx(0.78)
    assert cfg.rail_extension.v_max_m_s == pytest.approx(0.08)


def test_episode_keys_are_ignored_and_do_not_create_runtime_fields():
    raw = deepcopy(_raw())
    raw["inner"]["a_max_rail_escape_m_s2"] = 0.8
    raw["inner"]["qp"].update(
        sigma_escape_enter=0.10,
        sigma_escape_exit=0.12,
        rail_escape_v_min_m_s=0.01,
        rail_escape_v_max_m_s=0.02,
    )
    raw["inner"]["rail_extension"].update(
        weight_hard_max=4.5,
        task_weight_max_frac=0.2,
        escape_max_travel_m=0.08,
    )
    cfg = build_joint_ik_config(raw)
    assert not hasattr(cfg, "a_max_rail_escape_m_s2")
    assert not hasattr(cfg.qp, "sigma_escape_enter")
    assert not hasattr(cfg.qp, "rail_escape_v_max_m_s")
    assert not hasattr(cfg.rail_extension, "escape_max_travel_m")
    assert not hasattr(cfg.rail_extension, "weight_hard_max")


def test_restored_call_shapes_drop_episode_plumbing():
    rail_params = inspect.signature(RailExtensionTask.__call__).parameters
    assert {"sigma_scale", "sigma_grad_rail", "vel_ff", "dt_s"} <= set(
        rail_params
    )

    qp_params = inspect.signature(QpIkController.step).parameters
    assert "rail_task_vel_m_s" in qp_params
    assert "rail_task_weight" in qp_params
    assert not {
        "rail_escape_active",
        "rail_escape_sign",
        "rail_escape_stop",
        "rail_escape_v_min_m_s",
        "rail_escape_v_max_m_s",
        "rail_escape_accel_m_s2",
    } & set(qp_params)


@pytest.mark.parametrize(
    "key,value",
    [("soft_min_m", 0.0), ("soft_max_m", 0.81), ("soft_min_m", float("nan"))],
)
def test_invalid_soft_bounds_still_fail_closed(key: str, value: float):
    raw = _raw()
    raw["inner"]["rail"][key] = value
    with pytest.raises(ValueError, match="soft[- ]limit|finite"):
        build_joint_ik_config(raw)
