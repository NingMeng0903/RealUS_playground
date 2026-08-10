"""Deterministic startup validation for the Stage-1 escape envelope."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config


_PRODUCTION = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"


def _raw() -> dict:
    """Return a small, independent loader input for each test."""

    return {
        "inner": {
            "qp": {},
            "rail": {"travel_m": 0.80, "soft_min_m": 0.01, "soft_max_m": 0.78},
            "rail_extension": {},
        }
    }


def _set(raw: dict, section: str, key: str, value: object) -> dict:
    out = deepcopy(raw)
    out["inner"][section][key] = value
    return out


def test_production_escape_values_are_shared_by_qp_and_rail_extension():
    raw = yaml.safe_load(_PRODUCTION.read_text(encoding="utf-8")) or {}
    cfg = build_joint_ik_config(raw)

    assert 0.0 < cfg.qp.sigma_escape_enter
    assert (
        cfg.qp.sigma_escape_enter
        <= cfg.qp.sigma_limit_escape_enter
        <= cfg.qp.sigma_escape_exit
    )
    assert cfg.qp.rail_escape_v_min_m_s == pytest.approx(
        cfg.rail_extension.escape_v_min_m_s
    )
    assert cfg.qp.rail_escape_v_max_m_s == pytest.approx(
        cfg.rail_extension.escape_v_max_m_s
    )
    assert cfg.qp.rail_task_weight_hard_max == pytest.approx(
        cfg.rail_extension.weight_hard_max
    )
    assert cfg.qp.rail_task_weight_max_frac == pytest.approx(
        cfg.rail_extension.task_weight_max_frac
    )
    assert cfg.a_max_rail_escape_m_s2 == pytest.approx(0.80)
    assert cfg.qp.rail_escape_accel_m_s2 == pytest.approx(
        cfg.a_max_rail_escape_m_s2
    )
    assert cfg.qp.rail_soft_min_m == pytest.approx(cfg.rail.soft_min_m)
    assert cfg.qp.rail_soft_max_m == pytest.approx(cfg.rail.soft_max_m)


@pytest.mark.parametrize(
    "key,value",
    [
        ("sigma_escape_enter", 0.0),
        ("sigma_escape_enter", 0.13),
        ("sigma_limit_escape_enter", 0.09),
        ("sigma_limit_escape_enter", 0.13),
        ("sigma_escape_exit", 0.11),
        ("sigma_escape_enter", float("nan")),
    ],
)
def test_invalid_escape_threshold_order_fails_at_load(key: str, value: float):
    raw = _set(_raw(), "qp", key, value)
    # The parametrized cases either violate the ordering directly or provide
    # a non-finite threshold; both must be rejected before a controller starts.
    with pytest.raises(ValueError, match="threshold|finite"):
        build_joint_ik_config(raw)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("rail_extension", "escape_v_min_m_s", -1.0e-6),
        ("rail_extension", "escape_v_max_m_s", -1.0e-6),
        ("rail_extension", "escape_v_min_m_s", 0.03),
        ("rail_extension", "escape_v_max_m_s", float("nan")),
    ],
)
def test_invalid_escape_speed_envelope_fails_at_load(
    section: str, key: str, value: float
):
    raw = _set(_raw(), section, key, value)
    if key == "escape_v_min_m_s" and value == 0.03:
        raw["inner"]["rail_extension"]["escape_v_max_m_s"] = 0.02
    with pytest.raises(ValueError, match="velocity|finite"):
        build_joint_ik_config(raw)


def test_qp_escape_speed_alias_must_match_rail_extension():
    raw = _raw()
    raw["inner"]["qp"].update(
        rail_escape_v_min_m_s=0.01,
        rail_escape_v_max_m_s=0.02,
    )
    raw["inner"]["rail_extension"].update(
        escape_v_min_m_s=0.012,
        escape_v_max_m_s=0.02,
    )
    with pytest.raises(ValueError, match="escape_v_min_m_s|mismatch"):
        build_joint_ik_config(raw)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("rail_extension", "escape_max_travel_m", -1.0e-9),
        ("rail_extension", "escape_max_travel_m", 0.78),
        ("rail_extension", "escape_max_travel_m", float("nan")),
    ],
)
def test_escape_travel_is_nonnegative_and_inside_soft_band(
    section: str, key: str, value: float
):
    raw = _set(_raw(), section, key, value)
    with pytest.raises(ValueError, match="escape_max_travel|finite"):
        build_joint_ik_config(raw)


@pytest.mark.parametrize(
    "key,value",
    [
        ("soft_min_m", -1.0e-6),
        ("soft_max_m", 0.0),
        ("soft_max_m", 0.01),
        ("soft_max_m", 0.81),
        ("soft_min_m", float("nan")),
    ],
)
def test_invalid_soft_bounds_fail_at_load(key: str, value: float):
    raw = _set(_raw(), "rail", key, value)
    with pytest.raises(ValueError, match="soft limits|finite"):
        build_joint_ik_config(raw)


def test_qp_only_escape_alias_is_propagated_to_rail_extension():
    raw = _raw()
    raw["inner"]["qp"].update(
        rail_escape_v_min_m_s=0.0,
        rail_escape_v_max_m_s=0.015,
    )
    cfg = build_joint_ik_config(raw)
    assert cfg.rail_extension.escape_v_min_m_s == pytest.approx(0.0)
    assert cfg.rail_extension.escape_v_max_m_s == pytest.approx(0.015)


@pytest.mark.parametrize(
    "qp_key,rail_key,qp_value,rail_value",
    [
        ("rail_task_weight_hard_max", "weight_hard_max", 4.5, 4.0),
        ("rail_task_weight_max_frac", "task_weight_max_frac", 0.8, 0.7),
    ],
)
def test_qp_and_rail_weight_aliases_must_match(
    qp_key: str,
    rail_key: str,
    qp_value: float,
    rail_value: float,
):
    raw = _raw()
    raw["inner"]["qp"][qp_key] = qp_value
    raw["inner"]["rail_extension"][rail_key] = rail_value
    with pytest.raises(ValueError, match="weight.*mismatch|mismatch"):
        build_joint_ik_config(raw)


@pytest.mark.parametrize("value", [-1.0e-9, float("nan"), float("inf")])
def test_invalid_escape_acceleration_fails_at_load(value: float):
    raw = deepcopy(_raw())
    raw["inner"]["a_max_rail_escape_m_s2"] = value
    with pytest.raises(ValueError, match="escape.*accel|finite|non-negative"):
        build_joint_ik_config(raw)


def test_qp_escape_acceleration_alias_must_match_canonical_value():
    raw = deepcopy(_raw())
    raw["inner"]["a_max_rail_escape_m_s2"] = 0.80
    raw["inner"]["qp"]["rail_escape_accel_m_s2"] = 0.60
    with pytest.raises(ValueError, match="acceleration mismatch"):
        build_joint_ik_config(raw)


@pytest.mark.parametrize(
    "section,key,value,match",
    [
        ("rail_extension", "weight_hard_max", -0.1, "hard weight"),
        ("rail_extension", "task_weight_max_frac", -0.1, "fraction"),
        ("rail_extension", "task_weight_max_frac", 1.1, "fraction"),
        ("qp", "limit_escape_activation", -0.1, "activation"),
        ("qp", "limit_escape_activation", 1.1, "activation"),
        ("qp", "limit_escape_activation", float("nan"), "finite"),
    ],
)
def test_invalid_escape_hierarchy_values_fail_at_load(
    section: str, key: str, value: float, match: str
):
    with pytest.raises(ValueError, match=match):
        build_joint_ik_config(_set(_raw(), section, key, value))
