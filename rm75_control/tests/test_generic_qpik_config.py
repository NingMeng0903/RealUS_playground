"""Configuration contract for the fixed RM75 single-shot QPIK."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"


def _raw() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_production_yaml_declares_fixed_single_qp_and_hard_limits():
    cfg = build_joint_ik_config(_raw())
    generic = cfg.generic_qpik

    assert generic.solver.backend == "proxqp"
    assert generic.solver.max_iter == 20
    assert generic.solver.max_iter_in == 10
    assert generic.solver.max_solve_ms == pytest.approx(3.0)
    assert generic.solver.feasibility_tolerance == pytest.approx(1.0e-5)
    np.testing.assert_allclose(
        generic.solver.protected_limits, [0.010, 0.050, 0.050, 0.050]
    )
    assert not hasattr(generic, "task_profile")
    assert generic.dexterity_d_safe == pytest.approx(0.04)
    assert generic.dexterity_d_activate == pytest.approx(0.08)
    assert generic.working_arm_margin_rad == pytest.approx(0.30)
    assert generic.rail_macro_tau_s == pytest.approx(0.15)
    assert generic.risk_attack_s == pytest.approx(0.05)
    assert generic.wrist_danger_deg == pytest.approx(10.0)
    assert cfg.collision.d_safe == pytest.approx(0.01)
    assert cfg.collision.d_activate == pytest.approx(0.04)
    assert cfg.limit_damper_band_rad == pytest.approx(0.15)
    assert cfg.limit_damper_band_rail_m == pytest.approx(0.02)
    assert cfg.rail.soft_min_m == pytest.approx(0.005)
    assert cfg.rail.soft_max_m == pytest.approx(0.78)


@pytest.mark.parametrize(
    "retired",
    [
        {"protected_task": {"rows": [2, 3, 4, 5]}},
        {"scalable_tasks": [{"group_id": "motion", "rows": [0, 1]}]},
        {"compatibility": {"overforce_task_row": 2}},
        {"solver": {"max_rows": 128}},
        {"solver": {"max_scalable_groups": 4}},
        {"solver": {"regularization": 1.0e-6}},
    ],
)
def test_retired_multilevel_configuration_is_rejected(retired):
    raw = {"qpik": retired}
    with pytest.raises(ValueError, match="retired multi-level QPIK"):
        build_joint_ik_config(raw)


def test_minimal_fixed_configuration_uses_documented_defaults():
    cfg = build_joint_ik_config({"qpik": {"solver": {"backend": "scipy"}}})
    generic = cfg.generic_qpik
    assert generic.solver.max_iter == 20
    assert generic.solver.max_iter_in == 10
    assert generic.rail_indices == (0,)
    assert generic.wrist_indices == (5, 6, 7)


@pytest.mark.parametrize("backend", ["osqp", "unknown"])
def test_backend_must_be_explicit_and_supported(backend):
    with pytest.raises(ValueError, match="backend"):
        build_joint_ik_config({"qpik": {"solver": {"backend": backend}}})


@pytest.mark.parametrize(
    "section,key",
    [
        ("solver", "totally_unknown_legacy_knob"),
        ("dexterity", "projector_damping"),
        ("working_set", "silent_override"),
        ("whole_body", "escape_controller"),
        ("hard_limits", "collapse_to_zero"),
    ],
)
def test_unknown_qpik_keys_fail_fast(section, key):
    raw = {"qpik": {section: {key: 1.0}}}
    with pytest.raises(ValueError, match=key):
        build_joint_ik_config(raw)
