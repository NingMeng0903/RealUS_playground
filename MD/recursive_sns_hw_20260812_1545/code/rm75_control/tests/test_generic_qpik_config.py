"""Configuration contract for the generic two-level QPIK adapter."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"


def _raw() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_production_yaml_declares_generic_profile_and_hard_limits():
    cfg = build_joint_ik_config(_raw())
    generic = cfg.generic_qpik

    assert generic.solver.backend == "proxqp"
    assert generic.solver.max_iter == 80
    assert generic.solver.max_solve_ms == pytest.approx(3.0)
    assert generic.solver.max_rows == 128
    assert generic.solver.max_scalable_groups == 4
    assert generic.dexterity_d_safe == pytest.approx(0.04)
    assert generic.dexterity_d_activate == pytest.approx(0.08)
    assert generic.dexterity_gamma == pytest.approx(5.0)
    assert generic.dexterity_k_d == pytest.approx(0.15)
    assert generic.working_arm_margin_rad == pytest.approx(0.30)
    assert generic.working_rail_margin_m == pytest.approx(0.02)
    assert generic.working_gamma == pytest.approx(8.0)
    assert generic.solver.comfort_k_g == pytest.approx(2.0)
    assert generic.solver.rail_handoff_weight == pytest.approx(0.0)
    assert generic.solver.margin_weight == pytest.approx(5.0e-4)
    np.testing.assert_array_equal(
        np.argmax(generic.task_profile.protected_selection, axis=1), [2, 3, 4, 5]
    )
    assert generic.task_profile.protected_selection.shape == (4, 6)
    assert [g.group_id for g in generic.task_profile.scalable_groups] == ["motion"]
    np.testing.assert_array_equal(
        generic.task_profile.scalable_groups[0].selection,
        np.eye(6)[:2],
    )
    assert generic.overforce_task_row == 2
    assert generic.health.joint_danger_deg == pytest.approx(15.0)
    assert generic.health.joint_exit_deg == pytest.approx(25.0)
    assert cfg.collision.d_safe == pytest.approx(0.01)
    assert cfg.collision.d_activate == pytest.approx(0.04)
    assert cfg.limit_damper_band_rad == pytest.approx(0.15)
    assert cfg.limit_damper_band_rail_m == pytest.approx(0.02)
    assert cfg.rail.soft_min_m == pytest.approx(0.01)
    assert cfg.rail.soft_max_m == pytest.approx(0.78)
    assert not hasattr(cfg, "qp")
    assert not hasattr(cfg, "nullspace")
    assert not hasattr(cfg, "rail_extension")


def test_explicit_matrices_and_multiple_groups_are_preserved():
    raw = {
        "qpik": {
            "solver": {
                "backend": "scipy",
                "max_iter": 31,
                "max_rows": 40,
                "max_scalable_groups": 3,
            },
            "protected_task": {
                "selection": [[0.5, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]],
                "row_scales": [0.2, 0.7],
                "residual_limits": [0.01, 0.02],
            },
            "scalable_tasks": [
                {
                    "group_id": "position",
                    "selection": [[0, 1, 0, 0, 0, 0]],
                    "row_scales": [0.1],
                    "slack_limits": [[-0.01, 0.02]],
                },
                {"group_id": "attitude", "rows": [4, 5]},
            ],
            "health": {
                "arm": {"warn": 0.2, "danger": 0.1, "exit": 0.3},
                "joint_margin": {"danger_deg": 10, "warn_deg": 12, "exit_deg": 18},
                "wrist_margin": {"danger_deg": 11, "warn_deg": 15, "exit_deg": 22},
            },
            "indices": {"rail": [0], "wrist": [4, 5]},
        }
    }
    cfg = build_joint_ik_config(raw)
    generic = cfg.generic_qpik
    assert generic.solver.backend == "scipy"
    assert generic.solver.max_iter == 31
    assert generic.solver.max_rows == 40
    assert generic.solver.max_scalable_groups == 3
    np.testing.assert_allclose(
        generic.task_profile.protected_selection,
        [[0.5, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]],
    )
    assert [g.group_id for g in generic.task_profile.scalable_groups] == [
        "position",
        "attitude",
    ]
    np.testing.assert_allclose(
        generic.task_profile.scalable_groups[0].slack_limits, [[-0.01, 0.02]]
    )
    np.testing.assert_allclose(
        generic.task_profile.scalable_groups[1].selection,
        [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]],
    )
    assert generic.health.arm_danger == pytest.approx(0.1)
    assert generic.health.wrist_exit_deg == pytest.approx(22)
    assert generic.rail_indices == (0,)
    assert generic.wrist_indices == (4, 5)


def test_missing_qpik_keeps_compatibility_profile_all_protected():
    cfg = build_joint_ik_config({})
    generic = cfg.generic_qpik
    assert generic.solver.backend == "proxqp"
    np.testing.assert_allclose(generic.task_profile.protected_selection, np.eye(6))
    assert generic.task_profile.scalable_groups == ()
    assert generic.overforce_task_row is None


@pytest.mark.parametrize("backend", ["osqp", "unknown"])
def test_generic_backend_must_be_explicit_and_supported(backend):
    raw = {"qpik": {"solver": {"backend": backend}}}
    with pytest.raises(ValueError, match="backend"):
        build_joint_ik_config(raw)


def test_selection_indices_and_matrix_validation_is_fail_closed():
    raw = {"qpik": {"protected_task": {"rows": [0, 6]}}}
    with pytest.raises(ValueError, match="row|selection"):
        build_joint_ik_config(raw)

    raw = {
        "qpik": {
            "scalable_tasks": [
                {"group_id": "bad", "selection": [[1, 0, 0, 0, 0]]}
            ]
        }
    }
    with pytest.raises(ValueError, match="shape|selection"):
        build_joint_ik_config(raw)
