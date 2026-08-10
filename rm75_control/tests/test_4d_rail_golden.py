"""Frozen 4d15c1d rail-task and ProxQP allocation results."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig, QpIkController
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import (
    RailExtensionConfig,
    RailExtensionTask,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


class _LinearRailKinematics:
    """Small exact model: TCP Y is rail Y plus a fixed 0.30 m extension."""

    q_lower = np.array([0.0] + [-np.pi] * 7)
    q_upper = np.array([0.80] + [np.pi] * 7)

    @staticmethod
    def fk_placement(q):
        return SimpleNamespace(
            translation=np.array([0.0, float(q[0]) + 0.30, 0.0])
        )

    @staticmethod
    def jacobian(_q):
        jac = np.zeros((6, 8))
        jac[1, 0] = 1.0
        return jac


def _golden_task() -> RailExtensionTask:
    cfg = RailExtensionConfig(
        k_ext=2.0,
        k_ff=1.0,
        v_ff_thr_m_s=0.005,
        v_ff_span_m_s=0.015,
        e0_m=0.02,
        e1_m=0.08,
        w_max=2.0,
        soft_min_m=0.01,
        soft_max_m=0.78,
        v_max_m_s=0.08,
        limit_margin_m=0.08,
        k_sigma_boost=2.0,
        k_esc=0.5,
        w_sigma_floor=1.0,
        k_pose=2.0,
        pose_e0_m=0.005,
        pose_e1_m=0.04,
        pose_w_max=4.0,
        sigma_guard_enter=0.45,
        sigma_guard_exit=0.70,
        v_guard_max_m_s=0.04,
        v_lpf_tau_s=0.12,
    )
    return RailExtensionTask(_LinearRailKinematics(), cfg)


def test_reach_and_pose_attract_match_frozen_4d_values():
    q = np.zeros(8)
    q[0] = 0.40
    task = _golden_task()
    task.reset(q)
    task.d_pref_m = 0.24  # extension=.30 -> +.06 m reach error
    vel_ff = np.zeros(6)
    vel_ff[1] = 0.03

    v_reach, w_reach = task(
        q,
        sigma_scale=0.5,
        sigma_grad_rail=0.02,
        vel_ff=vel_ff,
    )
    assert v_reach == pytest.approx(0.08, abs=1e-12)
    assert w_reach == pytest.approx(5.444444444444445, abs=1e-12)

    task.set_mode("pose_attract")
    task.set_rail_pose_target(0.46)
    v_pose, w_pose = task(q, sigma_scale=0.4, sigma_grad_rail=0.02)
    assert v_pose == pytest.approx(0.08, abs=1e-12)
    assert w_pose == pytest.approx(4.6, abs=1e-12)


def test_gradient_reversal_and_positive_soft_bound_match_frozen_4d_values():
    q = np.zeros(8)
    q[0] = 0.40
    task = _golden_task()
    task.reset(q)

    v_pos, w_pos = task(q, sigma_scale=0.0, sigma_grad_rail=0.05)
    v_neg, w_neg = task(q, sigma_scale=0.0, sigma_grad_rail=-0.05)
    assert (v_pos, w_pos) == pytest.approx((0.025, 3.0), abs=1e-12)
    assert (v_neg, w_neg) == pytest.approx((-0.025, 3.0), abs=1e-12)

    q[0] = 0.76
    v_fade, w_fade = task(q, sigma_scale=0.0, sigma_grad_rail=0.05)
    assert (v_fade, w_fade) == pytest.approx((0.00390625, 0.46875), abs=1e-12)
    v_inward, w_inward = task(q, sigma_scale=0.0, sigma_grad_rail=-0.05)
    assert (v_inward, w_inward) == pytest.approx((-0.025, 3.0), abs=1e-12)

    q[0] = 0.78
    assert task(q, sigma_scale=0.0, sigma_grad_rail=0.05) == (0.0, 0.0)


def test_production_proxqp_rail_allocation_matches_frozen_result():
    kin = RobotKinematics()
    accel = np.full(8, 20.0)
    accel[0] = 0.30
    limits = SafetyLimits.from_kinematics(
        kin,
        v_scale=0.5,
        a_max=accel,
        position_margin=np.zeros(8),
    )
    cfg = QpConfig(
        backend="proxqp",
        collision=CollisionConfig(enabled=False),
        task_weight=np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]),
        reg=np.array([1e-3, 1e-2, 1e-2, 1e-2, 1e-2, 5e-3, 5e-3, 5e-3]),
        use_mass_weighted_reg=False,
        use_dyn_nullspace=False,
        task_weight_lpf_tau_s=0.0,
    )
    ctrl = QpIkController(kin, limits, cfg)
    assert "proxqp" in ctrl.backend_name and "osqp" not in ctrl.backend_name
    q = np.array(
        [0.40, -0.905938, 1.117987, 0.459109, 1.775407, -0.342094, 1.06775, 0.749873]
    )
    ctrl.reset(q)
    ctrl.qdot_prev[0] = 0.018
    result = ctrl.step(
        q,
        np.zeros(6),
        0.005,
        rail_task_vel_m_s=0.02,
        rail_task_weight=6.0,
    )
    assert result.rail_task_weight_effective == 6.0
    assert result.qdot[0] == pytest.approx(0.019205311544, abs=2e-8)
    assert result.slack_norm == pytest.approx(5.516406257e-05, abs=2e-8)
