"""Config-agnostic QPIK preference tests: set-based σ, branch barrier, rail latch."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.solver.branch_barrier import (
    BranchBarrierBuilder,
    BranchBarrierConfig,
    latch_q_star_signs,
)
from rm75_control.control.joint_admittance_8dof.solver.joint_comfort import (
    COMFORT_SLACK0,
    JointComfortBuilder,
    JointComfortConfig,
)
from rm75_control.control.joint_admittance_8dof.solver.sigma_setbased import (
    SigmaSetBasedConfig,
    SigmaSetBasedTracker,
)
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


def test_latch_q_star_signs_preserves_magnitude() -> None:
    qn = np.array([0.0, 0.0, -0.7, 0.0, 1.57, 0.0, 0.78, 0.0])
    qm = np.array([0.2, 0.1, -0.5, 0.0, -1.2, 0.1, -0.4, 0.0])
    out = latch_q_star_signs(qn, qm)
    assert np.isclose(out[4], -1.57)
    assert np.isclose(out[6], -0.78)
    assert np.isclose(out[2], -0.7)


def test_sigma_setbased_hysteresis() -> None:
    tr = SigmaSetBasedTracker(
        SigmaSetBasedConfig(activate=0.10, exit=0.14, safe=0.05, enabled=True)
    )
    assert tr.update_hysteresis(0.20) is False
    assert tr.update_hysteresis(0.09) is True
    assert tr.update_hysteresis(0.12) is True  # still active below exit
    assert tr.update_hysteresis(0.15) is False


def test_branch_barrier_tightens_box_against_zero() -> None:
    bb = BranchBarrierBuilder(
        BranchBarrierConfig(activate_rad=0.52, box_activate_rad=0.87, eps_rad=0.35)
    )
    q_star = np.array([0.0, -1.58, -1.63, 1.15, 1.82, 1.65, 1.05, 1.46])
    q = q_star.copy()
    q[4] = np.deg2rad(22.0)
    lo = -np.ones(8)
    hi = np.ones(8)
    lo2, hi2 = bb.tighten_box(lo, hi, q, q_star, np.ones(8))
    assert lo2[4] > -0.2
    assert hi2[4] == pytest.approx(1.0)


def test_tighten_box_blocks_j4_toward_stop_when_travel_open() -> None:
    from rm75_control.kinematics.srs_ik import Q_LOWER, Q_UPPER

    bb = BranchBarrierBuilder(
        BranchBarrierConfig(activate_rad=0.52, box_activate_rad=0.87, eps_rad=0.35)
    )
    q_star = np.array([0.0, -1.56, -1.65, 1.14, 1.68, 1.56, 1.06, 1.65])
    q = q_star.copy()
    lo = -np.ones(8)
    hi = np.ones(8)
    q_lo = np.concatenate([[0.0], Q_LOWER])
    q_hi = np.concatenate([[0.8], Q_UPPER])
    q[4] = np.deg2rad(115.0)
    _lo115, hi115 = bb.tighten_box(
        lo,
        hi,
        q,
        q_star,
        np.ones(8),
        rail_open_travel=True,
        q_lower=q_lo,
        q_upper=q_hi,
    )
    assert hi115[4] > 0.3
    q[4] = np.deg2rad(130.0)
    lo2, hi2 = bb.tighten_box(
        lo,
        hi,
        q,
        q_star,
        np.ones(8),
        rail_open_travel=True,
        q_lower=q_lo,
        q_upper=q_hi,
    )
    assert hi2[4] < 0.15
    lo3, hi3 = bb.tighten_box(
        lo,
        hi,
        q,
        q_star,
        np.ones(8),
        rail_open_travel=False,
        q_lower=q_lo,
        q_upper=q_hi,
    )
    assert hi3[4] == pytest.approx(1.0)


def test_tighten_box_blocks_j1_overfold_not_startup_fold() -> None:
    bb = BranchBarrierBuilder(
        BranchBarrierConfig(activate_rad=0.52, box_activate_rad=0.87, eps_rad=0.35)
    )
    q_star = np.array([0.0, np.deg2rad(-89.5), -1.65, 1.14, 1.68, 1.56, 1.06, 1.65])
    lo = -np.ones(8)
    hi = np.ones(8)
    q0 = q_star.copy()
    q0[1] = 0.0
    _lo0, hi0 = bb.tighten_box(lo, hi, q0, q_star, np.ones(8))
    assert hi0[1] < 0.15
    assert _lo0[1] == pytest.approx(-1.0)
    q90 = q_star.copy()
    q90[1] = np.deg2rad(-90.0)
    lo90, hi90 = bb.tighten_box(lo, hi, q90, q_star, np.ones(8))
    assert lo90[1] < -0.3
    assert hi90[1] == pytest.approx(1.0)
    q120 = q_star.copy()
    q120[1] = np.deg2rad(-120.0)
    lo120, _hi120 = bb.tighten_box(lo, hi, q120, q_star, np.ones(8))
    assert lo120[1] < -0.3
    q140 = q_star.copy()
    q140[1] = np.deg2rad(-140.0)
    lo140, _hi140 = bb.tighten_box(lo, hi, q140, q_star, np.ones(8))
    assert lo140[1] > -0.15


def test_preferred_escape_sign_follows_unload() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            soft_min_m=0.10,
            soft_max_m=0.70,
            pin_margin_m=0.008,
            escape_leave_m=0.04,
            escape_sign_policy="minus",
        ),
    )
    assert task._preferred_escape_sign(0.40) == pytest.approx(-1.0)
    assert task._preferred_escape_sign(0.40, unload_sign=1.0) == pytest.approx(1.0)
    assert task._preferred_escape_sign(0.40, unload_sign=-1.0) == pytest.approx(-1.0)


def test_branch_barrier_start_j1_still_allows_fold() -> None:
    bb = BranchBarrierBuilder(
        BranchBarrierConfig(activate_rad=0.52, box_activate_rad=0.87, eps_rad=0.35)
    )
    q_star = np.array([0.0, -1.58, -1.63, 1.15, 1.82, 1.65, 1.05, 1.46])
    q = np.array([0.31, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0])
    lo = -np.ones(8)
    hi = np.ones(8)
    lo2, hi2 = bb.tighten_box(lo, hi, q, q_star, np.ones(8))
    assert hi2[1] <= 1.0e-12
    assert lo2[1] < -0.5


def test_branch_barrier_wrong_side_stays_active() -> None:
    bb = BranchBarrierBuilder(BranchBarrierConfig(activate_rad=0.52, eps_rad=0.35))
    q_star = np.array([0.0, -1.58, -1.63, 1.15, 1.82, 1.65, 1.05, 1.46])
    q = q_star.copy()
    q[4] = np.deg2rad(-45.0)
    rows = bb.build_rows(q, q_star)
    assert rows.active
    j4_rows = [k for k in range(rows.jacobian.shape[0]) if abs(rows.jacobian[k, 4]) > 0.5]
    assert j4_rows
    assert rows.jacobian[j4_rows[0], 4] > 0.0
    assert rows.lower[j4_rows[0]] > 0.0


def test_branch_barrier_blocks_zero_crossing() -> None:
    bb = BranchBarrierBuilder(
        BranchBarrierConfig(activate_rad=0.5, eps_rad=0.05, gamma=6.0)
    )
    q_star = np.array([0.0, 0.0, 0.0, 0.0, 1.57, 0.0, 0.78, 0.0])
    # Inside eps band on the +side of 0 → rhs > 0 forbids crossing toward −.
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.02, 0.0])
    rows = bb.build_rows(q, q_star)
    assert rows.active
    assert rows.jacobian.shape[0] >= 1
    j4_rows = [k for k in range(rows.jacobian.shape[0]) if abs(rows.jacobian[k, 4]) > 0.5]
    assert j4_rows
    k = j4_rows[0]
    assert rows.jacobian[k, 4] > 0.0
    assert rows.lower[k] > 0.0


def test_branch_barrier_requires_open_at_legacy_j6_floor() -> None:
    """2.8° used to sit on eps=0.05; 15° eps must demand a positive qdot."""
    bb = BranchBarrierBuilder(
        BranchBarrierConfig(activate_rad=0.35, eps_rad=0.26, gamma=6.0)
    )
    q_star = np.array([0.0, 0.0, 0.0, 0.0, 1.57, 0.0, 0.78, 0.0])
    q = np.zeros(8)
    q[6] = np.deg2rad(2.8)
    rows = bb.build_rows(q, q_star)
    assert rows.active
    j6_rows = [
        k for k in range(rows.jacobian.shape[0]) if abs(rows.jacobian[k, 6]) > 0.5
    ]
    assert j6_rows
    k = j6_rows[0]
    assert rows.jacobian[k, 6] > 0.0
    assert rows.lower[k] > 1.0


def test_sigma_fade_spares_j4_and_j6() -> None:
    kin = RobotKinematics()
    q_nom = np.array([0.0, 0.0, -0.7, 0.0, 1.57, 0.0, 0.78, 0.0])
    centering = JointCenteringTask.from_kinematics(
        kin, NullspaceTaskConfig(k_center=1.0, q_nominal_rad=q_nom)
    )
    composer = SecondaryComposer(centering, None, max_qdot_frac=0.0)
    q = q_nom.copy()
    q[2] = 0.0
    q[4] = 0.20
    q[6] = 0.05
    healthy = composer.compose(
        q, None, None, arm_suppressed=True, sigma_min=1.0, centering_sigma_fade=True
    )
    singular = composer.compose(
        q, None, None, arm_suppressed=True, sigma_min=0.02, centering_sigma_fade=True
    )
    assert singular[4] == pytest.approx(healthy[4])
    assert singular[6] == pytest.approx(healthy[6])
    assert abs(singular[2]) < abs(healthy[2]) - 1e-9


def test_near_straight_elbow_blocks_rail_escape() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_esc=1.0,
            k_ext=0.0,
            sigma_escape_enter=0.9,
            sigma_escape_exit=0.95,
            escape_enter_dwell_s=0.0,
            v_lpf_tau_s=0.0,
        ),
    )
    task.set_mode("reach")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    q[4] = np.deg2rad(22.0)
    task.capture_reference(q)
    v, _ = task(
        q,
        sigma_scale=0.2,
        sigma_grad_rail=2.0,
        vel_ff=None,
        dt_s=0.005,
        block_escape=True,
    )
    assert not task._escape_active
    assert abs(task.last_v_escape) < 1e-12
    assert abs(v) < 1e-6


def test_rail_escape_latches_sign_against_grad_flip() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_esc=1.0,
            v_ff_thr_m_s=0.01,
            v_lpf_tau_s=0.0,
            escape_sign_policy="auto",
        ),
    )
    task.set_mode("reach")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    task.capture_reference(q)
    v1, _ = task(
        q,
        sigma_scale=0.2,
        sigma_grad_rail=2.0,
        vel_ff=None,
        dt_s=0.005,
        press_escape_allowed=True,
    )
    v2, _ = task(
        q,
        sigma_scale=0.2,
        sigma_grad_rail=-2.0,
        vel_ff=None,
        dt_s=0.005,
        press_escape_allowed=True,
    )
    assert abs(v1) > 1e-6
    assert np.sign(v1) == np.sign(v2)
    assert np.sign(task._escape_sign) == np.sign(v1)


def test_rail_ff_gate_zeros_escape() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_esc=1.0,
            k_ff=1.0,
            v_ff_thr_m_s=0.005,
            v_lpf_tau_s=0.0,
            k_ext=0.0,
        ),
    )
    task.set_mode("reach")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    task.capture_reference(q)
    j_rail = kin.jacobian(q)[:3, 0]
    n = float(np.linalg.norm(j_rail))
    assert n > 1e-6
    vel_ff = np.zeros(6)
    vel_ff[:3] = 0.05 * (j_rail / n)
    v_esc_only, _ = task(
        q, sigma_scale=0.1, sigma_grad_rail=5.0, vel_ff=None, dt_s=0.005
    )
    assert not task._escape_active
    assert abs(task.last_v_escape) < 1e-12
    v_with_ff, _ = task(
        q, sigma_scale=0.1, sigma_grad_rail=5.0, vel_ff=vel_ff, dt_s=0.005
    )
    assert abs(v_esc_only) < 1e-12
    assert abs(v_with_ff) < 1e-12
    assert not task._escape_active
    assert abs(task.last_v_ff) > 1e-4


def test_secondary_composer_adds_manip_to_centering() -> None:
    kin = RobotKinematics()
    centering = JointCenteringTask.from_kinematics(
        kin,
        NullspaceTaskConfig(
            k_center=1.0,
            q_nominal_rad=np.array([0.0, 0.0, -0.7, 0.0, 1.57, 0.0, 0.78, 0.0]),
        ),
    )

    class _FakeManip:
        def __call__(self, q, sigma_min=1.0, exclude_rail=True, dt_s=None):
            del sigma_min, exclude_rail, dt_s
            out = np.zeros_like(q)
            out[2] = 0.5
            return out

    composer = SecondaryComposer(
        centering, None, manipulability=_FakeManip(), max_qdot_frac=0.0
    )
    q = centering.q_target + np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_center = composer.compose(
        q, None, None, arm_suppressed=True, manipulability_active=False
    )
    q_both = composer.compose(
        q, None, None, arm_suppressed=True, manipulability_active=True, sigma_min=0.05
    )
    assert abs(q_center[2]) > 1e-6
    assert abs(q_both[2]) > abs(q_center[2]) - 1e-9


def test_near_limit_boosts_rail_weight() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True, k_margin_boost=4.0, w_ext_cap=20.0, w_max=2.0, k_esc=0.0
        ),
    )
    task.set_mode("reach")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    task.capture_reference(q)
    task.d_pref_m = task.extension(q) - 0.20
    _, w_ok = task(q, sigma_scale=1.0, joint_margin_frac=1.0, dt_s=0.005)
    _, w_near = task(q, sigma_scale=1.0, joint_margin_frac=0.0, dt_s=0.005)
    assert w_near > w_ok


def test_margin_does_not_authorize_escape_without_press() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_esc=1.2,
            k_ext=0.0,
            v_lpf_tau_s=0.0,
            v_lpf_tau_escape_s=0.0,
        ),
    )
    task.set_mode("reach")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    task.capture_reference(q)
    v_ok, _ = task(
        q,
        sigma_scale=1.0,
        sigma_grad_rail=1.0,
        joint_margin_frac=1.0,
        dt_s=0.005,
    )
    v_near, _ = task(
        q,
        sigma_scale=1.0,
        sigma_grad_rail=1.0,
        joint_margin_frac=0.2,
        dt_s=0.005,
    )
    assert abs(v_ok) < 1e-6
    assert abs(v_near) < 1e-6
    assert not task._escape_active
    v_press, _ = task(
        q,
        sigma_scale=1.0,
        sigma_grad_rail=1.0,
        joint_margin_frac=1.0,
        dt_s=0.005,
        press_escape_allowed=True,
    )
    assert abs(v_press) > 1e-3
    assert task._escape_active


def test_latched_escape_still_exposes_e_mid() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_esc=1.2,
            k_ext=5.0,
            e0_m=0.0,
            e1_m=0.01,
            sigma_escape_enter=0.99,
            escape_enter_dwell_s=0.0,
            v_lpf_tau_s=0.0,
            v_lpf_tau_escape_s=0.0,
            escape_grad_floor=1.0,
            d_band_m=0.0,
        ),
    )
    task.set_mode("reach")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    task.capture_reference(q)
    task.d_pref_m = task.extension(q) - 0.30
    v, _ = task(
        q,
        sigma_scale=0.2,
        sigma_grad_rail=2.0,
        joint_margin_frac=1.0,
        dt_s=0.005,
        press_escape_allowed=True,
    )
    assert task._escape_active
    assert abs(task.last_e_mid_m) > 0.20
    assert task.last_v_reach == pytest.approx(0.0, abs=1e-12)
    assert abs(float(v)) == pytest.approx(abs(task.last_v_escape), abs=1e-9)
    assert abs(task.last_v_escape) > 1e-9


def test_escape_stays_out_of_the_rail_limit_band() -> None:
    """Inside the soft-limit fade the carriage has nowhere to escape to.

    Measured on hardware: the latch fired on 29-31% of ticks at the stop
    versus 5-9% mid-travel, and fighting the reach term against the wall is
    what the operator feels as rail chatter.  Limit avoidance there belongs
    to the weighted-least-norm reg, which hands the stroke to the arm.
    """
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_esc=1.2,
            k_ext=0.0,
            limit_margin_m=0.08,
            sigma_escape_enter=0.99,
            escape_enter_dwell_s=0.0,
            v_lpf_tau_s=0.0,
            v_lpf_tau_escape_s=0.0,
            escape_grad_floor=1.0,
        ),
    )
    task.set_mode("reach")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    q[0] = float(kin.q_upper[0]) - 0.02  # inside the 0.08 m fade band
    task.capture_reference(q)
    for _ in range(2):
        task(
            q, sigma_scale=0.2, sigma_grad_rail=2.0, joint_margin_frac=1.0, dt_s=0.005
        )
        assert not task._escape_active
        assert abs(task.last_v_escape) < 1.0e-12

    q_mid = 0.5 * (kin.q_lower + kin.q_upper)
    task.capture_reference(q_mid)
    task(
        q_mid,
        sigma_scale=0.2,
        sigma_grad_rail=2.0,
        joint_margin_frac=1.0,
        dt_s=0.005,
        press_escape_allowed=True,
    )
    assert task._escape_active


def test_healthy_vel_ff_follows_without_escape_latch() -> None:
    """Any MotionReference FF owns the rail when σ/margin are healthy."""
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_esc=0.5,
            k_ff=1.0,
            k_ext=0.0,
            v_ff_thr_m_s=0.005,
            sigma_escape_enter=0.55,
            sigma_escape_exit=0.80,
            margin_escape_enter=0.12,
            margin_escape_exit=0.25,
            v_lpf_tau_s=0.0,
            escape_grad_floor=0.0,
            v_max_m_s=0.08,
        ),
    )
    task.set_mode("reach")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    task.capture_reference(q)
    # Arbitrary MotionReference FF aligned with the rail column (not Y-hardcoded).
    j_rail = kin.jacobian(q)[:3, 0]
    n = float(np.linalg.norm(j_rail))
    assert n > 1e-6
    vel_ff = np.zeros(6)
    vel_ff[:3] = 0.05 * (j_rail / n)
    v, _ = task(
        q,
        sigma_scale=0.85,  # mild σ dip — must NOT latch
        sigma_grad_rail=0.5,
        joint_margin_frac=1.0,
        vel_ff=vel_ff,
        dt_s=0.005,
    )
    assert not task._escape_active
    assert abs(task.last_v_escape) < 1e-12
    assert abs(task.last_v_ff) > 1e-4
    assert abs(v) < 1e-9


def test_deep_sigma_does_not_latch_escape() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_esc=0.5,
            k_ext=0.0,
            v_lpf_tau_s=0.0,
            v_lpf_tau_escape_s=0.0,
        ),
    )
    task.set_mode("reach")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    task.capture_reference(q)
    v1, _ = task(
        q, sigma_scale=0.3, sigma_grad_rail=2.0, joint_margin_frac=1.0, dt_s=0.005
    )
    v2, _ = task(
        q, sigma_scale=0.3, sigma_grad_rail=-2.0, joint_margin_frac=1.0, dt_s=0.005
    )
    assert not task._escape_active
    assert abs(v1) < 1e-12
    assert abs(v2) < 1e-12
    assert abs(task.last_v_escape) < 1e-12


def test_true_near_limit_does_not_latch_without_press() -> None:
    kin = RobotKinematics()
    cfg = RailExtensionConfig(
        enabled=True,
        k_esc=0.5,
        k_ext=0.0,
        v_lpf_tau_s=0.0,
    )
    q = 0.5 * (kin.q_lower + kin.q_upper)
    task_ok = RailExtensionTask(kin, cfg)
    task_ok.set_mode("reach")
    task_ok.capture_reference(q)
    v_ok, _ = task_ok(
        q,
        sigma_scale=1.0,
        sigma_grad_rail=2.0,
        joint_margin_frac=0.5,
        dt_s=0.005,
    )
    assert abs(v_ok) < 1e-6
    assert not task_ok._escape_active

    task_near = RailExtensionTask(kin, cfg)
    task_near.set_mode("reach")
    task_near.capture_reference(q)
    v_near, _ = task_near(
        q,
        sigma_scale=1.0,
        sigma_grad_rail=2.0,
        joint_margin_frac=0.05,
        dt_s=0.005,
    )
    assert not task_near._escape_active
    assert abs(v_near) < 1e-6


def test_escape_enter_ignores_dwell_without_press() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_esc=1.0,
            k_ext=0.0,
            escape_enter_dwell_s=0.05,
            v_lpf_tau_s=0.0,
        ),
    )
    task.set_mode("reach")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    task.capture_reference(q)
    task(q, sigma_scale=0.2, sigma_grad_rail=2.0, dt_s=0.005)
    assert not task._escape_active
    for _ in range(10):
        task(q, sigma_scale=0.2, sigma_grad_rail=2.0, dt_s=0.005)
    assert not task._escape_active
    task(q, sigma_scale=0.2, sigma_grad_rail=2.0, dt_s=0.005, press_escape_allowed=True)
    assert task._escape_active


def test_joint_comfort_inactive_when_centered() -> None:
    kin = RobotKinematics()
    b = JointComfortBuilder(
        JointComfortConfig(m_comfort_rad=0.26, activate_rad=0.44, gamma=6.0)
    )
    q = np.array([0.4, 0.0, -0.8, 0.0, 1.57, 0.0, 0.78, 0.0])
    rows = b.build_rows(q, kin.q_lower, kin.q_upper)
    assert rows.active is False


def test_joint_comfort_is_j4_only_and_still_binds_near_stop() -> None:
    """Comfort rows other than J4 never bound; the J4 row still holds the band."""
    kin = RobotKinematics()
    b = JointComfortBuilder(
        JointComfortConfig(m_comfort_rad=0.26, activate_rad=0.44, gamma=6.0)
    )
    q_j2 = np.array([0.4, 0.0, -2.20, 0.0, 1.57, 0.0, 0.78, 0.0])
    rows_j2 = b.build_rows(q_j2, kin.q_lower, kin.q_upper)
    assert rows_j2.active is False

    q_j4 = q_j2.copy()
    q_j4[4] = float(kin.q_upper[4]) - 0.20
    rows = b.build_rows(q_j4, kin.q_lower, kin.q_upper)
    assert rows.active
    j4 = [k for k in range(rows.jacobian.shape[0]) if abs(rows.jacobian[k, 4]) > 1e-6]
    assert j4
    assert int(rows.slack_col[j4[0]]) == COMFORT_SLACK0 + 3
    assert abs(rows.jacobian[j4[0], 2]) <= 1e-12
    # Near the upper stop: ∇h points −q4.
    assert rows.jacobian[j4[0], 4] < 0.0


def test_beyond_rail_cli_defaults_force_on(tmp_path) -> None:
    import argparse
    from pathlib import Path

    # Smoke the argparse + beyond force default logic via a mini replica.
    ap = argparse.ArgumentParser()
    ap.add_argument("--beyond-rail-cm", type=float, default=None)
    ap.add_argument(
        "--enable-force", action=argparse.BooleanOptionalAction, default=None
    )
    args = ap.parse_args(["--beyond-rail-cm", "5"])
    if args.beyond_rail_cm is not None and args.enable_force is None:
        args.enable_force = True
    assert args.enable_force is True
    args2 = ap.parse_args(["--beyond-rail-cm", "5", "--no-enable-force"])
    assert args2.enable_force is False
    del tmp_path, Path


def test_escape_sign_auto_holds_across_midpoint_and_dead_end() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_esc=0.5,
            k_ext=0.0,
            soft_min_m=0.10,
            soft_max_m=0.70,
            limit_margin_m=0.02,
            pin_margin_m=0.008,
            v_lpf_tau_s=0.0,
            v_lpf_tau_escape_s=0.0,
            escape_sign_policy="auto",
        ),
    )
    task.set_mode("reach")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    q[0] = 0.25
    task.capture_reference(q)
    task(
        q,
        sigma_scale=0.2,
        sigma_grad_rail=2.0,
        joint_margin_frac=1.0,
        dt_s=0.005,
        press_escape_allowed=True,
    )
    assert task._escape_active
    assert task._escape_sign > 0.0
    locked = float(task._escape_sign)
    q[0] = 0.55
    task(
        q,
        sigma_scale=0.2,
        sigma_grad_rail=-2.0,
        joint_margin_frac=1.0,
        dt_s=0.005,
        press_escape_allowed=True,
    )
    assert task._escape_sign == pytest.approx(locked)
    q[0] = 0.70
    v_end, _ = task(
        q,
        sigma_scale=0.2,
        sigma_grad_rail=-2.0,
        joint_margin_frac=1.0,
        dt_s=0.005,
        press_escape_allowed=True,
    )
    assert task._escape_sign == pytest.approx(locked)
    assert abs(task.last_v_escape) > 1e-9
    assert v_end * locked <= 1e-12
    task(
        q,
        sigma_scale=0.2,
        sigma_grad_rail=2.0,
        dt_s=0.005,
        press_escape_allowed=False,
    )
    assert not task._escape_active
    assert abs(task._escape_sign) < 1e-12
    assert abs(task.last_v_escape) < 1e-12


def test_escape_sign_minus_policy_holds_without_flip() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_esc=0.5,
            k_ext=0.0,
            soft_min_m=0.10,
            soft_max_m=0.70,
            limit_margin_m=0.02,
            pin_margin_m=0.008,
            v_lpf_tau_s=0.0,
            v_lpf_tau_escape_s=0.0,
            escape_sign_policy="minus",
        ),
    )
    task.set_mode("reach")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    q[0] = 0.51
    task.capture_reference(q)
    task(
        q,
        sigma_scale=0.2,
        sigma_grad_rail=2.0,
        joint_margin_frac=1.0,
        dt_s=0.005,
        press_escape_allowed=True,
    )
    assert task._escape_active
    assert task._escape_sign < 0.0
    locked = float(task._escape_sign)
    q[0] = 0.10
    v_end, _ = task(
        q,
        sigma_scale=0.2,
        sigma_grad_rail=2.0,
        joint_margin_frac=1.0,
        dt_s=0.005,
        press_escape_allowed=True,
    )
    assert task._escape_sign == pytest.approx(locked)
    assert abs(task.last_v_escape) > 1e-9
    assert v_end * locked <= 1e-12


def test_preferred_escape_is_minus_until_soft_min_pin() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            soft_min_m=0.10,
            soft_max_m=0.70,
            pin_margin_m=0.008,
            escape_leave_m=0.04,
            escape_sign_policy="minus",
        ),
    )
    assert task._preferred_escape_sign(0.51) < 0.0
    assert task._preferred_escape_sign(0.14) == 0.0
    assert task._preferred_escape_sign(0.105) > 0.0
    assert task._preferred_escape_sign(0.14, backoff=True) > 0.0
    assert task._in_plus_leave(0.67)


def test_press_stall_keeps_escape_despite_healthy_sigma() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_esc=1.2,
            k_ext=0.0,
            sigma_escape_enter=0.99,
            escape_enter_dwell_s=0.0,
            v_lpf_tau_s=0.0,
            v_lpf_tau_escape_s=0.0,
            escape_grad_floor=1.0,
            press_y_err_m=0.005,
        ),
    )
    task.set_mode("reach")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    task.capture_reference(q)
    task(
        q,
        sigma_scale=1.0,
        sigma_grad_rail=2.0,
        joint_margin_frac=1.0,
        sigma_raw=0.12,
        dt_s=0.005,
    )
    assert not task._escape_active
    assert abs(task.last_v_escape) < 1.0e-12
    v, _ = task(
        q,
        sigma_scale=1.0,
        sigma_grad_rail=2.0,
        joint_margin_frac=1.0,
        sigma_raw=0.12,
        dt_s=0.005,
        press_escape_allowed=True,
        tool_y_err_m=0.0,
    )
    assert abs(v) > 1e-4 or abs(task.last_v_escape) > 1e-4
    assert np.sign(task.last_v_escape) == np.sign(task._policy_escape_sign(float(q[0])))


def test_press_stall_allows_escape_in_limit_band_toward_open() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_esc=1.2,
            k_ext=0.0,
            limit_margin_m=0.08,
            sigma_escape_enter=0.99,
            escape_enter_dwell_s=0.0,
            v_lpf_tau_s=0.0,
            v_lpf_tau_escape_s=0.0,
            escape_grad_floor=1.0,
            press_y_err_m=0.005,
        ),
    )
    task.set_mode("reach")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    q[0] = float(kin.q_upper[0]) - 0.02
    task.capture_reference(q)
    task(
        q,
        sigma_scale=0.2,
        sigma_grad_rail=2.0,
        joint_margin_frac=1.0,
        dt_s=0.005,
        press_escape_allowed=True,
        tool_y_err_m=0.0,
    )
    assert task.last_v_escape * task._policy_escape_sign(float(q[0])) >= -1e-12
    assert abs(task.last_v_escape) > 1e-12 or task._escape_active


def _reach_budget_task(total_budget: float | None) -> RailExtensionTask:
    return RailExtensionTask(
        RobotKinematics(),
        RailExtensionConfig(
            enabled=True,
            k_ext=1.0,
            k_ff=1.0,
            k_esc=0.0,
            v_ff_thr_m_s=0.005,
            v_max_m_s=0.08,
            v_reach_cap_m_s=0.05,
            v_reach_total_max_m_s=total_budget,
            v_lpf_tau_s=0.0,
            v_lpf_tau_escape_s=0.0,
            escape_grad_floor=0.0,
        ),
    )


def _drive_reach(task: RailExtensionTask, *, v_ff_m_s: float) -> float:
    kin = task.kin
    q = 0.5 * (kin.q_lower + kin.q_upper)
    task.set_mode("reach")
    task.capture_reference(q)
    # Same-sign reach error: the stick pushes +rail and the posture
    # preference wants +rail too, which is when the two share a budget.
    task.d_pref_m = task.extension(q) - 0.20
    j_rail = kin.jacobian(q)[:3, 0]
    vel_ff = np.zeros(6)
    vel_ff[:3] = v_ff_m_s * j_rail
    v, _w = task(
        q,
        sigma_scale=1.0,
        sigma_grad_rail=0.0,
        joint_margin_frac=1.0,
        sigma_raw=0.5,
        vel_ff=vel_ff,
        dt_s=0.005,
    )
    return float(v)


def test_reach_records_measured_ff_but_does_not_apply_it() -> None:
    shared = _reach_budget_task(None)
    v_shared = _drive_reach(shared, v_ff_m_s=0.12)
    assert shared.last_v_ff == pytest.approx(0.12, rel=1e-6)
    assert shared.last_v_reach == pytest.approx(0.0, abs=1e-12)
    assert abs(v_shared) < 1e-9


def test_reach_budget_still_clips_beyond_the_new_total() -> None:
    task = _reach_budget_task(0.17)
    v = _drive_reach(task, v_ff_m_s=0.30)
    assert abs(v) <= task.cfg.reach_budget_m_s() + 1e-12


def test_pose_attract_keeps_the_original_cap_when_reach_budget_is_raised() -> None:
    kin = RobotKinematics()
    task = _reach_budget_task(0.17)
    task.set_mode("pose_attract")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    task.capture_reference(q)
    task.y_rail_target_m = float(q[0]) + 0.30  # k_pose 2.0 * 0.30 = 0.60 m/s
    v, _w = task(q, sigma_scale=1.0, sigma_grad_rail=0.0, dt_s=0.005)
    assert float(v) == pytest.approx(0.08, abs=1e-9)


def _w_ff_task(*, v_ff_thr_m_s: float = 0.005) -> RailExtensionTask:
    return RailExtensionTask(
        RobotKinematics(),
        RailExtensionConfig(
            enabled=True,
            k_ext=0.0,
            k_ff=1.0,
            k_esc=0.0,
            w_max=2.0,
            w_sigma_floor=0.0,
            v_ff_thr_m_s=v_ff_thr_m_s,
            v_ff_span_m_s=0.015,
            e0_m=0.20,
            e1_m=0.40,
            v_lpf_tau_s=0.0,
            escape_grad_floor=0.0,
        ),
    )


def _drive_ff_only(task: RailExtensionTask, v_ff_m_s: float) -> tuple[float, float]:
    kin = task.kin
    q = 0.5 * (kin.q_lower + kin.q_upper)
    task.set_mode("reach")
    task.capture_reference(q)
    j_rail = kin.jacobian(q)[:3, 0]
    vel_ff = np.zeros(6)
    if abs(v_ff_m_s) > 0.0:
        vel_ff[:3] = v_ff_m_s * j_rail
    _v, w = task(
        q,
        sigma_scale=1.0,
        sigma_grad_rail=0.0,
        joint_margin_frac=1.0,
        sigma_raw=0.5,
        vel_ff=vel_ff,
        dt_s=0.005,
    )
    return float(task.last_v_ff), float(w)


def test_w_ff_is_live_below_the_ownership_threshold() -> None:
    from rm75_control.control.joint_admittance_8dof.filters import smoothstep01 as _smoothstep01

    task = _w_ff_task()
    v_ff, w = _drive_ff_only(task, 0.0023)
    assert v_ff == pytest.approx(0.0023, rel=0.05)
    expected = 2.0 * _smoothstep01(abs(v_ff) / 0.015)
    assert w == pytest.approx(expected, abs=1e-6)
    assert w > 0.1


def test_w_ff_is_zero_when_feedforward_is_zero() -> None:
    task = _w_ff_task()
    v_ff, w = _drive_ff_only(task, 0.0)
    assert v_ff == pytest.approx(0.0, abs=1e-12)
    assert w == pytest.approx(0.0, abs=1e-12)


def test_ff_owns_still_flips_at_the_threshold() -> None:
    task = _w_ff_task(v_ff_thr_m_s=0.005)
    kin = task.kin
    q = 0.5 * (kin.q_lower + kin.q_upper)
    task.set_mode("reach")
    task.capture_reference(q)
    task.d_pref_m = task.extension(q) - 0.10
    j_rail = kin.jacobian(q)[:3, 0]
    vel_lo = np.zeros(6)
    vel_lo[:3] = 0.004 * j_rail
    vel_hi = np.zeros(6)
    vel_hi[:3] = 0.006 * j_rail
    task(
        q,
        sigma_scale=1.0,
        sigma_grad_rail=0.0,
        vel_ff=vel_lo,
        dt_s=0.005,
        sigma_raw=0.5,
    )
    assert abs(task.last_v_ff) < 0.005
    assert task.last_k_ff_scale < 1.0
    task(
        q,
        sigma_scale=1.0,
        sigma_grad_rail=0.0,
        vel_ff=vel_hi,
        dt_s=0.005,
        sigma_raw=0.5,
    )
    assert abs(task.last_v_ff) > 0.005
    assert task.last_k_ff_scale == pytest.approx(1.0)


def test_backoff_without_press_cannot_authorize_escape() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_esc=1.0,
            k_ext=0.0,
            soft_min_m=0.10,
            soft_max_m=0.70,
            pin_margin_m=0.008,
            escape_leave_m=0.04,
            escape_sign_policy="minus",
            v_lpf_tau_s=0.0,
        ),
    )
    task.set_mode("reach")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    q[0] = 0.12  # minus leave-band
    task.capture_reference(q)
    v, _ = task(
        q,
        sigma_scale=0.05,
        sigma_grad_rail=2.0,
        dt_s=0.005,
        press_escape_allowed=False,
        tool_y_err_m=0.02,
    )
    assert not task._escape_active
    assert abs(v) < 1e-12
    assert abs(task.last_v_escape) < 1e-12


def test_press_and_low_sigma_still_allows_escape() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(enabled=True, k_esc=1.0, k_ext=0.0, v_lpf_tau_s=0.0),
    )
    task.set_mode("reach")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    task.capture_reference(q)
    v, _ = task(
        q,
        sigma_scale=0.05,
        sigma_grad_rail=2.0,
        dt_s=0.005,
        press_escape_allowed=True,
    )
    assert task._escape_active
    assert abs(task.last_v_escape) > 1e-4
    assert abs(v) > 1e-4

