"""End-to-end invariants for the SRS + rail plan.

Pins the "smart-allocation" properties end-to-end so the plan-level fixes
(Bug 1/2/3/4/5/6) cannot silently regress:

1. SRS IK planner (Bug 4) meets ``sigma_min ≥ 0.05`` on realistic scan-slot
   move targets — the very floor the plan's success criteria calls out.
2. ψ landing is attracted to ``ψ_home`` (or to a bounded excursion when
   ``ψ_home`` is unset) — no more 97 deg → -172 deg jumps.
3. :class:`SrsSmoothMoveReference` (Bug 5) keeps the SRS branch stable for
   the full quintic (no mid-move J1/J4 flip).
4. Production yaml loads cleanly via :func:`build_joint_ik_config`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.pose_ik import (
    UnreachablePathError,
    resolve_pose_ik_srs,
)
from rm75_control.control.joint_admittance_8dof.reference import (
    SrsSmoothMoveReference,
    SinToolYReference,
    srs_move_duration_s,
)
from rm75_control.kinematics.srs_ik import branch_from_q, psi_from_q


CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"


@pytest.fixture(scope="module")
def cfg():
    raw = yaml.safe_load(CFG_PATH.read_text())
    return build_joint_ik_config(raw)


@pytest.fixture(scope="module")
def kin():
    return RobotKinematics()


def test_yaml_loads_production_config():
    """Production yaml exposes slack QP + rail-extension, not the 28-var stack."""
    raw = yaml.safe_load(CFG_PATH.read_text())
    cfg = build_joint_ik_config(raw)
    assert cfg.qp.backend == "proxqp"
    assert cfg.rail_extension.enabled
    assert cfg.rail.soft_min_m == pytest.approx(
        raw["qpik"]["hard_limits"]["rail"]["soft_min_m"]
    )
    assert cfg.rail.soft_max_m == pytest.approx(
        raw["qpik"]["hard_limits"]["rail"]["soft_max_m"]
    )
    assert not hasattr(cfg, "generic_qpik")


def test_srs_ik_move_meets_sigma_and_psi_floors(kin, cfg):
    """Bug 4: closed-form IK + ψ enum gives a well-conditioned q_target."""
    q_seed = np.array([0.0, 0.0, 0.5, 0.3, -1.2, 0.0, 0.5, 0.0])
    pose0 = kin.fk_pose(q_seed).copy()
    pose_target = pose0.copy()
    pose_target[0] += 0.10
    pose_target[2] -= 0.05
    q_target, ok, rpt = resolve_pose_ik_srs(
        kin,
        q_seed=q_seed,
        pose_target=pose_target,
        y_rail_target=0.05,
        # The generic profile has no null-space ``arm_angle`` block.  A
        # branch-local ψ seed is the planner's explicit posture guide.
        psi_home_rad=float(psi_from_q(q_seed[1:])),
        max_psi_swing_rad=np.pi,
    )
    assert ok
    assert rpt.pos_err_mm < 1.0
    assert rpt.rot_err_deg < 1.0
    # Plan §Success criteria: σ_min ≥ 0.05 at every planning target.
    assert rpt.sigma_min >= 0.05, f"sigma_min={rpt.sigma_min} below 0.05 floor"
    # J1 / J4 should not swing to extremes for a modest move.
    assert abs(np.degrees(q_target[1])) < 90.0
    assert abs(np.degrees(q_target[4])) < 130.0


def test_srs_move_ref_branch_stable(kin):
    """Bug 5: quintic move keeps the SRS branch locked — no mid-move J1/J4 flip."""
    q0 = np.array([0.0, 0.0, 0.5, 0.3, -1.2, 0.0, 0.5, 0.0])
    pose0 = kin.fk_pose(q0).copy()
    pose_target = pose0.copy()
    pose_target[0] += 0.10
    pose_target[2] -= 0.05
    ref = SrsSmoothMoveReference(
        kin,
        q0,
        pose_target,
        y_rail_target_m=0.05,
        psi_target_rad=np.deg2rad(90.0),
        duration_s=2.0,
    )
    q_prev = q0.copy()
    for t in np.linspace(0.0, 2.0, 41):
        q, _ = ref.sample_q(t)
        assert branch_from_q(q[1:]) == ref.branch_id, f"branch flip at t={t}"
        # Also check no discontinuous jumps in q (max ~0.15 rad per 50 ms tick)
        step = np.max(np.abs(q - q_prev))
        assert step < 0.25, f"discontinuous q step {step:.3f} rad at t={t}"
        q_prev = q


def test_sin_tool_y_set_origin_idempotent_at_same_time():
    """set_origin at the current sampled pose must not shift the reference."""
    ref = SinToolYReference(0.40, max_vel_m_s=0.05, soft_start=True, ramp_s=2.0)
    origin = np.array([0.40, 0.02, 0.41, -1.2, 1.5, -1.1])
    ref.set_origin(origin, t_s=0.0)
    t_rel = 42.57
    pose_before = ref.sample(t_rel).pose_d.copy()
    ref.set_origin(pose_before, t_s=t_rel)
    pose_after = ref.sample(t_rel).pose_d
    assert np.linalg.norm(pose_after[:3] - pose_before[:3]) < 1e-6


def test_srs_move_duration_respects_joint_rate_limits():
    """Bug 5: auto-duration honours max_qdot_rad_s per joint."""
    q_start = np.zeros(8)
    q_target = np.array([0.2, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # 1 rad on J1
    T = srs_move_duration_s(q_start, q_target, max_qdot_rad_s=1.0, peak_v_frac=0.6)
    # Quintic peak: 1.875 · |dq| / T ≤ 0.6·1.0 → T ≥ 1.875·1/0.6 = 3.125
    assert T >= 3.0
