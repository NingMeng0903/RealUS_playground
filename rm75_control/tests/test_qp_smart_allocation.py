"""QP-level "smart allocation" invariants for the 8-DOF slack QP.

These tests pin the desired cost hierarchy of the WBC slack QP:

    W_task (100) >> rail_task_weight (<= 4.5) >> reg (~1e-2)

so that on a *feasible* twist the QP recruits the rail (or any joint) to keep
slack near zero (100% tracking), and on a *truly infeasible* twist (v_cmd
outside the reachable subspace, e.g. asking a stretched arm to extend
further) the QP accepts slack without spending the rail on wasted motion.

If a future tuning change breaks either invariant these tests fail loudly.
The whole "let rail rescue tracking near singularities" story falls apart if
the cost hierarchy is inverted, so this behaviour must be kept in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    full_q_from_arm,
)
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import (
    QpConfig,
    QpIkController,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


# Nearly-stretched arm pointing +X: shoulder tilted to horizontal, elbow and
# wrist opened almost straight.  sigma_min ~ 0.014 (deep singularity), rail
# Jacobian column = (0, 1, 0, 0, 0, 0) so pure-Y twist is exactly rail-solvable
# and pure-X twist (further along the arm) is *not* — the arm is already
# stretched, no joint can extend it further within v_max.
Q_ARM_SINGULAR_XSTRETCH = np.array(
    [0.0, np.pi / 2 - 0.1, 0.0, 0.15, 0.0, 0.15, 0.0], dtype=float
)

# Rail parked mid-travel.  These tests ask "does the QP recruit the rail", so
# the carriage has to be free to move both ways: rail travel is [0, 0.80] m
# (it used to be ±0.25 about 0), which turned the old rail=0 fixture into a
# hard end stop where the velocity box is pinned and the answer is forced.
RAIL_MID_M = 0.40


def _make_ctrl(kin: RobotKinematics) -> QpIkController:
    """Production-like weighting: W_task (100/50) far above rail-ext w_max."""
    limits = SafetyLimits.from_kinematics(kin, v_scale=0.5, a_max=20.0)
    return QpIkController(
        kin,
        limits,
        QpConfig(
            task_weight=np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]),
            collision=CollisionConfig(enabled=False),
        ),
    )


@pytest.fixture(scope="module")
def kin() -> RobotKinematics:
    return RobotKinematics()


def test_singular_pose_is_deeply_ill_conditioned(kin: RobotKinematics) -> None:
    """Sanity: the pose we chose really is near-singular and rail column is +Y."""
    q_full = full_q_from_arm(Q_ARM_SINGULAR_XSTRETCH, rail_m=RAIL_MID_M)
    J = kin.jacobian(q_full)
    sigma = kin.singular_values(J)
    assert sigma.min() < 0.05, f"pose not singular enough: sigma_min={sigma.min():.4f}"
    # J[:,0] must be almost exactly the world-Y translation direction — the
    # rail is a pure prismatic Y joint, so any test that "rail can rescue Y"
    # relies on this identity.
    assert abs(J[1, 0] - 1.0) < 1e-9
    assert np.linalg.norm(J[[0, 2, 3, 4, 5], 0]) < 1e-9


# ---------------------------------------------------------------------------
# Test 1 — feasible via rail: TCP wants +Y, rail column is +Y, arm cannot
# supply it near this singularity, so the QP MUST push the rail (qdot[0] > 0)
# and hold slack near zero (tracking preserved 100 %).
# ---------------------------------------------------------------------------
def test_qp_uses_rail_when_singular_and_feasible(kin: RobotKinematics) -> None:
    q_full = full_q_from_arm(Q_ARM_SINGULAR_XSTRETCH, rail_m=RAIL_MID_M)
    ctrl = _make_ctrl(kin)
    ctrl.reset(q_full)
    v_cmd = np.array([0.0, 0.02, 0.0, 0.0, 0.0, 0.0])  # 2 cm/s along +Y

    r = ctrl.step(q_full, v_cmd, 0.005)

    assert r.slack_norm < 1e-3, (
        f"slack unexpectedly large ({r.slack_norm:.3e}); QP failed to keep tracking"
    )
    assert r.qdot[0] > 0.01, (
        f"rail idle ({r.qdot[0]:.4f} m/s); QP failed to recruit the rail for a feasible twist"
    )
    # Sanity: primary equality is (near-)satisfied by the arm+rail combined.
    v_tcp = kin.jacobian(q_full) @ r.qdot
    assert np.linalg.norm(v_tcp - v_cmd) < 5e-3


# ---------------------------------------------------------------------------
# Test 2 — truly infeasible: TCP wants +X (further extension) but the arm is
# already stretched and the rail is Y-only.  The QP MUST accept slack and NOT
# waste rail motion on a direction the rail can't fix.
# ---------------------------------------------------------------------------
def test_qp_accepts_slack_when_no_rail_can_help(kin: RobotKinematics) -> None:
    q_full = full_q_from_arm(Q_ARM_SINGULAR_XSTRETCH, rail_m=RAIL_MID_M)
    ctrl = _make_ctrl(kin)
    ctrl.reset(q_full)
    v_cmd = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])  # 2 cm/s along +X

    r = ctrl.step(q_full, v_cmd, 0.005)

    # slack must be non-trivial: the QP is *softening* the primary task
    # because there is no feasible qdot that satisfies J qdot = v_cmd within
    # the box.  Threshold picked comfortably above numerical noise (~1e-5).
    assert r.slack_norm > 1e-3, (
        f"slack too small ({r.slack_norm:.3e}); QP wasted joints trying to be exact"
    )
    # Rail must not spend authority on a direction it can't help — the rail
    # column is Y, the twist is X, so qdot[0] gains nothing.  Any non-zero
    # rail velocity here means the cost hierarchy is inverted.
    assert abs(r.qdot[0]) < 1e-3, (
        f"rail bogged down ({r.qdot[0]:.4f} m/s) on a rail-orthogonal twist"
    )


# ---------------------------------------------------------------------------
# Test 3 — rail hint amplifies the smart allocation: the same feasible Y
# twist, but the caller also passes a rail-preferred velocity through the
# rail_task_vel_m_s cost channel (Bug 2 will drive w_ext up when sigma drops).
# The QP should now use *more* rail (closer to the hint) and keep slack tiny.
# ---------------------------------------------------------------------------
def test_qp_rail_hint_pulls_rail_toward_hinted_velocity(kin: RobotKinematics) -> None:
    q_full = full_q_from_arm(Q_ARM_SINGULAR_XSTRETCH, rail_m=RAIL_MID_M)
    ctrl = _make_ctrl(kin)
    ctrl.reset(q_full)
    v_cmd = np.array([0.0, 0.02, 0.0, 0.0, 0.0, 0.0])

    r = ctrl.step(
        q_full, v_cmd, 0.005, rail_task_vel_m_s=0.02, rail_task_weight=1.5
    )
    assert r.slack_norm < 1e-3
    assert r.qdot[0] > 0.015, f"rail hint ignored: qdot[0]={r.qdot[0]:.4f}"


# ---------------------------------------------------------------------------
# Test 4 — rail hint on an orthogonal twist must NOT waste rail motion.
# Verifies that the primary Cartesian equality dominates the soft rail cost,
# i.e. the rail is not driven off just because the hint says so.
# ---------------------------------------------------------------------------
def test_qp_rail_hint_does_not_move_rail_on_orthogonal_twist(kin: RobotKinematics) -> None:
    q_full = full_q_from_arm(Q_ARM_SINGULAR_XSTRETCH, rail_m=RAIL_MID_M)
    ctrl = _make_ctrl(kin)
    ctrl.reset(q_full)
    v_cmd = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Hint asks for zero rail velocity but a positive rail weight; this is
    # exactly what the current rail_extension task emits inside the dead zone.
    r = ctrl.step(
        q_full, v_cmd, 0.005, rail_task_vel_m_s=0.0, rail_task_weight=1.5
    )
    assert abs(r.qdot[0]) < 1e-3


# ---------------------------------------------------------------------------
# Test 5 — hierarchy anti-inversion: the primary Cartesian equality must
# retain the SAME sign as v_cmd even with an aggressive rail hint.  This is
# the minimal invariant we need — the plan calls the weight ratio
# W_task=100 vs w_ext_max=4.5 a 22:1 hierarchy; if that ever inverts (bugs
# like accidentally scaling W_task by 0.01 or raising w_ext to 100), the QP
# would follow the rail hint in preference to v_cmd, and this test would
# catch it (TCP Y sign flipping or tracking error > 30 % of target).
# ---------------------------------------------------------------------------
def test_task_hierarchy_cartesian_beats_rail_soft_cost(kin: RobotKinematics) -> None:
    q_arm = np.array(
        [-0.949552, 0.095255, 0.646858, 1.469911, 0.502701, 0.666503, -0.338137]
    )
    q_full = full_q_from_arm(q_arm, rail_m=RAIL_MID_M)
    ctrl = _make_ctrl(kin)
    ctrl.reset(q_full)
    v_cmd = np.array([0.0, 0.02, 0.0, 0.0, 0.0, 0.0])

    # Rail hint asks qdot[0] = -0.10 m/s (WRONG SIGN, opposite of v_cmd_y).
    # If the hierarchy inverted, the rail would pull TCP negative Y and slack
    # would swallow the positive-Y demand.
    r = ctrl.step(
        q_full, v_cmd, 0.005, rail_task_vel_m_s=-0.10, rail_task_weight=4.5
    )
    v_tcp = kin.jacobian(q_full) @ r.qdot

    # Sign must be preserved: TCP Y > 0.
    assert v_tcp[1] > 0.5 * v_cmd[1], (
        f"hierarchy inverted: v_tcp_y = {v_tcp[1]:.4f} (< half of v_cmd_y = 0.02)"
    )
    # Rail must not drive TCP more than 30% off the commanded Y velocity.
    assert abs(v_tcp[1] - v_cmd[1]) < 0.3 * abs(v_cmd[1]) + 5e-3
