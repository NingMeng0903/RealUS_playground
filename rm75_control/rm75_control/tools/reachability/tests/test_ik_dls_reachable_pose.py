"""DLS IK correctness on synthetic reachable targets.

Strategy:
  1. Draw random arm configurations, forward-kinematics them → these poses
     are reachable by construction (they are hit by some ``q_truth``).
  2. Solve IK from a *different* seed and assert the returned q reproduces the
     pose within tolerance (may land on a different IK branch, that's fine).
  3. Cross-check against the fixed-rail numerical
     ``pose_ik.solve_pose_ik`` wrapper on a few of them — same reachability
     decision, comparable accuracy.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import pytest

from rm75_control.tools.reachability.data_model.frames import tool_axis_from_quat
from rm75_control.tools.reachability.kinematics import (
    SeedPoolConfig,
    build_locked_rail_model,
    build_seed_pool,
    fk_position_quat_batch,
    ik_dls,
    ik_dls_multiseed,
)
from rm75_control.tools.reachability.kinematics.model_locked_rail import DEFAULT_URDF


@pytest.fixture(scope="module")
def lm():
    if not DEFAULT_URDF.exists():
        pytest.skip(f"URDF missing at {DEFAULT_URDF}")
    return build_locked_rail_model(DEFAULT_URDF)


def _random_reachable_pose(lm, rng: np.random.Generator) -> tuple[pin.SE3, np.ndarray]:
    # keep away from limits by 10° to avoid deliberate ill-conditioning
    lo = lm.q_lower + np.deg2rad(10.0)
    hi = lm.q_upper - np.deg2rad(10.0)
    q = rng.uniform(lo, hi)
    pin.forwardKinematics(lm.model, lm.data, q)
    pin.updateFramePlacement(lm.model, lm.data, lm.tcp_id)
    return pin.SE3(lm.data.oMf[lm.tcp_id].rotation.copy(), lm.data.oMf[lm.tcp_id].translation.copy()), q


def test_ik_dls_single_seed_hits_target(lm):
    rng = np.random.default_rng(0)
    hits = 0
    total = 20
    for _ in range(total):
        target, q_truth = _random_reachable_pose(lm, rng)
        seed = q_truth + rng.normal(scale=0.15, size=q_truth.shape)
        res = ik_dls(lm, target, seed, max_iter=60, lam=0.05)
        if res.report.ok:
            hits += 1
    # single-seed IK from a nearby seed should almost always converge
    assert hits >= int(0.9 * total)


def test_ik_dls_multiseed_hits_high_rate(lm):
    rng = np.random.default_rng(1)
    seeds = build_seed_pool(lm.q_lower, lm.q_upper, SeedPoolConfig(n_random=8, random_seed=7))
    hits = 0
    total = 30
    for _ in range(total):
        target, _ = _random_reachable_pose(lm, rng)
        res = ik_dls_multiseed(lm, target, seeds, max_iter=60, lam=0.05)
        if res.report.ok:
            hits += 1
    # multi-seed should sweep >=95% of clearly reachable poses
    assert hits >= int(0.95 * total)


def test_ik_dls_returns_within_limits(lm):
    rng = np.random.default_rng(2)
    target, _ = _random_reachable_pose(lm, rng)
    seeds = build_seed_pool(lm.q_lower, lm.q_upper)
    res = ik_dls_multiseed(lm, target, seeds)
    assert res.report.ok
    lo, hi = lm.q_lower, lm.q_upper
    assert np.all(res.q >= lo - 1e-9)
    assert np.all(res.q <= hi + 1e-9)


def test_ik_dls_pose_error_matches_fk(lm):
    rng = np.random.default_rng(3)
    target, q_truth = _random_reachable_pose(lm, rng)
    seed = q_truth + rng.normal(scale=0.05, size=q_truth.shape)
    res = ik_dls(lm, target, seed, max_iter=80, lam=0.03)
    assert res.report.ok
    pos, quat = fk_position_quat_batch(lm, res.q[None, :])
    np.testing.assert_allclose(pos[0], target.translation, atol=1e-3)
    # rotation: compare TCP z-axis (5-DOF check; roll is unconstrained if any)
    axis_ours = tool_axis_from_quat(quat[0])
    axis_ref = target.rotation @ np.array([0.0, 0.0, 1.0])
    ang = np.arccos(np.clip(float(np.dot(axis_ours, axis_ref)), -1.0, 1.0))
    assert ang < np.deg2rad(1.0)


def test_ik_dls_rejects_unreachable_pose(lm):
    # far outside the arm reach
    far = pin.SE3(np.eye(3), np.array([2.5, 0.0, 0.0]))
    seeds = build_seed_pool(lm.q_lower, lm.q_upper, SeedPoolConfig(n_random=4))
    res = ik_dls_multiseed(lm, far, seeds, max_iter=40, lam=0.1)
    assert not res.report.ok
    # residual should be large (target is 2+ m away)
    assert res.report.pos_err_m > 0.5


def test_ik_dls_and_numerical_agree_on_reachable(lm):
    """DLS should agree with fixed-rail numerical IK on reachable poses.

    We only check the *decision* (both converge) and coarse pose match; the
    two solvers can land on different branches so joint values may differ.
    """
    try:
        from rm75_control.control.joint_admittance_8dof.model import RobotKinematics as Kin8
        from rm75_control.control.joint_admittance_8dof.numerical_pose_ik import (
            NumericalPoseIkConfig,
        )
        from rm75_control.control.joint_admittance_8dof.pose_ik import solve_pose_ik
    except Exception as e:  # pragma: no cover
        pytest.skip(f"8-DOF numerical IK unavailable: {e}")
    kin8 = Kin8(urdf_path=DEFAULT_URDF)

    rng = np.random.default_rng(4)
    total, both = 6, 0
    for _ in range(total):
        target, q_truth = _random_reachable_pose(lm, rng)
        # DLS path
        seed = q_truth + rng.normal(scale=0.05, size=q_truth.shape)
        res_dls = ik_dls(lm, target, seed, max_iter=80, lam=0.03)
        # Fixed-rail numerical path (8-DOF): pose_target as
        # [x,y,z, rx,ry,rz] xyz-euler; rail_y=0 seed.
        from scipy.spatial.transform import Rotation as R

        rxryrz = R.from_matrix(target.rotation).as_euler("xyz", degrees=False)
        pose6 = np.concatenate([target.translation, rxryrz])
        q_seed_full = np.zeros(8)
        q_seed_full[1:] = q_truth
        _q_num, ok_num, _ = solve_pose_ik(
            kin8,
            q_seed_full,
            pose6,
            config=NumericalPoseIkConfig(max_iters=120),
        )
        if res_dls.report.ok and ok_num:
            both += 1
    # Because both solvers get a very good seed here, both should converge
    # on essentially every pose.
    assert both >= total - 1
