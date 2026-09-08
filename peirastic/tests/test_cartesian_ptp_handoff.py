"""Recorded PTP endpoint must be maintainable under online position margins."""

import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from peirastic.realman8dof.binding import load_yaml
from peirastic.realman8dof.modes.cartesian import (
    build_cartesian_ptp_phase, controller_position_bounds, resolve_pose_q,
)
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.force.compensation.tool_pose import apply_kin_tcp_offset


DATA = json.loads((Path(__file__).with_name("data") / "phantom_standoff_handoff_20260908.json").read_text())


def _context():
    kin = RobotKinematics()
    apply_kin_tcp_offset(kin, DATA["tool"])
    cfg = build_joint_ik_config(load_yaml(Path(__file__).parents[1] / "configs/controller.yaml"))
    inner = SimpleNamespace(q_cmd=np.array(DATA["q_seed"]), cfg=cfg,
                            posture_retarget=SimpleNamespace(psi_star_rad=DATA["psi_star"], d_star_m=DATA["d_star"]))
    return SimpleNamespace(kin=kin, inner=inner, euler_order="xyz")


def test_recorded_position_margin_conflict_is_avoided_during_ik_selection():
    ctx = _context()
    old, fault = np.array(DATA["old_ik_q"]), np.array(DATA["q_fault"])
    lo, hi = controller_position_bounds(ctx)
    assert old[4] < ctx.kin.q_upper[4]  # physical SRS limit admitted this goal
    assert old[4] > hi[4]  # but the online HOLD must force it back out
    assert (hi[4] - fault[4]) / ctx.inner.cfg.dt == pytest.approx(DATA["j4_box_velocity"], abs=1e-10)
    result = resolve_pose_q(ctx, DATA["pose"], require_path=False)
    assert np.all(result >= lo) and np.all(result <= hi)
    # Rail/arm redundancy should solve the same pose without modifying TCP.
    assert abs(result[0] - old[0]) > 0.01
    pose = ctx.kin.fk_pose(result)
    np.testing.assert_allclose(pose[:3], DATA["pose"][:3], atol=1e-6)
    error = Rotation.from_euler("xyz", pose[3:]).inv() * Rotation.from_euler("xyz", DATA["pose"][3:])
    assert error.magnitude() < 1e-5  # rounded URDF versus analytic SRS constants


def test_explicit_cartesian_joint_target_cannot_bypass_controller_margins():
    ctx = _context()
    with pytest.raises(ValueError, match="outside controller position margin: J4"):
        build_cartesian_ptp_phase(ctx, {"pose": DATA["pose"], "q_target": DATA["old_ik_q"]})


def test_runtime_position_bounds_include_rail_hard_limits_without_mutating_kin():
    ctx = _context()
    original = ctx.kin.q_upper.copy()
    lo, hi = ctx.kin.q_lower.copy(), ctx.kin.q_upper.copy()
    lo[0], hi[0] = 0.05, 0.70
    margin = np.full(8, 0.02)
    margin[0] = 0.003
    ctx.inner.limits = SimpleNamespace(q_lower=lo, q_upper=hi, position_margin=margin)
    actual_lo, actual_hi = controller_position_bounds(ctx)
    np.testing.assert_allclose(actual_lo, lo + margin)
    np.testing.assert_allclose(actual_hi, hi - margin)
    np.testing.assert_array_equal(ctx.kin.q_upper, original)


def test_replanned_standoff_can_enter_native_hold_with_same_constraints():
    from rm75_control.control.joint_admittance_8dof.wbc_rt.client import find_wbc_rt_binary
    if find_wbc_rt_binary() is None:
        pytest.skip("native binary is not built")
    ctx = _context()
    cfg = ctx.inner.cfg
    cfg.backend = "native"
    cfg.native_shm_prefix = "phantom_handoff_test_" + uuid.uuid4().hex
    cfg.control_cpu, cfg.native_cpu = 6, 8
    qt = resolve_pose_q(ctx, DATA["pose"], require_path=False)
    inner = JointIkController(ctx.kin, cfg)
    try:
        # Confirm this fixture still exercises the original failure mechanism.
        fault = np.array(DATA["q_fault"])
        inner.reset(np.array(DATA["q_cmd_fault"]))
        bad = inner.update(np.zeros(6), q_meas=fault, rail_exec_vel_m_s=0.0, auto_commit=False)
        assert bad.solver_fault_latched
        inner.reset(qt)
        q = qt.copy()
        for _ in range(100):
            step = inner.update(np.zeros(6), q_meas=q, rail_exec_vel_m_s=0.0)
            assert not step.solver_fault_latched, step.fallback_reason
            assert step.qpik_hard_residual_max <= 1e-5
            q = step.q_send.copy()
    finally:
        inner.close()
