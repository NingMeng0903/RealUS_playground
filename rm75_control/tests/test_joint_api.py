"""Offline tests for joint_admittance.api compile layer."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.admittance_common.controller import AdmittanceConfig, AdmittanceController
from rm75_control.control.joint_admittance.api import (
    CompileContext,
    SecondaryPolicy,
    TaskMode,
    compile_phase,
    compute_move_plan,
    phase_cartesian_goto,
    phase_hybrid_track,
)
from rm75_control.control.joint_admittance.config import build_joint_ik_config
from rm75_control.control.joint_admittance.loop import (
    AdmittanceOuterLoop,
    CartesianTrackOuterLoop,
    JointIkController,
    JointTrackOuterLoop,
)
from rm75_control.control.joint_admittance.model import RobotKinematics, deg2rad
from rm75_control.control.joint_admittance.pose_ik import solve_pose_ik
from rm75_control.control.joint_admittance.reference import JointSmoothMoveReference, SinToolYReference
import yaml
from pathlib import Path


@pytest.fixture
def ctx():
    raw = yaml.safe_load(Path("configs/joint_admittance.yaml").read_text())
    kin = RobotKinematics()
    cfg = build_joint_ik_config(raw)
    inner = JointIkController(kin, cfg)
    return CompileContext(kin=kin, inner=inner, euler_order=cfg.euler_order, control_frame=cfg.control_frame, v_scale=cfg.v_scale)


def test_compute_move_plan_auto_joint(ctx):
    q0 = deg2rad(np.zeros(7))
    q1 = deg2rad(np.array([85.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    pose_d = ctx.kin.fk_pose(q1)
    plan = compute_move_plan(
        ctx.kin, q0, q1, pose_d, v_scale=ctx.v_scale, auto_select_joint=True, move_mode="cartesian"
    )
    assert plan.move_mode == "joint"
    assert plan.duration_s >= 2.5


def test_compile_cartesian_goto(ctx):
    q0 = deg2rad(np.array([10.0, -40.0, 15.0, 85.0, -10.0, 50.0, 5.0]))
    pose_d = ctx.kin.fk_pose(q0)
    q_tgt, ok, _ = solve_pose_ik(
        ctx.kin, q_seed=q0, pose_target=pose_d, qp_cfg=ctx.inner.cfg.qp, nullspace_cfg=ctx.inner.cfg.nullspace
    )
    assert ok
    ref = JointSmoothMoveReference(ctx.kin, q0, q_tgt, 2.5)
    spec = phase_cartesian_goto(
        ref,
        label="move->d",
        pose_target=pose_d,
        q_target_rad=q_tgt,
        move_mode="cartesian",
    )
    compiled = compile_phase(spec, ctx)
    assert isinstance(compiled.outer, CartesianTrackOuterLoop)
    assert compiled.phase.qdot_ff_provider is not None
    assert compiled.phase.on_enter is not None


def test_compile_joint_goto(ctx):
    q0 = deg2rad(np.zeros(7))
    q1 = deg2rad(np.array([30.0, -20.0, 10.0, 70.0, 0.0, 40.0, 5.0]))
    ref = JointSmoothMoveReference(ctx.kin, q0, q1, 3.0)
    spec = phase_cartesian_goto(ref, move_mode="joint", pose_target=ctx.kin.fk_pose(q1), q_target_rad=q1)
    compiled = compile_phase(spec, ctx)
    assert isinstance(compiled.outer, JointTrackOuterLoop)
    assert compiled.phase.governor_err_max_mm == 0.0


def test_compile_hybrid_track(ctx):
    ref = SinToolYReference(0.08, period_s=4.0)
    ctrl = AdmittanceController(0.005, AdmittanceConfig.from_dict({}))
    spec = phase_hybrid_track(
        ref,
        ctrl,
        desired_force=np.array([0, 0, 3.0, 0, 0, 0]),
        duration_s=10.0,
        psi_rad_on_enter=0.5,
    )
    compiled = compile_phase(spec, ctx)
    assert isinstance(compiled.outer, AdmittanceOuterLoop)
    assert spec.mode == TaskMode.HYBRID_TRACK
    assert spec.secondary.qdot_ff == "off"


def test_secondary_policy_move_preset(ctx):
    pol = SecondaryPolicy(preset="move")
    pol.apply(ctx.inner)
    assert ctx.inner._arm_task_suppressed is True
    assert ctx.inner._centering_suppressed is True
    assert ctx.inner._manipulability_active is True
