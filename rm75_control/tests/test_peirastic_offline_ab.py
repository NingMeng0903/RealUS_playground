"""Offline A/B: peirastic track_cartesian vs the old ellipse program.

Hardware p95 vs run_20260820_145541 (0.20 mm) still needs Window A on the
robot. This file only checks that the new compile emits the same twist as
the library outer / old ellipse factory. Old d_*.py / run_joint_admittance
/ phase_ipc stay until that hardware gate passes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PLAYGROUND = Path(__file__).resolve().parents[2]
if str(_PLAYGROUND) not in sys.path:
    sys.path.insert(0, str(_PLAYGROUND))

import numpy as np
import yaml

from rm75_control.control.admittance_common.phase_ipc import SinToolYTaskParams
from rm75_control.control.joint_admittance_8dof.api import CompileContext
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.ellipse_track_program import (
    build_ellipse_track_program,
)
from rm75_control.control.joint_admittance_8dof.loop import (
    CartesianTrackOuterLoop,
    JointIkController,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from peirastic.core.modes import Mode, ModeRequest
from peirastic.realman8dof.session import compile_request


from peirastic.configs import DEFAULT_CONTROLLER_YAML

_CFG = DEFAULT_CONTROLLER_YAML
_SEED = np.array([0.375, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370])
_DESIGN_SEED = np.r_[0.375, np.deg2rad([-89.5, -94.5, 65.2, 96.0, 89.3, 61.0, 94.6])]
_OLD_STACK = (
    Path(__file__).resolve().parents[1]
    / "apps"
    / "joint_admittance_8dof"
    / "run_joint_admittance.py",
    Path(__file__).resolve().parents[1]
    / "rm75_control"
    / "control"
    / "admittance_common"
    / "phase_ipc.py",
)


def _ctx(seed=_SEED):
    raw = yaml.safe_load(_CFG.read_text())
    cfg = build_joint_ik_config(raw)
    cfg.backend = "python"
    cfg.native_shm_prefix = f"rm75_wbc_peir_ab_{os.getpid()}"
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    inner.reset(seed)
    ctx = CompileContext(
        kin=kin,
        inner=inner,
        euler_order=cfg.euler_order,
        control_frame=cfg.control_frame,
        v_scale=cfg.v_scale,
    )
    return raw, ctx


def test_old_window_a_stack_still_present() -> None:
    for path in _OLD_STACK:
        assert path.is_file(), f"keep old stack until hardware p95 vs 145541: {path}"


def test_peirastic_ellipse_matches_old_program_outer() -> None:
    raw, ctx = _ctx()
    payload = {
        "reference": "ellipse",
        "x_pp_cm": 10.0,
        "y_pp_cm": 30.0,
        "max_vel_cm_s": 4.0,
        "duration_s": 5.0,
        "max_lin_vel_m_s": 0.15,
    }
    phase = compile_request(ctx, ModeRequest(Mode.TRACK_CARTESIAN, payload), raw=raw)
    params = SinToolYTaskParams(
        config_path=str(_CFG),
        x_pp_cm=10.0,
        y_pp_cm=30.0,
        max_vel_cm_s=4.0,
        scan_duration=5.0,
        cartesian_max_lin_vel=0.15,
        plan_duration_s=0.0,
    )
    old = build_ellipse_track_program(params, raw=raw)
    old_outer = old.compiled[-1].outer
    assert isinstance(old_outer, CartesianTrackOuterLoop)
    pose0 = ctx.kin.fk_pose(_SEED)
    phase.outer.set_origin(pose0, t_s=0.0)
    old_outer.set_origin(pose0, t_s=0.0)
    for t in (0.0, 0.25, 0.8, 1.5, 2.4):
        a = np.asarray(phase.outer.sample(t, pose0, np.zeros(6)), dtype=float)
        b = np.asarray(old_outer.sample(t, pose0, np.zeros(6)), dtype=float)
        assert np.allclose(a, b, atol=1e-9), t


def _run_ellipse(seed):
    raw, ctx = _ctx(seed)
    phase = compile_request(
        ctx,
        ModeRequest(
            Mode.TRACK_CARTESIAN,
            {
                "reference": "ellipse",
                "x_pp_cm": 10.0,
                "y_pp_cm": 30.0,
                "max_vel_cm_s": 4.0,
                "max_lin_vel_m_s": 0.15,
            },
        ),
        raw=raw,
    )
    pose0 = ctx.kin.fk_pose(seed)
    phase.outer.set_origin(pose0, t_s=0.0)
    dt = 0.005
    errs = []
    q = seed.copy()
    qdot = np.zeros(8)
    steps = []
    for i in range(400):
        t = i * dt
        pose = ctx.kin.fk_pose(q)
        twist = np.asarray(phase.outer.sample(t, pose, np.zeros(6)), dtype=float)
        step = ctx.inner.update(
            twist,
            dt,
            q_meas=q,
            rail_exec_vel_m_s=float(qdot[0]),
            dt_wall_s=dt,
        )
        q = np.asarray(step.q_send, dtype=float).copy()
        qdot = np.asarray(step.qdot, dtype=float).copy()
        errs.append(float(phase.outer.last_err_mm))
        steps.append(step)
    p95 = float(np.percentile(np.asarray(errs[40:], dtype=float), 95))
    return p95, steps


def test_peirastic_ellipse_closed_loop_python_inner() -> None:
    # Exercise the configured design family away from the hard J4 comfort
    # boundary. The legacy seed below reaches that boundary during this arc;
    # its old tracking score relied on independent Cartesian slack.
    p95, steps = _run_ellipse(_DESIGN_SEED)
    # Offline python inner, not the hardware 0.20 mm log. Keep this tight
    # enough that a broken compile cannot hide behind "just a sim".
    assert p95 < 2.0, f"offline ellipse p95 {p95:.3f} mm"
    assert not any(s.task_paused for s in steps)
    assert np.percentile([s.task_progress for s in steps[40:]], 5) > 0.95


def test_ellipse_at_j4_boundary_reduces_progress_without_direction_slack() -> None:
    _, steps = _run_ellipse(_SEED)
    for s in steps:
        assert not s.task_paused
        np.testing.assert_allclose(s.v_tcp_estimated, s.v_cmd_feasible, atol=2e-5)
        assert s.qpik_hard_residual_max < 1e-5
        assert s.qp1_status in ("solved", "max_iter")
