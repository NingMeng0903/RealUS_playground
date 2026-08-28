"""Offline A/B: replay ellipse + gamepad (v_cmd, q_meas, rail) Python vs native.

Gates from the C++ inner-loop plan: arm q_cmd p95 < 1e-4 rad, rail < 0.1 mm,
slack / wall sign / QP status the same order of magnitude.

Do not switch production yaml to native until this passes.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.reference import ellipse_xy_motion
from rm75_control.control.joint_admittance_8dof.wbc_rt.client import find_wbc_rt_binary


_CFG = Path(__file__).resolve().parents[2] / "peirastic" / "configs" / "controller.yaml"
if not _CFG.is_file():
    _CFG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"
_SEED_Q = np.array([0.375, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370])
_ARM_P95_RAD = 1.0e-4
_RAIL_MM = 0.1


def _base_cfg():
    raw = yaml.safe_load(_CFG.read_text())
    cfg = build_joint_ik_config(raw)
    cfg.ird.enabled = False
    return cfg


def _ellipse_gamepad_vcmd(dt: float, n_ell: int = 80, n_pad: int = 80) -> np.ndarray:
    rows = []
    period = 2.5
    for i in range(n_ell):
        t = i * dt
        _dx, _dy, vx, vy = ellipse_xy_motion(
            t, 0.02, 0.04, 2.0 * np.pi / period, soft_start=True, ramp_s=0.4
        )
        rows.append(np.array([vx, vy, 0.0, 0.0, 0.0, 0.0], dtype=float))
    pad = [
        np.array([0.0, 0.04, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, -0.03, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.0]),
    ]
    for i in range(n_pad):
        rows.append(pad[(i // 20) % len(pad)].copy())
    return np.asarray(rows, dtype=float)


def _record_python(v_cmd: np.ndarray) -> dict:
    cfg = _base_cfg()
    cfg.backend = "python"
    inner = JointIkController(RobotKinematics(), cfg)
    inner.reset(_SEED_Q)
    q_meas = []
    rail_v = []
    q_cmd = []
    qdot = []
    slack = []
    wall = []
    qdot_prev = np.zeros(8)
    for twist in v_cmd:
        q = inner.q_cmd.copy()
        step = inner.update(
            twist,
            cfg.dt,
            q_meas=q,
            rail_exec_vel_m_s=float(qdot_prev[0]),
            dt_wall_s=cfg.dt,
        )
        q_meas.append(q)
        rail_v.append(float(qdot_prev[0]))
        q_cmd.append(inner.q_cmd.copy())
        qdot.append(np.asarray(step.qdot, dtype=float).reshape(8).copy())
        slack.append(float(step.slack_norm))
        wall.append(int(bool(step.wall_active)))
        qdot_prev = np.asarray(step.qdot, dtype=float).reshape(8)
    return {
        "v_cmd": v_cmd,
        "q_meas": np.asarray(q_meas),
        "rail_v": np.asarray(rail_v),
        "q_cmd": np.asarray(q_cmd),
        "qdot": np.asarray(qdot),
        "slack": np.asarray(slack),
        "wall": np.asarray(wall),
        "dt": float(cfg.dt),
    }


def _replay_native(rec: dict) -> dict:
    cfg = _base_cfg()
    cfg.backend = "native"
    cfg.native_shm_prefix = f"rm75_wbc_ab_{os.getpid()}"
    inner = JointIkController(RobotKinematics(), cfg)
    try:
        inner._native.timeout_s = 0.50
        inner.reset(_SEED_Q)
        q_cmd = []
        slack = []
        wall = []
        qdot_seed = np.zeros(8)
        for i, twist in enumerate(rec["v_cmd"]):
            step = inner.update(
                twist,
                rec["dt"],
                q_meas=rec["q_meas"][i],
                qdot_ff=qdot_seed,
                rail_exec_vel_m_s=float(rec["rail_v"][i]),
                dt_wall_s=rec["dt"],
                seed_q_cmd=True,
            )
            qdot_seed = rec["qdot"][i]
            q_cmd.append(inner.q_cmd.copy())
            slack.append(float(step.slack_norm))
            wall.append(int(bool(step.wall_active)))
        return {
            "q_cmd": np.asarray(q_cmd),
            "slack": np.asarray(slack),
            "wall": np.asarray(wall),
        }
    finally:
        if inner._native is not None:
            inner._native.shutdown()


@pytest.mark.skipif(find_wbc_rt_binary() is None, reason="wbc_rt binary not built")
def test_offline_ab_ellipse_gamepad_gates() -> None:
    v_cmd = _ellipse_gamepad_vcmd(0.005)
    py = _record_python(v_cmd)
    nt = _replay_native(py)
    dq = nt["q_cmd"] - py["q_cmd"]
    arm = np.abs(dq[:, 1:])
    rail_mm = np.abs(dq[:, 0]) * 1000.0
    p95_arm = float(np.percentile(arm, 95))
    p95_rail = float(np.percentile(rail_mm, 95))
    assert p95_arm < _ARM_P95_RAD, (
        f"arm q_cmd p95={p95_arm:.3e} rad max={arm.max():.3e} "
        f"rail_p95={p95_rail:.3f} mm"
    )
    assert p95_rail < _RAIL_MM, f"rail q_cmd p95={p95_rail:.3f} mm max={rail_mm.max():.3f}"
    py_s = np.maximum(py["slack"], 1e-9)
    nt_s = np.maximum(nt["slack"], 1e-9)
    ratio = np.median(nt_s / py_s)
    assert 0.1 < ratio < 10.0, f"slack ratio {ratio:.3f}"
    sign_agree = float(np.mean(py["wall"] == nt["wall"]))
    assert sign_agree >= 0.8, f"wall sign agree {sign_agree:.2f}"
