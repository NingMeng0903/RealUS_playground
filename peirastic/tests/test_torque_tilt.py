"""Torque-guided tool-ωy tilt: keep CoP, stick leftover, hold in air."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PLAYGROUND = Path(__file__).resolve().parents[2]
if str(_PLAYGROUND) not in sys.path:
    sys.path.insert(0, str(_PLAYGROUND))

import numpy as np
import pytest
import yaml

from peirastic.core.modes import Mode
from peirastic.realman8dof.force.fce import FceAdmittanceLaw
from peirastic.realman8dof.force.legacy import LegacyForceLaw
from peirastic.realman8dof.force.torque_tilt import (
    LegacyForceWithTilt,
    TorqueTilt,
    TorqueTiltConfig,
    apply_tilt_selection,
)
from peirastic.realman8dof.modes.track import HybridTffOuter
from peirastic.realman8dof.session import compile_request
from peirastic.configs import DEFAULT_CONTROLLER_YAML, DEFAULT_FORCE_YAML
from rm75_control.control.joint_admittance_8dof.api import CompileContext
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics

DT = 0.005
_SEED = np.array([0.375, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370])


def _tilt(**over) -> TorqueTilt:
    cfg = TorqueTiltConfig(a_max=50.0, j_max=800.0, **over)
    return TorqueTilt(cfg)


def _step(tilt: TorqueTilt, f_ext: np.ndarray, f_des: np.ndarray, *, contact=True):
    return tilt.update(f_ext, f_des, dt_s=DT, contact=contact)


def test_leftover_torque_sticks() -> None:
    tilt = _tilt()
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    leftover = np.array([0.0, 0.0, 2.0, 0.0, 0.020, 0.0])
    out = 0.0
    for _ in range(40):
        out = _step(tilt, leftover, f_des)
    assert abs(out) < 1e-9
    assert tilt.stuck is True


def test_cop_moment_is_kept_and_rotates_against_tau() -> None:
    tilt = _tilt()
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    wrench = np.array([0.0, 0.0, 2.0, 0.0, 0.12, 0.0])
    out = 0.0
    for _ in range(30):
        out = _step(tilt, wrench, f_des)
    assert out < -1e-3
    assert tilt.tau_y == pytest.approx(0.12)
    assert tilt.telemetry()["omega_y"] < -1e-3


def test_air_does_not_tilt() -> None:
    tilt = _tilt()
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    wrench = np.array([0.0, 0.0, 2.0, 0.0, 0.12, 0.0])
    out = 0.0
    for _ in range(20):
        out = _step(tilt, wrench, f_des, contact=False)
    assert abs(out) < 1e-9
    assert tilt.engaged is False


def test_large_force_error_scales_without_killing() -> None:
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    cop = np.array([0.0, 0.0, 2.0, 0.0, 0.12, 0.0])
    far = np.array([0.0, 0.0, -1.0, 0.0, 0.12, 0.0])
    full = _tilt()
    muted = _tilt()
    out_full = 0.0
    out_far = 0.0
    for _ in range(30):
        out_full = _step(full, cop, f_des)
        out_far = _step(muted, far, f_des)
    assert out_full < -1e-3
    assert abs(out_far) < 0.25 * abs(out_full)
    assert muted.engaged is True


def test_force_error_flicker_does_not_reset() -> None:
    tilt = _tilt()
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    cop = np.array([0.0, 0.0, 2.0, 0.0, 0.12, 0.0])
    far = np.array([0.0, 0.0, -1.0, 0.0, 0.12, 0.0])
    for _ in range(20):
        _step(tilt, cop, f_des)
    omega_before = float(tilt.omega_y)
    theta_before = float(tilt.theta_tilt)
    assert omega_before < -1e-3
    for _ in range(4):
        _step(tilt, far, f_des)
    out = _step(tilt, cop, f_des)
    assert out < 0.0
    assert abs(out) > 0.5 * abs(omega_before)
    assert abs(tilt.theta_tilt) >= abs(theta_before) - 1e-4


def test_rising_contact_resets_tool_theta() -> None:
    tilt = _tilt(theta_max_rad=0.02)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    wrench = np.array([0.0, 0.0, 2.0, 0.0, 0.20, 0.0])
    for _ in range(80):
        _step(tilt, wrench, f_des)
    assert abs(tilt.theta_tilt) == pytest.approx(0.02, abs=1e-4)
    _step(tilt, wrench, f_des, contact=False)
    _step(tilt, wrench, f_des, contact=True)
    assert abs(tilt.theta_tilt) < 5e-3


def test_angle_limit_blocks_outward_rate() -> None:
    tilt = _tilt(theta_max_rad=0.02, vmax_rad_s=0.40)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    wrench = np.array([0.0, 0.0, 2.0, 0.0, 0.20, 0.0])
    for _ in range(80):
        _step(tilt, wrench, f_des)
    assert tilt.theta_tilt == pytest.approx(-0.02, abs=1e-4)
    out = _step(tilt, wrench, f_des)
    assert out >= -1e-6


def test_yaml_defaults_match_plan() -> None:
    raw = yaml.safe_load(DEFAULT_FORCE_YAML.read_text(encoding="utf-8"))
    cfg = TorqueTiltConfig.from_dict(raw)
    assert cfg.enabled is True
    assert cfg.axis == 4
    assert cfg.damping == pytest.approx(0.30)
    assert cfg.coulomb_nm == pytest.approx(0.025)
    assert cfg.theta_max_rad == pytest.approx(0.52)
    assert cfg.engage_err_n == pytest.approx(0.25)
    assert cfg.engage_fade_n == pytest.approx(2.0)


def test_apply_tilt_selection_opens_wy_only_on_default_mask() -> None:
    cfg = TorqueTiltConfig()
    opened = apply_tilt_selection(None, cfg)
    assert np.allclose(opened, [1, 1, 0, 1, 0, 1])
    explicit = apply_tilt_selection(np.array([1, 1, 0, 1, 1, 1]), cfg)
    assert np.allclose(explicit, [1, 1, 0, 1, 1, 1])
    off = apply_tilt_selection(None, TorqueTiltConfig(enabled=False))
    assert np.allclose(off, [1, 1, 0, 1, 1, 1])


def _ctx():
    raw = yaml.safe_load(DEFAULT_CONTROLLER_YAML.read_text())
    cfg = build_joint_ik_config(raw)
    cfg.backend = "python"
    cfg.native_shm_prefix = f"rm75_wbc_tilt_{os.getpid()}"
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    inner.reset(_SEED)
    ctx = CompileContext(
        kin=kin,
        inner=inner,
        euler_order=cfg.euler_order,
        control_frame=cfg.control_frame,
        v_scale=cfg.v_scale,
    )
    return raw, ctx


def test_hybrid_tff_opens_wy_and_masks_path() -> None:
    from peirastic.core.modes import ModeRequest

    raw, ctx = _ctx()
    phase = compile_request(
        ctx,
        ModeRequest(
            Mode.TRACK_HYBRID,
            {"reference": "hold", "use_tff_split": True, "desired_z": 2.0},
        ),
        raw=raw,
    )
    assert isinstance(phase.outer, HybridTffOuter)
    assert np.allclose(phase.outer.selection, [1, 1, 0, 1, 0, 1])
    assert isinstance(phase.outer.force_law, LegacyForceWithTilt)
    assert isinstance(phase.outer.force_law.z_law, LegacyForceLaw)
    pose = ctx.kin.fk_pose(_SEED)
    phase.outer.set_origin(pose)
    pos = phase.outer.position

    def _path_sample(_t, current, _f):
        v = np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.0])
        pos.last_path_twist = v.copy()
        pos.last_feedback_twist = np.zeros(6)
        pos.last_vel_ff = v.copy()
        pos.last_pose_d = np.asarray(current, dtype=float).copy()
        return v

    pos.sample = _path_sample
    wrench = np.array([0.0, 0.0, 2.0, 0.0, 0.12, 0.0])
    out = np.zeros(6)
    for i in range(25):
        out = np.asarray(
            phase.outer.sample(i * DT, pose, wrench, contact=True),
            dtype=float,
        )
    assert out[4] != pytest.approx(0.3)
    assert out[4] < 0.0


def test_hover_fce_is_not_wrapped() -> None:
    from peirastic.core.modes import ModeRequest

    raw, ctx = _ctx()
    phase = compile_request(
        ctx,
        ModeRequest(
            Mode.TRACK_HYBRID,
            {
                "reference": "hold",
                "law": "fce",
                "use_tff_split": True,
                "force_axes": [1, 1, 1, 1, 1, 1],
                "desired_force": [0.0] * 6,
            },
        ),
        raw=raw,
    )
    assert isinstance(phase.outer, HybridTffOuter)
    assert isinstance(phase.outer.force_law, FceAdmittanceLaw)
    assert not isinstance(phase.outer.force_law, LegacyForceWithTilt)
    assert np.allclose(phase.outer.selection, np.zeros(6))
