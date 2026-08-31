"""Gates for nullspace self-excitation: LPF, continuous manip weight, soft cap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.api import SecondaryPolicy
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof import loop as loop_mod
from rm75_control.control.joint_admittance_8dof.loop import JointIkController, _TickLogger
from rm75_control.control.joint_admittance_8dof.utils.safety import (
    clamp_command_step,
    integration_period,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.tasks.manipulability_task import (
    ManipulabilityTask,
    ManipulabilityTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.nullspace_task import (
    JointCenteringTask,
    NullspaceTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.secondary_composer import (
    SecondaryComposer,
    _soft_cap_per_joint,
)
from rm75_control.control.joint_admittance_8dof.teleop.gamepad_twist import (
    GamepadTwistConfig,
    normalize_trigger,
    map_pad_to_world_lin_tool_ang,
)
from rm75_control.control.joint_admittance_8dof.teleop.xbox_pad import FakePad


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "joint_admittance_8dof.yaml"

Q_D = np.array(
    [0.40, -0.949552, 0.095255, 0.646858, 1.469911, 0.502701, 0.666503, -0.338137]
)


def _inner(q: np.ndarray) -> JointIkController:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg = build_joint_ik_config(raw)
    cfg.backend = "python"
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    inner = JointIkController(RobotKinematics(), cfg)
    inner.reset(q)
    return inner


def test_yaml_parses_manipulability_qdot_tau() -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg = build_joint_ik_config(raw)
    assert cfg.manipulability.qdot_tau_s == pytest.approx(0.05)
    assert cfg.manipulability.grad_period_ticks == 10


def test_manipulability_qdot_is_first_order_smoothed() -> None:
    kin = RobotKinematics()

    class _Stepped(ManipulabilityTask):
        def __init__(self) -> None:
            super().__init__(
                kin,
                ManipulabilityTaskConfig(
                    k_mu=0.8, grad_period_ticks=1000, qdot_tau_s=0.05
                ),
            )
            self._forced = np.zeros(kin.nv)
            self._forced[2] = 1.0

        def _gradient_cached(self, q_rad, *, exclude_rail):
            self.last_grad_norm = 1.0
            return self._forced.copy()

    task = _Stepped()
    q = Q_D.copy()
    first = task(q, exclude_rail=True, dt_s=0.005, sigma_min=0.05)
    task._forced = -task._forced
    seq = [first]
    for _ in range(8):
        seq.append(task(q, exclude_rail=True, dt_s=0.005, sigma_min=0.05))
    jumps = [float(np.linalg.norm(seq[i + 1] - seq[i])) for i in range(len(seq) - 1)]
    assert max(jumps) < 0.35
    assert max(jumps) > 1.0e-4


def test_manip_weight_half_scales_term() -> None:
    kin = RobotKinematics()
    centering = JointCenteringTask.from_kinematics(
        kin, NullspaceTaskConfig(k_center=1.0, k_limit=2.0, activation=0.75)
    )
    manip = ManipulabilityTask(
        kin, ManipulabilityTaskConfig(k_mu=0.8, qdot_tau_s=0.0, grad_period_ticks=1)
    )
    composer = SecondaryComposer(
        centering, None, manipulability=manip, v_max=kin.v_max, max_qdot_frac=0.0
    )
    q = Q_D.copy()
    off = composer.compose(
        q,
        None,
        np.zeros(8),
        arm_suppressed=True,
        manipulability_active=False,
        sigma_min=0.05,
        centering_sigma_fade=False,
        dt_s=0.005,
    )
    full = composer.compose(
        q,
        None,
        np.zeros(8),
        arm_suppressed=True,
        manipulability_active=True,
        sigma_min=0.05,
        centering_sigma_fade=False,
        dt_s=0.005,
    )
    half = composer.compose(
        q,
        None,
        np.zeros(8),
        arm_suppressed=True,
        manipulability_active=0.5,
        sigma_min=0.05,
        centering_sigma_fade=False,
        dt_s=0.005,
    )
    delta_full = full - off
    delta_half = half - off
    assert np.linalg.norm(delta_full) > 1.0e-6
    np.testing.assert_allclose(delta_half, 0.5 * delta_full, rtol=1e-6, atol=1e-9)


def test_soft_cap_is_c1_and_hard_ceiling() -> None:
    cap = np.full(8, 0.628)
    raw = np.array([0.10, 0.55, 0.62, 0.80, -0.70, 0.0, 0.628, 1.5])
    out = _soft_cap_per_joint(raw, cap)
    assert float(out[0]) == pytest.approx(0.10)
    assert abs(float(out[3])) <= 0.628 + 1e-12
    assert abs(float(out[7])) <= 0.628 + 1e-12
    assert 0.55 < float(out[1]) <= 0.628
    nearby = _soft_cap_per_joint(raw + 1.0e-4, cap)
    assert float(np.max(np.abs(nearby - out))) < 5.0e-4


def test_trigger_deadzone_kills_rest_noise() -> None:
    assert normalize_trigger(-1.0, 0.08) == 0.0
    assert normalize_trigger(-0.90, 0.08) == 0.0  # ~0.05 after map
    assert normalize_trigger(1.0, 0.08) == pytest.approx(1.0)
    pad = FakePad(axes=np.array([0.0, 0.0, -0.95, 0.0, 0.0, -0.95]))
    v, w = map_pad_to_world_lin_tool_ang(pad.read(), GamepadTwistConfig(trigger_deadzone=0.08))
    assert float(np.linalg.norm(v)) == 0.0
    assert float(np.linalg.norm(w)) == 0.0


def test_tick_logger_has_nullspace_terms() -> None:
    header = _TickLogger._HEADER
    for name in (
        "qpik_nullspace_norm",
        "qpik_nullspace_centering_norm",
        "qpik_nullspace_manip_norm",
        "qpik_nullspace_arm_angle_norm",
        "qpik_nullspace_damping_norm",
        "qpik_nullspace_rail_lock_norm",
        "qpik_sat_scale",
        "qpik_sec_target_norm",
    ):
        assert name in header
    assert len(header) == len(set(header))


def _arm_jerk_rms(qs: np.ndarray, dt: float) -> float:
    j = np.diff(np.asarray(qs, dtype=float), n=3, axis=0) / (dt ** 3)
    return float(np.sqrt(np.mean(j[:, 1:] ** 2)))


def test_zero_twist_nullspace_is_quieter_with_lpf() -> None:
    """Nullspace must not self-excite when the Cartesian command is zero."""

    def _run(tau: float, *, force_manip: bool) -> tuple[float, float]:
        inner = _inner(Q_D.copy())
        inner.cfg.manipulability.qdot_tau_s = tau
        if inner.manipulability_task is not None:
            inner.manipulability_task.cfg.qdot_tau_s = tau
            inner.manipulability_task.reset()
        SecondaryPolicy(preset="track").apply(inner)
        if force_manip:
            inner.set_manipulability_active(True)
        q = inner.q_cmd.copy()
        qs = [q.copy()]
        for _ in range(400):
            inner.update(np.zeros(6), dt=inner.cfg.dt, q_meas=q.copy())
            q = inner.q_cmd.copy()
            qs.append(q.copy())
        qs_a = np.asarray(qs)
        jerk = _arm_jerk_rms(qs_a, float(inner.cfg.dt))
        kin = inner.kin
        drift_mm = float(
            np.linalg.norm(kin.fk_pose(qs_a[-1])[:3] - kin.fk_pose(qs_a[0])[:3]) * 1000.0
        )
        return jerk, drift_mm

    jerk_idle, drift_idle = _run(0.05, force_manip=False)
    jerk_forced, drift_forced = _run(0.05, force_manip=True)
    # Diagnosed idle self-excitation (run_20260819_055437): arm jerk RMS 214.5,
    # TCP p95 2.97 mm / max 34.6 mm.  LPF vs raw q-jerk is <1% here; the
    # command staircase is gated in test_manipulability_qdot_is_first_order_smoothed.
    assert jerk_idle < 120.0
    assert jerk_forced < 120.0
    assert drift_idle < 15.0
    assert drift_forced < 15.0


def test_nullspace_enter_fade_is_smoothstep() -> None:
    inner = _inner(Q_D.copy())
    inner.set_centering_suppressed(True)
    inner.set_centering_suppressed(False)
    assert inner._nullspace_enter_scale(0.0) == pytest.approx(0.0)
    s = 0.0
    for _ in range(60):
        s = inner._nullspace_enter_scale(0.005)
    assert 0.35 < s < 0.65
    for _ in range(80):
        s = inner._nullspace_enter_scale(0.005)
    assert s == pytest.approx(1.0)


def test_post_qp_clamp_uses_dt_nom(monkeypatch) -> None:
    """Box / collapse / integrate / post-QP clamp share T = dt_nom."""

    seen: list[float] = []
    real = clamp_command_step

    def _spy(q_prev, q_desired, dq_prev, a_max, dt_nom):
        seen.append(float(dt_nom))
        return real(q_prev, q_desired, dq_prev, a_max, dt_nom)

    monkeypatch.setattr(loop_mod, "clamp_command_step", _spy)
    inner = _inner(Q_D.copy())
    SecondaryPolicy(preset="track").apply(inner)
    dt_nom = 0.005
    dt_wall = 0.007
    inner.update(
        np.zeros(6),
        dt=dt_nom,
        q_meas=inner.q_cmd.copy(),
        dt_wall_s=dt_wall,
    )
    assert seen
    assert seen[-1] == pytest.approx(dt_nom)
    assert seen[-1] != pytest.approx(integration_period(dt_nom, dt_wall))
