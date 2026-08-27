"""Deterministic integration timebase and slim command-step clamp."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkConfig,
    JointIkController,
    _CStateGuard,
    _pin_control_cpu,
    _set_realtime_priority,
    reference_time_step,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, full_q_from_arm
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig
from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.utils.safety import (
    clamp_command_step,
    integration_period,
)


Q_SAFE = full_q_from_arm(
    np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), 0.40
)


def test_integration_period_clips_overrun_and_short_ticks() -> None:
    dt_nom = 0.007
    assert integration_period(dt_nom, None) == pytest.approx(dt_nom)
    assert integration_period(dt_nom, 0.006) == pytest.approx(dt_nom)
    assert integration_period(dt_nom, 0.008) == pytest.approx(0.008)
    assert integration_period(dt_nom, 0.020) == pytest.approx(1.25 * dt_nom)


def test_command_step_clamp_bounds_ddq() -> None:
    q_prev = np.zeros(3)
    dq_prev = np.array([0.001, 0.0, 0.0])
    a_max = np.array([3.0, 3.0, 3.0])
    dt_nom = 0.007
    q_desired = q_prev + np.array([0.050, 0.0, 0.0])
    q_safe, dq, acc_clamped = clamp_command_step(
        q_prev, q_desired, dq_prev, a_max, dt_nom
    )
    assert acc_clamped
    assert abs(dq[0] - dq_prev[0]) <= a_max[0] * dt_nom * dt_nom + 1.0e-15
    assert q_safe[0] == pytest.approx(dq[0])


def _ptp_controller() -> JointIkController:
    qp = QpConfig(backend="proxqp", collision=CollisionConfig(enabled=False))
    cfg = JointIkConfig(
        dt=0.005,
        control_frame="base",
        qp=qp,
        collision=CollisionConfig(enabled=False),
        a_max_arm_rad_s2=3.0,
        a_max_rail_m_s2=0.60,
    )
    controller = JointIkController(RobotKinematics(), cfg)
    controller.reset(Q_SAFE)
    controller.set_direct_joint_ptp(True)
    return controller


def test_jittering_dt_wall_keeps_integration_timebase_constant() -> None:
    controller = _ptp_controller()
    dt_nom = float(controller.cfg.dt)
    qdot_ff = np.zeros(8)
    qdot_ff[2] = 0.010
    walls = [0.004, 0.006, 0.012, 0.005, 0.009]
    expected = [integration_period(dt_nom, w) for w in walls]
    q = controller.q_cmd.copy()
    travelled = 0.0
    for dt_wall, dt_int in zip(walls, expected):
        q_before = controller.q_cmd.copy()
        step = controller.update(
            np.zeros(6),
            dt_nom,
            q_meas=q,
            qdot_ff=qdot_ff,
            dt_wall_s=dt_wall,
        )
        dq2 = float(controller.q_cmd[2] - q_before[2])
        assert dq2 == pytest.approx(step.qdot[2] * dt_int, abs=1.0e-9)
        travelled += dq2
        q = controller.q_cmd.copy()
        assert np.isfinite(step.qdot).all()
        assert np.all(step.qdot >= controller.core.last_lo_box - 1.0e-9)
        assert np.all(step.qdot <= controller.core.last_hi_box + 1.0e-9)
    assert travelled <= qdot_ff[2] * sum(expected) + 1.0e-9


def test_box_periods_use_the_clipped_integration_period() -> None:
    qp = QpConfig(backend="proxqp", collision=CollisionConfig(enabled=False))
    cfg = JointIkConfig(
        dt=0.007,
        control_frame="base",
        qp=qp,
        collision=CollisionConfig(enabled=False),
        a_max_arm_rad_s2=3.0,
        a_max_rail_m_s2=0.60,
    )
    controller = JointIkController(RobotKinematics(), cfg)
    controller.reset(Q_SAFE)
    seen: list[float] = []
    real = controller._measure_box_periods

    def _wrap(dt: float) -> tuple[float, float | None]:
        seen.append(float(dt))
        return real(dt)

    controller._measure_box_periods = _wrap  # type: ignore[method-assign]
    controller.update(
        np.array([0.0, 0.02, 0.0, 0.0, 0.0, 0.0]),
        0.007,
        q_meas=Q_SAFE,
        dt_wall_s=0.014,
    )
    assert seen == [pytest.approx(1.25 * 0.007)]


def test_rt_helpers_are_best_effort() -> None:
    assert _set_realtime_priority() in (True, False)
    assert _pin_control_cpu(None) is False
    try:
        before = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        before = None
    assert _pin_control_cpu(0) in (True, False)
    if before is not None:
        try:
            os.sched_setaffinity(0, before)
        except (OSError, PermissionError, AttributeError):
            pass
    with _CStateGuard() as guard:
        assert isinstance(guard.active, bool)


def test_yaml_dt_ms_is_5ms() -> None:
    from pathlib import Path

    import yaml

    raw = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml").read_text()
    )
    assert float(raw["timing"]["dt_ms"]) == pytest.approx(5.0)


def test_reference_time_step_follows_elapsed_wall() -> None:
    assert reference_time_step(0.00664, 1.0) == pytest.approx(0.00664)
    assert reference_time_step(0.00664, 0.5) == pytest.approx(0.00332)
    assert reference_time_step(0.0, 1.0) == pytest.approx(0.0)
    assert reference_time_step(float("nan"), 1.0) == pytest.approx(0.0)


def test_period_ladder_holds_until_slack_passes() -> None:
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "apps"
        / "joint_admittance_8dof"
        / "analyze_qpik_quality.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_qpik_quality", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    assert mod.next_period_ms(7.0, 0.98) == pytest.approx(7.0)
    assert mod.next_period_ms(7.0, 0.99) == pytest.approx(6.0)
    assert mod.next_period_ms(6.0, 0.99) == pytest.approx(5.0)
    assert mod.next_period_ms(5.0, 1.0) == pytest.approx(5.0)
    assert mod.raise_period_ms(5.0) == pytest.approx(7.0)
    assert mod.PERIOD_LADDER_MS == (7.0, 6.0, 5.0)


def test_step_clamp_flags_acc_clamped_on_large_qdot_jump() -> None:
    controller = _ptp_controller()
    dt_nom = float(controller.cfg.dt)
    q = controller.q_cmd.copy()
    slow = np.zeros(8)
    slow[2] = 0.002
    controller.update(np.zeros(6), dt_nom, q_meas=q, qdot_ff=slow)
    q = controller.q_cmd.copy()
    jump = np.zeros(8)
    jump[2] = 0.80
    step = controller.update(np.zeros(6), dt_nom, q_meas=q, qdot_ff=jump)
    # The direct FF jump is rejected by the QP's acceleration/jerk box before
    # publication.  No second post-QP clamp is needed or allowed to rewrite
    # the certified command.
    assert step.solver_fault_latched
    assert step.fallback_level == "stop"
    np.testing.assert_allclose(step.qdot, 0.0, atol=1.0e-12)
    assert not step.post_step_clamp_applied


def test_tick_logger_writes_step_controller_mode_and_ab_fields(tmp_path) -> None:
    import csv

    from rm75_control.control.joint_admittance_8dof.loop import JointIkStep, _TickLogger

    path = tmp_path / "wbc.csv"
    logger = _TickLogger(str(path), verbose_json=True)
    step = JointIkStep(
        q_send=np.linspace(0.1, 0.8, 8),
        qdot=np.zeros(8),
        twist_base=np.zeros(6),
        sigma_min=0.1,
        manip=1.0,
        slack_norm=0.0,
        n_cbf_active=0,
        follow_err_rad=0.0,
        controller_mode="qpik",
        post_qp_step_clamp_enabled=True,
        post_step_would_clamp=True,
        post_step_clamp_applied=False,
        qdot_raw=np.full(8, 0.01),
        qdot_pre_commit=np.full(8, 0.01),
        qdot_committed=np.full(8, 0.01),
        arm_send_mono_ns=2_000_000_000,
    )
    logger.write(
        0.012345678,
        "ellipse_track",
        0.0,
        step,
        np.zeros(8),
        np.zeros(6),
        np.zeros(6),
    )
    logger.close()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    row = rows[0]
    assert list(row.keys()) == list(_TickLogger._HEADER)
    assert row["controller_mode"] == "qpik"
    assert row["post_qp_step_clamp_enabled"] == "1"
    assert row["post_step_would_clamp"] == "1"
    assert row["post_step_clamp_applied"] == "0"
    assert row["arm_send_mono_ns"] == "2000000000"
    assert "0.01" in row["qpik_qdot_raw_json"]


def test_ab_lomb_scargle_and_verdicts() -> None:
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "apps"
        / "joint_admittance_8dof"
        / "analyze_qpik_quality.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_qpik_quality", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    t = np.linspace(0.0, 1.0, 64)
    y = np.sin(2.0 * np.pi * 60.0 * t)
    power_60 = float(mod.band_power(t, y, 55.0, 65.0))
    power_20 = float(mod.band_power(t, np.sin(2.0 * np.pi * 20.0 * t), 55.0, 65.0))
    assert power_60 > power_20
    assert mod.hysteresis_flip_count(np.array([0.2, -0.2, 0.2]), 0.05) == 2

    def _rows(power_scale: float, flips: int) -> list[dict]:
        n = 32
        rows = []
        for i in range(n):
            qdot = [0.0] * 7
            qdot[0] = power_scale * math.sin(2.0 * np.pi * 60.0 * i * 0.005)
            if i % 4 == 0 and i < 2 * flips:
                qdot[1] = 2.0 if (i // 4) % 2 == 0 else -2.0
            rows.append(
                {
                    "qpik_qp1_status": "solved",
                    "qpik_qp2_status": "solved",
                    "qpik_qp2_fallback": "0",
                    "qpik_solver_fault_latched": "0",
                    "qpik_fallback_reason": "",
                    "psi_ref_deg": "68.0",
                    "qpik_qdot_raw_json": "[0,0,0,0,0,0,0,0]",
                    "qpik_qdot_pre_commit_json": "[0,0,0,0,0,0,0,0]",
                    "arm_send_mono_ns": str(1_000_000_000 + i * 5_000_000),
                    "arm_qdot_target_wall_json": str(qdot).replace(" ", ""),
                }
            )
        return rows

    amp = mod.evaluate_post_qp_ab(_rows(1.0, 8), _rows(0.2, 2), _rows(1.0, 8))
    assert amp["verdict"] in {"amplifier", "inconclusive"}
    drift = mod.evaluate_post_qp_ab(_rows(1.0, 8), _rows(0.2, 2), _rows(2.0, 8))
    assert drift["verdict"] == "no_conclusion"
    bad = mod.evaluate_post_qp_ab(
        _rows(1.0, 8),
        [{**_rows(0.2, 2)[0], "qpik_qp2_fallback": "1"}],
        _rows(1.0, 8),
    )
    assert bad["verdict"] == "invalid"


def test_commit_does_not_rewrite_qdot_prev2() -> None:
    controller = _ptp_controller()
    q_prev = controller.q_cmd.copy()
    controller.q_cmd = q_prev + np.array([0.0, 0.0, 0.01, 0, 0, 0, 0, 0])
    prev2 = np.linspace(0.1, 0.8, 8)
    controller.core.qdot_prev2 = prev2.copy()
    seen = np.linspace(0.2, 0.9, 8)
    controller.core._qdot_prev_seen = seen.copy()
    controller._commit_command_step(q_prev, 0.005, 0.005)
    assert np.allclose(controller.core.qdot_prev2, prev2)
    assert np.allclose(controller.core._qdot_prev_seen, seen)
    assert np.isfinite(controller.core.qdot_prev).all()
