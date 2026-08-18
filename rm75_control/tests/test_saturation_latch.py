"""Saturation latch, governor floor, secondary fade, and HW-log replay."""

from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.api import GovernorSpec, SecondaryPolicy
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import (
    GovernorFilter,
    JointIkController,
    Phase,
    _TickLogger,
    _reference_governor_scale,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, deg2rad
from rm75_control.control.joint_admittance_8dof.saturation_latch import (
    SaturationConfig,
    SaturationLatch,
    predict_rail_position_m,
)
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
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "joint_admittance_8dof.yaml"
LOGS = ROOT / "apps" / "logs"
Q_LOWER = None
Q_UPPER = None


def _limits():
    global Q_LOWER, Q_UPPER
    if Q_LOWER is None:
        kin = RobotKinematics()
        Q_LOWER = np.asarray(kin.q_lower, dtype=float)
        Q_UPPER = np.asarray(kin.q_upper, dtype=float)
    return Q_LOWER, Q_UPPER


def _update(
    latch: SaturationLatch,
    q: np.ndarray,
    slack: float,
    dt: float,
    **kwargs,
):
    lo, hi = _limits()
    return latch.update(
        q_cmd=q,
        q_lower=lo,
        q_upper=hi,
        rail_soft_min_m=float(kwargs.get("rail_soft_min_m", 0.030)),
        rail_soft_max_m=float(kwargs.get("rail_soft_max_m", 0.755)),
        near_arm_margin_rad=float(kwargs.get("near_arm_margin_rad", 0.08)),
        branch_eps_rad=float(kwargs.get("branch_eps_rad", 0.35)),
        slack_norm=float(slack),
        dt_s=float(dt),
    )


def _mid_q() -> np.ndarray:
    lo, hi = _limits()
    q = 0.5 * (lo + hi)
    q[0] = 0.40
    return q


def test_slack_alone_never_latches() -> None:
    latch = SaturationLatch()
    q = _mid_q()
    for _ in range(400):
        flags = _update(latch, q, 0.20, 0.005)
    assert not flags.pinned
    assert flags.slack_over
    assert not flags.cannot_follow


def test_pin_without_slack_never_latches() -> None:
    latch = SaturationLatch()
    q = _mid_q()
    q[4] = 0.35
    for _ in range(400):
        flags = _update(latch, q, 0.001, 0.005)
    assert flags.near_branch
    assert flags.pinned
    assert not flags.cannot_follow


def test_and_dwell_then_hysteresis_exit() -> None:
    latch = SaturationLatch()
    q = _mid_q()
    q[4] = 0.35
    for _ in range(20):  # 100 ms < 150 ms
        flags = _update(latch, q, 0.04, 0.005)
    assert not flags.cannot_follow
    for _ in range(20):  # +100 ms
        flags = _update(latch, q, 0.04, 0.005)
    assert flags.cannot_follow
    flags = _update(latch, q, 0.020, 0.005)
    assert flags.cannot_follow  # still above exit 0.015
    flags = _update(latch, q, 0.010, 0.005)
    assert not flags.cannot_follow


def test_near_rail_uses_soft_not_hard() -> None:
    latch = SaturationLatch()
    q = _mid_q()
    q[0] = 0.760
    flags = _update(latch, q, 0.0, 0.005)
    assert flags.near_rail
    flags_hard = _update(
        latch, q, 0.0, 0.005, rail_soft_min_m=0.005, rail_soft_max_m=0.78
    )
    assert not flags_hard.near_rail


def test_predict_rail_position_coasts_without_filter() -> None:
    pred = predict_rail_position_m(0.401, 0.024, 0.0187)
    assert pred == pytest.approx(0.401 + 0.024 * 0.0187)
    clipped = predict_rail_position_m(0.401, 0.50, 1.00, lo_m=0.005, hi_m=0.78)
    assert clipped == pytest.approx(0.78)


def test_governor_floor_zero_only_when_saturated() -> None:
    phase = Phase(
        outer=SimpleNamespace(),
        governor_err_ok_mm=10.0,
        governor_err_max_mm=40.0,
        governor_scale_min=0.25,
        governor_joint_err_max_deg=0.0,
    )
    assert _reference_governor_scale(
        phase, outer_err_mm=80.0, joint_err_deg=None, physical_saturated=False
    ) == 1.0
    assert _reference_governor_scale(
        phase, outer_err_mm=80.0, joint_err_deg=None, physical_saturated=True
    ) == pytest.approx(0.05)
    mid = _reference_governor_scale(
        phase, outer_err_mm=25.0, joint_err_deg=None, physical_saturated=True
    )
    assert mid == pytest.approx(0.5)


def test_governor_freeze_timeout_defaults() -> None:
    assert GovernorSpec().freeze_timeout_s == 9.0
    phase = Phase(outer=SimpleNamespace())
    assert phase.governor_freeze_timeout_s == 9.0
    assert phase.governor_crawl_floor == pytest.approx(0.05)
    filt = GovernorFilter(tau_s=0.05, freeze_below=0.02, release_above=0.10, scale_min=0.0)
    for _ in range(400):
        out = filt.update(0.0, 0.005)
    assert out == 0.0 and filt.frozen


def test_tick_logger_header_has_saturation_and_nullspace_terms() -> None:
    header = _TickLogger._HEADER
    for name in (
        "physical_saturated",
        "sat_near_arm",
        "sat_near_rail",
        "sat_near_branch",
        "sat_slack_over",
        "sat_secondary_scale",
        "rail_task_alpha",
        "rail_margin_escape_active",
        "pad_twist_slewed",
        "governor_freeze_s",
        "governor_scale_raw",
        "qpik_nullspace_centering_norm",
        "qpik_nullspace_manip_norm",
        "qpik_nullspace_arm_angle_norm",
        "qpik_nullspace_damping_norm",
        "qpik_nullspace_rail_lock_norm",
    ):
        assert name in header
    assert len(header) == len(set(header))


def test_secondary_soft_scale_fades_soft_tasks_not_ff() -> None:
    kin = RobotKinematics()
    centering = JointCenteringTask.from_kinematics(
        kin, NullspaceTaskConfig(k_center=1.0, k_limit=2.0, activation=0.75)
    )
    composer = SecondaryComposer(
        centering, None, v_max=kin.v_max, max_qdot_frac=0.2
    )
    q = np.array([0.40, 0.0, -0.4, 0.2, 1.2, 0.1, 0.8, 0.0])
    full = composer.compose(q, None, np.zeros(8), arm_suppressed=True, soft_scale=1.0)
    fade = composer.compose(q, None, np.zeros(8), arm_suppressed=True, soft_scale=0.15)
    assert composer.last_centering_norm > 0.0
    assert np.linalg.norm(fade) == pytest.approx(0.15 * np.linalg.norm(full), rel=1e-6)
    ff = np.full(8, 0.05)
    with_ff = composer.compose(q, ff, np.zeros(8), arm_suppressed=True, soft_scale=0.15)
    assert np.linalg.norm(with_ff - ff) == pytest.approx(np.linalg.norm(fade), rel=1e-6)


def test_manipulability_output_is_first_order_smoothed() -> None:
    kin = RobotKinematics()

    class _Stepped(ManipulabilityTask):
        def __init__(self) -> None:
            super().__init__(
                kin,
                ManipulabilityTaskConfig(
                    k_mu=0.8, grad_period_ticks=10, qdot_tau_s=0.05
                ),
            )
            self._forced = np.zeros(kin.nv)
            self._forced[2] = 1.0

        def _gradient_cached(self, q_rad, *, exclude_rail):
            self.last_grad_norm = 1.0
            return self._forced.copy()

    task = _Stepped()
    q = deg2rad(np.array([0.0, -90.0, -30.0, 60.0, 90.0, 90.0, 60.0, 90.0]))
    first = task(q, exclude_rail=True, dt_s=0.005, sigma_min=0.05)
    task._forced = -task._forced
    seq = [first]
    for _ in range(8):
        seq.append(task(q, exclude_rail=True, dt_s=0.005, sigma_min=0.05))
    jumps = [float(np.linalg.norm(seq[i + 1] - seq[i])) for i in range(len(seq) - 1)]
    assert max(jumps) < 0.35
    assert max(jumps) > 1.0e-4


_REPLAY_CACHE: dict[str, dict] = {}


def _replay_cached(path: Path) -> dict:
    key = str(path)
    if key not in _REPLAY_CACHE:
        _REPLAY_CACHE[key] = _replay_csv(path)
    return _REPLAY_CACHE[key]


def _replay_csv(path: Path) -> dict:
    latch = SaturationLatch()
    first_t = None
    first_latch_t = None
    longest_and_s = 0.0
    and_s = 0.0
    any_latch = False
    t_prev = None
    track = []
    lo, hi = _limits()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                t = float(row.get("t_wall_s", ""))
            except ValueError:
                continue
            if not math.isfinite(t):
                continue
            if first_t is None:
                first_t = t
            dt = 0.005 if t_prev is None else max(t - t_prev, 0.0)
            t_prev = t
            q = np.zeros(8)
            for i in range(8):
                raw = row.get(f"q_cmd_{i}") or row.get(f"q_meas_{i}") or ""
                try:
                    q[i] = float(raw)
                except ValueError:
                    q[i] = float("nan")
            try:
                slack = float(row.get("slack_norm") or 0.0)
            except ValueError:
                slack = 0.0
            flags = latch.update(
                q_cmd=q,
                q_lower=lo,
                q_upper=hi,
                rail_soft_min_m=0.030,
                rail_soft_max_m=0.755,
                near_arm_margin_rad=0.08,
                branch_eps_rad=0.35,
                slack_norm=slack,
                dt_s=dt if dt > 0.0 else 0.005,
            )
            if flags.pinned and flags.slack_over:
                and_s += dt if dt > 0.0 else 0.005
                longest_and_s = max(longest_and_s, and_s)
            else:
                and_s = 0.0
            if flags.cannot_follow:
                any_latch = True
                if first_latch_t is None:
                    first_latch_t = t - first_t
            err = row.get("track_err_mm") or row.get("motion_err_rms_mm") or ""
            try:
                ev = float(err)
            except ValueError:
                ev = float("nan")
            if math.isfinite(ev):
                track.append(ev)
    arr = np.asarray(track, dtype=float) if track else np.array([])
    return {
        "any_latch": any_latch,
        "first_latch_s": first_latch_t,
        "longest_and_s": longest_and_s,
        "track_p50": float(np.percentile(arr, 50)) if arr.size else float("nan"),
        "track_p95": float(np.percentile(arr, 95)) if arr.size else float("nan"),
    }


@pytest.mark.parametrize(
    "rel",
    [
        "ellipse_track/run_20260819_022330.csv",
        "ellipse_track/run_20260819_022459.csv",
        "gamepad_vcmd/run_20260819_022056.csv",
    ],
)
def test_healthy_hw_logs_never_latch(rel: str) -> None:
    path = LOGS / rel
    if not path.is_file():
        pytest.skip(f"missing HW log {path}")
    report = _replay_cached(path)
    assert report["any_latch"] is False
    assert report["longest_and_s"] < 0.15


def test_runaway_022415_latches_before_40mm() -> None:
    path = LOGS / "ellipse_track" / "run_20260819_022415.csv"
    if not path.is_file():
        pytest.skip(f"missing HW log {path}")
    report = _replay_cached(path)
    assert report["any_latch"] is True
    assert 11.5 <= float(report["first_latch_s"]) <= 13.0
    assert report["longest_and_s"] > 1.0


def test_022330_historical_track_stays_submillimetre() -> None:
    path = LOGS / "ellipse_track" / "run_20260819_022330.csv"
    if not path.is_file():
        pytest.skip(f"missing HW log {path}")
    report = _replay_cached(path)
    assert report["track_p50"] == pytest.approx(0.26, abs=0.08)
    assert report["track_p95"] == pytest.approx(0.79, abs=0.15)


def test_closed_loop_unreachable_y_latches_and_caps_error() -> None:
    raw = yaml.safe_load(CONFIG.read_text())
    cfg = build_joint_ik_config(raw)
    cfg.collision.enabled = False
    cfg.ird.enabled = False
    cfg.control_frame = "base"
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    q0 = np.array(
        [0.77, -0.949552, 0.095255, 0.646858, 1.469911, 0.502701, 0.666503, -0.338137]
    )
    inner.reset(q0)
    SecondaryPolicy(preset="track").apply(inner)
    dt = float(inner.cfg.dt)
    m0 = kin.fk_placement(q0)
    p0 = m0.translation.copy()
    phase = Phase(
        outer=SimpleNamespace(),
        governor_err_ok_mm=10.0,
        governor_err_max_mm=40.0,
        governor_scale_min=0.25,
        governor_joint_err_max_deg=0.0,
        governor_freeze_timeout_s=3.0,
    )
    gov = GovernorFilter(tau_s=0.05, freeze_below=0.02, release_above=0.10, scale_min=0.25)
    t_ref = 0.0
    freeze_s = 0.0
    errs: list[float] = []
    saw_latch = False
    for _ in range(700):
        q = inner.q_cmd
        mc = kin.fk_placement(q)
        s = min(0.04 * t_ref, 0.40)
        p_d = p0 + np.array([0.0, s, 0.0])
        perr = p_d - mc.translation
        err_mm = float(np.linalg.norm(perr)) * 1000.0
        twist = np.zeros(6)
        twist[:3] = np.clip(10.0 * perr, -0.15, 0.15)
        step = inner.update(twist, dt, q_meas=q.copy())
        raw_scale = _reference_governor_scale(
            phase,
            outer_err_mm=err_mm,
            joint_err_deg=None,
            physical_saturated=bool(step.physical_saturated),
        )
        gov.scale_min = 0.05 if step.physical_saturated else 0.25
        scale = gov.update(raw_scale, dt)
        if step.physical_saturated and (gov.frozen or float(scale) <= 1.0e-9):
            freeze_s += dt
        else:
            freeze_s = 0.0
        t_ref += dt * float(scale)
        errs.append(err_mm)
        saw_latch = saw_latch or bool(step.physical_saturated)
        if freeze_s >= 3.0:
            break
    assert saw_latch
    assert max(errs) < 80.0
    assert max(errs) > 5.0


def test_poll_hz_probe_follows_synthetic_tone() -> None:
    path = (
        ROOT
        / "apps"
        / "joint_admittance_8dof"
        / "analyze_qpik_quality.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_qpik_quality", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    dt = 0.005
    n = 400

    def _rows(freq: float) -> list[dict]:
        w = 2.0 * math.pi * freq
        rows = []
        for i in range(n):
            t = i * dt
            q = -0.002 / (w * w) * math.sin(w * t)
            row = {f"q_cmd_{j}": q for j in range(8)}
            row["arm_send_mono_ns"] = int(t * 1.0e9)
            rows.append(row)
        return rows

    for hz in (37.0, 60.0, 100.0):
        report = mod.evaluate_poll_hz_probe(_rows(hz), poll_hz=hz)
        assert report["follows_poll_hz"], report
    report_mismatch = mod.evaluate_poll_hz_probe(_rows(37.0), poll_hz=60.0)
    assert report_mismatch["follows_poll_hz"] is False
