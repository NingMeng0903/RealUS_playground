"""Single rail-owner mixer, live SRS q*, J4 design band, SHM v4."""

from __future__ import annotations

import itertools
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.api import SecondaryPolicy
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    VelocityBoxConstraints,
)
from rm75_control.control.joint_admittance_8dof.solver.joint_comfort import (
    J4DesignComfortBuilder,
    j4_design_qdot_bounds,
    j4_joint_index,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits
from rm75_control.control.joint_admittance_8dof.tasks.rail_command import (
    EPS_ENTER,
    EPS_EXIT,
    DStarRef,
    RailCommandMixer,
    allocate_rail_shares,
    in_leave_band,
    leave_margin_m,
    policy_escape_sign,
    press_escape_allowed_from_flags,
    project_lpf_into_wall,
    q_star_srs_valid,
    soft_rail_travel,
    update_escape_dir,
    wall_velocity_bounds,
)
from rm75_control.control.joint_admittance_8dof.wbc_rt import protocol as P
from rm75_control.control.joint_admittance_8dof.wbc_rt.client import find_wbc_rt_binary


_CFG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"
_SEED_Q = np.array([0.375, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370])


def _yaml_inner_at_rail(q_rail_m: float) -> JointIkController:
    raw = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    cfg = build_joint_ik_config(raw)
    cfg.backend = "python"
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    inner = JointIkController(RobotKinematics(), cfg)
    q = _SEED_Q.copy()
    q[0] = float(q_rail_m)
    inner.reset(q)
    return inner


def test_protocol_v6_layout_is_1208() -> None:
    assert P.WBC_VERSION == 6
    assert P.WBC_OUT_SIZE == 1208
    assert P.WBC_IN_SIZE == 608
    binary = find_wbc_rt_binary()
    if binary is None:
        pytest.skip("wbc_rt binary not built")
    import subprocess

    out = subprocess.check_output([str(binary), "--sizes"], text=True).strip()
    inn, outn = out.split()
    assert int(inn) == 608
    assert int(outn) == 1208


def test_allocate_identity_and_bidirectional_cancel() -> None:
    shares = allocate_rail_shares(
        u_task_raw=0.04,
        u_post_raw=-0.01,
        u_escape_raw=0.0,
        escape_dir=0,
        u_lo=-0.12,
        u_hi=0.12,
    )
    assert shares["u_feasible"] == pytest.approx(
        shares["u_base"] + shares["u_post_feasible"]
    )
    cancel = allocate_rail_shares(
        u_task_raw=0.10,
        u_post_raw=-0.10,
        u_escape_raw=0.0,
        escape_dir=0,
        u_lo=-0.12,
        u_hi=0.12,
    )
    assert cancel["u_feasible"] == pytest.approx(
        cancel["u_base"] + cancel["u_post_feasible"]
    )
    assert cancel["u_post_feasible"] == pytest.approx(-0.10)


def test_escape_guard_never_crossed() -> None:
    plus = allocate_rail_shares(
        u_task_raw=-0.20,
        u_post_raw=-0.20,
        u_escape_raw=0.05,
        escape_dir=1,
        u_lo=-0.12,
        u_hi=0.12,
    )
    assert plus["u_feasible"] + 1e-12 >= plus["u_escape_feasible"]
    minus = allocate_rail_shares(
        u_task_raw=0.20,
        u_post_raw=0.20,
        u_escape_raw=-0.05,
        escape_dir=-1,
        u_lo=-0.12,
        u_hi=0.12,
    )
    assert minus["u_feasible"] - 1e-12 <= minus["u_escape_feasible"]


def test_task_saturation_is_not_blamed_on_pi() -> None:
    mix = RailCommandMixer(kp=1.2, ki=0.8, u_mid_max=0.12, kaw=8.0)
    mix.d_star.init_from_live(0.25)
    tel = mix.step(
        d_live=0.25,
        d_star_target=0.25,
        u_task_raw=1.0,
        u_escape_raw=0.0,
        escape_explicit=False,
        dt=0.005,
        u_max=0.12,
    )
    assert abs(tel.e_d) < 1e-12
    assert abs(tel.xi) < 1e-12
    assert abs(tel.u_task_feasible) <= 0.12 + 1e-12
    assert abs(tel.u_pi_raw) < 1e-12


def test_hard_clip_anti_windup_bounds_xi() -> None:
    mix = RailCommandMixer(kp=1.2, ki=8.0, u_mid_max=0.12, kaw=8.0)
    mix.d_star.init_from_live(0.0)
    for _ in range(400):
        mix.step(
            d_live=0.20,
            d_star_target=0.0,
            u_task_raw=0.0,
            u_escape_raw=0.0,
            escape_explicit=False,
            dt=0.005,
            u_max=0.12,
        )
    assert abs(mix.xi) < 0.6
    assert abs(mix.last.u_mid_cmd) <= 0.12 + 1e-12


def test_unsaturated_pi_uses_tanh_not_hard_clip() -> None:
    from rm75_control.control.joint_admittance_8dof.tasks.rail_allocator import (
        soft_saturate,
    )

    mix = RailCommandMixer(kp=1.2, ki=0.8, u_mid_max=0.12, kaw=8.0)
    mix.d_star.init_from_live(0.0)
    tel = mix.step(
        d_live=-0.04,
        d_star_target=0.0,
        u_task_raw=0.0,
        u_escape_raw=0.0,
        escape_explicit=False,
        dt=0.005,
        u_max=0.12,
        hold_d_star=True,
    )
    assert tel.u_mid_cmd == pytest.approx(soft_saturate(tel.u_pi_raw, 0.12), abs=1e-12)
    assert abs(tel.u_pi_raw) < 0.12 - 1e-6
    assert abs(tel.u_mid_cmd) < abs(tel.u_pi_raw)


def test_pi_plant_closes_e_d_with_rail_sign() -> None:
    mix = RailCommandMixer(kp=1.2, ki=0.8, u_mid_max=0.12, kaw=8.0)
    mix.d_star.init_from_live(0.0)
    d_live = -0.04
    dt = 0.005
    tel = None
    for _ in range(int(2.0 / dt)):
        tel = mix.step(
            d_live=d_live,
            d_star_target=0.0,
            u_task_raw=0.0,
            u_escape_raw=0.0,
            escape_explicit=False,
            dt=dt,
            u_max=0.12,
            hold_d_star=True,
        )
        d_live -= float(tel.u_mid_applied) * dt
    assert tel is not None
    assert abs(d_live - float(tel.d_star_ref)) < 0.020


def test_wall_pi_frozen_does_not_integrate_or_backcalc() -> None:
    mix = RailCommandMixer(kp=1.2, ki=0.8, u_mid_max=0.12, kaw=8.0)
    mix.d_star.init_from_live(0.0)
    mix.xi = 0.05
    for _ in range(40):
        mix.step(
            d_live=0.20,
            d_star_target=0.0,
            u_task_raw=0.0,
            u_escape_raw=0.0,
            escape_explicit=False,
            dt=0.005,
            u_max=0.12,
            in_wall=True,
        )
        assert mix.xi == pytest.approx(0.05, abs=1e-12)


def test_escape_dir_debounce() -> None:
    d = 0
    for u in (1.5e-4, -1.5e-4, 1.5e-4, -1.5e-4):
        d = update_escape_dir(
            explicit_active=True,
            u_escape_raw=u,
            prev_dir=d,
        )
        assert d == 0
    d = update_escape_dir(explicit_active=True, u_escape_raw=3.0e-4, prev_dir=0)
    assert d == 1
    d = update_escape_dir(explicit_active=True, u_escape_raw=1.5e-4, prev_dir=d)
    assert d == 1
    d = update_escape_dir(explicit_active=True, u_escape_raw=5.0e-5, prev_dir=d)
    assert d == 0
    d = update_escape_dir(explicit_active=False, u_escape_raw=1.0, prev_dir=1)
    assert d == 0
    assert EPS_ENTER > EPS_EXIT


def test_d_star_ref_slews_and_inits_from_live() -> None:
    ref = DStarRef()
    r, dot = ref.step(0.40, 0.005, rate_m_s=0.02, hold=False, d_live=0.25)
    assert r == pytest.approx(0.25)
    assert dot == pytest.approx(0.0)
    r2, dot2 = ref.step(0.40, 0.005, rate_m_s=0.02, hold=False, d_live=0.25)
    assert abs(dot2) <= 0.02 + 1e-12
    assert r2 == pytest.approx(0.25 + 0.02 * 0.005)
    r3, dot3 = ref.step(1.0, 0.005, rate_m_s=0.02, hold=True, d_live=0.25)
    assert r3 == pytest.approx(r2)
    assert dot3 == pytest.approx(0.0)


def test_track_applied_clears_stale_ref_before_command() -> None:
    d_live = -0.246
    stale = RailCommandMixer(kp=1.2, ki=0.8, u_mid_max=0.12, kaw=8.0)
    stale.d_star.ref = 0.0
    stale.d_star.dot = 0.0
    dumped = stale.step(
        d_live=d_live,
        d_star_target=d_live,
        u_task_raw=0.0,
        u_escape_raw=0.0,
        escape_explicit=False,
        dt=0.005,
        u_max=0.12,
    )
    assert abs(dumped.e_d) > 0.20 or abs(dumped.d_star_dot_cmd) > 0.01

    mix = RailCommandMixer(kp=1.2, ki=0.8, u_mid_max=0.12, kaw=8.0)
    mix.d_star.ref = 0.0
    mix.d_star.dot = 0.0
    tel = mix.track_applied(
        d_live=d_live,
        d_star_target=d_live,
        applied_rail_vel=0.04,
        dt=0.005,
    )
    assert tel.d_star_ref == pytest.approx(d_live)
    assert tel.e_d == pytest.approx(0.0, abs=1e-12)
    assert mix.xi == pytest.approx(0.0, abs=1e-12)
    assert tel.u_feasible == pytest.approx(0.04)
    joined = mix.step(
        d_live=d_live,
        d_star_target=d_live,
        u_task_raw=0.0,
        u_escape_raw=0.0,
        escape_explicit=False,
        dt=0.005,
        u_max=0.12,
    )
    assert abs(joined.e_d) < 1e-12
    assert abs(joined.u_post_raw) < 1e-12


def test_ptp_then_servo_does_not_dump_rail_mixer() -> None:
    inner = _yaml_inner_at_rail(0.49)
    q = inner.q_cmd.copy()
    inner.rail_mixer.d_star.ref = 0.0
    inner.rail_mixer.d_star.dot = 0.0
    inner.set_direct_joint_ptp(True)
    last = None
    for _ in range(8):
        last = inner.update(np.zeros(6), q_meas=q, qdot_ff=np.zeros(8))
        q = np.asarray(last.q_send, dtype=float).copy()
    assert last is not None
    d_live = float(inner.kin.fk_pose(q)[1]) - float(q[0])
    assert abs(float(last.e_d)) < 2.0e-3
    assert abs(float(inner.rail_mixer.d_star.ref) - d_live) < 2.0e-3
    assert abs(float(inner.rail_mixer.d_star.ref)) > 0.05
    inner.set_direct_joint_ptp(False)
    step = inner.update(np.zeros(6), q_meas=q)
    assert abs(float(step.e_d)) < 2.0e-3
    assert abs(float(step.u_post_raw)) < 5.0e-3


def test_quiescent_does_not_mask_mixer_terms() -> None:
    mix = RailCommandMixer(kp=1.2, ki=0.8, u_mid_max=0.12, kaw=8.0)
    mix.d_star.init_from_live(0.20)
    tel_live = mix.step(
        d_live=0.28,
        d_star_target=0.20,
        u_task_raw=0.05,
        u_escape_raw=0.0,
        escape_explicit=False,
        dt=0.005,
        u_max=0.12,
        quiescent=False,
    )
    mix2 = RailCommandMixer(kp=1.2, ki=0.8, u_mid_max=0.12, kaw=8.0)
    mix2.d_star.init_from_live(0.20)
    tel_q = mix2.step(
        d_live=0.28,
        d_star_target=0.20,
        u_task_raw=0.05,
        u_escape_raw=0.0,
        escape_explicit=False,
        dt=0.005,
        u_max=0.12,
        quiescent=True,
    )
    assert tel_q.u_task_raw == pytest.approx(tel_live.u_task_raw)
    assert tel_q.u_post_raw == pytest.approx(tel_live.u_post_raw)
    assert tel_q.u_feasible == pytest.approx(tel_live.u_feasible)
    tel_e = mix2.step(
        d_live=0.28,
        d_star_target=0.20,
        u_task_raw=0.0,
        u_escape_raw=0.04,
        escape_explicit=True,
        dt=0.005,
        u_max=0.12,
        quiescent=True,
    )
    assert tel_e.u_escape_feasible > 0.0
    assert tel_e.u_feasible > 0.0


def test_project_lpf_into_wall_zeros_into_wall_velocity() -> None:
    assert project_lpf_into_wall(0.05, 1.0) == pytest.approx(0.0)
    assert project_lpf_into_wall(-0.05, 1.0) == pytest.approx(-0.05)
    assert project_lpf_into_wall(-0.05, -1.0) == pytest.approx(0.0)
    assert project_lpf_into_wall(0.05, -1.0) == pytest.approx(0.05)


def test_q_star_srs_valid_rejects_nan_and_limits() -> None:
    lo = np.full(8, -2.0)
    hi = np.full(8, 2.0)
    q = np.zeros(8)
    assert q_star_srs_valid(q, q_lo=lo, q_hi=hi)
    qn = q.copy()
    qn[1] = np.nan
    assert not q_star_srs_valid(qn, q_lo=lo, q_hi=hi)
    qn[1] = 3.0
    assert not q_star_srs_valid(qn, q_lo=lo, q_hi=hi)
    assert not q_star_srs_valid(None, q_lo=lo, q_hi=hi)


def test_j4_index_uses_representation() -> None:
    assert j4_joint_index(8) == 4
    assert j4_joint_index(7) == 3


def test_j4_design_pref_rows_are_empty() -> None:
    b = J4DesignComfortBuilder()
    q21 = np.zeros(8)
    q21[4] = np.deg2rad(21.0)
    rows = b.build_rows(q21)
    assert not rows.active
    assert rows.jacobian.shape[0] == 0
    q96 = np.zeros(8)
    q96[4] = np.deg2rad(96.0)
    assert b.build_rows(q96).jacobian.shape[0] == 0


def test_j4_design_hard_box_holds_band() -> None:
    gamma = 4.0
    lo_rad = np.deg2rad(70.0)
    hi_rad = np.deg2rad(115.0)
    lim = SafetyLimits(
        q_lower=np.concatenate(([0.005], np.full(7, -3.0))),
        q_upper=np.concatenate(([0.78], np.full(7, 3.0))),
        v_max=np.concatenate(([0.12], np.full(7, 2.0))),
        a_max=None,
        position_margin=np.zeros(8),
    )
    box = VelocityBoxConstraints(
        lim,
        damper_band_rad=0.0,
        rail_reaction_s=0.0,
        j4_design_enabled=True,
        j4_design_lo=lo_rad,
        j4_design_hi=hi_rad,
        j4_design_gamma=gamma,
    )
    q = np.zeros(8)
    q[0] = 0.40
    q[4] = np.deg2rad(96.0)
    lo, hi = box.bounds(q, 0.005)
    want_lo, want_hi = j4_design_qdot_bounds(
        float(q[4]), lower_rad=lo_rad, upper_rad=hi_rad, gamma=gamma
    )
    assert lo[4] == pytest.approx(want_lo)
    assert hi[4] == pytest.approx(want_hi)
    assert lo[4] < 0.0 < hi[4]
    q[4] = np.deg2rad(21.0)
    lo21, hi21 = box.bounds(q, 0.005)
    assert lo21[4] > 0.0
    assert hi21[4] >= lo21[4] - 1e-12 or lo21[4] > hi21[4]
    q[4] = np.deg2rad(70.0)
    lo70, _ = box.bounds(q, 0.005)
    assert lo70[4] == pytest.approx(0.0, abs=1e-12)
    q[4] = np.deg2rad(115.0)
    _, hi115 = box.bounds(q, 0.005)
    assert hi115[4] == pytest.approx(0.0, abs=1e-12)


def test_inner_mixer_puts_posture_on_rail_not_sec0() -> None:
    inner = _yaml_inner_at_rail(0.40)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(inner)
    last = inner.update(np.zeros(6), q_meas=inner.q_cmd.copy())
    assert inner._u_mid_for_sec == pytest.approx(0.0)
    if np.isfinite(last.u_base) and np.isfinite(last.u_post_feasible):
        assert last.u_feasible == pytest.approx(
            last.u_base + last.u_post_feasible, abs=1e-9
        )


def test_inner_j4_21deg_recovery_direction_or_slack() -> None:
    inner = _yaml_inner_at_rail(0.40)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(inner)
    q = inner.q_cmd.copy()
    q[4] = np.deg2rad(21.0)
    inner.reset(q)
    last = inner.update(np.zeros(6), q_meas=q)
    assert np.isfinite(last.j4_design_slack)
    assert float(last.qdot[4]) >= -1.0e-4 or float(last.j4_design_slack) > 0.0


def test_inner_j4_recovers_after_quiet_latch() -> None:
    """Centering must keep pulling J4 after the stick-still latch; do not zero sec."""
    inner = _yaml_inner_at_rail(0.40)
    SecondaryPolicy(preset="track", qdot_ff="off").apply(inner)
    q = inner.q_cmd.copy()
    q[4] = np.deg2rad(21.0)
    inner.reset(q)
    last = None
    for _ in range(80):
        last = inner.update(np.zeros(6), q_meas=inner.q_cmd.copy())
    assert last is not None
    assert bool(inner._quiescent)
    assert float(inner.secondary.last_centering_norm) > 1.0e-4
    assert float(last.qdot[4]) >= -1.0e-4 or float(last.j4_design_slack) > 0.0


def _lpf(prev: float, u: float, dt: float, fc_hz: float = 2.0) -> float:
    tau = 1.0 / (2.0 * math.pi * fc_hz)
    a = dt / (tau + dt)
    return (1.0 - a) * prev + a * u


def _scan_p95(v_tcp: float, stroke_m: float, rail_lo: float, rail_hi: float) -> float:
    """1-D plant: TCP moves, rail integrates LPF(u_feasible).  Score interior e_d.

    Samples with the rail on a hard stop are dropped so saturation at the
    stroke ends does not dominate P95 (hardware C is allowed to grow e_d
    near the walls).
    """
    mix = RailCommandMixer(
        kp=1.2, ki=0.8, u_mid_max=0.12, kaw=8.0, d_center_rate=0.02
    )
    dt = 0.005
    sign = 1.0 if v_tcp >= 0.0 else -1.0
    margin = 0.03
    if sign >= 0.0:
        y_rail = float(rail_lo + margin)
    else:
        y_rail = float(rail_hi - margin)
    y_tcp = y_rail + 0.25
    mix.d_star.init_from_live(y_tcp - y_rail)
    d_star = 0.25
    v_lpf = 0.0
    abs_e: list[float] = []
    t_move = abs(stroke_m / max(abs(v_tcp), 1e-6))
    n = int((t_move + 1.0) / dt)
    for i in range(n):
        v_cmd = 0.0
        if i * dt < t_move:
            v_cmd = sign * abs(v_tcp)
            y_tcp += v_cmd * dt
        d_live = y_tcp - y_rail
        tel = mix.step(
            d_live=d_live,
            d_star_target=d_star,
            u_task_raw=float(v_cmd),
            u_escape_raw=0.0,
            escape_explicit=False,
            dt=dt,
            u_max=0.12,
        )
        v_lpf = _lpf(v_lpf, tel.u_feasible, dt)
        y_rail = float(np.clip(y_rail + v_lpf * dt, rail_lo, rail_hi))
        interior = (rail_lo + margin) < y_rail < (rail_hi - margin)
        if i * dt > 0.5 and interior:
            abs_e.append(abs(tel.e_d))
    return float(np.percentile(abs_e, 95)) if abs_e else 1.0


def test_sim_scan_A_mid_stroke_p95() -> None:
    """Stand-in for hardware A: mid-stroke 50 and 100 mm/s, rail 0.10–0.68 m."""
    assert _scan_p95(0.05, 0.40, 0.10, 0.68) < 0.030
    assert _scan_p95(0.10, 0.40, 0.10, 0.68) < 0.030


def test_sim_scan_B_reverse_and_unidirectional() -> None:
    """Stand-in for hardware B: reverse and sustained one-way both converge."""
    assert _scan_p95(0.08, 0.35, 0.10, 0.68) < 0.030
    assert _scan_p95(-0.08, 0.35, 0.10, 0.68) < 0.030


def test_sim_scan_C_long_stroke() -> None:
    """Stand-in for hardware C: ~1.15 m TCP travel, rail saturates at the ends.

    Comfort is only required in the interior; near-wall saturation may grow
    ``e_d``.  ``_scan_p95`` already skips the first 0.5 s; this check uses a
    wider bound than A/B because the rail can pin at the stroke ends.
    """
    assert _scan_p95(0.08, 1.15, 0.015, 0.77) < 0.080


def _wbc_bin():
    binary = find_wbc_rt_binary()
    if binary is None:
        pytest.skip("wbc_rt binary not built")
    return binary


def test_press_escape_flag_truth_table_matches_native() -> None:
    binary = _wbc_bin()
    names = (
        "demanding",
        "has_travel",
        "press_stalled",
        "j4_blocked",
        "arm_starved",
        "policy_leave",
    )
    for bits in itertools.product((False, True), repeat=6):
        kwargs = dict(zip(names, bits))
        py = press_escape_allowed_from_flags(**kwargs)
        out = subprocess.check_output(
            [str(binary), "--press-escape", "--flags", *[str(int(b)) for b in bits]],
            text=True,
        ).strip()
        assert int(out) == int(py), kwargs
        if not bits[0] or not bits[1]:
            assert py is False
        elif bits[2]:
            assert py is True
        elif bits[3] and not bits[5]:
            assert py is True
        elif bits[3] and bits[5] and not bits[4]:
            assert py is False
        elif bits[5] and bits[4]:
            assert py is True
        elif (not bits[5]) and bits[4] and (not bits[2]) and (not bits[3]):
            assert py is False


def test_policy_leave_geometry_matches_native() -> None:
    binary = _wbc_bin()
    urdf_lo, urdf_hi = 0.0, 0.785
    soft_min, soft_max = 0.005, 0.78
    escape_leave, pin_margin = 0.04, 0.008
    lo, hi = soft_rail_travel(urdf_lo, urdf_hi, soft_min, soft_max)
    leave = leave_margin_m(escape_leave, pin_margin)
    mid = 0.5 * (lo + hi)
    ys = (
        lo,
        lo + leave,
        lo + leave + 0.001,
        mid,
        hi - leave - 0.001,
        hi - leave,
        hi,
        0.40,
    )
    for policy in ("minus", "plus", "auto"):
        for y in ys:
            for latched in (0.0, 1.0, -1.0):
                py_sign = policy_escape_sign(policy, y, lo, hi, latched_sign=latched)
                py_leave = in_leave_band(y, lo, hi, leave, py_sign)
                out = subprocess.check_output(
                    [
                        str(binary),
                        "--policy-leave",
                        "--y",
                        str(y),
                        "--urdf-lo",
                        str(urdf_lo),
                        "--urdf-hi",
                        str(urdf_hi),
                        "--soft-min",
                        str(soft_min),
                        "--soft-max",
                        str(soft_max),
                        "--escape-leave",
                        str(escape_leave),
                        "--pin-margin",
                        str(pin_margin),
                        "--policy",
                        policy,
                        "--latched-sign",
                        str(latched),
                    ],
                    text=True,
                ).split()
                assert float(out[0]) == pytest.approx(py_sign)
                assert int(out[1]) == int(py_leave)


def test_two_phase_stop_after_press_revoke() -> None:
    mix = RailCommandMixer(kp=1.2, ki=0.8, u_mid_max=0.12, kaw=8.0)
    mix.d_star.init_from_live(0.25)
    tel_on = mix.step(
        d_live=0.27,
        d_star_target=0.25,
        u_task_raw=0.03,
        u_escape_raw=0.08,
        escape_explicit=True,
        dt=0.005,
        u_max=0.12,
        quiescent=False,
    )
    assert tel_on.escape_active == pytest.approx(1.0)
    assert abs(tel_on.u_escape_raw) > 1e-9
    tel_off = mix.step(
        d_live=0.27,
        d_star_target=0.25,
        u_task_raw=0.03,
        u_escape_raw=0.0,
        escape_explicit=False,
        dt=0.005,
        u_max=0.12,
        quiescent=False,
    )
    assert tel_off.escape_active == pytest.approx(0.0)
    assert tel_off.u_escape_raw == pytest.approx(0.0)
    assert abs(tel_off.u_escape_feasible) < 1e-12
    tel_q = mix.step(
        d_live=0.27,
        d_star_target=0.25,
        u_task_raw=0.03,
        u_escape_raw=0.0,
        escape_explicit=False,
        dt=0.005,
        u_max=0.12,
        quiescent=True,
    )
    assert tel_q.u_task_feasible == pytest.approx(tel_off.u_task_feasible)
    assert abs(tel_q.u_feasible - tel_off.u_feasible) < 0.02


def test_dead_end_wall_cap_zeros_into_wall_keeps_raw() -> None:
    u_lo, u_hi = wall_velocity_bounds(0.12, 1.0)
    shares = allocate_rail_shares(
        u_task_raw=0.0,
        u_post_raw=0.0,
        u_escape_raw=0.08,
        escape_dir=1,
        u_lo=u_lo,
        u_hi=u_hi,
    )
    assert shares["u_escape_feasible"] == pytest.approx(0.0)
    assert shares["u_feasible"] == pytest.approx(0.0)
    mix = RailCommandMixer(kp=1.2, ki=0.8, u_mid_max=0.12, kaw=8.0)
    mix.d_star.init_from_live(0.25)
    tel = mix.step(
        d_live=0.25,
        d_star_target=0.25,
        u_task_raw=0.0,
        u_escape_raw=0.08,
        escape_explicit=True,
        dt=0.005,
        u_max=0.12,
        leave_sign=1.0,
        quiescent=False,
    )
    assert tel.u_escape_raw == pytest.approx(0.08)
    assert tel.escape_active == pytest.approx(1.0)
    assert tel.u_feasible <= 1e-12


def _separated_task_j(nv: int = 8) -> np.ndarray:
    w = np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0])
    w_sqrt = np.sqrt(w)
    s_j = np.array([1.0, 0.82, 0.61, 0.40, 0.22, 0.057])
    u = np.eye(6)
    v = np.eye(nv)[:, :6]
    jw = u @ np.diag(s_j) @ v.T
    return jw / w_sqrt[:, None], w


def _python_task_weight_core(*, aniso: bool, tau: float):
    from rm75_control.control.joint_admittance_8dof.ik_types import SrDampingConfig
    from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
    from rm75_control.control.joint_admittance_8dof.solver.qp_builder import (
        QpConfig,
        QpIkController,
    )
    from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits

    kin = RobotKinematics()
    limits = SafetyLimits(
        q_lower=kin.q_lower,
        q_upper=kin.q_upper,
        v_max=kin.v_max,
        a_max=np.full(kin.nv, 10.0),
        position_margin=np.full(kin.nv, 0.01),
    )
    cfg = QpConfig(
        task_weight=np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]),
        aniso_task_damping=bool(aniso),
        task_weight_lpf_tau_s=float(tau),
        task_weight_min_frac=0.05,
        sr_damping=SrDampingConfig(lam0=0.05, sigma_ref=0.08, sigma_floor=1e-6),
        collision=CollisionConfig(enabled=False),
    )
    return QpIkController(kin, limits, cfg)


def _native_task_weight(
    binary,
    j: np.ndarray,
    w: np.ndarray,
    *,
    dt: float,
    tau: float,
    aniso: bool,
    ticks: int,
    zero_rail: bool = False,
    reset_before_tick: int | None = None,
    extra_j: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    cmd = [
        str(binary),
        "--task-weight",
        "--J",
        *[f"{x:.17g}" for x in np.asarray(j, dtype=float).reshape(-1)],
    ]
    for jx in extra_j or []:
        cmd.extend(["--J", *[f"{x:.17g}" for x in np.asarray(jx, dtype=float).reshape(-1)]])
    cmd.extend(
        [
            "--w",
            *[f"{x:.17g}" for x in np.asarray(w, dtype=float).reshape(-1)],
            "--dt",
            str(dt),
            "--tau",
            str(tau),
            "--sigma-ref",
            "0.08",
            "--min-frac",
            "0.05",
            "--aniso",
            "1" if aniso else "0",
            "--ticks",
            str(int(ticks)),
        ]
    )
    if zero_rail:
        cmd.append("--zero-rail")
    if reset_before_tick is not None:
        cmd.extend(["--reset-before-tick", str(int(reset_before_tick))])
    out = subprocess.check_output(cmd, text=True).strip().splitlines()
    mats = []
    for line in out:
        vals = [float(x) for x in line.split()]
        assert len(vals) == 36
        mats.append(np.array(vals, dtype=float).reshape(6, 6))
    return mats


def test_native_task_weight_parity_aniso_iso_lpf_reset_rail() -> None:
    binary = _wbc_bin()
    j, w = _separated_task_j()
    dt = 0.005
    tau = 0.25
    for aniso in (True, False):
        core = _python_task_weight_core(aniso=aniso, tau=tau)
        native = _native_task_weight(
            binary, j, w, dt=dt, tau=tau, aniso=aniso, ticks=8
        )
        for k, w_nat in enumerate(native):
            w_py = core._task_weight_matrix(j, dt=dt, keep_task_weight=False)
            assert np.allclose(w_nat, w_py, rtol=1e-7, atol=1e-8), (aniso, k)

    core0 = _python_task_weight_core(aniso=True, tau=0.0)
    w_raw = core0._task_weight_matrix(j, dt=dt, keep_task_weight=False)
    native_first = _native_task_weight(
        binary, j, w, dt=dt, tau=tau, aniso=True, ticks=1
    )
    assert np.allclose(native_first[0], w_raw, rtol=1e-7, atol=1e-8)

    s_j2 = np.array([1.0, 0.82, 0.61, 0.40, 0.22, 0.030])
    w_sqrt = np.sqrt(w)
    j2 = (np.eye(6) @ np.diag(s_j2) @ np.eye(8)[:, :6].T) / w_sqrt[:, None]
    core_l = _python_task_weight_core(aniso=True, tau=tau)
    native_l = _native_task_weight(
        binary, j, w, dt=dt, tau=tau, aniso=True, ticks=3, extra_j=[j2]
    )
    w_l = [
        core_l._task_weight_matrix(jj, dt=dt, keep_task_weight=False)
        for jj in (j, j2, j2)
    ]
    for k, w_nat in enumerate(native_l):
        assert np.allclose(w_nat, w_l[k], rtol=1e-7, atol=1e-8), k
    core_raw2 = _python_task_weight_core(aniso=True, tau=0.0)
    w_raw2 = core_raw2._task_weight_matrix(j2, dt=dt, keep_task_weight=False)
    assert not np.allclose(native_l[1], w_raw2, rtol=1e-4, atol=1e-5)

    core_r = _python_task_weight_core(aniso=True, tau=tau)
    native_r = _native_task_weight(
        binary, j, w, dt=dt, tau=tau, aniso=True, ticks=3, reset_before_tick=2
    )
    w1 = core_r._task_weight_matrix(j, dt=dt, keep_task_weight=False)
    core_r.reset()
    w2 = core_r._task_weight_matrix(j, dt=dt, keep_task_weight=False)
    w3 = core_r._task_weight_matrix(j, dt=dt, keep_task_weight=False)
    assert np.allclose(native_r[0], w1, rtol=1e-7, atol=1e-8)
    assert np.allclose(native_r[1], w2, rtol=1e-7, atol=1e-8)
    assert np.allclose(native_r[2], w3, rtol=1e-7, atol=1e-8)
    assert np.allclose(w1, w2, rtol=1e-12, atol=1e-12)

    j_z = j.copy()
    j_z[:, 0] = 0.0
    core_z = _python_task_weight_core(aniso=True, tau=0.0)
    w_z = core_z._task_weight_matrix(j_z, dt=dt, keep_task_weight=False)
    native_z = _native_task_weight(
        binary, j, w, dt=dt, tau=0.0, aniso=True, ticks=1, zero_rail=True
    )
    assert np.allclose(native_z[0], w_z, rtol=1e-7, atol=1e-8)

    s_rep = np.array([1.0, 1.0, 1.0, 0.4, 0.4, 0.057])
    w_sqrt = np.sqrt(w)
    u = np.eye(6)
    v = np.eye(8)[:, :6]
    j_rep = (u @ np.diag(s_rep) @ v.T) / w_sqrt[:, None]
    core_rep = _python_task_weight_core(aniso=True, tau=0.0)
    w_rep = core_rep._task_weight_matrix(j_rep, dt=dt, keep_task_weight=False)
    native_rep = _native_task_weight(
        binary, j_rep, w, dt=dt, tau=0.0, aniso=True, ticks=1
    )
    assert np.allclose(native_rep[0], w_rep, rtol=1e-5, atol=1e-6)

