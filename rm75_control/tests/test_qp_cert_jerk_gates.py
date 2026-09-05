"""Offline S1/S2/S3-1/protocol gates for the jerk-elimination patch."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.qp_cert import (
    dual_cancel_frac,
    inbox_brake,
    measure_qdot_box,
    qp_status_name,
    raised_cosine_alpha,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_command import (
    RailCommandMixer,
    allocate_rail_shares,
)
from rm75_control.control.joint_admittance_8dof.wbc_rt import protocol as P


def test_protocol_v5_sizes() -> None:
    assert P.WBC_VERSION == 7
    assert P.WBC_IN_SIZE == 616
    assert P.WBC_OUT_SIZE == 1440


def test_inbox_brake_stays_inside_accel_box() -> None:
    prev = np.array([0.10, 0.08, -0.06, 0.0, 0.04, 0.0, 0.0, 0.0])
    a_max = np.full(8, 3.0)
    h1 = 0.005
    step = a_max * h1
    lo = prev - step
    hi = prev + step
    out = inbox_brake(prev, lo, hi, a_max, h1)
    assert np.all(out >= lo - 1e-12)
    assert np.all(out <= hi + 1e-12)
    assert out[0] == pytest.approx(0.10 - 0.015)
    assert not np.allclose(out, 0.85 * prev)


def test_measure_box_ignores_float_noise() -> None:
    lo = np.zeros(8)
    hi = np.full(8, 0.015)
    qdot = np.full(8, 0.015 + 5e-7)
    excess, deg, inf, subst = measure_qdot_box(qdot, lo, hi)
    assert excess < 1e-6
    assert not subst
    assert not inf


def test_raised_cosine_alpha_is_continuous() -> None:
    assert raised_cosine_alpha(0.0, 0.03, 0.15, 0.2, 0.12) == pytest.approx(1.0)
    assert raised_cosine_alpha(0.15, 0.03, 0.15, 0.2, 0.12) == pytest.approx(0.0)
    mid = raised_cosine_alpha(0.09, 0.03, 0.15, 0.2, 0.12)
    assert 0.0 < mid < 1.0
    assert raised_cosine_alpha(0.0, 0.03, 0.15, 0.0, 0.12) == pytest.approx(0.0)


def test_mixer_adds_posture_to_u_feasible() -> None:
    shares = allocate_rail_shares(
        u_task_raw=0.04,
        u_post_raw=-0.03,
        u_escape_raw=0.0,
        escape_dir=0,
        u_lo=-0.12,
        u_hi=0.12,
    )
    assert shares["u_feasible"] == pytest.approx(
        shares["u_base"] + shares["u_post_feasible"]
    )
    mix = RailCommandMixer(kp=1.2, ki=0.8, u_mid_max=0.03, kaw=8.0)
    mix.d_star.init_from_live(0.0)
    tel = mix.step(
        d_live=0.05,
        d_star_target=0.0,
        u_task_raw=0.04,
        u_escape_raw=0.0,
        escape_explicit=False,
        dt=0.005,
        u_max=0.12,
        secondary_alpha=1.0,
    )
    assert tel.u_feasible == pytest.approx(tel.u_base + tel.u_post_feasible)
    tel0 = mix.step(
        d_live=0.05,
        d_star_target=0.0,
        u_task_raw=0.04,
        u_escape_raw=0.0,
        escape_explicit=False,
        dt=0.005,
        u_max=0.12,
        secondary_alpha=0.0,
    )
    assert tel0.u_post_raw == pytest.approx(0.0)
    assert tel0.u_task_feasible == pytest.approx(0.04)


def test_measure_box_rail_excess_is_always_substantial() -> None:
    lo = np.full(8, -0.06)
    hi = np.full(8, 0.06)
    qdot = np.zeros(8)
    qdot[0] = 0.066  # 6 mm over; 5% of width, old 10% rule would keep it
    _e, _d, _i, subst = measure_qdot_box(qdot, lo, hi)
    assert subst
    qdot[0] = 0.0
    qdot[1] = 0.066
    _e, _d, _i, subst_arm = measure_qdot_box(qdot, lo, hi)
    assert not subst_arm


def test_measure_box_rejects_degenerate_excess() -> None:
    lo = np.full(8, 0.0804)
    hi = np.full(8, 0.0804)
    qdot = np.full(8, 0.0804)
    qdot[0] = 0.1012
    excess, deg, inf, subst = measure_qdot_box(qdot, lo, hi)
    assert deg
    assert not inf
    assert excess == pytest.approx(0.0208)
    assert subst


def test_soft_saturate_never_hits_hard_rail() -> None:
    mix = RailCommandMixer(kp=1.2, ki=0.8, u_mid_max=0.03, kaw=8.0)
    mix.d_star.init_from_live(0.0)
    mix.xi = 0.2
    tel = mix.step(
        d_live=0.15,
        d_star_target=0.0,
        u_task_raw=0.0,
        u_escape_raw=0.0,
        escape_explicit=False,
        dt=0.005,
        u_max=0.12,
        secondary_alpha=1.0,
    )
    assert abs(tel.u_mid_cmd) < 0.03
    assert tel.u_mid_cmd != 0.03
    assert abs(tel.u_mid_cmd) > 0.9 * 0.03


def test_csv_header_has_jerk_s0_columns() -> None:
    from rm75_control.control.joint_admittance_8dof.loop import _TickLogger

    header = _TickLogger._HEADER
    for name in (
        "qpik_box_degenerate",
        "qpik_box_infeasible",
        "qpik_box_excess_max",
        "qpik_manip_active",
        "qpik_qdot_qp_vs_sent_max",
        "qpik_dual_cancel",
        "secondary_alpha",
    ):
        assert name in header
    start = header.index("qpik_box_degenerate")
    assert header[start:start + 7] == [
        "qpik_box_degenerate",
        "qpik_box_infeasible",
        "qpik_box_excess_max",
        "qpik_manip_active",
        "qpik_qdot_qp_vs_sent_max",
        "qpik_dual_cancel",
        "secondary_alpha",
    ]
    assert len(header) == len(set(header))


def test_qp_status_name_and_dual_cancel() -> None:
    assert qp_status_name(1) == "solved"
    assert qp_status_name(2) == "max_iter"
    assert qp_status_name("PROXQP_MAX_ITER_REACHED") == "max_iter"
    assert dual_cancel_frac(0.03, -0.03) == pytest.approx(1.0)
    assert dual_cancel_frac(0.03, 0.03) == pytest.approx(0.0)
    assert math.isfinite(dual_cancel_frac(0.0, 0.03))
