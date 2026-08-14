"""Quintic+dwell scan profile and lateral force-chase softener."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.admittance_common.controller import AdmittanceConfig
from rm75_control.control.joint_admittance_8dof.reference import (
    SinToolYReference,
    quintic_move_s_for_peak_vel,
    quintic_dwell_y_motion,
    smoothstep_scalar,
)


def test_quintic_endpoints_have_zero_velocity():
    A = 0.08
    move_s = 4.0
    dwell_s = 0.2
    # End of outbound move / start of dwell.
    dy, vy = quintic_dwell_y_motion(
        move_s, A, move_s, dwell_s, soft_start=False
    )
    assert abs(dy - A) < 1e-9
    assert abs(vy) < 1e-9
    # Mid dwell still held.
    dy2, vy2 = quintic_dwell_y_motion(
        move_s + 0.5 * dwell_s, A, move_s, dwell_s, soft_start=False
    )
    assert abs(dy2 - A) < 1e-9
    assert abs(vy2) < 1e-9


def test_quintic_accel_near_zero_at_endpoint():
    A = 0.08
    move_s = 4.0
    dwell_s = 0.2
    h = 1e-3
    t = move_s
    _, v0 = quintic_dwell_y_motion(t - h, A, move_s, dwell_s, soft_start=False)
    _, v1 = quintic_dwell_y_motion(t + h, A, move_s, dwell_s, soft_start=False)
    # Across the endpoint into dwell, acceleration estimate stays tiny.
    assert abs((v1 - v0) / (2 * h)) < 0.05


def test_sine_has_large_accel_at_endpoint():
    """Contrast: pure sine peaks acceleration where velocity is zero."""
    A = 0.08
    omega = 2.0 * np.pi / 10.0
    # Endpoint of sine: omega*t = pi/2
    t = 0.5 * np.pi / omega
    h = 1e-3
    from rm75_control.control.joint_admittance_8dof.reference import sin_y_motion

    _, v0 = sin_y_motion(t - h, A, omega, soft_start=False)
    _, v1 = sin_y_motion(t + h, A, omega, soft_start=False)
    a_est = abs((v1 - v0) / (2 * h))
    assert a_est > 0.02  # A*omega^2 ≈ 0.0316


def test_sin_tool_y_reference_default_quintic_dwell():
    ref = SinToolYReference(0.08, max_vel_m_s=0.015, soft_start=False)
    assert ref.profile == "quintic_dwell"
    assert ref.dwell_s > 0.0
    origin = np.zeros(6)
    ref.set_origin(origin)
    # Sample near first end.
    mref = ref.sample(ref.move_s)
    assert abs(mref.vel_ff[1]) < 1e-6


def test_quintic_soft_start_never_exceeds_requested_peak_velocity():
    amplitude = 0.05
    requested = 0.002
    move_s = quintic_move_s_for_peak_vel(amplitude, requested)
    samples = np.linspace(0.0, 6.0, 2001)
    values = [
        quintic_dwell_y_motion(
            t, amplitude, move_s, 0.20, soft_start=True, ramp_s=2.0
        )[1]
        for t in samples
    ]
    assert max(abs(value) for value in values) <= requested * (1.0 + 1.0e-9)
    # Startup is bumpless at the scan centre.
    dy0, vy0 = quintic_dwell_y_motion(
        0.0, amplitude, move_s, 0.20, soft_start=True, ramp_s=2.0
    )
    assert abs(dy0) <= 1.0e-12
    assert vy0 == 0.0


def test_quintic_soft_start_acceleration_is_continuous_at_both_boundaries():
    amplitude = 0.05
    move_s = quintic_move_s_for_peak_vel(amplitude, 0.002)
    ramp_s = 2.0
    h = 1.0e-4

    def velocity(t_s: float) -> float:
        return quintic_dwell_y_motion(
            t_s, amplitude, move_s, 0.20, soft_start=True, ramp_s=ramp_s
        )[1]

    start_acceleration = (velocity(h) - velocity(0.0)) / h
    before = (velocity(ramp_s) - velocity(ramp_s - h)) / h
    after = (velocity(ramp_s + h) - velocity(ramp_s)) / h
    assert abs(start_acceleration) <= 1.0e-6
    assert abs(before - after) <= 1.0e-5


def test_smoothstep_c2_endpoints():
    s0, ds0 = smoothstep_scalar(0.0, 1.0)
    s1, ds1 = smoothstep_scalar(1.0, 1.0)
    assert s0 == 0.0 and abs(ds0) < 1e-12
    assert s1 == 1.0 and abs(ds1) < 1e-12


def test_lateral_chase_scale_softens_at_low_speed():
    cfg = AdmittanceConfig(
        force_lateral_soft_m_s=0.006,
        force_lateral_full_m_s=0.018,
        force_lateral_gain_floor=0.35,
    )
    from rm75_control.control.admittance_common.controller import (
        AdmittanceController,
    )

    ctrl = AdmittanceController(0.005, cfg)
    # No prior scan motion → keep full chase (force-hold / Z tracking).
    assert ctrl._lateral_chase_scale(0.0, dt_s=0.005) == 1.0
    # Arm after scan speed, then soften at turnaround.
    assert ctrl._lateral_chase_scale(0.05, dt_s=0.005) == 1.0
    assert ctrl._lat_soften_hold_s > 0.0
    slow = ctrl._lateral_chase_scale(0.0, dt_s=0.005)
    assert slow == cfg.force_lateral_gain_floor
    mid = ctrl._lateral_chase_scale(0.012, dt_s=0.005)
    assert cfg.force_lateral_gain_floor < mid < 1.0


def test_lissajous_x_is_double_frequency():
    from rm75_control.control.joint_admittance_8dof.reference import (
        lissajous_period_for_peak_vel,
    )

    ay, ax, vmax = 0.15, 0.04, 0.03
    period = lissajous_period_for_peak_vel(ay, ax, vmax)
    ref = SinToolYReference(
        ay,
        max_vel_m_s=vmax,
        soft_start=False,
        profile="sine",
        amplitude_x_m=ax,
    )
    assert ref.period_s == pytest.approx(period)
    origin = np.zeros(6)
    ref.set_origin(origin)
    # x=Ax sin(2ωt) peaks at ωt=π/4; y=Ay sin(ωt) peaks at ωt=π/2.
    m_x = ref.sample(0.125 * ref.period_s)
    assert abs(m_x.pose_d[0] - ax) < 1.0e-9
    m_y = ref.sample(0.25 * ref.period_s)
    assert abs(m_y.pose_d[1] - ay) < 1.0e-9
    assert abs(m_y.pose_d[0]) < 1.0e-9
    speeds = []
    for t in np.linspace(0.0, ref.period_s, 361):
        vel = ref.sample(t).vel_ff[:2]
        speeds.append(float(np.hypot(vel[0], vel[1])))
    assert max(speeds) <= vmax * (1.0 + 1.0e-6)


def test_lissajous_world_y_span_includes_tilted_tool_x():
    ref = SinToolYReference(
        0.15,
        max_vel_m_s=0.03,
        soft_start=False,
        profile="sine",
        amplitude_x_m=0.04,
    )
    identity = np.array([0.4, 0.2, 0.3, 0.0, 0.0, 0.0])
    ref.set_origin(identity)
    y_c0, y_half0 = ref.world_y_span()
    assert abs(y_c0 - 0.2) < 1.0e-6
    assert abs(y_half0 - 0.15) < 1.0e-6
    # Yaw 50°: tool-X leaks into world Y, so the envelope is not (y0, Ay).
    origin = np.array([0.4, 0.2, 0.3, 0.0, 0.0, np.deg2rad(50.0)])
    ref.set_origin(origin)
    y_c, y_half = ref.world_y_span()
    pts = ref.path_world_xyz(n=16)
    assert abs(y_half - 0.15) > 0.005 or abs(y_c - 0.2) > 0.005
    assert float(np.max(pts[:, 1]) - np.min(pts[:, 1])) == pytest.approx(
        2.0 * y_half, abs=1.0e-9
    )
