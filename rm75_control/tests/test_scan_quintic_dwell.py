"""Quintic+dwell scan profile and lateral force-chase softener."""

from __future__ import annotations

import numpy as np

from rm75_control.control.admittance_common.controller import AdmittanceConfig
from rm75_control.control.joint_admittance_8dof.reference import (
    SinToolYReference,
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
