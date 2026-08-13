"""Tool-Z admittance contact behaviour: engagement ramp, unified vz cap, closed-loop."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.admittance_common.proactive_force_ff import ProactiveFfConfig
from rm75_control.control.hybrid_motion.controller import AdmittanceConfig, AdmittanceController


def _base_cfg(**over) -> AdmittanceConfig:
    kw = dict(
        contact_threshold_n=0.8,
        contact_use_fz_only=True,
        admittance_mass_z=1.0,
        admittance_damping_z=25.0,
        deadband_n=0.0,
        deadband_width_n=0.0,
        max_vz_tool_m_s=0.05,
        max_velocity=np.array([0.2, 0.2, 0.05, 0.5, 0.5, 0.5]),
        desired_force_ramp_s=0.0,
        var_damping_enabled=False,
    )
    kw.update(over)
    cfg = AdmittanceConfig(**kw)
    cfg.proactive_ff = ProactiveFfConfig(enabled=False)
    cfg.adaptive_ke.enabled = False
    return cfg


def _tick(ctrl: AdmittanceController, fz: float, f_des_z: float = 3.0) -> float:
    f_ext = np.zeros(6)
    f_ext[2] = fz
    f_des = np.zeros(6)
    f_des[2] = f_des_z
    ctrl.compute_velocity_command(np.zeros(6), np.zeros(6), np.zeros(6), f_ext, f_des)
    return ctrl.v_force_z


def test_hybrid_episode_reset_is_bumpless_and_clears_episode_state():
    cfg = _base_cfg(var_damping_enabled=True)
    cfg.bidirectional_flow.Tmin = 0.001
    cfg.bidirectional_flow.T0 = 0.003
    cfg.bidirectional_flow.Tmax = 0.004
    ctrl = AdmittanceController(0.005, cfg)
    ctrl._bidirectional_flow.tank_energy = 0.0015
    ctrl._in_contact_latched = True
    ctrl.force_task_latched = True
    ctrl.contact_present = True
    ctrl._contact_time_s = 2.0
    ctrl._episode_seen = True
    ctrl._episode_detached_s = 0.2
    ctrl._proactive_ff.v_r = 0.02
    ctrl.v_r_z = 0.02
    ctrl._force_dob.u_dob = 0.8
    ctrl.u_dob_z = 0.8
    ctrl._ke_estimator.ke_est = 1234.0
    seed = np.array([0.01, -0.02, 0.015, 0.03, -0.04, 0.05])

    ctrl.begin_hybrid_episode(seed)

    np.testing.assert_allclose(ctrl.last_v_cmd, seed)
    assert ctrl.v_force_z == pytest.approx(seed[2])
    assert ctrl._bidirectional_flow.tank_energy == pytest.approx(0.0015)
    assert not ctrl._in_contact_latched
    assert not ctrl.force_task_latched
    assert not ctrl.contact_present
    assert ctrl._contact_time_s == 0.0
    assert not ctrl._episode_seen
    assert ctrl._episode_detached_s == 0.0
    assert ctrl._proactive_ff.v_r == 0.0
    assert ctrl.v_r_z == 0.0
    assert ctrl._force_dob.u_dob == 0.0
    assert ctrl.u_dob_z == 0.0
    assert ctrl._ke_estimator.ke_est == pytest.approx(cfg.adaptive_ke.ke_initial)


def test_hybrid_episode_first_force_sample_has_no_filter_or_barrier_spike():
    cfg = _base_cfg(var_damping_enabled=True)
    ctrl = AdmittanceController(0.005, cfg)
    ctrl.begin_hybrid_episode(np.zeros(6))

    ctrl._update_instability_index(6.0)
    assert ctrl.instability_index == pytest.approx(0.0, abs=1.0e-12)
    assert ctrl._p_hi == pytest.approx(0.0, abs=1.0e-12)
    assert ctrl._force_barrier.update_fdot(6.0, 0.005) == pytest.approx(0.0)


def test_admittance_outer_episode_seeds_applied_twist_in_task_frame():
    from scipy.spatial.transform import Rotation as Rsc

    from rm75_control.control.joint_admittance_8dof.loop import AdmittanceOuterLoop
    from rm75_control.control.joint_admittance_8dof.reference import HoldReference

    cfg = _base_cfg(control_frame="tool")
    ctrl = AdmittanceController(0.005, cfg)
    outer = AdmittanceOuterLoop(ctrl, HoldReference())
    pose = np.array([0.2, 0.1, 0.3, 0.4, -0.2, 0.3])
    applied_base = np.array([0.01, -0.02, 0.03, 0.04, -0.05, 0.06])
    rotation = Rsc.from_euler(cfg.euler_order, pose[3:6]).as_matrix()
    expected = np.concatenate(
        (rotation.T @ applied_base[:3], rotation.T @ applied_base[3:])
    )

    outer.begin_hybrid_episode(applied_base, pose)

    np.testing.assert_allclose(ctrl.last_v_cmd, expected, atol=1.0e-12)
    assert ctrl.v_force_z == pytest.approx(expected[2])


def test_single_press_cap_no_free_space_switch():
    """One tool-Z cap in and out of contact — the 5× press-speed jump at
    contact latch that produced the dual press-speed-tier jitter is gone."""
    ctrl = AdmittanceController(0.005, _base_cfg())
    # Externally inject a large negative state, no contact ever latched.
    ctrl.v_force_z = -0.15
    v = _tick(ctrl, fz=0.0)
    # Same unified cap governs both directions.
    cap = ctrl._v_z_cap()
    assert -cap - 1e-9 <= v <= cap + 1e-9
    # The cap must not depend on contact latch alone (only on |F_err| press-side).
    ctrl2 = AdmittanceController(0.005, _base_cfg())
    for _ in range(50):  # latch contact at setpoint → f_err ≈ 0
        _tick(ctrl2, fz=3.0)
    assert ctrl2._v_z_cap() == cap


def test_engagement_force_ramp():
    """Tool-Z setpoint ramps from ~contact threshold to full value on contact."""
    cfg = _base_cfg(desired_force_ramp_s=1.0)
    ctrl = AdmittanceController(0.005, cfg)
    _tick(ctrl, fz=1.0, f_des_z=3.0)
    early = ctrl.f_des_z_eff
    assert early < 1.5, f"setpoint must start low, got {early}"
    for _ in range(300):  # 1.5 s of latched contact
        _tick(ctrl, fz=1.0, f_des_z=3.0)
    assert abs(ctrl.f_des_z_eff - 3.0) < 1e-9


def test_unified_vz_cap_no_state_windup():
    """``v_force_z`` must never exceed the single symmetric tool-Z cap."""
    dt = 0.005
    cfg = _base_cfg(
        max_vz_tool_m_s=0.20,
        max_velocity=np.array([0.2, 0.2, 0.05, 0.5, 0.5, 0.5]),
        admittance_damping_z=5.0,
    )
    ctrl = AdmittanceController(dt, cfg)
    for _ in range(400):
        _tick(ctrl, fz=7.0, f_des_z=3.0)
    cap = ctrl._v_z_cap()
    assert cap == pytest.approx(0.05)
    assert abs(ctrl.v_force_z) <= cap + 1e-9


def test_closed_loop_stiff_surface_no_bounce():
    """Closed-loop regression on a stiff unilateral spring with the SHIPPED
    yaml: approach → impact → settle at the setpoint with no sustained
    contact flipping. Regression guard against the /tmp/scan_v5.csv bounce
    cascade (71 contact losses in 45 s, fz range 0..9.4 N)."""
    import yaml
    from pathlib import Path

    dt = 0.005
    raw = yaml.safe_load(Path("configs/joint_admittance_8dof.yaml").read_text())
    cfg = AdmittanceConfig.from_dict(raw)
    ctrl = AdmittanceController(dt, cfg)

    ke_true = 8000.0  # hard surface
    x = -0.005        # 5 mm above the surface (at x=0)
    flips = 0
    was_contact = False
    fz_hist: list[float] = []
    for _ in range(2400):  # 12 s
        fz = ke_true * max(x, 0.0)
        v = _tick(ctrl, fz=fz, f_des_z=3.0)
        x += v * dt
        # Physical contact state, not the controller's enter-only latch.
        in_c = fz >= cfg.contact_threshold_n
        if in_c != was_contact:
            flips += 1
        was_contact = in_c
        fz_hist.append(fz)

    tail = np.asarray(fz_hist[-400:])  # last 2 s
    # The 27c1689 restore lets the stiff-first K̂_e absorb the first impact;
    # a few contact flips while the estimator settles are acceptable, but
    # never a limit cycle.
    assert flips <= 8, f"contact flipped {flips} times -- bounce limit cycle"
    assert np.max(np.asarray(fz_hist)) < 7.0, "impact overshoot too large"
    assert abs(tail.mean() - 3.0) < 0.8, f"force did not settle at 3N (mean {tail.mean():.2f})"
    assert tail.std() < 0.6, f"force still oscillating (std {tail.std():.2f})"


def test_closed_loop_very_hard_surface_no_bounce_cascade():
    """A very stiff (20 kN/m) surface with the SHIPPED yaml must not enter a
    bounce cascade even on repeated re-impact. This is the direct guard
    against the scan_v5.csv failure mode (71 contact losses in 45 s)."""
    import yaml
    from pathlib import Path

    dt = 0.005
    raw = yaml.safe_load(Path("configs/joint_admittance_8dof.yaml").read_text())
    cfg = AdmittanceConfig.from_dict(raw)
    ctrl = AdmittanceController(dt, cfg)

    ke_true = 20000.0  # very hard
    x = -0.008
    flips = 0
    was_contact = False
    fz_hist: list[float] = []
    for _ in range(3000):  # 15 s
        fz = ke_true * max(x, 0.0)
        v = _tick(ctrl, fz=fz, f_des_z=3.0)
        x += v * dt
        in_c = fz >= cfg.contact_threshold_n
        if in_c != was_contact:
            flips += 1
        was_contact = in_c
        fz_hist.append(fz)

    tail = np.asarray(fz_hist[-600:])  # last 3 s
    # A "bounce cascade" is dozens of contact flips per second (scan_v5 had
    # 142 flips in 45s ≈ 3/s). Post-fix we accept a few initial bounces
    # while K̂_e is learning, then contact must stay latched.
    assert flips <= 16, (
        f"bounce cascade: {flips} contact flips in 15s (scan_v5 had 142 in 45s). "
        "Stiff-first K̂_e + Dimeas inertia + single vz cap must keep re-impact damped."
    )
    # Very-hard-surface first-impact peak: acceptable up to ~2.7× setpoint
    # (still well inside the safe envelope, whereas scan_v5 saw 9.4 N ≈ 3.1×).
    assert np.max(np.asarray(fz_hist)) < 8.0, (
        f"impact overshoot too large: {np.max(fz_hist):.2f} N"
    )
    assert abs(tail.mean() - 3.0) < 0.8, f"force did not settle at 3N (mean {tail.mean():.2f})"
    assert tail.std() < 0.6, f"force still oscillating (std {tail.std():.2f})"


def test_dimeas_5hz_forced_oscillation_adds_zero_centered_damping():
    """A 5 Hz forced fz oscillation (in the contact-resonance band that
    ``_update_instability_index``'s HP-filter targets) must raise Iₛ and
    add zero-centered damping.  The shipped Stage-2 policy deliberately keeps
    ``m_u=0``: Dimeas is a detector/dissipative layer, not the primary impact
    mechanism and not an online inertia switch.
    """
    import yaml
    from pathlib import Path

    dt = 0.005
    raw = yaml.safe_load(Path("configs/joint_admittance_8dof.yaml").read_text())
    cfg = AdmittanceConfig.from_dict(raw)
    ctrl = AdmittanceController(dt, cfg)
    ctrl._in_contact_latched = True

    m_base = ctrl._m_z_now
    assert ctrl.instability_index == 0.0
    assert abs(m_base - cfg.admittance_mass_z) < 1e-6

    import math as _m
    max_mass = m_base
    max_dimeas_damping = 0.0
    max_total_damping = ctrl.damping_z_eff
    for i in range(2000):  # 10 s of forced 5 Hz oscillation on raw fz
        t = i * dt
        fz = 3.0 + 3.0 * _m.sin(2.0 * _m.pi * 5.0 * t)
        f_ext = np.zeros(6); f_ext[2] = fz
        f_des = np.zeros(6); f_des[2] = 3.0
        f_raw = np.zeros(6); f_raw[2] = fz
        ctrl.compute_velocity_command(
            np.zeros(6), np.zeros(6), np.zeros(6), f_ext, f_des,
            in_contact=True, f_ext_raw=f_raw,
        )
        max_mass = max(max_mass, ctrl._m_z_now)
        max_dimeas_damping = max(
            max_dimeas_damping,
            ctrl.damping_dimeas_z,
        )
        max_total_damping = max(max_total_damping, ctrl.damping_z_eff)

    assert ctrl.instability_index > 0.1, (
        f"5 Hz forced oscillation must raise Iₛ above 0.1, got "
        f"{ctrl.instability_index:.4f}"
    )
    assert cfg.var_damping_m_u == pytest.approx(0.0)
    assert ctrl._m_z_now == pytest.approx(m_base)
    assert max_mass == pytest.approx(m_base)
    assert max_dimeas_damping > 0.2
    assert max_total_damping > cfg.admittance_damping_z


def test_dimeas_disabled_leaves_mass_static():
    """With ``var_damping_enabled=False`` the inertia channel is inert:
    Iₛ stays at 0 and m_z stays at the configured admittance_mass_z, even
    under the same 5 Hz forced oscillation.
    """
    cfg = _base_cfg(admittance_mass_z=1.0, var_damping_enabled=False)
    ctrl = AdmittanceController(0.005, cfg)
    import math as _m
    for i in range(1000):
        t = i * 0.005
        fz = 3.0 + 3.0 * _m.sin(2.0 * _m.pi * 5.0 * t)
        f_ext = np.zeros(6); f_ext[2] = fz
        f_raw = np.zeros(6); f_raw[2] = fz
        ctrl.compute_velocity_command(
            np.zeros(6), np.zeros(6), np.zeros(6), f_ext, np.zeros(6),
            in_contact=True, f_ext_raw=f_raw,
        )
    assert ctrl.instability_index == 0.0
    assert abs(ctrl._m_z_now - cfg.admittance_mass_z) < 1e-9


def test_closed_loop_soft_surface_converges():
    """Same loop on soft tissue (300 N/m): must reach the setpoint (no
    permanent over-damped stall)."""
    import yaml
    from pathlib import Path

    dt = 0.005
    raw = yaml.safe_load(Path("configs/joint_admittance_8dof.yaml").read_text())
    cfg = AdmittanceConfig.from_dict(raw)
    ctrl = AdmittanceController(dt, cfg)

    ke_true = 300.0
    x = -0.005
    fz_hist: list[float] = []
    for _ in range(2400):  # 12 s
        fz = ke_true * max(x, 0.0)
        v = _tick(ctrl, fz=fz, f_des_z=3.0)
        x += v * dt
        fz_hist.append(fz)
    tail = np.asarray(fz_hist[-400:])
    assert abs(tail.mean() - 3.0) < 0.8, f"soft surface: mean {tail.mean():.2f}"
    assert tail.std() < 0.5


def test_production_stack_tracks_moving_surface_at_1n_and_5n():
    """The shipped 8-DoF stack (adaptive K̂e + Dimeas both enabled) must
    follow a compliant surface in either normal direction without restoring
    the old target-force-dependent response.

    A constant-velocity surface is deliberately used here: after the
    transient, the TCP velocity must match it and the residual force bias must
    stay inside the passive-admittance bias budget at both 1 N and 5 N.
    ``proactive_retract_only`` is intentionally asymmetric in the shipped
    safety baseline, so equal press/retract force error is not an invariant.
    """
    import yaml
    from pathlib import Path

    dt = 0.005
    raw = yaml.safe_load(
        Path("configs/joint_admittance_8dof.yaml").read_text()
    )
    ke_true = 800.0
    results: dict[tuple[float, float], tuple[float, float]] = {}

    for desired in (1.0, 5.0):
        for surface_velocity in (-0.01, 0.01):
            cfg = AdmittanceConfig.from_dict(raw)
            # Surface-velocity match uses the passive+proactive chase; DOB is
            # covered separately for steady-force bias rejection.
            cfg.force_dob.enabled = False
            ctrl = AdmittanceController(dt, cfg)
            tcp_z = desired / ke_true
            surface_z = 0.0
            force_tail: list[float] = []
            velocity_tail: list[float] = []

            for tick in range(2000):
                force_z = max(
                    0.0,
                    ke_true * (tcp_z - surface_z),
                )
                force = np.zeros(6)
                force[2] = force_z
                target = np.zeros(6)
                target[2] = desired
                pose = np.zeros(6)
                pose[2] = tcp_z
                velocity = ctrl.compute_velocity_command(
                    pose,
                    pose,
                    np.zeros(6),
                    force,
                    target,
                    f_ext_raw=force,
                    dt_actual=dt,
                )[2]
                tcp_z += velocity * dt
                surface_z += surface_velocity * dt
                if tick >= 800:
                    force_tail.append(force_z)
                    velocity_tail.append(velocity)

            force_arr = np.asarray(force_tail)
            results[(desired, surface_velocity)] = (
                float(np.mean(np.abs(force_arr - desired))),
                float(np.mean(velocity_tail)),
            )

    # A passive steady chase needs approximately D*v plus the smooth-deadband
    # offset.  At D=25 Ns/m and v=10 mm/s this is ~0.38 N, independent of the
    # 1/5 N setpoint.  The old 0.25 N assertion contradicted that configured
    # plant and encouraged re-enabling proactive press merely to satisfy a
    # test.
    passive_bias_budget = (
        cfg.admittance_damping_z * 0.01
        + cfg.deadband_n
        + 0.5 * cfg.deadband_width_n
        + 0.05
    )
    for desired in (1.0, 5.0):
        assert results[(desired, -0.01)][0] <= passive_bias_budget
        assert results[(desired, 0.01)][0] <= passive_bias_budget
    for desired in (1.0, 5.0):
        assert results[(desired, -0.01)][1] == pytest.approx(
            -0.01,
            abs=2e-4,
        )
        assert results[(desired, 0.01)][1] == pytest.approx(
            0.01,
            abs=2e-4,
        )
