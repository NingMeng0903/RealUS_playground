"""Leaky ∫F_err proactive reference (bidirectional chase + bounce guards)."""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from pathlib import Path

from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from rm75_control.control.admittance_common.scaling import (
    scale_admittance_for_desired_z,
)
from rm75_control.control.admittance_common.proactive_force_ff import (
    ProactiveFfConfig,
    ProactiveForceIntegrator,
)

DT = 0.005


def test_integrator_bidirectional_press_and_retract():
    ff = ProactiveForceIntegrator(
        ProactiveFfConfig(enabled=True, retract_only=False, gain=0.10, leak_s=10.0)
    )
    for _ in range(200):
        ff.update(1.0, in_contact=True, dt_eff=DT, instability_index=0.0, v_force_z=0.0, v_z_cap=0.10)
    assert ff.v_r > 0.01
    ff.reset()
    for _ in range(200):
        ff.update(-1.0, in_contact=True, dt_eff=DT, instability_index=0.0, v_force_z=0.0, v_z_cap=0.10)
    assert ff.v_r < -0.01


def test_instability_gates_press_but_keeps_retract_escape_open():
    ff_lo = ProactiveForceIntegrator(ProactiveFfConfig(gain=0.10, leak_s=10.0, press_is_gate=0.5))
    ff_hi = ProactiveForceIntegrator(ProactiveFfConfig(gain=0.10, leak_s=10.0, press_is_gate=0.5))
    ff_retract_lo = ProactiveForceIntegrator(ProactiveFfConfig(gain=0.10, leak_s=10.0, press_is_gate=0.5))
    ff_retract_hi = ProactiveForceIntegrator(ProactiveFfConfig(gain=0.10, leak_s=10.0, press_is_gate=0.5))
    for _ in range(50):
        ff_lo.update(2.0, in_contact=True, dt_eff=DT, instability_index=0.0, v_force_z=0.0, v_z_cap=0.10)
        ff_hi.update(2.0, in_contact=True, dt_eff=DT, instability_index=0.25, v_force_z=0.0, v_z_cap=0.10)
        ff_retract_lo.update(-2.0, in_contact=True, dt_eff=DT, instability_index=0.0, v_force_z=0.0, v_z_cap=0.10)
        ff_retract_hi.update(-2.0, in_contact=True, dt_eff=DT, instability_index=2.0, v_force_z=0.0, v_z_cap=0.10)
    assert ff_hi.v_r < ff_lo.v_r - 1e-6
    assert ff_retract_hi.v_r == pytest.approx(ff_retract_lo.v_r)
    assert ff_retract_hi.last_instability_scale == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("instability", "expected_scale"),
    [(0.10, 1.0), (0.20, 1.0), (0.40, 0.5), (0.60, 0.0), (1.0, 0.0)],
)
def test_press_gate_has_noise_floor_and_unchanged_hard_stop(
    instability: float,
    expected_scale: float,
):
    ff = ProactiveForceIntegrator(
        ProactiveFfConfig(
            gain=0.10,
            leak_s=10.0,
            press_is_gate_start=0.20,
            press_is_gate=0.60,
        )
    )
    ff.update(
        1.0,
        in_contact=True,
        dt_eff=DT,
        instability_index=instability,
        v_force_z=0.0,
        v_z_cap=0.10,
    )
    assert ff.last_instability_scale == pytest.approx(expected_scale)


def test_rising_edge_clears_press_v_r():
    ctrl = AdmittanceController(
        DT,
        AdmittanceConfig(
            proactive_ff=ProactiveFfConfig(enabled=True),
            adaptive_ke=AdmittanceConfig().adaptive_ke,
        ),
    )
    ctrl.cfg.adaptive_ke.enabled = False
    ctrl._proactive_ff.v_r = 0.04
    ctrl._update_proactive_v_r(0.0, True, DT, rising_edge=True)
    assert ctrl.v_r_z == pytest.approx(0.0)


def test_stable_controller_normalizes_setpoint_and_is_small_signal_symmetric():
    cfg = AdmittanceConfig(
        proactive_ff=ProactiveFfConfig(
            enabled=True,
            gain=0.10,
            leak_s=1e6,
            press_is_gate=0.5,
        ),
    )

    def integrate(error: float, desired: float, instability: float) -> float:
        ctrl = AdmittanceController(DT, cfg)
        ctrl._in_contact_latched = True
        ctrl.instability_index = instability
        for _ in range(100):
            ctrl._update_proactive_v_r(
                error,
                True,
                DT,
                rising_edge=False,
                desired_force_n=desired,
            )
        return ctrl.v_r_z

    # Half-scale errors exercise the unsaturated normalized law.  With no
    # detected instability both signs have exactly the same small-error gain.
    press_1n = integrate(0.15, 1.0, 0.0)
    press_5n = integrate(0.375, 5.0, 0.0)
    retract_1n = integrate(-0.15, 1.0, 0.0)
    retract_5n = integrate(-0.375, 5.0, 0.0)
    assert press_5n == pytest.approx(press_1n, rel=1e-9)
    assert retract_5n == pytest.approx(retract_1n, rel=1e-9)
    assert retract_1n == pytest.approx(-press_1n, rel=1e-9)


def test_same_contact_reversal_discards_old_reference_in_both_directions():
    cfg = ProactiveFfConfig(
        gain=0.10,
        retract_gain=0.10,
        leak_s=0.3,
        press_is_gate=0.5,
        reset_on_reversal=True,
    )
    ff = ProactiveForceIntegrator(cfg)
    for _ in range(100):
        ff.update(
            1.0,
            in_contact=True,
            dt_eff=DT,
            instability_index=0.0,
            v_force_z=0.0,
            v_z_cap=0.10,
        )
    assert ff.v_r > 0.0
    ff.update(
        -1.0,
        in_contact=True,
        dt_eff=DT,
        instability_index=2.0,
        v_force_z=0.0,
        v_z_cap=0.10,
    )
    assert ff.last_reversal_reset is True
    assert ff.v_r < 0.0
    assert ff.last_reference_accel_m_s2 < 0.0

    for _ in range(100):
        ff.update(
            -1.0,
            in_contact=True,
            dt_eff=DT,
            instability_index=0.0,
            v_force_z=0.0,
            v_z_cap=0.10,
        )
    assert ff.v_r < 0.0
    ff.update(
        1.0,
        in_contact=True,
        dt_eff=DT,
        instability_index=0.0,
        v_force_z=0.0,
        v_z_cap=0.10,
    )
    assert ff.last_reversal_reset is True
    assert ff.v_r > 0.0


def test_zero_effective_error_leaks_without_false_reversal_reset():
    ff = ProactiveForceIntegrator(
        ProactiveFfConfig(leak_s=0.3, reset_on_reversal=True)
    )
    ff.v_r = 0.02
    ff.update(
        0.0,
        in_contact=True,
        dt_eff=DT,
        instability_index=1.0,
        v_force_z=0.0,
        v_z_cap=0.10,
    )
    assert ff.last_reversal_reset is False
    assert 0.0 < ff.v_r < 0.02


def test_reference_drive_and_antiwindup_are_bounded_on_both_signs():
    ff = ProactiveForceIntegrator(
        ProactiveFfConfig(
            gain=0.10,
            retract_gain=0.10,
            leak_s=1e6,
            press_drive_max=1.0,
            retract_drive_max=1.0,
            v_r_max_m_s=0.06,
        )
    )
    ff.update(
        -100.0,
        in_contact=True,
        dt_eff=DT,
        instability_index=100.0,
        v_force_z=0.0,
        v_z_cap=0.10,
        desired_force_n=1.0,
    )
    assert ff.last_drive == pytest.approx(-1.0)
    assert ff.last_reference_accel_m_s2 == pytest.approx(-0.10)

    # Outward integration freezes at the force-velocity cap.
    ff.update(
        -100.0,
        in_contact=True,
        dt_eff=DT,
        instability_index=100.0,
        v_force_z=-0.10,
        v_z_cap=0.10,
        desired_force_n=1.0,
    )
    assert ff.last_reference_accel_m_s2 == pytest.approx(0.0)


def test_stable_controller_rising_edge_clears_either_old_direction():
    ctrl = AdmittanceController(DT, AdmittanceConfig())
    for old_reference in (-0.04, 0.04):
        ctrl._proactive_ff.v_r = old_reference
        ctrl._update_proactive_v_r(
            0.0,
            True,
            DT,
            rising_edge=True,
            desired_force_n=2.0,
        )
        assert ctrl.v_r_z == pytest.approx(0.0)


def test_stable_controller_keeps_2965_parameters_fixed_across_setpoints():
    raw = yaml.safe_load(Path("configs/joint_admittance_8dof.yaml").read_text())
    assert "controller_mode" not in raw["hybrid_motion"]
    cfg1 = scale_admittance_for_desired_z(raw, 1.0)
    cfg5 = scale_admittance_for_desired_z(raw, 5.0)
    assert cfg1.admittance_mass_z == pytest.approx(cfg5.admittance_mass_z)
    assert cfg1.var_damping_f_max_n == pytest.approx(
        cfg5.var_damping_f_max_n
    )
    assert cfg1.adaptive_ke.bd_max == pytest.approx(cfg5.adaptive_ke.bd_max)
    ctrl = AdmittanceController(DT, cfg1)
    assert ctrl.controller_mode == "legacy_symmetric"


def test_stable_controller_tracks_moving_surface_at_1n_and_5n_without_bias():
    raw = yaml.safe_load(Path("configs/joint_admittance_8dof.yaml").read_text())
    ke_n_m = 800.0
    results = {}
    for desired in (1.0, 5.0):
        for surface_velocity in (-0.01, 0.01):
            cfg = scale_admittance_for_desired_z(raw, desired)
            # Isolate the corrected force-reference dynamics in this linear
            # contact test. Ke/Dimeas retain their 2965fea implementations and
            # have their own regression tests.
            cfg.adaptive_ke.enabled = False
            cfg.var_damping_enabled = False
            cfg.force_dob.enabled = False
            cfg.force_barrier.enabled = False
            cfg.delay_damping_enabled = False
            # Hardware passivity baseline tightens press; this unit test needs
            # bidirectional chase on a moving linear spring surface.
            cfg.proactive_ff.retract_only = False
            cfg.delay_press_budget_enabled = False
            cfg.low_force_press_cap_m_s = 0.0
            cfg.suspect_recovery_enabled = False
            cfg.v_force_aw_enabled = False
            # Linear spring track: no bounce-cycle retract brake / interlock.
            cfg.fast_retract_guard.retract_stop_prediction_s = 0.0
            cfg.retract_brake_damping_ns_m = 0.0
            cfg.reverse_interlock_enter_m_s = 0.0
            cfg.impact_fdot_arm_n_s = 1.0e9
            cfg.impact_fpred_over_n = 1.0e9
            cfg.impact_danger_f_over_n = 1.0e9
            cfg.impact_danger_fdot_n_s = 1.0e9
            cfg.press_energy_tank.enabled = False
            cfg.port_passivity.enabled = False
            cfg.deadband_soft_tanh = False
            cfg.deadband_n = 0.0
            cfg.deadband_width_n = 0.0
            cfg.force_slew_press_m_s2 = 100.0
            cfg.force_slew_retract_m_s2 = 100.0
            cfg.force_slew_press_to_retract_m_s2 = 100.0
            cfg.force_slew_zero_cross_m_s2 = 100.0
            cfg.free_seek_accel_m_s2 = 100.0
            cfg.contact_press_cap_m_s = 0.10
            ctrl = AdmittanceController(DT, cfg)
            ctrl.free_seek_active = True
            tcp_z = desired / ke_n_m
            surface_z = 0.0
            samples = []
            velocity_samples = []
            for tick in range(1600):
                force_z = max(0.0, ke_n_m * (tcp_z - surface_z))
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
                )[2]
                tcp_z += velocity * DT
                surface_z += surface_velocity * DT
                if tick >= 600:
                    samples.append(force_z)
                    velocity_samples.append(velocity)
            results[(desired, surface_velocity)] = (
                float(np.mean(np.abs(np.asarray(samples) - desired))),
                float(np.mean(velocity_samples)),
            )

    assert results[(1.0, -0.01)][0] <= 0.20
    assert results[(1.0, 0.01)][0] <= 0.20
    assert results[(5.0, -0.01)][0] <= 0.50
    assert results[(5.0, 0.01)][0] <= 0.50
    for desired in (1.0, 5.0):
        negative_error = results[(desired, -0.01)][0]
        positive_error = results[(desired, 0.01)][0]
        # Asymmetric press/retract proactive gains (faster over-force escape)
        # allow a modest directional bias on moving surfaces.
        assert max(negative_error, positive_error) <= 2.0 * min(
            negative_error,
            positive_error,
        )
        assert results[(desired, -0.01)][1] == pytest.approx(
            -0.01,
            abs=2e-4,
        )
        assert results[(desired, 0.01)][1] == pytest.approx(
            0.01,
            abs=2e-4,
        )


def _controller(**over) -> AdmittanceController:
    from rm75_control.control.admittance_common.press_energy_tank import (
        PortPassivityConfig,
        PressEnergyTankConfig,
    )

    kw = dict(
        contact_threshold_n=0.8,
        deadband_n=0.0,
        deadband_width_n=0.0,
        max_vz_tool_m_s=0.10,
        max_velocity=np.array([0.2, 0.2, 0.10, 0.5, 0.5, 0.5]),
        admittance_mass_z=1.0,
        admittance_damping_z=25.0,
        var_damping_enabled=False,
        delay_press_budget_enabled=False,
        low_force_press_cap_m_s=0.0,
        suspect_recovery_enabled=False,
        v_force_aw_enabled=False,
        reverse_interlock_enter_m_s=0.0,
        press_energy_tank=PressEnergyTankConfig(enabled=False),
        port_passivity=PortPassivityConfig(enabled=False),
        free_seek_vz_m_s=0.10,
        contact_press_cap_m_s=0.10,
        impact_danger_f_over_n=1.0e9,
        impact_danger_fdot_n_s=1.0e9,
        proactive_ff=ProactiveFfConfig(
            enabled=True,
            retract_only=False,
            gain=0.10,
            leak_s=0.3,
            v_r_max_m_s=0.06,
        ),
    )
    kw.update(over)
    cfg = AdmittanceConfig(**kw)
    cfg.adaptive_ke.enabled = False
    return AdmittanceController(DT, cfg)


def test_proactive_boosts_velocity_under_sustained_error():
    ctrl = _controller()
    ctrl.cfg.force_barrier.enabled = False
    ctrl.cfg.delay_damping_enabled = False
    ctrl._in_contact_latched = True
    for _ in range(400):
        ctrl._admittance_z(
            2.0,
            True,
            dt_eff=DT,
            rising_edge=False,
            physical_contact=True,
            f_ext_z=0.0,
            desired_force_n=2.0,
        )
    assert ctrl.v_r_z > 0.015
    assert ctrl.v_force_z > 0.08


def test_high_instability_cannot_delay_overforce_escape_after_reversal():
    ctrl = _controller(var_damping_enabled=True)
    ctrl.cfg.force_barrier.enabled = False
    ctrl.cfg.delay_damping_enabled = False
    ctrl._in_contact_latched = True

    # Build the exact stale state seen in the hardware logs: positive TCP-Z
    # velocity and a positive active reference immediately before a fast
    # over-force push.
    for _ in range(200):
        ctrl._m_z_now = 1.0
        ctrl.instability_index = 0.0
        ctrl._admittance_z(
            1.0,
            True,
            dt_eff=DT,
            rising_edge=False,
            desired_force_n=2.0,
            physical_contact=True,
            f_ext_z=1.0,
        )
    assert ctrl.v_force_z > 0.0
    assert ctrl.v_r_z > 0.0

    first_tick_reset = False
    ticks_to_retract = None
    for tick in range(1, 41):
        # Is=0.7 is above the configured press gate.  Dimeas mass and damping
        # remain active, but they may not close the over-force escape branch.
        ctrl._m_z_now = 3.8
        ctrl.instability_index = 0.7
        ctrl._admittance_z(
            -2.0,
            True,
            dt_eff=DT,
            rising_edge=False,
            desired_force_n=2.0,
            physical_contact=True,
            f_ext_z=4.0,
        )
        if tick == 1:
            first_tick_reset = ctrl.force_reference_reversal_reset
            assert ctrl.v_r_z < 0.0
            assert ctrl.force_reference_gate_scale == pytest.approx(1.0)
        if ctrl.v_force_z < 0.0:
            ticks_to_retract = tick
            break

    assert first_tick_reset is True
    assert ticks_to_retract is not None
    assert ticks_to_retract * DT <= 0.10


def test_yaml_proactive_retract_only_passivity_baseline():
    raw = yaml.safe_load(Path("configs/joint_admittance_8dof.yaml").read_text())
    hm = raw["hybrid_motion"]
    assert hm["proactive_feedforward"] is True
    # Passivity A/B: no proactive press injection (Dv_r active power).
    assert hm["proactive_retract_only"] is True
    assert float(hm["proactive_gain"]) > 0.0
    assert float(hm["proactive_retract_gain"]) >= float(hm["proactive_gain"])
    assert 0.0 <= hm["proactive_press_is_gate_start"] < hm[
        "proactive_press_is_gate"
    ]
    assert hm.get("proactive_gate_press_on_is", True) is False
    assert float(hm["proactive_press_drive_max"]) >= 1.0
    assert float(hm["proactive_retract_drive_max"]) >= float(
        hm["proactive_press_drive_max"]
    )
    assert hm["proactive_reset_on_reversal"] is True
    assert hm["v_r_max_m_s"] < hm["max_vz_tool_m_s"]
    assert "li2022" not in hm
    assert hm.get("v_force_aw_enabled", True) is False
    assert hm.get("force_dob", {}).get("enabled", True) is False
    assert hm.get("var_damping_enabled", True) is False
