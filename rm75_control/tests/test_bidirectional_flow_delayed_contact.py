"""Deterministic 1-D delayed-contact smoke tests for the BEFM adapter.

The plant below is intentionally small: a velocity-servo lag drives a point
mass against a Kelvin--Voigt spring.  Force samples are delayed, while the
actual velocity sample remains fresh (the normal-axis implementation requires
that distinction).  This is a speed-level engineering smoke test, not a
torque-level passivity theorem or a substitute for the hardware acceptance
sequence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rm75_control.control.admittance_common.bidirectional_flow import (
    BidirectionalFlowConfig,
    BidirectionalFlowController,
)


def _simulation_config() -> BidirectionalFlowConfig:
    """Conservative, deterministic gains for the scalar simulation."""

    return BidirectionalFlowConfig(
        mode="active",
        sign_verified=True,
        feedback_delay_verified=True,
        require_sign_verification=True,
        require_delay_verification=True,
        Kd=0.5,
        Kp=5.0,
        Ki=0.0,
        lambda_gain=1.0,
        Dtrack=0.5,
        M_p=0.05,
        D_p=0.05,
        M_a=0.02,
        D_a=0.6,
        K_a=20.0,
        aux_max_retract_m_s=0.08,
        T0=0.003,
        Tmax=0.004,
        Tmin=0.0001,
        # The requested press is intentionally cheap enough to reach the
        # contact surface before the small engineering tank is depleted.
        active_press_debit_n=0.01,
        nominal_damping=0.1,
        alpha_attack_s=0.02,
        alpha_release_s=0.15,
        max_feedback_age_s=0.020,
    )


def _simulate_delayed_contact(
    delay_s: float,
    stiffness_n_m: float,
    *,
    contact_loss: bool = False,
) -> dict[str, Any]:
    """Run one deterministic delayed-force/velocity-servo contact trace."""

    dt = 0.005  # 200 Hz normal-axis tick
    steps = 600  # 3 s; long enough to include contact and settling
    delay_ticks = int(round(delay_s / dt))
    controller = BidirectionalFlowController(dt, _simulation_config())

    # x=0 is the undeformed surface; x>0 is spring compression.  Start only
    # 5 mm away so every grid point reaches contact without a long free-space
    # transient.  The delayed command queue models the black-box velocity
    # servo, and the delayed force queue models force transport/sensing.
    x_actual = -0.005
    v_actual = 0.0
    command_delay = [0.0] * (delay_ticks + 1)
    force_delay = [0.0] * (delay_ticks + 1)
    plant_mass_kg = 0.20
    servo_tau_s = 0.030

    trace: dict[str, list[float]] = {
        "x": [],
        "v": [],
        "force": [],
        "force_measured": [],
        "command": [],
        "alpha": [],
        "tank": [],
        "Pe": [],
        "Pc": [],
        "P_phys": [],
        "P_mismatch": [],
        "stale": [],
    }

    for tick in range(steps):
        t = tick * dt
        physical_force = stiffness_n_m * max(x_actual, 0.0)
        delayed_force = force_delay.pop(0)
        force_delay.append(physical_force)

        # A short force/feedback dropout emulates contact loss.  During that
        # interval the active press path must fail closed, while retract stays
        # available.  Actual velocity itself is still supplied every tick.
        lost = contact_loss and 1.35 <= t < 1.55
        force_age = 0.10 if lost else 0.0
        force_fresh = not lost
        vp_request = 0.015 if t < 1.80 else 0.0

        command = controller.update(
            vp_cmd=vp_request,
            x_actual=x_actual,
            v_actual=v_actual,
            force=delayed_force,
            dt_actual=dt,
            feedback_age_s=force_age,
            feedback_fresh=force_fresh,
        )

        delayed_command = command_delay.pop(0)
        command_delay.append(command)

        # Point-mass plant with a first-order velocity servo and spring force.
        # No numerical clipping is needed for the nominal grid, but a generous
        # velocity guard keeps a malformed implementation from poisoning the
        # rest of the deterministic trace with infinities.
        acceleration = (
            (delayed_command - v_actual) / servo_tau_s
            - physical_force / plant_mass_kg
        )
        v_actual = float(np.clip(v_actual + dt * acceleration, -0.30, 0.30))
        x_actual += dt * v_actual
        if x_actual < -0.040:
            x_actual = -0.040
            v_actual = max(v_actual, 0.0)

        trace["x"].append(float(x_actual))
        trace["v"].append(float(v_actual))
        trace["force"].append(float(physical_force))
        trace["force_measured"].append(float(delayed_force))
        trace["command"].append(float(command))
        trace["alpha"].append(float(controller.alpha))
        trace["tank"].append(float(controller.tank_energy))
        trace["Pe"].append(float(controller.Pe))
        trace["Pc"].append(float(controller.Pc))
        trace["P_phys"].append(float(controller.P_phys))
        trace["P_mismatch"].append(float(controller.P_mismatch))
        trace["stale"].append(float(controller.feedback_stale))

    return {
        "controller": controller,
        "trace": {name: np.asarray(values, dtype=float) for name, values in trace.items()},
        "dt": dt,
        "loss_slice": slice(int(1.35 / dt), int(1.55 / dt)) if contact_loss else None,
    }


@pytest.mark.parametrize("delay_s", (0.015, 0.045, 0.090))
@pytest.mark.parametrize("stiffness_n_m", (50.0, 500.0, 5000.0))
def test_delayed_contact_grid_is_finite_and_non_growing(
    delay_s: float,
    stiffness_n_m: float,
) -> None:
    """15/45/90 ms and 50/500/5000 N/m remain bounded in the smoke plant."""

    result = _simulate_delayed_contact(delay_s, stiffness_n_m)
    trace = result["trace"]
    controller: BidirectionalFlowController = result["controller"]

    # Every logged scalar is an audit field of the normal-axis structure.
    assert all(np.all(np.isfinite(values)) for values in trace.values())
    assert trace["tank"].min() >= controller.cfg.Tmin - 1.0e-12
    assert trace["force"].max() > 0.0  # contact was actually reached

    # A late impact envelope must not grow over the early envelope.  The
    # additive tolerance handles a zero-force first half without making that
    # case vacuous; this is deliberately a smoke-test bound, not a theorem.
    midpoint = trace["force"].size // 2
    early_peak = float(np.max(trace["force"][:midpoint]))
    late_peak = float(np.max(trace["force"][midpoint:]))
    assert late_peak <= 1.35 * early_peak + 0.50


def test_contact_loss_closes_press_but_keeps_retract_path() -> None:
    """A deterministic dropout cannot cause a positive press burst."""

    result = _simulate_delayed_contact(0.015, 500.0, contact_loss=True)
    trace = result["trace"]
    controller: BidirectionalFlowController = result["controller"]
    loss_slice = result["loss_slice"]
    assert loss_slice is not None

    assert np.all(trace["stale"][loss_slice] > 0.5)
    assert np.max(trace["command"][loss_slice]) <= 1.0e-10
    assert trace["tank"].min() >= controller.cfg.Tmin - 1.0e-12
    assert all(np.all(np.isfinite(values)) for values in trace.values())

    # Explicitly exercise retract-through after the dropout.  It must not be
    # alpha-gated even though positive velocity is fail-closed.
    retract = controller.update(
        vp_cmd=-0.020,
        x_actual=float(trace["x"][-1]),
        v_actual=0.0,
        force=0.0,
        dt_actual=result["dt"],
        feedback_age_s=0.0,
        feedback_fresh=True,
    )
    assert retract < 0.0


def test_alpha_zero_real_port_tracks_proxy() -> None:
    """With no positive port power, the real port follows the proxy speed."""

    dt = 0.005
    cfg = _simulation_config()
    controller = BidirectionalFlowController(dt, cfg)
    x_actual = 0.0
    proxy_speed = 0.012
    commands: list[float] = []
    for _ in range(40):
        command = controller.update(
            vp_cmd=proxy_speed,
            x_actual=x_actual,
            v_actual=proxy_speed,
            force=0.0,
            dt_actual=dt,
            feedback_age_s=0.0,
            feedback_fresh=True,
        )
        commands.append(command)
        x_actual += proxy_speed * dt

    assert np.all(np.asarray(commands) == pytest.approx(proxy_speed, abs=1.0e-9))
    assert controller.alpha == pytest.approx(0.0, abs=1.0e-12)
    assert controller.alpha_raw == pytest.approx(0.0, abs=1.0e-12)
    # ``xp`` is integrated at the end of a tick while the supplied encoder
    # position is the beginning-of-tick sample, so one sample of transport
    # skew is expected even with identical velocities.
    assert abs(controller.xp - controller.xa) <= proxy_speed * dt + 1.0e-9


def test_alpha_one_proxy_is_implicitly_pulled_by_real_mismatch() -> None:
    """The alpha=1 branch removes press output and applies proxy coupling."""

    dt = 0.005
    cfg = _simulation_config()
    cfg.Kd = 1.0
    cfg.Kp = 100.0
    cfg.Dtrack = 1.0
    cfg.lambda_gain = 1.0
    cfg.alpha_attack_s = 0.0
    cfg.alpha_release_s = 0.0
    controller = BidirectionalFlowController(dt, cfg)

    # Positive environment power closes alpha.  Keep the nominal proxy drive
    # positive in the following ticks so the Lee branch remains in Pe>0; the
    # implicit -lambda*alpha*Fc term then visibly reduces proxy speed.
    controller.update(
        vp_cmd=0.020,
        x_actual=0.0,
        v_actual=0.0,
        force=10.0,
        dt_actual=dt,
        feedback_age_s=0.0,
        feedback_fresh=True,
    )
    assert controller.alpha == pytest.approx(1.0)
    controller.update(
        vp_cmd=0.020,
        x_actual=0.0,
        v_actual=0.0,
        force=10.0,
        dt_actual=dt,
        feedback_age_s=0.0,
        feedback_fresh=True,
    )

    assert controller.alpha == pytest.approx(1.0)
    assert controller.fc > 0.0
    assert 0.0 <= controller.vp < 0.020
    assert controller.command <= 1.0e-10


def test_stale_feedback_fail_closed_and_fresh_retract_open() -> None:
    """A stale positive sample closes press while negative speed passes."""

    controller = BidirectionalFlowController(0.005, _simulation_config())
    press = controller.update(
        vp_cmd=0.020,
        x_actual=0.0,
        v_actual=0.0,
        force=5.0,
        dt_actual=0.005,
        feedback_age_s=0.100,
        feedback_fresh=True,
    )
    assert press <= 1.0e-10
    assert controller.alpha == pytest.approx(1.0)
    assert controller.feedback_stale

    retract = controller.update(
        vp_cmd=-0.020,
        x_actual=0.0,
        v_actual=0.0,
        force=0.0,
        dt_actual=0.005,
        feedback_age_s=0.0,
        feedback_fresh=True,
    )
    assert retract < 0.0
