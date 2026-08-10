"""Deterministic checks for the scalar bidirectional-flow (BEFM) adapter.

The adapter is intentionally exercised through its public ``update`` API and
telemetry.  These tests do not assert a torque-level theorem: they pin down
the engineering adapter's one-sided retract path, press gate, feedback
fail-closed behavior, and energy bookkeeping.
"""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.admittance_common.bidirectional_flow import (
    BidirectionalFlowConfig,
    BidirectionalFlowController,
)


def _active_cfg(**overrides: object) -> BidirectionalFlowConfig:
    """Small deterministic config with no proxy/real-port mismatch terms."""

    values: dict[str, object] = dict(
        mode="active",
        sign_verified=True,
        feedback_delay_verified=True,
        require_sign_verification=True,
        require_delay_verification=True,
        Kd=1.0,
        Kp=0.0,
        Ki=0.0,
        lambda_gain=2.0,
        M_p=1.0,
        D_p=0.0,
        aux_tau_s=0.0,
        aux_max_retract_m_s=1.0,
        alpha_attack_s=0.0,
        alpha_release_s=0.0,
        nominal_damping=0.0,
        positive_switching_cost_j=0.0,
        T0=0.003,
        Tmax=0.004,
        Tmin=0.001,
    )
    values.update(overrides)
    return BidirectionalFlowConfig(**values)


def _step(
    ctrl: BidirectionalFlowController,
    vp: float,
    *,
    va: float = 0.0,
    force: float = 0.0,
    age: float | None = 0.0,
    fresh: bool | None = True,
    dt: float | None = None,
) -> float:
    return ctrl.update(
        vp_cmd=vp,
        x_actual=0.0,
        v_actual=va,
        force=force,
        dt_actual=dt,
        feedback_age_s=age,
        feedback_fresh=fresh,
    )


def test_alpha_piecewise_nonpositive_pe_and_positive_power_cases() -> None:
    """Lee's nonpositive, Pe, and Pc branches are exact and deterministic."""

    no_press = BidirectionalFlowController(0.01, _active_cfg())
    out = _step(no_press, 0.0, force=1.0)
    assert out == pytest.approx(0.0)
    assert no_press.alpha_raw == pytest.approx(0.0)
    assert no_press.alpha == pytest.approx(0.0)
    assert no_press.alpha_case == "nonpositive"

    environment_power = BidirectionalFlowController(
        0.01,
        _active_cfg(lambda_gain=1.0),
    )
    out = _step(environment_power, 0.10, force=10.0)
    assert out == pytest.approx(0.0)  # alpha=1 closes the active press path
    assert environment_power.Pe == pytest.approx(
        (environment_power.vp - environment_power.va) * 10.0
    )
    assert environment_power.Pc == pytest.approx(
        (environment_power.vp - environment_power.va) * environment_power.fc
    )
    assert environment_power.Pe > environment_power.Pc > 0.0
    assert environment_power.alpha_raw == pytest.approx(1.0)
    assert environment_power.alpha == pytest.approx(1.0)
    assert environment_power.alpha_case == "Pe"

    mismatch_power = BidirectionalFlowController(0.01, _active_cfg(lambda_gain=2.0))
    out = _step(mismatch_power, 0.10, force=0.1)
    assert mismatch_power.Pe == pytest.approx(
        (mismatch_power.vp - mismatch_power.va) * 0.1
    )
    assert mismatch_power.Pc == pytest.approx(
        (mismatch_power.vp - mismatch_power.va) * mismatch_power.fc
    )
    assert mismatch_power.Pe == pytest.approx(0.01)
    assert mismatch_power.Pc == pytest.approx(0.01)
    assert mismatch_power.alpha_raw == pytest.approx(0.5)
    assert mismatch_power.alpha == pytest.approx(0.5)
    assert mismatch_power.alpha_case == "Pc"
    assert out > 0.0


def test_alpha_one_is_fail_closed_for_stale_feedback_and_low_tank() -> None:
    """Stale feedback and a depleted tank close only the active press path."""

    stale = BidirectionalFlowController(0.01, _active_cfg())
    out = _step(stale, 0.03, age=0.10, fresh=True)
    assert out == pytest.approx(0.0)
    assert stale.feedback_stale
    assert stale.alpha_raw == pytest.approx(1.0)
    assert stale.alpha == pytest.approx(1.0)
    assert stale.alpha_case == "stale"
    assert stale.blocked_reason == "feedback_stale"

    low_tank = BidirectionalFlowController(
        0.01,
        _active_cfg(T0=0.001, Tmin=0.001),
    )
    out = _step(low_tank, 0.03)
    assert out == pytest.approx(0.0)
    assert low_tank.tank_energy == pytest.approx(low_tank.cfg.Tmin)
    assert low_tank.alpha_raw == pytest.approx(1.0)
    assert low_tank.alpha == pytest.approx(1.0)
    assert low_tank.alpha_case == "tank_low"
    assert low_tank.gamma_effective == pytest.approx(0.0)


def test_retract_through_is_not_alpha_gated() -> None:
    """A negative proxy request passes through even while the press gate is closed."""

    ctrl = BidirectionalFlowController(0.01, _active_cfg())
    # First close the gate with stale positive feedback.
    _step(ctrl, 0.03, age=0.10, fresh=True)
    assert ctrl.alpha == pytest.approx(1.0)

    # Retract with a fresh sample.  The one-sided auxiliary path never commands
    # a positive velocity, while retract-through remains negative; alpha only
    # multiplies positive press velocity.
    out = _step(ctrl, -0.02, age=0.0, fresh=True)
    assert ctrl.retract_through < 0.0
    assert ctrl.v_aux <= 0.0
    assert out < 0.0
    assert out == pytest.approx(ctrl.v_aux + ctrl.retract_through)
    assert ctrl.press == pytest.approx(0.0)


def test_tank_depletion_is_monotone_and_clamps_at_tmin() -> None:
    """Positive active flow debits the tank until the hard lower bound."""

    cfg = _active_cfg(
        T0=0.0015,
        Tmax=0.004,
        Tmin=0.001,
        active_press_debit_n=1.0,
        lambda_gain=2.0,
    )
    ctrl = BidirectionalFlowController(0.01, cfg)
    energies: list[float] = []
    cases: list[str] = []
    for _ in range(8):
        _step(ctrl, 0.10, force=0.1)
        energies.append(ctrl.tank_energy)
        cases.append(ctrl.alpha_case)

    assert all(b <= a + 1e-12 for a, b in zip(energies, energies[1:]))
    assert all(e >= cfg.Tmin - 1e-12 for e in energies)
    assert energies[-1] == pytest.approx(cfg.Tmin)
    assert ctrl.alpha == pytest.approx(1.0)
    assert "Pc" in cases and "tank_low" in cases
    assert ctrl.gamma_effective == pytest.approx(0.0)
    assert ctrl.command == pytest.approx(0.0)
    # Once depleted, no press command is emitted; the retract side remains
    # available independently of the tank gate.
    assert _step(ctrl, -0.02) < 0.0


def test_alpha_gate_structure_is_bounded_and_telemetry_is_finite() -> None:
    """Alpha and its raw piecewise value are always in [0, 1]."""

    ctrl = BidirectionalFlowController(0.005, _active_cfg())
    for vp, force, age in (
        (0.02, 0.1, 0.0),
        (0.0, 0.1, 0.0),
        (-0.02, 0.1, 0.0),
        (0.02, 0.1, 0.2),
    ):
        _step(ctrl, vp, force=force, age=age, fresh=True)
        assert 0.0 <= ctrl.alpha_raw <= 1.0
        assert 0.0 <= ctrl.alpha <= 1.0
        assert np.isfinite(ctrl.tank_energy)
        assert np.isfinite(ctrl.command)
        assert np.isfinite(ctrl.P_phys)
        assert np.isfinite(ctrl.P_mismatch)
        assert ctrl.alpha_case in {"Pe", "Pc", "nonpositive", "stale", "tank_low"}
        assert ctrl.telemetry.alpha_case == ctrl.alpha_case


def test_delayed_one_dimensional_hard_contact_does_not_press_on_stale_sample() -> None:
    """Compact delayed-contact smoke test for the fail-closed press gate.

    A point mass approaches a wall at x=0.  The force sample is delayed by one
    tick; while delayed, feedback is marked stale and the active flow adapter
    must not add a positive press command.  A negative/retract request remains
    available, which is the safety property used by the larger simulation.
    """

    ctrl = BidirectionalFlowController(
        0.01,
        _active_cfg(max_feedback_age_s=0.015),
    )
    x = -0.03
    v = 0.02
    delayed_force = 0.0
    for tick in range(8):
        # Delayed hard-contact force (positive on penetration) and one-tick
        # stale flag emulate a sensor/transport delay without a simulator
        # dependency.
        contact_force = max(0.0, x) * 100.0
        stale = tick > 0
        cmd = ctrl.update(
            vp_cmd=0.02,
            x_actual=x,
            v_actual=v,
            force=delayed_force,
            dt_actual=0.01,
            feedback_age_s=0.02 if stale else 0.0,
            feedback_fresh=True,
        )
        if stale:
            assert cmd <= 1e-12
        delayed_force = contact_force
        # Integrate the proxy command as a compact 1-D plant, with a hard wall
        # projection that supplies a retract velocity after penetration.
        v = cmd
        x = min(0.0, x + v * 0.01)
