"""Low-latency raw-force guard for the active retract reference."""

from __future__ import annotations

import math

import pytest

from rm75_control.control.admittance_common.fast_retract_guard import (
    FastRetractGuard,
    FastRetractGuardConfig,
)
from rm75_control.control.admittance_common.proactive_force_ff import (
    ProactiveFfConfig,
    ProactiveForceIntegrator,
)


DT = 0.005


def _guard(**kwargs) -> FastRetractGuard:
    cfg = dict(
        cutoff_hz=20.0,
        stop_margin_n=0.25,
        stop_margin_fraction=0.05,
        rearm_margin_n=0.45,
        rearm_margin_fraction=0.10,
        stop_confirm_s=0.015,
        rearm_confirm_s=0.010,
        min_hold_s=0.025,
        max_sensor_age_s=0.020,
        # Disable prediction in legacy tests unless opted in.
        retract_stop_prediction_s=0.0,
        retract_stop_margin_n=0.10,
        retract_stop_confirm_s=0.005,
        retract_stop_fdot_n_s=15.0,
    )
    cfg.update(kwargs)
    return FastRetractGuard(FastRetractGuardConfig(**cfg))


def _step(
    guard: FastRetractGuard,
    raw_force_n: float,
    *,
    desired_force_n: float = 2.0,
    filtered_eff_n: float = -1.0,
    active_reference_m_s: float = -0.02,
    sensor_age_s: float | None = 0.001,
    instability_index: float = 0.0,
) -> bool:
    return guard.update(
        raw_force_n=raw_force_n,
        desired_force_n=desired_force_n,
        filtered_eff_n=filtered_eff_n,
        active_reference_m_s=active_reference_m_s,
        dt_s=DT,
        sensor_age_s=sensor_age_s,
        instability_index=instability_index,
    )


def _arm(guard: FastRetractGuard, target: float) -> None:
    for _ in range(8):
        assert not _step(
            guard,
            target + 1.0,
            desired_force_n=target,
        )
    assert guard.armed


def _trigger_hold(
    guard: FastRetractGuard,
    target: float,
    *,
    instability_index: float = 0.0,
) -> int:
    """Return raw-force ticks needed to enter the confirmed hold."""
    _arm(guard, target)
    stop_margin = max(
        guard.cfg.stop_margin_n,
        guard.cfg.stop_margin_fraction * target,
    )
    high = target + 0.8
    low = target - stop_margin - 0.25
    descending = [
        high + (low - high) * index / 20.0
        for index in range(21)
    ]
    descending.extend([low] * 20)
    for tick, raw_force in enumerate(descending, start=1):
        if _step(
            guard,
            raw_force,
            desired_force_n=target,
            instability_index=instability_index,
        ):
            return tick
    raise AssertionError("continuous raw-force fall did not trigger hold")


@pytest.mark.parametrize("target", [1.0, 5.0])
def test_confirmed_raw_fall_clears_only_negative_active_reference(
    target: float,
) -> None:
    guard = _guard()
    ticks = _trigger_hold(guard, target)
    assert ticks >= math.ceil(guard.cfg.stop_confirm_s / DT)
    assert guard.hold
    assert guard.stop_count == 1

    ff = ProactiveForceIntegrator(
        ProactiveFfConfig(
            gain=0.10,
            retract_gain=0.10,
            leak_s=0.3,
        )
    )
    ff.v_r = -0.02
    result = ff.update(
        -1.0,
        in_contact=True,
        dt_eff=DT,
        instability_index=2.0,
        v_force_z=-0.03,
        v_z_cap=0.10,
        desired_force_n=target,
        retract_fast_hold=guard.hold,
    )

    assert result == pytest.approx(0.0)
    assert ff.last_fast_retract_clear is True
    # The hold vetoes only the active reference; it does not synthesize a
    # positive command from the raw-force path.
    assert ff.last_reference_accel_m_s2 == pytest.approx(0.0)


def test_fast_hold_never_clears_a_positive_active_reference() -> None:
    ff = ProactiveForceIntegrator(
        ProactiveFfConfig(leak_s=0.3)
    )
    ff.v_r = 0.02

    result = ff.update(
        0.0,
        in_contact=True,
        dt_eff=DT,
        instability_index=2.0,
        v_force_z=0.0,
        v_z_cap=0.10,
        desired_force_n=2.0,
        retract_fast_hold=True,
    )

    assert 0.0 < result < 0.02
    assert ff.last_fast_retract_clear is False


def test_single_raw_force_noise_spike_cannot_trigger_stop() -> None:
    guard = _guard()
    target = 2.0
    _arm(guard, target)

    # Median-of-three plus time confirmation must reject an isolated low
    # sample while the filtered controller still reports over-force.
    samples = [target + 1.0, target - 2.0, target + 1.0]
    samples.extend([target + 1.0] * 8)
    holds = [
        _step(guard, sample, desired_force_n=target)
        for sample in samples
    ]

    assert not any(holds)
    assert guard.armed
    assert guard.stop_count == 0


def test_falling_force_still_above_target_does_not_stop_retract() -> None:
    guard = _guard()  # prediction off — legacy low-side only
    target = 2.0
    _arm(guard, target)

    # A legitimate moving-surface retract may settle with filtered/raw force
    # still above the target.  Merely falling below the high-side arm level is
    # not evidence that the delayed 6 Hz force has missed a target crossing.
    for index in range(30):
        raw_force = target + 0.7 - 0.6 * index / 29.0
        assert not _step(
            guard,
            raw_force,
            desired_force_n=target,
        )

    assert guard.fast_force_n > target
    assert guard.armed
    assert guard.stop_count == 0


def test_predictive_stop_before_low_side_crossing() -> None:
    """Fast fall: stop when F_pred,down ≤ Fd+margin, still above 1.75 N."""
    guard = _guard(
        retract_stop_prediction_s=0.045,
        retract_stop_margin_n=0.10,
        retract_stop_confirm_s=0.005,
        retract_stop_fdot_n_s=15.0,
    )
    target = 2.0
    _arm(guard, target)
    # ~40 N/s fall from 3.2 → pred reaches ~Fd while force still > 2.0.
    forces = [3.2 - 0.20 * i for i in range(12)]
    held_at = None
    for fz in forces:
        if _step(guard, fz, desired_force_n=target):
            held_at = fz
            break
    assert held_at is not None
    assert held_at > target - 0.25  # before legacy low-side 1.75
    assert guard.predictive_stop_count == 1
    assert guard.hold


def test_slow_fall_above_target_ignores_prediction() -> None:
    """Surface follow: small |ḟ| must not predictive-stop while F > Fd."""
    guard = _guard(
        retract_stop_prediction_s=0.045,
        retract_stop_margin_n=0.10,
        retract_stop_confirm_s=0.005,
        retract_stop_fdot_n_s=15.0,
    )
    target = 2.0
    _arm(guard, target)
    # ~4 N/s — below fdot gate; stay above low-side.
    for i in range(40):
        fz = 2.8 - 0.02 * i  # ends ~2.0
        assert not _step(guard, fz, desired_force_n=target)
    assert guard.predictive_stop_count == 0
    assert guard.armed


def test_stale_sensor_fails_open_and_keeps_active_escape_available() -> None:
    guard = _guard()
    _trigger_hold(guard, 2.0)
    assert guard.hold

    hold = _step(
        guard,
        1.0,
        sensor_age_s=guard.cfg.max_sensor_age_s + 0.001,
    )
    assert hold is False
    assert guard.valid is False
    assert guard.hold is False
    assert guard.armed is False
    assert math.isnan(guard.fast_force_n)

    # Recovery primes a fresh episode; pre-dropout high-force history cannot
    # manufacture an arm/stop transition.
    assert not _step(guard, 1.0)
    assert guard.fast_force_n == pytest.approx(1.0)
    assert not guard.armed

    ff = ProactiveForceIntegrator(
        ProactiveFfConfig(
            gain=0.10,
            retract_gain=0.10,
            leak_s=0.3,
        )
    )
    ff.v_r = -0.01
    result = ff.update(
        -1.0,
        in_contact=True,
        dt_eff=DT,
        instability_index=5.0,
        v_force_z=-0.02,
        v_z_cap=0.10,
        desired_force_n=2.0,
        retract_fast_hold=hold,
    )
    assert result < 0.0
    assert ff.last_fast_retract_clear is False
    assert ff.last_reference_accel_m_s2 < 0.0


def test_hold_uses_hysteresis_before_rearming_active_retract() -> None:
    guard = _guard()
    target = 2.0
    _trigger_hold(guard, target)
    assert guard.hold

    stop_level = target - max(
        guard.cfg.stop_margin_n,
        guard.cfg.stop_margin_fraction * target,
    )
    rearm_level = target + max(
        guard.cfg.rearm_margin_n,
        guard.cfg.rearm_margin_fraction * target,
    )
    between_levels = 0.5 * (stop_level + rearm_level)

    # Staying between the stop and rearm levels for longer than min_hold_s
    # must not chatter the guard back open.
    for _ in range(20):
        assert _step(
            guard,
            between_levels,
            desired_force_n=target,
        )
    assert guard.rearm_count == 0

    # A sustained force rise beyond the separate rearm threshold restores
    # active escape after the confirmation interval.
    released = False
    for _ in range(30):
        if not _step(
            guard,
            rearm_level + 0.5,
            desired_force_n=target,
        ):
            released = True
            break

    assert released
    assert guard.hold is False
    assert guard.armed is True
    assert guard.rearm_count == 1
