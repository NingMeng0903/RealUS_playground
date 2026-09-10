from dataclasses import replace

import numpy as np
import pytest

from peirastic.contact_qp.energy import (
    EnergyLedger, ExposureSegment, PortBounds, PortInterval, branch_exposure, mixed_device_branches,
)


def bounds(**kwargs):
    return PortBounds(calibration_version="synthetic", verified=True, **kwargs)


def interval(start, end, power, **kwargs):
    return PortInterval(start, end, [-power, 0, 0, 0, 0, 0], [-power, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0],
                        calibration_version="synthetic", time_aligned=True, **kwargs)


def test_untrusted_flags_clock_and_overflow_cannot_mint_energy():
    with pytest.raises(ValueError, match="boolean"):
        PortBounds(verified="false")
    with pytest.raises(ValueError, match="boolean"):
        replace(interval(1, 1.01, 1), time_aligned="false")
    ledger = EnergyLedger(.5, 1, .1, bounds())
    ledger.reserve(1, [ExposureSegment(1, 1.01, 1)])
    with pytest.raises(ValueError, match="clock"):
        ledger.settle(interval(100, 100.01, -10), now_s=float("nan"))
    explosive = replace(interval(1, 1.01, 1), wrench_start=np.full(6, 1e308),
                        velocity_start=np.full(6, 1e308))
    assert not ledger.settle(explosive, now_s=1.01)
    assert ledger.balance_j == .5 and ledger.liability(1) > 0
    assert not ledger.certification_valid


def test_one_balance_protects_stop_reserve_and_never_spends_future_recovery():
    ledger = EnergyLedger(.2, .2, .1, bounds())
    pieces = [ExposureSegment(1, 1.01, 8., "new_only"), ExposureSegment(1.01, 1.02, -20., "new_only")]
    assert ledger.reserve(1, pieces)
    assert ledger.balance_j == .2 and ledger.available_j == pytest.approx(.02)
    assert not ledger.reserve(2, [ExposureSegment(1, 1.01, 3)])
    assert ledger.reserve(3, [ExposureSegment(1, 1.01, 3)], stopping=True)
    assert ledger.balance_j == .2  # reservations did not become a second tank
    with pytest.raises(ValueError, match="future"):
        ledger.settle(interval(1.01, 1.02, -10.), now_s=1.)


def test_rejection_does_not_refund_old_held_or_mixed_exposure():
    ledger = EnergyLedger(1, 1, .1, bounds())
    assert ledger.reserve(1, [ExposureSegment(1., 1.02, 2., "old_or_mixed")])
    before = ledger.snapshot()
    assert not ledger.mark_no_send(1)
    assert ledger.reserved_j == before["reserved_j"]
    ledger.settle(interval(1, 1.01, 1.), now_s=1.01)
    assert ledger.balance_j == pytest.approx(.99)
    assert ledger.liability(1) == pytest.approx(.02)
    assert ledger.mark_no_send(1, old_exposure_proven_absent=True)
    assert ledger.balance_j == pytest.approx(.99) and ledger.reserved_j == 0


def test_unexposed_release_is_not_recharge_and_ids_cannot_be_reused():
    ledger = EnergyLedger(1, 1, .1, bounds())
    ledger.reserve(1, [ExposureSegment(1, 1.02, 2, "new_only")])
    assert ledger.mark_no_send(1)
    assert ledger.balance_j == 1 and ledger.reserved_j == 0
    with pytest.raises(ValueError):
        ledger.reserve(1, [ExposureSegment(1, 1.02, 2)])


def test_overlap_reservations_settle_physical_work_once_and_release_each_earmark():
    ledger = EnergyLedger(1, 1, .1, bounds())
    ledger.reserve(1, [ExposureSegment(1, 1.02, 2)])
    ledger.reserve(2, [ExposureSegment(1, 1.02, 3)])
    ledger.settle(interval(1, 1.01, 1.5), now_s=1.02)
    assert ledger.balance_j == pytest.approx(.985)
    assert ledger.reserved_j == pytest.approx(.05)
    with pytest.raises(ValueError, match="overlaps"):
        ledger.settle(interval(1, 1.01, 1.5), now_s=1.02)
    with pytest.raises(ValueError, match="overlaps"):
        ledger.settle(interval(1.005, 1.015, 1.5), now_s=1.02)
    ledger.settle(interval(1.01, 1.02, 1.5), now_s=1.02)
    assert ledger.balance_j == pytest.approx(.97) and ledger.reserved_j == 0
    assert ledger._settled == [(1., 1.02)]


def test_missing_measurement_retains_liability_and_wrong_port_loses_certification():
    ledger = EnergyLedger(1, 1, .1, bounds())
    ledger.reserve(1, [ExposureSegment(1, 1.02, 2)])
    missing = interval(1, 1.01, 1., valid=False)
    assert not ledger.settle(missing, now_s=1.01)
    assert ledger.balance_j == 1 and ledger.reserved_j == pytest.approx(.04)
    wrong = replace(interval(1, 1.01, 1.), calibration_version="other")
    assert not ledger.settle(wrong, now_s=1.01)
    assert not ledger.certification_valid
    assert not ledger.reserve(2, [ExposureSegment(1, 1.01, 0)])


def test_full_six_dimensional_corner_and_error_bounds_cover_mixed_execution():
    rng = np.random.default_rng(2)
    b = bounds(wrench_error=np.full(6, .1), velocity_error=np.full(6, .02))
    vectors = rng.normal(size=(5, 6))
    w = rng.normal(size=6)
    branches = mixed_device_branches(*vectors, stop_arm=vectors[4], stop_rail=np.zeros(6))
    segment = branch_exposure(1, .02, w, branches, b)
    for _ in range(500):
        v = branches[rng.integers(0, len(branches))]+rng.uniform(-.02, .02, 6)
        actual_w = w+rng.uniform(-.1, .1, 6)
        assert -actual_w @ v <= segment.power_upper_w+1e-12


def test_asynchronous_stop_destroys_arm_rail_cancellation():
    x = np.array([1., 0, 0, 0, 0, 0])
    branches = mixed_device_branches(2*x, -2*x, x, -x, x*0)
    segment = branch_exposure(1, .01, -x, branches, bounds())
    assert segment.power_upper_w == 2.
    with pytest.raises(ValueError):
        mixed_device_branches(x, x, x, x, x*0, stop_arm=x)


def test_subpicosecond_repeated_work_and_string_proof_are_rejected():
    ledger = EnergyLedger(.1, 2, .01, bounds())
    with pytest.raises(ValueError):
        ledger.reserve(1, [ExposureSegment(1, 1.01, 1)], stopping="false")
    ledger.reserve(1, [ExposureSegment(1, 1.01, 1)])
    with pytest.raises(ValueError):
        ledger.mark_no_send(1, old_exposure_proven_absent="false")
    sample = interval(1, 1+5e-13, -1e12)
    ledger.settle(sample, now_s=1.01)
    value = ledger.balance_j
    with pytest.raises(ValueError, match="overlaps"):
        ledger.settle(sample, now_s=1.01)
    assert ledger.balance_j == value


def test_energy_overdraw_is_exposed_without_a_lower_clip_or_automatic_refund():
    ledger = EnergyLedger(.1, .1, .01, bounds())
    ledger.reserve(1, [ExposureSegment(1, 1.02, 1)])
    ledger.settle(interval(1, 1.01, 20), now_s=1.01)
    assert ledger.balance_j == pytest.approx(-.1)
    assert not ledger.certification_valid
    assert ledger.liability(1) == pytest.approx(.01)
    assert any(e.get("reason") == "execution_exceeded_reserved_power_envelope" for e in ledger.events)


def test_recovery_is_credited_only_after_time_aligned_measurement():
    ledger = EnergyLedger(.2, 1, .1, bounds())
    ledger.reserve(1, [ExposureSegment(1, 1.02, 2)])
    ledger.settle(interval(1, 1.01, -2), now_s=1.01)
    assert ledger.balance_j == pytest.approx(.22)
    assert ledger.reserved_j == pytest.approx(.02)
    monitor = EnergyLedger(1, 1, .1, PortBounds())
    assert not monitor.certification_valid and "monitoring" in monitor.certification_reason
