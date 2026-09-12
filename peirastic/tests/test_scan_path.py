"""Geometry, compact API, force gating and task-local response; no hardware."""
import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from peirastic.scan_path import (
    AMPLITUDE_LOAD_MAX_M,
    ForearmReference,
    PEAK_RANGE_M,
    make_spec,
    force_profile,
    SCAN_FORCE_AXES,
    TILT_PROFILE,
    _offset,
)
from peirastic.realman8dof.modes.contact_reference import ContactGatedReference
from peirastic.realman8dof.force.torque_tilt import TorqueTilt, TorqueTiltConfig

D = np.array([0.2, 0.1, 0.3, np.pi, 0.0, np.deg2rad(179)])
P = np.array([0.4, 0.1, 0.31, np.pi-0.02, 0.01, np.deg2rad(-179)])


@pytest.mark.parametrize("shape", list("LCS"))
@pytest.mark.parametrize("direction", ["DtP", "PtD"])
def test_shape_endpoints_side_speed_and_smooth_ramps(shape, direction):
    spec = make_spec(D, P, shape, direction, 15)
    ref = ForearmReference(spec)
    assert len(json.dumps(spec).encode()) < 10_000
    first, last = (D, P) if direction == "DtP" else (P, D)
    for actual, wanted in [(ref.sample(0).pose_d, first), (ref.sample(ref.duration_s).pose_d, last)]:
        np.testing.assert_allclose(actual[:3], wanted[:3], atol=1e-12)
        assert (Rotation.from_euler("xyz", actual[3:]).inv() * Rotation.from_euler("xyz", wanted[3:])).magnitude() < 1e-12
    assert np.linalg.norm(ref.sample(0).vel_ff) == 0
    assert np.linalg.norm(ref.sample(ref.duration_s).vel_ff) == 0
    samples = [ref.sample(t) for t in np.linspace(.5, ref.duration_s-.5, 100)]
    speeds = np.array([np.linalg.norm(s.vel_ff[:3]) for s in samples])
    np.testing.assert_allclose(speeds, .02, rtol=0.002)
    u = np.linspace(0, 1, 401)
    poses = np.array([ref.at(t)[0] for t in u])
    lateral = (poses[:, :3] - D[:3] - u[:, None]*(P[:3]-D[:3])) @ ref.lateral
    if shape == "L":
        np.testing.assert_allclose(lateral, 0, atol=1e-14)
    else:
        assert .008 - 1e-6 <= np.max(lateral) <= .010
        assert np.all(lateral[:200] >= -1e-14)
        if shape == "C":
            assert np.all(lateral >= -1e-14)
        else:
            assert -.010 <= np.min(lateral) <= -.008 + 1e-6
            assert np.all(lateral[201:] <= 1e-14)
    for end in (0, 1):
        assert abs(float(ref.at(end)[1] @ ref.lateral)) < 1e-12
    angles = (Rotation.from_euler("xyz", poses[0, 3:]).inv() * Rotation.from_euler("xyz", poses[:, 3:])).magnitude()
    assert angles.max() < np.deg2rad(5)  # Never rotate 358 degrees across Euler wrap.
    # Feedforward agrees with an actual finite difference of the commanded path.
    t, h = ref.duration_s * .37, 1e-5
    fd = (ref.sample(t+h).pose_d[:3]-ref.sample(t-h).pose_d[:3])/(2*h)
    np.testing.assert_allclose(ref.sample(t).vel_ff[:3], fd, atol=1e-7)


def test_noise_is_repeatable_independent_and_always_on_the_fixed_side():
    for shape in "CS":
        a = make_spec(D, P, shape, "DtP", 100)
        assert a == make_spec(D, P, shape, "DtP", 100)
        b = make_spec(D, P, shape, "PtD", 101)
        assert a["lateral"] == b["lateral"]
        assert a["noise_coefficients"] != b["noise_coefficients"]
        for seed in range(25):
            ref = ForearmReference(make_spec(D, P, shape, "DtP", seed))
            for u in (.1, .25, .75, .9):
                displacement = ref.at(u)[0][:3] - (D[:3]+u*(P[:3]-D[:3]))
                signed = float(displacement @ ref.lateral)
                assert np.sign(signed) == (1 if shape == "C" or u < .5 else -1)
                assert abs(signed) <= .010


@pytest.mark.parametrize("shape", ["C", "S"])
def test_actual_noisy_peaks_stay_in_range_and_metadata_matches(shape):
    maxima = []
    u = np.linspace(0, 1, 20001)
    for seed in range(100):
        spec = make_spec(D, P, shape, "DtP", seed)
        offset, _ = _offset(u, shape, spec["noise_coefficients"], spec["amplitude_m"])
        peaks = np.array([offset.max(), -offset.min()])
        np.testing.assert_allclose(peaks, spec["peak_offsets_m"], atol=2e-9)
        lobes = peaks[:1] if shape == "C" else peaks
        assert np.all(lobes >= .008 - 2e-9)
        assert np.all(lobes <= .010)
        maxima.append(max(lobes))
        # Scaling must not introduce a derivative jump at the S crossing.
        ref = ForearmReference(spec)
        np.testing.assert_allclose(ref.at(.5-1e-8)[1], ref.at(.5+1e-8)[1], atol=1e-7)
    assert min(maxima) < .009
    assert max(maxima) > .009


def test_planning_peaks_are_8_to_10_mm_and_old_20_mm_specs_still_load():
    assert PEAK_RANGE_M == (0.008, 0.010)
    assert AMPLITUDE_LOAD_MAX_M == 0.020
    spec = make_spec(D, P, "C", "DtP", 3)
    assert spec["peak_range_m"] == [0.008, 0.010]
    spec["amplitude_m"] = 0.020
    ForearmReference(spec)


def test_legacy_fixed_amplitude_specs_remain_readable():
    spec = make_spec(D, P, "S", "PtD", 20)
    for field in ("peak_offsets_m", "peak_range_m"):
        spec.pop(field)
    spec["amplitude_m"] = .02
    u = np.linspace(0, 1, 4097)
    _, derivative = _offset(u, "S", spec["noise_coefficients"], .02)
    tangent = (P[:3]-D[:3]) + derivative[:, None]*np.asarray(spec["lateral"])
    rate = np.linalg.norm(tangent, axis=1)
    arc = np.r_[0., np.cumsum((rate[:-1]+rate[1:])*.5/4096)]
    spec["arc_m"] = arc[::32].tolist()
    ref = ForearmReference(spec)
    np.testing.assert_allclose(ref.sample(ref.duration_s).pose_d, D, atol=1e-12)


@pytest.mark.parametrize("amplitude", [float("nan"), float("inf"), 0., -.01, .03])
def test_invalid_amplitude_rejected(amplitude):
    spec = make_spec(D, P, "C", "DtP", 0)
    spec["amplitude_m"] = amplitude
    with pytest.raises(ValueError, match="invalid ICRA"):
        ForearmReference(spec)


def test_gate_needs_air_then_continuous_fresh_4n_without_changing_physical_contact():
    ref = ForearmReference(make_spec(D, P, "L", "DtP", 0))
    gate = ContactGatedReference(ref, lambda: True, start_force_n=4, start_force_s=.1)
    for t in np.arange(0, .3, .005):
        gate.observe_force(4.3, t)
        gate.sample(t)
    assert not gate.started  # No observed air baseline.
    gate.observe_force(0, .3)
    for t in np.arange(.305, .36, .005):
        gate.observe_force(4.3, t)
        gate.sample(t)
    gate.observe_force(3.9, .36)
    for t in np.arange(.365, .46, .005):
        gate.observe_force(4.3, t)
        gate.sample(t)
    assert not gate.started
    gate.observe_force(4.3, .47)
    gate.sample(.47)
    assert gate.started
    gate.sample(.57)
    assert gate.elapsed_s == pytest.approx(.1)
    gate.set_origin(D)
    gate.observe_force(0, 1)
    gate.observe_force(4.3, 1.01)
    gate.observe_force(4.3, 1.25)  # Gap does not count as sustained force.
    gate.sample(1.25)
    assert not gate.started


def test_task_profile_keeps_deadband_and_respects_acceleration_and_symmetry():
    assert force_profile("baseline") == {}
    cfg = replace(TorqueTiltConfig(), **TILT_PROFILE)
    assert TorqueTiltConfig().mass == .065  # No global baseline mutation.
    plus, minus = TorqueTilt(cfg), TorqueTilt(cfg)
    previous = 0
    for torque in np.repeat([.015, .025, .05, .10, -.10, 0], 200):
        a = plus.update(np.array([0, 0, 4, 0, torque, 0.]), np.array([0, 0, 4, 0, 0, 0.]), dt_s=.005, contact=True)
        b = minus.update(np.array([0, 0, 4, 0, -torque, 0.]), np.array([0, 0, 4, 0, 0, 0.]), dt_s=.005, contact=True)
        assert a == pytest.approx(-b)
        assert abs(a) <= .28
        assert abs(a-previous) <= 3*.005 + 1e-12
        previous = a


def test_hfpc_compiles_compact_path_with_open_my_and_independent_4n_gate(monkeypatch):
    from peirastic.tests.test_api_facade import _ctx
    from peirastic.api import PeirasticArm
    from peirastic.realman8dof.session import compile_request
    _, ctx = _ctx()
    arm = PeirasticArm(attach=False)
    arm.set_force_raw_override(force_profile())
    assert arm.hfpc(reference="icra_path", path_spec=make_spec(D, P, "C", "DtP", 8),
                    force_axes=SCAN_FORCE_AXES, force=4, wait_for_contact=True,
                    scan_contact_n=4, scan_contact_s=.1, block=0) == 0
    result = compile_request(ctx, arm.last_request)
    outer = result.outer
    assert outer.selection.tolist() == [1, 1, 0, 1, 0, 1]
    assert outer.force_law.tilt.cfg.mass == .051
    assert outer.contact_gate.start_force_n == 4
    assert isinstance(outer.contact_gate.reference, ForearmReference)
    assert len(json.dumps(arm.last_request.to_json()).encode()) < 16384
    clock = [0.0]
    monkeypatch.setattr("peirastic.realman8dof.modes.track.time.monotonic", lambda: clock[0])
    controller = outer.controller
    outer.set_origin(D)
    for i in range(100):
        clock[0] = i*.005
        fz = 0 if i < 20 else 4.3
        outer.sample(clock[0], D, np.array([0, 0, fz, 0, .05, 0.]),
                     dt_actual=.005, feedback_fresh_tick=True, sensor_age_s=0,
                     feedback_age_s=0, v_tcp_z_actual=0, feedback_velocity_valid=True)
        if i < 40:
            assert not outer.contact_gate.started
    assert outer.contact_gate.started and outer.contact_gate.elapsed_s > .2
    assert outer.controller is controller  # No force-state restart on contact.
    assert abs(outer.last_omega_y) > .01


def test_controller_seek_guard_works_without_external_supervisor(monkeypatch):
    from peirastic.scan_path import outward
    now = [10.0]
    monkeypatch.setattr("peirastic.realman8dof.modes.contact_reference.time.monotonic", lambda: now[0])
    ref = ForearmReference(make_spec(D, P, "L", "DtP", 0))
    gate = ContactGatedReference(ref, lambda: True, start_force_n=4)
    gate.guard_approach(D)
    now[0] += 25.01
    with pytest.raises(RuntimeError, match="timed out"):
        gate.guard_approach(D)
    gate.set_origin(D)
    too_deep = D.copy()
    too_deep[:3] -= .011 * outward(D)
    with pytest.raises(RuntimeError, match="10 mm"):
        gate.guard_approach(too_deep)
