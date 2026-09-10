from dataclasses import replace

import numpy as np
import pytest

from peirastic.contact_qp.features import FeatureConfig, FeatureExtractor
from peirastic.contact_qp.history import MotionHistory, ResponseConsistency, ResponseConfig
from peirastic.contact_qp.plant import FiniteAreaPlant, PlantConfig
from peirastic.contact_qp.types import ContactObservation, ProbeGeometry


def test_finite_area_force_is_unilateral_and_torque_mirrors():
    left, right = FiniteAreaPlant("left_gap", 3), FiniteAreaPlant("right_gap", 3)
    right.k, right.d = left.k[::-1].copy(), left.d[::-1].copy()
    for p in (left, right):
        p.path = .03
        p._update_contact()
    np.testing.assert_allclose(left.pressures, right.pressures[::-1], atol=1e-12)
    assert left.control_wrench(noise=False)[2] == pytest.approx(right.control_wrench(noise=False)[2])
    assert left.control_wrench(noise=False)[4] == pytest.approx(-right.control_wrench(noise=False)[4])
    left.z = -.01
    left._update_contact()
    assert not left.pressures.any()


def test_full_physical_port_matches_spring_work_under_tcp_rotation():
    p = FiniteAreaPlant("healthy", 8, replace(PlantConfig(), friction=0.))
    p.theta, p.z = .08, .008
    p.velocity[[2, 4]] = [.001, .03]
    p._update_contact()
    rates = np.cos(p.theta)*(p.velocity[2]-p.x*p.velocity[4])
    spring_power = p.pressures @ rates
    spin_loss = .004*np.tanh(p.velocity[4]/.01)*p.velocity[4]
    assert -p.wrench_environment @ p.velocity == pytest.approx(spring_power+spin_loss)


def test_sent_command_is_not_immediately_measured_motion():
    p = FiniteAreaPlant("delayed_execution", 1)
    v = np.array([0, .02, 0, 0, .1, 0.])
    p.command(v)
    for _ in range(10):
        p.step()
    np.testing.assert_array_equal(p.velocity, np.zeros(6))
    for _ in range(20):
        p.step()
    assert 0 < p.path < .02*p.t
    assert p.velocity[4] > 0.


@pytest.mark.parametrize("scenario", ["left_gap", "curvature", "moving_surface"])
def test_curved_surface_energy_derivative_includes_scanning_reaction_and_external_motion(scenario):
    cfg = replace(PlantConfig(), damping_n_s_m=0., friction=0., rocking_friction_nm=0.)
    p = FiniteAreaPlant(scenario, 2, cfg)
    p.path, p.z, p.theta, p.t = .025, .008, .08, 1.8
    p.velocity[[1, 2, 4]] = [.02, .001, .03]
    p._update_contact()
    power = -p.wrench_environment @ p.velocity + p.surface_power_w
    q = np.array([p.world_x, p.path, p.z, p.theta, p.t])
    dq = np.array([np.sin(p.theta)*p.velocity[2], p.velocity[1],
                   np.cos(p.theta)*p.velocity[2], p.velocity[4], 1.])
    eps = 1e-6
    energies = []
    for sign in (-1, 1):
        p.world_x, p.path, p.z, p.theta, p.t = q+sign*eps*dq
        p._update_contact()
        energies.append(.5*np.sum(p.k*np.maximum(p.depth, 0)**2))
    assert (energies[1]-energies[0])/(2*eps) == pytest.approx(power, rel=2e-6, abs=2e-8)


def test_measured_pose_difference_matches_tool_twist_and_unsupported_motion_rejected():
    p = FiniteAreaPlant("healthy", 1)
    p.theta = .2
    p.held = p.velocity = np.array([.003, .02, .004, 0., .1, 0.])
    before = p.pose()
    p.step()
    expected = np.array([np.cos(.2)*.003+np.sin(.2)*.004, .02,
                         -np.sin(.2)*.003+np.cos(.2)*.004, 0., .1, 0.])
    np.testing.assert_allclose((p.pose()-before)/p.config.dt_s, expected, atol=1e-12)
    with pytest.raises(ValueError, match="reduced plant"):
        p.command([0, 0, 0, .1, 0, 0])


def test_acoustic_shadow_passes_through_actual_feature_chain_independent_of_contact():
    cfg = FeatureConfig(width=32, height=48, calibration_version="synthetic_v1")
    healthy, shadow = FiniteAreaPlant("healthy", 4), FiniteAreaPlant("shadow", 4)
    out = []
    for p in (healthy, shadow):
        assert p.window_coupling(cfg)[[0, 2]].min() == 1.
        obs, _ = FeatureExtractor(cfg).extract(p.render(0), frame_seq=1, source_id=p.scenario,
                    capture_time_s=2., received_time_s=2.01)
        out.append(obs)
    assert out[0].quality[0] > .3 and out[1].quality[0] < .25*out[0].quality[0]
    assert out[1].valid.all()
    assert shadow.window_acoustic_coupling(cfg)[0] == 0.
    assert healthy.window_acoustic_coupling(cfg)[[0, 2]].min() == 1.


def test_registered_actual_motion_rejects_gap_and_does_not_use_a_command_sign():
    h = MotionHistory(ProbeGeometry.synthetic())
    for i in range(21):
        h.add(1+i*.005, [0, 0, -.001, 0, 0, 0], i)
    du, _ = h.displacement(1.02, 1.08)
    np.testing.assert_allclose(du, -.00006)
    assert not h.add(1.11, np.zeros(6), 20)
    h.add(1.2, np.zeros(6), 22)
    assert h.displacement(1.05, 1.2) is None
    policy = ResponseConsistency()
    first = ContactObservation(1, "camera", 1.02, 1.03, [.5]*3, [True]*3, "reg", "win")
    second = replace(first, frame_seq=2, effective_time_s=1.08, received_time_s=1.09, quality=[.6]*3)
    policy.update(first, h, now_s=1.03)
    np.testing.assert_array_equal(policy.update(second, h, now_s=1.09), np.ones(2))
    # Unloading cannot be interpreted as a failed positive repair action.
    updates = policy.updates.copy()
    policy.update(second, h, now_s=1.1)
    np.testing.assert_array_equal(updates, policy.updates)
    changed = replace(second, frame_seq=3, effective_time_s=1.09, received_time_s=1.1, window_version="new")
    np.testing.assert_array_equal(policy.update(changed, h, now_s=1.1), np.ones(2))
    policy.update(changed, h, now_s=2.)
    assert policy.previous is None
    h.reconfigure(ProbeGeometry.synthetic(image_x_sign=-1))
    assert h.displacement(1.02, 1.08) is None
    assert not h.add(1.3, np.zeros(6), 20)


def test_explicit_invalid_measurement_breaks_even_a_short_interpolation_span():
    h = MotionHistory(ProbeGeometry.synthetic())
    v = [0, 0, .001, 0, 0, 0]
    assert h.add(1., v, 1)
    assert not h.add(1.005, v, 2, valid=False)
    assert h.add(1.010, v, 3)
    assert h.displacement(1., 1.01) is None
    assert not h.add(1.02, v, 2)


def response_observation(seq, effective, *, quality=.1, received=None, **changes):
    base = ContactObservation(seq, 'camera', effective, effective+.01 if received is None else received,
                              [quality]*3, [True]*3, 'reg', 'win')
    return replace(base, **changes)


def motion_history(velocity):
    history = MotionHistory(ProbeGeometry.synthetic())
    for i in range(101):
        t=1.+i*.005
        history.add(t, [0,0,velocity(t),0,0,0], i)
    return history


def test_static_shadow_accumulates_small_executed_motion_and_discounts_no_response():
    history=motion_history(lambda t: .00005)  # only 2 um per camera interval
    policy=ResponseConsistency()
    for i in range(7):
        t=1.+i*.04
        policy.update(response_observation(i,t),history,now_s=t+.01)
    assert np.all(policy.gamma < 1.)
    np.testing.assert_array_equal(policy.updates, [1,1])
    # The same frame cannot consume that action evidence twice.
    before=policy.gamma.copy()
    policy.update(response_observation(6,1.24),history,now_s=1.26)
    np.testing.assert_array_equal(policy.gamma,before)


@pytest.mark.parametrize('quality', [.1,.8])
def test_no_executed_motion_or_texture_change_alone_does_not_discount(quality):
    history=motion_history(lambda t: 0.)
    policy=ResponseConsistency()
    policy.update(response_observation(0,1.),history,now_s=1.01)
    for i in range(1,8):
        t=1.+i*.04
        policy.update(response_observation(i,t,quality=quality),history,now_s=t+.01)
    np.testing.assert_array_equal(policy.gamma,np.ones(2))
    np.testing.assert_array_equal(policy.updates,np.zeros(2))


def test_response_waits_minimum_window_and_ignores_motion_after_exposure():
    policy=ResponseConsistency();history=motion_history(lambda t: .001 if t>1.081 else 0.)
    policy.update(response_observation(0,1.),history,now_s=1.01)
    policy.update(response_observation(1,1.08,received=1.25),history,now_s=1.25)
    np.testing.assert_array_equal(policy.gamma,np.ones(2))
    fast=motion_history(lambda t: .001);policy=ResponseConsistency()
    policy.update(response_observation(0,1.),fast,now_s=1.01)
    policy.update(response_observation(1,1.04),fast,now_s=1.05)
    np.testing.assert_array_equal(policy.gamma,np.ones(2))
    assert np.all(policy.update(response_observation(2,1.08),fast,now_s=1.09)<1.)


def test_negative_motion_and_within_frame_unloading_reset_positive_evidence():
    for velocity in (lambda t: -.001, lambda t: -.001 if 1.03<t<1.07 else .001):
        history=motion_history(velocity);policy=ResponseConsistency()
        policy.update(response_observation(0,1.),history,now_s=1.01)
        policy.update(response_observation(1,1.1),history,now_s=1.11)
        np.testing.assert_array_equal(policy.gamma,np.ones(2))
        assert policy.reason=='unloading_resets_evidence'


def test_missing_images_versions_and_motion_gaps_cannot_bridge_response_evidence():
    history=motion_history(lambda t: .001);policy=ResponseConsistency()
    policy.update(response_observation(0,1.),history,now_s=1.01)
    policy.update(None,history,now_s=1.07)
    policy.update(response_observation(1,1.08),history,now_s=1.09)
    np.testing.assert_array_equal(policy.gamma,np.ones(2))
    policy.update(response_observation(2,1.16),history,now_s=1.17)
    assert np.all(policy.gamma<1.)
    policy.update(response_observation(3,1.24,registration_version='new'),history,now_s=1.25)
    np.testing.assert_array_equal(policy.gamma,np.ones(2))
    history=MotionHistory(ProbeGeometry.synthetic())
    for i in range(21):history.add(1.+i*.005,[0,0,.001,0,0,0],i,valid=i!=8)
    policy=ResponseConsistency();policy.update(response_observation(0,1.),history,now_s=1.01)
    policy.update(response_observation(1,1.1),history,now_s=1.11)
    np.testing.assert_array_equal(policy.gamma,np.ones(2))


def test_registered_positive_improvement_recovers_only_after_action():
    history=motion_history(lambda t: .001);policy=ResponseConsistency()
    policy.update(response_observation(0,1.),history,now_s=1.01)
    policy.update(response_observation(1,1.08),history,now_s=1.09)
    low=policy.gamma.copy()
    policy.update(response_observation(2,1.16,quality=.2),history,now_s=1.17)
    assert np.all(policy.gamma>low)


def test_motion_evidence_integrates_unloading_across_zero_crossing():
    history=MotionHistory(ProbeGeometry.synthetic())
    history.add(1.,[0,0,-.001,0,0,0],0)
    history.add(1.01,[0,0,.001,0,0,0],1)
    local,planar,unloading=history.motion_evidence(1.,1.01)
    np.testing.assert_allclose(local,0.,atol=1e-15)
    np.testing.assert_allclose(unloading,.0000025)
    assert planar==0.
