"""Nominal law equivalence and command/observation transaction regressions."""
from copy import deepcopy
import math

import numpy as np
import pytest

from peirastic.realman8dof.force.legacy import LegacyForceLaw
from peirastic.realman8dof.force.config import apply_force_payload, load_force_raw
from peirastic.realman8dof.force.torque_tilt import LegacyForceWithTilt, TorqueTilt, TorqueTiltConfig
from peirastic.scan_path import force_profile, TILT_PROFILE, SCAN_FORCE_AXES
from rm75_control.control.admittance_common.controller import AdmittanceConfig, AdmittanceController

# Exactly the force-mode compiler's reload/overlay path: force.yaml is the
# force controller source, with scan_path's ICRA torque profile applied last.
# /media/camp/EXT_DRIVE/ICRA_YM/script/scan_robot.py:316-325 supplies
# these force defaults/overrides before hfpc(law="tff", force=4.0).
SCAN_PAYLOAD = dict(force_profile('icra'), force_axes=SCAN_FORCE_AXES,
                    control_frame='tool', desired_z=4.,
                    max_vz_tool_m_s=.010, v_seek_free_m_s=.010)
RAW = apply_force_payload(load_force_raw(), SCAN_PAYLOAD)


def make_law():
    controller = AdmittanceController(0.005, AdmittanceConfig.from_dict(deepcopy(RAW)))
    return LegacyForceWithTilt(LegacyForceLaw(controller), TorqueTilt(TorqueTiltConfig.from_dict(RAW)))


def sample(i, *, fz=None, tau=None):
    phase = i % 2000
    forces = (0., 1.2, 3.8, 4.04, 6.5, 2.5, 0., 4.)
    torques = (0., .1, .02, -.12, -.015, .15, 0., -.1)
    fz = forces[phase // 250] if fz is None else fz
    tau = torques[phase // 250] if tau is None else tau
    pose = np.array([.12 + i * 1e-7, .02, .15 + .0002 * np.sin(i * .03),
                     0., .3 * np.sin(i * .002), 0.])
    return dict(dt_s=.005, dt_actual=.005, pose=pose,
                f_ext=np.array([0., 0., fz, 0., tau, 0.]),
                f_ext_raw=np.array([0., 0., fz + .02 * np.sin(i), 0., tau, 0.]),
                f_des=np.array([0., 0., 4., 0., 0., 0.]),
                path_twist=np.array([.008 * np.cos(i * .001), 0., 0., 0., 0., 0.]),
                sensor_age_s=.003, feedback_age_s=.003,
                v_tcp_z_actual=.0005 * np.cos(i * .005), slack_norm=0.)


def assert_state_equal(old, new):
    for name in ('last_v_cmd', 'v_force_cmd_z', '_v_zoh_z', 'v_r_z',
                 '_force_point_base', 'x_adm_z', 'x_d_z', 'x_tilde_z',
                 '_u_force_slewed', '_u_force_slew_dot', 'ke_est', '_hp_zi'):
        np.testing.assert_allclose(getattr(old.controller, name), getattr(new.controller, name),
                                   atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(old.tilt.theta_tilt, new.tilt.theta_tilt, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(old.controller._tdpa.e_obs_j, new.controller._tdpa.e_obs_j,
                               atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(old.controller._bidirectional_flow.xp,
                               new.controller._bidirectional_flow.xp, atol=1e-10, rtol=1e-10)


def test_100000_ticks_effective_icra_nominal_equivalence():
    """Every tick compares the unchanged old update and prepare/commit paths."""
    old, new = make_law(), make_law()
    for name, expected in TILT_PROFILE.items():
        assert getattr(old.tilt.cfg, name) == expected
    assert RAW['force']['desired_z_n'] == 4.
    assert old.controller.cfg.max_vz_tool_m_s == .010
    assert old.controller.cfg.force_barrier.v_seek_free_m_s == .010
    np.testing.assert_array_equal(old.controller.cfg.force_axes, SCAN_FORCE_AXES)
    maximum_error = 0.
    for i in range(100000):
        kwargs = sample(i)
        if i % 10000 == 0:
            for law in (old, new):
                law.reset(pose=kwargs['pose'], f_ext=kwargs['f_ext'])
        expected = old.update(**kwargs)
        actual = new.prepare(measurement_id=i, **kwargs)
        maximum_error = max(maximum_error, float(np.max(np.abs(expected.v_force - actual.v_force))))
        if not np.allclose(expected.v_force, actual.v_force, atol=1e-10, rtol=1e-10):
            pytest.fail(f'nominal divergence at tick {i}: {expected.v_force} != {actual.v_force}')
        assert expected.v_force_z == actual.v_force_z
        assert expected.telemetry['theta_tilt'] == actual.telemetry['theta_tilt']
        new.commit_applied(actual.v_force)
        if i % 1000 == 0:
            assert_state_equal(old, new)
    assert_state_equal(old, new)
    assert maximum_error <= 1e-10
    print(dict(ticks=100000, max_absolute_twist_error=maximum_error,
               force_source=RAW['_path'], torque_profile=TILT_PROFILE,
               force_mass=old.controller.cfg.admittance_mass_z,
               force_damping=old.controller.cfg.admittance_damping_z,
               ke_initial=old.controller.cfg.adaptive_ke.ke_initial,
               scan_payload=SCAN_PAYLOAD))


def test_abort_preserves_measurements_and_prevents_windup():
    law = make_law()
    c = law.controller
    initial = law.z_law._transaction._capture()
    for i in range(500):
        out = law.prepare(measurement_id=i, **sample(i, fz=2., tau=.12))
        assert out.v_force_z == out.telemetry['v_cmd'][2]
        law.abort()
    assert c.contact_present
    assert c._contact_time_s > 2.
    assert c.physical_contact_state == 'contact'
    assert c._force_barrier._f_prev is not None
    assert not np.array_equal(c._hp_zi, np.zeros_like(c._hp_zi))
    assert c._bidirectional_flow.xa != initial[2].get('xa', 0.)
    for name, value in initial[0].items():
        if name in ('_force_point_base', '_force_point_inited', 'force_point_z', 'last_pose_d_combined'):
            continue  # The consumed contact edge retains its observed origin.
        np.testing.assert_array_equal(getattr(c, name), value)
    assert c._proactive_ff.v_r == 0.
    assert c._bidirectional_flow.xp == initial[2]['xp']
    assert law.tilt.theta_tilt == law.tilt.omega_y == 0.
    assert law.tilt.engaged and law.tilt.cop_valid


def test_modified_acceptance_rebases_force_tilt_and_full_command():
    law = make_law()
    for i in range(300):
        out = law.prepare(measurement_id=i, **sample(i, fz=2., tau=.12))
        law.commit_applied(out.v_force)
    c = law.controller
    before_x, before_theta = c.x_adm_z, law.tilt.theta_tilt
    out = law.prepare(measurement_id=300, **sample(300, fz=2., tau=.12))
    final = out.v_force.copy()
    final[2], final[4] = -.001, .02
    full = np.array([.003, .004, -.001, .005, .02, .006])
    law.commit_applied(final, final_full_twist=full)
    np.testing.assert_array_equal(c.last_v_cmd, full)
    assert c._v_zoh_z == c.v_force_cmd_z == -.001
    assert c._u_force_slewed == -.001
    assert c._proactive_ff.v_r == c.v_r_z == 0.
    assert c.x_adm_z == pytest.approx(before_x - .001 * .005, abs=1e-14)
    assert law.tilt.theta_tilt == pytest.approx(before_theta + .02 * .005, abs=1e-14)
    assert law.tilt._w == law.tilt.omega_y == .02
    assert c._safety_shield._u_prev == -.001


def test_duplicate_stale_pending_and_reset_do_not_reconsume():
    law = make_law()
    out = law.prepare(measurement_id=10, **sample(0, fz=3.))
    filtered = law.controller._hp_zi.copy()
    with pytest.raises(RuntimeError):
        law.prepare(measurement_id=11, **sample(1))
    law.abort()
    for seq in (10, 9):
        with pytest.raises(ValueError, match='duplicate or stale'):
            law.prepare(measurement_id=seq, **sample(1))
    np.testing.assert_array_equal(filtered, law.controller._hp_zi)
    with pytest.raises(RuntimeError):
        law.commit_applied(out.v_force)
    out = law.prepare(measurement_id=11, **sample(1))
    law.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    with pytest.raises(RuntimeError):
        law.commit_applied(out.v_force)
    with pytest.raises(ValueError):
        law.prepare(measurement_id=11, **sample(1))


def test_rejected_contact_edge_and_measured_pose_do_not_become_command_angle():
    tilt = TorqueTilt()
    tilt.theta_tilt = .7
    pose = np.array([0., 0., 0., .1, .4, .2])
    args = dict(dt_s=.005, contact=True, pose=pose)
    tilt.prepare(np.array([0., 0., 4., 0., .1, 0.]), np.array([0., 0., 4., 0., 0., 0.]),
                 measurement_id=1, **args)
    tilt.abort()
    assert tilt.theta_tilt == 0.
    np.testing.assert_array_equal(tilt.measured_pose_euler, pose[3:])
    pose[4] = 9.
    assert tilt.measured_pose_euler[1] == .4


def test_returned_proposal_cannot_mutate_internal_candidate():
    law = make_law()
    out = law.prepare(measurement_id=1, **sample(1, fz=4.))
    expected = out.v_force.copy()
    out.telemetry['v_cmd'][:] = 999.
    out.v_force[:] = 999.
    law.commit_applied(expected)
    assert np.max(np.abs(law.controller.last_v_cmd)) < 1.


def test_abort_after_motion_keeps_last_accepted_command_histories():
    law = make_law()
    for i in range(300):
        out = law.prepare(measurement_id=i, **sample(i, fz=3., tau=.12))
        law.commit_applied(out.v_force)
    state = law.z_law._transaction._capture()
    angle, omega = law.tilt.theta_tilt, law.tilt.omega_y
    for i in range(300, 800):
        law.prepare(measurement_id=i, **sample(i, fz=2., tau=-.12))
        law.abort()
    for name, value in state[0].items():
        np.testing.assert_array_equal(getattr(law.controller, name), value)
    assert law.tilt.theta_tilt == angle
    assert law.tilt.omega_y == omega


def test_remapped_nominal_and_distinct_final_full_twist():
    old, new = make_law(), make_law()
    for i in range(60):
        kwargs = sample(i, fz=4., tau=.1)
        kwargs['slack_norm'] = .2
        expected = old.update(**kwargs)
        out = new.prepare(measurement_id=i, **kwargs)
        new.commit_applied(out.v_force)
        np.testing.assert_array_equal(expected.v_force, out.v_force)
        assert_state_equal(old, new)
    out = new.prepare(measurement_id=60, **kwargs)
    full = np.array([.01, .02, .03, .04, .05, .06])
    new.commit_applied(out.v_force, final_full_twist=full, accepted_normal_z=.001)
    np.testing.assert_array_equal(new.controller.last_v_cmd, full)
    assert new.controller.v_force_cmd_z == .001


def test_optional_active_modes_fail_before_consuming_measurement():
    law = make_law()
    law.controller.cfg.bidirectional_flow.mode = 'active'
    with pytest.raises(ValueError, match='diagnostic'):
        law.prepare(measurement_id=1, **sample(1))
    assert law.z_law._transaction.last_measurement_id == -1


def test_air_bias_filter_consumed_once_on_abort_and_modified_commit():
    new = make_law()
    for i in range(30):
        kwargs = sample(i, fz=.3, tau=0.)
        before = new.controller._tdpa.f_bias_n
        out = new.prepare(measurement_id=i, **kwargs)
        cfg = new.controller._tdpa.cfg
        expected = before
        if abs(out.v_force_z) <= cfg.v_bias_gate_m_s:
            expected += (1. - math.exp(-.005 / cfg.bias_lpf_s)) * (.3 - before)
        assert new.controller._tdpa.f_bias_n == expected
        if i % 2:
            new.commit_applied(np.zeros(6))
        else:
            new.abort()
        assert new.controller._tdpa.f_bias_n == expected
    assert new.controller._tdpa.f_bias_n > 0.
    assert new.controller._tdpa.e_obs_j == 0.


def test_modified_tilt_history_does_not_hide_an_accepted_angle_excursion():
    tilt = TorqueTilt()
    tilt._was_contact = True
    tilt.theta_tilt = tilt.cfg.theta_max_rad
    tilt.prepare(np.array([0., 0., 4., 0., .1, 0.]), np.array([0., 0., 4., 0., 0., 0.]),
                 measurement_id=1, dt_s=.005, contact=True)
    tilt.commit_applied(.1)
    assert tilt.theta_tilt == tilt.cfg.theta_max_rad + .0005


def test_rejected_lateral_candidate_does_not_arm_chase_softening():
    law = make_law()
    kwargs = sample(0, fz=3.)
    kwargs['path_twist'] = np.array([.03, 0., 0., 0., 0., 0.])
    law.prepare(measurement_id=0, **kwargs)
    law.abort()
    assert law.controller._lat_soften_hold_s == 0.
    out = law.prepare(measurement_id=1, **kwargs)
    law.commit_applied(out.v_force, final_full_twist=np.zeros(6))
    assert law.controller._lat_soften_hold_s == 0.


def test_rejected_contact_episode_retains_reseed_without_command_integration():
    law = make_law()
    for i in range(300):
        out = law.prepare(measurement_id=i, **sample(i, fz=2.))
        law.commit_applied(out.v_force)
    for i in range(300, 800):
        kwargs = sample(i, fz=0.)
        kwargs['pose'][2] += .03
        law.prepare(measurement_id=i, **kwargs)
        law.abort()
    assert law.controller.x_adm_z == law.controller._v_zoh_z == 0.
    kwargs = sample(800, fz=2.)
    kwargs['pose'][2] += .03
    law.prepare(measurement_id=800, **kwargs)
    assert law.controller.contact_episode_rearm_event
    origin = law.controller._force_point_base.copy()
    law.abort()
    law.prepare(measurement_id=801, **kwargs)
    assert not law.controller.contact_episode_rearm_event
    np.testing.assert_array_equal(law.controller._force_point_base, origin)
    proposed = law.z_law._transaction.pending
    np.testing.assert_allclose(proposed.proposed[0]['_force_point_base'],
                               origin + proposed.normal * proposed.nominal[2] * proposed.dt,
                               atol=1e-14, rtol=0.)
    law.abort()
