"""Offline tests for the real mechanical transaction and pad allocation."""
from types import SimpleNamespace
import inspect
import time

import numpy as np
import pytest

from peirastic.core.modes import Mode, ModeRequest
from peirastic.realman8dof.session import compile_request
from peirastic.realman8dof.modes.pad_contact import PadContactOuter
from peirastic.realman8dof.modes.pad_contact_config import load_pad_contact_config
from peirastic.realman8dof.modes.pad_visual import pad_hybrid_payload, wrap_pad_hybrid_log
from peirastic.realman8dof.modes.contact_recording import drain_study_records
from peirastic.tests.test_api_facade import _ctx, _SEED


class Evidence:
    def __init__(self, omega=0., alpha=1.):
        self.omega, self.alpha = omega, alpha
        self.last_facts = {}

    def update(self, **kwargs):
        return SimpleNamespace(omega_request_rad_s=self.omega, alpha_target=self.alpha,
                               valid=True, reason='test_observation', facts={})

    def close(self, **kwargs):
        pass


def outer(omega=0., alpha=1., stick=None):
    raw, ctx = _ctx()
    payload = pad_hybrid_payload('ultrapoc', desired_z=4.)
    payload.pop('pad_contact'); payload.pop('pad_visual')
    phase = compile_request(ctx, ModeRequest(Mode.TRACK_HYBRID, payload), raw=raw,
                            twist_read=lambda: np.zeros(6) if stick is None else stick)
    clock = [time.monotonic()]
    obj = PadContactOuter(phase.outer, evidence=Evidence(omega, alpha), clock=lambda: clock[0])
    # Most unit cases exercise instantaneous allocation. Dedicated publication
    # tests below exercise the production dispatch window with delayed IK.
    obj.settings.update(minimum_publication_period_s=0.)
    obj._dispatch_times = lambda now, source_deadline: (now, now)
    pose = ctx.kin.fk_pose(_SEED)
    obj.set_origin(pose)
    return obj, pose, clock


def sample(obj, pose, clock, *, my=0., fz=4., **kw):
    clock[0] += .005
    return obj.sample(clock[0], pose, [0.,0.,fz,0.,my,0.], contact=True,
                      dt_actual=.005, sensor_age_s=.001, feedback_age_s=.001,
                      feedback_velocity_valid=True, v_tcp_z_actual=0., **kw)


def publish(obj, cmd, clock):
    assert obj.review(cmd, now_s=clock[0])
    assert obj.started()
    obj.commit(cmd, now_s=clock[0])


def test_prepare_does_not_integrate_until_sent_and_abort_clears_candidate():
    obj, pose, clock = outer(.08)
    theta = obj.force_law.tilt.theta_tilt
    cmd = sample(obj, pose, clock, my=.04)
    assert obj.force_law.tilt.theta_tilt == theta
    assert obj.force_law.tilt._pending_command is not None
    obj.abort('rejected', definitely_not_sent=True)
    assert obj.force_law.tilt._pending_command is None
    assert obj.force_law.tilt.theta_tilt == theta
    with pytest.raises(RuntimeError, match='latched'):
        sample(obj, pose, clock)


def test_published_total_omega_updates_state_and_unified_slew():
    obj, pose, clock = outer(.08)
    previous = 0.; previous_a = 0.
    for _ in range(35):
        cmd = sample(obj, pose, clock, my=.04)
        acceleration = (cmd[4]-previous)/.005
        assert abs(acceleration) <= 2.+1e-6
        assert abs(acceleration-previous_a)/.005 <= obj.rocking.jerk_limit+1e-4
        publish(obj, cmd, clock)
        assert obj.force_law.tilt._w == pytest.approx(cmd[4])
        assert obj._last_final[4] == cmd[4]
        previous, previous_a = cmd[4], acceleration


def test_cop_is_increment_pairing_not_total_rotation():
    obj, pose, clock = outer(.08)
    cmd = sample(obj, pose, clock, my=-.04)
    p = obj._pending
    assert p['cop'] == pytest.approx(.01)
    residual = cmd[2]-p['nominal'][2]-.01*(cmd[4]-p['nominal'][4])
    assert residual == pytest.approx(0., abs=1e-7)
    loading = cmd[2]-.01*(cmd[4]-p['nominal'][4])
    publish(obj, cmd, clock)
    assert obj.controller.v_force_cmd_z == pytest.approx(loading)


def test_invalid_cop_withdraws_visual_and_targets_pause():
    obj, pose, clock = outer(.10)
    cmd = sample(obj, pose, clock, my=.3)
    assert not obj.last_facts['cop_valid']
    assert not obj.last_facts['visual_task_valid']
    assert obj.last_facts['alpha_target'] == 0.
    assert cmd[4] <= 0.
    obj.abort('test', definitely_not_sent=True)


def test_final_review_rejects_ik_change_beyond_normal_and_rotation_guards():
    obj, pose, clock = outer(.10)
    cmd = sample(obj, pose, clock)
    bad = cmd.copy(); bad[2] = .02
    assert not obj.review(bad, now_s=clock[0])
    bad = cmd.copy(); bad[4] = .1
    assert not obj.review(bad, now_s=clock[0])
    obj.abort('test', definitely_not_sent=True)


def test_high_force_and_stale_feedback_stop_before_proposal():
    obj, pose, clock = outer(.1)
    cmd = sample(obj, pose, clock, fz=6.2)
    assert obj._force_supervisor_since is not None
    publish(obj, cmd, clock)
    with pytest.raises(RuntimeError, match='8 N'):
        sample(obj, pose, clock, fz=8.)
    assert obj._pending is None
    with pytest.raises(RuntimeError, match='feedback'):
        obj.sample(clock[0], pose, np.zeros(6), sensor_age_s=.1, feedback_age_s=.01)
    assert obj._pending is None


def test_duplicate_dispatch_or_modified_commit_is_rejected():
    obj, pose, clock = outer()
    cmd = sample(obj, pose, clock)
    assert obj.review(cmd, now_s=clock[0]) and obj.started()
    assert not obj.started()
    with pytest.raises(RuntimeError, match='exact reviewed'):
        obj.commit(cmd+1e-6, now_s=clock[0])
    obj.abort('partial_publish', facts={'arm':'sent','rail':'unknown'})
    assert obj._last_commit_s is None


def test_pause_does_not_accumulate_position_reference():
    stick = np.array([.02,0.,0.,0.,0.,0.])
    obj, pose, clock = outer(alpha=0., stick=stick)
    for _ in range(35):
        cmd = sample(obj, pose, clock)
        publish(obj, cmd, clock)
    assert abs(cmd[0]) < 1e-8
    obj.pad_visual.alpha = 1.
    stick[:] = 0.
    cmd = sample(obj, pose, clock)
    assert abs(cmd[0]) < 1e-8
    np.testing.assert_array_equal(obj.last_pose_d, pose)
    obj.abort('test', definitely_not_sent=True)


def test_log_wrapper_preserves_injected_feedback_and_command_equivalence(tmp_path):
    # Exercise the same signature-based keyword selection used by the runner.
    expected = dict(dt_actual=.007, f_ext_raw=np.ones(6), sensor_age_s=.02,
                    feedback_age_s=.03, feedback_fresh_tick=False,
                    feedback_velocity_valid=True, v_tcp_z_actual=.002, slack_norm=.04)
    seen = []
    class Probe:
        def sample(self,t_s,current_pose,f_ext,*,dt_actual=None,f_ext_raw=None,sensor_age_s=None,
                   feedback_age_s=None,feedback_fresh_tick=None,feedback_velocity_valid=None,
                   v_tcp_z_actual=None,slack_norm=None):
            seen.append((dt_actual,f_ext_raw,sensor_age_s,feedback_age_s,feedback_fresh_tick,
                         feedback_velocity_valid,v_tcp_z_actual,slack_norm))
            return np.array([dt_actual,v_tcp_z_actual,slack_norm,0.,0.,0.])
    probe = Probe()
    signature = inspect.signature(probe.sample)
    plain = probe.sample(0.,np.zeros(6),np.zeros(6),**expected)
    phase = SimpleNamespace(outer=probe,on_exit=None)
    wrap_pad_hybrid_log(phase, {'pad_log':str(tmp_path/'pad.jsonl')})
    assert inspect.signature(probe.sample) == signature
    injected = {k:v for k,v in expected.items() if k in inspect.signature(probe.sample).parameters}
    recorded = probe.sample(0.,np.zeros(6),np.zeros(6),**injected)
    np.testing.assert_array_equal(plain, recorded)
    phase.on_exit(); drain_study_records()
    assert len(seen) == 2 and seen[1][0] == .007 and seen[1][-1] == .04


def test_independent_pad_config_and_legacy_payload():
    config = load_pad_contact_config()
    assert config['pad']['normal_max_m_s'] == .015
    assert config['pad']['max_visual_omega_rad_s'] == .10
    assert 'pad_contact' not in pad_hybrid_payload('legacy')
    assert 'pad_contact' not in pad_hybrid_payload('ac2d')


def test_old_pad_config_cannot_reinstate_short_dispatch_window():
    config = load_pad_contact_config()
    config['pad'].pop('dispatch_policy')
    config['pad'].update(dispatch_lead_s=.008, dispatch_window_s=.002)
    loaded = load_pad_contact_config(config)['pad']
    assert loaded['dispatch_policy'] == 'full_command_lifetime_v1'
    assert 'dispatch_lead_s' not in loaded and 'dispatch_window_s' not in loaded
    assert loaded['command_max_age_s'] == .08
    assert loaded['hard_stop_persist_s'] == .15
    assert loaded['hard_stop_latch_n'] == 8.


def test_infeasible_handoff_records_solver_input_and_aborts():
    from peirastic.realman8dof.force.torque_tilt import rotation_from_pose
    obj, pose, clock = outer()
    records = []
    obj.record_sink = SimpleNamespace(emit=lambda event, **fields: records.append((event, fields)))
    r = rotation_from_pose(pose)
    # Deliberately impossible entry: normal speed cannot reach its hard limit
    # in one cycle. Preserve the failure and record enough to identify its rows.
    obj.begin_hybrid_episode(np.r_[r@np.array([0.,0.,.1]), np.zeros(3)], pose)
    with pytest.raises(RuntimeError, match='allocation failed'):
        sample(obj, pose, clock)
    failed = next(fields for event, fields in records if event == 'pad_allocation_failed')
    assert failed['diagnostics']['reason'] == 'mechanical_rows_infeasible'
    assert 'replay_input' in failed['diagnostics']
    assert failed['previous_tool'][2] == pytest.approx(.1)
    assert obj._pending is None and obj._faulted
    assert obj._last_commit_s is None


def test_pause_rejects_final_ik_drift_and_rows_fit_native_capacity():
    obj, pose, clock = outer(alpha=0.)
    cmd = sample(obj, pose, clock)
    rows = obj.command_constraints()
    assert len(rows['command_twist_rows_base']) <= 16
    bad = cmd.copy(); bad[0] = .0002
    assert not obj.review(bad, now_s=clock[0])
    publish(obj, cmd, clock)


def test_publication_infeasible_abort_does_not_latch():
    obj, pose, clock = outer()
    sample(obj, pose, clock)
    obj.abort('qpik_fault:stop:publication_infeasible', definitely_not_sent=True)
    assert not obj._faulted
    sample(obj, pose, clock)
    obj.abort('test', definitely_not_sent=True)


def test_publication_history_uses_dispatch_not_variable_transport_completion():
    obj, pose, clock = outer(.08)
    for delay in (.003, .0, .002):
        cmd = sample(obj, pose, clock)
        dispatch = clock[0]
        assert obj.review(cmd, now_s=dispatch) and obj.started()
        obj.commit(cmd, now_s=dispatch+delay)
        assert obj.rocking.time_s == dispatch
        assert abs(obj.rocking.acceleration_base @ obj.pending_rotation_base_tcp[:, 1]) <= 2.


def test_regressed_force_source_rejected_before_new_proposal():
    obj, pose, clock = outer()
    cmd = sample(obj, pose, clock, wrench_source_time_s=10., wrench_source_id='device')
    publish(obj, cmd, clock)
    with pytest.raises(RuntimeError, match='regressed'):
        sample(obj, pose, clock, wrench_source_time_s=9.99, wrench_source_id='device')
    assert obj._pending is None


def test_real_nominal_and_allocation_are_equal_with_logging_on_or_off(tmp_path):
    plain, pose, clock = outer(.08, alpha=.2)
    logged, logged_pose, logged_clock = outer(.08, alpha=.2)
    logged_clock[0] = clock[0]
    plain.set_origin(pose)
    logged.set_origin(logged_pose)
    phase = SimpleNamespace(outer=logged, on_exit=logged.close)
    wrap_pad_hybrid_log(phase, {'pad_log': str(tmp_path/'real.jsonl')})
    try:
        for _ in range(8):
            a = sample(plain, pose, clock, my=.04, wrench_source_time_s=clock[0])
            b = sample(logged, logged_pose, logged_clock, my=.04, wrench_source_time_s=logged_clock[0])
            np.testing.assert_allclose(a, b, atol=1e-12)
            publish(plain, a, clock); publish(logged, b, logged_clock)
    finally:
        phase.on_exit(); drain_study_records()


@pytest.mark.parametrize('backend', ['python', 'native'])
@pytest.mark.parametrize('omega,alpha,stick', [
    (0., 0., np.zeros(6)), (.08, 0., np.zeros(6)),
    (0., 1., np.array([.02,0.,0.,0.,0.,0.])),
])
def test_real_inner_ik_enforces_contact_rows_before_final_review(omega, alpha, stick, backend):
    from peirastic.realman8dof.force.torque_tilt import rotation_from_pose
    obj, pose, clock = outer(omega=omega, alpha=alpha, stick=stick)
    _, ctx = _ctx()
    inner = ctx.inner
    if backend == 'native':
        from copy import deepcopy
        from rm75_control.control.joint_admittance_8dof.loop import JointIkController
        cfg = deepcopy(inner.cfg)
        cfg.backend = 'native'
        inner = JointIkController(ctx.kin, cfg)
        inner._native.timeout_s = .5
        inner.reset(_SEED.copy())
    cmd = sample(obj, pose, clock, my=.04 if omega else 0.)
    step = inner.update(cmd, dt=.005, q_meas=_SEED.copy(), vel_ff=obj.last_vel_ff,
        pose_d=obj.last_pose_d, f_ext_z=4., f_des_z=4., contact_active=True,
        path_twist=obj.last_path_twist, feedback_twist=obj.last_feedback_twist,
        commit_history=False, auto_commit=False, v_force_z=obj.controller.v_force_z,
        rail_exec_vel_m_s=0., **obj.command_constraints())
    r = rotation_from_pose(pose)
    final = np.r_[r.T@step.v_tcp_estimated[:3], r.T@step.v_tcp_estimated[3:]]
    assert not step.fallback_reason
    assert obj.review(final, now_s=clock[0]) and obj.started()
    inner.commit_publication(step.qdot)
    obj.commit(final, now_s=clock[0])
    obj.close()
    if inner._native is not None:
        inner._native.shutdown()


def test_actual_dispatch_gaps_bound_jerk_even_below_nominal_period():
    obj, pose, clock = outer(.08)
    last_w = last_a = 0.
    for gap in (.005, .002, .008, .003):
        clock[0] += gap
        cmd = obj.sample(clock[0], pose, [0.,0.,4.,0.,.04,0.], contact=True,
            dt_actual=.005, sensor_age_s=.001, feedback_age_s=.001)
        a = (cmd[4]-last_w)/gap
        assert abs(a-last_a)/gap <= obj.rocking.jerk_limit+1e-4
        publish(obj, cmd, clock)
        last_w, last_a = cmd[4], a


def test_final_review_also_covers_fixed_manual_rotation_subspace():
    obj, pose, clock = outer()
    cmd = sample(obj, pose, clock)
    bad = cmd.copy(); bad[3] = .001
    assert not obj.review(bad, now_s=clock[0])
    obj.abort('test', definitely_not_sent=True)


@pytest.mark.parametrize('axis', [3, 5])
def test_manual_rotation_start_and_release_slew_without_fault(axis):
    stick = np.zeros(6); stick[axis] = .1
    obj, pose, clock = outer(stick=stick)
    cmd = sample(obj, pose, clock)
    assert 0 < cmd[axis] <= obj.solver.config.max_acceleration[axis]*.005+1e-8
    publish(obj, cmd, clock)
    stick[:] = 0.
    cmd = sample(obj, pose, clock)
    assert abs(cmd[axis]) < 1e-8
    publish(obj, cmd, clock)


@pytest.mark.parametrize('next_stick', [0., -.02])
def test_tangential_release_or_reversal_brakes_without_accumulated_motion(next_stick):
    stick = np.array([.02,0.,0.,0.,0.,0.])
    obj, pose, clock = outer(stick=stick)
    for _ in range(8):
        cmd = sample(obj, pose, clock); publish(obj, cmd, clock)
    previous = cmd[0]
    assert previous > .005
    stick[0] = next_stick
    for _ in range(12):
        cmd = sample(obj, pose, clock)
        assert abs(cmd[0]-previous) <= obj.solver.config.max_acceleration[0]*.005+1e-8
        publish(obj, cmd, clock)
        previous = cmd[0]
    assert cmd[0] <= 1e-8
    if next_stick == 0.: assert abs(cmd[0]) < 1e-8


def test_pad_config_uses_scan_five_millisecond_period():
    cfg = load_pad_contact_config()
    assert cfg['pad']['minimum_publication_period_s'] == pytest.approx(.005)


def test_wall_stretch_keeps_five_millisecond_force_and_cop_hold():
    fast, pose, clock_f = outer()
    slow, _, clock_s = outer()
    for obj, clock in ((fast, clock_f), (slow, clock_s)):
        publish(obj, sample(obj, pose, clock, fz=3.), clock)
    cmd_fast = sample(fast, pose, clock_f, fz=3.)
    clock_s[0] += .020
    cmd_slow = slow.sample(clock_s[0], pose, [0., 0., 3., 0., 0., 0.], contact=True,
                           dt_actual=.020, sensor_age_s=.001, feedback_age_s=.001)
    assert slow.last_facts['command_hold_s'] == pytest.approx(.005)
    assert fast.last_facts['command_hold_s'] == pytest.approx(.005)
    assert cmd_slow[2] == pytest.approx(cmd_fast[2], abs=1e-12)
    slow.abort('test', definitely_not_sent=True)
    fast.abort('test', definitely_not_sent=True)


def test_five_millisecond_ticks_do_not_wait_for_publication():
    obj, pose, clock = outer()
    del obj.__dict__['_dispatch_times']
    obj.settings.update(minimum_publication_period_s=.005)
    waits = []
    for _ in range(40):
        cmd = sample(obj, pose, clock)
        waits.append(obj.dispatch_wait_s(clock[0]))
        publish(obj, cmd, clock)
    assert waits[0] == pytest.approx(0.)
    assert max(waits[1:]) < 1e-12


def test_recorded_1723_overshoot_and_55ms_period_keep_force_loop():
    obj, pose, clock = outer()
    cmd = sample(obj, pose, clock, fz=5.412)
    publish(obj, cmd, clock)
    cmd = sample(obj, pose, clock, fz=6.1)
    assert cmd[2] < 0.
    publish(obj, cmd, clock)
    clock[0] += .055
    cmd = obj.sample(clock[0], pose, [0., 0., 4.217, 0., .039, 0.], contact=True,
                     dt_actual=.055, sensor_age_s=.001, feedback_age_s=.001)
    publish(obj, cmd, clock)
    obj.abort('test', definitely_not_sent=True)


def test_six_newton_persist_still_stops():
    obj, pose, clock = outer()
    for _ in range(5):
        cmd = sample(obj, pose, clock, fz=6.2)
        publish(obj, cmd, clock)
    clock[0] += .16
    with pytest.raises(RuntimeError, match='6 N'):
        obj.sample(clock[0], pose, [0., 0., 6.2, 0., 0., 0.], contact=True,
                   dt_actual=.03, sensor_age_s=.001, feedback_age_s=.001)
    assert obj._pending is None
