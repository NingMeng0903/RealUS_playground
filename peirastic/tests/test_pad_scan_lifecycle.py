"""Offline production-payload and accepted-publication tests for pad scan."""
from types import SimpleNamespace
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from peirastic.core.modes import Mode, ModeRequest
from peirastic.realman8dof.session import compile_request
from peirastic.realman8dof.modes.pad_scan_config import load_pad_scan_config
from peirastic.realman8dof.modes.pad_visual import pad_hybrid_payload
from peirastic.tests.test_api_facade import _ctx, _SEED
from rm75_control.control.joint_admittance_8dof.command_lifecycle import CommandLifecycleTransaction


def outer(omega=.08, stick=None, log_path=None, config=None):
    raw, ctx = _ctx()
    payload = pad_hybrid_payload('ultrapoc', desired_z=4., log_path=log_path, contact_config=config)
    phase = compile_request(ctx, ModeRequest(Mode.TRACK_HYBRID, payload), raw=raw,
        twist_read=lambda: np.zeros(6) if stick is None else np.asarray(stick))
    obj = phase.outer
    from peirastic.realman8dof.modes.pad_scan_lifecycle import PadScanLifecycle
    if getattr(obj, 'command_lifecycle', None) is None:
        obj.command_lifecycle = PadScanLifecycle(
            obj, payload.get('pad_scan_config') or load_pad_scan_config())
    owner = obj.command_lifecycle
    if getattr(obj, 'pad_log_sink', None) is not None:
        owner.record_sink = obj.pad_log_sink
    owner._test_phase = phase
    clock = [time.monotonic()]
    owner.clock = lambda: clock[0]
    guide = obj.pad_visual
    guide.omega_target, guide.last_alpha, guide.evidence_valid = omega, 1., True
    guide.preview = lambda **kw: omega
    guide.commit = lambda *args: None
    pose = ctx.kin.fk_pose(_SEED)
    obj.set_origin(pose)
    return obj, owner, pose, clock


def sample(obj, pose, clock, *, my=0., fz=4., dt=.005, **kwargs):
    clock[0] += dt
    return obj.sample(clock[0], pose, [0., 0., fz, 0., my, 0.], contact=True,
        dt_actual=dt, sensor_age_s=.001, feedback_age_s=.001,
        feedback_velocity_valid=True, v_tcp_z_actual=0., **kwargs)


def step_for(owner, final, *, tier=4, low=-np.inf, high=np.inf):
    rotation = owner.pending_rotation_base_tcp
    return SimpleNamespace(qdot=np.zeros(8), q_send=np.zeros(8),
        v_tcp_estimated=np.r_[rotation@final[:3], rotation@final[3:]],
        rocking_policy_tier=tier, rocking_lower_rad_s=low, rocking_upper_rad_s=high)


def publish(owner, final, clock, *, tx=None, tier=4, low=-np.inf, high=np.inf):
    tx = tx or CommandLifecycleTransaction(owner)
    if tx.state == 'idle':
        tx.begin_candidate()
    assert tx.review(step_for(owner, final, tier=tier, low=low, high=high),
                     published_rail_m=0., dt_s=.005, now_s=clock[0])
    assert tx.started()
    clock[0] += .0002
    assert tx.arm_dispatch(now_s=clock[0])
    tx.commit(now_s=clock[0]+.001, rail_enabled=True)
    return tx


@pytest.mark.parametrize('my', [-.04, .04])
def test_same_and_opposite_visual_moment_commit_nominal_separately(my):
    obj, owner, pose, clock = outer()
    before = obj.force_law.tilt._w
    cmd = sample(obj, pose, clock, my=my)
    nominal = owner._pending['nominal'].copy()
    assert obj.force_law.tilt._w == before
    assert abs(cmd[2]) <= .015+1e-8 and abs(cmd[4]) <= .28+1e-8
    publish(owner, cmd, clock)
    assert obj.force_law.tilt._w == pytest.approx(nominal[4])
    assert owner._last_final == pytest.approx(cmd)
    assert obj.controller.v_force_cmd_z == pytest.approx(nominal[2])
    assert owner.rocking.time_s == clock[0]
    assert owner._pending is None


def test_invalid_cop_withdraws_visual_and_hard_pauses_path():
    obj, owner, pose, clock = outer(stick=[.02, 0., 0., 0., 0., 0.])
    cmd = sample(obj, pose, clock, my=.3)
    assert not obj.last_facts['cop_valid']
    assert not obj.last_facts['visual_task_valid']
    assert obj.last_facts['alpha_target'] == 0.
    assert np.max(np.abs(cmd[:2])) < 1e-10
    assert owner._pending['path_paused']
    owner.abort('test', definitely_not_sent=True)


@pytest.mark.parametrize('filtered,raw', [(6., 4.), (4., 6.), (6.1, 6.2)])
def test_fresh_high_force_stops_immediately_before_nominal_prepare(filtered, raw):
    obj, owner, pose, clock = outer()
    with pytest.raises(RuntimeError, match='6 N'):
        sample(obj, pose, clock, fz=filtered, f_ext_raw=[0., 0., raw, 0., 0., 0.])
    assert owner._pending is None
    assert obj.force_law.tilt._pending_command is None


@pytest.mark.parametrize('axis,value', [(2, .01501), (4, .28002)])
def test_tier_four_never_relaxes_final_hard_speed_caps(axis, value):
    obj, owner, pose, clock = outer()
    cmd = sample(obj, pose, clock).copy()
    cmd[axis] = value
    assert not owner.review(cmd, now_s=clock[0], facts={'rocking_policy_tier': 4})
    assert owner.last_review_reason == 'final_hard_velocity_limit'
    owner.abort('test', definitely_not_sent=True)


@pytest.mark.parametrize('sent', [False, True])
def test_failed_or_partial_send_rolls_back_nominal_and_never_advances_history(sent):
    obj, owner, pose, clock = outer()
    tx = CommandLifecycleTransaction(owner); tx.begin_candidate()
    cmd = sample(obj, pose, clock, my=.04)
    before_time = owner.rocking.time_s
    if sent:
        assert tx.review(step_for(owner, cmd), published_rail_m=0., dt_s=.005, now_s=clock[0])
        assert tx.started()
    tx.abort('partial_publish' if sent else 'ik_rejected', definitely_not_sent=not sent)
    assert owner._pending is None and obj.force_law.tilt._pending_command is None
    assert obj.force_law.tilt._w == 0.
    assert owner.rocking.time_s == before_time
    assert np.all(owner._last_final == 0.)
    if sent:
        with pytest.raises(RuntimeError, match='latched'):
            sample(obj, pose, clock)
    else:
        sample(obj, pose, clock)
        owner.abort('cleanup', definitely_not_sent=True)


def test_qp_failure_is_logged_stop_without_additive_fallback(monkeypatch):
    obj, owner, pose, clock = outer()
    monkeypatch.setattr(owner.solver, 'solve', lambda *a, **kw: SimpleNamespace(
        qp_twist=None, status='failed', diagnostics={'reason': 'test_infeasible'}))
    with pytest.raises(RuntimeError, match='QP failed: test_infeasible'):
        sample(obj, pose, clock)
    assert obj.last_facts['allocation_success'] is False
    assert owner._pending is None and obj.force_law.tilt._pending_command is None


@pytest.mark.parametrize('tier', [1, 2, 3, 4])
def test_native_selected_interval_is_reviewed_without_narrow_recomputation(tier):
    obj, owner, pose, clock = outer()
    cmd = sample(obj, pose, clock)
    constraints = owner.command_constraints()
    assert set(constraints) == {'rocking_axis_base', 'rocking_bounds',
                                'command_twist_rows_base', 'command_twist_lower', 'command_twist_upper'}
    rows = constraints['command_twist_rows_base']
    assert rows.shape == (2, 6)
    omega = load_pad_scan_config()['pad']['angular_max_rad_s']
    np.testing.assert_allclose(constraints['command_twist_lower'], [-.015, -omega])
    np.testing.assert_allclose(constraints['command_twist_upper'], [.015, omega])
    rotation = owner.pending_rotation_base_tcp
    probe_tool = np.array([.01, -.02, .013, -.04, .19, .06])
    probe_base = np.r_[rotation @ probe_tool[:3], rotation @ probe_tool[3:]]
    np.testing.assert_allclose(rows @ probe_base, probe_tool[[2, 4]], atol=1e-12)
    assert owner.rocking._preview_jerk_scale == 1.
    # Acceptance uses the native interval for its actual selected tier.
    publish(owner, cmd, clock, tier=tier, low=cmd[4]-1e-4, high=cmd[4]+1e-4)


def test_jitter_keeps_nominal_hold_five_ms_and_records_actual_dispatch():
    obj, owner, pose, clock = outer()
    cmd = sample(obj, pose, clock, dt=.013)
    assert owner._pending['metadata']['command_hold_s'] == .005
    assert owner._pending['actual_dt'] == .013
    assert owner._pending['result'].diagnostics['command_hold_s'] == .005
    publish(owner, cmd, clock)
    assert owner.rocking.time_s == clock[0]


@pytest.mark.parametrize('seed_omega', [0., -.05, .05])
def test_guarded_mechanical_interval_is_sent_to_ik_before_publication(seed_omega):
    obj, owner, pose, clock = outer()
    rotation = Rotation.from_euler('xyz', pose[3:]).as_matrix()
    obj.begin_hybrid_episode(np.r_[np.zeros(3), rotation @ [0., seed_omega, 0.]], pose)
    sample(obj, pose, clock, slack_norm=1000.)
    assert owner._pending['guarded']
    assert not obj.last_facts['visual_task_valid']
    assert obj.last_facts['visual_request_rad_s'] == 0.
    nominal = owner._pending['nominal'][4]
    if seed_omega:
        assert 0 < abs(nominal) < abs(seed_omega)
        assert np.sign(nominal) == np.sign(seed_omega)
    constraints = owner.command_constraints()
    assert constraints['command_twist_rows_base'].shape == (2, 6)
    np.testing.assert_allclose(constraints['command_twist_lower'], [-.015, min(0., nominal)])
    np.testing.assert_allclose(constraints['command_twist_upper'], [.015, max(0., nominal)])
    owner.abort('test', definitely_not_sent=True)
    sample(obj, pose, clock)
    assert not owner._pending['guarded']
    released = owner.command_constraints()
    omega = load_pad_scan_config()['pad']['angular_max_rad_s']
    np.testing.assert_allclose(released['command_twist_lower'], [-.015, -omega])
    np.testing.assert_allclose(released['command_twist_upper'], [.015, omega])
    owner.abort('test', definitely_not_sent=True)


def test_measured_relative_pose_drives_both_tilt_guard_and_qp(monkeypatch):
    obj, owner, pose, clock = outer(omega=0.)
    sample(obj, pose, clock)
    owner.abort('unsent', definitely_not_sent=True)
    rotation = Rotation.from_euler('xyz', pose[3:])
    changed = pose.copy()
    changed[3:] = (rotation*Rotation.from_rotvec([0., .12, 0.])).as_euler('xyz')
    observed = {}
    original = owner.force_law.prepare
    def prepare(**kw):
        observed['nominal_theta'] = owner.force_law.tilt.theta_tilt
        return original(**kw)
    monkeypatch.setattr(owner.force_law, 'prepare', prepare)
    sample(obj, changed, clock)
    assert observed['nominal_theta'] == pytest.approx(.12)
    assert owner._pending['theta'] == pytest.approx(.12)
    owner.abort('test', definitely_not_sent=True)


def test_handoff_seeds_supplied_twist_and_reexpresses_previous_frame():
    obj, owner, pose, clock = outer()
    rotation = Rotation.from_euler('xyz', pose[3:]).as_matrix()
    tool = np.array([0., 0., .003, 0., .04, 0.])
    base = np.r_[rotation@tool[:3], rotation@tool[3:]]
    obj.begin_hybrid_episode(base, pose)
    assert owner._last_final == pytest.approx(tool)
    assert owner.controller.last_v_cmd == pytest.approx(tool)
    assert owner.force_law.tilt._w == pytest.approx(.04)
    changed = pose.copy()
    changed[3:] = (Rotation.from_matrix(rotation)*Rotation.from_rotvec([0., 0., .1])).as_euler('xyz')
    sample(obj, changed, clock)
    current = Rotation.from_euler('xyz', changed[3:]).as_matrix()
    assert owner._pending['previous'] == pytest.approx(np.r_[current.T@base[:3], current.T@base[3:]])
    owner.abort('test', definitely_not_sent=True)


def test_legacy_record_payload_has_no_publication_adapter():
    raw, ctx = _ctx()
    payload = pad_hybrid_payload('legacy', desired_z=4.)
    phase = compile_request(ctx, ModeRequest(Mode.TRACK_HYBRID, payload), raw=raw,
                            twist_read=lambda: np.zeros(6))
    assert getattr(phase.outer, 'command_lifecycle', None) is None


def test_ultrapoc_compile_has_no_publication_adapter():
    raw, ctx = _ctx()
    payload = pad_hybrid_payload('ultrapoc', desired_z=4.)
    phase = compile_request(ctx, ModeRequest(Mode.TRACK_HYBRID, payload), raw=raw,
                            twist_read=lambda: np.array([0.02, 0., 0., 0., 0., 0.]))
    assert getattr(phase.outer, 'command_lifecycle', None) is None


def test_ik_saturation_uses_scan_antiwindup_without_visual_state_reinjection():
    obj, owner, pose, clock = outer()
    for index in range(3):
        cmd = sample(obj, pose, clock, my=-.04).copy()
        nominal = owner._pending['nominal'].copy()
        cmd[4] = 0.
        publish(owner, cmd, clock)
        if index < 2:
            assert obj.force_law.tilt._w == pytest.approx(nominal[4])
        else:
            assert abs(obj.force_law.tilt._w) < abs(nominal[4])
        assert owner._last_final[4] == 0.


def test_force_gate_applies_once_and_visual_cap_is_soft_request():
    obj, owner, pose, clock = outer(omega=.2)
    sample(obj, pose, clock, fz=4.25)
    diag = obj.last_facts['allocation_diagnostics']
    from peirastic.realman8dof.modes.pad_scan_config import pad_visual_force_gate
    gate = pad_visual_force_gate(4.25, 4., 5.5)
    visual_cap = load_pad_scan_config()['pad']['max_visual_omega_rad_s']
    assert diag['visual_request_rad_s'] == pytest.approx(visual_cap)
    assert diag['visual_request_gated_rad_s'] == pytest.approx(visual_cap * gate)
    assert diag['visual_shortfall_rad_s'] > 0.
    owner.abort('test', definitely_not_sent=True)


def test_raw_force_spike_below_stop_does_not_gate_filtered_contact_evidence():
    obj, owner, pose, clock = outer(omega=.08)
    evidence_forces = []
    obj.pad_visual.preview = lambda **kw: evidence_forces.append(kw['force_n']) or .08
    sample(obj, pose, clock, fz=4., f_ext_raw=[0., 0., 4.6, 0., 0., 0.])
    assert evidence_forces == [4.]
    assert owner.supervisory_force_n == 4.6
    assert owner.force_n == 4.
    diag = obj.last_facts['allocation_diagnostics']
    assert diag['visual_request_rad_s'] == pytest.approx(.08)
    assert diag['visual_request_gated_rad_s'] == pytest.approx(.08)
    assert obj.last_facts['visual_task_valid']
    owner.abort('test', definitely_not_sent=True)


def test_outer_normal_reserve_preserves_nominal_and_final_physical_caps():
    obj, owner, pose, clock = outer()
    assert obj.controller.cfg.max_vz_tool_m_s == .015
    assert owner.solver.config.max_velocity[2] == pytest.approx(.0149)
    cmd = sample(obj, pose, clock).copy()
    cmd[2] = .015
    assert owner.review(cmd, now_s=clock[0], facts={'rocking_policy_tier': 4})
    snapshot = owner.configuration_snapshot()
    assert snapshot['effective_pad_settings']['normal_allocation_margin_m_s'] == .0001
    assert snapshot['normal_allocation_max_m_s'] == pytest.approx(.0149)
    assert snapshot['final_max_velocity'][2] == .015
    assert snapshot['final_normal_tolerance_m_s'] == 1e-6
    owner.abort('test', definitely_not_sent=True)


def test_contact_guard_withdraws_image_before_preview_and_rejects_ik_bypass():
    obj, owner, pose, clock = outer()
    observed = []
    obj.pad_visual.preview = lambda **kw: observed.append(kw['contact_present'])
    cmd = sample(obj, pose, clock, slack_norm=1000.).copy()
    assert observed == [False]
    assert not obj.last_facts['visual_task_valid']
    cmd[4] = .01
    assert not owner.review(cmd, now_s=clock[0], facts={'rocking_policy_tier': 4})
    assert owner.last_review_reason == 'final_mechanical_guard'
    owner.abort('test', definitely_not_sent=True)


@pytest.mark.parametrize('omega,accepted', [(1.77e-5, True), (-1.77e-5, True),
                                           (1.01e-4, False), (-1.01e-4, False)])
def test_mechanical_guard_uses_existing_task_residual_tolerance(omega, accepted):
    obj, owner, pose, clock = outer()
    cmd = sample(obj, pose, clock, slack_norm=1000.).copy()
    assert owner._pending['guarded']
    assert owner._pending['nominal'][4] == 0.
    cmd[4] = omega
    assert owner.review(cmd, now_s=clock[0], facts={'rocking_policy_tier': 4}) is accepted
    assert owner.last_review_reason == (None if accepted else 'final_mechanical_guard')
    snapshot = owner.configuration_snapshot()
    assert snapshot['mechanical_guard_tolerance_rad_s'] == 1e-4
    assert snapshot['final_omega_tolerance_rad_s'] == 1e-5
    owner.abort('test', definitely_not_sent=True)


@pytest.mark.parametrize('configured,expected,omega', [(.01, 1e-4, .009), (1e-5, 1e-5, 1.77e-5)])
def test_custom_task_tolerance_cannot_relax_mechanical_guard(configured, expected, omega):
    from peirastic.realman8dof.modes.pad_scan_config import load_pad_scan_config
    config = load_pad_scan_config()
    tolerance = np.array(config['qp'].get('subspace_tolerance', [1e-4]*6), dtype=float)
    tolerance[4] = configured
    config['qp']['subspace_tolerance'] = tolerance.tolist()
    obj, owner, pose, clock = outer(config=config)
    cmd = sample(obj, pose, clock, slack_norm=1000.).copy()
    assert owner._pending['guarded'] and owner._pending['nominal'][4] == 0.
    assert owner.mechanical_guard_tolerance_rad_s == expected
    cmd[4] = omega
    assert not owner.review(cmd, now_s=clock[0], facts={'rocking_policy_tier': 4})
    assert owner.last_review_reason == 'final_mechanical_guard'
    assert owner.configuration_snapshot()['mechanical_guard_tolerance_rad_s'] == expected
    owner.abort('test', definitely_not_sent=True)


def test_duplicate_recent_wrench_keeps_evidence_eligible_without_reobserving():
    obj, owner, pose, clock = outer()
    source = clock[0]
    for index in range(2):
        sample(obj, pose, clock, wrench_source_time_s=source, wrench_source_id='sensor')
        assert owner._pending['fresh'] is (index == 0)
        assert owner._pending['preview_contact']
        assert obj.last_facts['visual_task_valid']
        owner.abort('test', definitely_not_sent=True)


def test_reanchor_aborts_pending_before_reset():
    obj, owner, pose, clock = outer()
    sample(obj, pose, clock)
    obj.set_origin(pose)
    assert owner._pending is None
    assert obj.force_law.tilt._pending_command is None


def test_logging_does_not_change_compiled_control_or_commit(tmp_path):
    import json
    from peirastic.realman8dof.modes.contact_recording import drain_study_records
    results = []
    path = tmp_path/'pad.jsonl'
    for log in (None, str(path)):
        obj, owner, pose, clock = outer(log_path=log)
        cmd = sample(obj, pose, clock, my=.04)
        publish(owner, cmd, clock)
        results.append((cmd.copy(), obj.force_law.tilt._w, obj.controller.v_force_cmd_z))
        if owner._test_phase.on_exit:
            owner._test_phase.on_exit()
    drain_study_records()
    assert results[0][0] == pytest.approx(results[1][0])
    assert results[0][1:] == pytest.approx(results[1][1:])
    events = [json.loads(line) for line in path.read_text().splitlines()]
    commits = [row for row in events if row.get('event') == 'pad_publication_commit']
    assert len(commits) == 1
    assert commits[0]['final_is_measurement'] is False
    assert 'nominal_tool' in commits[0] and 'rocking_audit' in commits[0]


def test_custom_config_limits_are_executed_and_snapshotted():
    from peirastic.realman8dof.modes.pad_scan_config import load_pad_scan_config
    config = load_pad_scan_config()
    config['pad'].update(normal_max_m_s=.01, angular_max_rad_s=.2,
        angular_acceleration_rad_s2=1.5, max_visual_omega_rad_s=.06,
        cop_min_force_n=1., cop_max_m=.02)
    obj, owner, pose, clock = outer(config=config, omega=.1)
    assert owner.solver.config.max_velocity[2] == pytest.approx(.0099)
    assert owner.solver.config.max_velocity[4] == .2
    assert owner.rocking.acceleration_limit == 1.5
    assert owner.solver.config.cop_min_force_n == 1.
    assert owner.solver.config.cop_max_m == .02
    snapshot = owner.configuration_snapshot()
    assert snapshot['effective_qp_config']['max_visual_omega_rad_s'] == .06
    sample(obj, pose, clock)
    assert obj.last_facts['visual_request_rad_s'] == .06
    owner.abort('test', definitely_not_sent=True)


def test_invalid_cop_is_withdrawn_before_evidence_preview():
    obj, owner, pose, clock = outer()
    observed = []
    obj.pad_visual.preview = lambda **kw: observed.append(kw['contact_present'])
    sample(obj, pose, clock, my=.3)
    assert observed == [False]
    owner.abort('test', definitely_not_sent=True)
