"""Recorded startup/contact inputs, real IK backends, no robot or transport I/O."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest
import yaml

from peirastic.configs import DEFAULT_CONTROLLER_YAML
from peirastic.core.modes import Mode, ModeRequest
from peirastic.realman8dof.session import CompileContext, compile_request
from peirastic.realman8dof.force.torque_tilt import rotation_from_pose
from peirastic.realman8dof.modes.pad_contact import PadContactOuter
from peirastic.realman8dof.modes.pad_visual import pad_hybrid_payload
from peirastic.tests.test_pad_contact_outer import Evidence
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics

CASES = json.loads((Path(__file__).parent/'data/pad_install_20260921.json').read_text())
CONTACT_1650 = json.loads((Path(__file__).parent/'data/pad_contact_1650.json').read_text())['sessions']
CONTACT_CASES = (json.loads((Path(__file__).parent/'data/pad_contact_1622.json').read_text())['sessions']
                 + [case for case in CONTACT_1650 if case['rows']])


@pytest.mark.parametrize('case', CASES, ids=lambda r:r['source'])
@pytest.mark.parametrize('backend', ['python', 'native'])
def test_recorded_startup_with_actual_period_and_exact_rotation_guard(case, backend):
    raw = yaml.safe_load(DEFAULT_CONTROLLER_YAML.read_text())
    cfg = build_joint_ik_config(raw)
    cfg.backend = backend
    cfg.ird.enabled = False
    cfg.native_shm_prefix = 'pad_install_test_'+str(os.getpid())
    kin = RobotKinematics()
    kin.apply_link7_to_tcp_offset(case['tcp_offset_pose'])
    inner = JointIkController(kin, cfg)
    obj = None
    try:
        if inner._native: inner._native.timeout_s = .5
        q = np.array(case['q_meas'])
        inner.reset(q)
        ctx = CompileContext(kin=kin, inner=inner, euler_order=cfg.euler_order,
                             control_frame=cfg.control_frame, v_scale=cfg.v_scale)
        payload = pad_hybrid_payload('ultrapoc')
        payload.pop('pad_contact'); payload.pop('pad_visual')
        # Actual released-stick tails from the second failed installation.
        phase = compile_request(ctx, ModeRequest(Mode.TRACK_HYBRID, payload), raw=raw,
            twist_read=lambda: np.array([6e-10, 2.732e-9, 0., 0., 0., 0.]))
        clock = [time.monotonic()]
        obj = PadContactOuter(phase.outer, evidence=Evidence(0., 0.), clock=lambda: clock[0])
        obj.set_origin(kin.fk_pose(q))
        rail = 0.
        for dt in [.02, .0268, .02135, .02]*5:
            clock[0] += dt
            pose = kin.fk_pose(q)
            cmd = obj.sample(clock[0], pose, [0.,0.,.4,0.,-.1,0.], contact=False,
                dt_actual=dt, sensor_age_s=.002, feedback_age_s=.002)
            assert obj.last_facts['allocation_diagnostics']['zero_path']
            step = inner.update(cmd, dt, q_meas=q, vel_ff=obj.last_vel_ff, pose_d=obj.last_pose_d,
                f_ext_z=.4, f_des_z=4., contact_active=False, path_twist=obj.last_path_twist,
                feedback_twist=obj.last_feedback_twist, v_force_z=obj.controller.v_force_z,
                rail_exec_vel_m_s=rail, dt_wall_s=dt, auto_commit=False, commit_history=False,
                **obj.command_constraints())
            assert not step.fallback_reason, step.fallback_reason
            r = rotation_from_pose(pose)
            final = np.r_[r.T@step.v_tcp_estimated[:3], r.T@step.v_tcp_estimated[3:]]
            assert abs(final[4]) <= 1e-8
            clock[0] += obj.dispatch_wait_s(clock[0])
            assert obj.review(final, now_s=clock[0]) and obj.started()
            assert obj.arm_dispatch_time(clock[0])
            inner.commit_publication(step.qdot)
            obj.commit(final, now_s=clock[0])
            q = step.q_send.copy()
            rail = .3*step.qdot[0]+.7*rail
    finally:
        if obj is not None: obj.close()
        if inner._native: inner._native.shutdown()


@contextmanager
def recorded_chain(backend, q, *, pose=None, injected_qdot=None):
    """Real solver and publication history; ideal tracking, no hardware I/O."""
    raw = yaml.safe_load(DEFAULT_CONTROLLER_YAML.read_text())
    cfg = build_joint_ik_config(raw)
    cfg.backend = backend
    cfg.ird.enabled = False
    cfg.native_shm_prefix = 'pad_contact_test_'+str(os.getpid())
    kin = RobotKinematics()
    kin.apply_link7_to_tcp_offset(CASES[0]['tcp_offset_pose'])
    inner = JointIkController(kin, cfg)
    obj = None
    try:
        if inner._native: inner._native.timeout_s = .5
        q = np.asarray(q, dtype=float)
        inner.reset(q)
        ctx = CompileContext(kin=kin, inner=inner, euler_order=cfg.euler_order,
                             control_frame=cfg.control_frame, v_scale=cfg.v_scale)
        payload = pad_hybrid_payload('ultrapoc')
        payload.pop('pad_contact'); payload.pop('pad_visual')
        phase = compile_request(ctx, ModeRequest(Mode.TRACK_HYBRID, payload), raw=raw,
                                twist_read=lambda: np.zeros(6))
        evidence = Evidence(0., 0.)
        original = evidence.update
        def update(**kwargs):
            result = original(**kwargs)
            result.valid = False
            result.reason = 'recorded_image_invalid'
            return result
        evidence.update = update
        clock = [time.monotonic()]
        obj = PadContactOuter(phase.outer, evidence=evidence, clock=lambda: clock[0])
        pose = kin.fk_pose(q) if pose is None else np.asarray(pose)
        obj.set_origin(pose)
        rail = 0.
        if injected_qdot is not None:
            # This is a KNOWN injected command, not an inferred recorded handoff.
            inner.begin_hybrid_episode(q, injected_qdot)
            obj.begin_hybrid_episode(kin.jacobian(q) @ injected_qdot, pose)
            rail = float(injected_qdot[0])
        previous_sample = clock[0]
        first = True
        cycle_index = 0
        replay_trace = []
        def cycle(row):
            nonlocal q, rail, previous_sample, first, cycle_index
            cycle_index += 1
            clock[0] = max(clock[0]+.001, previous_sample+row['dt_actual_s'])
            dt = clock[0]-previous_sample
            previous_sample = clock[0]
            first_sample = first
            sample_pose = pose if first else kin.fk_pose(q)
            first = False
            cmd = obj.sample(clock[0], sample_pose, row['f_ext'], contact=row['contact'],
                dt_actual=dt, sensor_age_s=.002, feedback_age_s=.002)
            assert not obj.last_facts['visual_task_valid']
            assert obj.last_facts['visual_request_rad_s'] == 0.
            step = inner.update(cmd, dt, q_meas=q, vel_ff=obj.last_vel_ff, pose_d=obj.last_pose_d,
                f_ext_z=row['f_ext'][2], f_des_z=4., contact_active=row['contact'],
                path_twist=obj.last_path_twist, feedback_twist=obj.last_feedback_twist,
                v_force_z=obj.controller.v_force_z, rail_exec_vel_m_s=rail,
                dt_wall_s=dt, auto_commit=False, commit_history=False, **obj.command_constraints())
            replay_trace.append(dict(cycle=cycle_index, sample_time_s=clock[0],
                                     dt_actual_s=dt, q=q.copy(), row=row.copy()))
            if step.fallback_reason:
                def json_value(value):
                    if isinstance(value, np.ndarray): return value.tolist()
                    if isinstance(value, np.generic): return value.item()
                    if hasattr(value, '__dict__'): return vars(value)
                    return repr(value)
                pending = {key: value for key, value in obj._pending.items()
                           if key not in ('result', 'mechanical')}
                certificate = obj._pending['result'].hard_constraints
                native = inner._native
                snapshot = dict(backend=backend, cycle=cycle_index, q=q, pose=sample_pose,
                    row=row, command=cmd, dt_actual_s=dt, rail_exec_vel_m_s=rail,
                    step=vars(step), pending=pending,
                    certificate=dict(A=certificate.A, lower=certificate.lower,
                                     upper=certificate.upper, labels=certificate.labels),
                    rocking=dict(time_s=obj.rocking.time_s, omega_base=obj.rocking.omega_base,
                                 acceleration_base=obj.rocking.acceleration_base),
                    last_facts=obj.last_facts, trace=replay_trace,
                    native_output=({name: native._out[name].copy()
                                    for name in native._out.dtype.names} if native else None),
                    native_diagnostics=native.timeout_diagnostics() if native else None)
                path = Path('/tmp') / f'pad_recorded_failure_{os.getpid()}_{cycle_index}.json'
                path.write_text(json.dumps(snapshot, default=json_value, indent=2)+'\n')
                pytest.fail(f'{step.fallback_reason}; cycle {cycle_index}; snapshot {path}')
            if first_sample and injected_qdot is not None:
                np.testing.assert_allclose(step.qdot_prev_used, injected_qdot, atol=1e-10)
            r = rotation_from_pose(sample_pose)
            final = np.r_[r.T@step.v_tcp_estimated[:3], r.T@step.v_tcp_estimated[3:]]
            # Source-to-allocation recorded latency plus injected 4 ms inner solve.
            # Keep the production 50 ms lease and let dispatch affect the NEXT dt.
            latency = row.get('recorded_compute_time_s', .004)+.004
            clock[0] += latency
            clock[0] += obj.dispatch_wait_s(clock[0])
            assert obj.review(final, now_s=clock[0]), obj.last_review_reason
            assert obj.started()
            assert obj.arm_dispatch_time(clock[0])
            elapsed = clock[0]-obj.rocking.time_s
            previous = float(r[:, 1] @ obj.rocking.omega_base)
            previous_a = float(r[:, 1] @ obj.rocking.acceleration_base)
            acceleration = (final[4]-previous)/elapsed
            assert abs(acceleration) <= obj.rocking.acceleration_limit+1e-6
            assert abs(acceleration-previous_a) <= obj.rocking.jerk_limit*elapsed+1e-6
            assert abs(final[4]) <= obj.rocking.velocity_limit+1e-8
            inner.commit_publication(step.qdot)
            obj.commit(final, now_s=clock[0]+.001)
            clock[0] += .001
            q = step.q_send.copy()
            rail = .3*step.qdot[0]+.7*rail
            return final
        yield obj, cycle
    finally:
        if obj is not None: obj.close()
        if inner._native: inner._native.shutdown()


@pytest.mark.parametrize('case', CONTACT_CASES, ids=lambda r:r['session'])
@pytest.mark.parametrize('backend', ['python', 'native'])
def test_recorded_contact_wrench_with_real_ik_and_dispatch_latency(case, backend):
    """Recorded wrench/contact/timing, then explicitly synthetic continuation."""
    rows = case['rows']
    with recorded_chain(backend, rows[0]['q_meas8']) as (obj, cycle):
        for row in rows:
            final = cycle(row)
        continuation = dict(rows[-1], dt_actual_s=.03)
        if case['session'] == '20260921_165013':
            # Preserve the recorded contact wrench until meaningful rocking,
            # then withdraw contact while publication history remains moving.
            for _ in range(250):
                if abs(final[4]) > .23:
                    break
                final = cycle(continuation)
            assert abs(final[4]) > .23
            dropped = dict(continuation, contact=False)
            before = abs(final[4])
            first_brake = cycle(dropped)
            assert 0. < abs(first_brake[4]) < before+.02
            for _ in range(160):
                final = cycle(dropped)
            assert abs(final[4]) < 1e-5
        else:
            for _ in range(25):
                cycle(continuation)


@pytest.mark.parametrize('backend', ['python', 'native'])
@pytest.mark.parametrize('direction', [0., 1., -1.], ids=['zero', 'positive', 'negative'])
@pytest.mark.parametrize('first_sample_dt', [.005, .025], ids=['fast', 'slow'])
def test_165023_raw_entry_with_known_injected_handoff(backend, direction, first_sample_dt):
    """Raw H5 pose/wrench are recorded; previous applied velocity is NOT."""
    case = next(case for case in CONTACT_1650 if case['session'] == '20260921_165023')
    assert case['missing_handoff'] and not case['rows']
    entry = case['entry_snapshot']
    q = np.asarray(entry['q_meas8'])
    kin = RobotKinematics()
    kin.apply_link7_to_tcp_offset(CASES[0]['tcp_offset_pose'])
    rotation = rotation_from_pose(entry['pose_m_rad'])
    requested = np.r_[np.zeros(3), rotation @ np.array([0., direction*.02, 0.])]
    injected_qdot = np.linalg.lstsq(kin.jacobian(q), requested, rcond=None)[0]
    np.testing.assert_allclose(kin.jacobian(q) @ injected_qdot, requested, atol=1e-10)
    row = dict(f_ext=entry['f_ext_force_aligned'], contact=False,
               dt_actual_s=.02, recorded_compute_time_s=.004)
    with recorded_chain(backend, q, pose=entry['pose_m_rad'],
                        injected_qdot=injected_qdot) as (obj, cycle):
        final = cycle(dict(row, dt_actual_s=first_sample_dt))
        for _ in range(59):
            final = cycle(row)
        assert abs(final[4]) < 1e-5
