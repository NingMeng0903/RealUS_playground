"""Dispatch latency, speed saturation and reversal without robot I/O."""
import numpy as np
import pytest
import json
from pathlib import Path

from peirastic.contact_qp.rocking_smoothing import RockingSmoothing
from peirastic.realman8dof.modes.pad_publication import dispatch_rocking_bounds
from peirastic.tests.test_pad_contact_outer import outer


@pytest.mark.parametrize('acceleration', [-1.8, -.8, 0., .8, 1.8])
def test_window_bounds_cover_interior_extrema_and_speed_braking(acceleration):
    history = RockingSmoothing(.28, 2., .24, 1.)
    history.seed([0., .1, 0.], 0.)
    history.acceleration_base[1] = acceleration
    _, bounds = dispatch_rocking_bounds(history, np.eye(3), .02, .045)
    low, high = max(bounds[::2]), min(bounds[1::2])
    for t in np.linspace(.02, .045, 501):
        _, instantaneous, _ = history.preview(np.eye(3), t, 1e-9)
        assert low >= max(instantaneous[::2])-1e-12
        assert high <= min(instantaneous[1::2])+1e-12


def scheduled_outer():
    obj, pose, clock = outer(0., 0.)
    obj.settings.update(minimum_publication_period_s=.020)
    del obj.__dict__['_dispatch_times']
    # No image correction, matching the four 16:22 episodes.
    original = obj.pad_visual.update
    def mechanical_only(**kwargs):
        result = original(**kwargs)
        result.valid = False
        return result
    obj.pad_visual.update = mechanical_only
    return obj, pose, clock


def dispatch(obj, cmd, clock, fraction):
    p = obj._pending
    # Actual compute jitter; the certificate itself covers the full 50 ms
    # lifetime, not a fabricated 2 ms acceptance window.
    clock[0] = p['dispatch_earliest_s']+fraction*.012
    assert obj.review(cmd, now_s=clock[0])
    assert obj.started()
    _, bounds, h = obj.rocking.preview(p['rotation'], clock[0], 1e-9)
    assert max(bounds[::2])-1e-8 <= cmd[4] <= min(bounds[1::2])+1e-8
    a = (cmd[4]-h['previous_rad_s'])/h['elapsed_s']
    assert abs(a) <= 2.+1e-6
    assert abs(a-h['previous_acceleration_rad_s2'])/h['elapsed_s'] <= obj.rocking.jerk_limit+1e-5
    obj.commit(cmd, now_s=clock[0]+.01)


def test_sustained_mechanical_rotation_reaches_cap_and_reverses_with_variable_latency():
    obj, pose, clock = scheduled_outer()
    previous_sample = clock[0]
    positive_peak = negative_peak = 0.
    for i in range(220):
        # Includes fast retained-rail ACKs and ordinary 50 Hz rail responses.
        clock[0] += [.003, .02, .026][i % 3]
        dt = min(.049, clock[0]-previous_sample)
        previous_sample = clock[0]
        cmd = obj.sample(clock[0], pose, [0.,0.,4.,0.,-.12 if i < 90 else .12,0.],
            contact=True, dt_actual=dt, sensor_age_s=.001, feedback_age_s=.001)
        dispatch(obj, cmd, clock, (i % 5)/5.)
        positive_peak = max(positive_peak, cmd[4]); negative_peak = min(negative_peak, cmd[4])
        # This test isolates rate/accel/jerk saturation from the separate
        # integrated angle guard; real tests cover the angle stop separately.
        obj.force_law.tilt.theta_tilt = 0.
    assert positive_peak > .275
    assert negative_peak < -.275


def test_expired_dispatch_window_never_commits():
    obj, pose, clock = scheduled_outer()
    clock[0] += .02
    cmd = obj.sample(clock[0], pose, [0.,0.,4.,0.,-.08,0.],
        contact=True, dt_actual=.02, sensor_age_s=.001, feedback_age_s=.001)
    clock[0] = obj._pending['dispatch_latest_s']+.0001
    assert not obj.review(cmd, now_s=clock[0])
    assert not obj.started()
    obj.abort('missed_window', definitely_not_sent=True)
    assert obj._last_commit_s is None


def test_angle_certificate_covers_late_arm_admission():
    obj, pose, clock = scheduled_outer()
    # Establish contact before setting an angle close to the limit; a newly
    # observed contact intentionally resets the mechanical angle reference.
    clock[0] += .025
    cmd = obj.sample(clock[0], pose, [0.,0.,4.,0.,0.,0.], contact=True,
        dt_actual=.025, sensor_age_s=.001, feedback_age_s=.001)
    dispatch(obj, cmd, clock, 0.)
    obj.force_law.tilt.theta_tilt = obj.solver.config.angle_limit_rad-1e-5
    clock[0] += .025
    cmd = obj.sample(clock[0], pose, [0.,0.,4.,0.,-.12,0.], contact=True,
        dt_actual=.025, sensor_age_s=.001, feedback_age_s=.001)
    p = obj._pending
    gap = p['dispatch_latest_s']-obj.rocking.time_s
    assert p['theta']+cmd[4]*gap <= obj.solver.config.angle_limit_rad+1e-8
    clock[0] = p['dispatch_latest_s']
    assert obj.review(cmd, now_s=clock[0]) and obj.started()
    assert obj.arm_dispatch_time(clock[0])
    obj.commit(cmd, now_s=clock[0]+.01)
    assert abs(obj.force_law.tilt.theta_tilt) <= obj.solver.config.angle_limit_rad+1e-8


def test_review_rejects_ik_error_at_interior_jerk_extremum():
    obj, pose, clock = scheduled_outer()
    previous_sample = clock[0]
    for _ in range(3):
        clock[0] += .02
        dt = clock[0]-previous_sample; previous_sample = clock[0]
        cmd = obj.sample(clock[0], pose, [0.,0.,4.,0.,-.12,0.], contact=True,
            dt_actual=dt, sensor_age_s=.001, feedback_age_s=.001)
        dispatch(obj, cmd, clock, 0.)
    clock[0] += .02
    cmd = obj.sample(clock[0], pose, [0.,0.,4.,0.,-.12,0.], contact=True,
        dt_actual=.02, sensor_age_s=.001, feedback_age_s=.001)
    p = obj._pending
    acceleration = float(p['axis']@obj.rocking.acceleration_base)
    vertex = obj.rocking.time_s+acceleration/(2*obj.rocking.jerk_limit)
    assert p['dispatch_earliest_s'] < vertex < p['dispatch_latest_s']
    final = cmd.copy(); final[4] = max(p['bounds'][::2])-5e-6
    clock[0] = p['dispatch_earliest_s']
    # Generic Cartesian tolerance formerly accepted this at the early review,
    # although admission at the interior vertex violated the stricter jerk.
    assert not obj.review(final, now_s=clock[0])
    assert obj.last_review_reason == 'final_dispatch_rocking_limit'
    obj.abort('test', definitely_not_sent=True)


def test_moving_handoff_and_single_angle_commit_use_final_arm_timestamp():
    from peirastic.realman8dof.force.torque_tilt import rotation_from_pose
    obj, pose, clock = scheduled_outer()
    r = rotation_from_pose(pose)
    applied = np.r_[np.zeros(3), r@np.array([0., .04, 0.])]
    obj.begin_hybrid_episode(applied, pose)
    handoff = clock[0]
    assert obj.rocking.time_s == handoff
    # First sample has only 1 ms history, despite a nominal 5 ms period.
    clock[0] += .001
    cmd = obj.sample(clock[0], pose, [0.,0.,4.,0.,-.12,0.],
        contact=True, dt_actual=.005, sensor_age_s=.001, feedback_age_s=.001)
    assert obj.rocking.time_s == handoff
    origin = obj.force_law.tilt.theta_tilt
    # Model a small accepted IK change inside the certified interval.
    p = obj._pending
    final = cmd.copy()
    final[4] = .5*(max(p['bounds'][::2])+min(p['bounds'][1::2]))
    final[2] = cmd[2]  # COP invalid in this torque sample, no geometric shift.
    clock[0] = p['dispatch_earliest_s']
    assert obj.review(final, now_s=clock[0]) and obj.started()
    clock[0] += .001
    assert obj.arm_dispatch_time(clock[0])
    arm_time = clock[0]
    obj.commit(final, now_s=arm_time+.025)
    assert obj.rocking.time_s == arm_time
    assert obj.force_law.tilt.theta_tilt == pytest.approx(origin+final[4]*(arm_time-handoff))
    assert obj.force_law.tilt._pending_command is None


def test_production_windows_remain_feasible_near_caps_in_rotated_frame():
    # Follow reachable histories, including both extrema of dispatch jitter.
    angle = .73
    r = np.array([[np.cos(angle), -np.sin(angle), 0.],
                  [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    history = RockingSmoothing(.28, 2., .051, .22)
    history.seed(r@np.array([0., -.27, 0.]), 0.)
    clock = 0.
    for i in range(700):
        early = clock+[.02, .028, .034][i % 3]
        latest = early+[.008, .027, .050][i % 3]
        _, b = dispatch_rocking_bounds(history, r, early, latest, minimum_period_s=.02)
        lo, hi = max(b[::2]), min(b[1::2])
        assert lo <= hi+1e-12
        desired = .28 if (i//100) % 2 == 0 else -.28
        velocity = float(np.clip(desired, lo, hi))
        for t in np.linspace(early, latest, 9):
            _, point, _ = history.preview(r, t, 1e-9)
            assert max(point[::2])-1e-12 <= velocity <= min(point[1::2])+1e-12
        clock = early if i % 2 == 0 else latest
        history.commit([0.,0.,0.,0.,velocity,0.], r, clock)


@pytest.mark.parametrize('delay', [.010742, .012268, .010367, .028])
def test_recorded_computation_delays_accept_safe_force_command(delay):
    obj, pose, clock = scheduled_outer()
    previous_sample = clock[0]
    for _ in range(12):
        clock[0] += .016
        start = clock[0]
        dt = start-previous_sample; previous_sample = start
        cmd = obj.sample(start, pose, [0.,0.,0.,0.,.02,0.], contact=False,
            dt_actual=dt, sensor_age_s=.003, feedback_age_s=.003)
        # These first three delays caused 165005/17/20 to exit under the old
        # arbitrary [8,10] ms schedule, while their physical omega was zero.
        clock[0] = start+delay
        assert obj.review(cmd, now_s=clock[0]) and obj.started()
        assert obj.arm_dispatch_time(clock[0])
        obj.commit(cmd, now_s=clock[0]+.02)


@pytest.mark.parametrize('omega', [-.24, -.005, .005, .24])
def test_uncontacted_moving_entry_brakes_instead_of_conflicting_with_guard(omega):
    from peirastic.realman8dof.force.torque_tilt import rotation_from_pose
    obj, pose, clock = scheduled_outer()
    r = rotation_from_pose(pose)
    applied_tool = np.array([.02, -.018, .008, 0., omega, 0.])
    obj.begin_hybrid_episode(np.r_[r@applied_tool[:3], r@applied_tool[3:]], pose)
    previous = clock[0]
    for i in range(60):
        clock[0] += .025
        dt = clock[0]-previous; previous = clock[0]
        cmd = obj.sample(clock[0], pose, [0.,0.,0.,0.,0.,0.], contact=False,
            dt_actual=dt, sensor_age_s=.003, feedback_age_s=.003)
        assert obj.last_facts['guarded_braking']
        assert not obj.last_facts['visual_task_valid']
        dispatch(obj, cmd, clock, .2)
    assert abs(cmd[4]) < 1e-8
    assert np.max(np.abs(cmd[:2])) < 1e-8


def test_contact_drop_at_recorded_rotation_brakes_with_final_jerk_limit():
    obj, pose, clock = scheduled_outer()
    # Reach a rotating contact state using the new policy, then lose contact.
    previous_sample = clock[0]
    for i in range(140):
        clock[0] += .026
        dt = clock[0]-previous_sample; previous_sample = clock[0]
        # Force/tilt keep the 5 ms hold even when this fixture spaces
        # publishes at 26 ms, so rocking takes more ticks than a 26 ms
        # integrator to reach the same speed.
        contact = i < 55
        cmd = obj.sample(clock[0], pose, [0.,0.,2.8 if contact else 0.,0.,-.1,0.],
            contact=contact, dt_actual=dt, sensor_age_s=.003, feedback_age_s=.003)
        dispatch(obj, cmd, clock, .3)
        if i == 54: assert cmd[4] > .23
        if not contact: assert obj.last_facts['guarded_braking']
    assert abs(cmd[4]) < 1e-8


RECORDED = json.loads((Path(__file__).parent/'data/pad_contact_1622.json').read_text())['sessions']


@pytest.mark.parametrize('session', RECORDED, ids=lambda s:s['session'])
def test_recorded_contact_force_and_pose_with_jittered_dispatch(session):
    obj, _, clock = scheduled_outer()
    rows = session['rows']
    obj.set_origin(rows[0]['pose_m_rad'])
    previous_sample = clock[0]
    for i, row in enumerate(rows):
        # Replay source observations; move dispatch inside the new certified
        # window. Recorded image feedback cannot represent the new movement.
        clock[0] = max(clock[0]+.001, previous_sample+row['dt_actual_s'])
        dt = clock[0]-previous_sample
        previous_sample = clock[0]
        cmd = obj.sample(clock[0], row['pose_m_rad'], row['f_ext'],
            contact=row['contact'], dt_actual=dt, sensor_age_s=.003, feedback_age_s=.003)
        dispatch(obj, cmd, clock, (i % 4)/4.)
    # Extend the last contact torque so saturation, not merely startup, is
    # exercised. This was missing from the original no-contact regression.
    for i in range(40):
        clock[0] += .02
        dt = clock[0]-previous_sample
        previous_sample = clock[0]
        cmd = obj.sample(clock[0], rows[-1]['pose_m_rad'], rows[-1]['f_ext'],
            contact=rows[-1]['contact'], dt_actual=dt, sensor_age_s=.003, feedback_age_s=.003)
        dispatch(obj, cmd, clock, (i % 4)/4.)
