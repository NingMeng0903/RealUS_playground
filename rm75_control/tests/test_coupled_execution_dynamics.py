"""Offline execution regressions; model claims stay separate from hardware claims."""
import numpy as np
import pytest

from .test_strict_hqp_rail_compensation import _controller, Q_SAFE
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import RailMode
from rm75_control.control.joint_admittance_8dof.tasks.rail_command import RailCommandMixer
from rm75_control.control.joint_admittance_8dof.tasks.rail_allocator import RailReferenceModel
from rm75_control.control.joint_admittance_8dof.tasks.execution_observer import ActuatorModel, ExecutionObserver
from rm75_control.control.joint_admittance_8dof.qp_cert import inbox_brake


def test_final_limit_and_publication_history_use_original_velocity():
    model = RailReferenceModel(f_c_hz=0, a_max=1, j_max=1000, v_max=.102)
    model.reset(.1)
    model.state.initialized = True
    model.state.a = 1
    model.step(.2, .005, x_m=.4, apply_wall=False)
    assert model.state.v == pytest.approx(.102)
    assert model.state.a == pytest.approx(.4)
    model.commit(.101, .005)
    assert model.state.a == pytest.approx(.2)
    # A limit/fault override has truthful history even if it breaks a smooth limit.
    model.track(0, .005)
    assert model.state.a == pytest.approx(-20.2)


def test_filtered_feedforward_reversal_is_not_integrated_as_posture():
    mix = RailCommandMixer()
    total, base = RailReferenceModel(), RailReferenceModel()
    mix.reset(.25)
    dt = .005
    for tick in range(2800):
        ff = .02 if tick < 2600 else -.02
        tel = mix.step(d_live=.25, d_star_target=.25, u_task_raw=ff,
                       u_escape_raw=0, escape_explicit=False, dt=dt, u_max=.12,
                       defer_commit=True)
        v = total.step(tel.u_feasible, dt, x_m=.4)
        b = base.step(tel.u_base, dt, x_m=.4)
        mix.commit_final(v, b, dt)
        total.commit(v, dt)
        assert abs(mix.xi) < 1e-12
        assert tel.u_post_committed == pytest.approx(0, abs=1e-12)
        xi = mix.xi
        mix.commit_final(99, -99, dt)  # one integration per committed tick
        assert mix.xi == xi


def test_qp_derating_of_filtered_feedforward_cannot_create_pi_cancellation():
    mix = RailCommandMixer()
    mix.reset(.25)
    total, base = RailReferenceModel(), RailReferenceModel()
    for _ in range(2600):
        mix.step(d_live=.25, d_star_target=.25, u_task_raw=.04,
                 u_escape_raw=0, escape_explicit=False, dt=.005, u_max=.12,
                 defer_commit=True)
        t = total.step(mix.last.u_feasible, .005, x_m=.4)
        b = base.step(mix.last.u_base, .005, x_m=.4)
        final = min(t, .001)
        tel = mix.commit_final(final, b, .005, total_shaped=t, brake=0)
        base.commit(tel.u_base_committed, .005)
        total.commit(final, .005)
        assert tel.u_post_committed == pytest.approx(0, abs=1e-12)
        assert tel.u_base_committed == pytest.approx(final)
        assert mix.xi == pytest.approx(0, abs=1e-12)


def test_reference_and_pi_share_fractional_authority():
    mix = RailCommandMixer(kp=1, ki=1, kaw=4, d_center_rate=.02)
    mix.reset(.25)
    tel = mix.step(d_live=.25, d_star_target=.35, u_task_raw=0, u_escape_raw=0,
                   escape_explicit=False, dt=.005, u_max=.12, secondary_alpha=0,
                   defer_commit=True)
    assert tel.d_star_ref == .25 and tel.d_star_dot_cmd == 0
    mix.commit_final(0, 0, .005)
    tel = mix.step(d_live=.25, d_star_target=.35, u_task_raw=0, u_escape_raw=0,
                   escape_explicit=False, dt=.005, u_max=.12, secondary_alpha=.25,
                   defer_commit=True)
    assert tel.d_star_dot_cmd == pytest.approx(.005)
    before = mix.xi
    mix.commit_final(tel.u_feasible, tel.u_base, .005)
    expected = before + .005 * (.25 * mix.ki * tel.e_d + mix.kaw *
                               (tel.u_mid_applied - .25 * tel.u_pi_raw))
    assert mix.xi == pytest.approx(expected)


@pytest.mark.parametrize('dof', [7, 8])
def test_continuous_z_reaches_full_progress_without_jerk_dead_end(dof):
    ctrl = _controller()
    ctrl.set_rail_mode(RailMode.COUPLED if dof == 8 else RailMode.LOCKED)
    ctrl.posture_retarget = None  # fixed d*: isolate task feedforward
    q = Q_SAFE.copy()
    for _ in range(100):
        previous_v = float(ctrl.core.qdot_prev[0])
        step = ctrl.update(np.array([0, 0, .008, 0, 0, 0]), .005, q_meas=q,
                           rail_exec_vel_m_s=float(ctrl.core.qdot_prev[0]))
        assert not step.task_paused, step.task_pause_reason
        assert step.u_alloc == pytest.approx(0, abs=1e-12)
        np.testing.assert_allclose(step.v_tcp_estimated, step.v_cmd_feasible, atol=1e-5)
        assert ctrl.core.validate_final_qdot(step.qdot)[0] < 1e-5
        if dof == 8:
            lo, hi = ctrl.rail_ref_model.last_command_bounds
            assert lo - 1e-5 <= step.qdot[0] <= hi + 1e-5
            assert step.rail_ref_acceleration == pytest.approx(
                (step.qdot[0] - previous_v) / .005, abs=1e-7)
        q = step.q_send.copy()
    assert step.task_progress > .99


def test_long_z_seek_then_small_chirp_has_no_allocation_cancelling_integral():
    ctrl = _controller()
    ctrl.set_rail_mode(RailMode.COUPLED)
    ctrl.posture_retarget = None
    q = Q_SAFE.copy()
    peak_integral = 0.0
    for i in range(2800):
        # A fixed Jacobian was not sufficient to catch the warm-start issue;
        # run the actual kinematics and absolute-position command integration.
        vz = -.008 if i < 2600 else .0004 * np.sin((i - 2600) * .04)
        step = ctrl.update(np.array([0, 0, vz, 0, 0, 0]), .005, q_meas=q,
                           rail_exec_vel_m_s=float(ctrl.core.qdot_prev[0]))
        q = step.q_send.copy()
        assert abs(step.u_alloc) < 1e-12
        peak_integral = max(peak_integral, abs(ctrl.rail_mixer.xi))
        if i < 2600:
            assert not step.task_paused
        if i == 2599:
            dz = ctrl.kin.fk_pose(q)[2] - ctrl.kin.fk_pose(Q_SAFE)[2]
            assert -.106 < dz < -.100
        if not step.task_paused:
            np.testing.assert_allclose(step.v_tcp_estimated, step.v_cmd_feasible, atol=1e-5)
    # This is an offline regression threshold, not a hardware error budget.
    assert peak_integral < 5e-4
    assert abs(ctrl.rail_mixer.xi) < 5e-4


def test_dstar_motion_retains_nonzero_rail_and_protected_tcp():
    ctrl = _controller()
    ctrl.set_rail_mode(RailMode.COUPLED)
    q = Q_SAFE.copy()
    travel = 0.0
    for _ in range(80):
        step = ctrl.update(np.zeros(6), .005, q_meas=q,
                           rail_exec_vel_m_s=float(ctrl.core.qdot_prev[0]))
        assert not step.task_paused
        np.testing.assert_allclose(step.v_tcp_estimated, 0, atol=1e-5)
        travel += abs(step.qdot[0]) * .005
        q = step.q_send.copy()
    assert travel > 1e-4  # a permanently locked rail would fail this test


def test_slow_rail_and_delayed_arm_expose_leak_and_stop_tail_in_observer():
    dt = .005
    observer = ExecutionObserver([ActuatorModel(.05, .02, .9)] * 7,
                                 ActuatorModel(.02, .04))
    observer.reset(0, np.zeros(8))
    J = np.zeros((6, 8)); J[1, 0] = J[1, 1] = 1
    arm_position = 0.0
    seq, rail_written = 0, 0.0
    residual, tails = [], []
    for i in range(1, 601):
        t = i * dt
        desired = .01 if t < 1 else (-.01 if t < 2 else 0.)
        # 50 Hz worker, including skipped writes; ARM remains 200 Hz.
        if i % 4 == 0 and desired != rail_written:
            seq += 1; rail_written = desired
            observer.record_rail_write(t, seq, rail_written)
        arm_position -= rail_written * dt
        observer.record_arm_send(t, np.r_[0, arm_position, np.zeros(6)])
        value = observer.sample(t, J)[1]
        residual.append(value)
        if 2 < t < 2.1:
            tails.append(value)
    assert max(abs(np.asarray(residual))) > .002
    assert max(abs(np.asarray(tails))) > .001
    assert abs(residual[-1]) < 1e-6
    assert observer.mode == 'observe'


def test_brake_does_not_command_sign_reversal_when_zero_is_reachable():
    out = inbox_brake([.001, -.001], [-.1, -.1], [.1, .1], [1, 1], .005)
    np.testing.assert_array_equal(out, [0, 0])


def test_closed_loop_low_rate_rail_delayed_arm_preserves_model_contract():
    """Hardware lag remains visible; it cannot become a false QP certificate."""
    ctrl = _controller()
    ctrl.set_rail_mode(RailMode.COUPLED)
    ctrl.posture_retarget = None
    q = Q_SAFE.copy()
    plant = ExecutionObserver([ActuatorModel(.035, .015, .97)] * 7,
                              ActuatorModel(.015, .025))
    plant.reset(0, q)
    observed, rail, progress = [], [], []
    for tick in range(1, 401):
        now = tick * .005
        J = ctrl.kin.jacobian(q)
        actual_twist = plant.sample(now, J)
        q[0] += plant.channels[7].value * .005
        q[1:] = [c.value for c in plant.channels[:7]]
        requested = np.array([0, .004 if tick < 250 else 0, 0, 0, 0, 0])
        step = ctrl.update(requested, .005, q_meas=q.copy(),
                           rail_exec_vel_m_s=plant.channels[7].value,
                           rail_refresh_dt_s=.02)
        plant.record_arm_send(now, step.q_send)
        if tick % 4 == 0:
            plant.record_rail_write(now, tick // 4, float(step.qdot[0]))
        if not step.task_paused:
            np.testing.assert_allclose(step.v_tcp_estimated,
                                       step.v_cmd_feasible, atol=2e-5)
        assert np.isfinite(step.q_send).all()
        observed.append(actual_twist)
        rail.append(q[0])
        progress.append(step.task_progress)
    assert max(progress[:250]) > .9
    assert np.ptp(rail) > 1e-4
    assert np.isfinite(observed).all()
    # An observation model never silently enables a compensation controller.
    assert ctrl.execution_observer is None and plant.mode == 'observe'
