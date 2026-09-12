"""Offline fixed rocking row admission: task smoothness yields to P0."""
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.rocking_envelope import validate_rocking, rocking_interval

Q = np.array([.375, *np.deg2rad([-89.5, -94.5, 65.2, 96., 89.3, 61., 94.6])])
AXIS = np.array([0., 1., 0.])
CASES = [
    ([-.28, .28, -.03, .03, -.001, .001], 1),
    ([-.28, .28, -.03, .03, .08, .09], 2),
    ([-.28, .28, .5, .6, .51, .52], 3),
    ([100., 101., 100., 101., 100., 101.], 4),
]


def controller(backend):
    raw = yaml.safe_load((Path(__file__).parents[1] / 'configs/joint_admittance_8dof.yaml').read_text())
    cfg = build_joint_ik_config(raw)
    cfg.backend = backend
    cfg.control_frame = 'base'
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    cfg.native_shm_prefix = f'rocking_test_{os.getpid()}_{backend}'
    obj = JointIkController(RobotKinematics(), cfg)
    if obj._native is not None:
        obj._native.timeout_s = .5
    obj.reset(Q)
    return obj


def close(obj):
    if obj._native is not None:
        obj._native.shutdown()


def test_fixed_contract_and_empty_jerk_interval():
    axis, bounds = validate_rocking(AXIS, [-.2, .2, -.1, .1, .05, -.05])
    assert rocking_interval(bounds, 1) == (.05, -.05)
    assert rocking_interval(bounds, 2) == (-.1, .1)
    assert rocking_interval(bounds, 3) == (-.2, .2)
    with pytest.raises(ValueError, match='unit'):
        validate_rocking([0, 2, 0], bounds)
    with pytest.raises(ValueError, match='together'):
        validate_rocking(AXIS, None)


@pytest.mark.parametrize('bounds,tier', CASES)
def test_python_core_relaxes_only_task_row(bounds, tier):
    obj = controller('python')
    try:
        r = obj.core.step(Q, np.array([0., 0., 0., 0., .2, 0.]), .005,
                          q_meas=Q, rail_exec_vel_m_s=.015,
                          rocking_axis_base=AXIS, rocking_bounds=bounds)
        assert not obj.core.last_failed
        assert obj.core.last_rocking_policy_tier == tier
        assert obj.core.last_rocking_limited == (tier > 1)
        qdot = obj.core.last_qdot_qp1
        assert np.all(qdot >= obj.core.last_lo_box - 1e-5)
        assert np.all(qdot <= obj.core.last_hi_box + 1e-5)
        omega = float(AXIS @ (obj.kin.jacobian(Q)[3:] @ qdot))
        lo, hi = rocking_interval(bounds, tier)
        assert lo - 1e-5 <= omega <= hi + 1e-5
        hard, _ = obj.core.validate_final_qdot(qdot)
        assert hard < 1e-5
    finally:
        close(obj)


@pytest.mark.parametrize('bounds,tier', CASES)
def test_native_python_fixed_envelope_parity(bounds, tier):
    for backend in ('python', 'native'):
        obj = controller(backend)
        try:
            step = obj.update(np.array([0., 0., 0., 0., .2, 0.]), .005,
                              q_meas=Q, rail_exec_vel_m_s=.015, dt_wall_s=.005,
                              rocking_axis_base=AXIS, rocking_bounds=bounds)
            assert not step.solver_fault_latched, (backend, step.fallback_reason)
            assert step.rocking_policy_tier == tier
            assert step.rocking_limited == (tier > 1)
            final = obj.kin.jacobian(Q) @ step.qdot
            omega = float(AXIS @ final[3:])
            lo, hi = rocking_interval(bounds, tier)
            assert lo - 1e-5 <= omega <= hi + 1e-5
            assert step.rocking_lower_rad_s == lo
            assert step.rocking_upper_rad_s == hi
            # The rail is prismatic: even the final Python rail rewrite has
            # exactly zero angular contribution under this Jacobian snapshot.
            changed = step.qdot.copy(); changed[0] += .1
            assert np.allclose((obj.kin.jacobian(Q) @ changed)[3:], final[3:], atol=1e-12)
        finally:
            close(obj)
    # Parity is the chosen admission policy and certified envelope; the
    # backends retain their existing different secondary objectives/limiters.


def test_mechanics_only_matches_disabled_backend():
    for backend in ('python', 'native'):
        outputs = []
        for enabled in (False, True):
            obj = controller(backend)
            try:
                kw = ({'rocking_axis_base': AXIS, 'rocking_bounds': CASES[-1][0]}
                      if enabled else {})
                step = obj.update(np.array([0., 0., 0., 0., .2, 0.]), .005,
                                  q_meas=Q, rail_exec_vel_m_s=.015, dt_wall_s=.005, **kw)
                assert not step.solver_fault_latched
                assert step.rocking_policy_tier == (4 if enabled else 0)
                outputs.append(step.qdot.copy())
            finally:
                close(obj)
        np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-7, rtol=1e-5)


def test_collision_priority_relaxes_jerk_and_keeps_cbf(monkeypatch):
    from rm75_control.control.joint_admittance_8dof.solver import qp_builder as builder
    obj = controller('python')
    try:
        row = (AXIS @ obj.kin.jacobian(Q)[3:]).reshape(1, 8)
        monkeypatch.setattr(builder, 'build_cbf_rows', lambda *a, **kw:
                            builder.CbfRows(jacobian=row, lower=np.array([.005])))
        obj.core.collision = type('CollisionStub', (), {'closest_pair': lambda self: None})()
        obj.core.collision_cfg.enabled = True
        obj.core.step(Q, np.array([0., 0., 0., 0., .2, 0.]), .005,
                      q_meas=Q, rocking_axis_base=AXIS, rocking_bounds=CASES[0][0])
        assert not obj.core.last_failed
        assert obj.core.last_rocking_policy_tier == 2
        omega = float((row @ obj.core.last_qdot_qp1)[0])
        assert .005 - 1e-5 <= omega <= .03 + 1e-5
        hard, _ = obj.core.validate_final_qdot(obj.core.last_qdot_qp1)
        assert hard < 1e-5
    finally:
        close(obj)


def test_final_mechanical_rewrite_relaxes_only_task_metadata():
    obj = controller('python')
    try:
        obj.core.step(Q, np.array([0., 0., 0., 0., .2, 0.]), .005,
                      q_meas=Q, rocking_axis_base=AXIS, rocking_bounds=CASES[0][0])
        core = obj.core
        history = core.qdot_prev.copy()
        row = core._last_rocking_row.copy()
        candidate = .04 * row / float(row @ row)
        candidate_before = candidate.copy()
        box_before = (core.last_lo_box.copy(), core.last_hi_box.copy())
        assert core.relax_rocking_for_final_qdot(candidate) == 3
        assert core.last_rocking_limited
        assert core.last_rocking_lower_rad_s == -.28
        np.testing.assert_array_equal(candidate, candidate_before)
        np.testing.assert_array_equal(core.qdot_prev, history)
        np.testing.assert_array_equal(core.last_lo_box, box_before[0])
        np.testing.assert_array_equal(core.last_hi_box, box_before[1])
        # Relaxation does not certify or excuse any mechanical violation.
        unsafe = core.last_hi_box + 1.
        core.relax_rocking_for_final_qdot(unsafe)
        hard, _ = core.validate_final_qdot(unsafe)
        assert hard >= 1. - 1e-9
        tier = core.last_rocking_policy_tier
        assert core.relax_rocking_for_final_qdot(np.full(8, np.nan)) == tier
        assert np.isinf(core.validate_final_qdot(np.full(8, np.nan))[0])
    finally:
        close(obj)


def test_real_native_aborted_late_reply_uses_fresh_frame_and_envelope(monkeypatch):
    from rm75_control.control.joint_admittance_8dof.wbc_rt import protocol as P
    obj = controller('native')
    try:
        native = obj._native
        wait = native._wait_seq
        # Force a soft deadline miss without delaying or changing the real
        # native solve. It produces an actual, uncommitted late proposal.
        monkeypatch.setattr(native, '_wait_seq', lambda seq, **kw: False)
        old_twist = np.array([0., 0., 0., 0., .2, 0.])
        first = obj.update(old_twist, .005, q_meas=Q, auto_commit=False,
                           rocking_axis_base=AXIS, rocking_bounds=CASES[0][0])
        assert first.fallback_reason == 'native_timeout_coast'
        old_seq = native._inflight_seq
        native.abort_pending()
        monkeypatch.setattr(native, '_wait_seq', wait)
        assert wait(old_seq, timeout_s=.5)
        q_before = obj.q_cmd.copy()
        accepted = []
        accept = native._accept_ok_step
        def accept_fresh(twist, seq, **kw):
            assert seq != old_seq
            accepted.append(seq)
            return accept(twist, seq, **kw)
        monkeypatch.setattr(native, '_accept_ok_step', accept_fresh)
        monkeypatch.setattr(native, '_sync_q', lambda: pytest.fail('prepare cannot synchronize q'))
        new_axis = np.array([1., 0., 0.])
        new_bounds = [-.15, .15, -.02, .02, -.0005, .0005]
        new_twist = np.array([0., 0., 0., .15, 0., 0.])
        fresh = obj.update(new_twist, .005, q_meas=Q, auto_commit=False,
                           rocking_axis_base=new_axis, rocking_bounds=new_bounds)
        assert accepted == [old_seq + 1]
        assert not fresh.solver_fault_latched
        assert int(native._in['flags'][0]) & P.IN_ABORT_PREV
        assert not int(native._in['flags'][0]) & P.IN_COMMIT_PREV
        np.testing.assert_allclose(fresh.v_cmd_received, new_twist)
        np.testing.assert_array_equal(obj.q_cmd, q_before)
        np.testing.assert_array_equal(native._in['rocking_axis_base'][0], new_axis)
        np.testing.assert_array_equal(native._in['rocking_bounds'][0], new_bounds)
        np.testing.assert_allclose(native._out['qdot_prev'][0], np.zeros(8), atol=1e-12)
        assert native._pending_commit_seq == old_seq + 1
        omega = float(new_axis @ (obj.kin.jacobian(Q) @ fresh.qdot)[3:])
        assert fresh.rocking_lower_rad_s - 1e-5 <= omega <= fresh.rocking_upper_rad_s + 1e-5
    finally:
        close(obj)
