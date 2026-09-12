"""Recorded /001 budget failures and full commanded-rail admission, no hardware."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.command_power import final_command_qdot, validate_command_power
from peirastic.contact_qp.command_budget import CommandBudget
from peirastic.contact_qp.port_constraint import PortEnergyConstraint

CASES = json.loads((Path(__file__).parents[2] / 'analysis_artifacts/contact_execution_20260912/session001_energy_failures.json').read_text())
Q = np.array([.375, *np.deg2rad([-89.5, -94.5, 65.2, 96., 89.3, 61., 94.6])])


def controller(backend):
    raw = yaml.safe_load((Path(__file__).parents[1] / 'configs/joint_admittance_8dof.yaml').read_text())
    cfg = build_joint_ik_config(raw)
    cfg.backend = backend
    cfg.control_frame = 'base'
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    cfg.native_shm_prefix = f'power_test_{os.getpid()}_{backend}'
    obj = JointIkController(RobotKinematics(), cfg)
    if obj._native is not None:
        obj._native.timeout_s = .5
    obj.reset(Q)
    return obj


def close(obj):
    if obj._native is not None:
        obj._native.shutdown()


def test_rail_rebase_does_not_spend_energy_but_real_clamp_does():
    qdot = np.arange(8.) * .001
    qdot[0] = .01
    # Current observer base .25 differs from previous command .251 by 1 mm.
    proposal = .25 + .005 * qdot[0]
    result = final_command_qdot(qdot, proposed_rail_m=proposal, published_rail_m=proposal, dt_s=.005)
    np.testing.assert_array_equal(result, qdot)
    assert (proposal-.251)/.005 == pytest.approx(-.19)  # old fictitious rate
    clamped = final_command_qdot(qdot, proposed_rail_m=proposal, published_rail_m=proposal-.00001, dt_s=.005)
    assert clamped[0] == pytest.approx(.008)
    np.testing.assert_array_equal(clamped[1:], qdot[1:])


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['attempt'])
def test_recorded_final_model_really_exceeded_budget(case):
    wrench = np.array(case['wrench_environment'])
    nominal = np.array(case['nominal_twist_tool'])
    np.testing.assert_array_equal(nominal, case['candidate_twist_tool'])
    bound = PortEnergyConstraint(wrench, case['energy']['available_j'], .05,
        task_power_w=max(0., -wrench @ nominal), assurance='two_port_command_model')
    assert bound.admissible(nominal, tolerance_w=0.)
    assert not bound.admissible(case['old_final_tool'], tolerance_w=0.)


@pytest.mark.parametrize('backend', ['python', 'native'])
@pytest.mark.parametrize('case', CASES, ids=lambda c: c['attempt'])
def test_recorded_low_energy_admits_final_command_through_both_qps(backend, case):
    obj = controller(backend)
    try:
        q = np.array(case['q_meas'])
        obj.reset(q)
        rotation = np.array(case['rotation_base_tcp'])
        wrench = np.array(case['wrench_environment'])
        nominal = np.array(case['nominal_twist_tool'])
        twist = np.r_[rotation @ nominal[:3], rotation @ nominal[3:]]
        wrench_base = np.r_[rotation @ wrench[:3], rotation @ wrench[3:]]
        budget = CommandBudget(.05+case['energy']['available_j'], .15, .05,
            max_command_interval_s=.05, settlement_port='logical_final_model',
            wrench_convention='negative_control_raw_tcp_v1', task_power_source='nominal_command')
        for i in range(400):
            now = 10.+i*.005
            bound = budget.snapshot(now_s=now, wrench_control_raw=-wrench,
                rotation_base_tcp=rotation, nominal_twist_tool=nominal)
            minimum = -(bound.task_power_w + bound.beta*bound.available_j/bound.hold_s)
            step = obj.update(twist, .005, q_meas=q, rail_exec_vel_m_s=.003*np.sin(i/8.),
                command_power_wrench_base=wrench_base, command_power_min_w=minimum,
                auto_commit=False, commit_history=False)
            assert not step.solver_fault_latched, (backend, i, step.fallback_reason)
            vbase = obj.kin.jacobian(q) @ step.qdot
            final = np.r_[rotation.T @ vbase[:3], rotation.T @ vbase[3:]]
            assert budget.reserve(i+1, bound, final, now_s=now+.0001), (backend, i, bound.margin_power_w(final))
            budget.publication_started(i+1)
            budget.commit(i+1, final, now_s=now+.0002, rotation_base_tcp=rotation, dual_success=True)
            obj.commit_publication(step.qdot)
            assert budget.balance_j >= .05 and budget.latched_reason is None
            assert budget.active.expires_s == pytest.approx(now+.0501)
            q = step.q_send.copy()
    finally:
        close(obj)


@pytest.mark.parametrize('backend', ['python', 'native'])
def test_zero_wrench_does_not_create_positive_power_requirement(backend):
    obj = controller(backend)
    try:
        step = obj.update(np.array([0., .005, 0., 0., 0., 0.]), .005, q_meas=Q,
            command_power_wrench_base=np.zeros(6), command_power_min_w=0.)
        assert not step.solver_fault_latched
    finally:
        close(obj)


@pytest.mark.parametrize('backend', ['python', 'native'])
def test_impossible_power_constraint_is_not_relaxed_or_sent(backend):
    obj = controller(backend)
    try:
        step = obj.update(np.zeros(6), .005, q_meas=Q,
            command_power_wrench_base=np.array([0., 1., 0., 0., 0., 0.]), command_power_min_w=1000.)
        assert step.solver_fault_latched
    finally:
        close(obj)


@pytest.mark.parametrize('backend', ['python', 'native'])
def test_rejected_power_prepare_preserves_published_velocity_history(backend):
    obj = controller(backend)
    try:
        first = obj.update(np.array([0., .005, 0., 0., 0., 0.]), .005, q_meas=Q,
            auto_commit=False, commit_history=False)
        assert not first.solver_fault_latched
        obj.commit_publication(first.qdot)
        q = first.q_send.copy()
        rejected = obj.update(np.zeros(6), .005, q_meas=q,
            command_power_wrench_base=np.array([0., 1., 0., 0., 0., 0.]), command_power_min_w=1000.,
            auto_commit=False, commit_history=False)
        assert rejected.solver_fault_latched
        obj.abort_publication()
        fresh = obj.update(np.zeros(6), .005, q_meas=q, auto_commit=False, commit_history=False)
        assert not fresh.solver_fault_latched
        np.testing.assert_allclose(fresh.qdot_prev_used, first.qdot, atol=1e-12)
        np.testing.assert_allclose(fresh.qdot_prev2_used, np.zeros(8), atol=1e-12)
    finally:
        close(obj)


@pytest.mark.parametrize('wrench,minimum', [(np.full(6,np.nan),0.), (np.zeros(6),float('nan')), (np.zeros(6),None)])
def test_invalid_power_never_becomes_disabled(wrench, minimum):
    with pytest.raises(ValueError):
        validate_command_power(wrench, minimum)
