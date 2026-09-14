"""Final-command Cartesian admission in both inner QP levels, no hardware."""
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics

Q = np.array([.375, *np.deg2rad([-89.5, -94.5, 65.2, 96., 89.3, 61., 94.6])])

def controller(backend):
    raw = yaml.safe_load((Path(__file__).parents[1] / 'configs/joint_admittance_8dof.yaml').read_text())
    cfg = build_joint_ik_config(raw)
    cfg.backend = backend
    cfg.control_frame = 'base'
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    cfg.native_shm_prefix = f'twist_test_{os.getpid()}_{backend}'
    obj = JointIkController(RobotKinematics(), cfg)
    if obj._native is not None:
        obj._native.timeout_s = .5
    obj.reset(Q)
    return obj


def close(obj):
    if obj._native is not None:
        obj._native.shutdown()


from rm75_control.control.joint_admittance_8dof.command_twist import validate_command_twist

@pytest.mark.parametrize('backend',['python','native'])
def test_command_twist_caps_tracking_despite_measured_rail_compensation(backend):
    obj=controller(backend)
    try:
        q=Q.copy()
        cap=np.array([.015,.015,.015,.28,.28,.28])
        for i in range(60):
            step=obj.update(np.array([0.,.025,.025,0.,0.,0.]),.005,q_meas=q,
                rail_exec_vel_m_s=.008*np.sin(i/7.),
                command_twist_rows_base=np.eye(6),command_twist_lower=-cap,command_twist_upper=cap,
                auto_commit=False,commit_history=False)
            assert not step.solver_fault_latched,(i,step.fallback_reason)
            rail_exec=.008*np.sin(i/7.)
            J=obj.kin.jacobian(q)
            final=J@step.qdot+J[:,0]*(rail_exec-step.qdot[0])
            np.testing.assert_allclose(step.v_tcp_estimated,final,atol=1e-10)
            assert np.all(abs(final)<=cap+1e-8),(i,final)
            obj.commit_publication(step.qdot);q=step.q_send.copy()
    finally:close(obj)

@pytest.mark.parametrize('backend',['python','native'])
def test_impossible_command_twist_is_not_relaxed_or_committed(backend):
    obj=controller(backend)
    try:
        step=obj.update(np.zeros(6),.005,q_meas=Q,
            command_twist_rows_base=np.eye(6)[[2]],command_twist_lower=[100.],command_twist_upper=[101.],
            auto_commit=False,commit_history=False)
        assert step.solver_fault_latched
        obj.abort_publication()
        fresh=obj.update(np.zeros(6),.005,q_meas=Q,auto_commit=False,commit_history=False)
        assert not fresh.solver_fault_latched
        np.testing.assert_allclose(fresh.qdot_prev_used,np.zeros(8),atol=1e-12)
    finally:close(obj)

@pytest.mark.parametrize('rows,lo,hi',[(np.eye(6),None,np.ones(6)),(np.zeros((17,6)),np.zeros(17),np.ones(17)),
    (np.full((1,6),np.nan),[0.],[1.]),(np.zeros((1,6)),[1.],[0.])])
def test_invalid_command_intervals_rejected(rows,lo,hi):
    with pytest.raises(ValueError):validate_command_twist(rows,lo,hi)


@pytest.mark.parametrize('backend',['python','native'])
def test_rail_does_not_chase_its_own_execution(backend):
    obj=controller(backend)
    try:
        q=Q.copy()
        rail_exec=0.0
        previous=np.zeros(6)
        max_acc=np.array([1.,1.,.8,2.,3.,2.])
        max_vel=np.array([.04,.04,.01,.6,.28,.6])
        dt=.005
        for i in range(100):
            rows=np.vstack((np.eye(6),np.eye(6)))
            lower=np.r_[-max_vel,previous-max_acc*dt]
            upper=np.r_[max_vel,previous+max_acc*dt]
            step=obj.update(np.array([0.,0.,.015,0.,0.,0.]),dt,q_meas=q,
                rail_exec_vel_m_s=rail_exec,
                command_twist_rows_base=rows,command_twist_lower=lower,command_twist_upper=upper,
                auto_commit=False,commit_history=False)
            assert not step.solver_fault_latched,(i,step.fallback_reason)
            J=obj.kin.jacobian(q)
            final=J@step.qdot+J[:,0]*(rail_exec-step.qdot[0])
            assert abs(step.qdot[0])<=.03,(i,step.qdot[0])
            assert abs(final[1])<=.002,(i,final[1])
            obj.commit_publication(step.qdot)
            q=step.q_send.copy()
            rail_exec+=.3*(step.qdot[0]-rail_exec)
            previous=final
    finally:close(obj)
