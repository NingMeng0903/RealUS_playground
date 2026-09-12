"""Actual torque-state feedback and visual velocity fusion stay bounded."""
from dataclasses import replace

import numpy as np
import pytest

from peirastic.contact_qp.qp import ContactQp,QpConfig
from peirastic.contact_qp.repair_policy import DifferentialRepairConfig
from peirastic.contact_qp.port_constraint import PortEnergyConstraint
from peirastic.realman8dof.force.torque_tilt import TorqueTilt,TorqueTiltConfig
from peirastic.tests.test_contact_qp_solver import datum,observation


def config():
    return QpConfig(allocation_policy='differential_repair_v8',
        differential_repair=DifferentialRepairConfig(revision='v8r3_confidence_balance',
            permission_mode='continuous'),max_velocity=np.array([.04,.04,.01,.6,.22,.6]))


@pytest.mark.parametrize('torque_nm',[0.,-.04,.04])
def test_total_velocity_request_does_not_accumulate_when_committed_to_torque_state(torque_nm):
    tilt=TorqueTilt(TorqueTiltConfig());qp=ContactQp(config())
    wrench=np.array([0.,0.,4.,0.,torque_nm,0.]);desired=np.array([0.,0.,4.,0.,0.,0.])
    previous=np.array([.02,0.,0.,0.,0.,0.]);history=[]
    for i in range(240):
        now=1+i*.005
        nominal=np.array([.02,0.,0.,0.,0.,0.])
        nominal[4]=tilt.prepare(wrench,desired,measurement_id=i+1,dt_s=.005,
            contact=True,pose=np.zeros(6),slack_norm=0.)
        data=datum((.6,.8,.9),nominal_twist=nominal,previous_twist=previous,now_s=now,
            observation=observation((.6,.8,.9),now=1+(i//10)*.05,seq=i//10+1),
            measured_angle=tilt.theta_tilt)
        result=qp.solve(data)
        assert result.success
        assert result.hard_constraints.violation(result.qp_twist)<=qp.config.feasibility_tolerance
        tilt.commit_applied(result.qp_twist[4]);previous=result.qp_twist.copy();history.append(previous[4])
    assert 0.<history[-1]<(.07 if torque_nm<0 else .02)
    # An additive per-cycle visual correction in this interconnection reaches
    # the .22rad/s rate cap even for zero/opposed torque and fixed confidence.
    assert max(history)<.08
    assert tilt.theta_tilt<.08


def test_sufficient_nominal_is_transparent_and_diagnostics_separate_contributions():
    data=datum((.6,.8,.9));nominal=data.nominal_twist.copy();nominal[4]=.02
    result=ContactQp(config()).solve(replace(data,nominal_twist=nominal,previous_twist=nominal))
    assert result.success and result.diagnostics['transparent']
    np.testing.assert_array_equal(result.qp_twist,nominal)
    d=result.diagnostics
    assert d['differential_reference']=='total_velocity'
    assert d['differential_achieved_m_s']==pytest.approx(d['differential_total_achieved_m_s'])
    assert d['differential_nominal_achieved_m_s']==pytest.approx(d['differential_total_achieved_m_s'])
    assert d['differential_increment_achieved_m_s']==0.


def test_task_supply_obeys_bounded_total_power():
    wrench=np.zeros(6);wrench[4]=-1.
    energy=PortEnergyConstraint(wrench,0.,.05,task_power_w=.005,assurance='two_port_command_model')
    result=ContactQp(config()).solve(datum((.6,.8,.9),energy=energy))
    assert result.success and 1e-4<result.qp_twist[4]<=.005+1e-8
    assert energy.admissible(result.qp_twist,tolerance_w=1e-8)
    assert result.diagnostics['differential_shortfall_m_s']>0


def test_unsupported_increment_reference_cannot_be_enabled():
    with pytest.raises(TypeError):
        DifferentialRepairConfig(revision='v8r3_confidence_balance',differential_reference='nominal_increment')
