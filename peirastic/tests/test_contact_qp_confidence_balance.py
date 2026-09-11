"""V8r3 confidence direction and nominal torque compete on one physical omega."""
from dataclasses import replace

import numpy as np
import pytest

from peirastic.contact_qp.qp import ContactQp, QpConfig
from peirastic.contact_qp.repair_policy import DifferentialRepairConfig
from peirastic.contact_qp.types import ProbeGeometry, TwistConstraints
from peirastic.tests.test_contact_qp_solver import datum, observation


def config(**kwargs):
    return QpConfig(allocation_policy='differential_repair_v8',
        differential_repair=DifferentialRepairConfig(revision='v8r3_confidence_balance'),
        max_acceleration=np.full(6,100.),**kwargs)


def test_healthy_imbalance_repairs_by_rotation_and_mirrors_image_geometry():
    data=datum((.6,.8,.9))
    result=ContactQp(config()).solve(data)
    assert result.success and result.diagnostics['acquisition_loss']==0.
    assert result.diagnostics['differential_request_m_s']==pytest.approx(.002*.2/.9)
    assert result.qp_twist[4]>1e-3
    assert result.qp_twist[2]==pytest.approx(data.nominal_twist[2],abs=1e-8)
    assert result.alpha==pytest.approx(1.)
    mirrored=ContactQp(config()).solve(datum((.9,.8,.6)))
    reversed_axis=ContactQp(config()).solve(replace(data,geometry=ProbeGeometry.synthetic(image_x_sign=-1)))
    for other in (mirrored,reversed_axis):
        assert other.success
        assert other.qp_twist[4]==pytest.approx(-result.qp_twist[4],abs=1e-7)


def test_deadband_and_sufficient_nominal_are_exactly_transparent():
    data=datum((.8,.8,.85))
    small=ContactQp(config()).solve(data)
    assert small.diagnostics['transparent']
    np.testing.assert_array_equal(small.qp_twist,data.nominal_twist)
    nominal=data.nominal_twist.copy();nominal[4]=.02
    sufficient=replace(data,observation=observation((.6,.8,.9)),nominal_twist=nominal,previous_twist=nominal)
    result=ContactQp(config()).solve(sufficient)
    assert result.diagnostics['differential_request_m_s']>0.
    assert result.diagnostics['transparent']
    np.testing.assert_array_equal(result.qp_twist,nominal)


def test_stopping_alpha_cannot_remove_rotation_or_manufacture_shortfall():
    data=datum((.6,.8,.9))
    free=ContactQp(config()).solve(data)
    stop=TwistConstraints(np.array([[1.,0.,0.,0.,0.,0.]]),np.zeros(1),np.zeros(1),valid_until_s=2.)
    blocked=ContactQp(config()).solve(replace(data,mechanical=stop))
    assert blocked.success and blocked.alpha==pytest.approx(0.,abs=1e-8)
    assert blocked.qp_twist[4]==pytest.approx(free.qp_twist[4],abs=1e-7)
    assert blocked.diagnostics['differential_auxiliary_shortfall']==pytest.approx(
        blocked.diagnostics['differential_shortfall_m_s']/config().normal_scale_m_s,abs=1e-7)


def test_consistency_does_not_manufacture_or_reverse_direction():
    policy=config().differential_repair
    assert policy.confidence_imbalance([.8,.8],[0.,1.])==0.
    assert policy.confidence_imbalance([.6,.9],[.2,1.])==pytest.approx(.2*.2/.9)
    assert policy.confidence_imbalance([.6,.9],[1.,.2])==pytest.approx(.2*.2/.9)


def test_persistent_healthy_imbalance_exhausts_without_healthy_refunds():
    qp=ContactQp(config())
    def solve(t,quality,seq):
        return qp.solve(datum(quality,now_s=t,observation=observation(quality,now=t,seq=seq)))
    assert solve(0.,(.6,.8,.9),1).diagnostics['repair_episode']['repair_allowed']
    for seq,t in enumerate((2.,2.1,2.2,2.3,2.4),2):
        result=solve(t,(.6,.8,.9),seq)
        assert result.success
        assert result.diagnostics['repair_episode']['exhausted']
        assert result.diagnostics['repair_episode']['healthy_resets']==0
        assert result.diagnostics['differential_request_m_s']==0.
    for seq,t in enumerate((2.5,2.6,2.7),10):
        result=solve(t,(.8,.8,.85),seq)
    assert result.diagnostics['repair_episode']['healthy_resets']==1
    assert solve(2.8,(.6,.8,.9),13).diagnostics['repair_episode']['repair_allowed']


def test_old_revision_remains_threshold_only_and_deadband_is_validated():
    old=ContactQp(replace(config(),differential_repair=DifferentialRepairConfig())).solve(datum((.6,.8,.9)))
    assert old.diagnostics['transparent']
    assert old.diagnostics['differential_request_m_s']==0.
    for value in (True,-.1,1.,float('nan')):
        with pytest.raises(ValueError):DifferentialRepairConfig(balance_deadband=value)


def test_energy_admission_is_independent_of_healthy_balance_request():
    from peirastic.contact_qp.port_constraint import PortEnergyConstraint
    data=datum((.6,.8,.9))
    wrench=np.zeros(6);wrench[4]=-1.
    energy=PortEnergyConstraint(wrench_environment=wrench,available_j=0.,hold_s=data.dt_s)
    result=ContactQp(config()).solve(replace(data,energy=energy))
    assert result.success
    assert result.qp_twist[4]<=1e-7
    assert result.diagnostics['differential_shortfall_m_s']>0.
    assert energy.admissible(result.qp_twist,tolerance_w=1e-8,velocity_tolerance=1e-8)


def test_high_threshold_and_opposing_nominal_still_use_same_omega():
    data=datum((.82,.9,.97))
    result=ContactQp(config(c_min=.8)).solve(data)
    assert result.success and result.diagnostics['differential_request_m_s']>0.
    assert result.qp_twist[4]>0.
    assert result.qp_twist[2]==pytest.approx(0.,abs=1e-8)
    nominal=data.nominal_twist.copy();nominal[4]=-.005
    opposing=ContactQp(config(c_min=.8)).solve(replace(data,nominal_twist=nominal,previous_twist=nominal))
    assert opposing.success and opposing.qp_twist[4]>nominal[4]
