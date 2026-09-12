"""Incremental visual repair shares the physical command's hard constraints."""
from dataclasses import replace

import numpy as np
import pytest

from peirastic.contact_qp.qp import ContactQp, QpConfig
from peirastic.contact_qp.repair_policy import DifferentialRepairConfig
from peirastic.contact_qp.port_constraint import PortEnergyConstraint
from peirastic.contact_qp.types import ProbeGeometry
from peirastic.tests.test_contact_qp_solver import datum


def config(**kwargs):
    values=dict(allocation_policy='differential_repair_v8',
        differential_repair=DifferentialRepairConfig(revision='v8r3_confidence_balance',
            permission_mode='continuous',differential_reference='nominal_increment'),
        max_acceleration=np.full(6,100.))
    values.update(kwargs)
    return QpConfig(**values)


def with_nominal(omega,quality=(.6,.8,.9),**kwargs):
    data=datum(quality,**kwargs)
    nominal=data.nominal_twist.copy();nominal[4]=omega
    return replace(data,nominal_twist=nominal,previous_twist=nominal)


def test_matching_nominal_requires_additional_visual_motion_and_same_reference_diagnostics():
    data=with_nominal(.02)
    cfg=config();result=ContactQp(cfg).solve(data)
    assert result.success and not result.diagnostics['transparent']
    assert result.qp_twist[4]>.02+1e-3
    d=result.diagnostics
    assert d['differential_reference']=='nominal_increment'
    assert d['differential_achieved_m_s']==pytest.approx(d['differential_increment_achieved_m_s'])
    assert d['differential_total_achieved_m_s']==pytest.approx(
        d['differential_nominal_achieved_m_s']+d['differential_increment_achieved_m_s'])
    assert d['differential_auxiliary_shortfall']*cfg.normal_scale_m_s==pytest.approx(
        d['differential_shortfall_m_s'],abs=1e-8)
    legacy=ContactQp(replace(cfg,differential_repair=replace(cfg.differential_repair,
        differential_reference='total_velocity'))).solve(data)
    assert legacy.success and legacy.diagnostics['transparent']
    np.testing.assert_array_equal(legacy.qp_twist,data.nominal_twist)


def test_opposing_nominal_is_corrected_and_mirrored_geometry_reverses_increment():
    data=with_nominal(-.005)
    result=ContactQp(config()).solve(data)
    assert result.success and result.qp_twist[4]>data.nominal_twist[4]+1e-3
    mirrored_nominal=data.nominal_twist.copy();mirrored_nominal[4]*=-1
    mirrored=replace(data,geometry=ProbeGeometry.synthetic(image_x_sign=-1),
        nominal_twist=mirrored_nominal,previous_twist=mirrored_nominal)
    other=ContactQp(config()).solve(mirrored)
    assert other.success
    assert other.qp_twist[4]==pytest.approx(-result.qp_twist[4],abs=1e-7)


def test_zero_imbalance_keeps_nominal_exactly_transparent():
    data=with_nominal(.02,quality=(.85,.85,.85))
    result=ContactQp(config()).solve(data)
    assert result.success and result.diagnostics['transparent']
    assert result.diagnostics['differential_request_m_s']==0
    np.testing.assert_array_equal(result.qp_twist,data.nominal_twist)


@pytest.mark.parametrize('constraint',['angle','slew','aperture','force','energy'])
def test_increment_cannot_bypass_physical_limits(constraint):
    cfg=config();data=datum((.6,.8,.9))
    if constraint=='angle':
        cfg=replace(cfg,angle_limit_rad=.12);data=replace(data,measured_angle=.12)
    elif constraint=='slew':
        acceleration=np.full(6,100.);acceleration[4]=.001;cfg=replace(cfg,max_acceleration=acceleration)
    elif constraint=='aperture':cfg=replace(cfg,aperture_budget_m_s=0.)
    elif constraint=='force':data=replace(data,force_n=4.5)
    else:
        wrench=np.zeros(6);wrench[4]=-1.
        data=replace(data,energy=PortEnergyConstraint(wrench,0.,.05))
    result=ContactQp(cfg).solve(data)
    assert result.success
    assert result.qp_twist[4]<=((.001*data.dt_s+1e-8) if constraint=='slew' else 1e-8)
    assert result.hard_constraints.violation(result.qp_twist)<=cfg.feasibility_tolerance
    if constraint=='force':assert result.diagnostics['differential_request_m_s']==0
    else:assert result.diagnostics['differential_shortfall_m_s']>0
    if data.energy is not None:assert data.energy.admissible(result.qp_twist,tolerance_w=1e-8)


def test_task_supply_enables_only_its_bounded_total_power():
    wrench=np.zeros(6);wrench[4]=-1.
    energy=PortEnergyConstraint(wrench,0.,.05,task_power_w=.005,assurance='two_port_command_model')
    result=ContactQp(config()).solve(datum((.6,.8,.9),energy=energy))
    assert result.success
    assert 1e-4<result.qp_twist[4]<=.005+1e-8
    assert energy.admissible(result.qp_twist,tolerance_w=1e-8)
    assert result.diagnostics['differential_shortfall_m_s']>0


def test_reference_is_explicit_legacy_default_and_r3_only():
    assert DifferentialRepairConfig().differential_reference=='total_velocity'
    with pytest.raises(ValueError,match='requires v8r3'):
        DifferentialRepairConfig(differential_reference='nominal_increment')
    with pytest.raises(ValueError,match='unsupported differential reference'):
        DifferentialRepairConfig(differential_reference='implicit')
