"""Independent supplied-work checks; task authorization is never recovery."""
from dataclasses import replace
import math

import numpy as np
import pytest

from peirastic.contact_qp.command_budget import CommandBudget
from peirastic.contact_qp.port_constraint import PortEnergyConstraint


def budget(**overrides):
    config=dict(initial_j=.2,capacity_j=.3,stopping_reserve_j=.01,
        max_command_interval_s=.05,settlement_port='logical_final_model',
        wrench_convention='negative_control_raw_tcp_v1',task_power_source='nominal_command')
    return CommandBudget(**(config|overrides))


def speed(x):
    return np.array([x,0.,0.,0.,0.,0.])


def snapshot(tank,now,nominal=1.,raw=2.):
    return tank.snapshot(now_s=now,wrench_control_raw=speed(raw),
        rotation_base_tcp=np.eye(3),nominal_twist_tool=speed(nominal))


def publish(tank,key,now,final=1.,nominal=1.,raw=2.,delay=0.):
    snap=snapshot(tank,now,nominal,raw)
    v=speed(final)
    assert tank.reserve(key,snap,v,now_s=now)
    tank.publication_started(key)
    tank.commit(key,v,now_s=now+delay,rotation_base_tcp=np.eye(3),dual_success=True)
    return snap


def test_long_baseline_spends_source_without_draining_small_initial_tank():
    tank=budget(initial_j=.010001)
    for k in range(1000):
        publish(tank,k,k*.01)
        assert tank.reserved_j==0.
    tank.stop(now_s=10.)
    assert tank.balance_j==pytest.approx(.010001)
    assert tank.cumulative_port_work_j==pytest.approx(-20.)
    assert tank.cumulative_task_source_used_j==pytest.approx(20.)
    assert tank.cumulative_tank_work_j==pytest.approx(0.)
    assert tank.facts['energy_assurance']=='two_port_command_model'
    assert not tank.facts['physical_certified']


def test_extra_output_debits_difference_and_external_recovery_stays_absolute():
    tank=budget()
    snap=publish(tank,0,0.,final=2.)
    assert snap.lower_power_w(speed(2.))==-4.
    assert snap.lower_work_j(speed(2.),.02)==pytest.approx(-.08)
    assert tank.reserved_j==pytest.approx(.1)
    tank.advance(.02)
    assert tank.balance_j==pytest.approx(.16)
    assert tank.cumulative_port_work_j==pytest.approx(-.08)
    assert tank.cumulative_task_source_used_j==pytest.approx(.04)
    publish(tank,1,.025,final=-2.)
    tank.advance(.045)
    assert tank.balance_j==pytest.approx(.23)
    assert tank.cumulative_task_source_used_j==pytest.approx(.05)
    assert tank.cumulative_port_work_j==pytest.approx(-.02)
    for event in tank.drain_events():
        if event['event']=='logical_epoch_work':
            assert event['work_j']==event['port_work_j']
            assert event['tank_work_j']==pytest.approx(event['port_work_j']+event['task_source_used_j'])


@pytest.mark.parametrize('final',[0.,.25,1.])
def test_unused_source_cannot_fill_tank_when_motion_is_cancelled_or_reduced(final):
    tank=budget()
    publish(tank,0,0.,final=final)
    tank.advance(.04)
    assert tank.balance_j==.2
    assert tank.cumulative_task_source_used_j==pytest.approx(.08*final)
    assert tank.cumulative_tank_work_j==0.


def test_capacity_clips_only_external_recovery_and_logs_discard():
    tank=budget(initial_j=.29)
    publish(tank,0,0.,final=-1.)
    tank.advance(.04)
    assert tank.balance_j==.3
    assert tank.cumulative_port_work_j==pytest.approx(.08)
    assert tank.cumulative_task_source_used_j==0.
    assert tank.cumulative_capacity_discard_j==pytest.approx(.07)
    assert tank.balance_j-.29==pytest.approx(tank.cumulative_port_work_j+
        tank.cumulative_task_source_used_j-tank.cumulative_capacity_discard_j)


def test_snapshot_freezes_independent_nominal_and_refuses_replaced_qp_snapshot():
    tank=budget()
    baseline=speed(1.)
    snap=tank.snapshot(now_s=0.,wrench_control_raw=speed(2.),
        rotation_base_tcp=np.eye(3),nominal_twist_tool=baseline)
    baseline[0]=1000.
    assert snap.task_power_w==2.
    assert not tank.reserve('forged',replace(snap,task_power_w=2000.),speed(100.),now_s=0.)
    assert tank.reserve('ok',snap,speed(1.),now_s=0.)
    assert tank.pending.nominal_twist_tool[0]==1.
    with pytest.raises(ValueError):tank.pending.nominal_twist_tool[0]=99.
    tank.publication_started('ok')
    tank.commit('ok',speed(1.),now_s=.001,rotation_base_tcp=np.eye(3),dual_success=True)
    next_snap=snapshot(tank,.01,nominal=3.)
    assert next_snap.task_power_w==6.
    assert tank.active.task_power_w==2.
    assert tank.cumulative_task_source_used_j==pytest.approx(.018)


@pytest.mark.parametrize('started,known',[(False,False),(True,False),(True,True)])
def test_unknown_publication_never_earns_source_or_refunds_liability(started,known):
    tank=budget()
    snap=snapshot(tank,0.)
    assert tank.reserve('pending',snap,speed(2.),now_s=0.)
    if started:tank.publication_started('pending')
    assert not tank.reject_new_only('pending',definitely_not_sent=known,now_s=.01)
    assert tank.reserved_j==pytest.approx(.1)
    assert tank.balance_j==.2
    assert tank.cumulative_task_source_used_j==0.
    tank.stop(now_s=.1)
    assert tank.reserved_j==pytest.approx(.1)
    assert tank.cumulative_task_source_used_j==0.


def test_review_delay_and_expiry_only_fund_successful_active_prefix():
    tank=budget()
    publish(tank,0,0.,delay=.01)
    assert tank.cumulative_task_source_used_j==0.
    assert not tank.advance(.06)
    assert tank.cumulative_task_source_used_j==pytest.approx(.08)
    assert tank.cumulative_port_work_j==pytest.approx(-.08)
    assert tank.active is None
    with pytest.raises(ValueError):snapshot(tank,.07)


def test_supplied_constraint_has_absolute_port_and_affine_affordability():
    energy=PortEnergyConstraint(speed(-2.),.05,.05,task_power_w=2.,
        assurance='two_port_command_model')
    assert energy.admissible(speed(1.5),tolerance_w=0.)
    assert not energy.admissible(speed(1.5001),tolerance_w=0.)
    assert energy.lower_power_w(speed(1.5))==-3.
    assert energy.margin_power_w(speed(1.5))==0.
    assert energy.lower_work_j(speed(1.5),.05)==pytest.approx(-.15)
    assert energy.aged(.02).task_power_w==2.


@pytest.mark.parametrize('source',[None,True,'damping','nominal'])
def test_invalid_source_mode_refused(source):
    with pytest.raises(ValueError):budget(task_power_source=source)


@pytest.mark.parametrize('power',[True,-1.,float('nan'),float('inf')])
def test_invalid_power_and_unmodeled_parameter_injection_refused(power):
    with pytest.raises(ValueError):
        PortEnergyConstraint(speed(-1.),.1,.05,task_power_w=power,assurance='two_port_command_model')
    with pytest.raises(ValueError):budget(constraint={'task_power_w':power})


def test_source_requires_explicit_label_and_nominal_finite_vector():
    with pytest.raises(ValueError):PortEnergyConstraint(speed(-1.),.1,.05,task_power_w=1.)
    for nominal in (None,[1.,2.],np.full(6,float('nan')),np.full(6,float('inf'))):
        tank=budget()
        with pytest.raises(ValueError):tank.snapshot(now_s=0.,wrench_control_raw=speed(2.),
            rotation_base_tcp=np.eye(3),nominal_twist_tool=nominal)
        assert tank.balance_j==.2 and tank.pending is None
    with pytest.raises(ValueError):snapshot(budget(task_power_source='none'),0.)


def test_overflowed_nominal_and_combined_allowance_are_rejected():
    tank=budget()
    with pytest.raises(ValueError):snapshot(tank,0.,nominal=1e308,raw=1e308)
    assert tank.cumulative_task_source_used_j==0.
    with pytest.raises(ValueError):
        PortEnergyConstraint(speed(-1.),1e308,1.,task_power_w=1e308,assurance='two_port_command_model')


def test_independent_random_six_axis_supplied_account():
    rng=np.random.default_rng(91011)
    tank=budget(initial_j=10.,capacity_j=10.1)
    absolute=[];sources=[];discards=[]
    balance=10.
    for k in range(500):
        now=k*.02
        w=rng.normal(size=6);vb=rng.normal(size=6);vf=rng.normal(size=6)
        snap=tank.snapshot(now_s=now,wrench_control_raw=-w,rotation_base_tcp=np.eye(3),nominal_twist_tool=vb)
        assert tank.reserve(k,snap,vf,now_s=now)
        tank.publication_started(k)
        tank.commit(k,vf,now_s=now+.002,rotation_base_tcp=np.eye(3),dual_success=True)
        # Previous epoch continues during this publication delay. Independent
        # reference works only from recorded inputs and explicit active durations.
        if k:
            a=.02*old_p;s=.02*min(old_a,max(0.,-old_p))
            absolute.append(a);sources.append(s)
            excess=max(0.,balance+a+s-10.1);discards.append(excess)
            balance=min(10.1,balance+a+s)
        old_p=float(w@vf);old_a=max(0.,-float(w@vb))
        assert tank.balance_j>=.01
    tank.stop(now_s=10.)
    a=.018*old_p;s=.018*min(old_a,max(0.,-old_p))
    absolute.append(a);sources.append(s)
    discards.append(max(0.,balance+a+s-10.1))
    balance=min(10.1,balance+a+s)
    assert tank.cumulative_port_work_j==pytest.approx(math.fsum(absolute),abs=1e-11)
    assert tank.cumulative_task_source_used_j==pytest.approx(math.fsum(sources),abs=1e-11)
    assert tank.cumulative_capacity_discard_j==pytest.approx(math.fsum(discards),abs=1e-11)
    assert tank.balance_j==pytest.approx(balance,abs=1e-11)


def test_legacy_default_keeps_strict_one_port_balance():
    tank=budget(task_power_source='none')
    snap=tank.snapshot(now_s=0.,wrench_control_raw=speed(2.),rotation_base_tcp=np.eye(3))
    assert snap.assurance=='command_model' and snap.task_power_w==0.
    assert tank.reserve(0,snap,speed(1.),now_s=0.)
    tank.publication_started(0)
    tank.commit(0,speed(1.),now_s=0.,rotation_base_tcp=np.eye(3),dual_success=True)
    tank.advance(.04)
    assert tank.balance_j==pytest.approx(.12)
    assert tank.cumulative_task_source_used_j==0.
