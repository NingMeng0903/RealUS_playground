"""Guarded real-endpoint gap recovery; no transport, hardware, or pseudo-samples."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from peirastic.contact_qp.runtime_source import SourceClock, SourceGapContext
from peirastic.tests.test_contact_qp_variable_timebase import observer
from rm75_control.control.admittance_common.variable_step_filter import VariableLowpass1


SOURCE=dict(period_s=.005,timebase='variable_step_bilinear_v1',max_interval_s=.015,max_age_s=.015)
RECOVERY=dict(gap_policy='lease_fresh_foh_v1',max_recovery_interval_s=.05)


def context(previous=1.,new=1.03,now=1.031,**kw):
    return replace(SourceGapContext('A',previous,new,17,1.002,1.052,now,.05),**kw)


def clock(**kw):
    c=SourceClock(**SOURCE,**kw);c.observe('A',1.,1,now_s=1.001)
    return c


@pytest.mark.parametrize('delta',[.015,.015000001,.03,.05])
def test_ingress_boundaries_require_original_live_lease(delta):
    c=clock(**RECOVERY);t=1.+delta;now=t+.001
    gap=None if delta<=.015 else context(new=t,now=now)
    step=c.observe('A',t,2,now_s=now,gap_context=gap)
    assert step.gap_recovered==(delta>.015)
    assert step.source_dt_s==pytest.approx(delta)
    assert step.gap_context is gap and c.source_updates==2
    with pytest.raises(FrozenInstanceError):step.fresh=False
    held=c.observe('A',t,2,now_s=now+.0001)
    assert not held.fresh and not held.gap_recovered and held.gap_context is None


@pytest.mark.parametrize('change,stamp,now',[
    ({},1.050001,1.051001),
    ({},1.03,1.045001),
    ({},1.03,1.029),
    ({'lease_expires_s':1.031},1.03,1.031),
    ({'lease_committed_s':1.03},1.03,1.031),
    ({'lease_id':0},1.03,1.031),
    ({'lease_committed_s':'1.002'},1.03,1.031),
    ({'previous_source_t_s':.999},1.03,1.031),
    ({'source_id':'B'},1.03,1.031),
    ({'max_gap_s':.1},1.03,1.031),
    ({'lease_expires_s':1.2},1.03,1.031),
])
def test_ingress_rejected_recovery_leaves_watermarks_unchanged(change,stamp,now):
    c=clock(**RECOVERY);previous=c.last;last_now=c.last_now
    ctx=context(new=stamp,now=now,**change)
    with pytest.raises(ValueError):c.observe('A',stamp,2,now_s=now,gap_context=ctx)
    assert c.last is previous and c.last_now==last_now and c.source_updates==1


def test_ingress_default_strict_and_held_epoch_guards():
    with pytest.raises(ValueError,match='maximum'):clock().observe('A',1.03,2,now_s=1.031)
    c=clock(**RECOVERY)
    for source,t,ctx in [('A',1.,context()),('B',1.03,context(source_id='B')),
                         ('A',1.005,context(new=1.005,now=1.006))]:
        with pytest.raises(ValueError):c.observe(source,t,2,now_s=max(t+.001,1.002),gap_context=ctx)
    with pytest.raises(ValueError):SourceClock(**SOURCE,gap_policy='lease_fresh_foh_v1')
    with pytest.raises(ValueError):SourceClock(**SOURCE,gap_policy='lease_fresh_foh_v1',max_recovery_interval_s=.01)


def prepared_observer(tmp_path,monkeypatch):
    obs=observer(tmp_path)
    monkeypatch.setattr('rm75_control.control.admittance_common.observer.wrench_sensor_to_link7',
                        lambda raw,contract:np.asarray(raw,dtype=float).copy())
    monkeypatch.setattr(obs,'_rows',lambda *args:(np.zeros((6,len(obs.phi))),)*2)
    obs.configure_source_period(.005,variable_dt=True,max_interval_s=.015,new_epoch=True,**RECOVERY)
    obs.update(1.,np.zeros(6),np.ones(6),source_id='A',sensor_age_s=.001)
    return obs


@pytest.mark.parametrize('delta',[.015000001,.03,.05])
def test_observer_gap_uses_own_watermark_preserves_real_history(tmp_path,monkeypatch,delta):
    obs=prepared_observer(tmp_path,monkeypatch);t=1.+delta;ctx=context(new=t,now=t+.001)
    filt=obs._variable_lpf;before=filt.output.copy();oldinput=filt.previous_input.copy();n=obs._n_updates
    raw=np.full(6,4.5);z=filt.omega*delta;phi=-np.expm1(-z)/z
    expected=np.exp(-z)*before+(phi-np.exp(-z))*oldinput+(1-phi)*raw
    signed,actual=obs.update(t,np.zeros(6),raw,source_id='A',sensor_age_s=.001,
                             source_dt_s=.005,gap_context=ctx)
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
    assert obs._variable_lpf is filt and obs._n_updates==n+1
    assert list(obs._t_ring)==[1.,t] and obs.last_t_s==t
    assert not hasattr(obs,'source_filter_epoch_reset')
    np.testing.assert_array_equal(obs.f_ext_raw_last,raw)
    held=obs.update(t,np.zeros(6),np.full(6,99.),measurement_fresh=False,source_id='A')[1]
    np.testing.assert_array_equal(held,actual)
    assert obs._n_updates==n+1


@pytest.mark.parametrize('change,stamp,age',[
    ({'previous_source_t_s':.995},1.03,.001),
    ({'source_id':'B'},1.03,.001),
    ({'source_t_s':1.025},1.03,.001),
    ({'lease_expires_s':1.031},1.03,.001),
    ({},1.03,.015001),
    ({},1.03,-.001),
    ({},1.050001,.001),
    ({'max_gap_s':.06},1.03,.001),
    ({'lease_expires_s':1.04},1.03,.011),
    ({'lease_committed_s':'1.002'},1.03,.001),
])
def test_observer_independently_rejects_untrusted_recovery(tmp_path,monkeypatch,change,stamp,age):
    obs=prepared_observer(tmp_path,monkeypatch);ctx=replace(context(new=stamp,now=stamp+.001),**change)
    state=obs._variable_lpf.output.copy();n=obs._n_updates;history=list(obs._t_ring)
    with pytest.raises(ValueError):
        obs.update(stamp,np.zeros(6),np.full(6,4.5),source_id='A',sensor_age_s=age,gap_context=ctx)
    assert obs.last_t_s==1. and obs._n_updates==n and list(obs._t_ring)==history
    np.testing.assert_array_equal(obs._variable_lpf.output,state)


def test_observer_no_context_or_new_epoch_cannot_recover_running_gap(tmp_path,monkeypatch):
    obs=prepared_observer(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='maximum'):
        obs.update(1.03,np.zeros(6),np.full(6,4.5),source_id='A',sensor_age_s=.001)
    with pytest.raises(ValueError,match='held'):
        obs.update(1.,np.zeros(6),np.zeros(6),measurement_fresh=False,gap_context=context())
    obs.configure_source_period(.005,variable_dt=True,max_interval_s=.015,new_epoch=True,**RECOVERY)
    with pytest.raises(ValueError):
        obs.update(1.03,np.zeros(6),np.full(6,4.5),source_id='A',sensor_age_s=.001,gap_context=context())


def test_15ms_normal_interval_retains_bilinear_filter(tmp_path,monkeypatch):
    obs=prepared_observer(tmp_path,monkeypatch);old=obs._variable_lpf.output.copy()
    oldinput=obs._variable_lpf.previous_input.copy();raw=np.full(6,4.5);wh=obs._variable_lpf.omega*.015
    expected=((2-wh)*old+wh*(oldinput+raw))/(2+wh)
    actual=obs.update(1.015,np.zeros(6),raw,source_id='A',sensor_age_s=.001)[1]
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)


def test_exact_foh_matches_independent_ode_and_avoids_false_4point5n_peak():
    # A preceding rising edge leaves old filtered state below the old raw value.
    # A long trapezoidal step can then report >6 N despite both raw endpoints 4.5.
    foh=VariableLowpass1(45.,.005,[1.],[4.5]);bilinear=VariableLowpass1(45.,.005,[1.],[4.5])
    assert bilinear.update([4.5],.03)[0]>6.
    assert 1.<foh.update_foh([4.5],.03)[0]<=4.5
    old=1.2;u0=2.;u1=4.5;dt=.03
    foh=VariableLowpass1(45.,.005,[old],[u0])
    oracle=solve_ivp(lambda t,y:foh.omega*(u0+(u1-u0)*t/dt-y),(0,dt),[old],rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(foh.update_foh([u1],dt),oracle.y[:,-1],atol=2e-11,rtol=0)


def test_recovered_raw_force_above_6n_is_not_hidden(tmp_path,monkeypatch):
    obs=prepared_observer(tmp_path,monkeypatch)
    signed,_=obs.update(1.03,np.zeros(6),np.full(6,6.2),source_id='A',sensor_age_s=.001,gap_context=context())
    assert signed[2]==6.2 and obs.f_ext_raw_last[2]==6.2
    # The outer raw-6-N guard consumes this unchanged raw value; it does not
    # consult the interpolated filter result for its independent hard limit.
