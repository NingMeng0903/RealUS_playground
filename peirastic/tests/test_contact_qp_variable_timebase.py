"""Active-only measured timebase; no hardware or IPC connections."""
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.signal import butter,lfilter,tf2ss
from rm75_control.control.admittance_common.variable_step_filter import VariableLowpass1,VariableHighpass2
from peirastic.contact_qp.runtime_source import SourceClock

SOURCE=dict(period_s=.005,timebase='variable_step_bilinear_v1',max_interval_s=.015,max_age_s=.015,jitter_fraction=.5)


def observer(tmp_path):
    from rm75_control.control.admittance_common.observer import CompensatedForceObserver,ForceObserverConfig
    from rm75_control.force.compensation.regressor import PHI_NAMES
    params={key:0. for key in PHI_NAMES};params['m']=.5
    path=tmp_path/'phi.json';path.write_text(json.dumps({'phi_recommended':params}))
    return CompensatedForceObserver(ForceObserverConfig(phi_path=path,poll_hz=200.,causal_fc_hz=45.,causal_order=1))


def update(obs,t,force=4.,**kw):
    return obs.update(t,np.zeros(6),np.array([0.,0.,force,0.,0.,0.]),**kw)[1]


def test_fixed_step_matches_original_lp_hp_and_dynamic_observer_switch(tmp_path):
    rng=np.random.default_rng(4);u=4+rng.normal(size=4000)
    low=VariableLowpass1(45.,.005,np.zeros(1),np.zeros(1));high=VariableHighpass2(2.5,.005)
    lp=np.array([low.update([v],.005)[0] for v in u]);hp=np.array([high.update(v,.005) for v in u])
    for actual,order,fc,kind in ((lp,1,45.,'low'),(hp,2,2.5,'high')):
        b,a=butter(order,2*fc*.005,btype=kind)
        np.testing.assert_allclose(actual,lfilter(b,a,u),atol=1e-12,rtol=0)
    original=observer(tmp_path);variable=observer(tmp_path)
    for i,force in enumerate(u[:100]):
        update(original,1+i*.005,force);update(variable,1+i*.005,force)
    before=variable._f_ext_last.copy()
    assert not np.array_equal(before,variable.f_ext_raw_last)
    variable.configure_source_period(.005,variable_dt=True,max_interval_s=.015,new_epoch=True)
    # The already-consumed source must not be filtered a second time on entry.
    np.testing.assert_array_equal(update(variable,1+99*.005,u[99],measurement_fresh=False),before)
    for i,force in enumerate(u[100:200],100):
        np.testing.assert_allclose(update(variable,1+i*.005,force),update(original,1+i*.005,force),atol=1e-12,rtol=0)
    variable.configure_source_period(None)
    for i,force in enumerate(u[200:300],200):
        np.testing.assert_allclose(update(variable,1+i*.005,force),update(original,1+i*.005,force),atol=1e-12,rtol=0)


@pytest.mark.parametrize('interval',[.00215,.01091])
def test_observer_first_new_timestamp_uses_existing_history_not_new_clock_default(tmp_path,interval):
    obs=observer(tmp_path);update(obs,1.,2.);update(obs,1.005,6.)
    y=obs._f_ext_last.copy();previous=obs.f_ext_raw_last.copy()
    obs.configure_source_period(.005,variable_dt=True,max_interval_s=.015,new_epoch=True)
    actual=update(obs,1.005+interval,3.,source_dt_s=.005)
    w=2/.005*np.tan(np.pi*45*.005)
    expected=((2-w*interval)*y+w*interval*(previous+obs.f_ext_raw_last))/(2+w*interval)
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)


def test_irregular_hp_matches_independent_continuous_matrix_oracle():
    high=VariableHighpass2(2.5,.005);w=2/.005*np.tan(np.pi*2.5*.005)
    b,a=butter(2,w,btype='high',analog=True);A,B,C,D=tf2ss(b,a)
    state=np.zeros(2);previous=0.
    for i in range(1200):
        dt=(.00215,.005,.01091)[i%3];value=4+np.sin(i*.13)
        state=np.linalg.solve(np.eye(2)-dt*A/2,(np.eye(2)+dt*A/2)@state+dt*B[:,0]*(previous+value)/2)
        expected=float((C@state+D[:,0]*value)[0])
        assert high.update(value,dt)==pytest.approx(expected,abs=1e-12)
        previous=value


def test_new_epoch_gap_seeds_current_measurement_once_then_running_gap_rejects(tmp_path):
    obs=observer(tmp_path);update(obs,1.,1.)
    obs.configure_source_period(.005,variable_dt=True,max_interval_s=.015,new_epoch=True)
    actual=update(obs,2.,4.)
    np.testing.assert_array_equal(actual,obs.f_ext_raw_last)
    assert obs.source_filter_epoch_reset['steady_current_measurement_seed']
    watermark=obs.last_t_s;count=obs._n_updates
    with pytest.raises(ValueError,match='maximum'):update(obs,2.02,4.)
    assert obs.last_t_s==watermark and obs._n_updates==count


def test_clock_actual_intervals_duplicates_and_faults():
    clock=SourceClock(**SOURCE);clock.observe('A',1.,1,now_s=1.001)
    assert clock.observe('A',1.,1,now_s=1.003).fresh is False
    assert clock.observe('A',1.01091,2,now_s=1.012).source_dt_s==pytest.approx(.01091)
    for identity,t,wall,now in [('A',1.027,3,1.028),('A',1.005,3,1.014),('B',1.015,3,1.016),('A',1.015,3,1.04)]:
        with pytest.raises(ValueError):clock.observe(identity,t,wall,now_s=now)
    assert clock.last.source_t_s==1.01091


def test_active_adapter_observer_repeated_measurement_and_aborted_command(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from peirastic.tests.test_contact_qp_runtime import make_outer,MemorySink
    from peirastic.tests.test_contact_qp_runtime_config import active_config
    from peirastic.realman8dof.modes.contact_active import ContactQpOuter
    config=active_config();config.update(source=SOURCE,energy=None,energy_constraint_enabled=False)
    baseline,pose=make_outer();active=ContactQpOuter(baseline,config,sink=MemorySink(),
        feature_receiver=SimpleNamespace(observation=None,close=lambda **kw:None));active.set_origin(pose)
    obs=observer(tmp_path);obs.configure_source_period(.005,variable_dt=True,max_interval_s=.015,new_epoch=True)
    now=[1.001];monkeypatch.setattr('peirastic.realman8dof.modes.contact_active.time.monotonic',lambda:now[0])
    for index,(stamp,wall) in enumerate(((1.,1),(1.,1),(1.01091,2))):
        source=active.prepare_source('A',stamp,wall,now_s=now[0])
        f=update(obs,stamp,4.,measurement_fresh=source.fresh and stamp!=obs.last_t_s,source_dt_s=source.source_dt_s)
        command=active.sample(0.,pose,f,f_ext_raw=obs.f_ext_raw_last,dt_actual=.005)
        active.publication_abort('test refusal',definitely_not_sent=True)
        assert active.reference_time_s==0. and active.nominal.z_law._transaction.pending is None
        if index==0:hp=active.controller._variable_hp.state.copy();lp=obs._variable_lpf.output.copy()
        if index==1:
            np.testing.assert_array_equal(active.controller._variable_hp.state,hp)
            np.testing.assert_array_equal(obs._variable_lpf.output,lp)
        now[0]=1.003 if index==0 else 1.012
    assert obs._n_updates==2 and active.controller._measurement_updates==2
    before=obs._variable_lpf.output.copy();obs.configure_source_period(None)
    np.testing.assert_array_equal(obs._f_ext_last,before)
    active.close()


def test_actual_active_configuration_is_valid_offline():
    from peirastic.contact_qp.runtime_config import validate_study_config
    assert validate_study_config('peirastic/config/contact_qp/active_probe50.yaml')['command_authority']=='outer_qp_original_ik'
