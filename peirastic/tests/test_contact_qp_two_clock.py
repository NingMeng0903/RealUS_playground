"""Independent source ingress and 200 Hz command transactions, no hardware."""
from copy import deepcopy
import numpy as np
import pytest
from peirastic.realman8dof.force.contact_nominal import build_contact_nominal


def law(period):
    result,_=build_contact_nominal(.005)
    result.controller.configure_source_period(period)
    return result


def prepare(controller,tick,source_tick,*,fresh=True,period=.01,force=3.):
    wrench=np.array([0.,0.,force,0.,0.,0.])
    return controller.prepare(control_step_id=tick,source_sample_id=('source-A',source_tick),
        source_t_s=1.+source_tick*period,measurement_fresh=fresh,source_dt_s=period,
        dt_s=.005,dt_actual=.005,pose=np.array([.12,.02,.15,0.,0.,0.]),
        f_ext=wrench,f_ext_raw=wrench,f_des=np.array([0.,0.,4.,0.,0.,0.]),
        path_twist=np.zeros(6),sensor_age_s=0. if fresh else .005,
        feedback_age_s=0.,v_tcp_z_actual=0.)


def test_100hz_ingress_preserves_100_control_steps_and_constant_force_dynamics():
    slow,fast=law(.01),law(.005)
    accepted_integral=0.
    reference=0.
    for tick in range(100):
        a=prepare(slow,tick,tick//2,fresh=tick%2==0)
        b=prepare(fast,tick,tick,period=.005)
        np.testing.assert_allclose(a.v_force,b.v_force,atol=1e-13,rtol=0)
        slow.commit_applied(a.v_force);fast.commit_applied(b.v_force)
        if slow.controller.contact_present:accepted_integral+=a.v_force[2]*.005
        reference+=.005
    assert slow.controller._measurement_updates==50
    assert fast.controller._measurement_updates==100
    assert slow.z_law._transaction.last_measurement_id==99
    assert slow.controller.x_adm_z==pytest.approx(accepted_integral,abs=1e-12)
    assert reference==pytest.approx(.5)
    assert slow.controller._ke_estimator._contact_ticks==fast.controller._ke_estimator._contact_ticks


def test_abort_consumes_fresh_once_then_held_step_can_propose_without_filter_learning():
    controller=law(.01)
    first=prepare(controller,0,0,force=3.)
    hp=controller.controller._hp_zi.copy()
    ke_previous=controller.controller._ke_estimator._last_x
    controller.abort()
    held=prepare(controller,1,0,fresh=False,force=3.)
    assert controller.controller._measurement_updates==1
    np.testing.assert_array_equal(controller.controller._hp_zi,hp)
    assert controller.controller._ke_estimator._last_x==ke_previous
    controller.commit_applied(held.v_force)
    with pytest.raises(ValueError,match='duplicate or stale'):
        prepare(controller,1,0,fresh=False)
    with pytest.raises(ValueError,match='new source'):
        prepare(controller,2,0,fresh=True)
    assert controller.z_law._transaction.last_measurement_id==1


def test_force_derivative_uses_new_source_interval_and_does_not_decay_on_hold():
    controller=law(.01)
    a=prepare(controller,0,0,force=3.);controller.commit_applied(a.v_force)
    a=prepare(controller,1,0,fresh=False,force=3.);controller.commit_applied(a.v_force)
    barrier=deepcopy(controller.controller._force_barrier)
    expected=barrier.update_fdot(3.2,.01)
    a=prepare(controller,2,1,force=3.2)
    assert controller.controller.force_dot_z==pytest.approx(expected)
    controller.abort()
    a=prepare(controller,3,1,fresh=False,force=3.2)
    assert controller.controller.force_dot_z==pytest.approx(expected)
    controller.abort()


def test_epoch_filter_steady_contact_initialization_and_same_period_no_reset():
    controller=law(.01)
    a=prepare(controller,0,0,force=4.)
    assert controller.controller.instability_index==pytest.approx(0.,abs=1e-12)
    assert controller.controller._f_dc==4.
    controller.commit_applied(a.v_force)
    before=controller.controller._hp_zi.copy()
    controller.controller.configure_source_period(.01)
    np.testing.assert_array_equal(before,controller.controller._hp_zi)
    with pytest.raises(ValueError,match='fresh controller'):
        controller.controller.configure_source_period(.005)


def test_contact_confirmation_event_can_occur_on_a_held_control_step():
    controller=law(.01)
    controller.controller.cfg.physical_contact.enter_confirm_s=.010
    events=[]
    for tick in range(4):
        a=prepare(controller,tick,tick//2,fresh=tick%2==0,force=4.)
        events.append(controller.controller.physical_contact_acquire_event)
        controller.abort()
    assert sum(events)==1 and events[1]
    assert controller.controller._force_point_inited


def test_observer_source_binding_holds_samples_and_restores_legacy_coefficients(tmp_path):
    import json
    from rm75_control.control.admittance_common.observer import CompensatedForceObserver,ForceObserverConfig
    from rm75_control.force.compensation.regressor import PHI_NAMES
    parameters={key:0. for key in PHI_NAMES};parameters['m']=.5
    path=tmp_path/'phi.json';path.write_text(json.dumps({'phi_recommended':parameters}))
    observer=CompensatedForceObserver(ForceObserverConfig(phi_path=path,poll_hz=200.))
    original_b,original_a=observer._lpf_b.copy(),observer._lpf_a.copy()
    raw=np.array([0,0,4,0,0,0.])
    first=observer.update(1.,np.zeros(6),raw)
    observer.configure_source_period(.01)
    for tick in range(1,100):
        output=observer.update(1.+(tick//2)*.01,np.zeros(6),raw,
                               measurement_fresh=tick%2==0)
        np.testing.assert_allclose(output[1],first[1],atol=1e-12,rtol=0)
    assert observer._n_updates==50
    watermark=observer.last_t_s;history=list(observer._t_ring)
    observer.configure_source_period(None)
    np.testing.assert_array_equal(observer._lpf_b,original_b)
    np.testing.assert_array_equal(observer._lpf_a,original_a)
    assert observer.last_t_s==watermark and list(observer._t_ring)==history
    output=observer.update(watermark+.005,np.zeros(6),raw)
    np.testing.assert_allclose(output[1],first[1],atol=1e-12,rtol=0)
    before=observer._lpf_zi.copy()
    observer.configure_source_period(None)
    np.testing.assert_array_equal(observer._lpf_zi,before)


def test_declared_hp_period_changes_coefficients_once_and_preserves_dc_gain():
    from scipy.signal import butter
    controller=law(.01).controller
    b,a=butter(2,controller.cfg.var_damping_omega_c_hz*.02,btype='high')
    np.testing.assert_array_equal(controller._hp_b,b)
    np.testing.assert_array_equal(controller._hp_a,a)
    controller._episode_filter_seed_pending=True
    controller._update_instability_index(4.,source_dt_s=.01)
    assert controller.instability_index==pytest.approx(0.,abs=1e-12)
    controller.instability_index=1.
    controller._update_instability_index(4.,source_dt_s=.01)
    assert controller.instability_index==pytest.approx(controller.cfg.var_damping_lambda**2,abs=1e-12)


def test_ingress_source_age_never_refreshes_on_relay_republication():
    from peirastic.contact_qp.runtime_source import SourceClock
    clock=SourceClock(.01,jitter_fraction=.25,max_age_s=.02)
    first=clock.observe('source-A',1.,100,now_s=1.)
    held=clock.observe('source-A',1.,100,now_s=1.005)
    assert first.fresh and not held.fresh
    assert held.sample_id==first.sample_id and held.age_s==pytest.approx(.005)
    assert clock.source_updates==1
    next_sample=clock.observe('source-A',1.01,110,now_s=1.01)
    assert next_sample.fresh and next_sample.source_dt_s==pytest.approx(.01)
    with pytest.raises(ValueError,match='hold age'):clock.observe('source-A',1.01,110,now_s=1.031)
    with pytest.raises(ValueError,match='filter timebase'):clock.observe('source-A',1.03,130,now_s=1.03)
    with pytest.raises(ValueError,match='epoch'):clock.observe('source-B',1.02,120,now_s=1.02)
