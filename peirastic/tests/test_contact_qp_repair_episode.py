from dataclasses import replace
import math
import numpy as np
import pytest
from peirastic.contact_qp.repair_episode import RepairEpisode
from peirastic.tests.test_contact_qp_solver import observation,datum
from peirastic.contact_qp.qp import ContactQp,QpConfig
from peirastic.contact_qp.types import TwistConstraints


def update(state,t,angle=0.,*,quality=(.3,.8,.9),seq=None,valid=True,gate=1.,enabled=True,version=None,reference_reset=False):
    obs=observation(quality,now=t,seq=round(t*100)+1 if seq is None else seq)
    if version:obs=replace(obs,window_version=version)
    return state.update(now_s=t,measured_angle=angle,observation=obs,image_valid=valid,
        c_min=.5,force_gate=gate,execution_enabled=enabled,angle_reference_reset=reference_reset)


def test_permission_elapsed_bounds_persistent_bad_and_never_refunds():
    state=RepairEpisode()
    assert update(state,0.)['repair_allowed']
    assert update(state,1.)['repair_allowed']
    end=update(state,2.);assert end['exhausted'] and not end['repair_allowed']
    for i in range(3,8):
        facts=update(state,i,seq=3,valid=i%2==0,version='new_version')
        assert facts['exhausted'] and facts['permission_elapsed_s']==2.


def test_angle_counts_total_measured_travel_during_pause_and_nominal_rebasing():
    state=RepairEpisode();update(state,0.)
    update(state,.1,.02)
    paused=update(state,.2,.03,gate=0.)
    assert paused['permission_elapsed_s']==pytest.approx(.2)
    exhausted=update(state,.8,.055,gate=0.,valid=False)
    assert exhausted['permission_elapsed_s']==pytest.approx(.2)
    assert exhausted['measured_angle_travel_rad']==pytest.approx(.055)
    assert exhausted['exhausted']
    reset=update(state,.9,0.,reference_reset=True)
    assert reset['measured_angle_travel_rad']==pytest.approx(.055)
    assert reset['exhausted']  # changing nominal/angle origin is not a refund


def test_seek_does_not_arm_and_three_distinct_healthy_frames_restore():
    state=RepairEpisode()
    for i in range(4):assert not update(state,i,angle=i*.1,enabled=False)['armed']
    update(state,4.,.4);update(state,6.,.4)
    first=update(state,6.1,.4,quality=(.9,.9,.9),seq=700)
    assert first['exhausted']
    for t in (6.11,6.12):assert update(state,t,.4,quality=(.9,.9,.9),seq=700)['exhausted']
    assert update(state,6.2,.4,quality=(.9,.9,.9),seq=701)['exhausted']
    fresh=update(state,6.3,.4,quality=(.9,.9,.9),seq=702)
    assert not fresh['exhausted'] and fresh['healthy_resets']==1
    assert update(state,6.4,.4)['repair_allowed']


def test_invalid_version_bad_and_reversed_time_break_health_confirmation():
    state=RepairEpisode();update(state,0.);update(state,2.)
    update(state,2.1,quality=(.9,.9,.9))
    update(state,2.2,valid=False)
    assert update(state,2.3,quality=(.9,.9,.9))['healthy_confirmation_count']==1
    assert update(state,2.4,quality=(.9,.9,.9),version='changed')['healthy_confirmation_count']==1
    with pytest.raises(ValueError):update(state,2.3)
    assert state.permission_elapsed_s==2. and state.healthy_count==0
    with pytest.raises(ValueError):update(state,2.4,.1)
    assert state.angle_travel_rad==0.


def test_zero_request_cannot_manufacture_residual_to_pay_alpha():
    nominal=np.array([.005,0.,-.003,0.,-.01,0.]);path=nominal.copy();path[[2,4]]=0.
    mech=TwistConstraints(np.array([[1.,0,0,0,0,0]]),np.array([-1.]),np.array([.001]),valid_until_s=2.)
    data=datum((.4,.8,1.),nominal_twist=nominal,path_twist=path,previous_twist=nominal,force_n=4.6,mechanical=mech)
    qp=ContactQp(QpConfig(allocation_policy='differential_repair_v8',max_acceleration=np.full(6,100.)))
    result=qp.solve(data)
    assert result.qp_twist is not None
    assert result.diagnostics['differential_request_m_s']==0
    assert result.diagnostics['differential_auxiliary_shortfall']==pytest.approx(0.,abs=1e-8)


def test_exhausted_episode_removes_bad_window_zero_loading_row():
    qp=ContactQp(QpConfig(allocation_policy='differential_repair_v8',max_acceleration=np.full(6,100.)))
    data=datum((.1,.8,.9),now_s=0.,observation=observation((.1,.8,.9),now=0.))
    assert qp.solve(data).qp_twist is not None
    nominal=data.nominal_twist.copy();nominal[2]=-.001
    expired=replace(data,now_s=2.,nominal_twist=nominal,previous_twist=nominal,
                    observation=observation((.1,.8,.9),now=2.,seq=2))
    result=qp.solve(expired)
    assert result.qp_twist is not None
    assert result.diagnostics['repair_episode']['exhausted']
    assert result.diagnostics['visual_window_active'].tolist()==[False,True]
    assert result.qp_twist[2]==pytest.approx(-.001,abs=1e-7)
    assert result.diagnostics['acquisition_loss']>0
    assert result.diagnostics['differential_request_m_s']==0.


def test_real_adapter_air_without_gate_does_not_start_visual_episode(monkeypatch):
    from peirastic.tests.test_contact_qp_active import make_active
    active,pose,clock,_=make_active(monkeypatch)
    active.solver=ContactQp(replace(active.solver.config,allocation_policy='differential_repair_v8'))
    obs=observation((.1,.8,.9),now=clock[0])
    active.features.observation=replace(obs,registration_version=active.registration_version,
        window_version=active.feature_config.window_version,calibration_version=active.feature_config.calibration_version)
    try:
        assert active.contact_gate is None
        active.sample(0.,pose,np.zeros(6),f_ext_raw=np.zeros(6),dt_actual=.005,
            wrench_source_id='test-source',wrench_source_time_s=clock[0],wrench_source_wall_time_ns=1)
        assert not active.controller.contact_present
        facts=active.pending_result.diagnostics['repair_episode']
        assert not facts['armed'] and not facts['repair_allowed']
        assert active.pending_result.diagnostics['differential_request_m_s']==0.
        active.publication_abort('not sent',definitely_not_sent=True)
    finally:active.close()


def test_adapter_previously_started_gate_does_not_refund_or_continue_after_contact_loss(monkeypatch):
    from types import SimpleNamespace
    from peirastic.tests.test_contact_qp_active import make_active
    active,pose,clock,_=make_active(monkeypatch)
    active.solver=ContactQp(replace(active.solver.config,allocation_policy='differential_repair_v8'))
    active.contact_gate=SimpleNamespace(started=True,guard_approach=lambda *a:None,
        observe_force=lambda *a,**kw:None,sample=lambda *a:None)
    # Existing measurement-policy history from earlier contact; no model of a
    # force detector is substituted. This sample uses the actual original law.
    state=active.solver.repair_episode
    update(state,clock[0]-.2);update(state,clock[0]-.1,.01)
    obs=observation((.1,.8,.9),now=clock[0])
    active.features.observation=replace(obs,registration_version=active.registration_version,
        window_version=active.feature_config.window_version,calibration_version=active.feature_config.calibration_version)
    try:
        active.sample(0.,pose,np.zeros(6),f_ext_raw=np.zeros(6),dt_actual=.005,
            wrench_source_id='test-source',wrench_source_time_s=clock[0],wrench_source_wall_time_ns=1)
        assert not active.controller.contact_present
        facts=active.pending_result.diagnostics['repair_episode']
        assert facts['armed'] and not facts['repair_allowed']
        assert facts['permission_elapsed_s']==pytest.approx(.2)
        assert facts['measured_angle_travel_rad']>=.01
        active.publication_abort('not sent',definitely_not_sent=True)
        assert state.permission_elapsed_s==pytest.approx(.2)
    finally:active.close()
