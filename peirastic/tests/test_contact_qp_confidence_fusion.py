"""No hardware: actual force transactions plus the experimental component law."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from peirastic.contact_qp.confidence_fusion import (
    ConfidenceAngularTask, FEATURE_VERSION, accepted_loading, cop_preference, task_components)
from peirastic.contact_qp.command_budget import CommandBudget
from peirastic.contact_qp.runtime_config import load_study_config, validate_study_config
from peirastic.contact_qp.types import ProbeGeometry
from peirastic.tests.test_contact_qp_solver import observation
from peirastic.tests.test_contact_qp_runtime import make_outer, MemorySink
from peirastic.tests.test_contact_qp_execution_runtime import commit
from peirastic.realman8dof.modes.contact_active import ContactQpOuter

CONFIG = Path('peirastic/config/contact_qp/active_probe50_confidence_cop_tank.yaml')


def task():
    return ConfidenceAngularTask(mass=.051,damping=.22,repair_speed_m_s=.002,lever_m=.031,
        image_x_sign=-1,deadband=.03,c_min=.8,max_velocity=.28,max_acceleration=2.)


def frame(quality=(.5,.8,.95),centroid=.2,**kwargs):
    return replace(observation(quality,now=10.),confidence_centroid_x=centroid,
        confidence_centroid_valid=True,confidence_feature_version=FEATURE_VERSION,**kwargs)


def preview(state,obs,**kwargs):
    return state.preview(obs,rotation_base_tcp=np.eye(3),dt_s=.005,force_gate=1.,enabled=True,**kwargs)


def test_centroid_direction_target_hold_and_no_increment_accumulation():
    state=task();obs=frame()
    first,facts=preview(state,obs)
    assert first<0 and facts['visual_evidence_updated']
    for _ in range(200):
        repeated,info=preview(state,obs)
        assert repeated==first  # refused commands never become dynamics state
        assert not info['visual_evidence_updated']
    v=np.zeros(6)
    for _ in range(600):
        v[4],info=preview(state,obs);state.commit(v,np.eye(3))
    assert v[4]==pytest.approx(info['visual_gated_target_rad_s'],rel=1e-5)
    assert abs(v[4])<.02
    reverse=frame((.95,.8,.5),-.2,frame_seq=obs.frame_seq+1)
    value,info=preview(state,reverse)
    assert value>v[4] and info['visual_requested_omega_rad_s']>0


def test_healthy_or_paused_image_decelerates_without_zero_moment_target():
    state=task();state.commit([0,0,0,0,-.02,0],np.eye(3))
    omega,facts=preview(state,frame((.8,.8,.99),.3))
    assert -.02<omega<0 and facts['visual_requested_omega_rad_s']==0
    paused,_=preview(state,None)
    assert paused==omega


def test_angular_history_is_transformed_before_comparison():
    state=task();rot=Rotation.from_euler('z',90,degrees=True).as_matrix()
    state.commit([0,0,0,0,.02,0],rot)
    omega,facts=preview(state,frame((.9,.9,.9),0))
    assert abs(facts['visual_previous_final_omega_rad_s'])<1e-16
    assert abs(omega)<1e-16


def test_force_gate_changes_target_not_consumed_frame_or_additive_velocity():
    state=task();obs=frame()
    omega,_=preview(state,obs);state.commit([0,0,0,0,omega,0],np.eye(3))
    next_omega,info=state.preview(obs,rotation_base_tcp=np.eye(3),dt_s=.005,force_gate=0.,enabled=True)
    assert omega<next_omega<0
    assert info['visual_gated_target_rad_s']==0 and not info['visual_evidence_updated']


def test_cop_is_geometric_load_center_not_identified_stiffness():
    geometry=ProbeGeometry(half_length_m=.025)
    args=dict(geometry=geometry,contact=True,minimum_force_n=.8)
    assert cop_preference([0,0,4,0,.04,0],**args)==(-.01,'valid')
    assert cop_preference([0,0,.2,0,.04,0],**args)==(0.,'insufficient_compressive_load')
    assert cop_preference([0,0,4,0,.2,0],**args)==(0.,'outside_face')
    candidate=np.array([0,0,.001+(-.01)*(-.02),0,-.02,0])
    for _ in range(100):
        assert accepted_loading(candidate,.001,-.01)==pytest.approx(.001)
    assert accepted_loading([0,0,0,0,0,0],.001,0)==0


def budget():
    return CommandBudget(.1,.15,.05,max_command_interval_s=.05,
        settlement_port='logical_final_model',wrench_convention='negative_control_raw_tcp_v1',
        task_power_source='loading_scan_v1')


def test_supply_excludes_visual_even_after_many_successful_publications():
    tank=budget();wrench=np.array([0,0,4,0,-.04,0])
    # Pure rocking spends energy; inherited rocking never becomes task supply.
    final=np.array([0,0,0,0,-.02,0]);nominal=final.copy()
    loading,scan=task_components(nominal)
    for i in range(20):
        now=10+i*.005
        bound=tank.snapshot(now_s=now,wrench_control_raw=wrench,rotation_base_tcp=np.eye(3),
            loading_twist_tool=loading,scan_twist_tool=scan)
        assert bound.task_power_w==0
        assert tank.reserve(i,bound,final,now_s=now)
        tank.publication_started(i)
        tank.commit(i,final,now_s=now,rotation_base_tcp=np.eye(3),dual_success=True)
    assert tank.balance_j<.1 and tank.cumulative_task_source_used_j==0
    with pytest.raises(ValueError,match='mixed nominal'):
        tank.snapshot(now_s=10.1,wrench_control_raw=wrench,rotation_base_tcp=np.eye(3),nominal_twist_tool=nominal)


@pytest.mark.parametrize('kind',('loading','scan'))
def test_supply_rejects_angular_contamination(kind):
    tank=budget();loading=np.zeros(6);scan=np.zeros(6)
    (loading if kind=='loading' else scan)[4]=.01
    with pytest.raises(ValueError,match='rocking'):
        tank.snapshot(now_s=10,wrench_control_raw=np.ones(6),rotation_base_tcp=np.eye(3),
            loading_twist_tool=loading,scan_twist_tool=scan)


def test_loading_scan_source_rejects_overflow_before_rectification():
    tank=budget()
    with pytest.raises(ValueError,match='nonfinite loading/scan'):
        tank.snapshot(now_s=10,wrench_control_raw=[0,0,-1e200,0,0,0],rotation_base_tcp=np.eye(3),
            loading_twist_tool=[0,0,1e200,0,0,0],scan_twist_tool=np.zeros(6))


def test_loading_scan_sources_are_separately_accounted_without_visual_credit():
    tank=budget();loading=np.array([0,0,.001,0,0,0]);scan=np.array([.002,0,0,0,0,0])
    source=tank.snapshot(now_s=10,wrench_control_raw=[2,0,4,0,0,0],rotation_base_tcp=np.eye(3),
        loading_twist_tool=loading,scan_twist_tool=scan)
    final=loading+scan
    assert source.task_power_w==pytest.approx(.008)
    assert tank.reserve(0,source,final,now_s=10)
    tank.publication_started(0)
    tank.commit(0,final,now_s=10,rotation_base_tcp=np.eye(3),dual_success=True)
    tank.advance(10.01)
    work=next(e for e in tank.drain_events() if e['event']=='logical_epoch_work')
    assert work['loading_source_used_j']==pytest.approx(.00004)
    assert work['scan_source_used_j']==pytest.approx(.00004)
    assert tank.balance_j==pytest.approx(.1)


def runtime(monkeypatch,policy='confidence_cop_v1'):
    cfg=load_study_config(CONFIG);cfg['qp']['allocation_policy']=policy
    baseline,pose=make_outer();sink=MemorySink();clock=[10.]
    monkeypatch.setattr('peirastic.realman8dof.modes.contact_active.time.monotonic',lambda:clock[0])
    active=ContactQpOuter(baseline,cfg,sink=sink,
        feature_receiver=SimpleNamespace(observation=None,close=lambda **kw:None))
    active.set_origin(pose)
    active.features.observation=replace(frame(),registration_version=active.registration_version,
        window_version=active.feature_config.window_version,calibration_version=active.feature_config.calibration_version)
    return active,pose,clock,sink


def step(active,pose,clock,*,moment=.04,force=4.):
    w=np.array([0.,0.,force,0.,moment,0.])
    return active.sample(0.,pose,w,f_ext_raw=w,dt_actual=.005,
        wrench_source_id='A',wrench_source_time_s=clock[0],wrench_source_wall_time_ns=int(clock[0]*1e6))


def test_real_transaction_uses_visual_final_history_and_unmixed_loading(monkeypatch):
    active,pose,clock,sink=runtime(monkeypatch)
    try:
        # This fixture fixes measured pose; beyond acquisition it deliberately
        # drives the real force law into a retract/envelope conflict. Exercise
        # accepted component transactions here, not a fictitious contact plant.
        for i in range(8):
            command=step(active,pose,clock)
            wanted=active._pending_fusion['loading_requested_tool'][2]
            cp=active._pending_fusion['cop_preference_m']
            # Model a final IK normal residual along the rail: it is accounted
            # by the tank, but cannot contaminate outer loading integration.
            final=command.copy();final[2]+=.0001;final[4]*=.9
            commit(active,final,clock)
            assert active.controller.v_force_cmd_z==pytest.approx(accepted_loading(command,wanted,cp))
            np.testing.assert_allclose(active._visual_task.omega_base,[0,final[4],0])
            assert active.controller.last_v_cmd[4]==0
            assert active.command_budget.active.nominal_twist_tool[4]==0
            clock[0]+=.005
        angular_history=active._visual_task.omega_base.copy();balance=active.command_budget.balance_j
        command=step(active,pose,clock)
        active.publication_abort('test_unsent',definitely_not_sent=True)
        np.testing.assert_array_equal(active._visual_task.omega_base,angular_history)
        assert active.command_budget.balance_j<=balance+1e-8  # no reset or synthetic refund
        assert any(r['event']=='fusion_publication' for r in sink.records)
    finally:active.close()


def test_opposing_moment_cannot_reverse_image_target(monkeypatch):
    outputs=[]
    for moment in (-.04,.04):
        active,pose,clock,sink=runtime(monkeypatch,'confidence_angular_v1')
        try:
            for i in range(8):
                command=step(active,pose,clock,moment=moment);commit(active,command,clock);clock[0]+=.005
            outputs.append(command[4])
        finally:active.close()
    assert outputs[0]<0 and outputs==pytest.approx([outputs[0]]*2,abs=1e-8)


def test_outer_angular_slew_reconciles_final_history_without_rail_feedback(monkeypatch):
    active,pose,clock,sink=runtime(monkeypatch)
    try:
        for _ in range(3):
            command=step(active,pose,clock);commit(active,command,clock);clock[0]+=.005
        active._previous[4]=.20  # previously requested outer rotation
        active._visual_task.omega_base=np.array([0.,.10,0.])  # admitted final model
        z_history=active._previous[2]
        command=step(active,pose,clock)
        assert abs(command[4]-.1)<=.01+1e-8
        recorded=next(r for r in reversed(sink.records) if r['event']=='control_sample')
        assert recorded['previous_outer_command_tool'][4]==.1
        assert recorded['previous_outer_command_tool'][2]==z_history
        active.publication_abort('fixture',definitely_not_sent=True)
    finally:active.close()


def test_new_profiles_fail_closed_on_old_worker_or_mixed_supply():
    cfg=load_study_config(CONFIG)
    assert validate_study_config(cfg)['allocation_policy']=='confidence_cop_v1'
    for mutate in ('worker','supply','geometry'):
        bad=load_study_config(CONFIG)
        if mutate=='worker':bad['feature'].pop('confidence_feature_version')
        elif mutate=='supply':bad['energy']['task_power_source']='nominal_command'
        else:bad['geometry']['T_tcp_face'][0][3]=.01
        with pytest.raises(ValueError):validate_study_config(bad)


def test_new_mode_preserves_fresh_six_newton_supervisor(monkeypatch):
    active,pose,clock,sink=runtime(monkeypatch)
    try:
        with pytest.raises(RuntimeError,match='6 N'):
            step(active,pose,clock,force=6.)
        assert active.command_budget.active is None
        assert not active._nominal_pending
        assert any(r['event']=='force_supervisor_stop' for r in sink.records)
    finally:active.close()


def test_new_mode_stale_image_with_fresh_force_keeps_current_transaction(monkeypatch):
    active,pose,clock,sink=runtime(monkeypatch)
    try:
        for _ in range(3):
            command=step(active,pose,clock);commit(active,command,clock);clock[0]+=.005
        reference=active.reference_time_s
        active.features.observation=replace(active.features.observation,
            effective_time_s=clock[0]-.303,received_time_s=clock[0]-.1)
        command=step(active,pose,clock)
        assert active._pending_fusion['visual_reason']=='image_paused'
        assert active._pending_fusion['visual_requested_omega_rad_s']==0.
        commit(active,command,clock)
        assert active.reference_time_s>reference
        assert active.command_budget.latched_reason is None
        clock[0]+=.005
        active.features.observation=replace(active.features.observation,
            frame_seq=active.features.observation.frame_seq+1,
            effective_time_s=clock[0],received_time_s=clock[0])
        command=step(active,pose,clock)
        assert active._pending_fusion['visual_evidence_updated']
        commit(active,command,clock)
        assert active.command_budget.latched_reason is None
    finally:active.close()
