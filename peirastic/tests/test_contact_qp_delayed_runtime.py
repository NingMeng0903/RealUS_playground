"""Real adapter transactions using in-memory image/robot substitutes only."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from peirastic.contact_qp.command_lease import CommandLease
from peirastic.contact_qp.runtime_config import load_study_config,validate_study_config
from peirastic.contact_qp.types import REGION_FEATURE_VERSION, WEAK_SIDE_FEATURE_VERSION
from peirastic.contact_qp.geometry import twist_tcp_to_face,point_normal_row
from scipy.spatial.transform import Rotation
from peirastic.realman8dof.modes.contact_active import ContactQpOuter
from peirastic.tests.test_contact_qp_runtime import make_outer,MemorySink
from peirastic.tests.test_contact_qp_solver import observation


def fixture(monkeypatch,region_count=10):
    config=load_study_config(Path(__file__).parents[1]/'config/contact_qp/active_probe50_delay_kf_cop.yaml')
    from peirastic.contact_qp.features import FeatureConfig
    config['feature']['config']['region_count']=region_count
    config['feature']['region_layout_version']=FeatureConfig(**config['feature']['config']).region_layout_version
    config['image_kf']['channels']=region_count
    baseline,pose=make_outer();sink=MemorySink();clock=[10.]
    monkeypatch.setattr('peirastic.realman8dof.modes.contact_active.time.monotonic',lambda:clock[0])
    receiver=SimpleNamespace(observation=None,close=lambda **kw:None)
    active=ContactQpOuter(baseline,config,sink=sink,feature_receiver=receiver)
    active.set_origin(pose)
    receiver.observation=frame(active,clock[0],0,(.65,.9))
    return active,pose,clock,sink


def frame(active,now,seq,quality):
    quality=np.asarray(quality,dtype=float)
    if quality.shape==(2,):quality=np.repeat(quality,active.feature_config.region_count//2)
    return replace(observation((.9,.9,.9),now=now),frame_seq=seq,
        effective_time_s=now-.15,received_time_s=now,
        registration_version=active.registration_version,window_version=active.feature_config.window_version,
        calibration_version=active.feature_config.calibration_version,
        confidence_lr=np.ones(2),confidence_lr_valid=np.ones(2,dtype=bool),weakside_feature_version=WEAK_SIDE_FEATURE_VERSION,
        region_confidence=quality,region_valid=np.ones(active.feature_config.region_count,dtype=bool),
        region_edges=active.feature_config.region_edges,region_feature_version=REGION_FEATURE_VERSION,
        region_layout_version=active.feature_config.region_layout_version,timestamp_semantics='effective_image_time')


def sample(active,pose,clock,*,force=4.,torque=0.,source=None,dt=.005,wrench=None):
    stamp=clock[0] if source is None else source
    wrench=np.array([0.,0.,force,0.,torque,0.]) if wrench is None else np.asarray(wrench)
    return active.sample(0.,pose,wrench,f_ext_raw=wrench,dt_actual=dt,
        wrench_source_id='A',wrench_source_time_s=stamp,wrench_source_wall_time_ns=int(stamp*1e6))


def commit(active,command,clock):
    active.rocking_constraints()
    facts=dict(rocking_policy_tier=4,rocking_limited=True)
    assert active.publication_review(active.pending_id,command,now_s=clock[0],facts=facts)
    assert active.publication_started(active.pending_id)
    active.publication_commit(active.pending_id,command,now_s=clock[0],facts={'arm':'sent','rail':'sent'})


def test_default_delayed_profile_uses_no_energy_and_requires_new_capability():
    from peirastic.core.capabilities import study_capabilities,DELAY_KF_COP_CAPABILITY
    config=load_study_config(Path(__file__).parents[1]/'config/contact_qp/active_probe50_delay_kf_cop.yaml')
    facts=validate_study_config(config)
    assert facts['configuration_valid']
    assert facts['command_energy_budget_enabled'] is False
    assert DELAY_KF_COP_CAPABILITY in study_capabilities(config)
    assert not any('power' in cap or 'budget' in cap for cap in study_capabilities(config))
    config['energy_constraint_enabled']=True
    with pytest.raises(ValueError,match='removes energy'):
        validate_study_config(config)


def test_runtime_uses_delay_prediction_new_qp_and_separate_nominal_publication_history(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        assert active.command_budget is None and active.energy is None
        assert active.command_power_constraints()=={}
        assert active._visual_task is None
        for i in range(12):
            active.features.observation=frame(active,clock[0],i,(.65,.9))
            command=sample(active,pose,clock,torque=.06)
            mechanical=active._pending_mechanical_tcp.copy()
            assert 'visual_velocity_target_rad_s' not in active._preview.context
            assert active.pending_result.diagnostics['allocation_policy']=='delay_kf_cop_v1'
            assert active._pending_prediction.valid
            assert active._pending_fusion['cop_preference_m']==pytest.approx(-.06/4.)
            assert active._pending_fusion['visual_request_rad_s']<0  # left is face+x; negative Omega loads it
            commit(active,command,clock)
            np.testing.assert_allclose(active.controller.last_v_cmd,mechanical,atol=1e-14)
            np.testing.assert_array_equal(active._previous,command)
            assert active.nominal.tilt.omega_y==pytest.approx(mechanical[4])
            assert active._command_lease.active.expires_s==pytest.approx(clock[0]+.05)
            clock[0]+=.005
        assert any(abs(r['nominal_twist_tool'][4])>0 for r in sink.records if r['event']=='control_sample')
        assert not any(r['event']=='logical_command_energy' for r in sink.records)
        regional=[r for r in sink.records if r['event']=='final_regional_visual_task']
        assert len(regional)==12 and all(r['region_count']==10 for r in regional)
        for row in regional:
            expected=max(abs(row['gated_request_rad_s'])-np.sign(row['gated_request_rad_s'])*row['final_contact_omega_rad_s'],0.)
            assert row['shortfall_rad_s']==pytest.approx(expected)
        reasons=set().union(*(item['reasons'] for item in active._quality_intervals.active.values()))
        assert 'regional_deficit_moment' in reasons
        assert 'request_zero_or_deadband' not in reasons
    finally:active.close()


def test_failed_proposal_keeps_kf_measurement_once_and_old_lease_expiry(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        command=sample(active,pose,clock);commit(active,command,clock)
        expiry=active._command_lease.active.expires_s
        reference=active.reference_time_s
        previous=active._previous.copy()
        clock[0]+=.005
        fresh=frame(active,clock[0],1,(.5,.9));active.features.observation=fresh
        sample(active,pose,clock)
        active.publication_abort('not sent',definitely_not_sent=True)
        assert active.reference_time_s==reference
        np.testing.assert_array_equal(active._previous,previous)
        assert active._command_lease.active.expires_s==expiry
        first=[r for r in sink.records if r['event']=='image_kf_measurement' and r['feature']['frame_seq']==1]
        assert first[-1]['accepted']
        clock[0]+=.005
        sample(active,pose,clock)
        last=[r for r in sink.records if r['event']=='image_kf_measurement' and r['feature']['frame_seq']==1][-1]
        assert not last['accepted'] and last['reason']=='duplicate_frame'
        active.publication_abort('not sent',definitely_not_sent=True)
    finally:active.close()


def test_stale_and_unknown_confidence_withdraw_visual_without_delaying_force(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        old=frame(active,clock[0]-.2,0,(.3,.9));active.features.observation=old
        sample(active,pose,clock)
        assert not active._pending_prediction.valid
        assert active.pending_result.diagnostics['visual_rows_active'] is False
        assert active._pending_alpha_preferred==.25
        assert active._source_step.source_t_s==clock[0]
        active.publication_abort('not sent',definitely_not_sent=True)
        clock[0]+=.005
        active.features.observation=replace(frame(active,clock[0],1,(.3,.9)),region_valid=[True]*4+[False]+[True]*5)
        sample(active,pose,clock)
        assert active._pending_prediction.reason=='invalid_windows'
        assert active.pending_result.diagnostics['visual_rows_active'] is False
        active.publication_abort('not sent',definitely_not_sent=True)
    finally:active.close()


def test_final_mechanical_violation_and_partial_publication_never_commit_reference(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        command=sample(active,pose,clock);active.rocking_constraints()
        invalid=command.copy();invalid[4]=1.
        assert not active.publication_review(active.pending_id,invalid,now_s=clock[0],
            facts=dict(rocking_policy_tier=4))
        assert active._publication_rejection_reason=='final_mechanical_interval_violation'
        assert active.publication_review(active.pending_id,command,now_s=clock[0],facts=dict(rocking_policy_tier=4))
        assert active.publication_started(active.pending_id)
        before=active.reference_time_s
        with pytest.raises(RuntimeError,match='partial_publication'):
            active.publication_commit(active.pending_id,command,now_s=clock[0],facts={'arm':'sent','rail':'failed'})
        assert active.reference_time_s==before
        assert active._command_lease.latched_reason
        active.publication_abort('partial',definitely_not_sent=False)
    finally:active.close()


def test_delayed_runtime_oos_batch_reaches_kf_before_latest_filter(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        newer=frame(active,clock[0],2,(.6,.9))
        earlier=replace(frame(active,clock[0],1,(.5,.9)),effective_time_s=clock[0]-.18)
        active.features.observation=newer
        active.features.drain_observations=lambda:(newer,earlier)
        sample(active,pose,clock)
        records=[r for r in sink.records if r['event']=='image_kf_measurement']
        assert [r['feature']['frame_seq'] for r in records]==[2,1]
        assert all(r['accepted'] for r in records)
        assert records[-1]['latest_replayed_frame_seq']==2
        assert records[-1]['latest_replayed_effective_time_s']==newer.effective_time_s
        assert 'innovation' not in records[-1]
        active.publication_abort('not sent',definitely_not_sent=True)
    finally:active.close()


def test_command_lease_is_never_extended_by_unsent_retry_or_late_commit():
    lease=CommandLease(.05)
    assert lease.reserve(1,created_s=1.,now_s=1.001)
    assert lease.publication_started(1,now_s=1.002)
    lease.commit(1,now_s=1.003,dual_success=True)
    assert lease.active.expires_s==1.05
    assert lease.reserve(2,created_s=1.02,now_s=1.021)
    lease.reject_new_only(definitely_not_sent=True,now_s=1.022)
    assert lease.active.expires_s==1.05
    assert not lease.advance(1.05)
    assert not lease.reserve(3,created_s=1.05,now_s=1.05)


def test_command_lease_crossing_old_expiry_after_valid_dispatch_keeps_reserved_expiry():
    lease=CommandLease(.05)
    assert lease.reserve(1,created_s=1.,now_s=1.)
    assert lease.publication_started(1,now_s=1.)
    lease.commit(1,now_s=1.,dual_success=True)
    assert lease.reserve(2,created_s=1.04,now_s=1.04)
    assert lease.publication_started(2,now_s=1.049)
    lease.commit(2,now_s=1.06,dual_success=True)
    assert lease.active.expires_s==1.09


@pytest.mark.parametrize('image_sign',[-1,1])
@pytest.mark.parametrize('weak_side',[0,1])
def test_visual_request_increases_actual_weak_side_point_indentation(monkeypatch,image_sign,weak_side):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        active.geometry=replace(active.geometry,image_x_sign=image_sign)
        quality=[.9,.9];quality[weak_side]=.5
        active.features.observation=frame(active,clock[0],0,quality)
        sample(active,pose,clock)
        fusion=active._pending_fusion
        omega=fusion['visual_request_rad_s']
        x=np.asarray(fusion['region_task']['region_centers_m'])[[0,-1]]
        contact_twist=np.array([0.,0.,0.,0.,omega,0.])
        tcp_twist=np.linalg.solve(twist_tcp_to_face(active.geometry),contact_twist)
        indentation=np.array([point_normal_row(active.geometry,location) @ tcp_twist for location in x])
        assert indentation[weak_side]>0
        assert indentation[1-weak_side]<0
        assert len(fusion['region_task']['region_deficits'])==10
        active.publication_abort('not sent',definitely_not_sent=True)
    finally:active.close()


@pytest.mark.parametrize('rotated',[False,True])
def test_noncoincident_nominal_wrench_pose_and_cop_share_contact_frame(monkeypatch,rotated):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        t=np.eye(4);t[:3,3]=[.012,-.015,.08]
        if rotated:t[:3,:3]=Rotation.from_euler('xyz',[.15,-.2,.3]).as_matrix()
        active.geometry=replace(active.geometry,T_tcp_face=t)
        active._preview.geometry=active.geometry
        active.baseline.position.cfg.track_axes=np.ones(6)
        active.set_origin(pose)
        control_contact=np.array([1.,-.3,4.,0.,0.,0.])
        control_tcp=twist_tcp_to_face(active.geometry).T @ control_contact
        assert abs(control_tcp[4])>.01  # old TCP moment contains pure shift/rotation
        sample(active,pose,clock,wrench=control_tcp)
        np.testing.assert_allclose(active._pending_fusion['environment_wrench_contact'],-control_contact,atol=1e-14)
        assert active.nominal.tilt.tau_y==pytest.approx(0.,abs=1e-14)
        assert active._pending_fusion['cop_preference_m']==pytest.approx(0.,abs=1e-14)
        expected_rotation=Rotation.from_euler(active.controller.cfg.euler_order,pose[3:]).as_matrix() @ t[:3,:3]
        actual_rotation=Rotation.from_euler(active.controller.cfg.euler_order,active.nominal.tilt.measured_pose_euler).as_matrix()
        np.testing.assert_allclose(actual_rotation,expected_rotation,atol=1e-14)
        np.testing.assert_allclose(twist_tcp_to_face(active.geometry) @ active._pending_mechanical_tcp,
                                   active._pending_mechanical_contact,atol=1e-14)
        active.publication_abort('not sent',definitely_not_sent=True)
    finally:active.close()


def test_delayed_dispatch_rechecks_fresh_force_and_retry_cannot_extend_old_lease(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        command=sample(active,pose,clock);commit(active,command,clock)
        expiry=active._command_lease.active.expires_s
        clock[0]+=.005
        command=sample(active,pose,clock);active.rocking_constraints()
        assert active.publication_review(active.pending_id,command,now_s=clock[0],facts=dict(rocking_policy_tier=4))
        clock[0]+=.015
        assert not active.publication_started(active.pending_id)
        assert active._publication_rejection_reason=='dispatch_certificate_expired'
        active.publication_abort('not sent',definitely_not_sent=True)
        clock[0]+=1e-6
        assert active.retry_unsent_publication()
        assert active._command_lease.active.expires_s==expiry
        clock[0]=expiry
        assert not active.retry_unsent_publication()
        assert active._command_lease.active.expires_s==expiry
    finally:active.close()


def test_live_command_lease_permits_bounded_force_gap_and_no_reference_catchup(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        command=sample(active,pose,clock);commit(active,command,clock)
        reference=active.reference_time_s
        expiry=active._command_lease.active.expires_s
        clock[0]+=.025
        sample(active,pose,clock,dt=.025)
        assert active._source_step.gap_recovered
        assert active._source_step.gap_context.lease_expires_s==expiry
        assert active.reference_time_s==reference
        active.publication_abort('not sent',definitely_not_sent=True)
        assert active.reference_time_s==reference
    finally:active.close()


def test_realised_progress_subtracts_feedback_and_permits_task_residual(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        # Advance the path away from its initial ramp point to expose nonzero
        # path and feedback. Both are present in the final affine velocity.
        active.reference_time_s=.15
        moved=active.reference.reference.sample(.15).pose_d.copy();moved[1]-=.00001
        command=sample(active,moved,clock)
        assert np.linalg.norm(active.pending_basis[:,2])>0
        assert np.linalg.norm(active.pending_offset)>0
        alpha=active.pending_result.alpha*.5
        solution=active.pending_result.diagnostics['solution'].copy()
        solution[2]=alpha
        final=active.pending_offset+active.pending_basis @ solution[:3]
        # Orthogonal task tracking error remains below physical speed/acc caps.
        final[0]+=.0001
        active.rocking_constraints()
        assert active.publication_review(active.pending_id,final,now_s=clock[0],facts=dict(rocking_policy_tier=4))
        assert active.publication_started(active.pending_id)
        active.publication_commit(active.pending_id,final,now_s=clock[0],facts={'arm':'sent','rail':'sent'})
        record=[r for r in sink.records if r['event']=='publication'][-1]
        assert record['accepted_alpha']==pytest.approx(alpha,abs=1e-9)
        assert record['accepted_alpha']<=active.pending_result.alpha
        assert active.reference_time_s==pytest.approx(.15+alpha*.005)
    finally:active.close()


def test_last_dispatch_gate_rejects_expired_certificate_while_force_is_fresh(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        command=sample(active,pose,clock);active.rocking_constraints()
        clock[0]+=.009
        assert active.publication_review(active.pending_id,command,now_s=clock[0],facts=dict(rocking_policy_tier=4))
        clock[0]+=.003
        assert clock[0]-active._source_step.source_t_s<.015
        assert not active.publication_started(active.pending_id)
        assert active._publication_rejection_reason=='dispatch_certificate_expired'
        active.publication_abort('not sent',definitely_not_sent=True)
    finally:active.close()


def test_contact_face_force_supervisor_stops_when_tool_z_is_below_six(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        t=np.eye(4);t[:3,:3]=Rotation.from_euler('y',.7).as_matrix()
        active.geometry=replace(active.geometry,T_tcp_face=t);active._preview.geometry=active.geometry
        control_tcp=twist_tcp_to_face(active.geometry).T @ np.array([0.,0.,6.1,0.,0.,0.])
        assert control_tcp[2]<6.
        with pytest.raises(RuntimeError,match='6 N'):
            sample(active,pose,clock,wrench=control_tcp)
        stop=[r for r in sink.records if r['event']=='force_supervisor_stop'][-1]
        assert stop['filtered_contact_force_n']==pytest.approx(6.1)
        assert stop['primary_force_frame']=='contact_face'
        assert stop['additional_stop_frame']=='tcp_tool'
    finally:active.close()


def test_visual_force_gate_uses_contact_face_load(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        t=np.eye(4);t[:3,:3]=Rotation.from_euler('y',.3).as_matrix()
        active.geometry=replace(active.geometry,T_tcp_face=t);active._preview.geometry=active.geometry
        control_tcp=twist_tcp_to_face(active.geometry).T @ np.array([0.,0.,4.25,0.,0.,0.])
        sample(active,pose,clock,wrench=control_tcp)
        assert active.pending_result.diagnostics['visual_gamma']==pytest.approx(.5)
        assert active._pending_fusion['filtered_contact_force_n']==pytest.approx(4.25)
        active.publication_abort('not sent',definitely_not_sent=True)
    finally:active.close()


@pytest.mark.parametrize('region_count,bad',[(10,(4,5)),(10,(0,3,6,9)),(4,(1,2)),(12,(5,6))])
def test_runtime_detects_all_regional_defects_despite_healthy_legacy_statistics(monkeypatch,region_count,bad):
    active,pose,clock,sink=fixture(monkeypatch,region_count)
    try:
        quality=np.full(region_count,.9);quality[list(bad)]=.3
        active.features.observation=frame(active,clock[0],0,quality)
        assert np.all(active.features.observation.quality>.8)
        assert np.all(active.features.observation.confidence_lr>.8)
        sample(active,pose,clock)
        task=active._pending_fusion['region_task']
        assert task['bad_region_indices']==bad
        assert active._pending_prediction.state.shape==(2*region_count,)
        assert active._pending_fusion['visual_request_rad_s']==0.
        assert active._pending_alpha_preferred==pytest.approx(1.-.75*(active.solver.config.c_min-.3)/active.solver.config.c_min)
        assert active.pending_result.diagnostics['visual_rows_active'] is False
        active.publication_abort('not sent',definitely_not_sent=True)
    finally:active.close()


@pytest.mark.parametrize('mismatch',['legacy_only','wrong_layout'])
def test_runtime_refuses_legacy_only_or_mismatched_regional_evidence(monkeypatch,mismatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        obs=frame(active,clock[0],0,(.3,.9))
        if mismatch=='legacy_only':
            obs=replace(obs,region_confidence=None,region_valid=None,region_edges=None,
                region_feature_version=None,region_layout_version=None)
        else:obs=replace(obs,region_layout_version='wrong-layout')
        active.features.observation=obs
        sample(active,pose,clock)
        assert not active._pending_prediction.valid
        assert active._pending_alpha_preferred==.25
        assert active.pending_result.diagnostics['visual_rows_active'] is False
        active.publication_abort('not sent',definitely_not_sent=True)
    finally:active.close()


def test_delayed_profile_rejects_nonregional_feature_algorithm():
    from peirastic.contact_qp.features import FeatureConfig
    config=load_study_config(Path(__file__).parents[1]/'config/contact_qp/active_probe50_delay_kf_cop.yaml')
    config['feature']['config']['algorithm_version']='randomwalk_camp_bmode_v2'
    fc=FeatureConfig(**config['feature']['config'])
    config['feature']['window_version']=fc.window_version
    config['feature']['region_layout_version']=fc.region_layout_version
    with pytest.raises(ValueError,match='regional delay KF requires'):
        validate_study_config(config)
