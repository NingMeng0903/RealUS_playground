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
from peirastic.contact_qp.execution import ProposalDeferred
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
    assert config['qp']['repair_speed_m_s']==pytest.approx(0.010)


def test_stop_ramp_keeps_outer_alpha_when_path_projection_vanishes(monkeypatch):
    from peirastic.contact_qp.reference import FiniteIntervalReference
    from peirastic.scan_path import ForearmReference, make_spec

    active,pose,clock,sink=fixture(monkeypatch)
    try:
        assert active.solver.config.repair_speed_m_s==pytest.approx(0.010)
        distal=pose.copy();proximal=pose.copy();proximal[1]+=0.12
        source=ForearmReference(make_spec(distal,proximal,'S','PtD',5,speed=0.005))
        active.reference=FiniteIntervalReference(source,dt_s=0.005)
        active.pending_result=SimpleNamespace(alpha=0.9)
        active.pending_h_ref=0.005
        active.reference_time_s=source.duration_s-0.3
        assert active._accepted_progress_alpha(0.,1e-16)==pytest.approx(0.9)
        accepted=active.reference_time_s
        for _ in range(80):
            accepted=active.reference.commit_time(accepted,0.9,0.005)
        assert active.reference.exhaustion_reason(accepted)
        active.reference_time_s=1.0
        assert active._accepted_progress_alpha(0.,1e-16)==0.
        assert active._accepted_progress_alpha(0.,0.)==pytest.approx(0.9)
    finally:
        active.close()


def test_late_region_deficit_escalates_omega_floor_and_caps_alpha(monkeypatch):
    late=np.array([.555,.562,.581,.628,.708,.857,.931,.977,.989,.993])
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        first=frame(active,clock[0],0,late)
        second=frame(active,clock[0]+.31,1,late)
        active._quality_intervals.observe(first,clock[0],())
        active._quality_intervals.observe(second,clock[0]+.31,())
        assert active._quality_intervals.escalation_active
        active._quality_intervals.last_request_sign=-1.
        active.features.observation=second
        sample(active,pose,clock,torque=-0.021)
        assert active._pending_alpha_preferred==0.25
        assert active._pending_fusion['visual_request_rad_s']<=-0.05
        assert active.pending_result.alpha<=0.25+1e-12
    finally:
        active.close()


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
            accepted=float((twist_tcp_to_face(active.geometry)@command)[4])
            assert abs(active.nominal.tilt.omega_y)<=abs(mechanical[4])+1e-12
            if abs(mechanical[4])>1e-6 and abs(accepted)>=0.25*abs(mechanical[4]):
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


@pytest.mark.parametrize('old,updated,omega',[
    ((2.317580979532303e-5,3.992503824979431e-5),
     (-8.88937690719186e-5,1.4955447145205473e-4),1.4846832179818732e-4),
    ((2.124700092423118e-4,2.2802990552445109e-4),
     (6.951608056828782e-5,3.67910797337589e-4),3.6753162834268576e-4),
])
def test_008_seek_final_uses_exported_rocking_interval(monkeypatch,old,updated,omega):
    """Real failed omega/bounds; other mechanical constraints remain active."""
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        command=sample(active,pose,clock)
        hard=active.pending_result.hard_constraints
        index=hard.labels.index('published_contact_rocking_speed_acc_jerk')
        low,high=hard.lower.copy(),hard.upper.copy();low[index],high[index]=old
        active.pending_result=replace(active.pending_result,
            hard_constraints=replace(hard,lower=low,upper=high))
        bounds=np.array([-.28,.28,-.01,.01,*updated])
        monkeypatch.setattr(active._rocking,'preview',lambda *a:(np.array([0.,1.,0.]),bounds.copy(),{}))
        active.rocking_constraints()
        final=command.copy();final[4]=omega
        facts=dict(rocking_policy_tier=1,rocking_lower_rad_s=updated[0],
                   rocking_upper_rad_s=updated[1],rocking_tolerance_rad_s=1e-5)
        assert hard.A[index] @ final>old[1]
        assert active.publication_review(active.pending_id,final,now_s=clock[0],facts=facts)
        event=[r for r in sink.records if r['event']=='publication_review'][-1]
        assert event['mechanical_review']['superseded_proposal_rows']==['published_contact_rocking_speed_acc_jerk']
        assert event['mechanical_review']['violated_rows']==[]
        active.publication_abort('offline unsent check',definitely_not_sent=True)
    finally:active.close()


@pytest.mark.parametrize('failure',['new_interval','wrong_certificate','stale_export','linear_speed','linear_slew','angle'])
def test_refreshed_rocking_preserves_final_guards(monkeypatch,failure):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        command=sample(active,pose,clock)
        hard=active.pending_result.hard_constraints
        index=hard.labels.index('published_contact_rocking_speed_acc_jerk')
        lo,hi=hard.lower.copy(),hard.upper.copy();lo[index],hi[index]=-1e-5,1e-5
        if failure=='angle':
            hi[hard.labels.index('angle_limit_contact_y')]=5e-5
        active.pending_result=replace(active.pending_result,hard_constraints=replace(hard,lower=lo,upper=hi))
        bounds=np.array([-.28,.28,-.01,.01,-.001,.001])
        monkeypatch.setattr(active._rocking,'preview',lambda *a:(np.array([0.,1.,0.]),bounds.copy(),{}))
        active.rocking_constraints()
        final=command.copy();final[4]=1e-4
        facts=dict(rocking_policy_tier=1,rocking_lower_rad_s=-.001,
                   rocking_upper_rad_s=.001,rocking_tolerance_rad_s=1e-5)
        reason='final_mechanical_interval_violation'
        if failure=='new_interval':final[4]=.002;reason='final_rocking_interval_violation'
        if failure=='wrong_certificate':facts['rocking_upper_rad_s']=.002;reason='final_rocking_certificate_mismatch'
        if failure=='stale_export':
            active._rocking_export_id=active.pending_id-1
            facts=dict(rocking_policy_tier=4,rocking_tolerance_rad_s=1e-5)
            final[4]=.05
        if failure=='linear_speed':final[0]=1.
        if failure=='linear_slew':final[0]=.02
        if failure=='angle':
            facts=dict(rocking_policy_tier=4,rocking_tolerance_rad_s=1e-5)
            final[4]=.05
        assert not active.publication_review(active.pending_id,final,now_s=clock[0],facts=facts)
        assert active._publication_rejection_reason==reason
        assert active._command_lease.pending is None
        assert active.reference_time_s==0.
        if failure in ('linear_speed','linear_slew','angle','stale_export'):
            event=[r for r in sink.records if r['event']=='publication_review_rejected'][-1]
            assert event['mechanical_review']['violated_rows']
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
    active.solver.config=replace(active.solver.config,certificate_horizon_s=.01)
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
    active.solver.config=replace(active.solver.config,certificate_horizon_s=.01)
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


def test_seek_command_slew_uses_control_step_not_rocking_publication_gap(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        leftover=np.array([0.,-0.001,0.0003,0.001,0.001,0.001])
        active.begin_hybrid_episode(leftover,pose)
        clock[0]+=0.00064
        active.features.observation=frame(active,clock[0],1,(.4,.4))
        command=sample(active,pose,clock,force=-0.2)
        assert active.pending_result.success, active.pending_result.diagnostics
        assert active.pending_result.diagnostics['command_slew_dt_s']==pytest.approx(.005)
        assert command is not None
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


def test_session009_11ms_first_command_admitted_with_fresh_force(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        command=sample(active,pose,clock);active.rocking_constraints()
        created=clock[0];clock[0]+=.011182748
        assert active.publication_review(active.pending_id,command,now_s=clock[0],facts=dict(rocking_policy_tier=4))
        assert active.publication_started(active.pending_id)
        active.publication_commit(active.pending_id,command,now_s=clock[0],facts={'arm':'sent','rail':'sent'})
        assert active._command_lease.active.expires_s==pytest.approx(created+.05)
    finally:active.close()


def test_missing_and_stale_ultrasound_do_not_stop_mechanical_control(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        active.features.observation=None
        for i in range(75):
            if i==2:active.features.observation=frame(active,clock[0],i,(.5,.9))
            command=sample(active,pose,clock)
            assert np.isfinite(command).all()
            commit(active,command,clock);clock[0]+=.005
        assert not active._pending_prediction.valid
        assert not active.pending_result.diagnostics['visual_rows_active']
        assert active._command_lease.latched_reason is None
    finally:active.close()


@pytest.mark.parametrize('compute_s',[.016,.020,.030])
def test_default_profile_accepts_isolated_scheduling_overrun_without_renewing_deadlines(monkeypatch,compute_s):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        first=sample(active,pose,clock);commit(active,first,clock)
        original_expiry=active._command_lease.active.expires_s
        clock[0]+=.005
        command=sample(active,pose,clock);created=clock[0]
        active.rocking_constraints();clock[0]+=compute_s
        assert active.publication_review(active.pending_id,command,now_s=clock[0],facts=dict(rocking_policy_tier=4))
        assert active._command_lease.active.expires_s==original_expiry
        assert active.publication_started(active.pending_id)
        active.publication_commit(active.pending_id,command,now_s=clock[0],facts={'arm':'sent','rail':'sent'})
        assert active._command_lease.active.expires_s==pytest.approx(created+.05)
        clock[0]+=.005
        fresh=sample(active,pose,clock);commit(active,fresh,clock)
        assert active._command_lease.latched_reason is None
    finally:active.close()


def test_default_profile_still_rejects_force_older_than_bounded_window(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        command=sample(active,pose,clock);active.rocking_constraints();clock[0]+=.051
        assert not active.publication_review(active.pending_id,command,now_s=clock[0],facts=dict(rocking_policy_tier=4))
        assert active._publication_rejection_reason=='wrench_source_expired'
        assert active._command_lease.pending is None
        assert active.reference_time_s==0.
    finally:active.close()


@pytest.mark.parametrize('damping',[40.,400.])
def test_air_seek_uses_selected_study_speed_independent_of_damping_and_black_image(monkeypatch,damping):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        active.controller.cfg.admittance_damping_z=damping
        active.controller.cfg.ke_schedule.d_min=active.controller.cfg.ke_schedule.d_max=damping
        assert active.controller.cfg.max_vz_tool_m_s==pytest.approx(.015)
        assert active.solver.config.max_velocity[2]==pytest.approx(.015)
        assert active.controller._approach_governor_m_s()==pytest.approx(.015)
        for i in range(50):
            active.features.observation=frame(active,clock[0],i,np.zeros(10))
            command=sample(active,pose,clock,force=0.)
            assert not active.controller.contact_present
            assert not active.pending_result.diagnostics['visual_rows_active']
            assert abs(command[2])<=.015+1e-8
            commit(active,command,clock);clock[0]+=.005
        assert command[2]>.014
    finally:active.close()


@pytest.mark.parametrize('motion',[{'normal_max_m_s':.015,'seek_m_s':.02},
    {'normal_max_m_s':.015,'seek_m_s':0.}, {'normal_max_m_s':True,'seek_m_s':.01}])
def test_invalid_study_motion_limits_rejected(motion):
    config=load_study_config(Path(__file__).parents[1]/'config/contact_qp/active_probe50_delay_kf_cop.yaml')
    config['motion']=motion
    with pytest.raises(ValueError):validate_study_config(config)


@pytest.mark.parametrize('field',['force_age','candidate_lifetime'])
def test_timing_cannot_exceed_existing_command_window(field):
    config=load_study_config(Path(__file__).parents[1]/'config/contact_qp/active_probe50_delay_kf_cop.yaml')
    if field=='force_age':config['source']['max_age_s']=.1
    else:config['qp']['certificate_horizon_s']=.1
    with pytest.raises(ValueError,match='command lease'):validate_study_config(config)


def test_inner_compensation_does_not_poison_fixed_outer_feedback_but_final_slew_stays_hard(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        # Real 010 lateral compensation grew away from the near-zero target.
        for lateral in [.004,.008]:
            target=sample(active,pose,clock,force=0.)
            final=target.copy();final[1]=lateral
            commit(active,final,clock)
            np.testing.assert_array_equal(active._previous,target)
            np.testing.assert_array_equal(active._previous_final,final)
            clock[0]+=.005
        target=sample(active,pose,clock,force=0.)
        assert active.pending_result.success
        assert abs(target[1])<1e-4
        sent_before=active._previous_final.copy()
        hard=active._final_command_constraints();i=hard.labels.index('acceleration_1')
        assert hard.lower[i]==pytest.approx(.003)
        assert hard.upper[i]==pytest.approx(.013)
        active.rocking_constraints()
        far=target.copy();far[1]=-.003
        assert not active.publication_review(active.pending_id,far,now_s=clock[0],facts=dict(rocking_policy_tier=4))
        assert active._publication_rejection_reason=='final_mechanical_interval_violation'
        assert active.publication_review(active.pending_id,target,now_s=clock[0],facts=dict(rocking_policy_tier=4))
        accepted=[r for r in sink.records if r['event']=='publication_review'][-1]
        assert accepted['mechanical_review']['max_violation']==pytest.approx(.003,abs=1e-5)
        assert accepted['mechanical_review']['violated_rows']
        reasons=set().union(*(item['reasons'] for item in active._quality_intervals.active.values()))
        assert 'final_outside_outer_envelope' in reasons
        active.publication_abort('unsent',definitely_not_sent=True)
        np.testing.assert_array_equal(active._previous_final,sent_before)
        clock[0]+=.005
        target=sample(active,pose,clock,force=0.)
        final=target.copy();final[1]=.006
        commit(active,final,clock)
        assert active._command_lease.latched_reason is None
    finally:active.close()


def test_seek_knee_within_one_tick_acceleration_is_audited_not_rejected(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        command=sample(active,pose,clock,dt=.007)
        previous=np.zeros(6);previous[2]=.0148
        active._pending_previous_final=previous
        active.rocking_constraints()
        facts=dict(rocking_policy_tier=4)
        reject=command.copy();reject[2]=.025
        assert not active.publication_review(active.pending_id,reject,now_s=clock[0],facts=facts)
        assert active._publication_rejection_reason=='final_mechanical_interval_violation'
        accept=command.copy();accept[2]=.017
        assert active.publication_review(active.pending_id,accept,now_s=clock[0],facts=facts)
        event=[r for r in sink.records if r['event']=='publication_review'][-1]
        assert event['mechanical_review']['max_violation']==pytest.approx(.002,abs=2e-4)
        assert event['mechanical_review']['violated_rows']
        reasons=set().union(*(item['reasons'] for item in active._quality_intervals.active.values()))
        assert 'final_outside_outer_envelope' in reasons
        active.publication_abort('unsent',definitely_not_sent=True)
    finally:
        active.close()


def test_solver_retry_lease_expiry_is_deferred_until_bound(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        commit(active,sample(active,pose,clock),clock)
        lease=active._command_lease.active
        assert lease is not None
        active._publication_rejection_reason='solver_attempts_exhausted'
        active._publication_retry_source_t_s=clock[0]
        for expected in (1,2,3):
            clock[0]=lease.expires_s
            active._publication_rejection_reason='solver_attempts_exhausted'
            active._publication_retry_source_t_s=lease.created_s
            active._command_lease.active=lease
            active._command_lease.latched_reason=None
            active._command_lease.last_time_s=lease.expires_s-1e-6
            with pytest.raises(ProposalDeferred,match='lease expired'):
                active.waiting_for_retry_source(lease.created_s,now_s=clock[0])
            assert active._lease_expiry_retries==expected
        active._publication_rejection_reason='solver_attempts_exhausted'
        active._publication_retry_source_t_s=lease.created_s
        active._command_lease.active=lease
        active._command_lease.latched_reason=None
        active._command_lease.last_time_s=lease.expires_s-1e-6
        clock[0]=lease.expires_s
        with pytest.raises(RuntimeError,match='lease expired'):
            active.waiting_for_retry_source(lease.created_s,now_s=clock[0])
    finally:
        active.close()


def test_empty_jerk_intersection_drops_outer_rocking_tier(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        commit(active,sample(active,pose,clock),clock)
        rotation=np.eye(3)
        active._rocking.seed(np.zeros(3),clock[0]-.01)
        active._rocking.commit([0.,0.,0.,0.,1.,0.],rotation,clock[0])
        clock[0]+=.005
        command=sample(active,pose,clock)
        assert command is not None
        assert active._outer_rocking_tier>=2
    finally:
        active.close()


def test_exhausted_outer_solve_retries_next_rocking_tier(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        commit(active,sample(active,pose,clock),clock)
        real=active.solver.solve
        calls=[]
        def fail_first_rocking(data,**kwargs):
            calls.append(len(data.mechanical.A))
            result=real(data,**kwargs)
            if len(calls)==1 and len(data.mechanical.A):
                return replace(result,status=result.status,qp_twist=None,
                               diagnostics=dict(result.diagnostics,reason='solver_attempts_exhausted'))
            return result
        monkeypatch.setattr(active.solver,'solve',fail_first_rocking)
        clock[0]+=.005
        command=sample(active,pose,clock)
        assert command is not None
        assert any(item['event']=='outer_rocking_tier_relax' for item in sink.records)
        assert active._outer_rocking_tier>=2
    finally:
        active.close()


def test_retry_unsent_allows_same_tick_after_solver_deferral(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        commit(active,sample(active,pose,clock),clock)
        lease=active._command_lease.active
        active._publication_rejection_reason='solver_attempts_exhausted'
        active._command_lease.last_time_s=clock[0]
        assert active.retry_unsent_publication()
        assert active._command_lease.active is lease
    finally:
        active.close()


def test_retry_unsent_forgives_expired_solver_deadline_lease(monkeypatch):
    active,pose,clock,sink=fixture(monkeypatch)
    try:
        commit(active,sample(active,pose,clock),clock)
        lease=active._command_lease.active
        active._publication_rejection_reason='solver_deadline_exceeded'
        clock[0]=lease.expires_s+1e-6
        assert active.retry_unsent_publication()
        assert active._lease_expiry_retries==1
        assert active._command_lease.active is None
        clock[0]+=.005
        command=sample(active,pose,clock)
        assert command is not None
        commit(active,command,clock)
        assert active._command_lease.active is not None
    finally:
        active.close()
