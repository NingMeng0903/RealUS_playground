"""Fresh recomputation and temporary removal of unavailable visual feedback."""
import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from peirastic.tests.test_contact_qp_active import make_active, propose
from peirastic.tests.test_contact_qp_command_budget_active import enabled
from peirastic.tests.test_contact_qp_solver import observation


def image(active,now):
    return replace(observation((.6,.8,.9),now=now),
        registration_version=active.registration_version,
        window_version=active.feature_config.window_version,
        calibration_version=active.feature_config.calibration_version)


def commit(active,command,clock):
    assert active.publication_review(active.pending_id,command,now_s=clock[0])
    assert active.publication_started(active.pending_id)
    active.publication_commit(active.pending_id,command,now_s=clock[0],facts={'arm':'sent','rail':'sent'})


def retry_fixture(monkeypatch):
    active,pose,clock,sink=enabled(monkeypatch)
    commit(active,propose(active,pose,clock),clock)
    clock[0]+=.005
    command=propose(active,pose,clock)
    # Simulate a narrow source deadline without inventing a new source sample.
    active.source_clock.max_age_s=.006
    clock[0]+=.0061
    assert not active.publication_review(active.pending_id,command,now_s=clock[0])
    active.publication_abort('expired',definitely_not_sent=True)
    clock[0]+=.0001
    return active,pose,clock,sink


def test_retry_preserves_lease_and_command_then_requires_new_force_source(monkeypatch):
    active,pose,clock,sink=retry_fixture(monkeypatch)
    try:
        old=active.command_budget.active
        reference=active.reference_time_s
        previous=active._previous.copy()
        assert active.retry_unsent_publication()
        assert active.command_budget.active is old
        assert active.command_budget.active.expires_s==pytest.approx(10.05)
        assert active.command_budget.pending is None
        assert active.reference_time_s==reference
        np.testing.assert_array_equal(active._previous,previous)
        # A fresh accepted source is required independently of the old lease.
        active.source_clock.max_age_s=.02
        with pytest.raises(RuntimeError,match='newer force source'):
            active.sample(0.,pose,np.array([0,0,4.,0,0,0]),dt_actual=.005,
                f_ext_raw=np.array([0,0,4.,0,0,0]),wrench_source_id='test-source',
                wrench_source_time_s=10.005,wrench_source_wall_time_ns=1)
        # This fixture uses a fixed ingress clock; a newer 5 ms source can be
        # received late without relabeling its acquisition timestamp.
        command=active.sample(0.,pose,np.array([0,0,4.,0,0,0]),dt_actual=.005,
            f_ext_raw=np.array([0,0,4.,0,0,0]),wrench_source_id='test-source',
            wrench_source_time_s=10.01,wrench_source_wall_time_ns=2)
        assert active.reference_time_s==reference
        commit(active,command,clock)
        assert active.reference_time_s>reference
        assert active._publication_retry_count==0
        assert any(r['event']=='publication_fresh_retry' for r in sink.records)
    finally:active.close()


@pytest.mark.parametrize('invalid',['no_old','expired','latched','wrong_reason','started','pending'])
def test_retry_never_recovers_other_faults_or_renews_old_deadline(monkeypatch,invalid):
    active,_,clock,_=retry_fixture(monkeypatch)
    try:
        if invalid=='no_old':active.command_budget.active=None
        elif invalid=='expired':clock[0]=10.051
        elif invalid=='latched':active.command_budget.latched_reason='existing_fault'
        elif invalid=='wrong_reason':active._publication_rejection_reason='energy_reservation_rejected'
        elif invalid=='started':active._dispatch_time_s=clock[0]
        else:active._nominal_pending=True
        assert not active.retry_unsent_publication()
        if invalid=='expired':assert active.command_budget.latched_reason
    finally:active.close()


def test_retry_count_is_diagnostic_and_waiting_preserves_original_lease(monkeypatch):
    active,_,clock,_=retry_fixture(monkeypatch)
    try:
        active._publication_retry_count=3
        assert active.retry_unsent_publication()
        assert active._publication_retry_count==4
        expiry=active.command_budget.active.expires_s
        clock[0]+=.005
        assert active.waiting_for_retry_source(10.005,now_s=clock[0])
        assert active.command_budget.active.expires_s==expiry
        assert not active.waiting_for_retry_source(10.01,now_s=clock[0])
        clock[0]=expiry
        with pytest.raises(RuntimeError,match='lease expired'):
            active.waiting_for_retry_source(10.005,now_s=clock[0])
        assert active.command_budget.latched_reason
    finally:active.close()


def test_dispatch_retry_classifies_age_separately_from_missing_review(monkeypatch):
    active,pose,clock,_=enabled(monkeypatch)
    try:
        commit(active,propose(active,pose,clock),clock)
        clock[0]+=.005
        command=propose(active,pose,clock)
        assert not active.publication_started(active.pending_id)
        assert active._publication_rejection_reason=='missing_review_or_invalid_dispatch'
        active.publication_abort('no review',definitely_not_sent=True)
        clock[0]+=.0001
        assert not active.retry_unsent_publication()
    finally:active.close()


@pytest.mark.parametrize('gap',['stale','missing'])
def test_image_grace_uses_no_old_observation_and_recovers_once(monkeypatch,gap):
    active,pose,clock,sink=make_active(monkeypatch)
    try:
        active.config['feature']['required']=True
        active._image_dropout_grace_s=.1
        active.features.observation=image(active,clock[0])
        assert active._image_observation(clock[0],contact_enabled=True)[1]=='ok'
        clock[0]+=.301
        if gap=='missing':active.features.observation=None
        observed,reason=active._image_observation(clock[0],contact_enabled=True)
        assert observed is None and reason=='transient_'+gap
        assert active._image_observation(clock[0]+.099,contact_enabled=True)[0] is None
        active.features.observation=image(active,clock[0]+.099)
        assert active._image_observation(clock[0]+.099,contact_enabled=True)[1]=='ok'
        assert [r['event'] for r in sink.records].count('image_feedback_unavailable')==1
        assert [r['event'] for r in sink.records].count('image_feedback_recovered')==1
    finally:active.close()


@pytest.mark.parametrize('fault',['startup','future','version','expired','disabled'])
def test_image_grace_cannot_hide_startup_invalid_feedback_or_unbounded_loss(monkeypatch,fault):
    active,_,clock,_=make_active(monkeypatch)
    try:
        active.config['feature']['required']=True;active._image_dropout_grace_s=.1
        if fault!='startup':
            active.features.observation=image(active,clock[0])
            active._image_observation(clock[0],contact_enabled=True)
        if fault=='future':active.features.observation=image(active,clock[0]+1)
        elif fault=='version':active.features.observation=replace(image(active,clock[0]),registration_version='wrong')
        else:
            active.features.observation=None
            if fault=='expired':
                active._image_observation(clock[0],contact_enabled=True)
                clock[0]+=.101
            if fault=='disabled':active._image_dropout_grace_s=0
        with pytest.raises(RuntimeError,match='confidence feedback'):
            active._image_observation(clock[0],contact_enabled=True)
    finally:active.close()


def test_transient_image_is_removed_from_actual_qp_input(monkeypatch):
    active,pose,clock,sink=make_active(monkeypatch)
    try:
        active.config['feature']['required']=True;active._image_dropout_grace_s=.1
        active.features.observation=image(active,clock[0])
        active._image_observation(clock[0],contact_enabled=True)
        active.features.observation=None
        captured=[];solve=active.solver.solve
        def capture(data):captured.append(data);return solve(data)
        monkeypatch.setattr(active.solver,'solve',capture)
        propose(active,pose,clock)
        assert captured[-1].observation is None
        sample=next(r for r in sink.records if r['event']=='control_sample')
        assert sample['feature'] is None and not sample['image_compatible']
        assert sample['image_feedback_status']=='transient_missing'
        active.publication_abort('test',definitely_not_sent=True)
    finally:active.close()


@pytest.mark.parametrize('failure',['review','dispatch'])
def test_production_retry_branch_aborts_every_proposal_and_paces_without_send(failure):
    """Execute real runner send block; neither device nor commit/heartbeat runs."""
    from rm75_control.control.joint_admittance_8dof import loop
    tree=ast.parse(Path(loop.__file__).read_text());target=None
    for node in ast.walk(tree):
        body=getattr(node,'body',None)
        if not isinstance(body,list):continue
        starts=[i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_t_send0' for t in n.targets)]
        if starts:
            start=starts[0]
            end=next(i for i in range(start,len(body)) if isinstance(body[i],ast.If) and ast.unparse(body[i].test)=='not wd.fired')
            target=body[start:end+1];break
    assert target
    events=[]
    class Owner:
        pending_id=7;pending_rotation_base_tcp=np.eye(3)
        def publication_review(self,*a,**k):return failure!='review'
        def publication_started(self,*a):return failure!='dispatch'
        def publication_abort(self,reason,**kw):events.append(('outer_abort',kw))
        def publication_commit(self,*a,**kw):events.append('outer_commit')
        def retry_unsent_publication(self):
            assert events[-3:]==['rail_abort',('outer_abort',{'definitely_not_sent':True}),'inner_abort']
            events.append('retry');return True
    rail=NS(enabled=True,measured_m=0.,reserve_target_m=lambda *a,**k:True,
            commit_reservation=lambda:events.append('rail_commit'),abort_reservation=lambda:events.append('rail_abort'))
    env=dict(time=__import__('time'),np=np,publication_owner=Owner(),rail_bridge=rail,rail_coast_active=False,
        step=NS(qdot=np.zeros(8),q_send=np.zeros(8),rocking_policy_tier=0,
            rocking_limited=False,rocking_lower_rad_s=-np.inf,rocking_upper_rad_s=np.inf),q_prev=np.zeros(8),q_meas=np.zeros(8),
        inner=NS(cfg=NS(dt=.005,resync_err_rail_m=.01),limits=NS(q_lower=np.full(8,-1),q_upper=np.ones(8)),
            kin=NS(jacobian=lambda q:np.eye(6,8)),_direct_joint_ptp=False,_plan_drives_rail=False,
            abort_publication=lambda:events.append('inner_abort'),commit_publication=lambda q:events.append('inner_commit')),
        dt_wall_actual=.005,_qpik_rail_v_ff_m_s=lambda x:x,_wall_clock_rail_target=lambda x,*a,**kw:x,
        _reserve_rail_target=lambda *a,**k:(True,''),_fault_stop=lambda reason:events.append('stop'),
        _send_joint_canfd_cmd=lambda *a:events.append('arm_send'),robot=None,rad2deg=lambda x:x,
        arm_q_from_full=lambda x:x[1:],follow=True,canfd_proxy=None,ticks=9,next_tick=10.,dt=.005,
        wd=NS(fired=False,beat=lambda:events.append('heartbeat')),_wait_until=lambda t:events.append(('wait',t)))
    env.setdefault('wd',NS(fired=False));env.update(fault_epoch=[0],stop_check=None,on_control_state=None)
    module=ast.Module(body=[ast.For(target=ast.Name(id='_once',ctx=ast.Store()),iter=ast.Tuple(elts=[ast.Constant(1)],ctx=ast.Load()),body=target,orelse=[])],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'<production no-send retry>','exec'),env)
    assert events==['rail_abort',('outer_abort',{'definitely_not_sent':True}),'inner_abort','retry',('wait',10.005)]
    assert env['ticks']==10 and env['next_tick']==10.005


def test_native_abort_schedules_abort_instead_of_commit():
    from rm75_control.control.joint_admittance_8dof.wbc_rt.client import NativeWbcClient
    # No worker/process is started: exercise the exact pending transaction API.
    native=object.__new__(NativeWbcClient)
    native._pending_commit_seq=14;native._abort_next=False
    native.abort_pending()
    assert native._pending_commit_seq==0 and native._abort_next
    tree=ast.parse(Path(__import__(NativeWbcClient.__module__,fromlist=['']).__file__).read_text())
    target=None
    for node in ast.walk(tree):
        body=getattr(node,'body',None)
        if not isinstance(body,list):continue
        for i,n in enumerate(body):
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='auto_commit' for t in n.targets):
                end=next(j for j in range(i,len(body)) if isinstance(body[j],ast.Assign) and any(isinstance(t,ast.Name) and t.id=='seq' for t in body[j].targets))
                target=body[i:end];break
        if target is not None:break
    from rm75_control.control.joint_admittance_8dof.wbc_rt import protocol as P
    record={'cmd_u':np.zeros(1,dtype=np.uint32),'flags':np.zeros(1,dtype=np.uint32)}
    env=dict(self=native,kwargs={'auto_commit':False},flags=0,rec=record,P=P,np=np)
    exec(compile(ast.fix_missing_locations(ast.Module(body=target,type_ignores=[])),'<native next-request flags>','exec'),env)
    assert record['flags']&P.IN_ABORT_PREV
    assert not record['flags']&P.IN_COMMIT_PREV
    assert record['cmd_u'][0]==0


def test_frame_arriving_during_nominal_compute_does_not_look_future(monkeypatch):
    active,pose,clock,sink=make_active(monkeypatch)
    try:
        active.config['feature']['required']=True
        first=image(active,clock[0]);active.features.observation=first
        baseline_sample=active.baseline.sample
        def arrival(*args,**kwargs):
            result=baseline_sample(*args,**kwargs)
            active.features.observation=image(active,clock[0]+.001)
            return result
        monkeypatch.setattr(active.baseline,'sample',arrival)
        propose(active,pose,clock)
        sample=next(r for r in sink.records if r['event']=='control_sample')
        assert sample['feature']['received_time_s']==first.received_time_s
        assert sample['image_feedback_status']=='ok'
        active.publication_abort('test',definitely_not_sent=True)
    finally:active.close()


def test_pause_visual_keeps_old_images_disabled_for_long_transport_loss(monkeypatch):
    active,_,clock,sink=make_active(monkeypatch)
    try:
        active.config['feature']['required']=True
        active._image_dropout_policy='pause_visual'
        active.features.observation=image(active,clock[0])
        active._image_observation(clock[0],contact_enabled=True)
        for elapsed in (.301,.4,1.,60.):
            observed,reason=active._image_observation(clock[0]+elapsed,contact_enabled=True)
            assert observed is None and reason=='transient_stale'
        active.features.observation=image(active,clock[0]+60.)
        assert active._image_observation(clock[0]+60.,contact_enabled=True)[1]=='ok'
        assert not any(r['event']=='required_image_feedback_rejected' for r in sink.records)
        assert any(r.get('scan_quality_degraded') for r in sink.records)
    finally:active.close()


def test_runner_waits_for_new_retry_source_before_preparing_any_proposal():
    from rm75_control.control.joint_admittance_8dof import loop
    tree=ast.parse(Path(loop.__file__).read_text())
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
        and any(isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='wait_for_source' for t in x.targets) for x in n.body))
    events=[]
    owner=NS(waiting_for_retry_source=lambda *a,**k:True,
             prepare_source=lambda *a,**k:events.append('prepare'))
    env=dict(publication_owner=owner,snap=NS(t_s=10.),time=__import__('time'),ticks=1,next_tick=10.,dt=.005,
        _wait_until=lambda t:events.append(('wait',t)))
    module=ast.Module(body=[ast.For(target=ast.Name(id='_once',ctx=ast.Store()),iter=ast.Tuple(elts=[ast.Constant(1)],ctx=ast.Load()),body=[branch],orelse=[])],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'<production fresh-source wait>','exec'),env)
    assert events==[('wait',10.005)] and env['ticks']==2


@pytest.mark.parametrize('valid',[(False,True,True),(True,True,False),(False,True,False)])
def test_invalid_required_windows_cannot_establish_initial_contact_image(monkeypatch,valid):
    active,_,clock,sink=make_active(monkeypatch)
    try:
        active.config['feature']['required']=True;active._image_dropout_policy='pause_visual'
        active.features.observation=replace(image(active,clock[0]),valid=valid)
        assert not active.features.observation.fresh(clock[0],active.solver.config.max_image_age_s)
        with pytest.raises(RuntimeError,match='invalid_required_windows'):
            active._image_observation(clock[0],contact_enabled=True)
        assert not active._image_seen_in_contact
        assert not any(r['event']=='image_feedback_recovered' for r in sink.records)
    finally:active.close()


def test_invalid_required_windows_keep_visual_paused_without_false_recovery(monkeypatch):
    active,_,clock,sink=make_active(monkeypatch)
    try:
        active.config['feature']['required']=True;active._image_dropout_policy='pause_visual'
        active.features.observation=image(active,clock[0])
        active._image_observation(clock[0],contact_enabled=True)
        active.features.observation=None
        clock[0]+=.01
        active._image_observation(clock[0],contact_enabled=True)
        unavailable_since=active._image_unavailable_since_s
        for elapsed in (.02,.5,60.):
            active.features.observation=replace(image(active,clock[0]+elapsed),valid=[False,True,False])
            received,reason=active._image_observation(clock[0]+elapsed,contact_enabled=True)
            assert received is None and reason=='transient_invalid_required_windows'
            assert active._image_unavailable_since_s==unavailable_since
        assert not any(r['event']=='image_feedback_recovered' for r in sink.records)
        # Center is diagnostic, not a required boundary window.
        active.features.observation=replace(image(active,clock[0]+60.),valid=[True,False,True])
        received,reason=active._image_observation(clock[0]+60.,contact_enabled=True)
        assert received is not None and reason=='ok'
        assert active._image_unavailable_since_s is None
        assert sum(r['event']=='image_feedback_recovered' for r in sink.records)==1
    finally:active.close()
