"""Software-only active publication checks; geometry is explicitly a test fixture."""
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import pytest
from peirastic.tests.test_contact_qp_runtime import make_outer, MemorySink
from peirastic.contact_qp.features import FeatureConfig
from peirastic.realman8dof.modes.contact_active import ContactQpOuter


def make_active(monkeypatch):
    fc=FeatureConfig(calibration_version='test_fixture_only')
    config=dict(mode='active', source=dict(period_s=.005,jitter_fraction=.2,max_age_s=.02),
        force_axis_monotonicity_confirmed=True,
        geometry=dict(half_length_m=.025,T_tcp_face=np.eye(4),image_x_sign=1,
                      calibration_version=fc.calibration_version,verified=True,face_normal_convention='into_contact'),
        feature=dict(config=asdict(fc),window_version=fc.window_version,
                     registration_version='test_registration',quality_policy_version='test_policy',c_min=.5,verified=True),
        feature_endpoint='inproc://unused')
    outer,pose=make_outer();sink=MemorySink()
    active=ContactQpOuter(outer,config,sink=sink,feature_receiver=SimpleNamespace(observation=None,close=lambda **kw:None))
    active.set_origin(pose,t_s=0.)
    clock=[10.]
    monkeypatch.setattr('peirastic.realman8dof.modes.contact_active.time.monotonic',lambda:clock[0])
    return active,pose,clock,sink


def propose(active,pose,clock):
    return active.sample(0.,pose,np.array([0.,0.,4.,0.,0.,0.]),dt_actual=.005,
        f_ext_raw=np.array([0.,0.,4.,0.,0.,0.]),wrench_source_id='test-source',
        wrench_source_time_s=clock[0],wrench_source_wall_time_ns=1)


def test_active_abort_and_commit_own_nominal_and_reference(monkeypatch):
    active,pose,clock,sink=make_active(monkeypatch)
    try:
        initial=active.controller.last_v_cmd.copy()
        command=propose(active,pose,clock)
        assert active.reference_time_s==0.
        np.testing.assert_array_equal(active.controller.last_v_cmd,initial)
        active.publication_abort('test rejected',definitely_not_sent=True)
        assert active.reference_time_s==0.
        np.testing.assert_array_equal(active.controller.last_v_cmd,initial)
        clock[0]+=.005
        command=propose(active,pose,clock)
        candidate=active.pending_id
        assert active.publication_review(candidate,command,now_s=clock[0]+.001)
        active.publication_started(candidate)
        active.publication_commit(candidate,command,now_s=clock[0]+.001)
        assert 0 < active.reference_time_s <= .005
        assert not active._nominal_pending
        with pytest.raises(RuntimeError):active.publication_commit(candidate,command,now_s=clock[0]+.001)
    finally:active.close()


def test_active_expired_review_and_partial_publish_freeze(monkeypatch):
    active,pose,clock,sink=make_active(monkeypatch)
    try:
        command=propose(active,pose,clock)
        assert not active.publication_review(active.pending_id,command,now_s=clock[0]+1.)
        active.publication_abort('partial',facts={'arm':'sent','rail':'failed'})
        assert active.reference_time_s==0.
        assert sink.records[-1]['facts']['arm']=='sent'
    finally:active.close()


@pytest.mark.parametrize('failure',['none','review','dispatch','arm','rail'])
def test_runner_actual_publication_block_preserves_device_facts(failure):
    """Execute the production runner's final preparation/send/commit block."""
    import ast
    from pathlib import Path
    from types import SimpleNamespace as NS
    from rm75_control.control.joint_admittance_8dof import loop
    tree=ast.parse(Path(loop.__file__).read_text())
    target=None
    for node in ast.walk(tree):
        body=getattr(node,'body',None)
        if not isinstance(body,list):continue
        starts=[i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_t_send0' for t in n.targets)]
        if starts:
            start=starts[0]
            end=next(i for i in range(start,len(body)) if isinstance(body[i],ast.If) and ast.unparse(body[i].test)=='not wd.fired')
            target=body[start:end];break
    assert target
    events=[]
    class Owner:
        pending_id=7;pending_rotation_base_tcp=np.eye(3)
        def publication_review(self,*a,**kw):events.append('review');return failure!='review'
        def publication_started(self,*a):events.append('started');return failure!='dispatch'
        def publication_abort(self,reason,**kw):events.append(('abort',reason,kw))
        def publication_commit(self,*a,**kw):events.append(('commit',kw['facts']))
    rail=NS(enabled=True,measured_m=0.,reserve_target_m=lambda *a,**k:True,
            commit_reservation=lambda:failure!='rail',abort_reservation=lambda:events.append('rail_abort'))
    def send(*a):
        events.append('arm_send')
        if failure=='arm':raise TimeoutError('test unknown')
    env=dict(time=__import__('time'),np=np,publication_owner=Owner(),rail_bridge=rail,rail_coast_active=False,
        step=NS(qdot=np.zeros(8),q_send=np.zeros(8)),q_prev=np.zeros(8),q_meas=np.zeros(8),
        inner=NS(cfg=NS(dt=.005,resync_err_rail_m=.01),limits=NS(q_lower=np.full(8,-1),q_upper=np.ones(8)),
                 kin=NS(jacobian=lambda q:np.eye(6,8)),_direct_joint_ptp=False,_plan_drives_rail=False,
                 abort_publication=lambda:events.append('inner_abort'),commit_publication=lambda q:events.append('inner_commit')),
        dt_wall_actual=.005,_qpik_rail_v_ff_m_s=lambda x:x,_wall_clock_rail_target=lambda x,*a,**kw:x,
        _reserve_rail_target=lambda *a,**k:(True,''),_fault_stop=lambda reason:events.append(('stop',reason)),
        _send_joint_canfd_cmd=send,robot=None,rad2deg=lambda x:x,arm_q_from_full=lambda x:x[1:],follow=True,canfd_proxy=None)
    module=ast.Module(body=[ast.For(target=ast.Name(id='_once',ctx=ast.Store()),iter=ast.Tuple(elts=[ast.Constant(1)],ctx=ast.Load()),body=target,orelse=[])],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'<production publication>','exec'),env)
    if failure=='none':
        assert events.index('review')<events.index('started')<events.index('arm_send')<events.index('inner_commit')
        assert events[-1]==('commit',{'arm':'sent','rail':'sent'})
    else:
        assert not any(isinstance(e,tuple) and e[0]=='commit' for e in events)
        if failure=='review':assert 'arm_send' not in events
        if failure=='dispatch':
            assert 'arm_send' not in events
            assert 'rail_abort' in events
            assert any(isinstance(e,tuple) and e[:2]==('abort','dispatch_rejected') for e in events)
        if failure=='arm':assert any(isinstance(e,tuple) and e[:2]==('abort','arm_send_unknown') for e in events)
        if failure=='rail':assert any(isinstance(e,tuple) and e[:2]==('abort','partial_publish') for e in events)


def test_prepare_failure_after_nominal_candidate_retires_candidate(monkeypatch):
    active,pose,clock,sink=make_active(monkeypatch)
    original=active.baseline.sample
    def fail(*args,**kwargs):
        original(*args,**kwargs)
        raise RuntimeError('failure after delegated candidate')
    monkeypatch.setattr(active.baseline,'sample',fail)
    with pytest.raises(RuntimeError,match='delegated'):
        propose(active,pose,clock)
    assert active.nominal.z_law._transaction.pending is None
    assert active.nominal.tilt._pending_command is None
    assert active.reference_time_s==0.
    active.close()


def test_production_runner_exception_stops_before_cleanup_even_when_cleanup_fails():
    import ast
    from pathlib import Path
    from rm75_control.control.joint_admittance_8dof import loop
    tree=ast.parse(Path(loop.__file__).read_text())
    handler=next(n for n in ast.walk(tree) if isinstance(n,ast.ExceptHandler)
                 and 'contact_qp_exception:' in ast.unparse(n))
    events=[]
    class Owner:
        def publication_abort(self,reason):events.append('outer_abort');raise RuntimeError('log unavailable')
    env=dict(phase=SimpleNamespace(outer=SimpleNamespace(publication_owner=Owner())),
             inner=SimpleNamespace(abort_publication=lambda:events.append('inner_abort')),
             _fault_stop=lambda reason:events.append('stop'))
    trigger=ast.Raise(exc=ast.Call(func=ast.Name(id='ValueError',ctx=ast.Load()),args=[ast.Constant('source invalid')],keywords=[]),cause=None)
    module=ast.Module(body=[ast.Try(body=[trigger],handlers=[handler],orelse=[],finalbody=[])],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'<production exception handler>','exec'),env)
    assert events==['stop','inner_abort','outer_abort']
    assert env['phase_stopped'] and 'source invalid' in env['stop_reason']


@pytest.mark.parametrize('arrived',[False,True])
def test_daemon_done_requires_confirmed_active_arrival(arrived):
    import ast
    from pathlib import Path
    from peirastic.realman8dof import daemon
    from peirastic.core.ipc import Status
    tree=ast.parse(Path(daemon.__file__).read_text())
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and 'arrival_confirmed' in ast.unparse(n.test))
    events=[]
    svc=SimpleNamespace(hub=SimpleNamespace(publish=lambda **kw:events.append(kw)),mode=4,ticks=9,_cmd_seq=12,
                        _pending=None,_idle_request=lambda **kw:'idle')
    env=dict(self=svc,velocity_loop=True,active_contact=True,compiled=SimpleNamespace(outer=SimpleNamespace(arrival_confirmed=arrived)),
             phase=SimpleNamespace(label='active'),Status=Status)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[branch],type_ignores=[])),'<production done>','exec'),env)
    assert bool(events)==arrived
    if arrived:
        assert events[0]['status']==Status.DONE and events[0]['done_seq']==12
        assert svc._pending=='idle'


def test_geometric_endpoint_alone_does_not_confirm_settled_arrival(monkeypatch):
    active,pose,clock,sink=make_active(monkeypatch)
    active.reference_time_s=active.reference.duration_s
    target=active.reference.reference.sample(active.reference.duration_s).pose_d
    assert active.arrived(target)
    assert not active.arrival_confirmed
    active.close()


@pytest.mark.parametrize('settled',[False,True])
def test_runner_settling_gate_owns_arrival_confirmation(settled):
    import ast
    from pathlib import Path
    from rm75_control.control.joint_admittance_8dof import loop
    tree=ast.parse(Path(loop.__file__).read_text())
    gate=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test).startswith('arrival_gate.update('))
    owner=SimpleNamespace(arrival_confirmed=False)
    env=dict(arrival_gate=SimpleNamespace(update=lambda **kw:settled),phase_arrived=True,t_ref=1.,
        step=SimpleNamespace(qdot=np.zeros(8)),control_dt=.005,rail_bridge=None,
        phase=SimpleNamespace(arrival_rail_speed_m_s=.001),inner=SimpleNamespace(cfg=SimpleNamespace(feedback_timeout_s=.1),q_cmd=np.zeros(8)),
        q_meas=np.zeros(8),time=__import__('time'),_rail_settled_for_arrival=lambda *a,**kw:settled,publication_owner=owner)
    module=ast.Module(body=[ast.For(target=ast.Name(id='_once',ctx=ast.Store()),iter=ast.Tuple(elts=[ast.Constant(1)],ctx=ast.Load()),body=[gate],orelse=[])],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'<production settling gate>','exec'),env)
    assert owner.arrival_confirmed==settled
