"""Detached adapter checks: no live receiver, SDK, SHM, or hardware process."""
from copy import deepcopy
import json
import inspect
from types import SimpleNamespace
import numpy as np
import pytest
from peirastic.realman8dof.modes.contact_recording import ContactRecordSink
from peirastic.realman8dof.modes.contact_qp import ContactStudyOuter,study_config
from peirastic.realman8dof.modes.track import HybridTffOuter
from peirastic.realman8dof.force.contact_nominal import build_contact_nominal,build_contact_position
from peirastic.scan_path import ForearmReference,make_spec,SCAN_FORCE_AXES
from peirastic.api.payloads import HfpcPayload


class MemorySink:
    def __init__(self):self.records=[];self.closed=False
    def emit(self,event,**fields):self.records.append(dict(event=event,**fields));return True
    def close(self,**kwargs):self.closed=True


def make_outer():
    start=np.array([.12,.02,.15,0.,0.,0.]);end=start.copy();end[1]+=.06
    ref=ForearmReference(make_spec(start,end,'L','DtP',0))
    law,_=build_contact_nominal(.005);position=build_contact_position(ref)
    outer=HybridTffOuter(position,law,desired_force=[0,0,4,0,0,0],selection=1-np.array(SCAN_FORCE_AXES))
    outer.set_origin(start,t_s=0.)
    return outer,start


@pytest.mark.parametrize('mode',['baseline','shadow'])
def test_recording_modes_return_identical_legacy_commands_and_preserve_states(mode):
    baseline,pose=make_outer();recorded=deepcopy(baseline)
    sink=MemorySink();wrapped=ContactStudyOuter(recorded,dict(mode=mode),sink=sink)
    try:
        for i in range(600):
            wrench=np.array([0,0,4+.3*np.sin(i*.03),0,.05*np.sin(i*.02),0])
            kwargs=dict(dt_actual=.005,f_ext_raw=wrench,sensor_age_s=0.,feedback_age_s=0.,
                        feedback_fresh_tick=False,feedback_velocity_valid=False,v_tcp_z_actual=0.,slack_norm=0.)
            expected=baseline.sample(i*.005,pose,wrench,**kwargs)
            actual=wrapped.sample(i*.005,pose,wrench,**kwargs,measured_twist_valid=False,
                                  measured_twist_fresh=False,measurement_id=0)
            np.testing.assert_array_equal(actual,expected)
        for name in ('last_v_cmd','v_force_z','_v_zoh_z','x_adm_z','x_d_z','_force_point_base'):
            np.testing.assert_array_equal(getattr(baseline.controller,name),getattr(recorded.controller,name))
        assert baseline.force_law.tilt.theta_tilt==recorded.force_law.tilt.theta_tilt
        samples=[r for r in sink.records if r['event']=='control_sample']
        assert len(samples)==600
        assert samples[0]['measured_twist_valid'] is False
        assert samples[0]['measurement_id']==0
    finally:wrapped.close()


def test_record_sink_snapshots_mutable_values_and_final_drop_count(tmp_path):
    sink=ContactRecordSink(tmp_path/'study.jsonl',capacity=1)
    values=np.array([1.,np.nan])
    sink.emit('snapshot',value=values);values[0]=99
    for i in range(1000):sink.emit('busy',i=i)
    sink.close()
    rows=[json.loads(line) for line in sink.path.read_text().splitlines()]
    assert rows[0]['value']==[1.,None]
    assert rows[-1]['event']=='recording_close'
    assert rows[-1]['dropped_records']==sink.dropped
    assert sink.dropped>0
    with pytest.raises(FileExistsError):ContactRecordSink(sink.path)


def test_baseline_recording_does_not_require_geometry_or_image_calibration():
    payload=HfpcPayload(reference='icra_path',contact_qp={'mode':'baseline'}).to_json()
    assert 'law' not in payload
    assert payload['contact_qp']=={'mode':'baseline'}
    assert study_config(payload)=={'mode':'baseline'}
    with pytest.raises(ValueError):HfpcPayload(reference='hold',contact_qp={'mode':'baseline'}).to_json()
    with pytest.raises(ValueError):HfpcPayload(reference='icra_path',law='contact_qp').to_json()


def test_wrapper_measurement_signature_preserves_zero_and_false():
    names=inspect.signature(ContactStudyOuter.sample).parameters
    for name in ('dt_actual','slack_norm','measurement_id','measured_twist_valid','measured_twist_fresh',
                 'measurement_time_s','wrench_source_time_s','wrench_source_wall_time_ns'):
        assert name in names


def test_real_daemon_install_closure_preserves_publication_owner_and_metadata():
    """Execute the real nested installer, with its hardware peers replaced."""
    import ast
    from pathlib import Path
    from peirastic.realman8dof.session import ProxyOuter
    from peirastic.realman8dof.modes.contact_qp import wrap_study_phase
    from rm75_control.control.joint_admittance_8dof.loop import Phase
    tree=ast.parse(Path('peirastic/realman8dof/daemon.py').read_text())
    installer=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_install_velocity')
    copy_assignment=next(n for n in tree.body if isinstance(n,ast.Assign)
                         and any(isinstance(t,ast.Name) and t.id=='_PHASE_COPY' for t in n.targets))
    copied=ast.literal_eval(copy_assignment.value)
    first,pose=make_outer();second,_=make_outer()
    sinks=[MemorySink(),MemorySink()]
    import peirastic.realman8dof.modes.contact_qp as adapter
    old_sink=adapter.ContactRecordSink
    adapter.ContactRecordSink=lambda path:sinks.pop(0)
    try:
        ctx=SimpleNamespace(inner=SimpleNamespace(kin=SimpleNamespace(jacobian=lambda q:np.eye(6,8))))
        original=wrap_study_phase(Phase(outer=first,label='first'),{'reference':'icra_path','contact_qp':{'mode':'baseline'}},ctx)
        replacement=wrap_study_phase(Phase(outer=second,label='second'),{'reference':'icra_path','contact_qp':{'mode':'baseline'}},ctx)
    finally:adapter.ContactRecordSink=old_sink
    owner1,owner2=original.outer,replacement.outer
    proxy=ProxyOuter(owner1);original.outer=proxy
    inner=SimpleNamespace(q_cmd=np.zeros(8),core=SimpleNamespace(qdot_prev=np.zeros(8)),
        kin=ctx.inner.kin,begin_hybrid_episode=lambda *a:None)
    env=dict(np=np,phase=original,proxy=proxy,_PHASE_COPY=copied,
        _arm_install_ack=lambda *a:None,Status=SimpleNamespace(RUNNING=1),
        self=SimpleNamespace(inner=inner,panel=SimpleNamespace(event=lambda *a:None),
                             hub=SimpleNamespace(publish=lambda **kw:None)),MODE_LABEL={0:'test'})
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),installer],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'<real daemon installer>','exec'),env)
    metadata=dict(dt_actual=.005,measurement_id=0,measurement_time_s=0.,measured_twist_valid=False,
                  measured_twist_fresh=False,measured_twist_metadata={'port_verified':False},
                  wrench_source_time_s=0.,wrench_source_wall_time_ns=0,wrench_source_id='packet-source')
    forwarded={k:v for k,v in metadata.items() if k in inspect.signature(proxy.sample).parameters}
    step=SimpleNamespace(qdot=np.zeros(8),valid=True)
    proxy.sample(0.,pose,np.array([0,0,4,0,0,0]),**forwarded)
    original.on_tick(.005,step,np.zeros(8))
    env['_install_velocity'](replacement,SimpleNamespace(mode=0),pose,.005,announce=False)
    proxy.sample(.005,pose,np.array([0,0,4,0,0,0]),**forwarded)
    original.on_tick(.01,step,np.zeros(8))
    for owner in (owner1,owner2):
        records=owner.sink.records
        sample=next(r for r in records if r['event']=='control_sample')
        sent=[r for r in records if r['event']=='publication']
        assert len(sent)==1 and sent[0]['control_id']==sample['control_id']==1
        assert sample['measurement_id']==sample['measurement_time_s']==0
        assert sample['measured_twist_valid'] is sample['measured_twist_fresh'] is False
        assert sample['wrench_source_time_s']==sample['wrench_source_wall_time_ns']==0
        assert sample['wrench_source_id']=='packet-source'
    assert owner1.sink.closed and not owner2.sink.closed
    original.on_exit();assert owner2.sink.closed
    runner=Path('rm75_control/rm75_control/control/joint_admittance_8dof/loop.py').read_text()
    assert runner.index('phase.on_tick(t_ref, step, q_meas)')<runner.index('on_step(phase.label, t_ref, step, pose_pin, f_ext, t_wall)')


def test_cli_default_validation_has_no_transport_and_template_needs_no_geometry(capsys,monkeypatch):
    from peirastic.apps.contact_qp_run import main
    from peirastic.api.arm import PeirasticArm
    monkeypatch.setattr(PeirasticArm,'__init__',lambda *a,**kw:pytest.fail('transport attempted during validation'))
    assert main(['--config','peirastic/config/contact_qp/real_study.yaml'])==0
    report=json.loads(capsys.readouterr().out)
    assert report['hardware_connected'] is False
    assert report['command_authority']=='unchanged_baseline'
    assert 'geometry.T_tcp_face' in report['missing_active_calibration']


def test_active_rejects_missing_calibration_and_force_sign_follows_real_installation():
    from peirastic.realman8dof.modes.contact_config import validate_study_config,compression_force
    with pytest.raises(ValueError,match='active calibration missing'):
        validate_study_config({'mode':'active'})
    config={'geometry':dict(half_length_m=.025,T_tcp_face=np.eye(4),image_x_sign=1,
              calibration_version='measured_installation_A',verified=True,face_normal_convention='into_contact')}
    assert compression_force(config,[0,0,-4,0,0,0])==4
    config['geometry']['T_tcp_face']=np.diag([1,-1,-1,1])
    assert compression_force(config,[0,0,4,0,0,0])==4
    config['geometry']['face_normal_convention']='outward'
    assert compression_force(config,[0,0,-4,0,0,0])==4


def test_execute_prepares_the_actual_icra_local_defaults_without_hardware(tmp_path,monkeypatch,capsys):
    from peirastic.apps.contact_qp_run import main
    import peirastic.api.arm as api
    setter=inspect.signature(api.PeirasticArm.set_force_control)
    calls={}
    class FakeArm:
        def __init__(self,**kwargs):calls['constructor']=kwargs
        def set_force_control(self,**kwargs):setter.bind(self,**kwargs);calls['force_control']=kwargs
        def set_force_raw_override(self,payload):calls['raw']=payload
        def hfpc(self,**kwargs):calls['hfpc']=kwargs;return 0
    monkeypatch.setattr(api,'PeirasticArm',FakeArm)
    pose=np.array([.12,.02,.15,0,0,0]);end=pose.copy();end[1]+=.06
    path=tmp_path/'path.json';path.write_text(json.dumps(make_spec(pose,end,'L','DtP',0)))
    assert main(['--config','peirastic/config/contact_qp/real_study.yaml','--execute','--path-spec',str(path)])==0
    assert calls['force_control']==dict(control_frame='tool',max_vz_tool_m_s=.010)
    assert calls['raw']['max_vz_tool_m_s']==calls['raw']['v_seek_free_m_s']==.010
    assert calls['raw']['hybrid_motion']['torque_tilt']['mass']==.051
    assert calls['raw']['hybrid_motion']['torque_tilt']['damping']==.22
    assert calls['hfpc']['law']=='tff'
    assert calls['hfpc']['scan_contact_n']==4.
    assert calls['hfpc']['scan_contact_s']==.1


def test_actual_runner_exit_records_partial_failure_before_closing(tmp_path):
    import ast
    from pathlib import Path
    from peirastic.realman8dof.modes.contact_qp import wrap_study_phase
    from rm75_control.control.joint_admittance_8dof.loop import Phase
    outer,pose=make_outer();path=tmp_path/'failure.jsonl'
    phase=wrap_study_phase(Phase(outer=outer),dict(reference='icra_path',contact_qp=dict(mode='baseline',log_path=str(path))),
                           SimpleNamespace(inner=None))
    phase.outer.sample(0.,pose,np.array([0,0,4,0,0,0]),dt_actual=.005)
    tree=ast.parse(Path('rm75_control/rm75_control/control/joint_admittance_8dof/loop.py').read_text())
    stop=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and any(
              isinstance(c,ast.Name) and c.id=='study_reason' for c in ast.walk(n))
              and "getattr(phase.outer" in ast.unparse(n.test))
    exit_call=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='phase.on_exit is not None')
    env=dict(phase=phase,stop_reason='PARTIAL_ARM:rail_commit_failed',study_termination_reason='',phase_arrived=False,phase_idx=1)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[stop,exit_call],type_ignores=[])),'<runner exit>','exec'),env)
    phase.outer.record_stop('late_finally')
    phase.outer.close()
    records=[json.loads(line) for line in path.read_text().splitlines()]
    stops=[row for row in records if row['event']=='stop']
    assert len(stops)==1 and stops[0]['reason']=='PARTIAL_ARM:rail_commit_failed'
    assert records[-1]['event']=='recording_close'


def test_shadow_uses_declared_feature_policy_and_relative_pose(monkeypatch):
    from dataclasses import asdict
    from threading import Event
    from peirastic.contact_qp.features import FeatureConfig
    from peirastic.contact_qp.types import ContactObservation
    from peirastic.contact_qp.qp import ContactQp
    fc=FeatureConfig(calibration_version='installation-A')
    now=10.
    obs=ContactObservation(1,'camera-A',now-.1,now,np.array([.9,.9,.9]),np.ones(3,dtype=bool),
                           'registration-A',fc.window_version,calibration_version=fc.calibration_version)
    config=dict(mode='shadow',geometry=dict(half_length_m=.025,T_tcp_face=np.eye(4),image_x_sign=1,
                calibration_version=fc.calibration_version),feature=dict(config=asdict(fc),window_version=fc.window_version,
                registration_version='registration-A',c_min=.8,quality_policy_version='declared-A'))
    called=[];event=Event()
    def solve(solver,data):called.append((solver.config,data));event.set();raise ValueError('test solve capture')
    monkeypatch.setattr(ContactQp,'solve',solve)
    outer,pose=make_outer();pose[3:]=[.4,.6,-.2]
    sink=MemorySink();receiver=SimpleNamespace(observation=obs,error=None,close=lambda **kw:None)
    wrapped=ContactStudyOuter(outer,config,sink=sink,feature_receiver=receiver)
    wrapped.sample(0.,pose,np.array([0,0,4,0,0,0]),dt_actual=.005,wrench_source_id='source',wrench_source_time_s=now)
    assert event.wait(1.)
    assert called[0][0].c_min==.8
    assert called[0][1].measured_angle==pytest.approx(0.,abs=1e-12)
    wrapped.close()


def test_control_thread_shutdown_never_joins_writer(tmp_path,monkeypatch):
    sink=ContactRecordSink(tmp_path/'nonblocking.jsonl')
    join=sink._thread.join
    monkeypatch.setattr(sink._thread,'join',lambda *a,**kw:pytest.fail('control exit waited for writer'))
    sink.emit('stop',reason='test')
    sink.close(wait=False)
    assert sink.closed
    monkeypatch.setattr(sink._thread,'join',join)
    sink.close()
    assert json.loads(sink.path.read_text().splitlines()[-1])['event']=='recording_close'


def test_process_exit_drains_requested_writer_close(tmp_path):
    import subprocess,sys
    target=tmp_path/'exit.jsonl'
    script='from peirastic.realman8dof.modes.contact_recording import ContactRecordSink; import sys; s=ContactRecordSink(sys.argv[1]); s.emit("stop",reason="process_exit"); s.close(wait=False)'
    subprocess.run([sys.executable,'-c',script,str(target)],check=True,timeout=10)
    rows=[json.loads(line) for line in target.read_text().splitlines()]
    assert rows[0]['reason']=='process_exit' and rows[-1]['event']=='recording_close'


def test_published_rail_record_distinguishes_actual_wall_clock_target_from_inner_proposal():
    outer,_=make_outer();sink=MemorySink();wrapped=ContactStudyOuter(outer,{'mode':'baseline'},sink=sink)
    step=SimpleNamespace(q_send=np.zeros(8),qdot=np.zeros(8),study_rail_target_m=.0003,
                         study_rail_velocity_m_s=.004,study_rail_coast=False,study_rail_target_modified=True)
    wrapped.record_publication(.1,step,np.zeros(8))
    record=sink.records[-1]
    assert record['rail_target_published_m']==.0003 and record['q_send'][0]==0
    assert record['rail_target_modified'] is True
    step.study_rail_coast=True
    wrapped.record_publication(.105,step,np.zeros(8))
    assert sink.records[-1]['rail_target_published_m'] is None
    assert sink.records[-1]['rail_action']=='hold_current_target_unavailable'
    wrapped.close()


@pytest.mark.parametrize('version,threshold,allowed',[
    ('randomwalk_welleweerd2020_v3',.5,False),
    ('randomwalk_welleweerd2020_v3',.8,True),
    ('randomwalk_camp_bmode_v2',.5,True),
])
def test_v3_preview_and_control_quality_threshold_must_match(version,threshold,allowed):
    from dataclasses import asdict
    from peirastic.contact_qp.features import FeatureConfig
    from peirastic.contact_qp.runtime_config import validate_study_config
    fc=FeatureConfig(algorithm_version=version,calibration_version='test-only')
    config=dict(mode='shadow',feature_endpoint='inproc://test-only',
        geometry=dict(half_length_m=.025,T_tcp_face=np.eye(4),image_x_sign=1,
            calibration_version='test-only',verified=True,face_normal_convention='into_contact'),
        feature=dict(config=asdict(fc),window_version=fc.window_version,
            registration_version='test',quality_policy_version='test',c_min=threshold,verified=True))
    if allowed:assert validate_study_config(config)['configuration_valid']
    else:
        with pytest.raises(ValueError,match='must equal'):
            validate_study_config(config)
