"""Continuous visual task regression; all execution is software-only."""
from dataclasses import replace
from pathlib import Path
import uuid

import numpy as np
import pytest

from peirastic.contact_qp.qp import ContactQp, QpConfig
from peirastic.contact_qp.repair_policy import DifferentialRepairConfig
from peirastic.contact_qp.port_constraint import PortEnergyConstraint
from peirastic.contact_qp.runtime_config import load_study_config, validate_study_config
from peirastic.tests.test_contact_qp_solver import datum, observation


PROFILE = Path(__file__).parents[1]/'config/contact_qp/active_probe50_v8r3_tank.yaml'


def solver(**kwargs):
    return ContactQp(QpConfig(allocation_policy='differential_repair_v8',
        differential_repair=DifferentialRepairConfig(revision='v8r3_confidence_balance',
            permission_mode='continuous'), max_acceleration=np.full(6,100.), **kwargs))


def test_persistent_request_survives_time_and_travel_without_healthy_frames():
    qp=solver()
    for seq,t in enumerate((0.,1.,2.1,10.,60.),1):
        data=datum((.6,.8,.9), now_s=t, measured_angle=.1*(seq%2),
                   observation=observation((.6,.8,.9),now=t,seq=seq))
        result=qp.solve(data)
        assert result.success, result.diagnostics
        assert result.diagnostics['repair_episode']['repair_allowed']
        assert not result.diagnostics['repair_episode']['exhausted']
        assert result.diagnostics['differential_request_m_s']>0
        assert result.qp_twist[4]>1e-3
        assert result.hard_constraints.violation(result.qp_twist)<=qp.config.feasibility_tolerance


def test_equal_bad_images_cannot_spend_future_angular_repair():
    qp=solver(c_min=.8)
    for seq,t in enumerate((0.,2.1,15.),1):
        result=qp.solve(datum((.3,.3,.3),now_s=t,
            observation=observation((.3,.3,.3),now=t,seq=seq)))
        assert result.success
        assert result.diagnostics['differential_request_m_s']==0
        assert abs(result.qp_twist[4])<1e-7
    result=qp.solve(datum((.3,.7,.9),now_s=15.005,
        observation=observation((.3,.7,.9),now=15.005,seq=4)))
    assert result.success and result.qp_twist[4]>1e-3


@pytest.mark.parametrize('pause',['image_missing','image_stale','contact','force'])
def test_current_gates_pause_and_one_new_frame_resumes(pause):
    qp=solver()
    data=datum((.6,.8,.9))
    assert qp.solve(data).qp_twist[4]>1e-3
    changes=dict(now_s=20., observation=observation((.6,.8,.9),now=20.,seq=2))
    if pause=='image_missing':changes['observation']=None
    elif pause=='image_stale':changes['observation']=observation((.6,.8,.9),now=19.,seq=2)
    elif pause=='contact':changes['repair_execution_enabled']=False
    else:changes['force_n']=4.6
    stopped=qp.solve(replace(data,**changes))
    assert stopped.success
    assert not stopped.diagnostics['repair_episode']['repair_allowed']
    assert stopped.diagnostics['differential_request_m_s']==0
    resumed=qp.solve(replace(data,now_s=20.005,
        observation=observation((.6,.8,.9),now=20.005,seq=3)))
    assert resumed.success and resumed.qp_twist[4]>1e-3


@pytest.mark.parametrize('budget',[0.,.01])
def test_energy_admission_still_limits_continuous_visual_motion(budget):
    qp=solver()
    wrench=np.zeros(6);wrench[4]=-1.
    energy=PortEnergyConstraint(wrench,budget,.05)
    result=qp.solve(datum((.6,.8,.9),energy=energy))
    assert result.success and result.diagnostics['differential_request_m_s']>0
    assert energy.admissible(result.qp_twist,tolerance_w=1e-8,velocity_tolerance=1e-8)
    if budget==0:
        assert result.qp_twist[4]<=1e-8
        assert result.diagnostics['differential_shortfall_m_s']>0
    else:assert result.qp_twist[4]>1e-3


def test_actual_angle_limit_still_blocks_outward_rotation():
    qp=solver(angle_limit_rad=.12)
    result=qp.solve(datum((.6,.8,.9),measured_angle=.12))
    assert result.success and result.diagnostics['differential_request_m_s']>0
    assert result.qp_twist[4]<=1e-8
    assert result.diagnostics['differential_shortfall_m_s']>0


@pytest.mark.parametrize('change',['energy_off','optional_image','visual_off','wrong_allocation'])
def test_runtime_cannot_silently_disable_continuous_dependencies(change):
    config=load_study_config(PROFILE)
    assert validate_study_config(config)['repair_permission_mode']=='continuous'
    if change=='energy_off':config['energy_constraint_enabled']=False
    elif change=='optional_image':config['feature']['required']=False
    elif change=='visual_off':config['qp']['enable_visual']=False
    else:config['qp']['allocation_policy']='legacy_v7'
    with pytest.raises(ValueError,match='continuous visual repair'):
        validate_study_config(config)


def test_running_old_controller_cannot_silently_use_bounded_permission():
    from peirastic.core.capabilities import CapabilityAdvertisement, study_capabilities, CONTINUOUS_VISUAL_CAPABILITY
    from peirastic.core.ipc import CommandHub, CommandClient
    requirements=set(study_capabilities(load_study_config(PROFILE)))
    assert CONTINUOUS_VISUAL_CAPABILITY in requirements
    hub=CommandHub(prefix='continuous_'+uuid.uuid4().hex+'_')
    client=CommandClient(prefix=hub.ctl_name.removesuffix('peirastic_ctl_v2'))
    advert=CapabilityAdvertisement(hub,requirements-{CONTINUOUS_VISUAL_CAPABILITY})
    try:
        with pytest.raises(RuntimeError,match='restart Window A'):
            client.require_capability(CONTINUOUS_VISUAL_CAPABILITY)
        assert client.snapshot()['cmd_seq']==0
        advert.close();advert=CapabilityAdvertisement(hub,requirements)
        for capability in requirements:client.require_capability(capability)
        assert client.snapshot()['cmd_seq']==0
    finally:advert.close();client.close();hub.close()


@pytest.mark.parametrize('mode',[True,'forever',None])
def test_invalid_permission_mode_is_rejected(mode):
    with pytest.raises(ValueError,match='permission_mode'):
        DifferentialRepairConfig(permission_mode=mode)


def test_runtime_continuous_feedback_keeps_final_publication_and_required_image(monkeypatch,capsys):
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from peirastic.realman8dof.modes.contact_active import ContactQpOuter
    from peirastic.tests.test_contact_qp_runtime import make_outer, MemorySink
    from peirastic.tests.test_contact_qp_runtime_config import active_config
    config=active_config();config['feature']['required']=True
    config['source']=load_study_config(PROFILE)['source']
    config['qp']=dict(allocation_policy='differential_repair_v8',
        differential_repair=dict(revision='v8r3_confidence_balance',permission_mode='continuous'))
    baseline,pose=make_outer();sink=MemorySink()
    receiver=SimpleNamespace(observation=None,close=lambda **kw:None)
    clock=[10.]
    monkeypatch.setattr('peirastic.realman8dof.modes.contact_active.time.monotonic',lambda:clock[0])
    active=ContactQpOuter(baseline,config,sink=sink,feature_receiver=receiver)
    active.set_origin(pose,t_s=0.)
    assert 'visual=continuous' in capsys.readouterr().out
    try:
        # This checks transport routing. Long-duration availability is tested
        # above; this constant-wrench fixture is not a tissue dynamics model.
        for i in range(10):
            receiver.observation=replace(observation((.6,.8,.9),now=clock[0],seq=i),
                registration_version=active.registration_version,
                window_version=active.feature_config.window_version,
                calibration_version=active.feature_config.calibration_version)
            wrench=np.array([0.,0.,4.,0.,0.,0.])
            command=active.sample(0.,pose,wrench,dt_actual=.005,f_ext_raw=wrench,
                wrench_source_id='continuous-test',wrench_source_time_s=clock[0],wrench_source_wall_time_ns=i+1)
            assert active.publication_review(active.pending_id,command,now_s=clock[0])
            assert active.publication_started(active.pending_id)
            active.publication_commit(active.pending_id,command,now_s=clock[0],facts={'arm':'sent','rail':'sent'})
            # Ideal kinematic plant; do not freeze pose while commanding a path.
            rotation=Rotation.from_euler('xyz',pose[3:])
            pose=pose.copy();pose[:3]+=rotation.apply(command[:3])*.005
            pose[3:]=(rotation*Rotation.from_rotvec(command[3:]*.005)).as_euler('xyz')
            clock[0]+=.005
        assert active.pending_result.diagnostics['differential_request_m_s']>0
        assert active.command_budget.active is not None
        assert active.command_budget.latched_reason is None
        sample=next(r for r in reversed(sink.records) if r['event']=='control_sample')
        assert sample['repair_episode']['permission_mode']=='continuous'
        assert not sample['repair_episode']['exhausted']
        assert sample['allocation_diagnostics']['qp_delta_omega_y_rad_s']>0
        receiver.observation=None
        with pytest.raises(RuntimeError,match='confidence feedback missing'):
            active.sample(0.,pose,wrench,dt_actual=.005,f_ext_raw=wrench,
                wrench_source_id='continuous-test',wrench_source_time_s=clock[0],wrench_source_wall_time_ns=11)
    finally:active.close()
