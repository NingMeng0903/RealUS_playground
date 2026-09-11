from pathlib import Path
from copy import deepcopy
import hashlib
from types import SimpleNamespace
from unittest.mock import patch
import json
import numpy as np
from scipy.spatial.transform import Rotation
from peirastic.tests.test_contact_qp_runtime import make_outer,MemorySink
from peirastic.realman8dof.modes.contact_active import ContactQpOuter
from peirastic.contact_qp.runtime_config import load_study_config
from peirastic.contact_qp.types import ContactObservation

config=load_study_config('peirastic/config/contact_qp/active_probe50_v8r3_tank.yaml')
w=np.array([0.,1.,4.,0.,0.,0.])
def run(label,cfg,images=True,warm=0,move=False):
    outer,pose=make_outer();sink=MemorySink();recv=SimpleNamespace(observation=None,close=lambda **kw:None)
    active=ContactQpOuter(outer,cfg,sink=sink,feature_receiver=recv);active.set_origin(pose,t_s=0.)
    # Consume genuine production nominal measurements/updates before switching
    # to QP transactions; no controller states or gains are manually patched.
    if warm:
        for i in range(warm):
            v=active.nominal.update(pose=pose,f_des=np.array([0.,0.,4.,0.,0.,0.]),f_ext=w,
                f_ext_raw=w,dt_s=.005,dt_actual=.005,source_dt_s=.005,path_twist=np.zeros(6),
                sensor_age_s=0.,feedback_age_s=0.,feedback_fresh_tick=True,
                feedback_velocity_valid=True,v_tcp_z_actual=0.,slack_norm=0.).v_force
    clock=[10.]
    result=dict(label=label,warm=warm,images=images,pose_integrated=move,successes=0,
                warm_contact=bool(active.controller.contact_present))
    with patch('peirastic.realman8dof.modes.contact_active.time.monotonic',lambda:clock[0]):
        for i in range(200):
            clock[0]=10.+i*.005
            if images:
                recv.observation=ContactObservation(i,'fixture',clock[0]-.152,clock[0],np.array([.82,.90,.97]),np.ones(3,dtype=bool),
                    cfg['feature']['registration_version'],cfg['feature']['window_version'],calibration_version=cfg['geometry']['calibration_version'])
            try:
                cmd=active.sample(i*.005,pose,w,f_ext_raw=w,dt_actual=.005,wrench_source_time_s=clock[0],wrench_source_id='fixture',wrench_source_wall_time_ns=int(clock[0]*1e9))
                assert active.publication_review(active.pending_id,cmd,now_s=clock[0])
                active.publication_started(active.pending_id);clock[0]+=.0005
                active.publication_commit(active.pending_id,cmd,now_s=clock[0],facts={'arm':'sent','rail':'sent'})
                result['successes']+=1
                if move:
                    r=Rotation.from_euler('xyz',pose[3:]);pose[:3]+=r.apply(cmd[:3])*.005
                    pose[3:]=(r*Rotation.from_rotvec(cmd[3:]*.005)).as_euler('xyz')
            except Exception as e:
                result['failure']=str(e)
                rejected=[r for r in sink.records if r['event']=='qp_prepare_rejected']
                if rejected:
                    record=rejected[-1]
                    result['nominal']=record['nominal_twist_tool'].tolist();result['previous']=record['previous_twist_tool'].tolist()
                break
    result['contact_present']=bool(active.controller.contact_present)
    result['final_energy']=None if active.command_budget is None else active.command_budget.facts
    result['nominal_warm_last_twist']=None if not warm else v.tolist()
    active.close();return result
r2=deepcopy(config);r2['energy_constraint_enabled']=False;r2['energy']=None;r2['qp']['differential_repair']['revision']='v8r2_bounded_episode'
results=[run('r3_warm_static_image',config,True,warm=400),run('r3_cold_static_image',config),run('r2_cold_static_image',r2),run('r3_cold_static_noimage',config,False),run('r3_cold_kinematic_image',config,True,move=True)]
report=dict(assurance='detached software fixture, no hardware or physical certification',warmup='400 production nominal.update calls at 4 N / 5 ms, zero path, no internal-state or gain patch',profile_sha256=hashlib.sha256(Path('peirastic/config/contact_qp/active_probe50_v8r3_tank.yaml').read_bytes()).hexdigest(),results=results)
Path('analysis_artifacts/contact_qp_v8r3/profile_fixture_compare.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
