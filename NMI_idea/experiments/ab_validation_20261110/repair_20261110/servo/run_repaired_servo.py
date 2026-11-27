#!/usr/bin/env python3
"""Actual source-loop regression after public-observation and entrypoint repair."""
from __future__ import annotations
import copy,hashlib,json,os,platform,subprocess,sys,time
from importlib.metadata import version
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
SOURCE=Path('/media/camp/EXT_DRIVE/Among_US')
OUT=Path(__file__).resolve().parent
for p in [SOURCE,SOURCE/'src']:
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import DMP_playground.run_trajectory_extraction_demo as core
import DMP_playground.core_algorithms.module_c_smoothing as smoothing
from DMP_playground.configs.load_config import load_playground_config
from DMP_playground.prebake.build_patch import build_prebake
from DMP_playground.kmp_honest.prior import build_honest_prior
from DMP_playground.dmp_honest.prior import build_honest_dmp_prior
from DMP_playground.kmp_honest.imaging import build_open_bake
from DMP_playground.dmp_honest.imaging import build_dmp_open_bake
from DMP_playground.kmp_honest.servo import run_honest_kmp_servo
from DMP_playground.dmp_honest.servo import run_honest_dmp_servo

def metrics(r):
    valid=np.asarray(r.visible)>0
    raw=np.asarray(r.image_u_m)
    return dict(steps=len(r.p_cmd),visible_count=int(valid.sum()),visible_fraction=float(valid.mean()),
        truth_geometry_u_abs_mean_mm=float(np.mean(np.abs(raw))*1000),truth_geometry_u_abs_p95_mm=float(np.percentile(np.abs(raw),95)*1000),
        public_visible_u_abs_mean_mm=float(np.mean(np.abs(raw[valid]))*1000) if np.any(valid) else None,
        path_length_m=float(np.linalg.norm(np.diff(r.p_cmd,axis=0),axis=1).sum()),
        max_step_m=float(np.linalg.norm(np.diff(r.p_cmd,axis=0),axis=1).max()),last_phase=float(r.s_ctrl[-1]))

def save_result(name,r):
    valid=np.asarray(r.visible)>0
    # Raw geometric diagnostics are explicitly truth-prefixed. Missing public
    # measurements are NaN and never represented as a centered detection.
    data=dict(p_cmd=r.p_cmd,r_cmd=r.r_cmd,s_ctrl=r.s_ctrl,visible=r.visible,
        truth_image_u_m=r.image_u_m,truth_image_depth_m=r.image_depth_m,
        public_image_u_m=np.where(valid,r.image_u_m,np.nan),public_image_depth_m=np.where(valid,r.image_depth_m,np.nan),
        gamma_pos=r.gamma_pos,gamma_tangent=r.gamma_tangent,external_force=r.external_force,measured_vessel=r.measured_vessel)
    np.savez_compressed(OUT/f'{name}.npz',**data)

def main():
    os.chdir(SOURCE)
    original=smoothing.RuckigPositionPlanner
    cfg=load_playground_config(OUT/'causal_module_c_bypass_config.json')
    base,vessels,trunk=build_prebake(cfg,OUT/'prepared_scene')
    s,_=core._polyline_s(trunk.surface_points)
    s,state,_=core._trajectory_state_from_trunk(trunk,base,s,fit_cfg=cfg['controller'])
    ids,weights,canonical=core._pose_resources(cfg,base)
    def extract(deg):
        return core._extract_for_pose(label=f'knee{deg}',knee_bend_deg=deg,beta_scale=1.,config=cfg,output_dir=OUT/'prepared_scene',base_patch=base,vessels=vessels,trunk_label=trunk.label,atlas_full_vertex_indices=ids,atlas_weights=weights,canonical_dir=canonical,s_topology=s,state_topology=state)
    legal=[extract(0),extract(30)];test=extract(60)
    kp=build_honest_prior(legal);dp=build_honest_dmp_prior(legal)
    kb=build_open_bake(kp,test,cfg);db=build_dmp_open_bake(dp,test,cfg)
    db=replace(db,offset_vessel=kb.offset_vessel.copy(),offset_tangents=kb.offset_tangents.copy())
    configs={
        'ruckig_on':load_playground_config(SOURCE/'DMP_playground/configs/smplx_recovered_20261108.json'),
        'ruckig_off_fallback':load_playground_config(OUT/'causal_no_ruckig_config.json'),
        'module_c_bypass':cfg,
    }
    rows=[];missing={}
    def run(name,fn,cfg_run):
        calls=0
        def construct(*args,**kwargs):
            nonlocal calls
            calls+=1
            return original(*args,**kwargs)
        t0=time.perf_counter()
        with patch.object(smoothing,'RuckigPositionPlanner',side_effect=construct):r=fn()
        data=dict(id=name,wall_s=time.perf_counter()-t0,ruckig_constructor_calls=calls,
            effective_module_c_cfg=core._closed_loop_module_c_cfg(cfg_run),**metrics(r))
        rows.append(data);save_result(name,r)
        print(json.dumps(data),flush=True)
        (OUT/'metrics.json').write_text(json.dumps(dict(rows=rows,missing=missing),indent=2))
        return r
    for method,prior,bake,servo in [('kmp',kp,kb,run_honest_kmp_servo),('dmp',dp,db,run_honest_dmp_servo)]:
        for mode,config in configs.items():
            result=run(f'{method}_{mode}',lambda:servo(prior=prior,test_traj=test,cfg=config,module_c_cfg=core._closed_loop_module_c_cfg(config),output_dir=OUT/f'simulation/{method}_{mode}',bake=bake,gain_scale=1.,use_module_c=True),config)
            assert rows[-1]['ruckig_constructor_calls']==(1 if mode=='ruckig_on' else 0)
        pair=[]
        for shift in [-.010,.010]:
            vessel=kb.offset_vessel+shift*kb.r[:,:,0]-.120*kb.r[:,:,2]
            tangent=np.gradient(vessel,axis=0);tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
            hidden=replace(bake,offset_vessel=vessel,offset_tangents=tangent)
            name=f'{method}_missing_hidden_{shift:+.3f}'
            result=run(name,lambda:servo(prior=prior,test_traj=test,cfg=cfg,module_c_cfg=core._closed_loop_module_c_cfg(cfg),output_dir=OUT/f'simulation/{name}',bake=hidden,gain_scale=1.,use_module_c=True),cfg)
            pair.append(result)
        a,b=pair
        assert not np.any(a.visible) and not np.any(b.visible)
        difference=np.linalg.norm(a.p_cmd-b.p_cmd,axis=1)
        missing[method]=dict(same_public_prior=True,all_steps_missing=True,steps=len(a.p_cmd),
            hidden_lateral_offsets_m=[-.010,.010],depth_extra_m=.120,position_max_difference_m=float(difference.max()),
            rotation_max_difference=float(np.max(np.abs(a.r_cmd-b.r_cmd))),
            control_input_max_difference=float(np.max(np.abs(a.external_force-b.external_force))),
            all_missing_confidence_zero=bool(np.all(a.gamma_pos==0)&np.all(b.gamma_pos==0)),
            all_public_image_u_missing=bool(np.all(np.isnan(a.public_image_u_m))&np.all(np.isnan(b.public_image_u_m))))
        assert missing[method]['position_max_difference_m']==0
        assert missing[method]['rotation_max_difference']==0
        assert missing[method]['control_input_max_difference']==0
    primary=[]
    for mode in ['ruckig_off_fallback','module_c_bypass']:
        for scheduled in [True,False]:
            config=copy.deepcopy(configs[mode])
            if not scheduled:config['closed_loop_servo'].pop('scheduled_lateral_error',None)
            assert not hasattr(core,'use_scheduled_vessel')
            name=f'primary_{mode}_scheduled_{scheduled}'
            result=run(name,lambda:core.run_closed_loop_servo_on_bent(legal[0],test,config,core._closed_loop_module_c_cfg(config),normal_offset_m=.006,output_dir=OUT/f'simulation/{name}',open_loop=None),config)
            assert rows[-1]['ruckig_constructor_calls']==0
            primary.append(name)
    files=[SOURCE/'DMP_playground/kmp_honest/servo.py',SOURCE/'DMP_playground/run_trajectory_extraction_demo.py',SOURCE/'DMP_playground/kmp_honest/test_servo_causality.py',SOURCE/'DMP_playground/tests/test_servo_entrypoint.py',SOURCE/'DMP_playground/dmp_honest/servo.py',OUT/'causal_no_ruckig_config.json',OUT/'causal_module_c_bypass_config.json',Path(__file__).resolve()]
    out=dict(rows=rows,missing=missing,primary_runs=primary,
        provenance=dict(python=sys.executable,python_version=sys.version,packages={x:version(x) for x in ['numpy','scipy','matplotlib','ruckig']},source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()),
        information_contract=dict(honest='Prepared vessel-centered surface crop, known scan demos at0/30deg, held-out60deg body surface; online roll from visible history only; hidden geometry retained solely in truth-prefixed diagnostics.',primary='Oracle-assisted reference only: bent GT rho/roll, vessel tangent path, VesselOracle and anatomy nominal replay; not a causal unknown-target acquisition baseline.'),
        execution_scope='CPU probe-pose path simulation and geometric contact surrogate only; no robot joint trajectory, dIRD, physical force/contact, or real ultrasound execution.')
    (OUT/'metrics.json').write_text(json.dumps(out,indent=2))
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
