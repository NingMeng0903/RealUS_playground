#!/usr/bin/env python3
"""Reproducible CPU validation of imported generators, servo and production atlas."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,platform,subprocess,sys,time
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from adapter import SOURCE,generate_candidates,serializable
OUT=Path(__file__).resolve().parent
PROD=SOURCE/'dataset/processed/anatomy_retarget/leg_volume_coordinates/production'
CANON=SOURCE/'outputs/anatomy_retarget/recovered_smplx_20261108'

def stats(x):
    x=np.asarray(x,float)
    return dict(mean=float(np.mean(x)),p95=float(np.percentile(x,95)),max=float(np.max(x)))

def primitive_tests():
    start=np.array([.15,-.2,.3]);goal=start+np.array([.06,.015,.003])
    R=Rotation.from_rotvec([.4,-.6,.2]).as_matrix();t=np.array([.7,-.1,.2])
    base=generate_candidates(start,goal,rotation=np.eye(3))
    changed=generate_candidates(R@start+t,R@goal+t,normal_xyz=R@np.array([0,0,1.]),rotation=R)
    rows=[]
    for a,b in zip(base,changed):
        err=np.linalg.norm(b['xyz_m']-(a['xyz_m']@R.T+t),axis=1)
        rows.append(dict(id=a['id'],position_se3_max_m=float(err.max()),
            rotation_se3_max=float(np.max(np.abs(b['rotations']-R@a['rotations']))),**a['metrics']))
    (OUT/'candidate_paths.json').write_text(json.dumps(serializable(base),indent=2))
    return dict(rows=rows,claim='Actual primitive generation only; caller-defined candidate demonstration; no servo or feasibility guarantee.')

def geometry_tests():
    from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas,query_atlas_coordinates
    from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.surface_refine import SurfaceAtlasRefiner
    from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.volume_refine import VolumeTetRefiner
    from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.lbs_bridge import apply_lbs_pose,inverse_lbs_pose,LbsKinematicState
    from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import joint_global_transforms
    from DMP_playground.core_algorithms.patch_binding import stencil_matmul,load_canonical_bundle,make_pose_axis_angle_55
    a=load_leg_volume_atlas(PROD/'atlas/atlas_right.npz')
    rng=np.random.default_rng(20261110)
    ids=rng.choice(len(a.skin_faces),96,replace=False);faces=a.skin_faces[ids]
    bary=rng.dirichlet(np.ones(3),len(ids))
    points=np.einsum('ni,nij->nj',bary,a.skin_vertices[faces])
    theta=a.skin_theta[faces].astype(float)
    theta=theta[:,0,None]+(theta-theta[:,0,None]+np.pi)%(2*np.pi)-np.pi
    xi=np.column_stack(((bary*theta).sum(1)%(2*np.pi),(bary*a.skin_h[faces]).sum(1),np.zeros(len(ids))))
    ref=SurfaceAtlasRefiner.from_atlas(a)
    t0=time.perf_counter();decoded=ref.xi_to_p(xi);decode_s=time.perf_counter()-t0
    st=np.load(PROD/'material/surface_stencil_right.npz')
    canon=load_canonical_bundle(CANON)
    weights=stencil_matmul(st['stencil_indptr'],st['stencil_indices'],st['stencil_weights'],canon['lbs_weights'][st['source_full_vertex_indices']])
    weights/=weights.sum(1,keepdims=True)
    w=np.einsum('ni,nij->nj',bary,weights[faces]).astype(np.float32)
    G=joint_global_transforms(pose_axis_angle=make_pose_axis_angle_55(knee_bend_deg=60).reshape(-1),rest_joints=canon['rest_joints'],parents=canon['parents'])
    k=G.shape[0];tf=G@canon['inverse_bind'][:k]
    state=LbsKinematicState(tf)
    Rs=np.repeat(np.eye(3,dtype=np.float32)[None],len(points),axis=0)
    known=apply_lbs_pose(points,Rs,w,state)
    can3=apply_lbs_pose(points,Rs,w,state) # same matched registration and attachment inputs
    harmonic=apply_lbs_pose(decoded,Rs,w,state)
    back=inverse_lbs_pose(known.points,known.rotations,w,state)
    # Joint body-frame SE(3) changes after the same evaluated LBS transforms.
    R=Rotation.from_rotvec([.3,-.4,.6]).as_matrix();t=np.array([.31,-.12,.2])
    H=np.eye(4);H[:3,:3]=R;H[:3,3]=t
    moved=apply_lbs_pose(points,Rs,w,LbsKinematicState(H[None]@tf))
    moved_back=inverse_lbs_pose(moved.points,moved.rotations,w,LbsKinematicState(H[None]@tf))
    wrong=xi.copy();wrong[:,0]=(wrong[:,0]+.6)%(2*np.pi)
    wrong_pts=ref.xi_to_p(wrong)
    # Exercise actual harmonic volume interpolation on independently selected stored nodes.
    vid=rng.choice(len(a.harmonic_vertices),16,replace=False)
    vp=a.harmonic_vertices[vid]
    t0=time.perf_counter();vx,_=query_atlas_coordinates(a,vp);vol_s=time.perf_counter()-t0
    target=np.column_stack((a.harmonic_theta[vid],a.harmonic_h[vid],a.harmonic_d[vid]))
    vd=vx-target;vd[:,0]=(vd[:,0]+np.pi)%(2*np.pi)-np.pi
    from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.harmonic import compute_volume_d_distance_ratio
    from scipy.spatial import cKDTree
    skin_ids=np.linspace(0,len(a.skin_vertices)-1,16,dtype=int)
    skin_points=a.skin_vertices[skin_ids]
    skin_xi,_=query_atlas_coordinates(a,skin_points)
    from depth_adapter import corrected_distance_ratio_depth,query_with_corrected_skin_depth
    independent_d=corrected_distance_ratio_depth(a,skin_points)
    corrected_xi,_=query_with_corrected_skin_depth(a,skin_points)
    Rdepth=Rotation.from_rotvec([.2,-.3,.5]).as_matrix();tdepth=np.array([.31,-.2,.7])
    adepth=replace(a,skin_vertices=a.skin_vertices@Rdepth.T+tdepth,core_points=a.core_points@Rdepth.T+tdepth)
    corrected_depth_se3=corrected_distance_ratio_depth(adepth,vp@Rdepth.T+tdepth)-corrected_distance_ratio_depth(a,vp)
    assumed_distance,_=cKDTree(a.harmonic_vertices[:len(a.skin_vertices)]).query(skin_points)
    depth_contract=dict(skin_sample_indices=skin_ids.tolist(),runtime_xi=skin_xi.tolist(),expected_skin_d=np.zeros(16).tolist(),
        independent_distance_ratio_using_actual_skin=independent_d.tolist(),corrected_adapter_xi=corrected_xi.tolist(),corrected_adapter_skin_d_max=float(np.max(np.abs(corrected_xi[:,2]))),
        corrected_adapter_h_theta_unchanged_max=float(np.max(np.abs(corrected_xi[:,:2]-skin_xi[:,:2]))),corrected_adapter_depth_common_se3_max=float(np.max(np.abs(corrected_depth_se3))),
        correction_scope='Only fixes skin-geometry binding for existing analytic distance-ratio depth. Baked layered d is a different field; no full runtime-vs-baked depth equivalence claimed.',runtime_abs_d_max=float(np.abs(skin_xi[:,2]).max()),runtime_abs_d_mean=float(np.abs(skin_xi[:,2]).mean()),
        distance_actual_skin_to_assumed_skin_m=assumed_distance.tolist(),
        source_contract='atlas.query_atlas_coordinates constructs arange(skin_count) indices; harmonic.interpolate_volume_field recomputes distance ratio d from those presumed skin nodes instead of interpolating baked harmonic_d.')
    (OUT/'depth_contract.json').write_text(json.dumps(depth_contract,indent=2))
    # A continuous production skin curve, with no reference/true point passed to decoder.
    curve_xi=np.column_stack((np.full(80,float(np.median(xi[:,0]))),np.linspace(.1,.9,80),np.zeros(80)))
    curve=ref.xi_to_p(curve_xi)
    np.savez_compressed(OUT/'production_geometry_samples.npz',canonical_points=points,xi=xi,decoded=decoded,known_posed=known.points,harmonic_posed=harmonic.points,weights=w,transforms=tf,curve_xi=curve_xi,curve=curve)
    return dict(asset_counts=dict(skin_vertices=len(a.skin_vertices),skin_faces=len(a.skin_faces),harmonic_vertices=len(a.harmonic_vertices),tets=len(a.harmonic_tets)),
        shared_information='Same production mesh, corresponding faces/barycentric weights, canonical SMPL-X LBS weights, evaluated 60-degree knee transforms; no truth/reference points supplied to harmonic decoder.',
        samples=len(ids),surface_decode_error_m=stats(np.linalg.norm(decoded-points,axis=1)),surface_decode_wall_s=decode_s,
        canonical_registered3d_transport_error_m=stats(np.linalg.norm(can3.points-known.points,axis=1)),harmonic_transport_error_m=stats(np.linalg.norm(harmonic.points-known.points,axis=1)),
        exact_lbs_inverse_position_error_m=stats(np.linalg.norm(back.points-points,axis=1)),
        common_se3_transport_error_m=stats(np.linalg.norm(moved.points-(known.points@R.T+t),axis=1)),
        common_se3_inverse_position_error_m=stats(np.linalg.norm(moved_back.points-points,axis=1)),
        wrong_body_address_0_6rad_displacement_m=stats(np.linalg.norm(wrong_pts-decoded,axis=1)),
        harmonic_node_query_error_by_coordinate=np.max(np.abs(vd),axis=0).tolist(),harmonic_node_query_wall_s=vol_s,
        production_skin_runtime_depth_contract=depth_contract,
        production_curve_step_m=stats(np.linalg.norm(np.diff(curve,axis=0),axis=1)),
        limitation='Canonical registered3D baseline is algebraically exact under known matching attachments; harmonic does not outperform this baseline. No nonrigid inference or patient registration was learned/tested.')

def result_metrics(result,wall_s):
    p=result.p_cmd;dt=.1
    steps=np.linalg.norm(np.diff(p,axis=0),axis=1)
    ang=Rotation.from_matrix(np.swapaxes(result.r_cmd[:-1],1,2)@result.r_cmd[1:]).magnitude()
    vis=np.asarray(result.visible)>0
    good=vis&(np.abs(result.image_u_m)<=.003)
    first=np.flatnonzero(good)
    return dict(steps=len(p),visible_fraction=float(vis.mean()),u_abs_mm=stats(np.abs(result.image_u_m)*1000),
        first_synthetic_centered_time_s=float(first[0]*dt) if len(first) else None,
        path_length_m=float(steps.sum()),translation_step_m=stats(steps),rotation_step_rad=stats(ang),wall_s=wall_s,
        nonvisible_motion_length_m=float(steps[~vis[:-1]].sum()),last_phase=float(result.s_ctrl[-1]))

def servo_tests():
    import DMP_playground.run_trajectory_extraction_demo as core
    from DMP_playground.configs.load_config import load_playground_config
    from DMP_playground.prebake.build_patch import build_prebake
    from DMP_playground.kmp_honest.prior import build_honest_prior
    from DMP_playground.dmp_honest.prior import build_honest_dmp_prior
    from DMP_playground.kmp_honest.imaging import build_open_bake
    from DMP_playground.dmp_honest.imaging import build_dmp_open_bake
    from DMP_playground.kmp_honest.servo import run_honest_kmp_servo
    cfg=load_playground_config(SOURCE/'DMP_playground/configs/smplx_recovered_20261108.json')
    base,vessels,trunk=build_prebake(cfg,OUT/'controlled_servo')
    s,_=core._polyline_s(trunk.surface_points)
    s,state,_=core._trajectory_state_from_trunk(trunk,base,s,fit_cfg=cfg['controller'])
    ids,w,canonical=core._pose_resources(cfg,base)
    def extract(deg):
        return core._extract_for_pose(label=f'knee_{deg}',knee_bend_deg=deg,beta_scale=1.,config=cfg,output_dir=OUT/'controlled_servo',base_patch=base,vessels=vessels,trunk_label=trunk.label,atlas_full_vertex_indices=ids,atlas_weights=w,canonical_dir=canonical,s_topology=s,state_topology=state)
    legal=[extract(0),extract(30)];test=extract(60)
    kp=build_honest_prior(legal);dp=build_honest_dmp_prior(legal)
    kb=build_open_bake(kp,test,cfg);db=build_dmp_open_bake(dp,test,cfg)
    # Critical matched-world correction: both priors observe the SAME fixed curve.
    db=replace(db,offset_vessel=kb.offset_vessel.copy(),offset_tangents=kb.offset_tangents.copy())
    rows=[];stored={}
    for method,prior,bake in [('kmp',kp,kb),('dmp',dp,db)]:
        for scenario,shift in [('nominal',0.),('mismatch_lateral_plus10mm',.010),('mismatch_lateral_minus10mm',-.010),('not_detected_depth_plus120mm',0.),('not_detected_depth_plus120mm_lateral_plus10mm',.010),('not_detected_depth_plus120mm_lateral_minus10mm',-.010)]:
            bv=bake.offset_vessel.copy()
            if shift:
                bv+=shift*kb.r[:,:,0]
            if scenario.startswith('not_detected'):
                bv-=.120*kb.r[:,:,2]
            tangent=np.gradient(bv,axis=0);tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
            bb=replace(bake,offset_vessel=bv,offset_tangents=tangent)
            for gain in [0.,1.]:
                name=f'{method}_{scenario}_gain{gain:g}'
                t0=time.perf_counter()
                res=run_honest_kmp_servo(prior=prior,test_traj=test,cfg=cfg,module_c_cfg=core._closed_loop_module_c_cfg(cfg),output_dir=OUT/'controlled_servo'/name,bake=bb,gain_scale=gain,use_module_c=False)
                met=result_metrics(res,time.perf_counter()-t0)
                rows.append(dict(id=name,method=method,scenario=scenario,gain=gain,use_module_c=False,**met))
                stored[name]=res
                np.savez_compressed(OUT/'controlled_servo'/f'{name}.npz',p_cmd=res.p_cmd,r_cmd=res.r_cmd,visible=res.visible,image_u_m=res.image_u_m,image_depth_m=res.image_depth_m,s_ctrl=res.s_ctrl,truth_curve=bv)
        # Same production honest loop with its existing Ruckig Module C enabled.
        name=f'{method}_nominal_ruckig'
        t0=time.perf_counter();res=run_honest_kmp_servo(prior=prior,test_traj=test,cfg=cfg,module_c_cfg=core._closed_loop_module_c_cfg(cfg),output_dir=OUT/'controlled_servo'/name,bake=bake,gain_scale=1.,use_module_c=True)
        rows.append(dict(id=name,method=method,scenario='nominal',gain=1.,use_module_c=True,**result_metrics(res,time.perf_counter()-t0)))
        np.savez_compressed(OUT/'controlled_servo'/f'{name}.npz',p_cmd=res.p_cmd,r_cmd=res.r_cmd,visible=res.visible,image_u_m=res.image_u_m,image_depth_m=res.image_depth_m,s_ctrl=res.s_ctrl,truth_curve=bake.offset_vessel)
    # Whole-loop equivariance, including surface contact and reconstruction.
    R=Rotation.from_rotvec([.25,-.4,.3]).as_matrix();t=np.array([.2,-.1,.15])
    patch=replace(test.patch,vertices=test.patch.vertices@R.T+t,normals=test.patch.normals@R.T,frames=R@test.patch.frames,metadata=dict(test.patch.metadata))
    trunk2=replace(test.trunk,surface_points=test.trunk.surface_points@R.T+t,vessel_points=test.trunk.vessel_points@R.T+t,tangents=test.trunk.tangents@R.T)
    test2=replace(test,patch=patch,trunk=trunk2)
    kb2=replace(kb,p=kb.p@R.T+t,r=R@kb.r,offset_vessel=kb.offset_vessel@R.T+t,offset_tangents=kb.offset_tangents@R.T)
    t0=time.perf_counter();moved=run_honest_kmp_servo(prior=kp,test_traj=test2,cfg=cfg,module_c_cfg=core._closed_loop_module_c_cfg(cfg),output_dir=OUT/'controlled_servo/se3',bake=kb2,gain_scale=1.,use_module_c=False)
    ref=stored['kmp_nominal_gain1']
    eq=dict(position_max_m=float(np.max(np.linalg.norm(moved.p_cmd-(ref.p_cmd@R.T+t),axis=1))),rotation_max=float(np.max(np.abs(moved.r_cmd-R@ref.r_cmd))),image_u_max_m=float(np.max(np.abs(moved.image_u_m-ref.image_u_m))),wall_s=time.perf_counter()-t0)
    np.savez_compressed(OUT/'controlled_servo/se3.npz',p_cmd=moved.p_cmd,r_cmd=moved.r_cmd,image_u_m=moved.image_u_m,s_ctrl=moved.s_ctrl,R=R,t=t)
    # Strongest configurable primary loop. Its information contract differs:
    # bent replay roll/rho, exact vessel geometry and VesselOracle are inputs.
    t0=time.perf_counter()
    prodm={}
    try:
        prod=core.run_closed_loop_servo_on_bent(legal[0],test,cfg,core._closed_loop_module_c_cfg(cfg),normal_offset_m=.006,output_dir=OUT/'controlled_servo/primary_production',open_loop=None)
        prodm['stock']=result_metrics(prod,time.perf_counter()-t0)
    except Exception as e:
        import traceback
        (OUT/'logs/primary_production_stock_failure.log').write_text(traceback.format_exc())
        prodm['stock']={'error':repr(e),'wall_s':time.perf_counter()-t0}
        # Explicit standalone harness repair: the stock function references a
        # nonexistent global/local name. Bind it to its existing schedule flag.
        # This does not change original source and is a separate result.
        if isinstance(e,NameError) and 'use_scheduled_vessel' in str(e):
            core.use_scheduled_vessel=core.scheduled_observation_truth_enabled(cfg['closed_loop_servo'])
            t0=time.perf_counter()
            try:
                prod=core.run_closed_loop_servo_on_bent(legal[0],test,cfg,core._closed_loop_module_c_cfg(cfg),normal_offset_m=.006,output_dir=OUT/'controlled_servo/primary_production_harness_alias',open_loop=None)
                prodm['harness_global_alias_repair']=result_metrics(prod,time.perf_counter()-t0)
                prodm['harness_global_alias_repair']['change']='Set module global use_scheduled_vessel to scheduled_observation_truth_enabled(cfg[closed_loop_servo]); no source file modified.'
                np.savez_compressed(OUT/'controlled_servo/primary_production_harness_alias.npz',p_cmd=prod.p_cmd,r_cmd=prod.r_cmd,visible=prod.visible,image_u_m=prod.image_u_m,s_ctrl=prod.s_ctrl)
            except Exception as e2:
                prodm['harness_global_alias_repair']={'error':repr(e2),'wall_s':time.perf_counter()-t0}
                (OUT/'logs/primary_production_repair_failure.log').write_text(traceback.format_exc())
            finally:
                del core.use_scheduled_vessel
    prodm['source']='DMP_playground.run_trajectory_extraction_demo.run_closed_loop_servo_on_bent + SurfaceScanDMPController'
    prodm['privileged_inputs']='Exact bent anatomy/vessel points and tangents, bent replay roll/rho, VesselOracle; not a blind first-image baseline.'
    missing={}
    for method in ['kmp','dmp']:
        inv0=stored[f'{method}_not_detected_depth_plus120mm_gain0'];inv1=stored[f'{method}_not_detected_depth_plus120mm_gain1']
        plus=stored[f'{method}_not_detected_depth_plus120mm_lateral_plus10mm_gain1'];minus=stored[f'{method}_not_detected_depth_plus120mm_lateral_minus10mm_gain1']
        plus0=stored[f'{method}_not_detected_depth_plus120mm_lateral_plus10mm_gain0'];minus0=stored[f'{method}_not_detected_depth_plus120mm_lateral_minus10mm_gain0']
        observed=np.flatnonzero((plus.visible>0)|(minus.visible>0))
        prefix=int(observed[0]) if len(observed) else len(plus.visible)
        missing[method]=dict(same_world_gain1_vs_gain0_position_difference_m=stats(np.linalg.norm(inv1.p_cmd-inv0.p_cmd,axis=1)),
            two_hidden_offsets_visible_counts=[int(np.count_nonzero(plus.visible)),int(np.count_nonzero(minus.visible))],
            gain1_hidden_world_path_difference_m=stats(np.linalg.norm(plus.p_cmd-minus.p_cmd,axis=1)),
            identical_missing_prefix_steps=prefix,identical_missing_prefix_gain1_path_difference_m=stats(np.linalg.norm(plus.p_cmd[:prefix]-minus.p_cmd[:prefix],axis=1)),
            same_public_missing_history_gain0_hidden_world_path_difference_m=stats(np.linalg.norm(plus0.p_cmd-minus0.p_cmd,axis=1)),
            inference='Nominal motion while missing is allowed. Compare only common prefix before either world detects. Hidden-world dependence under identical visible=False observations arises because stock control additionally receives simulator geometric image_u even when invisible; this fails a missing-only sensor contract, not the internal simulator contract.')
    return dict(common_target='KMP scheduled fixed world vessel reused verbatim by both controllers; lateral/depth perturbations applied once in fixed world before execution.',legal_prior_knees=[0,30],test_knee=60,rows=rows,missing_observation_counterfactual=missing,full_honest_loop_se3=eq,primary_production=prodm,
        known_source_behavior='Lateral correction uses image_u even when visible=False; reconstruction ingestion alone is visibility-gated. Negative visibility tests diagnose this behavior, not acquisition success.',
        scope='Synthetic probe-plane geometry and surrogate contact only. No real US, robot, force feedback, target identity detector, stop/confirmation, or dIRD inside these servo runs.')

def provenance():
    from importlib.metadata import version,PackageNotFoundError
    files=list((SOURCE/'DMP_playground/kmp_honest').glob('*.py'))+list((SOURCE/'DMP_playground/dmp_honest').glob('*.py'))
    files += [SOURCE/'DMP_playground/run_trajectory_extraction_demo.py',SOURCE/'DMP_playground/core_algorithms/controller.py',SOURCE/'DMP_playground/core_algorithms/module_c_smoothing.py',SOURCE/'DMP_playground/core_algorithms/patch_binding.py',SOURCE/'DMP_playground/configs/smplx_recovered_20261108.json']
    files+=list((SOURCE/'src/projects/genesis_ue_sync/anatomy_retarget/leg_volume_coordinates').glob('*.py'))
    files+=[PROD/'atlas/atlas_right.npz',PROD/'material/surface_stencil_right.npz',PROD/'vessels/vessel_material_coordinates.npz',PROD/'manifest.json',PROD.parent/'bake/base_atlas/atlas_right.npz',CANON/'smpl_canonical_weights.npz',CANON/'source_manifest.json',CANON/'smpl_canonical_tpose.obj',CANON/'smpl_canonical_tpose_neutral.obj',CANON/'vessel_centerlines_rest.obj']
    files+=list((SOURCE/'DMP_playground/configs').glob('*.json'))
    files+=[OUT/'adapter.py',OUT/'depth_adapter.py',Path(__file__).resolve()]
    packages={}
    for package in ['numpy','scipy','matplotlib','ruckig']:
        try:packages[package]=version(package)
        except PackageNotFoundError:packages[package]='not recorded in package metadata'
    return dict(python=sys.executable,version=sys.version,platform=platform.platform(),packages=packages,numpy=np.__version__,threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),dirty_status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=SOURCE,text=True),seed=20261110)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sections',nargs='+',default=['primitive','production_geometry','servo'])
    args=parser.parse_args()
    os.chdir(SOURCE)
    out=json.loads((OUT/'metrics.json').read_text()) if (OUT/'metrics.json').is_file() else {}
    out['provenance']=provenance()
    (OUT/'metrics.json').write_text(json.dumps(out,indent=2))
    for name,fn in [('primitive',primitive_tests),('production_geometry',geometry_tests),('servo',servo_tests)]:
        if name not in args.sections:continue
        print('RUN',name,flush=True);t0=time.perf_counter()
        try:out[name]=fn();out[name]['total_wall_s']=time.perf_counter()-t0
        except Exception as e:
            import traceback
            traceback.print_exc();out[name]={'error':repr(e)}
        (OUT/'metrics.json').write_text(json.dumps(out,indent=2))
        print('DONE',name,flush=True)
    print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='source_sha256' and kk!='dirty_status'} for k,v in out.items() if k!='provenance'},indent=2))
if __name__=='__main__':main()
