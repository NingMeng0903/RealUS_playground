"""Independent saved native trace audit, including same-time FK and spline arc."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pinocchio as pin
from scipy.integrate import quad
from scipy.spatial.transform import Rotation,Slerp

HERE=Path(__file__).resolve().parent;REPAIR=HERE.parents[1];BASE=REPAIR/'qp_cross_track_fix';ROOT=Path('/media/camp/EXT_DRIVE/RealUS_playground')
sys.path.insert(0,str(ROOT/'rm75_control'))
from rm75_control.control.joint_admittance_8dof import scan_reference as source
from rm75_control.control.joint_admittance_8dof.collision_model import CollisionModel
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def maxabs(a):return float(np.max(np.abs(a)))
def geodesic(A,B):return np.degrees((Rotation.from_matrix(A).inv()*Rotation.from_matrix(B)).magnitude())
meta=json.loads((BASE/'smooth_reference_results.json').read_text());assert len(meta['runs'])==4,'only audit sealed complete four-case result'
assert meta['production_reference_sha256']==sha(source.__file__)
assert meta['protocol_sha256']==sha(BASE/'SMOOTH_REFERENCE_PROTOCOL.txt') and meta['code_sha256']==sha(BASE/'run_smooth_reference.py')
inputs=np.load(REPAIR/'audit/phase2_native_subset_inputs.npz');assert meta['input_sha256']==sha(REPAIR/'audit/phase2_native_subset_inputs.npz')
contract=json.loads((REPAIR/'dird/qp_native_urdf/manifest.json').read_text())['robot_contract']
model=pin.buildModelFromUrdf(contract['kinematics_urdf']);data=model.createData();fid=model.getFrameId('tcp')
collision=CollisionModel(model,collision_urdf=contract['collision_urdf'],pair_config=contract['collision_pairs'])
out={'scope':'all four saved smooth-reference runs, independent same-time URDF FK and derivatives; no native rerun','input_sha256':meta['input_sha256'],'source_sha256':meta['production_reference_sha256'],'result_sha256':sha(BASE/'smooth_reference_results.json'),'runs':{},'paired_weight_checks':{}}

def region_stats(actual,desired,times,duration,arc,tangent):
    e=desired[:,:3,3]-actual[:,:3,3];pos=np.linalg.norm(e,axis=1)*1000;along=np.sum(e*tangent,axis=1)*1000
    normal=np.linalg.norm(e-along[:,None]/1000*tangent,axis=1)*1000;angle=geodesic(desired[:,:3,:3],actual[:,:3,:3]);moving=times<duration
    masks={'all_motion':np.ones(len(times),bool),'middle_phase':(times>=.8)&(times<=duration-.8),'arc20_80':(arc>=.2)&(arc<=.8)&moving}
    masks.update({f'arc_decile_{i}':(arc>=i/10)&((arc<(i+1)/10) if i<9 else (arc<=1))&moving for i in range(10)})
    result={}
    for key,mask in masks.items():
        r={'samples':int(mask.sum())}
        if mask.any():
            r.update(time_min_s=float(times[mask].min()),time_max_s=float(times[mask].max()),arc_min=float(arc[mask].min()),arc_max=float(arc[mask].max()))
            for label,a in [('position',pos),('normal',normal),('along_abs',abs(along)),('angle',angle)]:
                unit='deg' if label=='angle' else 'mm';r[f'{label}_max_{unit}']=float(a[mask].max());r[f'{label}_p95_{unit}']=float(np.percentile(a[mask],95))
            r['pose_2mm_2deg']=bool(pos[mask].max()<=2 and angle[mask].max()<=2)
            r['fine_pose_0p2mm_0p1deg']=bool(pos[mask].max()<=.2 and angle[mask].max()<=.1)
        result[key]=r
    result['peaks']={}
    for name,values in [('position_mm',pos),('angle_deg',angle)]:
        k=int(values.argmax());result['peaks'][name]={'tick':k,'time_s':float(times[k]),'arc_fraction':float(arc[k]),'value':float(values[k])}
    return result

for r in meta['runs']:
    name=r['case'];path=int(name[4]);trace=BASE/(name+'_trace.npz');native=BASE/(name+'_native_config.txt')
    assert sha(trace)==r['trace_sha256'] and sha(native)==r['native_config_sha256']
    z=np.load(trace);T=inputs[f'branch4_path{path}_T'];q0=inputs[f'branch4_path{path}_q_seed'];dt=r['dt_s'];times=z['t_s'];n=len(times)
    ref=source.WorldPoseSplineReference(T[:,:3,3],rpy=Rotation.from_matrix(T[:,:3,:3]).as_euler('xyz'),speed_m_s=.02,ramp_s=.8)
    duration=ref.duration_s();assert duration==r['nominal_path_duration_s']
    poses_before=[];poses_after=[];J=[]
    for qb,qa in zip(z['q_meas_before'],z['q']):
        pin.computeJointJacobians(model,data,qb);pin.updateFramePlacements(model,data);poses_before.append(data.oMf[fid].homogeneous.copy());J.append(pin.getFrameJacobian(model,data,fid,pin.LOCAL_WORLD_ALIGNED).copy())
        pin.forwardKinematics(model,data,qa);pin.updateFramePlacements(model,data);poses_after.append(data.oMf[fid].homogeneous.copy())
    before=np.asarray(poses_before);after=np.asarray(poses_after);J=np.asarray(J)
    assert maxabs(after-z['pose_actual'])<1e-12
    assert np.array_equal(z['q_meas_before'][0],q0) and np.array_equal(z['q_meas_before'][1:],z['q'][:-1])
    assert np.array_equal(z['q_send_command'],z['q']) and np.array_equal(times,np.arange(n)*dt)
    assert maxabs(z['native_model_t_mono']-(1000+times))<1e-12
    expected=[];future=[];ff=[];u=[]
    for t in times:
        for collection,tt in [(expected,t),(future,t+dt)]:
            a=ref.sample(tt);pose=np.eye(4);pose[:3,3]=a.pose_d[:3];pose[:3,:3]=Rotation.from_euler('xyz',a.pose_d[3:]).as_matrix();collection.append(pose)
        ff.append(ref.sample(t).vel_ff);u.append(ref.parameter_state(t)[0])
    expected=np.asarray(expected);future=np.asarray(future);ff=np.asarray(ff);u=np.asarray(u)
    assert maxabs(expected-z['pose_desired'])<1e-12 and maxabs(ff-z['vel_ff'])<1e-12
    tangent=ref.position_spline(u,1);tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    knots=ref.knots;lengths=np.array([quad(lambda v:float(np.linalg.norm(ref.position_spline(v,1))),a,b,epsabs=1e-13,epsrel=1e-13)[0] for a,b in zip(knots[:-1],knots[1:])]);cum=np.r_[0.,np.cumsum(lengths)]
    segment=np.clip(np.searchsorted(knots,u,side='right')-1,0,len(knots)-2)
    nodes,weights=np.polynomial.legendre.leggauss(32);half=(u-knots[segment])/2;mid=(u+knots[segment])/2
    part=half*np.sum(np.linalg.norm(ref.position_spline(mid[:,None]+half[:,None]*nodes,1),axis=2)*weights,axis=1)
    arc=np.clip((cum[segment]+part)/cum[-1],0,1)
    ep=expected[:,:3,3]-before[:,:3,3];er=(Rotation.from_matrix(expected[:,:3,:3])*Rotation.from_matrix(before[:,:3,:3]).inv()).as_rotvec()
    received=ff+np.c_[12*ep,2*er];assert maxabs(received-z['v_cmd_received'])<1e-10
    body=np.einsum('nij,nj->ni',before[:,:3,:3],z['twist_requested'][:,:3]);bodyR=np.einsum('nij,nj->ni',before[:,:3,:3],z['twist_requested'][:,3:])
    assert maxabs(np.c_[body,bodyR]-z['v_cmd_received'])<1e-12
    Jtask=J.copy();Jtask[:,:,0]=0;affine=J[:,:,0]*np.r_[r['qdot_entry'][0],z['qdot'][:-1,0]][:,None]
    estimated=np.einsum('nij,nj->ni',Jtask,z['qdot'])+affine
    v=z['qdot'];a=np.diff(np.vstack([r['qdot_entry'],v]),axis=0)/dt;j=np.diff(np.vstack([r['acceleration_entry'],a]),axis=0)/dt
    assert maxabs(a-z['command_acceleration'])<1e-10 and maxabs(j-z['command_jerk'])<1e-8
    margins={label:float(np.min(np.asarray(r[key])-abs(vals))) for label,key,vals in [('velocity','velocity_limit_by_coordinate',v),('acceleration','acceleration_limit_by_coordinate',a),('jerk','jerk_limit_by_coordinate',j)]}
    for label,val in margins.items():assert abs(val-r[label+'_margin_min'])<1e-9
    qmargin=np.minimum(z['q']-model.lowerPositionLimit,model.upperPositionLimit-z['q'])
    indices=np.unique(np.r_[np.linspace(0,n-1,65,dtype=int),np.argmin(z['min_self_distance_m']),np.argmax(np.linalg.norm(ep,axis=1)),np.argmax(np.linalg.norm(er,axis=1)),np.unravel_index(np.argmax(abs(j)-np.asarray(r['jerk_limit_by_coordinate'])),j.shape)[0]])
    cerror=0.
    for k in indices:
        collision.update(z['q'][k]);d=min(x.min_distance for x in collision.geom_data.distanceResults);cerror=max(cerror,abs(d-z['min_self_distance_m'][k]))
    assert cerror<1e-12
    old_p=T[:,:3,3];edges=np.diff(old_p,axis=0);chord=np.linalg.norm(edges,axis=1);f=(u-knots[segment])/chord[segment]
    oldsame=np.tile(np.eye(4),(n,1,1));oldsame[:,:3,3]=old_p[segment]+f[:,None]*edges[segment];oldsame[:,:3,:3]=Slerp(knots,Rotation.from_matrix(T[:,:3,:3]))(u).as_matrix()
    strict=all(x>=0 for x in margins.values()) and qmargin.min()>=0 and z['min_self_distance_m'].min()>=.01 and not len(r['failures']) and not z['solver_fault_latched'].any()
    worst=np.unravel_index(np.argmax(abs(j)-np.asarray(r['jerk_limit_by_coordinate'])),j.shape)
    out['runs'][name]={'trace_sha256':sha(trace),'native_config_sha256':sha(native),'samples':n,'duration_s':duration,'integrated_duration_s':n*dt,'reported_soft_start_ramp_s':r['soft_start_ramp_s'],'actual_symmetric_ramp_s':ref.effective_ramp_s,
      'same_time_state_chain_exact':True,'FK_saved_post_update_max_abs':maxabs(after-z['pose_actual']),'reference_pose_max_abs':maxabs(expected-z['pose_desired']),'reference_FF_max_abs':maxabs(ff-z['vel_ff']),
      'outer_received_FF_plus_feedback_max_abs':maxabs(received-z['v_cmd_received']),'J_arm_plus_previous_rail_vs_estimated_max_abs':maxabs(estimated-z['v_tcp_estimated']),'feasible_vs_estimated_max_abs':maxabs(z['v_cmd_feasible']-z['v_tcp_estimated']),
      'requested_vs_feasible_linear_max_m_s':float(np.linalg.norm(z['v_cmd_received'][:,:3]-z['v_cmd_feasible'][:,:3],axis=1).max()),'requested_vs_feasible_angular_max_rad_s':float(np.linalg.norm(z['v_cmd_received'][:,3:]-z['v_cmd_feasible'][:,3:],axis=1).max()),
      'old_post_update_vs_same_tick_spline':region_stats(after,expected,times,duration,arc,tangent),
      'aligned_before_vs_same_tick_spline':region_stats(before,expected,times,duration,arc,tangent),
      'aligned_next_vs_next_tick_spline':region_stats(after,future,(np.arange(n)+1)*dt,duration,np.r_[arc[1:],arc[-1]],np.vstack([tangent[1:],tangent[-1]])),
      'aligned_before_vs_old_polyline_same_parameter':region_stats(before,oldsame,times,duration,arc,tangent),
      'mechanical':{'minimum_q_margin':float(qmargin.min()),'minimum_command_margins':margins,'worst_jerk':{'tick':int(worst[0]),'coordinate':int(worst[1]),'jerk':float(j[worst]),'box':[float(z['box_lo'][worst]),float(z['box_hi'][worst])]},'minimum_sampled_self_distance_m':float(z['min_self_distance_m'].min()),'collision_independent_subset_samples':len(indices),'collision_pairs_each':len(collision.geom_model.collisionPairs),'collision_subset_max_difference_m':cerror,'solver_fault_count':int(z['solver_fault_latched'].sum()),'pause_count':int(z['task_paused'].sum()),'strict_mechanical_gate':bool(strict),'reported_gate':r['strict_mechanical_gate']}}
    assert bool(strict)==r['strict_mechanical_gate']
    print(name,'ALIGNED',out['runs'][name]['aligned_before_vs_same_tick_spline']['middle_phase'],'MECHANICAL',out['runs'][name]['mechanical'],flush=True)
for path in (0,1):
    an=f'path{path}_smooth_baseline';bn=f'path{path}_smooth_unit';a=np.load(BASE/(an+'_trace.npz'));b=np.load(BASE/(bn+'_trace.npz'))
    ca=(BASE/(an+'_native_config.txt')).read_text().splitlines();cb=(BASE/(bn+'_native_config.txt')).read_text().splitlines();diff=[(x,y) for x,y in zip(ca,cb) if x!=y]
    assert len(ca)==len(cb) and len(diff)==1 and diff[0][0].startswith('qp.task_weight ')
    assert np.array_equal(a['pose_desired'],b['pose_desired']) and np.array_equal(a['vel_ff'],b['vel_ff']) and np.array_equal(a['t_s'],b['t_s'])
    out['paired_weight_checks'][str(path)]={'reference_and_clock_and_FF_bit_identical':True,'native_config_only_difference':diff,'q_max_abs_difference':maxabs(a['q']-b['q'])}
(HERE/'smooth_traces_independent.json').write_text(json.dumps(out,indent=2)+'\n')
print('ALL FOUR SMOOTH TRACES VERIFIED',flush=True)
