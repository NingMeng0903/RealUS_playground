"""Independent read-only reconstruction of sealed reference/QP trace evidence."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
import pinocchio as pin

HERE=Path(__file__).resolve().parent
REPAIR=HERE.parents[1]
clean_run=len(sys.argv)>1 and sys.argv[1]=='clean'
RUN=REPAIR/('qp_clock_repair' if clean_run else 'qp_reference_followup')
SRC=Path('/media/camp/EXT_DRIVE/RealUS_playground')
sys.path[:0]=[str(SRC),str(SRC/'rm75_control')]
from rm75_control.control.joint_admittance_8dof.reference import WorldPolylineReference

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=json.loads((RUN/'RESULTS.json').read_text());inputs=np.load(REPAIR/'audit/phase2_native_subset_inputs.npz');T=inputs['branch4_path0_T'];points=T[:,:3,3];rot=Rotation.from_matrix(T[:,:3,:3]);rpy=rot.as_euler('xyz');arcs=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
manifest=json.loads((REPAIR/'dird/qp_native_urdf/manifest.json').read_text());urdf=manifest['robot_contract']['kinematics_urdf'];model=pin.buildModelFromUrdf(urdf);data=model.createData();tcp_id=model.getFrameId('tcp')

def independent_reference(t,speed):
    t=max(0.,float(t));ramp=.4
    if t<ramp:
        u=t/ramp;tau=ramp*(u**6-3*u**5+2.5*u**4);td=6*u**5-15*u**4+10*u**3
    else:tau=t-.5*ramp;td=1.
    s=min(arcs[-1],speed*tau);i=int(np.clip(np.searchsorted(arcs,s,side='right')-1,0,len(arcs)-2));span=arcs[i+1]-arcs[i];u=(s-arcs[i])/span;R=Slerp([0,1],rot[i:i+2])([u])[0].as_matrix();p=(1-u)*points[i]+u*points[i+1];vel=np.zeros(6)
    if s<arcs[-1]:
        vel[:3]=(points[i+1]-points[i])/span*speed*td
        vel[3:]=rot[i].apply((rot[i].inv()*rot[i+1]).as_rotvec())*speed*td/span
    pose=np.eye(4);pose[:3,:3]=R;pose[:3,3]=p
    return pose,vel,s,i

out={'scope':'read-only sealed trace/reference recomputation, no QP execution or production edit','hashes':{str(RUN/x):sha(RUN/x) for x in ['REPORT.txt','PROTOCOL.txt','RESULTS.json','run_reference_protocol.py','derive_reference_velocities.py','run_clean_reference.py','input_manifest.json'] if (RUN/x).is_file()},'input_hash_matches':sha(REPAIR/'audit/phase2_native_subset_inputs.npz')==d['input_sha256'],'current_config_matches_earlier_sealed_config':sha(manifest['config'])==manifest['config_sha256'],'current_urdf_matches_earlier_sealed_urdf':sha(urdf)==manifest['robot_contract']['kinematics_urdf_sha256'],'earlier_config_eps_abs':manifest['effective_config']['qp']['eps_abs'],'runs':{}}
for result in d['runs']:
    name=result['case'];z=np.load(RUN/f'{name}_trace.npz');rr=np.load(RUN/f'{name}_reference.npz') if (RUN/f'{name}_reference.npz').exists() else None;dt=result['dt_s'];v=z['qdot'];a=np.diff(np.vstack((result['qdot_entry'],v)),axis=0)/dt;j=np.diff(np.vstack((result['acceleration_entry'],a)),axis=0)/dt
    rec={'trace_sha256_matches':sha(RUN/f'{name}_trace.npz')==result['trace_sha256'],'rows':len(v),'derivative_max_abs_error':{'acceleration':float(abs(a-z['command_acceleration']).max()),'jerk':float(abs(j-z['command_jerk']).max())},'calculated_margins':{},'reported_margin_errors':{}}
    for label,arr in [('velocity',v),('acceleration',a),('jerk',j)]:
        limits=np.asarray(result[f'{label}_limit_by_coordinate']);margins=limits-abs(arr);rec['calculated_margins'][label]=float(margins.min());rec['reported_margin_errors'][label]=float(margins.min()-result[f'{label}_margin_min']);rec[label+'_strict_violation_coordinate_ticks']=int((margins<0).sum());rec[label+'_max_limit_ratio']=float((abs(arr)/limits).max())
    pos=np.linalg.norm(z['pose_actual'][:,:3,3]-z['pose_desired'][:,:3,3],axis=1);rot_err=Rotation.from_matrix(np.einsum('nji,njk->nik',z['pose_desired'][:,:3,:3],z['pose_actual'][:,:3,:3])).magnitude()
    rec['pose_record_error']={'position':float(abs(pos-z['position_error_m']).max()),'rotation':float(abs(rot_err-z['rotation_error_rad']).max())};rec['pose_max_m_deg']=[float(pos.max()),float(np.degrees(rot_err.max()))]
    rec['pose_p95_m_deg']=[float(np.percentile(pos,95)),float(np.degrees(np.percentile(rot_err,95)))];rec['strict_pose_max_pass']=bool(pos.max()<=.002 and np.degrees(rot_err.max())<=2)
    rec['strict_mechanical_pass']=bool(not result['failures'] and np.all(z['q']>=model.lowerPositionLimit) and np.all(z['q']<=model.upperPositionLimit) and min(rec['calculated_margins'].values())>=0 and z['min_self_distance_m'].min()>=.01 and not np.any(z['solver_fault_latched']))
    rec['model_clock_max_error']=float(abs(z['native_model_t_mono']-(1000+np.arange(len(v))*dt)).max());rec['q_meas_chain_max_error']=float(abs(z['q_meas_before'][1:]-z['q'][:-1]).max());rec['q_exit_last_accepted_error']=float(abs(np.asarray(result['q_exit'])-z['q'][-1]).max())
    pose_diff=vel_diff=pose_log_diff=fk_error=fd_ang=fd_lin=0.;fd_count=0
    for n,t in enumerate(z['t_s']):
        expected,vel,s,i=independent_reference(t,result['speed_m_s']);saved_pose=np.eye(4)
        if rr is not None:
            saved_pose[:3,:3]=Rotation.from_euler('xyz',rr['pose_d'][n,3:]).as_matrix();saved_pose[:3,3]=rr['pose_d'][n,:3]
        if name=='original_2cm':vel[3:]=0
        if rr is not None:
            pose_diff=max(pose_diff,float(abs(expected-saved_pose).max()));vel_diff=max(vel_diff,float(abs(vel-rr['vel_ff'][n]).max()))
        pose_log_diff=max(pose_log_diff,float(abs(expected-z['pose_desired'][n]).max()))
        pin.forwardKinematics(model,data,z['q'][n]);pin.updateFramePlacements(model,data);fk_error=max(fk_error,float(abs(data.oMf[tcp_id].homogeneous-z['pose_actual'][n]).max()))
        # Separate finite difference of the reconstructed pose curve away from knots.
        if name!='original_2cm' and n%17==0 and t>1e-5 and s>arcs[i]+1e-6 and s<arcs[i+1]-1e-6:
            eps=1e-6;P0,_,_,_=independent_reference(t-eps,result['speed_m_s']);P1,_,_,_=independent_reference(t+eps,result['speed_m_s']);ww=Rotation.from_matrix(P1[:3,:3]@P0[:3,:3].T).as_rotvec()/(2*eps);vv=(P1[:3,3]-P0[:3,3])/(2*eps);fd_ang=max(fd_ang,float(abs(ww-vel[3:]).max()));fd_lin=max(fd_lin,float(abs(vv-vel[:3]).max()));fd_count+=1
    rec['reference_reconstruction_max_abs']={'pose_vs_saved_reference':pose_diff if rr is not None else None,'pose_vs_logged_desired':pose_log_diff,'vel_ff':vel_diff if rr is not None else None,'actual_pose_vs_independent_pinocchio_FK':fk_error,'angular_fd':fd_ang,'linear_fd':fd_lin,'fd_count':fd_count}
    if result['failures']:
        bad=result['failures'][0];k=bad['tick'];rec['failure']={'tick':k,'rail_send_m':float(z['q_send_command'][k,0]),'rail_measured_before_m':float(z['q_meas_before'][k,0]),'rail_trace_accepted_q_m':float(z['q'][k,0]),'q_bad_proposal_not_integrated':bool(np.array_equal(z['q'][k],z['q_meas_before'][k])),'native_q_send_agreement':bool(np.array_equal(np.asarray(bad['native_output']['q_cmd']),z['q_send_command'][k])),'exit_state_valid':result['exit_state_valid']}
    excess=abs(j)-np.asarray(result['jerk_limit_by_coordinate']);index=np.unravel_index(np.argmax(excess),excess.shape);k,c=map(int,index);rec['largest_jerk_excess']={'tick':k,'coordinate':c,'excess':float(excess[index]),'equivalent_velocity_box_excess':float(excess[index]*dt**2),'actual_box_excess_same_coordinate':float(max(z['qdot'][index]-z['box_hi'][index],z['box_lo'][index]-z['qdot'][index]))}
    out['runs'][name]=rec
duration_fast=d['runs'][0]['nominal_path_duration_s'];duration_slow=d['runs'][2]['nominal_path_duration_s'];out['retime']={'path_length_m':float(arcs[-1]),'nominal_speed_ratio':4.,'soft_start_ramp_both_s':.4,'settle_both_s':2.,'path_duration_fast_s':duration_fast,'path_duration_slow_s':duration_slow,'duration_ratio':duration_slow/duration_fast,'not_literal_global_t_over_4':True}
out['provenance_limit']=('Corrected-run input_manifest explicitly records native client, replay, configuration, binary and reference hashes; all file-backed entries were independently verified at sealing. URDF identity is linked separately to the earlier sealed robot manifest.' if clean_run else 'RESULTS records protocol/input/replay/reference and trace hashes, but does not directly record effective full controller config or native executable hash. Current config/URDF matches older sealed manifest; that link is reported separately, not invented as per-run capture.')
out['clock_integrity_limit']=('Corrected client writes explicit model_time_s before seq publication; separate actual-publication regression and fault injection audit apply. Logged t_mono alone is not a native-consumption transcript.' if clean_run else 'The logged native_model_t_mono and failure SHM input are read after the notify hook. They do not record the native consumed snapshot. Source audit found the hook writes model time after seq publication, allowing a wall/model timestamp race. Thus model_clock_max_error=0 does not certify clean native input timing; see CLOCK_PUBLICATION_AUDIT.txt.')
(HERE/('qp_clean_reference_independent.json' if clean_run else 'qp_reference_independent.json')).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
