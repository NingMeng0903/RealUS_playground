"""Independent algebra/source diagnosis of the saved clean 2 cm/s 6D trace."""
from pathlib import Path
import hashlib,json
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation,Slerp

HERE=Path(__file__).resolve().parent; REPAIR=HERE.parents[1]
SRC=Path('/media/camp/EXT_DRIVE/RealUS_playground')
z=np.load(REPAIR/'qp_clock_repair/6d_2cm_trace.npz')
f=np.load(REPAIR/'audit/phase2_native_subset_inputs.npz');T=f['branch4_path0_T']
P=T[:,:3,3];rots=Rotation.from_matrix(T[:,:3,:3]);arcs=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(P,axis=0),axis=1))]
model=pin.buildModelFromUrdf(str(SRC/'rm75_control/rm75_control/assets/robots/rm75_6f_8dof/RM75-6F-8dof.urdf'));data=model.createData();fid=model.getFrameId('tcp')
dt=.005;n=len(z['q']);Jall=[];preposes=[];ff=[];tangents=[]
for k,t in enumerate(z['t_s']):
    q=z['q_meas_before'][k];pin.computeJointJacobians(model,data,q);pin.updateFramePlacements(model,data)
    J=pin.getFrameJacobian(model,data,fid,pin.LOCAL_WORLD_ALIGNED);Jall.append(J.copy());preposes.append(data.oMf[fid].homogeneous.copy())
    if t<.4:
        u=t/.4;tau=.4*(u**6-3*u**5+2.5*u**4);td=6*u**5-15*u**4+10*u**3
    else:tau=t-.2;td=1.
    s=min(arcs[-1],.02*tau);i=int(np.clip(np.searchsorted(arcs,s,side='right')-1,0,len(arcs)-2));span=arcs[i+1]-arcs[i]
    tangent=(P[i+1]-P[i])/span;tangents.append(tangent)
    omega=rots[i].apply((rots[i].inv()*rots[i+1]).as_rotvec())/span
    ff.append(np.r_[tangent,omega]*(.02*td if s<arcs[-1] else 0.))
Jall=np.asarray(Jall);preposes=np.asarray(preposes);ff=np.asarray(ff);tangents=np.asarray(tangents)
rot_error=Rotation.from_matrix(np.einsum('nij,nkj->nik',z['pose_desired'][:,:3,:3],preposes[:,:3,:3])).as_rotvec()
pos_error=z['pose_desired'][:,:3,3]-preposes[:,:3,3]
expected=ff+np.c_[12*pos_error,2*rot_error]
roundtrip=np.empty_like(expected)
for k in range(n):
    R=preposes[k,:3,:3];roundtrip[k]=np.r_[R@z['twist_requested'][k,:3],R@z['twist_requested'][k,3:]]
Jtask=Jall.copy();Jtask[:,:,0]=0
rail_previous=np.r_[0.,z['qdot'][:-1,0]]
affine=Jall[:,:,0]*rail_previous[:,None]
predicted=np.einsum('nij,nj->ni',Jtask,z['qdot'])+affine
full_current=np.einsum('nij,nj->ni',Jall,z['qdot'])
lower=np.sum(Jtask*np.where(Jtask>=0,z['box_lo'][:,None,:],z['box_hi'][:,None,:]),axis=2)+affine
upper=np.sum(Jtask*np.where(Jtask>=0,z['box_hi'][:,None,:],z['box_lo'][:,None,:]),axis=2)+affine
request=z['v_cmd_received'];gap=request-z['v_cmd_feasible']
outside=np.maximum(np.maximum(lower-request,request-upper),0.)
e=z['pose_desired'][:,:3,3]-z['pose_actual'][:,:3,3]
along=np.einsum('ni,ni->n',e,tangents);cross=e-along[:,None]*tangents
actual_fd=(z['pose_actual'][:,:3,3]-preposes[:,:3,3])/dt
position_fd_q=(z['q']-z['q_meas_before'])/dt
fd_model=np.einsum('nij,nj->ni',Jall,position_fd_q)
lead=z['q_cmd_before']-z['q_meas_before']
lead_cap=np.sqrt((3*dt)**2+2*3*np.deg2rad(6))-3*dt
nearcap=np.minimum(abs(z['qdot'][:,4]-lead_cap),abs(z['qdot'][:,4]+lead_cap))<2e-6
thresholds={}
for tol in [1e-6,1e-4,.001,.01]:
    thresholds[str(tol)]={'linear_component_request_outside_recorded_box_ticks':int((outside[:,:3].max(axis=1)>tol).sum()),'angular_component_request_outside_recorded_box_ticks':int((outside[:,3:].max(axis=1)>tol).sum()),'primary_linear_residual_ticks':int((np.linalg.norm(gap[:,:3],axis=1)>tol).sum())}
important=[0,50,100,300,500,650,700,724,750,779,801,850,900,1179]
rows=[]
for k in important:
    direction=e[k]/max(np.linalg.norm(e[k]),1e-30)
    proj=direction@Jtask[k,:3,:];offset=float(direction@affine[k,:3])
    achievable=[float(np.sum(proj*np.where(proj>=0,z['box_lo'][k],z['box_hi'][k]))+offset),float(np.sum(proj*np.where(proj>=0,z['box_hi'][k],z['box_lo'][k]))+offset)]
    rows.append({'tick':k,'t_s':float(z['t_s'][k]),'post_update_position_error_m':e[k].tolist(),'position_error_norm_m':float(np.linalg.norm(e[k])),'along_last_or_current_segment_m':float(along[k]),'cross_last_or_current_segment_m':float(np.linalg.norm(cross[k])),'command_received_base':request[k].tolist(),'feasible_base':z['v_cmd_feasible'][k].tolist(),'primary_gap_base':gap[k].tolist(),'axis_velocity_lower':lower[k].tolist(),'axis_velocity_upper':upper[k].tolist(),'axis_request_outside_box':outside[k].tolist(),'correction_direction_command_m_s':float(direction@request[k,:3]),'correction_direction_feasible_m_s':float(direction@z['v_cmd_feasible'][k,:3]),'correction_direction_box_interval_m_s':achievable,'arm_4_q_rad':float(z['q_meas_before'][k,4]),'arm_4_velocity_rad_s':float(z['qdot'][k,4]),'arm_4_box':[float(z['box_lo'][k,4]),float(z['box_hi'][k,4])],'arm_4_previous_two_velocities':[float(z['qdot_prev_used'][k,4]),float(z['qdot_prev2_used'][k,4])],'narrow_box_coordinates':np.flatnonzero(z['box_hi'][k]-z['box_lo'][k]<1e-7).tolist(),'box_active_coordinates':np.flatnonzero(np.minimum(z['qdot'][k]-z['box_lo'][k],z['box_hi'][k]-z['qdot'][k])<1e-6).tolist()})
source_paths=[SRC/'rm75_control/native/wbc_rt/src/inner.cpp',SRC/'rm75_control/native/wbc_rt/include/wbc_rt/types.hpp',SRC/'rm75_control/native/wbc_rt/src/kinematics.cpp',SRC/'rm75_control/rm75_control/control/joint_admittance_8dof/loop.py',SRC/'rm75_control/rm75_control/control/joint_admittance_8dof/wbc_rt/client.py',SRC/'peirastic/configs/controller.yaml']
snap=HERE/'source_at_diagnosis';snap.mkdir(exist_ok=True)
hashes={}
for p in source_paths:
    raw=p.read_bytes();name=('native_' if '/native/' in str(p) else 'python_' if p.suffix=='.py' else '')+p.name
    (snap/name).write_bytes(raw);hashes[str(p)]=hashlib.sha256(raw).hexdigest()
result={'scope':'read-only frozen clean trace; no native or counterfactual control run; per-axis box certificates are necessary conditions, not full LP solves','trace_sha256':hashlib.sha256((REPAIR/'qp_clock_repair/6d_2cm_trace.npz').read_bytes()).hexdigest(),'source_snapshots_sha256':hashes,'frame_and_execution':{'outer_FF_plus_12p_2R_vs_received_max_abs':float(abs(expected-request).max()),'tool_twist_rotated_back_vs_received_max_abs':float(abs(roundtrip-request).max()),'J_arm_plus_previous_rail_vs_feasible_max_abs':float(abs(predicted-z['v_cmd_feasible']).max()),'J_arm_plus_previous_rail_vs_estimated_max_abs':float(abs(predicted-z['v_tcp_estimated']).max()),'feasible_vs_estimated_max_abs':float(abs(z['v_cmd_feasible']-z['v_tcp_estimated']).max()),'linear_feasible_vs_actual_FD_max_norm_m_s':float(np.linalg.norm(z['v_cmd_feasible'][:,:3]-actual_fd,axis=1).max()),'linear_FD_vs_J_position_FD_max_norm_m_s':float(np.linalg.norm(actual_fd-fd_model[:,:3],axis=1).max()),'rail_column_max_abs_difference_from_base_y':float(abs(Jall[:,:,0]-np.array([0,1,0,0,0,0])).max()),'position_FD_vs_command_qdot_max_by_coordinate':np.max(abs(position_fd_q-z['qdot']),axis=0).tolist(),'arm_lead_max_abs_rad':float(abs(lead[:,1:]).max()),'logged_model_clock_max_abs':float(abs(z['native_model_t_mono']-(1000+np.arange(n)*dt)).max())},'lead_speed_limit':{'distance_rad':float(np.deg2rad(6)),'acceleration_rad_s2':3.,'reaction_s':dt,'calculated_zero_lead_speed_cap_rad_s':float(lead_cap),'declared_nominal_arm_speed_limit_rad_s':2.512,'joint4_near_lead_cap_ticks':int(nearcap.sum()),'joint_limited_telemetry_ticks':int(z['joint_limited'].sum()),'note':'joint_limited only compares to nominal v_max, not lead/dynamic/branch boxes'},'residual_threshold_counts':thresholds,'position_error_max_m':float(np.linalg.norm(e,axis=1).max()),'max_along_current_segment_m':float(abs(along).max()),'max_cross_current_segment_m':float(np.linalg.norm(cross,axis=1).max()),'primary_linear_residual_max_norm_m_s':float(np.linalg.norm(gap[:,:3],axis=1).max()),'primary_angular_residual_max_norm_rad_s':float(np.linalg.norm(gap[:,3:],axis=1).max()),'sampled_min_self_distance_m':float(z['min_self_distance_m'].min()),'collision_activation_m_from_config':.04,'key_ticks':rows,'primary_secondary_scope':'QP2 source locks achieved J_task qdot from QP1; trace does not export QP1 raw qdot or QP2 status, so direct per-tick primary-lock optimality cannot be reconstructed. Near-equal feasible and estimated velocity rules out a material downstream velocity rewrite in these records, but not posture history changing later geometry/dynamic boxes.','recommendation_scope':'identify and test explicit command-lead/dynamic feasibility at requested 6D timing; do not infer safe cap removal or independently feasible whole path from these local certificates.'}
(HERE/'cross_track_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
