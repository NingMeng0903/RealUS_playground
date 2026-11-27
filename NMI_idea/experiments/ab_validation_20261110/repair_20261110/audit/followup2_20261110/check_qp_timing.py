"""Read-only independent checks of the frozen timing experiments; no native launch."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
REPAIR = HERE.parents[1]
SOURCE = Path('/media/camp/EXT_DRIVE/RealUS_playground')
sys.path[:0] = [str(REPAIR/'qp_terminal_retime'), str(SOURCE/'rm75_control')]
from run_terminal_retime import SymmetricReference
from rm75_control.control.joint_admittance_8dof.collision_model import CollisionModel

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def normmax(a):
    return float(np.max(np.abs(a)))

def time_law(t, length, both):
    t=max(0.,float(t)); v=.02; r=.4
    if both:
        r=min(r,length/v); c=max(0.,length/v-r); duration=2*r+c
        if t>=duration:
            return length,0.,duration
        if t<r:
            return v*(t/2-r/(2*np.pi)*np.sin(np.pi*t/r)),v/2*(1-np.cos(np.pi*t/r)),duration
        if t<r+c:
            return v*r/2+v*(t-r),v,duration
        u=t-r-c
        return v*r/2+v*c+v*(u/2+r/(2*np.pi)*np.sin(np.pi*u/r)),v/2*(1+np.cos(np.pi*u/r)),duration
    duration=length/v+r/2
    if t<r:
        u=t/r; s=v*r*(u**6-3*u**5+2.5*u**4); speed=v*(6*u**5-15*u**4+10*u**3)
    else:
        s=v*(t-r/2); speed=v
    return min(length,s),speed if s<length else 0.,duration

def curve(T):
    points=T[:,:3,3]; rots=Rotation.from_matrix(T[:,:3,:3])
    arcs=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    return points,rots,arcs

def evaluate(T,t,both=False,rate=1.):
    points,rots,arcs=curve(T); s,speed,duration=time_law(t,arcs[-1],both)
    i=int(np.clip(np.searchsorted(arcs,s,side='right')-1,0,len(arcs)-2)); span=arcs[i+1]-arcs[i]
    u=float(np.clip((s-arcs[i])/span,0,1)); P=np.eye(4)
    P[:3,3]=(1-u)*points[i]+u*points[i+1]
    P[:3,:3]=Slerp([0,1],rots[i:i+2])([u])[0].as_matrix()
    vel=np.r_[(points[i+1]-points[i])/span,rots[i].apply((rots[i].inv()*rots[i+1]).as_rotvec())/span]*speed*rate
    return P,vel,s,duration

inputs=np.load(REPAIR/'audit/phase2_native_subset_inputs.npz')
old_manifest=json.loads((REPAIR/'dird/qp_native_urdf/manifest.json').read_text())
contract=old_manifest['robot_contract']
model=pin.buildModelFromUrdf(contract['kinematics_urdf']); data=model.createData(); fid=model.getFrameId('tcp')
collision=CollisionModel(model,collision_urdf=contract['collision_urdf'],pair_config=contract['collision_pairs'])
out={'scope':'read-only saved traces; no QP/native execution, no production edits','runs':{},'time_law_contract':{},'fixed_baseline_cross_harness':{},'input_sha256':sha(REPAIR/'audit/phase2_native_subset_inputs.npz'),'source_hashes':{}}

for folder,runner in [('qp_governor_followup','run_governor.py'),('qp_terminal_retime','run_terminal_retime.py')]:
    base=REPAIR/folder; results=json.loads((base/'RESULTS.json').read_text())
    hashes={'protocol':sha(base/'PROTOCOL.txt')==results['protocol_sha256'],'runner':sha(base/runner)==results['code_sha256'],'input':out['input_sha256']==results['input_sha256']}
    assert all(hashes.values())
    out['source_hashes'][folder]={str(base/p):sha(base/p) for p in ['PROTOCOL.txt',runner,'RESULTS.json']}
    for row in results['runs']:
        name=row['case']; path=int(name[4]); T=inputs[f'branch4_path{path}_T']; q0=inputs[f'branch4_path{path}_q_seed']
        z=np.load(base/(name+'_trace.npz')); gov=folder=='qp_governor_followup'; both='both_ends' in name
        dt=.005; v=z['qdot']; q=z['q']; n=len(q); times=z['t_s']; qbefore=z['q_before' if gov else 'q_meas_before']; qcmd=z['command_q' if gov else 'q_send_command']
        a=np.diff(np.vstack((np.zeros(8),v)),axis=0)/dt; j=np.diff(np.vstack((np.zeros(8),a)),axis=0)/dt
        limits=[np.asarray(row[key]) for key in (['velocity_limit','acceleration_limit','jerk_limit'] if gov else ['velocity_limit_by_coordinate','acceleration_limit_by_coordinate','jerk_limit_by_coordinate'])]
        margins=[np.min(limit-np.abs(value),axis=0) for limit,value in zip(limits,[v,a,j])]
        reported=[np.asarray(row[key]) for key in ['velocity_margin_by_coordinate','acceleration_margin_by_coordinate','jerk_margin_by_coordinate']] if gov else [np.asarray(row[key]) for key in ['velocity_margin_min','acceleration_margin_min','jerk_margin_min']]
        qmargin=np.minimum(qcmd-model.lowerPositionLimit,model.upperPositionLimit-qcmd)
        pos=np.linalg.norm(z['pose_actual'][:,:3,3]-z['pose_desired'][:,:3,3],axis=1)
        rot=Rotation.from_matrix(np.einsum('nji,njk->nik',z['pose_desired'][:,:3,:3],z['pose_actual'][:,:3,:3])).magnitude()
        pose_error=fk_error=ff_error=0.
        for k,t in enumerate(times):
            ref_t=z['reference_time_s'][k] if gov else t
            expected,vel,_,duration=evaluate(T,ref_t,both,z['rate'][k] if gov else 1.)
            pose_error=max(pose_error,normmax(expected-z['pose_desired'][k]))
            if gov:
                ff_error=max(ff_error,normmax(vel-z['vel_ff'][k]))
            pin.forwardKinematics(model,data,q[k]);pin.updateFramePlacements(model,data)
            fk_error=max(fk_error,normmax(data.oMf[fid].homogeneous-z['pose_actual'][k]))
        indices=np.unique(np.r_[np.linspace(0,n-1,33,dtype=int),np.argmin(z['min_self_distance_m']),np.argmax(pos),np.unravel_index(np.argmax(np.abs(j)),j.shape)[0]])
        collision_error=0.
        for k in indices:
            collision.update(q[k]); value=min(x.min_distance for x in collision.geom_data.distanceResults)
            collision_error=max(collision_error,abs(value-z['min_self_distance_m'][k]))
        rec={'trace_sha256_matches':sha(base/(name+'_trace.npz'))==row['trace_sha256'],'rows':n,'reported_simulated_duration_s':float(times[-1]),'integration_count_times_dt_s':n*dt,'recorded_wall_compute_s':row['compute_s' if gov else 'wall_seconds'],'maximum_position_mm':float(pos.max()*1000),'maximum_orientation_deg':float(np.degrees(rot.max())),'position_error_record_max_abs':normmax(pos-z['position_error_m']),'orientation_error_record_max_abs':normmax(rot-z['rotation_error_rad']),'desired_pose_independent_max_abs':pose_error,'FK_independent_max_abs':fk_error,'governor_chain_rule_FF_max_abs':ff_error if gov else None,'command_acceleration_record_max_abs':normmax(a-z['command_acceleration']),'command_jerk_record_max_abs':normmax(j-z['command_jerk']),'minimum_command_margins':dict(zip(['velocity','acceleration','jerk'],[float(x.min()) for x in margins])),'reported_margin_max_abs':max(normmax(x-y) if gov else abs(float(x.min())-float(y)) for x,y in zip(margins,reported)),'minimum_q_margin_by_coordinate':qmargin.min(axis=0).tolist(),'collision_subset':{'sample_count':len(indices),'pair_count':len(collision.geom_model.collisionPairs),'max_difference_m':collision_error,'reported_minimum_m':float(z['min_self_distance_m'].min())},'q_feedback_chain_max_abs':normmax(qbefore[1:]-q[:-1]),'initial_q_error':normmax(qbefore[0]-q0),'proposal_equals_integrated_q_max_abs':normmax(qcmd-q),'logged_clock_max_abs':normmax(z['model_timestamp' if gov else 'native_model_t_mono']-(1000+np.arange(n)*dt)),'solver_fault_count':int(z['solver_fault_latched'].sum()),'paused_count':int(z['task_paused'].sum())}
        rec['recomputed_pose_gate']=bool(pos.max()<=.002 and np.degrees(rot.max())<=2.)
        rec['recomputed_strict_dynamic_gate']=bool(qmargin.min()>=0 and min(x.min() for x in margins)>=0 and z['min_self_distance_m'].min()>=.01 and not z['solver_fault_latched'].any())
        rec['reported_gates_match']=rec['recomputed_pose_gate']==row['pose_gate' if gov else 'strict_pose_max_gate'] and rec['recomputed_strict_dynamic_gate']==row['strict_mechanical_gate']
        implied_v=np.diff(np.vstack((q0,q)),axis=0)/dt
        rec['position_fd_vs_qdot_by_coordinate_max_abs']=np.max(np.abs(implied_v-v),axis=0).tolist()
        k,c=map(int,np.unravel_index(np.argmax(np.abs(j)-limits[2]),j.shape))
        rec['worst_jerk']={'tick':k,'coordinate':c,'command_jerk':float(j[k,c]),'limit':float(limits[2][c]),'excess':float(abs(j[k,c])-limits[2][c])}
        if not gov:
            rec['worst_jerk'].update(velocity_box_lower=float(z['box_lo'][k,c]),velocity_box_upper=float(z['box_hi'][k,c]),actual_box_excess=float(max(z['qdot'][k,c]-z['box_hi'][k,c],z['box_lo'][k,c]-z['qdot'][k,c])))
        k=int(np.argmax(pos)); rec['maximum_error_location']={'tick':k,'time_s':float(times[k]),'reference_phase_s':float(z['reference_time_s'][k]) if gov else float(times[k]),'path_duration_s':duration}
        if gov:
            target=np.minimum(np.clip((.0015-pos)/.001,0,1),np.clip((np.deg2rad(1.5)-rot)/np.deg2rad(1),0,1))
            rate=1.; phase=0.; rates=[]; phases=[]
            for k in range(n):
                if phase>=duration: rate=0.
                elif row['governed']:
                    want=1. if k==0 else target[k-1]
                    rate=float(np.clip(rate+.1*(want-rate),max(0.,rate-.02),min(1.,rate+.02)))
                else:rate=1.
                phases.append(phase);rates.append(rate);phase=min(duration,phase+rate*dt)
            rec['governor_public_rule']={'target_rate_max_abs':normmax(target-z['target_rate']),'rate_recurrence_max_abs':normmax(np.asarray(rates)-z['rate']),'phase_recurrence_max_abs':normmax(np.asarray(phases)-z['reference_time_s']),'maximum_rate_slew_per_s':float(np.max(np.abs(np.diff(z['rate'])))/dt),'rate_mean_including_settle':float(z['rate'].mean()),'path_reached_at_s':row['reached_at_s'],'real_endpoint_hold_s':float(times[-1]-row['reached_at_s'])}
        assert rec['trace_sha256_matches'] and rec['reported_gates_match']
        assert pose_error<1e-11 and fk_error<1e-12 and collision_error<1e-12
        assert rec['reported_margin_max_abs']==0 and rec['command_acceleration_record_max_abs']==0 and rec['command_jerk_record_max_abs']==0
        out['runs'][folder+'/'+name]=rec

for path in [0,1]:
    T=inputs[f'branch4_path{path}_T']; points,rots,arcs=curve(T)
    obj=SymmetricReference(points,rpy=rots.as_euler('xyz'),speed_m_s=.02,ramp_s=.4)
    errors=[]; fd=[]; geometry=[]
    for t in np.linspace(1e-4,obj.duration_s()-1e-4,301):
        P,vel,s,duration=evaluate(T,t,True); sample=obj.sample(t)
        Q=np.eye(4);Q[:3,3]=sample.pose_d[:3];Q[:3,:3]=Rotation.from_euler('xyz',sample.pose_d[3:]).as_matrix()
        errors.append(normmax(sample.vel_ff-vel));geometry.append(normmax(P-Q))
        h=1e-6;Pm,_,sm,_=evaluate(T,t-h,True);Pp,_,sp,_=evaluate(T,t+h,True)
        if np.any((arcs>sm)&(arcs<sp)):continue
        measured=np.r_[(Pp[:3,3]-Pm[:3,3])/(2*h),Rotation.from_matrix(Pp[:3,:3]@Pm[:3,:3].T).as_rotvec()/(2*h)]
        fd.append(normmax(measured-vel))
    a=np.load(REPAIR/'qp_governor_followup'/f'path{path}_fixed_trace.npz');b=np.load(REPAIR/'qp_terminal_retime'/f'path{path}_start_only_trace.npz')
    diffs={}
    for key in ['pose_desired','q','qdot','pose_actual']:
        d=np.max(np.abs(a[key]-b[key]),axis=tuple(range(1,a[key].ndim))); nonzero=np.flatnonzero(d)
        diffs[key]={'max_abs':float(d.max()),'first_nonzero_tick':int(nonzero[0]) if len(nonzero) else None,'first_nonzero_max_abs':float(d[nonzero[0]]) if len(nonzero) else 0.}
    out['fixed_baseline_cross_harness'][str(path)]=diffs
    out['time_law_contract'][str(path)]={'actual_class_vs_independent_twist_max_abs':max(errors),'actual_class_vs_independent_pose_max_abs':max(geometry),'independent_pose_fd_vs_twist_max_abs':max(fd),'fd_samples':len(fd),'start_twist_exactly_zero':bool(np.array_equal(obj.sample(0.).vel_ff,np.zeros(6))),'end_twist_exactly_zero':bool(np.array_equal(obj.sample(obj.duration_s()).vel_ff,np.zeros(6))),'duration_s':obj.duration_s(),'extra_nominal_duration_vs_start_only_s':obj.duration_s()-time_law(0.,arcs[-1],False)[2],'start_speed_at_0p1s_old_new_m_s':[time_law(.1,arcs[-1],False)[1],time_law(.1,arcs[-1],True)[1]],'scope':'same polyline/Slerp at equal arc; changes both start and end timing; interior spatial knots remain nonsmooth'}

out['provenance_limit']='Governor captures client/replay/native binary hashes; terminal RESULTS captures runner/replay/input but not direct client/config/binary hashes. Current artifacts can be linked to prior manifest; this does not invent per-run provenance.'
out['timing_scope']='Fixed-dt ideal integration elapsed time differs from computational wall time; governor compute excludes controller/assets setup, terminal wall time includes construction. Cross-harness wall-time overhead cannot be directly subtracted.'
(HERE/'qp_timing_independent.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
