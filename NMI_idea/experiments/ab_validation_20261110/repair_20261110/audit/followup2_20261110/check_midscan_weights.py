"""Independent saved-data audit; no native process and no experiment writes."""
from pathlib import Path
import hashlib, json
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
REPAIR=HERE.parents[1]
BASE=REPAIR/'qp_cross_track_fix'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def maxabs(a): return float(np.max(np.abs(a)))
inputs=np.load(REPAIR/'audit/phase2_native_subset_inputs.npz')
claimed=json.loads((BASE/'midscan_summary.json').read_text())
contract=json.loads((REPAIR/'dird/qp_native_urdf/manifest.json').read_text())['robot_contract']
model=pin.buildModelFromUrdf(contract['kinematics_urdf']);data=model.createData();fid=model.getFrameId('tcp')
out={'scope':'read-only independent FK/rotation/arc/derivative recomputation of saved ten cases; no native execution',
 'protocol_sha256':sha(BASE/'MIDSCAN_EVALUATION_PROTOCOL.txt'),'claimed_summary_sha256':sha(BASE/'midscan_summary.json'),
 'input_sha256':sha(REPAIR/'audit/phase2_native_subset_inputs.npz'),'runs':{},'paired_weight_checks':{}}
for folder,resultfile in [(BASE/'weight_ablation','RESULTS.json'),(BASE,'relaxed_execution_results.json')]:
    metadata=json.loads((folder/resultfile).read_text())
    assert out['input_sha256']==metadata['input_sha256']
    for r in metadata['runs']:
        name=r['case'];path=int(name[4]);trace=folder/r.get('trace_file',name+'_trace.npz')
        assert sha(trace)==r['trace_sha256']
        if folder.name=='weight_ablation':
            for filename,hashkey in [(name+'_native_config.txt','native_dump_sha256'),(name+'_reference.npz','reference_sha256'),(name+'_result.json','raw_result_sha256')]:
                assert sha(folder/filename)==r[hashkey],(name,hashkey)
        z=np.load(trace);T=inputs[f'branch4_path{path}_T'];vertices=T[:,:3,3]
        edges=np.diff(vertices,axis=0);lengths=np.linalg.norm(edges,axis=1);cum=np.r_[0.,np.cumsum(lengths)]
        positions=z['pose_desired'][:,:3,3];actual=z['pose_actual'][:,:3,3]
        # Separate segment loop, then choose nearest clamped line projection.
        dist=[];parameters=[]
        for origin,edge,length in zip(vertices[:-1],edges,lengths):
            u=np.clip((positions-origin)@edge/(length*length),0.,1.)
            dist.append(np.linalg.norm(positions-(origin+u[:,None]*edge),axis=1));parameters.append(u)
        dist=np.asarray(dist).T;parameters=np.asarray(parameters).T;segment=np.argmin(dist,axis=1)
        rowidx=np.arange(len(positions));fraction=(cum[segment]+parameters[rowidx,segment]*lengths[segment])/cum[-1]
        tangent=edges[segment]/lengths[segment,None]
        ep=positions-actual;along=np.sum(ep*tangent,axis=1)*1000
        pos=np.linalg.norm(ep,axis=1)*1000
        normal=np.linalg.norm(ep-along[:,None]/1000*tangent,axis=1)*1000
        angle=np.degrees((Rotation.from_matrix(z['pose_desired'][:,:3,:3]).inv()*Rotation.from_matrix(z['pose_actual'][:,:3,:3])).magnitude())
        times=z['reference_time_s'] if 'reference_time_s' in z.files else z['t_s']
        duration=r['nominal_path_duration_s'];moving=times<duration
        masks={'all_motion':np.ones(len(times),bool),'middle_phase':(times>=.8)&(times<=duration-.8),'arc20_80':(fraction>=.2)&(fraction<=.8)&moving}
        masks.update({f'arc_decile_{i}':(fraction>=i/10)&((fraction<(i+1)/10) if i<9 else (fraction<=1))&moving for i in range(10)})
        stats={};comparemax=0.
        for key,mask in masks.items():
            rec={'samples':int(mask.sum())}
            if mask.any():
                rec.update(phase_min_s=float(times[mask].min()),phase_max_s=float(times[mask].max()),arc_min_fraction=float(fraction[mask].min()),arc_max_fraction=float(fraction[mask].max()))
                for label,values in [('position',pos),('normal',normal),('along_abs',abs(along)),('angle',angle)]:
                    unit='deg' if label=='angle' else 'mm'
                    rec[f'{label}_p95_{unit}']=float(np.percentile(values[mask],95));rec[f'{label}_max_{unit}']=float(values[mask].max())
                rec['pose_gate']=bool(pos[mask].max()<=2 and angle[mask].max()<=2)
            for k,v in rec.items():
                expected=claimed[name][key][k]
                if isinstance(v,(bool,int)): assert expected==v,(name,key,k,expected,v)
                else: comparemax=max(comparemax,abs(v-expected));assert np.isclose(v,expected,rtol=1e-10,atol=1e-9),(name,key,k,expected,v)
            stats[key]=rec
        assert sum(stats[f'arc_decile_{i}']['samples'] for i in range(10))==int(moving.sum())
        fkerror=0.
        for q,pose in zip(z['q'],z['pose_actual']):
            pin.forwardKinematics(model,data,q);pin.updateFramePlacements(model,data)
            fkerror=max(fkerror,maxabs(data.oMf[fid].homogeneous-pose))
        dt=r['dt_s'];v=z['qdot'];a=np.diff(np.vstack((r['qdot_entry'],v)),axis=0)/dt
        jerk=np.diff(np.vstack((r['acceleration_entry'],a)),axis=0)/dt
        marg={label:float(np.min(np.asarray(r[key])-np.abs(values))) for label,key,values in [('velocity','velocity_limit_by_coordinate',v),('acceleration','acceleration_limit_by_coordinate',a),('jerk','jerk_limit_by_coordinate',jerk)]}
        for label in marg: assert abs(marg[label]-r[label+'_margin_min'])<1e-9
        worst=np.unravel_index(np.argmax(abs(jerk)-np.asarray(r['jerk_limit_by_coordinate'])),jerk.shape)
        peak=int(angle.argmax());over=np.flatnonzero(angle>2)
        rec={'trace_sha256':sha(trace),'samples':len(times),'nominal_duration_s':duration,
            'FK_max_abs':fkerror,'desired_point_polyline_max_distance_m':float(dist[rowidx,segment].max()),
            'recorded_position_max_difference_m':maxabs(pos/1000-z['position_error_m']),
            'recorded_rotation_max_difference_rad':maxabs(np.radians(angle)-z['rotation_error_rad']),
            'all_summary_max_abs_difference':comparemax,'all_reported_counts_and_gates_match':True,
            'command_acceleration_max_difference':maxabs(a-z['command_acceleration']),
            'command_jerk_max_difference':maxabs(jerk-z['command_jerk']),
            'minimum_command_margins':marg,'strict_command_dynamics_pass':all(x>=0 for x in marg.values()),
            'worst_jerk':{'tick':int(worst[0]),'coordinate':int(worst[1]),'time_s':float(z['t_s'][worst[0]]),'jerk':float(jerk[worst]),'box_lower':float(z['box_lo'][worst]),'box_upper':float(z['box_hi'][worst])},
            'orientation_peak':{'tick':peak,'time_s':float(z['t_s'][peak]),'phase_s':float(times[peak]),'arc_fraction':float(fraction[peak]),'degrees':float(angle[peak])},
            'orientation_above_2deg':{'count':len(over),'first_time_s':float(z['t_s'][over[0]]) if len(over) else None,'last_time_s':float(z['t_s'][over[-1]]) if len(over) else None,'contiguous_intervals_s':[[float(z['t_s'][g[0]]),float(z['t_s'][g[-1]])] for g in np.split(over,np.flatnonzero(np.diff(over)>1)+1) if len(g)]},
            'regional_statistics':stats}
        assert fkerror<1e-12
        out['runs'][name]=rec
        print(name,stats['middle_phase'],rec['orientation_peak'],marg,flush=True)
for path in (0,1):
    base=BASE/'weight_ablation';a=f'path{path}_baseline';b=f'path{path}_tolerance_unit'
    za=np.load(base/(a+'_trace.npz'));zb=np.load(base/(b+'_trace.npz'))
    ca=(base/(a+'_native_config.txt')).read_text().splitlines();cb=(base/(b+'_native_config.txt')).read_text().splitlines()
    differences=[{'baseline':x,'unit':y} for x,y in zip(ca,cb) if x!=y]
    assert len(ca)==len(cb) and len(differences)==1 and differences[0]['baseline'].startswith('qp.task_weight ')
    assert np.array_equal(za['t_s'],zb['t_s']) and np.array_equal(za['pose_desired'],zb['pose_desired'])
    wa=np.fromstring(differences[0]['baseline'].split(' ',1)[1],sep=' ');wb=np.fromstring(differences[0]['unit'].split(' ',1)[1],sep=' ')
    assert np.array_equal(wa,np.r_[np.full(3,100.),np.full(3,50.)]);assert np.array_equal(wb,np.r_[np.full(3,50*(np.deg2rad(2)/.002)**2),np.full(3,50.)])
    out['paired_weight_checks'][str(path)]={'reference_and_times_bit_identical':True,'native_dump_only_weight_difference':differences,'baseline_native_cfg_sha256':sha(base/(a+'_native_config.txt')),'unit_native_cfg_sha256':sha(base/(b+'_native_config.txt'))}
(HERE/'midscan_weights_independent.json').write_text(json.dumps(out,indent=2)+'\n')
print('ALL TEN RUNS AND ALL DECILES VERIFIED',flush=True)
