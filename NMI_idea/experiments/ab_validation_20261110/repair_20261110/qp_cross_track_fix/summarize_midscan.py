"""User-requested middle-scan summaries, with all deciles and transients retained."""
from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent
INPUT=HERE.parent/'audit'/'phase2_native_subset_inputs.npz'


def summarize(T,z,duration):
    desired=z['pose_desired'][:,:3,3]
    actual=z['pose_actual'][:,:3,3]
    p=T[:,:3,3];v=np.diff(p,axis=0);l=np.linalg.norm(v,axis=1);cum=np.r_[0,np.cumsum(l)]
    u=np.clip(((desired[:,None,:]-p[:-1])*v).sum(axis=2)/l**2,0,1)
    projected=p[:-1][None]+u[:,:,None]*v[None]
    idx=np.linalg.norm(desired[:,None]-projected,axis=2).argmin(axis=1)
    rows=np.arange(len(desired))
    fraction=(cum[idx]+u[rows,idx]*l[idx])/cum[-1]
    tangent=v[idx]/l[idx,None]
    err=desired-actual
    pos=np.linalg.norm(err,axis=1)*1000
    along=(err*tangent).sum(axis=1)*1000
    normal=np.sqrt(np.maximum(0,pos**2-along**2))
    angle=np.degrees(z['rotation_error_rad'])
    time=z['reference_time_s'] if 'reference_time_s' in z.files else z['t_s']
    masks=dict(all_motion=np.ones(len(pos),dtype=bool),
        middle_phase=(time>=.8)&(time<=duration-.8),
        arc20_80=(fraction>=.2)&(fraction<=.8)&(time<duration))
    masks.update({f'arc_decile_{i}':(fraction>=i/10)&(fraction<(i+1)/10 if i<9 else fraction<=1)&(time<duration)
        for i in range(10)})
    out={}
    for key,mask in masks.items():
        if not mask.any():
            out[key]=dict(samples=0)
            continue
        out[key]=dict(samples=int(mask.sum()),phase_min_s=float(time[mask].min()),phase_max_s=float(time[mask].max()),
            arc_min_fraction=float(fraction[mask].min()),arc_max_fraction=float(fraction[mask].max()),
            position_p95_mm=float(np.percentile(pos[mask],95)),position_max_mm=float(pos[mask].max()),
            normal_p95_mm=float(np.percentile(normal[mask],95)),normal_max_mm=float(normal[mask].max()),
            along_abs_p95_mm=float(np.percentile(np.abs(along[mask]),95)),along_abs_max_mm=float(np.abs(along[mask]).max()),
            angle_p95_deg=float(np.percentile(angle[mask],95)),angle_max_deg=float(angle[mask].max()),
            pose_gate=bool(pos[mask].max()<=2 and angle[mask].max()<=2))
    i=int(angle.argmax())
    out['orientation_peak_location']=dict(tick=i,phase_s=float(time[i]),arc_fraction=float(fraction[i]),degrees=float(angle[i]))
    return out


if __name__=='__main__':
    data=np.load(INPUT,allow_pickle=False)
    out={}
    for root,file in [(HERE/'weight_ablation','RESULTS.json'),(HERE,'relaxed_execution_results.json')]:
        if not (root/file).exists():
            continue
        for run in json.loads((root/file).read_text())['runs']:
            name=run['case'];path=int(name[4])
            trace=root/run.get('trace_file',f'{name}_trace.npz')
            with np.load(trace,allow_pickle=False) as z:
                out[name]=summarize(data[f'branch4_path{path}_T'],z,run['nominal_path_duration_s'])
            print(name, 'middle',out[name]['middle_phase'],flush=True)
    (HERE/'midscan_summary.json').write_text(json.dumps(out,indent=2))
