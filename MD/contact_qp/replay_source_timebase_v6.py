import json
from pathlib import Path
import numpy as np
from scipy.signal import butter,tf2ss
from peirastic.contact_qp.runtime_source import SourceClock
from rm75_control.control.admittance_common.variable_step_filter import VariableLowpass1,VariableHighpass2
root=Path('/media/camp/EXT_DRIVE/ICRA_2027/icra 2027_contact/real_characterization/shadow_probe50/001/attempts')
report={'scope':'Recorded measurement timing and filter numerics only; no unexecuted active motion or force-performance claim.','nominal_period_s':.005,'max_interval_s':.015,'max_age_s':.015,'scans':[]}
for name in ['RH_Per_L_DtP','RH_Per_C_DtP']:
    path=root/name/'001/contact_qp.jsonl';clock=SourceClock(.005,timebase='variable_step_bilinear_v1',max_interval_s=.015,max_age_s=.015)
    w=2/.005*np.tan(np.pi*2.5*.005);A,B,C,D=tf2ss(*butter(2,w,btype='high',analog=True));I=np.eye(2)
    wl=2/.005*np.tan(np.pi*45*.005);low=high=None; intervals=[];ages=[];failures=[];held=0;n=0;lp_error=hp_error=0.
    for line in path.open():
        r=json.loads(line)
        if r.get('event')!='control_sample':continue
        n+=1
        try:s=clock.observe(r['wrench_source_id'],r['wrench_source_time_s'],r['wrench_source_wall_time_ns'],now_s=r['record_monotonic_s'])
        except ValueError as e:
            failures.append({'control_id':r['control_id'],'reason':str(e)});continue
        ages.append(s.age_s)
        if not s.fresh:held+=1;continue
        u=np.asarray(r['raw_control_wrench_tool'],float)
        if low is None:
            low=VariableLowpass1(45.,.005,u,u);high=VariableHighpass2(2.5,.005);high.reset(u[2]);yprev=u.copy();prev=u.copy();state=np.linalg.solve(A,-B[:,0]*u[2])
        else:intervals.append(s.source_dt_s)
        h=s.source_dt_s
        y=((2-wl*h)*yprev+wl*h*(prev+u))/(2+wl*h)
        state=np.linalg.solve(I-h*A/2,(I+h*A/2)@state+h*B[:,0]*(prev[2]+u[2])/2)
        hp=float((C@state+D[:,0]*u[2])[0])
        lp_error=max(lp_error,float(np.max(np.abs(low.update(u,h)-y))))
        hp_error=max(hp_error,abs(high.update(u[2],h)-hp))
        prev=u;yprev=y
    report['scans'].append({'scan':name,'input':str(path),'records':n,'fresh':clock.source_updates,'held':held,'rejected':failures,'source_interval_min_s':min(intervals),'source_interval_max_s':max(intervals),'maximum_record_age_s':max(ages),'lp_oracle_max_abs_error':lp_error,'hp_oracle_max_abs_error':hp_error})
output=Path('MD/contact_qp/source_timebase_v6_replay.json');output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
assert all(not r['rejected'] and r['hp_oracle_max_abs_error']<1e-12 for r in report['scans'])
