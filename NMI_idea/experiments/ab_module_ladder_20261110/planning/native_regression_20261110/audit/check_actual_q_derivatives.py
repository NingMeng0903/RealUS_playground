"""Actual ideal-plant position increments, with full segment-boundary history."""
from pathlib import Path
import hashlib
import json
import numpy as np

here=Path(__file__).resolve().parent
folder=here.parent/'layout20_preserved_preferences'
data=json.loads((folder/'RESULTS.json').read_text())
rows=[]
for episode in data['episodes']:
    prior_v=np.zeros(8);prior_q=None;vm=am=float('inf');delta=0.;count=0
    for segment in episode['segments']:
        z=np.load(folder/segment['trace_file'])
        q=z['q_meas_before'];qn=z['q'];v=(qn-q)/.005
        if prior_q is not None:np.testing.assert_array_equal(q[0],prior_q)
        a=np.diff(np.vstack([prior_v,v]),axis=0)/.005
        limits=segment['result']['native']
        vm=min(vm,float(np.min(np.array(limits['velocity_limit_by_coordinate'])-abs(v))))
        am=min(am,float(np.min(np.array(limits['acceleration_limit_by_coordinate'])-abs(a))))
        delta=max(delta,float(np.max(abs(v-z['qdot']))))
        prior_v=v[-1];prior_q=qn[-1];count+=len(v)
    rows.append(dict(object=episode['object_index_for_audit_only'],original_context=episode['context'],
        samples=count,q_finite_difference_velocity_margin=vm,q_finite_difference_acceleration_margin=am,
        max_q_fd_velocity_vs_reported_qdot=delta))
out=dict(scope='Actual ideal measured-q increments, velocity and acceleration including segment boundaries; not derivative of only published qdot',
    result_sha256=hashlib.sha256((folder/'RESULTS.json').read_bytes()).hexdigest(),dt=.005,
    episodes=rows,min_velocity_margin=min(r['q_finite_difference_velocity_margin'] for r in rows),
    min_acceleration_margin=min(r['q_finite_difference_acceleration_margin'] for r in rows))
assert out['min_velocity_margin']>=0 and out['min_acceleration_margin']>=-1e-8
(here/'ACTUAL_Q_DERIVATIVE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
