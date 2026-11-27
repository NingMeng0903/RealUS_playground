"""Same spatial curve; analytically consistent start/stop time law."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qp_reference_followup"))
import run_reference_protocol as old


class SymmetricReference(old.Analytic6DReference):
    def __init__(self, *args, **kwargs):
        ramp = float(kwargs.get("ramp_s", .4))
        kwargs["soft_start"] = False
        super().__init__(*args, **kwargs)
        self.end_ramp_s = min(max(ramp, 0.), self._length/self.speed_m_s) if self.speed_m_s>0 else 0.
        self.cruise_s = max(0., self._length/self.speed_m_s-self.end_ramp_s) if self.speed_m_s>0 else 0.

    def duration_s(self):
        return max(0., 2*self.end_ramp_s+self.cruise_s)

    def time_law(self, t):
        t = max(0., float(t)-self._t_anchor)
        r, c, v = self.end_ramp_s, self.cruise_s, self.speed_m_s
        if t >= self.duration_s() or v<=0 or self._length<=1e-12:
            return self._length, 0.
        if r<=1e-12:
            return min(v*t,self._length), v
        if t < r:
            return v*(t/2-r/(2*np.pi)*np.sin(np.pi*t/r)), v/2*(1-np.cos(np.pi*t/r))
        if t < r+c:
            return v*r/2+v*(t-r), v
        u=t-r-c
        return v*r/2+v*c+v*(u/2+r/(2*np.pi)*np.sin(np.pi*u/r)), v/2*(1+np.cos(np.pi*u/r))

    def sample(self, t_s):
        s, speed = self.time_law(t_s)
        base_t = self._t_anchor + (s/self.speed_m_s if self.speed_m_s>0 else 0.)
        ref = super().sample(base_t)
        scale = speed/self.speed_m_s if self.speed_m_s>0 else 0.
        return replace(ref, vel_ff=np.asarray(ref.vel_ff)*scale, t_ref=float(t_s))


def validate(T):
    r=SymmetricReference(T[:,:3,3],rpy=Rotation.from_matrix(T[:,:3,:3]).as_euler('xyz'),speed_m_s=.02,ramp_s=.4)
    checks=[]
    for t in np.linspace(.0001,r.duration_s()-.0001,501):
        h=1e-6
        a,b,m=r.sample(t-h),r.sample(t+h),r.sample(t)
        # Exclude spatial knots: a piecewise polyline has two one-sided derivatives there.
        sa,_=r.time_law(t-h); sb,_=r.time_law(t+h)
        if np.any((r._s>sa)&(r._s<sb)):
            continue
        linear=(b.pose_d[:3]-a.pose_d[:3])/(2*h)
        R0=Rotation.from_euler('xyz',a.pose_d[3:]).as_matrix()
        R1=Rotation.from_euler('xyz',b.pose_d[3:]).as_matrix()
        angular=Rotation.from_matrix(R1@R0.T).as_rotvec()/(2*h)
        checks.append(np.max(np.abs(np.r_[linear,angular]-m.vel_ff)))
        p,angle,_=r._interp(r.time_law(t)[0])
        assert np.max(np.abs(p-m.pose_d[:3]))<1e-12
    assert max(checks)<1e-6
    assert np.array_equal(r.sample(0.).vel_ff,np.zeros(6))
    assert np.array_equal(r.sample(r.duration_s()+.1).vel_ff,np.zeros(6))
    return dict(samples=len(checks),max_twist_fd_error=max(checks),duration_s=r.duration_s())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__=='__main__':
    data=np.load(old.INPUT,allow_pickle=False)
    out=dict(protocol_sha256=sha(HERE/'PROTOCOL.txt'),code_sha256=sha(__file__),
        input_sha256=sha(old.INPUT),replay_sha256=sha(old.replay.__file__),checks={},runs=[])
    for path in (0,1):
        T,q=data[f'branch4_path{path}_T'],data[f'branch4_path{path}_q_seed']
        out['checks'][str(path)]=validate(T)
        for name,klass in [('start_only',old.Analytic6DReference),('both_ends',SymmetricReference)]:
            old.replay.WorldPolylineReference=klass
            bridge=old.replay.QPReplayBridge('native','urdf',native_clock='deterministic')
            raw=bridge.query_pose_path(T,q,speed_m_s=.02,settle_s=2.,return_trace=True,return_state=False,
                provenance=f'frozen two-ended time law path{path} {name}')
            trace=raw.pop('trace',[])
            old.summarize(raw,trace)
            case=f'path{path}_{name}'
            raw['case']=case
            p=HERE/f'{case}_trace.npz'
            np.savez_compressed(p,**{key:np.asarray([row[key] for row in trace]) for key in trace[0]})
            raw['trace_sha256']=sha(p)
            out['runs'].append(raw)
            (HERE/'RESULTS.json').write_text(json.dumps(old.clean_json(out),indent=2,allow_nan=False))
            print(case,'n',raw['n_steps'],'maxmm',1000*raw['full_position_max_m'],
                'maxdeg',raw['full_rotation_max_deg'],'pose',raw['strict_pose_max_gate'],
                'mechanical',raw['strict_mechanical_gate'],flush=True)
