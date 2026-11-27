"""Thin CPU adapter for actual local DMP/KMP path generators, not a servo.

Coordinates are meters in the caller's explicitly named frame. The demonstration
is a user-defined local candidate shape; fitting/rollout use the existing source
implementations. Constant supplied TCP orientation is preserved. No robot calls.
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
import numpy as np

SOURCE = Path('/media/camp/EXT_DRIVE/Among_US')
for p in (SOURCE, SOURCE / 'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from DMP_playground.dmp_honest.dmp import fit_discrete_dmp
from DMP_playground.kmp_honest.kmp import fit_phase_kmp

def generate_candidates(start_xyz, goal_xyz, normal_xyz=(0., 0., 1.), n=41,
                        rotation=None, frame='railbase', duration_s=1.,
                        lateral_offsets_m=(-0.015, 0., 0.015), methods=('dmp', 'kmp')):
    """Return list of {T_tcp[W,4,4], xyz_m, rotations, normal_xyz, metrics}.

    Goal and start are public proposed actions, never hidden target truth.
    The tangent and normal define the candidate plane; a supplied rotation is
    constant and is used directly, not inferred from the path. Duration is only
    a declared evaluation horizon, not a dynamic feasibility guarantee.
    """
    start = np.asarray(start_xyz, dtype=float).reshape(3)
    goal = np.asarray(goal_xyz, dtype=float).reshape(3)
    if n < 4 or duration_s <= 0:
        raise ValueError('n >= 4 and duration_s > 0 are required')
    normal = np.asarray(normal_xyz, dtype=float).reshape(3)
    normal /= np.linalg.norm(normal)
    delta = goal - start
    tangent = delta - np.dot(delta, normal) * normal
    if np.linalg.norm(tangent) < 1e-10:
        tangent = np.eye(3)[np.argmin(np.abs(normal))]
        tangent -= np.dot(tangent, normal)*normal
    tangent /= np.linalg.norm(tangent)
    lateral = np.cross(normal, tangent)
    if rotation is None:
        rotation = np.column_stack((lateral, tangent, normal))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
    rotation = np.asarray(rotation, dtype=float).reshape(3,3)
    if np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1e-6 or np.linalg.det(rotation) < .999:
        raise ValueError('rotation must be a proper orthonormal matrix')
    s_train = np.linspace(0., 1., 41)
    s = np.linspace(0., 1., n)
    rows=[]
    for offset in lateral_offsets_m:
        # Caller-defined demonstration in an origin-free local frame. This is
        # deliberately not claimed as an existing full production scan policy.
        demo = s_train[:,None]*delta + (float(offset)*np.sin(np.pi*s_train))[:,None]*lateral
        for method in methods:
            t0=time.perf_counter()
            if method == 'dmp':
                model=fit_discrete_dmp(s_train, demo, n_basis=30)
                # Keep primitive integration accuracy independent of how many
                # waypoints a downstream exact kinematics query can afford.
                s_dense=np.linspace(0.,1.,max(161,n))
                dense,_=model.rollout(s_dense)
                xyz_local=np.column_stack([np.interp(s,s_dense,dense[:,j]) for j in range(3)])
                source='DMP_playground.dmp_honest.dmp.fit_discrete_dmp + DiscreteDMP.rollout'
            elif method == 'kmp':
                cov=np.repeat((1e-6*np.eye(3))[None],s_train.size,axis=0)
                model=fit_phase_kmp(s_train,demo,cov,lengthscale=20.,ridge=1.)
                xyz_local=np.asarray([model.predict(float(x))[0] for x in s])
                source='DMP_playground.kmp_honest.kmp.fit_phase_kmp + PhaseKMP.predict'
            else:
                raise ValueError(method)
            xyz=xyz_local+start
            elapsed=time.perf_counter()-t0
            T=np.repeat(np.eye(4)[None],n,axis=0)
            T[:,:3,:3]=rotation
            T[:,:3,3]=xyz
            steps=np.linalg.norm(np.diff(xyz,axis=0),axis=1)
            rows.append(dict(id=f'{method}_offset_{offset:+.3f}m', method=method,
                source=source, implementation_level='actual_primitive_only',frame=frame, units='m',
                integration_samples=max(161,n) if method=='dmp' else None,
                duration_s=float(duration_s), phase=s, xyz_m=xyz,T_tcp=T,
                rotations=T[:,:3,:3], normal_xyz=normal.copy(), offset_m=float(offset),
                metrics=dict(wall_s=elapsed,path_length_m=float(steps.sum()),
                    max_step_m=float(steps.max()),start_error_m=float(np.linalg.norm(xyz[0]-start)),
                    goal_error_m=float(np.linalg.norm(xyz[-1]-goal)),
                    max_speed_m_s=float(steps.max()*(n-1)/duration_s))))
    return rows

def serializable(rows):
    def convert(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,dict): return {k:convert(v) for k,v in x.items()}
        if isinstance(x,list): return [convert(v) for v in x]
        return x
    return convert(rows)

if __name__ == '__main__':
    import argparse,json
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path(__file__).with_name('candidate_paths.json'))
    a=p.parse_args()
    a.out.write_text(json.dumps(serializable(generate_candidates([0,0,0],[.06,0,0])),indent=2))
