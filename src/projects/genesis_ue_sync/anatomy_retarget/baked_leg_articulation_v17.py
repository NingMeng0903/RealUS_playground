"""Fixed, local pose response compiled from a subject-independent angle grid.

The two flexion axes use bilinear interpolation. The other rotation axes
add their independently sampled response. This separable approximation is
explicitly recorded; mixed-axis accuracy must be checked on recorded motion.
There is no optimizer, nearest-point query or rebinding in ``evaluate``.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from .lower_chain_pose_fit_v17 import fit_leg_pose_v17, head_centers_local_v17


HIP_FLEXION_DEG=np.array([-120.,-90.,-60.,-30.,0.,30.])
KNEE_FLEXION_DEG=np.array([-15.,0.,30.,60.,90.,120.])
OFF_AXIS_DEG=np.full(7,90.)
OFF_KNOT_FRACTIONS=np.array([-1.,-.5,-1/6,0.,1/6,.5,1.])


@dataclass
class BakedLegArticulationV17:
    hip_axis: np.ndarray
    knee_axis: np.ndarray
    off_axis: np.ndarray
    values: np.ndarray
    head_centers_local: np.ndarray
    report: dict
    off_knots: np.ndarray | None = None

    def __post_init__(self):
        for name in ('hip_axis','knee_axis','off_axis','values','head_centers_local'):
            value=np.asarray(getattr(self,name),dtype=np.float64)
            if not np.isfinite(value).all():raise ValueError('nonfinite articulation '+name)
            setattr(self,name,value.copy())
        variants=9
        if self.off_knots is not None:
            self.off_knots=np.asarray(self.off_knots,dtype=np.float64).copy()
            if (self.off_knots.ndim!=1 or not np.isfinite(self.off_knots).all() or
                    np.any(np.diff(self.off_knots)<=0) or
                    not np.array_equal(self.off_knots[[0,-1]],[-1.,1.]) or
                    np.count_nonzero(self.off_knots==0)!=1):
                raise ValueError('invalid off-axis knots')
            variants=1+len(self.off_axis)*(len(self.off_knots)-1)
            self.off_knots.setflags(write=False)
        if (self.values.shape!=(len(self.hip_axis),len(self.knee_axis),variants,2,2,3) or
                self.off_axis.shape not in ((4,),(7,)) or self.head_centers_local.shape!=(2,3) or
                np.any(np.diff(self.hip_axis)<=0) or np.any(np.diff(self.knee_axis)<=0) or
                np.any(self.off_axis<=0)):
            raise ValueError('invalid articulation grid')
        for name in ('hip_axis','knee_axis','off_axis','values','head_centers_local'):
            getattr(self,name).setflags(write=False)

    @staticmethod
    def _interval(axis,value):
        if value<axis[0]-1e-6 or value>axis[-1]+1e-6:
            raise ValueError('pose outside baked leg flexion support; angle was not clipped')
        i=min(max(int(np.searchsorted(axis,value,side='right'))-1,0),len(axis)-2)
        return i,(value-axis[i])/(axis[i+1]-axis[i])

    def evaluate(self,pose55):
        pose=np.asarray(pose55,dtype=np.float64)
        if pose.shape!=(55,3) or not np.isfinite(pose).all():
            raise ValueError('55 finite axis-angle rotations required')
        pose=pose.copy()
        pose[[1,2,4,5,7,8]]=Rotation.from_rotvec(pose[[1,2,4,5,7,8]]).as_rotvec()
        output=np.zeros((2,2,3))
        for side,(hip,knee,ankle) in enumerate(((1,4,7),(2,5,8))):
            h=np.rad2deg(pose[hip]);k=np.rad2deg(pose[knee])
            other=np.r_[h[1:],k[1:]]
            if len(self.off_axis)==7:other=np.r_[other,np.rad2deg(pose[ankle])]
            if np.any(np.abs(other)>self.off_axis+1e-6):
                raise ValueError('pose outside baked leg off-axis support; angle was not clipped')
            i,u=self._interval(self.hip_axis,h[0]);j,v=self._interval(self.knee_axis,k[0])
            grid=((1-u)*(1-v)*self.values[i,j,:,side]+u*(1-v)*self.values[i+1,j,:,side]+
                  (1-u)*v*self.values[i,j+1,:,side]+u*v*self.values[i+1,j+1,:,side])
            result=grid[0].copy()
            for axis,value in enumerate(other):
                if self.off_knots is None:
                    variant=1+2*axis+int(value>0)
                    result+=(grid[variant]-grid[0])*abs(value)/self.off_axis[axis]
                else:
                    start=1+axis*(len(self.off_knots)-1)
                    knots=list(grid[start:start+len(self.off_knots)-1])
                    knots.insert(int(np.flatnonzero(self.off_knots==0)[0]),grid[0])
                    m,w=self._interval(self.off_knots,value/self.off_axis[axis])
                    result+=(1-w)*knots[m]+w*knots[m+1]-grid[0]
            # Exceeding a calibrated amplitude is an explicit unsupported case,
            # not a silent clipping or a zero-correction fallback.
            if np.any(np.linalg.norm(result,axis=1)>np.deg2rad(25)):
                raise ValueError('baked leg correction exceeds 25 degree norm support')
            output[side]=result
        selected=[1,2,4,5,7,8] if len(self.off_axis)==7 else [1,2,4,5]
        if not np.any(pose[selected]):output[:]=0
        return output

    def save(self,path):
        optional={} if self.off_knots is None else dict(off_knots=self.off_knots)
        np.savez_compressed(path,hip_axis=self.hip_axis,knee_axis=self.knee_axis,
            off_axis=self.off_axis,values=self.values,head_centers_local=self.head_centers_local,
            **optional,
            metadata_json=np.asarray(json.dumps(dict(schema='BakedLegArticulationV17',report=self.report),allow_nan=False)))

    @classmethod
    def load(cls,path):
        with np.load(Path(path),allow_pickle=False) as data:
            fields={'hip_axis','knee_axis','off_axis','values','head_centers_local','metadata_json'}
            if set(data.files) not in (fields,fields|{'off_knots'}):
                raise ValueError('invalid articulation archive fields')
            meta=json.loads(str(data['metadata_json']))
            if meta['schema']!='BakedLegArticulationV17':raise ValueError('invalid articulation schema')
            return cls(**{k:data[k].copy() for k in ('hip_axis','knee_axis','off_axis','values','head_centers_local')},
                       report=meta['report'],off_knots=data['off_knots'].copy() if 'off_knots' in data.files else None)


def bake_leg_articulation_v17(subject,calibration,model,progress=None):
    """Compatibility entry point; new compilation uses coupled skin constraints.

    The separable class above is retained only to replay archived prototypes.
    Its independent-axis fitting grid failed mixed rotations and is not used
    for new subject packages.
    """
    from .coupled_leg_articulation_v17 import bake_coupled_leg_articulation_v17
    return bake_coupled_leg_articulation_v17(subject,calibration,model,progress=progress)
