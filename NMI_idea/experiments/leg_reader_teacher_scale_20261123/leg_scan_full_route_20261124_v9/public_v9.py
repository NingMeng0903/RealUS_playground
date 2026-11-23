"""Causal public prefix features with a cache for immutable observed rows."""

import study_v9
from fast_embedding import (_atlas_features_fast, _finite, _frame_points, _material_features_fast, _merge_values,
                            _route_bounds, _same_frame_separations, ATLAS_DISTANCE_MM, FRAME_STATION_RADIUS,
                            STATIC_CORRELATION_MM, TRACK_DISTANCE_MM)
from connected_reader import _public_evidence as _legacy_public_evidence, _target_prior as _legacy_target_prior
from numba import njit
import numpy as np


@njit(cache=True, nogil=True)
def dynamic_features(s, a, points, q, p, frame_id, first, order, cuts, key):
    rows = len(s)
    result = np.zeros((rows, 6),dtype=np.float32)
    max_key = int(key.max())+1 if rows else 1
    support_seen = np.zeros(max_key,dtype=np.int64)
    history_seen = np.zeros(max_key,dtype=np.int64)
    for i in range(rows):
        bucket = int(np.floor(s[i]))
        low = cuts[max(bucket-4,0)]
        hi = cuts[min(bucket+5,len(cuts)-1)]
        best = before = after = corr = 0.
        views = frames = cross = 0
        for index in range(low,hi):
            j = order[index]
            ds = abs(s[j]-s[i])
            da = abs((a[j]-a[i]+12.)%24.-12.)
            if first[j] and frame_id[i]!=frame_id[j] and ds<=3.2:
                frames+=1
                cross+=int(da>=.75)
                corr=max(corr,np.exp(-.5*(ds/.55)**2-.5*(da/.75)**2))
                if history_seen[key[j]]!=i+1:
                    views+=1
                    history_seen[key[j]]=i+1
            valid=((.60<=ds<=FRAME_STATION_RADIUS) or (ds<=2.05 and da>=.75))
            if not valid or support_seen[key[j]]==i+1:
                continue
            distance=np.hypot(points[j,0]-points[i,0],points[j,1]-points[i,1])
            if distance>TRACK_DISTANCE_MM:
                continue
            similarity=np.exp(-.5*(distance/6.5)**2)*(.60+.40*min(q[i],q[j]))*(.55+.45*np.sqrt(min(p[i],p[j])))
            similarity*=1.-.70*np.exp(-.5*(ds/STATIC_CORRELATION_MM)**2-.5*(da/.75)**2)
            if similarity<=0.:
                continue
            support_seen[key[j]]=i+1
            best=max(best,similarity)
            if s[j]<s[i]-.20:
                before=max(before,similarity)
            if s[j]>s[i]+.20:
                after=max(after,similarity)
        result[i,0]=best
        result[i,1]=min(before,after)
        result[i,2]=1.-corr
        result[i,3]=min(1.,views/8.)
        result[i,4]=min(1.,frames/12.)
        result[i,5]=cross/frames if frames else 0.
    return result


class Prefix:
    """Cache candidate-local geometry and re-evaluate history at every prefix."""

    def __init__(self, env, spec):
        self.material=env.material
        self.begin,_,self.span=_route_bounds(spec)
        self.static=[]
        self.last_frame=None
        self.frame_count=0
        self.cache_atlas={}
        self.cache_material={}

    def fork(self):
        from copy import copy
        new=copy(self)
        new.static=self.static.copy()
        # Pose-chart mappings are public and independent of which branch
        # observed the geometry. The observed row list is branch-local.
        return new

    def _append(self, reader, index, frame):
        station,sector,echoes,_,quality,region,posterior,unknown,tick=frame
        points=_frame_points(reader,frame)
        if not 0<=region<6:
            return
        quality=float(np.clip(quality,0.,1.))
        for j, point in enumerate(points):
            if not isinstance(point,np.ndarray) or point.shape!=(2,) or not np.isfinite(point).all():
                continue
            p=float(np.clip(posterior[j] if j<len(posterior) else 0.,0.,1.))
            u=float(np.clip(unknown[j] if j<len(unknown) else 1.,0.,1.))
            merge, observed=_merge_values(reader,index,j)
            separated,separation_mask=_same_frame_separations({index:points},index,j)
            if callable(separated):
                separated,separation_mask=separated(reader,index,j)
            beam=np.asarray(echoes[j],dtype=float)
            angle=sector*2.*np.pi/24.
            baseline=np.array([0.,p,u,merge,separated,beam[0]/25.,beam[1]/80.,np.clip((station-self.begin)/self.span,-1.,2.),
                               np.sin(angle),np.cos(angle),0.,quality],dtype=np.float32)
            baseline_mask=np.ones(12,dtype=np.float32);baseline_mask[3:5]=(observed,separation_mask)
            record={"frame":index,"slot":j,"region":region,"station":station,"sector":sector,"world":point}
            material,material_mask=_material_features_fast(reader,self.material,record,beam,self.begin,self.span,self.cache_material)
            atlas,atlas_mask=_atlas_features_fast(reader,record,self.cache_atlas)
            prior=_legacy_target_prior(reader,station)
            value=0. if prior is None else float(np.exp(-.5*(np.linalg.norm(prior-point,axis=-1).min()/ATLAS_DISTANCE_MM)**2))
            fallback=.62*p+.08*value+.04*quality+.02*(1.-u)
            features=np.r_[baseline,baseline_mask,material,atlas,np.zeros(6),material_mask,atlas_mask,np.ones(6)].astype(np.float32)
            self.static.append((index,j,region,station,sector,float(frame[4]),quality,point,p,tick,fallback,features))

    def update(self,node):
        reader=node.reader
        frames=reader.frames
        if self.frame_count and (len(frames)<self.frame_count or frames[self.frame_count-1] is not self.last_frame):
            self.static=[]; self.frame_count=0
        for i in range(self.frame_count,len(frames)):
            self._append(reader,i,frames[i])
        self.frame_count=len(frames)
        self.last_frame=frames[-1] if frames else None
        if not self.static:
            return {"features":np.empty((0,96),dtype=np.float16),"points":np.empty((0,2),dtype=np.float32)} | {
                name:np.empty(0,dtype=dtype) for name,dtype in (("frames",int),("slots",int),("regions",int),
                  ("stations",np.float32),("sectors",np.float32),("quality",np.float32))}
        f,sl,r,s,a,q,clipped,point,p,t,base,x=(list(items) for items in zip(*self.static,strict=True))
        f,sl,r=np.asarray(f),np.asarray(sl),np.asarray(r)
        s,a=np.asarray(s),np.asarray(a)
        point=np.asarray(point)
        x=np.stack(x)
        existing=_legacy_public_evidence(reader)
        x[:,0]=[existing.get((i,j),fallback) for i,j,fallback in zip(f,sl,base,strict=True)]
        x[:,10]=np.clip(np.asarray(t,dtype=float)/max(1.,_finite(frames[-1][8])),0.,2.)
        bucket=np.floor(s).astype(np.int64); order=np.argsort(bucket,kind="stable")
        cuts=np.searchsorted(bucket[order],np.arange(bucket.max()+2),side="left")
        first=np.r_[True,f[1:]!=f[:-1]]
        key=(np.rint(s)*25+np.rint(a)).astype(np.int64)
        x[:,54:60]=dynamic_features(s,a,point,np.asarray(clipped),np.asarray(p),f,first,order,cuts,key)
        return {"features":x.astype(np.float16),"frames":f,"slots":sl,"regions":r,"stations":s.astype(np.float32),
                "sectors":a.astype(np.float32),"points":point.astype(np.float32),"quality":np.asarray(q,dtype=np.float32)}
