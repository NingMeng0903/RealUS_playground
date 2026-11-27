"""Construct short scan windows on named right-leg atlas geometry."""
from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from inputs import SKIN, SCAN_OUT, meshes, side_mesh, target_model, Slicer, write_json
from align_cloud_sources import section_outline

S_MIN, S_MAX = 2., 31.4
STATIONS = np.linspace(S_MIN, S_MAX, 121)
CONTACT_X = np.linspace(-30., 0., 61)
OUTWARD = {"posterior": np.array([0., -1.]),
           "anteromedial": np.array([.38, np.sqrt(1.-.38**2)])}

# Roles establish the ground-truth object tracked by the evaluator.
# Supplemental structures enter as anonymous echoes at the same cadence.
TERMS = {
    "BodyParts3D": {
        "fv":"right femoral vein", "fa":"right femoral artery",
        "dfv":"right deep femoral vein", "gsv":"right great saphenous vein",
        "pv":"right popliteal vein", "pa":"right popliteal artery", "ssv":"right small saphenous vein",
    },
    "Open3DModel": {
        "fv":"Femoral_vein.r", "fa":"Femoral_artery.r",
        "dfv":"Deep_femoral_vein.r", "gsv":"Great_saphenous_vein.r",
        "pv":"Popliteal_vein.r", "pa":"Popliteal_artery.r", "ssv":"Small_saphenous_vein.r",
    },
}

TASKS = {
    "popliteal": ("parallel", "posterior", ("pv", "pa", "ssv")),
    "thigh_trunk": ("parallel", "anteromedial", ("fv", "fa", "dfv")),
    "stay_on_deep_trunk": ("connection", "anteromedial", ("fv", "gsv", "fa", "dfv")),
    "follow_saphenous": ("connection", "anteromedial", ("gsv", "fv", "fa", "dfv")),
}
ROLE_KEYS = np.array([0,1,2,5], dtype=int)
LABELS = {"fv":"股静脉主干", "fa":"股动脉", "dfv":"股深静脉", "gsv":"大隐静脉",
          "pv":"腘静脉", "pa":"腘动脉", "ssv":"小隐静脉"}


@lru_cache(maxsize=1)
def frozen_starts():
    registry=os.environ.get("LEG_TRANSFER_TASK_REGISTRY")
    if not registry:
        return {}
    return {(row["source"],row["task"]):float(row["atlas_y_span_mm"][1])
            for row in json.loads(Path(registry).read_text())}


@lru_cache(maxsize=2)
def parts(source):
    archive=meshes()
    out={}
    for key, name in TERMS[source].items():
        mesh=side_mesh(archive[f"{source}|{name}"], "R")
        out[key]=Slicer(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    return out


def largest(slicer, y):
    """Reject sub-mm slivers and keep the main visible lumen of a named segment."""
    return max(slicer.at(float(y), min_area=.6), key=lambda ring:ring.area, default=None)


@lru_cache(maxsize=1)
def anatomical_axis():
    _, controls=target_model()
    return controls[:,0]


def skin_depth_profile(points, outward, tangent, lateral):
    endpoints=points[:,[0,2]].reshape(-1,2,2)
    u=endpoints @ tangent
    d=endpoints @ outward
    low=u.min(axis=1); high=u.max(axis=1)
    mask=((lateral[:,None]>=low) & (lateral[:,None]<=high) & (high>low+1.e-9))
    denominator=u[:,1]-u[:,0]
    t=(lateral[:,None]-u[:,0])/np.where(abs(denominator)>1.e-9,denominator,1.)
    profile=np.max(np.where(mask,d[:,0]+t*(d[:,1]-d[:,0]),-np.inf),axis=1)
    if not np.isfinite(profile).all():
        raise ValueError("contact strip leaves the SMPL-X skin")
    return profile


class ScanChart:
    """Coordinates use millimeters. Each same-station pose has a fixed in-plane tilt."""
    def __init__(self, y0, view):
        self.y0=float(y0)
        self.outward=OUTWARD[view]
        h,v=self.outward
        self.tangent=np.array([abs(v),-h*np.sign(v)])
        self.ys=self.y0-(STATIONS-S_MIN)
        control=anatomical_axis()
        anchor=np.column_stack([np.interp(self.ys,control[:,1],control[:,j]) for j in (0,2)])
        self.anchor=anchor @ self.tangent
        outline=section_outline(SKIN["neutral"])
        self.depth_grid=np.stack([
            skin_depth_profile(outline(y), self.outward,self.tangent,a+CONTACT_X+15.)
            for y,a in zip(self.ys,self.anchor,strict=True)
        ])
        self.interpolant=RegularGridInterpolator((STATIONS,CONTACT_X),self.depth_grid,
                                                  bounds_error=False,fill_value=None)

    def depth_of_contact(self, s, x):
        s,x=np.broadcast_arrays(np.asarray(s,dtype=float),np.asarray(x,dtype=float))
        query=np.column_stack((np.clip(s.ravel(),S_MIN,S_MAX),np.clip(x.ravel(),-30.,0.)))
        return self.interpolant(query).reshape(s.shape)

    def poses(self,s,x):
        ss=np.asarray(s,dtype=float)
        lateral=np.interp(ss,STATIONS,self.anchor)+np.asarray(x)+15.
        depth=self.depth_of_contact(ss,x)
        xz=lateral[:,None]*self.tangent+depth[:,None]*self.outward
        xyz=np.column_stack((xz[:,0],self.y0-(ss-S_MIN),xz[:,1]))
        # All views in this chart share one orientation. Costs use actual skin travel.
        ds=np.array([0.,-1.,0.])
        uv=np.array([self.tangent[0],0.,self.tangent[1]])
        frame=np.column_stack((ds,uv,np.cross(ds,uv)))
        return xyz,np.broadcast_to(frame,(len(ss),3,3))

    def cost(self,s,x):
        xyz,_=self.poses(s,x)
        return float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()),0.


@dataclass
class Site:
    source: str
    task: str
    family: str
    chart: ScanChart
    labels: tuple
    centers: np.ndarray
    areas: np.ndarray
    nearest_saphenous_station: dict | None

    def section(self, s, x, jitter=(0.,0.)):
        q=np.clip((float(s)-S_MIN)*(len(STATIONS)-1)/(S_MAX-S_MIN),0,len(STATIONS)-1.)
        i=min(int(q),len(STATIONS)-2)
        f=q-i
        # Do not interpolate a true branch across an absent labeled section.
        centers=(1.-f)*self.centers[i]+f*self.centers[i+1]
        areas=(1.-f)*self.areas[i]+f*self.areas[i+1]
        if f < 1.e-7:
            centers=self.centers[i]
        elif f > 1.-1.e-7:
            centers=self.centers[i+1]
        y0=np.interp(float(s),STATIONS,self.chart.anchor)
        means=(centers @ np.column_stack((self.chart.tangent,-self.chart.outward))
               + [-y0-15.-x, float(self.chart.depth_of_contact(s,x))] + np.asarray(jitter))
        return means,areas,ROLE_KEYS[:len(areas)]

    def preflight(self):
        from leg_venous_stress_world import FOV_MIN,FOV_MAX
        from adaptive_controller import EDGES
        m=np.stack([np.stack([self.section(s,x)[0] for x in CONTACT_X[::4]]) for s in STATIONS])
        visible=np.isfinite(m).all(axis=-1)&((m>=FOV_MIN)&(m<=FOV_MAX)).all(axis=-1)
        center_depths=np.stack([self.section(s,-15.)[0][:,1] for s in STATIONS])
        distances=np.linalg.norm(m[:,:,0]-m[:,:,1],axis=-1)
        refs=[]
        for k in range(len(self.labels)):
            on=visible[...,k].any(axis=1)
            refs.append({"role":int(ROLE_KEYS[k]),"label":LABELS[self.labels[k]],
                         "present_stations":int(np.isfinite(self.centers[:,k]).all(axis=1).sum()),
                         "visible_at_any_pose_stations":int(on.sum()),
                         "visible_regions": [j+1 for j,(lo,hi) in enumerate(zip(EDGES[:-1],EDGES[1:]))
                                             if on[((lo<=STATIONS)&(STATIONS<=hi))].any()],
                         "depth_range_at_x_minus_15_mm":bounds(center_depths[:,k])})
        return {"source":self.source,"task":self.task,"family":self.family,
                "atlas_y_span_mm":[float(self.chart.ys.min()),float(self.chart.ys.max())],
                "geometry_samples":len(STATIONS),"routes":refs,
                "co_visible_pose_count":int((visible[:,:,0]&visible[:,:,1]).sum()),
                "poses_sampled":int(np.prod(visible.shape[:2])),
                "target_vs_primary_distance_mm":bounds(distances[:,0]),
                "near_saphenous_landmark":self.nearest_saphenous_station}


def bounds(values):
    v=np.asarray(values)[np.isfinite(values)]
    return [float(v.min()),float(np.median(v)),float(v.max())] if len(v) else None


def near_saphenous(source):
    atlas=parts(source)
    pairs=[]
    for y in np.arange(-565.25,-474.,.5):
        a,b=(largest(atlas[key],y) for key in ("fv","gsv"))
        if a is not None and b is not None:
            pairs.append((float(np.linalg.norm(a.xyz-b.xyz)),float(y)))
    d,y=min(pairs)
    return {"closest_y_mm":y,"center_distance_mm":d}


@lru_cache(maxsize=8)
def site(source,task):
    family,view,labels=TASKS[task]
    landmark=None
    if task=="popliteal":
        y_start=float(anatomical_axis()[2,1])+14.7
    elif task=="thigh_trunk":
        y_start=float((anatomical_axis()[3,1]+anatomical_axis()[2,1])*.5)+14.7
    elif task=="stay_on_deep_trunk":
        landmark=near_saphenous(source)
        y_start=landmark["closest_y_mm"]+12.
    elif task=="follow_saphenous":
        landmark=near_saphenous(source)
        y_start=landmark["closest_y_mm"]-7.
    else:
        raise KeyError(task)
    chart=ScanChart(frozen_starts().get((source,task),y_start),view)
    atlas=parts(source)
    centers=np.full((len(STATIONS),len(labels),2),np.nan)
    areas=np.zeros(centers.shape[:2],dtype=float)
    for i,y in enumerate(chart.ys):
        for k,name in enumerate(labels):
            r=largest(atlas[name],y)
            if r is not None and r.xyz[0]<0:
                centers[i,k]=r.xyz[[0,2]]
                areas[i,k]=r.area
    return Site(source,task,family,chart,labels,centers,areas,landmark)


def main():
    SCAN_OUT.mkdir(parents=True,exist_ok=True)
    table=[site(source,task).preflight() for source in TERMS for task in TASKS]
    write_json(SCAN_OUT/"atlas_preflight.json",table)
    for row in table:
        print(row["source"],row["task"],row["atlas_y_span_mm"],
              [v["visible_at_any_pose_stations"] for v in row["routes"]],
              row["target_vs_primary_distance_mm"],
              [v["depth_range_at_x_minus_15_mm"] for v in row["routes"][:2]])


if __name__=="__main__":
    main()
