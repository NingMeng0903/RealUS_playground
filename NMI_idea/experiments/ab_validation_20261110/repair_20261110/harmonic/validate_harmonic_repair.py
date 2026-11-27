#!/usr/bin/env python3
"""Actual production query validation; no monkey patch or depth adapter."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

REPO=Path('/media/camp/EXT_DRIVE/Among_US');sys.path.insert(0,str(REPO/'src'))
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas,query_atlas_coordinates
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.harmonic import _distance_to_skin_surface
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.volume_refine import VolumeTetRefiner
OUT=Path(__file__).resolve().parent
PROD=REPO/'dataset/processed/anatomy_retarget/leg_volume_coordinates/production/atlas'

def summary(x):
    x=np.asarray(x,float)
    return dict(count=x.size,min=float(x.min()),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(x.max()))

def nearest_core(points,core):
    a,b=core[:-1].astype(float),core[1:].astype(float);edge=b-a
    d=points[:,None]-a[None]
    u=np.clip(np.einsum('nkj,kj->nk',d,edge)/np.maximum(np.sum(edge*edge,axis=1),1.e-30),0,1)
    closest=a[None]+u[:,:,None]*edge[None]
    distance=np.linalg.norm(points[:,None]-closest,axis=2)
    ids=np.argmin(distance,axis=1)
    return closest[np.arange(len(points)),ids],distance[np.arange(len(points)),ids]

def zero_faces(atlas):
    d0=atlas.harmonic_d==0
    t=atlas.harmonic_tets[np.sum(d0[atlas.harmonic_tets],axis=1)>=3]
    f=np.concatenate([t[:,ids] for ids in ([0,1,2],[0,1,3],[0,2,3],[1,2,3])])
    f=f[np.all(d0[f],axis=1)]
    return np.unique(np.sort(f,axis=1),axis=0)

def run(side):
    started=time.perf_counter();path=PROD/f'atlas_{side}.npz';a=load_leg_volume_atlas(path)
    rng=np.random.default_rng(20261110)
    skinids=np.linspace(0,len(a.skin_vertices)-1,16,dtype=int)
    nids=rng.choice(len(a.harmonic_vertices),96,replace=False)
    fids=rng.choice(len(a.skin_faces),256,replace=False)
    bary=rng.dirichlet(np.ones(3),len(fids))
    facepoints=np.einsum('ni,nij->nj',bary,a.skin_vertices[a.skin_faces[fids]])
    corepts=np.vstack([a.core_points,(a.core_points[:-1]+a.core_points[1:])/2])
    skinq,_=query_atlas_coordinates(a,a.skin_vertices)
    faceq,_=query_atlas_coordinates(a,facepoints)
    coreq,_=query_atlas_coordinates(a,corepts)
    oldq,_=query_atlas_coordinates(a,a.skin_vertices[skinids])
    node_ratio,_=query_atlas_coordinates(a,a.harmonic_vertices[nids])
    closest,_=nearest_core(facepoints,a.core_points)
    direction=closest-facepoints;direction/=np.linalg.norm(direction,axis=1)[:,None]
    offsets=np.array([0.,1.e-6,1.e-5,1.e-4,1.e-3,5.e-3])
    near=facepoints[:,None]+offsets[None,:,None]*direction[:,None]
    nearq,_=query_atlas_coordinates(a,near.reshape(-1,3));neard=nearq[:,2].reshape(len(facepoints),-1)
    # Independent all-triangle nearest-point implementation (trimesh), 16 samples.
    check=facepoints[:16]+.001*direction[:16]
    mesh=trimesh.Trimesh(vertices=a.skin_vertices,faces=a.skin_faces,process=False)
    _,independent_skin,_=trimesh.proximity.closest_point_naive(mesh,check)
    _,independent_medial=nearest_core(check,a.core_points)
    independent_ratio=independent_skin/(independent_skin+independent_medial)
    checkq,_=query_atlas_coordinates(a,check)
    R=Rotation.from_rotvec([.2,-.3,.5]).as_matrix();t=np.array([.31,-.2,.7])
    moved=replace(a,skin_vertices=(a.skin_vertices@R.T+t).astype(np.float32),
        core_points=(a.core_points@R.T+t).astype(np.float32),harmonic_vertices=(a.harmonic_vertices@R.T+t).astype(np.float32))
    movedq,_=query_atlas_coordinates(moved,check@R.T+t)
    # All sampled node coordinates undergo the same geometric conflict checks;
    # no direct nodal lookup shortcut bypasses overlapping tetrahedra.
    node_ids=np.unique(np.r_[nids,rng.choice(np.flatnonzero(a.harmonic_d==0),96,replace=False),rng.choice(np.flatnonzero(a.harmonic_d==1),96,replace=False)])
    baked=np.full((len(node_ids),3),np.nan,np.float32);node_conflicts=[]
    for qi,ni in enumerate(node_ids):
        try:baked[qi]=query_atlas_coordinates(a,a.harmonic_vertices[[ni]],depth_mode='baked')[0][0]
        except ValueError as e:
            if 'Ambiguous baked' not in str(e):raise
            node_conflicts.append(dict(node_index=int(ni),message=str(e)))
    node_valid=np.isfinite(baked).all(1)
    target=np.column_stack([a.harmonic_theta[node_ids],a.harmonic_h[node_ids],a.harmonic_d[node_ids]])
    node_error=baked-target;node_error[:,0]=(node_error[:,0]+np.pi)%(2*np.pi)-np.pi
    rng=np.random.default_rng(20261110);rng.choice(len(a.harmonic_vertices),96,replace=False);rng.choice(len(a.skin_faces),256,replace=False);rng.dirichlet(np.ones(3),256)
    tids=rng.choice(len(a.harmonic_tets),256,replace=False)
    weights=rng.dirichlet(np.ones(4),len(tids))
    tp=np.einsum('ni,nij->nj',weights,a.harmonic_vertices[a.harmonic_tets[tids]])
    bq=np.full((len(tp),3),np.nan,np.float32);ambiguous=[]
    for qi,point in enumerate(tp):
        try:
            bq[qi]=query_atlas_coordinates(a,point[None],depth_mode='baked')[0][0]
        except ValueError as e:
            if 'Ambiguous baked' not in str(e): raise
            # Independent exhaustive AABB locator and barycentric calculation.
            vv=a.harmonic_vertices[a.harmonic_tets].astype(float)
            candidates=np.flatnonzero(np.all(point>=vv.min(1)-1.e-9,axis=1)&np.all(point<=vv.max(1)+1.e-9,axis=1))
            matches=[]
            for ti in candidates:
                corners=a.harmonic_vertices[a.harmonic_tets[ti]].astype(float)
                try:local=np.linalg.solve((corners[1:]-corners[0]).T,point.astype(np.float32).astype(float)-corners[0])
                except np.linalg.LinAlgError:continue
                w=np.r_[1-local.sum(),local]
                if np.min(w)<-1.e-7:continue
                ids=a.harmonic_tets[ti];angles=a.harmonic_theta[ids].astype(float)
                vals=[float(np.arctan2(w@np.sin(angles),w@np.cos(angles))%(2*np.pi)),float(w@a.harmonic_h[ids]),float(w@a.harmonic_d[ids])]
                matches.append(dict(tet_index=int(ti),vertex_indices=ids.tolist(),barycentric_weights=w.tolist(),xi=vals))
            ratio_q=query_atlas_coordinates(a,point[None])[0][0]
            independent_skin_at_point=trimesh.proximity.closest_point_naive(mesh,point[None])[1][0]
            independent_core_at_point=nearest_core(point[None],a.core_points)[1][0]
            independent_ratio_at_point=float(independent_skin_at_point/(independent_skin_at_point+independent_core_at_point))
            ambiguous.append(dict(default_distance_ratio_xi=ratio_q.tolist(),independent_distance_ratio=independent_ratio_at_point,sample_index=qi,original_tet_index=int(tids[qi]),query_point_m=point.astype(np.float32).tolist(),message=str(e),independent_containing_tets=matches))
            del vv
    valid=np.all(np.isfinite(bq),axis=1)
    refiner=VolumeTetRefiner.from_atlas(a)
    roundtrip_points=tp[valid][:32]
    roundtrip_xi=refiner.p_to_xi(roundtrip_points)
    roundtrip_decoded=refiner.xi_to_p(roundtrip_xi)
    roundtrip_error=np.linalg.norm(roundtrip_decoded-roundtrip_points,axis=1)
    np.savez_compressed(OUT/f'{side}_precision_roundtrip.npz',points=roundtrip_points,xi=roundtrip_xi,decoded=roundtrip_decoded)
    (OUT/f'{side}_overlap_evidence.json').write_text(json.dumps(ambiguous,indent=2))
    td=np.einsum('ni,ni->n',weights,a.harmonic_d[a.harmonic_tets[tids]])
    bh=np.einsum('ni,ni->n',weights,a.harmonic_h[a.harmonic_tets[tids]])
    angles=a.harmonic_theta[a.harmonic_tets[tids]].astype(float)
    bt=np.mod(np.arctan2(np.sum(weights*np.sin(angles),axis=1),np.sum(weights*np.cos(angles),axis=1)),2*np.pi)
    te=(bq[:,0]-bt+np.pi)%(2*np.pi)-np.pi
    domain_errors={}
    for name,pts in [('actual_skin_16',a.skin_vertices[skinids]),('core_polyline_48',a.core_points),('far_outside',np.array([[5.,5.,5.]]))]:
        try:
            query_atlas_coordinates(a,pts,depth_mode='baked')
            domain_errors[name]='accepted'
        except ValueError as e:
            domain_errors[name]=str(e)
    try:
        extrapolated=query_atlas_coordinates(a,a.skin_vertices[skinids],depth_mode='baked',outside='nearest')[0][:,2].tolist()
    except ValueError as e:
        extrapolated=str(e)
    zfaces=zero_faces(a);outer=a.harmonic_vertices[a.harmonic_d==0]
    outer_distance=_distance_to_skin_surface(outer,a.skin_vertices,a.skin_faces)
    skin_distance=_distance_to_skin_surface(a.skin_vertices,a.harmonic_vertices,zfaces)
    orphan_ids=np.setdiff1d(np.arange(len(a.skin_vertices)),np.unique(a.skin_faces))
    orphan_distances=trimesh.proximity.closest_point_naive(mesh,a.skin_vertices[orphan_ids])[1].tolist() if len(orphan_ids) else []
    result=dict(side=side,asset_path=str(path),asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        counts=dict(skin_vertices=len(a.skin_vertices),skin_faces=len(a.skin_faces),baked_vertices=len(a.harmonic_vertices),baked_tets=len(a.harmonic_tets),baked_d0_nodes=int(sum(a.harmonic_d==0)),baked_d1_nodes=int(sum(a.harmonic_d==1))),
        runtime_distance_ratio=dict(all_skin_vertices_abs_d=summary(abs(skinq[:,2])),skin_triangle_interiors_abs_d=summary(abs(faceq[:,2])),core_points_and_segments_abs_d_minus1=summary(abs(coreq[:,2]-1)),
            original_16_skin_xi=oldq.tolist(),independent_all_triangle_ratio_abs_error=summary(abs(checkq[:,2]-independent_ratio)),common_rigid_motion_depth_abs_error=summary(abs(movedq[:,2]-checkq[:,2])),
            near_skin_offsets_m=offsets.tolist(),near_skin_depth_mean=neard.mean(0).tolist(),near_skin_depth_max=neard.max(0).tolist(),near_skin_monotone_fraction=float(np.mean(np.all(np.diff(neard,axis=1)>=-1.e-6,axis=1))),
            ratio_vs_baked_at_96_nodes_abs_error=summary(abs(node_ratio[:,2]-a.harmonic_d[nids]))),
        baked_native_tetrahedra=dict(sampled_node_count=len(node_ids),accepted_node_count=int(sum(node_valid)),ambiguous_node_count=len(node_conflicts),ambiguous_node_evidence=node_conflicts,node_coordinate_abs_error_max=np.max(abs(node_error[node_valid]),axis=0).tolist(),
            barycentric_sample_count=256,unambiguous_sample_count=int(sum(valid)),ambiguous_sample_count=len(ambiguous),baked_geometry_gate_passed=not bool(ambiguous or node_conflicts),
            barycentric_unambiguous_coordinate_abs_error_max=[float(abs(te[valid]).max()),float(abs(bq[valid,1]-bh[valid]).max()),float(abs(bq[valid,2]-td[valid]).max())],
            unassisted_32_point_precision_roundtrip_error_m=summary(roundtrip_error),overlap_evidence=ambiguous,
            sampled_d0_boundary_abs_error=float(abs(baked[node_valid & (target[:,2]==0),2]).max()),sampled_d1_boundary_abs_error=float(abs(baked[node_valid & (target[:,2]==1),2]-1).max()),domain_checks=domain_errors,
            actual_skin_16_with_explicit_nearest_depth=extrapolated),
        distinct_boundaries=dict(baked_outer_nodes_to_actual_skin_geometric_union_m=summary(outer_distance),actual_skin_vertices_to_baked_zero_surface_m=summary(skin_distance),
            isolated_skin_vertex_ids=orphan_ids.tolist(),isolated_vertex_distance_to_triangles_m=orphan_distances,actual_skin_h_range=[float(a.skin_h.min()),float(a.skin_h.max())],baked_h_range=[float(a.harmonic_h.min()),float(a.harmonic_h.max())]),wall_s=time.perf_counter()-started)
    np.savez_compressed(OUT/f'{side}_boundary_samples.npz',skin_face_ids=fids,skin_barycentric_weights=bary,skin_triangle_points=facepoints,near_skin_offsets_m=offsets,near_skin_points=near,near_skin_depth=neard,native_tet_ids=tids,native_barycentric_weights=weights,native_tet_points=tp,native_tet_xi=bq)
    (OUT/f'{side}_metrics.json').write_text(json.dumps(result,indent=2))
    # Production acceptance criteria for the corrected runtime and separate baked mode.
    assert abs(skinq[:,2]).max()<1.e-6
    assert abs(faceq[:,2]).max()<5.e-6
    assert abs(coreq[:,2]-1).max()<1.e-5
    assert abs(checkq[:,2]-independent_ratio).max()<5.e-6
    assert abs(movedq[:,2]-checkq[:,2]).max()<1.e-5
    assert abs(node_error[node_valid]).max()<1.e-6
    assert max(abs(te[valid]).max(),abs(bq[valid,1]-bh[valid]).max(),abs(bq[valid,2]-td[valid]).max())<2.e-5
    assert roundtrip_error.max()<1.e-4
    assert domain_errors['far_outside']!='accepted'
    return result

if __name__=='__main__':
    results={}
    for side in ('right','left'):
        results[side]=run(side)
        (OUT/'production_metrics.json').write_text(json.dumps(results,indent=2))
        print(side,json.dumps(results[side],indent=2),flush=True)
