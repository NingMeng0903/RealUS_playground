from pathlib import Path
import numpy as np,igl,json,time
OUT=Path(__file__).resolve().parent;ROOT=Path('/media/camp/EXT_DRIVE/Among_US')
def stats(d):return dict(count=len(d),mean=float(np.mean(d)),p95=float(np.percentile(d,95)),max=float(np.max(d)))
r={}
for side in ['right','left']:
 a=np.load(OUT/f'rebuild_work/owned_butterfly/atlas_{side}.npz');z=np.load(OUT/f'tetgen_prototype/{side}_owned_clean_meshfix_outer_only.npz');v=z['repaired_surface_vertices'];f=z['repaired_surface_faces'];original_v=a['skin_vertices'].astype(float);original_f=a['skin_faces'].astype(np.int64)
 d,idx,closest=igl.point_mesh_squared_distance(original_v,v,f.astype(np.int64));reverse,ri,rc=igl.point_mesh_squared_distance(v,original_v,original_f)
 # Keep cap distances explicitly separate: classify face centers by source skin
 # surface proximity; retained skin lies within numerical tolerance of it.
 fc=v[f].mean(1);fd,fi,fp=igl.point_mesh_squared_distance(fc,original_v,original_f);tris=v[f];area=np.linalg.norm(np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]),axis=1)/2
 prod=np.load(ROOT/f'dataset/processed/anatomy_retarget/leg_volume_coordinates/production/atlas/atlas_{side}.npz');pd,_,_=igl.point_mesh_squared_distance(prod['skin_vertices'].astype(float),v,f.astype(np.int64))
 dd=np.sqrt(d);rr=np.sqrt(reverse);boundarykeep=np.all(np.isclose(original_v[:,None] if False else 0,0))
 result=dict(source='clean owned Butterfly level3',original_vertices=len(original_v),repaired_vertices=len(v),original_faces=len(original_f),repaired_faces=len(f),clean_source_vertex_to_repaired_closed_surface_m=stats(dd),repaired_surface_vertex_to_clean_source_open_skin_m=stats(rr),original_mixed_production_vertex_to_repaired_closed_surface_m=stats(np.sqrt(pd)),source_vertex_coverage_below_1um=float(np.mean(dd<1.e-6)),source_vertex_coverage_below_1mm=float(np.mean(dd<1.e-3)),source_vertex_coverage_below_5mm=float(np.mean(dd<5.e-3)),changed_source_vertex_indices_over_1um=np.flatnonzero(dd>1.e-6).tolist(),changed_source_vertex_h=stats(a['skin_h'][dd>1.e-6]) if np.any(dd>1.e-6) else {},surface_area_m2=float(area.sum()),source_skin_coincident_face_area_m2=float(area[fd<1.e-12].sum()),cap_or_changed_face_area_m2=float(area[fd>=1.e-12].sum()),canonical_volume_det_percentiles=np.percentile(np.linalg.det(np.moveaxis(z['vertices'][z['tets']][:,1:]-z['vertices'][z['tets']][:,:1],1,2)),[0,1,50,99,100]).tolist())
 r[side]=result;print(side,result,flush=True);np.savez_compressed(OUT/f'tetgen_prototype/{side}_surface_correspondence.npz',source_vertex_distances=dd,source_vertex_closest_points=closest,repaired_vertex_source_face_indices=ri,repaired_vertex_closest_source_points=rc,repaired_vertex_source_distances=rr,source_face_center_distances=np.sqrt(fd),source_face_center_nearest_source_face=fi)
(OUT/'repaired_surface_metrics.json').write_text(json.dumps(r,indent=2))
