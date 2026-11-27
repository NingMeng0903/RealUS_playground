from pathlib import Path
import sys,numpy as np,json,time,igl,trimesh
ROOT=Path('/media/camp/EXT_DRIVE/Among_US');OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas,save_leg_volume_atlas
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.shaft_rebuild import build_shaft_roi,tet_determinants
TARGET=OUT/'shaft_roi_v1';TARGET.mkdir(exist_ok=True)
results={}
for side in ['right','left']:
 start=time.perf_counter();a=load_leg_volume_atlas(OUT/f'rebuild_work/owned_butterfly/atlas_{side}.npz');old=load_leg_volume_atlas(OUT/f'ownership_smooth/owned_smooth_{side}_skin.npz');ow=np.load(OUT/f'ownership_smooth/{side}_surface_lbs_weights.npz')['weights']
 dist,faces,closest=igl.point_mesh_squared_distance(a.skin_vertices.astype(float),old.skin_vertices.astype(float),old.skin_faces.astype(np.int64));bary=trimesh.triangles.points_to_barycentric(old.skin_vertices[old.skin_faces[faces]].astype(float),closest);bary=np.maximum(bary,0);bary/=bary.sum(1)[:,None];w=np.einsum('ni,nij->nj',bary,ow[old.skin_faces[faces]])
 try:b,attachments,metrics=build_shaft_roi(a,w)
 except Exception as error:print(side,'FAILED',repr(error),flush=True);raise
 metrics['surface_rebinding_distance_m']=dict(mean=float(np.sqrt(dist).mean()),p95=float(np.percentile(np.sqrt(dist),95)),max=float(np.sqrt(dist).max()))
 save_leg_volume_atlas(TARGET/f'atlas_{side}.npz',b);np.savez_compressed(TARGET/f'attachments_{side}.npz',**attachments,source_butterfly_vertex_nearest_old_owned_face=faces,source_butterfly_vertex_barycentric=bary)
 loaded=load_leg_volume_atlas(TARGET/f'atlas_{side}.npz');det=tet_determinants(loaded.harmonic_vertices,loaded.harmonic_tets);assert np.all(det>0)
 metrics['after_save_load_det_min']=float(det.min());metrics['wall_s']=time.perf_counter()-start;results[side]=metrics
 (TARGET/'generation_metrics.json').write_text(json.dumps(results,indent=2));print(side,metrics,flush=True)
