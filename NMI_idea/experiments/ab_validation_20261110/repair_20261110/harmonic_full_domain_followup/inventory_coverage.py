from pathlib import Path
import json,hashlib,numpy as np,igl
BASE=Path(__file__).resolve().parent;PRIOR=BASE.parent/'harmonic';ROOT=Path('/media/camp/EXT_DRIVE/Among_US');source=ROOT/'dataset/processed/anatomy_retarget/leg_volume_coordinates/production/vessels/vessel_material_coordinates.npz';a=np.load(source,allow_pickle=True);result={}
for side in ['right','left']:
 s=np.load(PRIOR/f'ownership_smooth/owned_smooth_{side}_skin.npz');mask=np.asarray(a['side'],str)==side;points=a['projected_points'][mask].astype(float);labels=np.asarray(a['labels'],str)[mask];xi=a['xi_skin'][mask];orig=a['original_points'][mask]
 d,f,c=igl.point_mesh_squared_distance(points,s['skin_vertices'].astype(float),s['skin_faces'].astype(np.int64))
 for name in np.unique(labels):
  m=labels==name;x=points[m];order=np.argsort(xi[m,1]);result[name]=dict(side=side,count=int(sum(m)),legacy_h_min=float(xi[m,1].min()),legacy_h_max=float(xi[m,1].max()),projected_bbox=[x.min(0).tolist(),x.max(0).tolist()],h_sorted_endpoints=x[order[[0,-1]]].tolist(),owned_skin_distance_mean_m=float(np.sqrt(d[m]).mean()),owned_skin_distance_max_m=float(np.sqrt(d[m]).max()),original_to_projected_mean_m=float(np.linalg.norm(orig[m]-x,axis=1).mean()))
np.savez_compressed(BASE/'legacy_template_evaluation_points.npz',labels=np.asarray(a['labels'],str),side=np.asarray(a['side'],str),original_points=a['original_points'],projected_points=a['projected_points'],legacy_xi=a['xi_skin'])
manifest=dict(source_path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),role='provided template/old-bake path; controller-prior eligibility follows explicit task authorization; not identified as held-out hidden patient truth by filename',frame='canonical anatomy coordinates from source bundle',labels=result);(BASE/'coverage_inventory.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
