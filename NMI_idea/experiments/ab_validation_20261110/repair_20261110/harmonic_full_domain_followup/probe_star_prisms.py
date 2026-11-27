from pathlib import Path
import sys,json,time,numpy as np,shapely
from shapely.strtree import STRtree
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE));from probe_surface_layers import geometry,PLAN,tet_determinants
ROOT=Path('/media/camp/EXT_DRIVE/Among_US');PRIOR=BASE.parent/'harmonic'
plan=dict(centerline_candidates=['skeletal_axis','linear_fit_core'],radial_fractions=PLAN['radial_fractions'],refinement='one cell center + 8 oriented tets; global shared-quad pulling diagonal; keep exact source outer facets',filter='all serialized star cells positive and material surface orientation positive; then exclude every face participating in periodic material overlap',scan_labels=['R_COM_FEM_V','R_SUPFEMV','R_POPV','R_POST_TIB_V','L_COM_FEM_V','L_SUPFEMV','L_POPV','L_POST_TIB_V'],selection='publish no candidate unless all later graph/material/spatial/pose/boundary gates pass; failures retained')
(BASE/'frozen_star_prism_plan.json').write_text(json.dumps(plan,indent=2))
def star_prisms(v,c,faces,fractions):
 nv=len(v);nf=len(faces);nodes=(c[None]+np.asarray(fractions)[:,None,None]*(v-c)[None]).reshape(-1,3).astype(np.float32);centers=[];all=[]
 for layer in range(len(fractions)-1):
  lo=faces+layer*nv;hi=faces+(layer+1)*nv;center=nodes[np.column_stack([lo,hi])].astype(float).mean(1).astype(np.float32);ci=np.arange(nf)+len(nodes)+layer*nf;centers.append(center);boundary=[lo[:,[0,2,1]],hi]
  for x,y in [(0,1),(1,2),(2,0)]:
   q=np.column_stack([lo[:,x],lo[:,y],hi[:,y],hi[:,x]]);swap=q[:,1]<q[:,0];q[swap]=q[swap][:,[1,2,3,0]];boundary +=[q[:,[0,1,2]],q[:,[0,2,3]]]
  t=np.stack([np.column_stack([ci,b]) for b in boundary],axis=1);all.append(t)
 return np.vstack([nodes]+centers),np.stack(all).astype(np.int32)
rows=[]
material=np.load(ROOT/'dataset/processed/anatomy_retarget/leg_volume_coordinates/production/vessels/vessel_material_coordinates.npz',allow_pickle=True)
for side in ['right','left']:
 for method in plan['centerline_candidates']:
  started=time.perf_counter();a,v,c,theta,h,axis=geometry(side,method);nodes,tblock=star_prisms(v,c,a.skin_faces,plan['radial_fractions']);f=a.skin_faces;det=tet_determinants(nodes,tblock.reshape(-1,4)).reshape(tblock.shape[:-1]);tx=theta[f];tx=tx[:,0,None]+(tx-tx[:,0,None]+np.pi)%(2*np.pi)-np.pi;hx=h[f];area=(tx[:,1]-tx[:,0])*(hx[:,2]-hx[:,0])-(tx[:,2]-tx[:,0])*(hx[:,1]-hx[:,0]);prekeep=(np.min(det,axis=(0,2))>1.e-18)&(area>1.e-14);keptids=np.flatnonzero(prekeep);coords=np.stack([hx[prekeep],tx[prekeep]-2*np.pi*np.floor(tx[prekeep].mean(1)/(2*np.pi))[:,None]],axis=2);poly=shapely.polygons(coords);n=len(poly);pp=np.concatenate([shapely.transform(poly,lambda q:q+np.array([0.,shift])) for shift in [-2*np.pi,0,2*np.pi]]);pairs=STRtree(pp).query(pp,predicate='intersects');pairs=pairs[:,pairs[0]<pairs[1]];aa=shapely.area(shapely.intersection(pp[pairs[0]],pp[pairs[1]]));proper=pairs[:,aa>1.e-12];excluded=np.unique(proper%n);keep=prekeep.copy();keep[keptids[excluded]]=False
  src=np.load(PRIOR/f'ownership_smooth/{side}_surface_lbs_weights.npz')['source_skin_vertex_indices'];old=np.load(ROOT/f'dataset/processed/anatomy_retarget/leg_volume_coordinates/production/atlas/atlas_{side}.npz');lookup={tuple(sorted(x)):i for i,x in enumerate(old['skin_faces'])};oldfaceids=np.array([lookup[tuple(sorted(row))] for row in src[f]]);supported=set(oldfaceids[keep]);counts={}
  for label in np.unique(np.asarray(material['labels'],str)[np.asarray(material['side'],str)==side]):
   mask=np.asarray(material['labels'],str)==label;ids=material['surface_face_indices'][mask];missing=[int(i) for i in np.flatnonzero(mask) if int(material['surface_face_indices'][i]) not in supported];counts[str(label)]={'count':len(ids),'supported':sum(int(i) in supported for i in ids),'missing_global_point_indices':missing}
  row=dict(side=side,method=method,all_faces=len(f),positive_star_faces=int(sum(np.min(det,axis=(0,2))>1.e-18)),positive_geom_and_material_faces=int(sum(prekeep)),periodic_overlap_pairs_before_filter=proper.shape[1],faces_removed_for_overlap=len(excluded),accepted_faces=int(sum(keep)),min_det_accepted=float(det[:,keep].min()),template_coverage=counts,wall_s=time.perf_counter()-started);rows.append(row);(BASE/'star_prism_screen.json').write_text(json.dumps(rows,indent=2));print(row,flush=True);np.savez_compressed(BASE/f'{side}_{method}_star_candidate.npz',vertices=nodes,tets=tblock[:,keep].reshape(-1,4),source_face_mask=keep,theta=theta,h=h,centers=c,source_vertex_indices=src,source_old_face_indices=oldfaceids)
