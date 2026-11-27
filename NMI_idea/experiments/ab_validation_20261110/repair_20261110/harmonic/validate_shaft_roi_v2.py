from pathlib import Path
from dataclasses import replace
import sys,numpy as np,json,time,igl,trimesh
from scipy.spatial import cKDTree
ROOT=Path('/media/camp/EXT_DRIVE/Among_US');OUT=Path(__file__).resolve().parent;TARGET=OUT/'shaft_roi_v2';sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas,query_atlas_coordinates
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.volume_refine import VolumeTetRefiner
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.shaft_rebuild import tet_determinants
from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import joint_global_transforms
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.lbs_bridge import blend_lbs_transforms
CANON=ROOT/'outputs/anatomy_retarget/recovered_smplx_20261108';native=np.load(CANON/'smpl_canonical_weights.npz')

def stats(x):
 x=np.asarray(x,float);return dict(count=x.size,min=float(x.min()),mean=float(x.mean()),p95=float(np.percentile(x,95)),max=float(x.max()))
def error(a,b):
 e=np.asarray(a,float)-b;e[:,0]=(e[:,0]+np.pi)%(2*np.pi)-np.pi;return np.max(abs(e),axis=0).tolist()
results={}
for side in ['right','left']:
 start=time.perf_counter();a=load_leg_volume_atlas(TARGET/f'atlas_{side}.npz');at=np.load(TARGET/f'attachments_{side}.npz');shape=tuple(at['grid_shape']);grid=np.arange(len(a.harmonic_vertices)).reshape(shape);skinids=grid[:,-1].ravel();innerids=grid[:,0].ravel();d=a.harmonic_d.reshape(shape);h=a.harmonic_h.reshape(shape);w=at['volume_lbs_weights'];corew=at['core_lbs_weights'];reference=load_leg_volume_atlas(OUT/f'ownership_smooth/owned_smooth_{side}_skin.npz')
 source_tri=reference.skin_vertices[reference.skin_faces[at['source_surface_face_indices']]];source_point=np.einsum('ni,nij->nj',at['source_surface_barycentric'],source_tri)
 outer_dist,_,_=igl.point_mesh_squared_distance(a.skin_vertices.astype(float),reference.skin_vertices.astype(float),reference.skin_faces.astype(np.int64))
 rng=np.random.default_rng(20261110);faceids=rng.choice(len(a.skin_faces),4096,replace=False);fb=rng.dirichlet(np.ones(3),len(faceids));facepoints=np.einsum('ni,nij->nj',fb,a.skin_vertices[a.skin_faces[faceids]])
 face_dist,_,_=igl.point_mesh_squared_distance(facepoints,reference.skin_vertices.astype(float),reference.skin_faces.astype(np.int64))
 refiner=VolumeTetRefiner.from_atlas(a);tids=rng.choice(len(a.harmonic_tets),256,replace=False);bary=rng.dirichlet(np.ones(4),len(tids));points=np.einsum('ni,nij->nj',bary,a.harmonic_vertices[a.harmonic_tets[tids]])
 xi=query_atlas_coordinates(a,points)[0];decoded=refiner.xi_to_p(xi);decode_error=np.linalg.norm(decoded-points,axis=1)
 qskin=query_atlas_coordinates(a,a.skin_vertices)[0];qinner=query_atlas_coordinates(a,a.harmonic_vertices[innerids])[0]
 seam=np.array([[theta,hh,.5] for hh in [.25,.5,.8] for theta in [0.,2*np.pi,1.e-6,2*np.pi-1.e-6]],np.float32);sp=refiner.xi_to_p(seam).reshape(3,4,3);seam_equal=np.linalg.norm(sp[:,0]-sp[:,1],axis=1);seam_near=np.linalg.norm(sp[:,2]-sp[:,3],axis=1)
 domains={}
 for name,p in [('far',np.array([[5.,5.,5.]])),('excluded_centerline',a.core_points[[len(a.core_points)//2]]),('outside_proximal_h',a.core_points[[0]]-at['frames'][[0],:,2]*.1)]:
  try:query_atlas_coordinates(a,p);domains[name]='ACCEPTED'
  except ValueError as e:domains[name]=str(e)
 canonical_det=tet_determinants(a.harmonic_vertices,a.harmonic_tets);pose_rows=[];homo=np.column_stack([a.harmonic_vertices,np.ones(len(w),np.float32)]).astype(np.float32)
 for angle in range(0,91,15):
  pose=np.zeros((55,3),np.float32);pose[5 if side=='right' else 4,0]=np.deg2rad(angle)
  global_tf=joint_global_transforms(pose_axis_angle=pose.ravel(),rest_joints=native['rest_joints'],parents=native['parents']);tf=global_tf@native['inverse_bind'];blend=blend_lbs_transforms(w,tf);posed=np.einsum('nij,nj->ni',blend,homo)[:,:3].astype(np.float32);det=tet_determinants(posed,a.harmonic_tets)
  coreblend=blend_lbs_transforms(corew,tf);coreposed=np.einsum('nij,nj->ni',coreblend,np.column_stack([a.core_points,np.ones(len(corew),np.float32)]))[:,:3].astype(np.float32)
  radial_distance=np.linalg.norm(posed.reshape(*shape,3)-coreposed[:,None,None],axis=-1)
  delta=np.diff(posed.reshape(*shape,3),axis=0);long_direction=np.diff(coreposed,axis=0);long_direction/=np.linalg.norm(long_direction,axis=1)[:,None];horder=np.einsum('srtj,sj->srt',delta,long_direction)
  posed_atlas=replace(a,harmonic_vertices=posed,skin_vertices=posed[skinids],core_points=coreposed)
  posed_points=np.einsum('ni,nij->nj',bary,a.harmonic_vertices[a.harmonic_tets[tids]]) if angle==0 else np.einsum('ni,nij->nj',bary,posed[a.harmonic_tets[tids]])
  try:pose_xi=query_atlas_coordinates(posed_atlas,posed_points)[0];coorderr=error(pose_xi,xi);query_status='passed'
  except ValueError as e:coorderr=None;query_status=str(e)
  row=dict(knee_deg=angle,nonpositive_tets=int(sum(det<=0)),pose_over_rest_det_min=float((det/canonical_det).min()),radial_distance_order_nonincreasing=int(sum((np.diff(radial_distance,axis=1)<=0).ravel())),neighbor_h_layer_order_nonpositive=int(sum((horder<=0).ravel())),material_label_coordinate_abs_error_max=coorderr,query_status=query_status)
  pose_rows.append(row);np.savez_compressed(TARGET/f'{side}_pose_{angle:02d}.npz',vertices=posed,core_points=coreposed)
 r=dict(radial_depth_step_min=float(np.min(-np.diff(d,axis=1))),radial_depth_nonstrict_ray_count=int(np.sum(np.any(np.diff(d,axis=1)>=0,axis=1))),h_label_step_min=float(np.min(np.diff(h,axis=0))),
  outer_source_triangle_barycentric_identity_error_m=stats(np.linalg.norm(a.skin_vertices-source_point,axis=1)),outer_nodes_to_original_owned_skin_m=stats(np.sqrt(outer_dist)),loft_4096_triangle_interior_to_original_owned_skin_m=stats(np.sqrt(face_dist)),
  boundary_outer_abs_d_max=float(abs(qskin[:,2]).max()),boundary_inner_abs_d_minus1_max=float(abs(qinner[:,2]-1).max()),unassisted_256_native_tet_roundtrip_error_m=stats(decode_error),periodic_seam_0_vs_2pi_position_error_m=stats(seam_equal),periodic_seam_plusminus_1e6rad_gap_m=stats(seam_near),domain_rejection=domains,pose_sweep=pose_rows,wall_s=time.perf_counter()-start)
 results[side]=r;(TARGET/'validation_metrics.json').write_text(json.dumps(results,indent=2));print(side,json.dumps(r,indent=2),flush=True)
 assert np.min(-np.diff(d,axis=1))>0 and np.min(np.diff(h,axis=0))>0
 assert decode_error.max()<1.e-5 and abs(qskin[:,2]).max()<1.e-6 and abs(qinner[:,2]-1).max()<1.e-6
 assert seam_equal.max()<1.e-6 and all(v!='ACCEPTED' for v in domains.values())
