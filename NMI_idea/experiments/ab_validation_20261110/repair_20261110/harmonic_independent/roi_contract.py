"""Independent closed-boundary, attachment and strict inverse/domain checks."""
from pathlib import Path
import argparse,json,time,numpy as np
from audit import HERE,SOURCE,CANON,sha,surface_stats,scaled_jacobian
from skin_intersections import check
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.volume_refine import VolumeTetRefiner
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.harmonic import _distance_to_skin_surface

def boundary(T,V):
 faces=np.vstack((T[:,[1,2,3]],T[:,[0,3,2]],T[:,[0,1,3]],T[:,[0,2,1]]));u,ix,ct=np.unique(np.sort(faces,axis=1),axis=0,return_index=True,return_counts=True);return faces[ix[ct==1]],int(sum(ct>2))

def main():
 p=argparse.ArgumentParser();p.add_argument('--asset',type=Path,required=True);p.add_argument('--attachments',type=Path,required=True);p.add_argument('--source-skin',type=Path,required=True);p.add_argument('--tag',required=True);a=p.parse_args();d=np.load(a.asset,allow_pickle=True);w=np.load(a.attachments,allow_pickle=True);s=np.load(a.source_skin,allow_pickle=True);V=d['harmonic_vertices'].astype(float);T=d['harmonic_tets'];F,nonmani=boundary(T,V);out={'volume_refine_source_sha256':sha(SOURCE/'src/projects/genesis_ue_sync/anatomy_retarget/leg_volume_coordinates/volume_refine.py'),'asset_sha256':sha(a.asset),'attachment_sha256':sha(a.attachments),'source_skin_sha256':sha(a.source_skin),'complete_tet_boundary_topology':surface_stats(V,F),'interior_nonmanifold_tet_faces':nonmani,'complete_tet_boundary_cgal':check(V,F,np.ones(len(F),int))}
 faces=s['skin_faces'][w['source_surface_face_indices']];expected=np.sum(w['source_surface_barycentric'][:,:,None]*s['skin_vertices'][faces].astype(float),axis=1);delta=np.linalg.norm(d['skin_vertices']-expected,axis=1);outer=d['skin_vertices'];OF=d['skin_faces'];centers=outer[OF].mean(1);dist=_distance_to_skin_surface(centers,s['skin_vertices'],s['skin_faces']);out['surface_binding']={'count':len(delta),'maximum_vertex_barycentric_error_m':float(max(delta)),'mean_error_m':float(np.mean(delta)),'source_bary_min':float(w['source_surface_barycentric'].min()),'source_bary_rowsum_max_error':float(max(abs(w['source_surface_barycentric'].sum(1)-1))),'outer_face_centroids_to_actual_source_triangles':{'count':len(dist),'mean_m':float(np.mean(dist)),'p95_m':float(np.quantile(dist,.95)),'max_m':float(max(dist))},'limit':'all vertices/centroids checked; not certified complete surface Hausdorff bound'}
 if 'source_owned_surface_weights' in w:
  from scipy.sparse import csr_matrix
  side=str(d['side']);stencil=SOURCE/f'dataset/processed/anatomy_retarget/leg_volume_coordinates/production/material/surface_stencil_{side}.npz';st=np.load(stencil);ca=np.load(CANON,allow_pickle=True);S=csr_matrix((st['stencil_weights'],st['stencil_indices'],st['stencil_indptr']),shape=(len(st['stencil_indptr'])-1,len(st['source_full_vertex_indices'])));oldW=np.asarray(S@ca['lbs_weights'][st['source_full_vertex_indices']]);ownedW=oldW[w['source_original_skin_vertex_indices']];source_error=float(np.max(abs(ownedW-w['source_owned_surface_weights'])));expectW=np.sum(w['source_surface_barycentric'][:,:,None]*ownedW[faces],axis=1);out['surface_weight_lineage']={'original_production_stencil_sha256':sha(stencil),'canonical_weights_sha256':sha(CANON),'max_source_owned_weight_difference':source_error,'max_outer_barycentric_weight_difference':float(np.max(abs(expectW-w['surface_lbs_weights'])))}
 shape=tuple(map(int,w['grid_shape']));ns,nr,nt=shape;W=w['volume_lbs_weights'].reshape(ns,nr,nt,55);core=w['core_lbs_weights'];ow=w['surface_lbs_weights'].reshape(ns,nt,55);r=w['radial_fractions'];WW=(1-r[None,:,None,None])*core[:,None,None,:]+r[None,:,None,None]*ow[:,None,:,:];out['volume_weight_rule']={'max_difference_from_frozen_radial_convex_combination':float(np.max(abs(W-WW))),'core_ring_mean_difference':float(np.max(abs(core-ow.mean(1)))),'all_volume_row_sum_max_error':float(np.max(abs(W.sum(-1)-1))),'all_volume_min':float(W.min())}
 # Independent material Jacobian with periodic theta edge deltas.
 X=V[T];C=np.stack((d['harmonic_theta'][T],d['harmonic_h'][T],d['harmonic_d'][T]),axis=-1).astype(float);E=C[:,1:]-C[:,:1];E[:,:,0]=np.angle(np.exp(1j*E[:,:,0]));detX=np.linalg.det(np.moveaxis(X[:,1:]-X[:,:1],1,2));J=np.linalg.det(np.moveaxis(E,1,2))/detX;out['material_jacobian']={'min':float(min(J)),'max':float(max(J)),'positive':int(sum(J>0)),'negative':int(sum(J<0)),'zero':int(sum(J==0)),'limit':'local piecewise coordinate test with periodic angle; not global inverse certification'}
 atlas=load_leg_volume_atlas(a.asset);mapper=VolumeTetRefiner.from_atlas(atlas);rng=np.random.default_rng(41109);ids=rng.integers(0,len(T),1024);bary=rng.dirichlet(np.ones(4),1024);points=np.einsum('ni,nij->nj',bary,V[T[ids]]);started=time.perf_counter();xi=mapper.p_to_xi(points);back=mapper.xi_to_p(xi);error=np.linalg.norm(points-back,axis=1);xi2=mapper.p_to_xi(back);xd=xi2-xi;xd[:,0]=np.angle(np.exp(1j*xd[:,0]));out['production_inverse_roundtrip']={'sample_count':1024,'seed':41109,'reference_points_supplied':False,'max_position_error_m':float(max(error)),'p95_position_error_m':float(np.quantile(error,.95)),'mean_position_error_m':float(np.mean(error)),'max_coordinate_error':np.max(abs(xd),axis=0).tolist(),'wall_seconds':time.perf_counter()-started}
 # Explicitly test outside targets; all results retained, no silent fallback.
 lo,hi=float(min(d['harmonic_h'])),float(max(d['harmonic_h']));targets=np.array([[0,lo-.04,.5],[1,hi+.04,.5],[2,(lo+hi)/2,-.1],[3,(lo+hi)/2,1.1]],float);outside=[]
 for q in targets:
  try:
   pp=mapper.xi_to_p(q[None]);rec={'xi':q.tolist(),'inverse_rejected':False,'returned_point':pp[0].tolist()}
   try:qx=mapper.p_to_xi(pp);rec.update(returned_forward_rejected=False,returned_actual_xi=qx[0].tolist())
   except Exception as e:rec.update(returned_forward_rejected=True,forward_exception=type(e).__name__,forward_message=str(e))
   outside.append(rec)
  except Exception as e:outside.append({'xi':q.tolist(),'inverse_rejected':True,'exception':type(e).__name__,'message':str(e)})
 out['outside_inverse_domain']=outside
 seam=np.array([[v,(lo+hi)/2,.5] for v in [-2*np.pi,0,2*np.pi,4*np.pi]],float);seam_p=mapper.xi_to_p(seam);out['periodic_inverse_seam']={'angles':seam[:,0].tolist(),'max_position_difference_m':float(np.max(np.linalg.norm(seam_p-seam_p[1],axis=1))),'points':seam_p.tolist()}
 grid=V.reshape(ns,nr,nt,3);frames=w['frames'];pp=[grid[0,nr//2,0]-.04*frames[0,:,2],grid[-1,nr//2,0]+.04*frames[-1,:,2]];outsideP=[]
 for q in pp:
  try:xx=mapper.p_to_xi(q[None]);outsideP.append({'point':q.tolist(),'rejected':False,'xi':xx[0].tolist()})
  except Exception as e:outsideP.append({'point':q.tolist(),'rejected':True,'exception':type(e).__name__,'message':str(e)})
 out['outside_forward_domain']=outsideP;(HERE/(a.tag+'_contract.json')).write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
