"""Independent fixed asset geometry/SAT/SMPL-X audit; production is read-only."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,time
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
HERE=Path(__file__).resolve().parent
SOURCE=Path('/media/camp/EXT_DRIVE/Among_US')
sys.path[:0]=[str(SOURCE),str(SOURCE/'src')]
from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import joint_global_transforms
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import _axis_point_and_tangent
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.harmonic import medial_point_at_station
CANON=SOURCE/'outputs/anatomy_retarget/recovered_smplx_20261108/smpl_canonical_weights.npz'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def dets(V,T):
 out=[]
 for start in range(0,len(T),100000):
  p=V[T[start:start+100000]];out.append(np.linalg.det(np.moveaxis(p[:,1:]-p[:,:1],1,2)))
 return np.concatenate(out)
def scaled_jacobian(V,T,orientation):
 values=[]
 for start in range(0,len(T),100000):
  pp=V[T[start:start+100000]];det=np.linalg.det(np.moveaxis(pp[:,1:]-pp[:,:1],1,2));den=[]
  for corner in range(4):den.append(np.prod(np.linalg.norm(np.delete(pp,corner,axis=1)-pp[:,corner:corner+1],axis=2),axis=1))
  values.append(orientation*det/np.maximum(np.max(den,axis=0),1e-300))
 z=np.concatenate(values)
 return {'definition':'oriented det / max corner product of three edge lengths; regular tet=1/sqrt(2)', 'min':float(min(z)),'quantiles':np.quantile(z,[0,.0001,.001,.01,.5,.99,1]).tolist(),'nonpositive':int(sum(z<=0)),'diagnostic_le_1e_6':int(sum(z<=1e-6)),'diagnostic_le_1e_3':int(sum(z<=1e-3)),'diagnostic_le_1e_2':int(sum(z<=1e-2))}

def volume_metrics(V,T,h,ref=None):
 d=dets(V,T);sgn=-1 if np.sum(d<0)>np.sum(d>0) else 1
 out={'tet_count':len(T),'positive_count':int(sum(d>1e-18)),'negative_count':int(sum(d< -1e-18)),'degenerate_count':int(sum(abs(d)<=1e-18)),'modal_orientation':sgn,'opposite_modal_orientation_count':int(sum(d*sgn< -1e-18)),'determinant_min_max':[float(min(d)),float(max(d))],'signed_volume_sum_m3':float(d.sum()/6),'absolute_volume_sum_m3':float(abs(d).sum()/6)}
 out['corner_scaled_jacobian']=scaled_jacobian(V,T,sgn)
 bins=np.floor(np.mean(h[T],axis=1)*10000).astype(int);bad=np.flatnonzero(d*sgn< -1e-18);uu,cc=np.unique(bins[bad],return_counts=True);out['opposite_orientation_mean_h_bins']={str(u/10000):int(c) for u,c in zip(uu,cc)}
 if ref is not None:
  keep=abs(ref)>1e-18;ratio=d[keep]/ref[keep];out['pose_det_ratio']={'min':float(min(ratio)),'quantiles':np.quantile(ratio,[0,.001,.01,.5,.99,1]).tolist(),'nonpositive_count':int(sum(ratio<=0)),'near_collapse_count_le_1e_6':int(sum(ratio<=1e-6))}
 return out,d

def binfile(path,V,T,h):
 low,high=float(min(h)),float(max(h));layer=np.floor((np.mean(h[T],axis=1)-low)/max(high-low,1e-12)*71).astype('<i4')
 with Path(path).open('wb') as f:
  np.array([len(V),len(T)],dtype='<u8').tofile(f);np.asarray(V,dtype='<f8').tofile(f);np.asarray(T,dtype='<u4').tofile(f);layer.tofile(f)
def lp_witness(V,T,pair):
 A=[];b=[]
 for tid in pair[:2]:
  pp=V[T[int(tid)]]
  for opposite in range(4):
   tri=np.delete(pp,opposite,axis=0);n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);n/=np.linalg.norm(n)
   if n@(pp[opposite]-tri[0])>0:n=-n
   A.append(np.r_[n,1.]);b.append(n@tri[0])
 result=linprog([0,0,0,-1],A_ub=A,b_ub=b,bounds=[(None,None)]*3+[(0,None)],method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
 slack=None if not result.success else float(np.min(np.asarray(b)-np.asarray(A)[:,:3]@result.x[:3]))
 return {'tet_ids':[int(x) for x in pair[:2]],'success':bool(result.success),'measured_min_face_slack_m':slack,'verified_strict_interior':bool(result.success and slack>1e-12),'interior_ball_radius_m':None if not result.success else float(result.x[3]),'xyz':None if not result.success else result.x[:3].tolist()}
def overlaps(label,V,T,h):
 path=HERE/(label+'.bin');binfile(path,V,T,h);t=time.perf_counter();result=subprocess.run([str(HERE/'tet_overlap'),str(path)],check=True,text=True,capture_output=True,timeout=1200);out=json.loads(result.stdout);out['wall_seconds']=time.perf_counter()-t
 out['independent_lp_witnesses']=[lp_witness(V,T,pair) for pair in out['examples'][:64]];save(HERE/(label+'_overlap.json'),out);path.unlink();return out

def surface_stats(V,F):
 ee=np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]));uni,inv,ct=np.unique(np.sort(ee,axis=1),axis=0,return_inverse=True,return_counts=True);bd=uni[ct==1];degree=np.bincount(bd.ravel(),minlength=len(V));wind=np.bincount(inv,weights=np.where(ee[:,0]<ee[:,1],1,-1),minlength=len(uni));parent=np.arange(len(V))
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 for a,b in bd:parent[find(a)]=find(b)
 comp={}
 for v in np.unique(bd):comp.setdefault(find(v),[]).append(int(v))
 return {'vertices':len(V),'faces':len(F),'boundary_edges':len(bd),'boundary_components':len(comp),'boundary_component_vertices':[len(x) for x in comp.values()],'boundary_degrees_not_two':int(sum(degree[np.unique(bd)]!=2)),'nonmanifold_edges':int(sum(ct>2)),'inconsistent_interior_winding':int(sum((ct==2)&(wind!=0))),'unused_vertices':int(len(V)-len(np.unique(F))),'watertight':bool(np.all(ct==2)),'boundary_vertex_ids':[x for x in comp.values()]}

def section_audit(asset):
 import trimesh
 V=asset['skin_vertices'].astype(float);F=asset['skin_faces'];mesh=trimesh.Trimesh(V,F,process=False);hvalues=np.linspace(float(min(asset['harmonic_h'])),float(max(asset['harmonic_h'])),72);theta=np.linspace(0,2*np.pi,108,endpoint=False);records=[];allsegments=[]
 for si,s in enumerate(hvalues):
  axis,tangent=_axis_point_and_tangent(asset['hip'],asset['knee'],asset['ankle'],s);core=medial_point_at_station(asset['core_h'],asset['core_points'],s).astype(float);ref=asset['pelvis']-asset['hip'];e1=ref-(ref@tangent)*tangent;e1=e1/np.linalg.norm(e1);e2=np.cross(tangent,e1);e2/=np.linalg.norm(e2)
  seg=trimesh.intersections.mesh_plane(mesh,np.asarray(tangent,float),core);allsegments.append(seg);rel=seg-core;uv=np.stack([rel@e1,rel@e2],axis=-1)
  counts=[];rays=[]
  for angle in theta:
   direction=np.array([np.cos(angle),np.sin(angle)]);aa=uv[:,0];delta=uv[:,1]-aa;cross=lambda a,b:a[...,0]*b[...,1]-a[...,1]*b[...,0];den=cross(direction,delta);good=abs(den)>1e-14
   ray=np.zeros(len(seg));u=np.zeros(len(seg));ray[good]=cross(aa[good],delta[good])/den[good];u[good]=cross(aa[good],direction)/den[good]
   rr=np.sort(ray[good&(ray>1e-10)&(u>=-1e-10)&(u<=1+1e-10)]);rr=rr[np.r_[True,np.diff(rr)>1e-8]] if len(rr) else rr;counts.append(len(rr));rays.append(rr.tolist())
  # Section connectivity is reconstructed from quantized segment endpoints;
  # no angular sorting is used to invent edges.
  if len(seg):
   ids,inv=np.unique(np.rint(seg.reshape(-1,3)/1e-8).astype(np.int64),axis=0,return_inverse=True);edges=inv.reshape(-1,2);dd=np.bincount(edges.ravel(),minlength=len(ids));bad=int(sum(dd!=2))
  else:bad=0
  records.append({'station_index':si,'station':float(s),'origin':core.tolist(),'plane_normal':tangent.tolist(),'segments':len(seg),'section_vertices_degree_not2':bad,'zero_hit_rays':counts.count(0),'one_hit_rays':counts.count(1),'multi_hit_rays':sum(x>1 for x in counts),'positive_radii_by_ray':rays})
 return {'planes':72,'rays_per_plane':108,'plane_source':'original intended geometric station core and piecewise bone tangent; actual skin triangle intersection','zero_hit_rays':sum(r['zero_hit_rays'] for r in records),'one_hit_rays':sum(r['one_hit_rays'] for r in records),'multi_hit_rays':sum(r['multi_hit_rays'] for r in records),'open_or_branched_section_planes':sum(r['section_vertices_degree_not2']>0 for r in records),'records':records},allsegments

def fixed_volume_weights(asset,side,stencil_path):
 c=np.load(CANON,allow_pickle=True);st=np.load(stencil_path,allow_pickle=False);S=csr_matrix((st['stencil_weights'],st['stencil_indices'],st['stencil_indptr']),shape=(len(st['stencil_indptr'])-1,len(st['source_full_vertex_indices'])));surfaceW=np.asarray(S@c['lbs_weights'][st['source_full_vertex_indices']]);skin=asset['skin_vertices'].astype(float)
 if len(surfaceW)!=len(skin) or not np.array_equal(st['surface_faces'],asset['skin_faces']):raise ValueError('surface stencil does not match asset topology')
 if min(surfaceW.ravel()) < -1e-10:raise ValueError('surface stencil creates negative skin weights; do not silently clip')
 dist,idx=cKDTree(skin).query(asset['harmonic_vertices'],k=6);a=1/(dist*dist+1e-8);a/=a.sum(1)[:,None];W=np.sum(a[:,:,None]*surfaceW[idx],axis=1);W/=W.sum(1)[:,None]
 return W,c,{'source':'fixed same-leg k6 nearest-skin inverse-square proxy using actual canonical SMPL-X and surface CSR lineage','canonical_file':str(CANON),'canonical_sha256':sha(CANON),'stencil_file':str(stencil_path),'stencil_sha256':sha(stencil_path),'nearest_skin_distance_max_m':float(max(dist[:,0])),'weight_min':float(W.min()),'row_sum_max_error':float(max(abs(W.sum(1)-1))), 'volume_weights_sha256':hashlib.sha256(W.tobytes()).hexdigest(),'limitation':'skin-derived volume weights are a proxy; no native volume skinning claim'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--asset',type=Path,required=True);p.add_argument('--side',choices=['left','right'],required=True);p.add_argument('--tag',required=True);p.add_argument('--stencil',type=Path);p.add_argument('--volume-weights',type=Path);p.add_argument('--no-poses',action='store_true');p.add_argument('--pose-degrees',type=int,nargs='+',default=[0,30,60]);a=p.parse_args();asset=np.load(a.asset,allow_pickle=True);V=asset['harmonic_vertices'].astype(float);T=asset['harmonic_tets'];h=asset['harmonic_h'].astype(float);label=a.tag+'_'+a.side
 metrics,ref=volume_metrics(V,T,h);quantized=V.astype(np.float32).astype(float);qd=dets(quantized,T);metrics['float32_roundtrip']={'max_node_displacement_m':float(np.max(np.linalg.norm(quantized-V,axis=1))),'sign_flip_relative_stored_count':int(sum(qd*ref<0)),'degenerate_count':int(sum(abs(qd)<=1e-18)),'modal_opposite_count':int(sum(qd*metrics['modal_orientation']< -1e-18))};out={'asset':str(a.asset),'asset_sha256':sha(a.asset),'side':a.side,'protocol_sha256':sha(HERE/'PROTOCOL.txt'),'input_vertex_dtype':str(asset['harmonic_vertices'].dtype),'canonical_volume':metrics,'skin':surface_stats(asset['skin_vertices'],asset['skin_faces']),'poses':{}};save(HERE/(label+'_audit.json'),out);section,segs=section_audit(asset);out['sections']=section;np.savez_compressed(HERE/(label+'_sections.npz'),**{f'plane_{i}':s for i,s in enumerate(segs)});out['canonical_overlap']=overlaps(label+'_canonical',V,T,h);save(HERE/(label+'_audit.json'),out)
 if not a.no_poses:
  if a.volume_weights is not None:
   wd=np.load(a.volume_weights,allow_pickle=True);key='weights' if 'weights' in wd else 'volume_lbs_weights';W=np.asarray(wd[key],dtype=float);canon=np.load(CANON,allow_pickle=True)
   if W.shape!=(len(V),55) or np.min(W)< -1e-10 or max(abs(W.sum(1)-1))>1e-5:raise ValueError('invalid sealed volume weight contract')
   wm={'source':'sealed candidate-provided volume LBS weights; fixed across all poses','file':str(a.volume_weights),'sha256':sha(a.volume_weights),'canonical_file':str(CANON),'canonical_sha256':sha(CANON),'weight_min':float(W.min()),'row_sum_max_error':float(max(abs(W.sum(1)-1))),'volume_weights_sha256':hashlib.sha256(W.tobytes()).hexdigest()}
  else:
   if a.stencil is None:raise ValueError('explicit matching skin stencil required for posed audit')
   W,canon,wm=fixed_volume_weights(asset,a.side,a.stencil)
  out['volume_weight_contract']=wm;np.savez_compressed(HERE/(label+'_fixed_volume_weights.npz'),weights=W);joint_names=[str(x) for x in canon['joint_names']];knee=joint_names.index(a.side+'_knee')
  for degrees in a.pose_degrees:
   pose=np.zeros((55,3));pose[knee,0]=np.deg2rad(degrees);tf=joint_global_transforms(pose_axis_angle=pose,rest_joints=canon['rest_joints'],parents=canon['parents'])@canon['inverse_bind'];posed=np.einsum('nj,jab,nb->na',W,tf[:,:3],np.c_[V,np.ones(len(V))]);m,_=volume_metrics(posed,T,h,ref);m['max_node_displacement_from_rest_m']=float(np.max(np.linalg.norm(posed-V,axis=1)));m['overlap']=overlaps(label+f'_pose{degrees}',posed,T,h);m['knee_joint_index']=knee;out['poses'][str(degrees)]=m;save(HERE/(label+'_audit.json'),out);print(label,'pose',degrees,'opposite',m['opposite_modal_orientation_count'],'overlaps',m['overlap']['positive_interior_overlap_pairs'],flush=True)

 save(HERE/(label+'_audit.json'),out)
if __name__=='__main__':main()
