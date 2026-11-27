from pathlib import Path
import sys,json,time,hashlib
import numpy as np
from scipy.interpolate import PchipInterpolator,UnivariateSpline
ROOT=Path('/media/camp/EXT_DRIVE/Among_US');BASE=Path(__file__).resolve().parent;PRIOR=BASE.parent/'harmonic';sys.path.insert(0,str(ROOT/'src'))
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.shaft_rebuild import tet_determinants
PLAN=dict(candidate_order=['skeletal_axis','mean_core_axis','linear_fit_core','pchip_core','smooth_core_1e4'],radial_fractions=[.04,.36,.68,1.],skin='CGAL-clean owned original production smooth skin; all original XYZ/facets preserved',selection='must pass both-sided all native float32 tet determinants, local material theta/axis-h orientation, then independent global/pose gates; no candidate is accepted by this screen alone')
(BASE/'frozen_surface_layer_plan.json').write_text(json.dumps(PLAN,indent=2))

def geometry(side,method):
 a=load_leg_volume_atlas(PRIOR/f'ownership_smooth/owned_smooth_{side}_skin.npz');core=load_leg_volume_atlas(PRIOR/f'rebuild_work/owned_base/atlas_{side}.npz');v=a.skin_vertices.astype(float);axis=(a.ankle-a.hip).astype(float);axis/=np.linalg.norm(axis);ref=(a.pelvis-a.hip).astype(float);e1=ref-axis*(ref@axis);e1/=np.linalg.norm(e1);e2=np.cross(axis,e1);origin=a.hip.astype(float)
 s=(v-origin)@axis;cp=core.core_points.astype(float);cs=(cp-origin)@axis;order=np.argsort(cs);cs,idx=np.unique(cs[order],return_index=True);cp=cp[order][idx];uv=np.column_stack([(cp-origin)@e1,(cp-origin)@e2])
 if method=='skeletal_axis':cuv=np.zeros((len(s),2))
 elif method=='mean_core_axis':cuv=np.broadcast_to(uv.mean(0),(len(s),2)).copy()
 elif method=='linear_fit_core':cuv=np.column_stack([np.polyval(np.polyfit(cs,uv[:,i],1),s) for i in [0,1]])
 elif method=='pchip_core':cuv=PchipInterpolator(cs,uv,axis=0)(np.clip(s,cs[0],cs[-1]))
 elif method=='smooth_core_1e4':cuv=np.column_stack([UnivariateSpline(cs,uv[:,i],k=3,s=1.e-4)(np.clip(s,cs[0],cs[-1])) for i in [0,1]])
 else:raise ValueError(method)
 centers=origin+s[:,None]*axis+cuv[:,0,None]*e1+cuv[:,1,None]*e2
 angles=np.mod(np.arctan2((v-centers)@e2,(v-centers)@e1),2*np.pi);h=(s-s.min())/(s.max()-s.min())
 return a,v,centers,angles,h,axis

def prism(v,centers,faces,fractions):
 nodes=(centers[None]+np.asarray(fractions)[:,None,None]*(v-centers)[None]).reshape(-1,3).astype(np.float32)
 nv=len(v);sf=np.sort(faces,axis=1);perm=np.argsort(faces,axis=1);odd=((perm[:,0]>perm[:,1]).astype(int)+(perm[:,0]>perm[:,2])+(perm[:,1]>perm[:,2]))%2==1;tets=[]
 for i in range(len(fractions)-1):
  a,b,c=(sf+i*nv).T;A,B,C=(sf+(i+1)*nv).T;t=np.stack([np.column_stack([a,b,c,C]),np.column_stack([a,b,C,B]),np.column_stack([a,A,B,C])],axis=1);t[odd,:,0],t[odd,:,1]=t[odd,:,1].copy(),t[odd,:,0].copy();tets.append(t.reshape(-1,4))
 return nodes,np.vstack(tets).astype(np.int32)

if __name__=='__main__':
 results=[]
 for side in ['right','left']:
  for method in PLAN['candidate_order']:
   started=time.perf_counter();a,v,c,theta,h,axis=geometry(side,method);nodes,tets=prism(v,c,a.skin_faces,PLAN['radial_fractions']);det=tet_determinants(nodes,tets);f=a.skin_faces;normal=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]);rad=(v-c)[f].mean(1);cos=np.einsum('ij,ij->i',normal,rad)/(np.linalg.norm(normal,axis=1)*np.linalg.norm(rad,axis=1));tx=theta[f];tx=tx[:,0,None]+(tx-tx[:,0,None]+np.pi)%(2*np.pi)-np.pi;hx=h[f];area=(tx[:,1]-tx[:,0])*(hx[:,2]-hx[:,0])-(tx[:,2]-tx[:,0])*(hx[:,1]-hx[:,0]);row=dict(side=side,method=method,nodes=len(nodes),tets=len(tets),nonpositive_tets=int(sum(det<=0)),det_min=float(det.min()),radial_normal_nonpositive_faces=int(sum(cos<=0)),radial_normal_cos_min=float(cos.min()),material_surface_area_positive=int(sum(area>1.e-14)),material_surface_area_negative=int(sum(area<-1.e-14)),material_surface_degenerate=int(sum(abs(area)<=1.e-14)),wall_s=time.perf_counter()-started);results.append(row);print(row,flush=True);(BASE/'surface_layer_screen.json').write_text(json.dumps(results,indent=2))
