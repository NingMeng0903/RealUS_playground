from pathlib import Path
from dataclasses import replace
import hashlib,json,sys,time
import numpy as np

BASE=Path(__file__).resolve().parent
ASSET=BASE.parent.parent/'ab_validation_20261110/repair_20261110/harmonic/shaft_roi_v2'
sys.path.insert(0,'/media/camp/EXT_DRIVE/Among_US/src')
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas,query_atlas_coordinates

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summary(a):
 a=np.asarray(a);return dict(n=a.size,median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max()))
def unit(x):return x/np.linalg.norm(x,axis=-1,keepdims=True)
def angle(a,b):return np.rad2deg(np.arccos(np.clip(np.einsum('ni,ni->n',unit(a),unit(b)),-1.,1.)))
def mats(p):return np.moveaxis(p[:,1:]-p[:,:1],1,2)

def run():
 rng=np.random.default_rng(720960);output={};arrays={};hashes={}
 for side in ('right','left'):
  a=load_leg_volume_atlas(ASSET/f'atlas_{side}.npz');hashes[str(ASSET/f'atlas_{side}.npz')]=sha(ASSET/f'atlas_{side}.npz')
  ids=rng.choice(len(a.harmonic_tets),256,replace=False);cells=a.harmonic_tets[ids]
  b=.1+.6*rng.dirichlet(np.full(4,2.),len(ids));v0=a.harmonic_vertices[cells].astype(float)
  X0=mats(v0);p0=np.einsum('ni,nij->nj',b,v0)
  xi0=query_atlas_coordinates(a,p0)[0].astype(float)
  labels=np.stack((a.harmonic_theta,a.harmonic_h,a.harmonic_d),-1)[cells].astype(float)
  theta=labels[:,:,0];ss=np.sum(b*np.sin(theta),axis=1);cc=np.sum(b*np.cos(theta),axis=1)
  # Differences are against vertex0, not adjacent pairs.
  theta_b=(cc[:,None]*(np.sin(theta[:,1:])-np.sin(theta[:,:1]))-ss[:,None]*(np.cos(theta[:,1:])-np.cos(theta[:,:1])))/(ss*ss+cc*cc)[:,None]
  Kb=np.stack((theta_b,labels[:,1:,1]-labels[:,:1,1],labels[:,1:,2]-labels[:,:1,2]),axis=1)
  v=unit(rng.normal(size=(len(ids),3)));normal=unit(np.cross(v,rng.normal(size=v.shape)))
  error=rng.normal(size=v.shape);A=np.diag([.1,10.,2.]);rows=[]
  arrays[f'{side}_cell_ids']=ids;arrays[f'{side}_barycentric']=b
  for degree in range(0,91,15):
   posefile=ASSET/f'{side}_pose_{degree:02d}.npz';hashes[str(posefile)]=sha(posefile)
   data=np.load(posefile);posed=data['vertices'].astype(float);vp=posed[cells];Xp=mats(vp)
   p=np.einsum('ni,nij->nj',b,vp);F=Xp@np.linalg.inv(X0)
   at=replace(a,harmonic_vertices=posed,core_points=data['core_points'])
   xi=query_atlas_coordinates(at,p)[0].astype(float);dxi=xi-xi0;dxi[:,0]=(dxi[:,0]+np.pi)%(2*np.pi)-np.pi
   transported_v=np.einsum('nij,nj->ni',F,v)
   transported_n=np.linalg.solve(np.swapaxes(F,1,2),normal[...,None])[...,0]
   wrong_n=np.einsum('nij,nj->ni',F,normal)
   dot=np.abs(np.einsum('ni,ni->n',unit(transported_v),unit(transported_n)))
   J=Xp@np.linalg.inv(Kb);J2=J@np.linalg.inv(A)
   gradient=np.einsum('nji,nj->ni',J,error);gradient2=np.einsum('nji,nj->ni',J2,error)
   raw=-np.einsum('nij,nj->ni',J,gradient);raw2=-np.einsum('nij,nj->ni',J2,gradient2)
   metric=-np.einsum('nij,nj->ni',J,np.linalg.solve(np.swapaxes(J,1,2)@J,gradient[...,None])[...,0])
   metric2=-np.einsum('nij,nj->ni',J2,np.linalg.solve(np.swapaxes(J2,1,2)@J2,gradient2[...,None])[...,0])
   row=dict(knee_deg=degree,address_absolute_error_max_by_component=np.max(abs(dxi),axis=0).tolist(),
    frozen_world_point_error_mm=summary(np.linalg.norm(p-p0,axis=1)*1000),
    material_vs_ordinary_barycentric_error_m=float(np.linalg.norm(p-(vp[:,0]+np.einsum('nij,nj->ni',Xp,b[:,1:])),axis=1).max()),
    frozen_tangent_error_deg=summary(angle(v,transported_v)),
    wrong_vector_normal_error_deg=summary(angle(wrong_n,transported_n)),
    correct_tangent_normal_abs_cosine=summary(dot),
    raw_gradient_reparameterization_angle_deg=summary(angle(raw,raw2)),
    metric_gradient_reparameterization_difference=summary(np.linalg.norm(unit(metric)-unit(metric2),axis=1)),
    metric_vs_cartesian_descent_difference=summary(np.linalg.norm(unit(metric)-unit(-error),axis=1)))
   rows.append(row)
   for k,value in [('address_error',dxi),('deformation',F),('world_point_error_m',np.linalg.norm(p-p0,axis=1)),
                   ('frozen_tangent_deg',angle(v,transported_v)),('wrong_normal_deg',angle(wrong_n,transported_n)),
                   ('raw_gradient_reparameterization_deg',angle(raw,raw2))]:arrays[f'{side}_{degree}_{k}']=value
   print(side,degree,'label',np.max(abs(dxi)),'frozenpointmm',row['frozen_world_point_error_mm']['p95'],flush=True)
  output[side]=rows
 np.savez_compressed(BASE/'RAW_RESULTS.npz',**arrays)
 result=dict(scope='Actual audited production harmonic volume and saved SMPL-X poses; local geometric transport/gradient checks only',
  protocol_sha256=sha(BASE/'PROTOCOL.txt'),source_sha256=sha(__file__),seed=720960,results=output,input_hashes=hashes,
  strong_baseline='Same tet/barycentric correspondence and Cartesian physical gradient match correct harmonic transport; no unique harmonic learning gain asserted')
 (BASE/'RESULTS.json').write_text(json.dumps(result,indent=2))
 return result

if __name__=='__main__':run()
