from pathlib import Path
import sys,json,numpy as np
from scipy.interpolate import PchipInterpolator
ROOT=Path('/media/camp/EXT_DRIVE/Among_US');OUT=Path(__file__).resolve().parent/'prism_prototype';OUT.mkdir(exist_ok=True)

def build(a,levels=6):
 skin=a['skin_vertices'].astype(float);faces=a['skin_faces'];axis=(a['ankle']-a['hip']).astype(float);axis/=np.linalg.norm(axis)
 ref=(a['pelvis']-a['hip']).astype(float);e1=ref-axis*(ref@axis);e1/=np.linalg.norm(e1);e2=np.cross(axis,e1)
 cp=a['core_points'].astype(float);cs=(cp-a['hip'])@axis;order=np.argsort(cs);cs,ci=np.unique(cs[order],return_index=True);cp=cp[order][ci]
 s=(skin-a['hip'])@axis;uv=np.column_stack([(cp-a['hip'])@e1,(cp-a['hip'])@e2]);interp=PchipInterpolator(cs,uv,axis=0);cuv=interp(np.clip(s,cs[0],cs[-1]));centers=a['hip']+s[:,None]*axis+cuv[:,0,None]*e1+cuv[:,1,None]*e2
 frac=np.linspace(.04,1.,levels);vertices=(centers[None]+frac[:,None,None]*(skin-centers)[None]).reshape(-1,3)
 nf=len(faces);nv=len(skin);sf=np.sort(faces,axis=1);perm=np.argsort(faces,axis=1);odd=((perm[:,0]>perm[:,1]).astype(int)+(perm[:,0]>perm[:,2])+(perm[:,1]>perm[:,2]))%2==1
 all=[]
 for i in range(levels-1):
  aa,bb,cc=(sf+i*nv).T;AA,BB,CC=(sf+(i+1)*nv).T
  t=np.stack([np.column_stack([aa,bb,cc,CC]),np.column_stack([aa,bb,CC,BB]),np.column_stack([aa,AA,BB,CC])],axis=1)
  t[odd,:,0],t[odd,:,1]=t[odd,:,1].copy(),t[odd,:,0].copy();all.append(t.reshape(-1,4))
 tets=np.vstack(all).astype(np.int32)
 theta=np.mod(np.arctan2((skin-centers)@e2,(skin-centers)@e1),2*np.pi);hh=(s-s.min())/(s.max()-s.min())
 xi=np.stack([np.tile(theta,levels),np.tile(hh,levels),np.repeat((1-frac)/(.96),nv)],axis=1)
 return vertices,tets,xi,centers,frac

results={}
for side in ['right','left']:
 a=np.load(ROOT/f'dataset/processed/anatomy_retarget/leg_volume_coordinates/production/atlas/atlas_{side}.npz');v,t,xi,c,frac=build(a)
 vv=v[t];m=np.moveaxis(vv[:,1:]-vv[:,:1],1,2);det=np.linalg.det(m);xx=xi[t];xx[:,:,0]=xx[:,0,0,None]+(xx[:,:,0]-xx[:,0,0,None]+np.pi)%(2*np.pi)-np.pi;dx=np.linalg.det(np.moveaxis(xx[:,1:]-xx[:,:1],1,2));j=dx/det
 result=dict(vertices=len(v),tets=len(t),positive=int(sum(det>1.e-20)),negative=int(sum(det<-1.e-20)),degenerate=int(sum(abs(det)<=1.e-20)),det_min=float(det.min()),det_max=float(det.max()),material_J_min=float(j.min()),material_J_max=float(j.max()),material_J_positive=int(sum(j>1.e-8)),material_J_negative=int(sum(j<-1.e-8)),material_J_degenerate=int(sum(abs(j)<=1.e-8)))
 results[side]=result;print(side,json.dumps(result));np.savez_compressed(OUT/f'{side}_prototype.npz',vertices=v,tets=t,xi=xi,centers=c,radial_fractions=frac)
(OUT/'geometry_metrics.json').write_text(json.dumps(results,indent=2))
