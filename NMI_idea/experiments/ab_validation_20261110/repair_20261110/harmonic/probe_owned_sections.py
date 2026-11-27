from pathlib import Path
import sys,json,numpy as np,trimesh
from scipy.interpolate import splprep,splev
ROOT=Path('/media/camp/EXT_DRIVE/Among_US');OUT=Path(__file__).resolve().parent

def frames(tangents,reference):
 t=tangents/np.linalg.norm(tangents,axis=1)[:,None];a=reference-t[0]*(reference@t[0]);a/=np.linalg.norm(a);all=[]
 for i,v in enumerate(t):
  if i:
   cross=np.cross(t[i-1],v);ss=np.linalg.norm(cross);cc=t[i-1]@v
   if ss>1.e-14:
    axis=cross/ss;a=a*cc+np.cross(axis,a)*ss+axis*(axis@a)*(1-cc)
   a-=v*(a@v);a/=np.linalg.norm(a)
  all.append(np.stack([a,np.cross(v,a),v],axis=1))
 return np.array(all)

def hits_uv(lines,angle):
 direction=np.array([np.cos(angle),np.sin(angle)]);a=lines[:,0];edge=lines[:,1]-a
 cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
 den=cross(direction,edge);good=abs(den)>1.e-14
 radius=np.full(len(a),np.inf);u=np.full(len(a),np.inf)
 radius[good]=cross(a[good],edge[good])/den[good];u[good]=cross(a[good],direction)/den[good]
 idx=np.flatnonzero(good&(radius>1.e-8)&(u>=-1.e-8)&(u<=1+1.e-8))
 idx=idx[np.argsort(radius[idx])];unique=[]
 for k in idx:
  if not unique or abs(radius[k]-radius[unique[-1]])>1.e-7:unique.append(k)
 return np.array(unique,int),radius

allresults={}
for side in ['right','left']:
 a=np.load(OUT/'rebuild_work/owned_butterfly'/f'atlas_{side}.npz');mesh=trimesh.Trimesh(a['skin_vertices'],a['skin_faces'],process=False)
 hs=np.linspace(.05,.97,72);tck,u=splprep(a['core_points'].T,u=a['core_h'],s=1.e-4,k=3);centers=np.array(splev(hs,tck)).T;tangents=np.array(splev(hs,tck,der=1)).T;F=frames(tangents,a['pelvis']-a['hip'])
 results=[]
 for si,(h,c,f) in enumerate(zip(hs,centers,F)):
  lines,faces=trimesh.intersections.mesh_plane(mesh,f[:,2],c,return_faces=True)
  uv=(lines-c)@f[:,:2]
  quant=np.round(uv.reshape(-1,2)/1.e-7).astype(np.int64);_,counts=np.unique(quant,axis=0,return_counts=True)
  hits=[hits_uv(uv,t)[0] for t in np.linspace(0,2*np.pi,108,endpoint=False)]
  nh=np.array([len(z) for z in hits]);results.append(dict(station_index=si,h=float(h),segment_count=len(lines),endpoint_degree_bad=int(sum(counts!=2)),ray_hit_count_min=int(nh.min()),ray_hit_count_max=int(nh.max()),multi_hit_rays=int(sum(nh>1)),missing_rays=int(sum(nh==0))))
 allresults[side]=results;print(side,'closed stations',sum(r['endpoint_degree_bad']==0 for r in results),'multi_hitstations',sum(r['multi_hit_rays']>0 for r in results),'missing',sum(r['missing_rays']>0 for r in results));print([r for r in results if r['endpoint_degree_bad'] or r['multi_hit_rays'] or r['missing_rays']])
(OUT/'owned_geometric_section_probe.json').write_text(json.dumps(allresults,indent=2))
