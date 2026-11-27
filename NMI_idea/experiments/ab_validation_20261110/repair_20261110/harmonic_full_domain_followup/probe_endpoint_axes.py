from pathlib import Path
import sys,json,numpy as np,trimesh
BASE=Path(__file__).resolve().parent;PRIOR=BASE.parent/'harmonic';ROOT=Path('/media/camp/EXT_DRIVE/Among_US');sys.path.insert(0,str(ROOT/'src'))
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.shaft_rebuild import ray_segment_hits

def loops_from_faces(faces):
 e=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);u,c=np.unique(np.sort(e,axis=1),axis=0,return_counts=True);boundary=set(map(tuple,u[c==1]));edges=[tuple(x) for x in e if tuple(sorted(x)) in boundary];degree=np.bincount(np.asarray(edges).ravel());nxt={}
 for a,b in edges:
  if a in nxt:raise ValueError('nonmanifold boundary vertex')
  nxt[a]=b
 loops=[]
 while nxt:
  start=next(iter(nxt));current=start;loop=[]
  while True:
   loop.append(current);current=nxt.pop(current)
   if current==start:break
  loops.append(np.asarray(loop))
 return loops

results={}
for side in ['right','left']:
 a=load_leg_volume_atlas(PRIOR/f'ownership_smooth/owned_smooth_{side}_skin.npz');axis=(a.ankle-a.hip).astype(float);axis/=np.linalg.norm(axis);mesh=trimesh.Trimesh(a.skin_vertices,a.skin_faces,process=False);loops=loops_from_faces(a.skin_faces);loops.sort(key=lambda idx:float(a.skin_vertices[idx].mean(0)@axis));rows=[]
 for which,idx in zip(['proximal','distal'],loops):
  p=a.skin_vertices[idx].astype(float);center=p.mean(0);_,_,vh=np.linalg.svd(p-center);normal=vh[-1];normal*=np.sign(normal@axis);e1=vh[0];e2=np.cross(normal,e1);distance=(p-center)@normal;inward=normal if which=='proximal' else -normal
  scans=[]
  for offset in [.001,.003,.005,.01,.02,.04,.06,.08,.10,.15]:
   origin=center+offset*inward;lines,f=trimesh.intersections.mesh_plane(mesh,normal,origin,return_faces=True);uv=(lines-origin)@np.column_stack([e1,e2]);hits=[len(ray_segment_hits(uv,x)[0]) for x in np.linspace(0,2*np.pi,108,endpoint=False)];scans.append(dict(offset_m=offset,segments=len(lines),zero_rays=sum(n==0 for n in hits),multi_rays=sum(n>1 for n in hits)))
  rows.append(dict(which=which,count=len(idx),center=center.tolist(),normal=normal.tolist(),plane_residual_min=float(distance.min()),plane_residual_max=float(distance.max()),plane_residual_std=float(distance.std()),axis_angle_deg=float(np.rad2deg(np.arccos(np.clip(normal@axis,-1,1)))),scans=scans))
 results[side]=rows;print(side,json.dumps(rows,indent=2))
(BASE/'endpoint_axis_probe.json').write_text(json.dumps(results,indent=2))
