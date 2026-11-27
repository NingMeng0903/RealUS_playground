from pathlib import Path
import numpy as np,trimesh,tetgen,json,time
ROOT=Path('/media/camp/EXT_DRIVE/Among_US');OUT=Path(__file__).resolve().parent/'tetgen_prototype';OUT.mkdir(exist_ok=True)

def boundary_loops(faces):
 e=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);u,c=np.unique(np.sort(e,axis=1),axis=0,return_counts=True);bound=set(map(tuple,u[c==1]));nxt=dict(tuple(x) for x in e if tuple(sorted(x)) in bound);loops=[]
 while nxt:
  first=next(iter(nxt));cur=first;loop=[]
  while True:
   loop.append(cur);cur=nxt.pop(cur)
   if cur==first:break
  loops.append(loop)
 return loops

def cap_mesh(a):
 used=np.unique(a['skin_faces']);remap=np.full(len(a['skin_vertices']),-1,int);remap[used]=np.arange(len(used));v=a['skin_vertices'][used].astype(float);f=remap[a['skin_faces']].astype(np.int32);loops=boundary_loops(f);caps=[]
 for loop in loops:
  center=len(v);v=np.vstack([v,v[loop].mean(0)])
  caps.extend([[center,b,a] for a,b in zip(loop,np.roll(loop,-1))])
 allfaces=np.vstack([f,np.array(caps,int)]).astype(np.int32);markers=np.r_[np.ones(len(f),int),np.full(len(caps),2,int)].astype(np.int32)
 return v,allfaces,markers,used,loops

if __name__=='__main__':
 for side in ['right','left']:
  start=time.perf_counter();a=np.load(ROOT/f'dataset/processed/anatomy_retarget/leg_volume_coordinates/production/atlas/atlas_{side}.npz');v,f,m,used,loops=cap_mesh(a);mesh=trimesh.Trimesh(v,f,process=False)
  print(side,'input',v.shape,f.shape,'watertight',mesh.is_watertight,'winding',mesh.is_winding_consistent,'volume',mesh.volume,flush=True)
  gen=tetgen.TetGen(v,f,m)
  nodes,tets,attr,markers=gen.tetrahedralize(plc=True,quality=False,nobisect=True,docheck=True,quiet=False,facesout=True)
  vv=nodes[tets];det=np.linalg.det(np.moveaxis(vv[:,1:]-vv[:,:1],1,2))
  np.savez_compressed(OUT/f'{side}_outer_only.npz',vertices=nodes,tets=tets,boundary_faces=gen.trifaces,boundary_markers=markers,source_skin_ids=used,input_vertices=v,input_faces=f)
  result=dict(side=side,nodes=len(nodes),tets=len(tets),det_min=float(det.min()),det_max=float(det.max()),negative=int(sum(det<=0)),time_s=time.perf_counter()-start)
  print(result,flush=True);(OUT/f'{side}_metrics.json').write_text(json.dumps(result,indent=2))
