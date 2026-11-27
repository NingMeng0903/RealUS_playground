from pathlib import Path
import numpy as np,trimesh,tetgen,json,time,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prototype_tetgen import cap_mesh,ROOT,OUT
for side in ['right']:
 start=time.perf_counter();a=np.load(Path(__file__).resolve().parent.parent/'harmonic_independent'/f'ownership_{side}_skin.npz');v,f,m,used,loops=cap_mesh(a);mesh=trimesh.Trimesh(v,f,process=False)
 print(side,'input',v.shape,f.shape,'watertight',mesh.is_watertight,'winding',mesh.is_winding_consistent,'volume',mesh.volume,flush=True)
 try:
  gen=tetgen.TetGen(v,f,m);nodes,tets,attr,markers=gen.tetrahedralize(plc=True,quality=False,nobisect=True,docheck=True,quiet=False,facesout=True)
 except RuntimeError as e:
  print(side,'FAILED',str(e),flush=True);continue
 vv=nodes[tets];det=np.linalg.det(np.moveaxis(vv[:,1:]-vv[:,:1],1,2));np.savez_compressed(OUT/f'{side}_owned_native_outer_only.npz',vertices=nodes,tets=tets,boundary_faces=gen.trifaces,boundary_markers=markers,source_skin_ids=used,input_vertices=v,input_faces=f)
 result=dict(side=side,nodes=len(nodes),tets=len(tets),det_min=float(det.min()),det_max=float(det.max()),negative=int(sum(det<=0)),time_s=time.perf_counter()-start);print(result,flush=True);(OUT/f'{side}_owned_native_metrics.json').write_text(json.dumps(result,indent=2))
