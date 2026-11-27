from pathlib import Path
import numpy as np,trimesh,tetgen,json,time,sys,pymeshfix
sys.path.insert(0,str(Path(__file__).resolve().parent));from prototype_tetgen import cap_mesh,ROOT,OUT
for side in ['right','left']:
 start=time.perf_counter();a=np.load(ROOT/f'dataset/processed/anatomy_retarget/leg_volume_coordinates/bake/base_atlas/atlas_{side}.npz');v,f,m,used,loops=cap_mesh(a);fix=pymeshfix.MeshFix(v,f);fix.repair(joincomp=False,remove_smallest_components=False);v=np.asarray(fix.points);f=np.asarray(fix.faces)
 mesh=trimesh.Trimesh(v,f,process=False);print(side,'repaired input',v.shape,f.shape,'watertight',mesh.is_watertight,'winding',mesh.is_winding_consistent,'volume',mesh.volume,flush=True)
 try:
  gen=tetgen.TetGen(v,f);nodes,tets,attr,markers=gen.tetrahedralize(plc=True,quality=True,minratio=1.6,mindihedral=5.,maxvolume=1.e-6,fixedvolume=True,steinerleft=100000,docheck=True,quiet=False,facesout=True)
 except RuntimeError as e:
  print(side,'FAILED',str(e),flush=True);continue
 vv=nodes[tets];det=np.linalg.det(np.moveaxis(vv[:,1:]-vv[:,:1],1,2));np.savez_compressed(OUT/f'{side}_meshfix_outer_only.npz',vertices=nodes,tets=tets,boundary_faces=gen.trifaces,boundary_markers=markers,repaired_surface_vertices=v,repaired_surface_faces=f,source_skin_ids=used,input_vertices=v,input_faces=f)
 result=dict(side=side,nodes=len(nodes),tets=len(tets),det_min=float(det.min()),det_max=float(det.max()),negative=int(sum(det<=0)),time_s=time.perf_counter()-start);print(result,flush=True);(OUT/f'{side}_meshfix_metrics.json').write_text(json.dumps(result,indent=2))
