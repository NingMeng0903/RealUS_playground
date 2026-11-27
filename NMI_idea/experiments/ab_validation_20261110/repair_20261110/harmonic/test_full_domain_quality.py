from pathlib import Path
import numpy as np,trimesh,tetgen,pymeshfix,json,time
OUT=Path(__file__).resolve().parent;TARGET=OUT/'tetgen_quality_trials';TARGET.mkdir(exist_ok=True)
results=[]
for side in ['right','left']:
 original=np.load(OUT/f'tetgen_prototype/{side}_owned_clean_meshfix_outer_only.npz')
 for digits in [6,5]:
  start=time.perf_counter();mesh=trimesh.Trimesh(original['repaired_surface_vertices'],original['repaired_surface_faces'],process=False);beforev=len(mesh.vertices);mesh.merge_vertices(digits_vertex=digits);mesh.update_faces(mesh.nondegenerate_faces(height=1.e-8));mesh.update_faces(mesh.unique_faces());mesh.remove_unreferenced_vertices();fix=pymeshfix.MeshFix(mesh.vertices,mesh.faces);fix.repair(joincomp=False,remove_smallest_components=False);v=np.asarray(fix.points);f=np.asarray(fix.faces)
  record=dict(side=side,merge_decimal_digits=digits,input_vertices_before=beforev,input_vertices_after=len(v),input_faces=len(f))
  try:
   gen=tetgen.TetGen(v,f);nodes,tets,_,markers=gen.tetrahedralize(plc=True,quality=True,nobisect=True,minratio=1.4,mindihedral=10.,maxvolume=1.e-6,fixedvolume=True,steinerleft=100000,docheck=True,quiet=True,facesout=True)
   vv=nodes[tets];det=np.linalg.det(np.moveaxis(vv[:,1:]-vv[:,:1],1,2));vf=nodes.astype(np.float32).astype(float)[tets];df=np.linalg.det(np.moveaxis(vf[:,1:]-vf[:,:1],1,2));edges=vf[:,1:]-vf[:,:1];scaled=df/np.maximum(np.prod(np.linalg.norm(edges,axis=2),axis=1),1.e-100)
   record.update(vertices=len(nodes),tets=len(tets),float64_det_min=float(det.min()),float32_negative=int(sum(df<0)),float32_degenerate=int(sum(df==0)),float32_det_min=float(df.min()),scaled_corner0_min=float(scaled.min()),scaled_corner0_below1e6=int(sum(scaled<1.e-6)))
   np.savez_compressed(TARGET/f'{side}_merge{digits}.npz',vertices=nodes.astype(np.float32),tets=tets,boundary_faces=gen.trifaces,boundary_markers=markers,input_vertices=v,input_faces=f)
  except Exception as error:record['error']=str(error)
  record['wall_s']=time.perf_counter()-start;results.append(record);(TARGET/'metrics.json').write_text(json.dumps(results,indent=2));print(record,flush=True)
