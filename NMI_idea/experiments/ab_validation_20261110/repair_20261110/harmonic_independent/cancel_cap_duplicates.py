from pathlib import Path
import argparse,json,numpy as np,trimesh,tetgen
from audit import surface_stats,sha
from skin_intersections import check
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--tag',required=True);a=p.parse_args();d=np.load(a.input);V=d['vertices'];F=d['faces'];tags=d['face_markers'];unique,iv,ct=np.unique(np.sort(F,axis=1),axis=0,return_inverse=True,return_counts=True);keep=np.ones(len(F),bool);removed=[]
for k in np.flatnonzero(ct>1):
 ids=np.flatnonzero(iv==k)
 if len(ids)!=2 or set(tags[ids])!={1,2}:raise ValueError('unexpected duplicate faces: do not discard')
 n=[np.cross(V[F[i,1]]-V[F[i,0]],V[F[i,2]]-V[F[i,0]]) for i in ids]
 if n[0]@n[1]>=0:raise ValueError('same orientation duplicates: not cancelling boundary chain')
 keep[ids]=False;removed.append({'face_ids':ids.tolist(),'triangles':F[ids].tolist(),'tags':tags[ids].tolist(),'reason':'exact opposite skin/cap coincident triangles cancel as oriented boundary chain'})
FF=F[keep];TT=tags[keep];stats=surface_stats(V,FF);intersect=check(V,FF,TT);mesh=trimesh.Trimesh(V,FF,process=False);result={'input':str(a.input),'input_sha256':sha(a.input),'cancelled_opposite_face_pairs':removed,'topology':stats,'cgal_intersections':intersect,'original_vertex_displacement_max_m':0.,'enclosed_signed_volume_m3':float(mesh.volume),'source_bbox':[V.min(0).tolist(),V.max(0).tolist()],'retained_surface_bbox':[V[np.unique(FF)].min(0).tolist(),V[np.unique(FF)].max(0).tolist()]};np.savez_compressed(H/(a.tag+'_closed_plc.npz'),vertices=V,faces=FF,face_markers=TT,source_skin_vertex_ids=d['source_skin_vertex_ids'],cancelled_face_ids=np.flatnonzero(~keep));(H/(a.tag+'_closed_metrics.json')).write_text(json.dumps(result,indent=2));print(a.tag,'topology',{k:v for k,v in stats.items() if k!='boundary_vertex_ids'},'CGAL',intersect['cgal_intersecting_face_pairs'],flush=True)
assert stats['watertight'] and stats['nonmanifold_edges']==0 and stats['inconsistent_interior_winding']==0 and intersect['cgal_intersecting_face_pairs']==0
# Geometric prototype only: no production fields/asset overwrite.
gen=tetgen.TetGen(V,FF,TT.astype(np.int32));nodes,tets,attrs,markers=gen.tetrahedralize(plc=True,quality=False,nobisect=True,docheck=True,quiet=False,facesout=True);pp=nodes[tets];det=np.linalg.det(np.moveaxis(pp[:,1:]-pp[:,:1],1,2));np.savez_compressed(H/(a.tag+'_tetgen.npz'),vertices=nodes,tets=tets,boundary_faces=gen.trifaces,boundary_markers=markers,input_vertices=V,input_faces=FF,input_face_markers=TT);result['tetgen']={'nodes':len(nodes),'tets':len(tets),'positive':int(sum(det>1e-18)),'negative':int(sum(det< -1e-18)),'degenerate':int(sum(abs(det)<=1e-18)),'det_min_max':[float(min(det)),float(max(det))]};(H/(a.tag+'_closed_metrics.json')).write_text(json.dumps(result,indent=2));print(result['tetgen'],flush=True)
