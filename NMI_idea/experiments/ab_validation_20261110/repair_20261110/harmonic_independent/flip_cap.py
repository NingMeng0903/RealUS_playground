from pathlib import Path
import argparse,json,numpy as np,tetgen
from shapely import Polygon
from audit import surface_stats
from skin_intersections import check
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--tag',required=True);a=p.parse_args();d=np.load(a.input);V=d['vertices'];F=d['faces'].copy();tags=d['face_markers'];skin=F[tags==1];skin_edges=set(map(tuple,np.sort(np.vstack((skin[:,[0,1]],skin[:,[1,2]],skin[:,[2,0]])),axis=1)));records=[]
for step in range(32):
 E=np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]));u,c=np.unique(np.sort(E,axis=1),axis=0,return_counts=True);bad=u[c>2]
 if not len(bad):break
 changed=False
 for edge in bad:
  ids=np.flatnonzero((tags==2)&(np.sum(np.isin(F,edge),axis=1)==2))
  if len(ids)!=2:continue
  t1,t2=F[ids];u0,v0=map(int,edge);i=int(np.flatnonzero(t1==u0)[0])
  if t1[(i+1)%3]!=v0:u0,v0=v0,u0
  aa=int(next(x for x in t1 if x not in edge));bb=int(next(x for x in t2 if x not in edge))
  if tuple(sorted((aa,bb))) in skin_edges:continue
  old=V[F[ids]];pts=V[[aa,u0,bb,v0]];origin=pts.mean(0);_,_,Q=np.linalg.svd(pts-origin,full_matrices=False);xy=(pts-origin)@Q[:2].T;poly=Polygon(xy)
  if not poly.is_valid or poly.area<=1e-14:continue
  new=np.array([[aa,u0,bb],[aa,bb,v0]],int);newxy=(V[new]-origin)@Q[:2].T
  if any(not poly.covers(Polygon(t)) for t in newxy):continue
  previous=F[ids].copy();F[ids]=new;inter=check(V,F,tags)
  if inter['cgal_intersecting_face_pairs']:
   F[ids]=previous;records.append({'edge':edge.tolist(),'accepted':False,'reason':'new cap creates proper CGAL intersections','intersection_count':inter['cgal_intersecting_face_pairs']});continue
  records.append({'edge':edge.tolist(),'accepted':True,'faces':ids.tolist(),'old_faces':previous.tolist(),'new_faces':new.tolist()});changed=True;break
 if not changed:break
stats=surface_stats(V,F);inter=check(V,F,tags);out={'input':str(a.input),'cap_only_flips':records,'unchanged_original_skin_faces':bool(np.array_equal(F[tags==1],skin)),'original_vertex_displacement_max_m':0.,'topology':stats,'cgal_intersections':inter};np.savez_compressed(H/(a.tag+'_flip_plc.npz'),vertices=V,faces=F,face_markers=tags,source_skin_vertex_ids=d['source_skin_vertex_ids']);(H/(a.tag+'_flip_metrics.json')).write_text(json.dumps(out,indent=2));print(a.tag,stats['watertight'],stats['nonmanifold_edges'],inter['cgal_intersecting_face_pairs'],records,flush=True)
if stats['watertight'] and inter['cgal_intersecting_face_pairs']==0:
 gen=tetgen.TetGen(V,F,tags.astype(np.int32));nodes,tets,_,markers=gen.tetrahedralize(plc=True,quality=False,nobisect=True,docheck=True,quiet=False,facesout=True);np.savez_compressed(H/(a.tag+'_flip_tetgen.npz'),vertices=nodes,tets=tets,boundary_faces=gen.trifaces,boundary_markers=markers,input_vertices=V,input_faces=F,input_face_markers=tags);pp=nodes[tets];det=np.linalg.det(np.moveaxis(pp[:,1:]-pp[:,:1],1,2));out['tetgen']={'nodes':len(nodes),'tets':len(tets),'positive':int(sum(det>1e-18)),'negative':int(sum(det< -1e-18)),'degenerate':int(sum(abs(det)<=1e-18))};(H/(a.tag+'_flip_metrics.json')).write_text(json.dumps(out,indent=2));print(out['tetgen'])
