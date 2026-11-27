from pathlib import Path
import argparse,json,numpy as np,tetgen
from audit import surface_stats
from skin_intersections import check
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--tag',required=True);a=p.parse_args();d=np.load(a.input);V=d['vertices'].copy();F=d['faces'].copy();tags=d['face_markers'].copy();skin=F[tags==1].copy();originalV=V.copy();attempts=[]
for _ in range(100):
 E=np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]));u,c=np.unique(np.sort(E,axis=1),axis=0,return_counts=True);bad=u[c>2]
 if not len(bad):break
 changed=False
 for edge in bad:
  ids=np.flatnonzero((tags==2)&(np.sum(np.isin(F,edge),axis=1)==2))
  if len(ids)!=2:continue
  old=F[ids];normals=np.cross(V[old[:,1]]-V[old[:,0]],V[old[:,2]]-V[old[:,0]]);n=normals.sum(0);n/=np.linalg.norm(n)
  for lift in [.0001,-.0001,.001,-.001,.005,-.005,.01,-.01,.02,-.02]:
   mid=len(V);VV=np.vstack((V,V[edge].mean(0)+lift*n));new=[]
   for t in old:
    i=next(i for i in range(3) if t[i] in edge and t[(i+1)%3] in edge);u0,v0,w=t[i],t[(i+1)%3],t[(i+2)%3];new.extend([[u0,mid,w],[mid,v0,w]])
   keep=np.ones(len(F),bool);keep[ids]=False;FF=np.vstack((F[keep],np.array(new,int)));TT=np.r_[tags[keep],np.full(4,2)];inter=check(VV,FF,TT);accepted=inter['cgal_intersecting_face_pairs']==0;attempts.append({'edge':edge.tolist(),'new_vertex':VV[-1].tolist(),'lift_m':lift,'cgal_intersections':inter['cgal_intersecting_face_pairs'],'accepted':accepted})
   if accepted:V,F,tags=VV,FF,TT;changed=True;break
  if changed:break
 if not changed:break
stats=surface_stats(V,F);inter=check(V,F,tags);out={'input':str(a.input),'cap_only_steiner_attempts':attempts,'unchanged_original_skin_faces':bool(np.array_equal(F[tags==1],skin)),'unchanged_original_vertices':bool(np.array_equal(V[:len(originalV)],originalV)),'added_cap_vertices':len(V)-len(originalV),'topology':stats,'cgal_intersections':inter};np.savez_compressed(H/(a.tag+'_steiner_plc.npz'),vertices=V,faces=F,face_markers=tags,source_skin_vertex_ids=d['source_skin_vertex_ids']);(H/(a.tag+'_steiner_metrics.json')).write_text(json.dumps(out,indent=2));print(a.tag,'watertight',stats['watertight'],'nonmanifold',stats['nonmanifold_edges'],'CGAL',inter['cgal_intersecting_face_pairs'],'attempts',attempts,flush=True)
if stats['watertight'] and inter['cgal_intersecting_face_pairs']==0:
 gen=tetgen.TetGen(V,F,tags.astype(np.int32));nodes,tets,_,markers=gen.tetrahedralize(plc=True,quality=False,nobisect=True,docheck=True,quiet=False,facesout=True);np.savez_compressed(H/(a.tag+'_steiner_tetgen.npz'),vertices=nodes,tets=tets,boundary_faces=gen.trifaces,boundary_markers=markers,input_vertices=V,input_faces=F,input_face_markers=tags);pp=nodes[tets];det=np.linalg.det(np.moveaxis(pp[:,1:]-pp[:,:1],1,2));out['tetgen']={'nodes':len(nodes),'tets':len(tets),'positive':int(sum(det>1e-18)),'negative':int(sum(det< -1e-18)),'degenerate':int(sum(abs(det)<=1e-18))};(H/(a.tag+'_steiner_metrics.json')).write_text(json.dumps(out,indent=2));print(out['tetgen'],flush=True)
