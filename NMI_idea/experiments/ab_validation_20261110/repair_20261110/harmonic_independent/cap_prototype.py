"""Nonplanar caps preserving original skin xyz via constrained 2D triangulation."""
from pathlib import Path
import json,argparse,numpy as np
from scipy.spatial import cKDTree
from shapely import Polygon,constrained_delaunay_triangles,is_valid_reason
from audit import surface_stats,sha
H=Path(__file__).resolve().parent

def loops(F):
 edges=np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]));uu,ct=np.unique(np.sort(edges,axis=1),axis=0,return_counts=True);bound=set(map(tuple,uu[ct==1]));nxt={int(u):int(v) for u,v in edges if tuple(sorted((u,v))) in bound};out=[]
 while nxt:
  start=min(nxt);cur=start;loop=[]
  while True:
   loop.append(cur);cur=nxt.pop(cur)
   if cur==start:break
  out.append(loop)
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--asset',type=Path,required=True);p.add_argument('--tag',required=True);a=p.parse_args();asset=np.load(a.asset,allow_pickle=True);V=asset['skin_vertices'].astype(float);F=asset['skin_faces'].astype(np.int32);used=np.unique(F);mapping=np.full(len(V),-1,int);mapping[used]=np.arange(len(used));V=V[used];F=mapping[F];LL=loops(F);caps=[];record=[]
 for li,loop in enumerate(LL):
  # Reverse oriented skin boundary to give the cap opposite edge incidence.
  loop=np.array(loop[::-1]);xyz=V[loop];origin=xyz.mean(0);_,ss,Q=np.linalg.svd(xyz-origin,full_matrices=False);basis=Q[:2].T;xy=(xyz-origin)@basis;poly=Polygon(xy);info={'loop':li,'vertices':len(loop),'svd_singular_values_m':ss.tolist(),'nonplanar_height_span_m':float(np.ptp((xyz-origin)@Q[2])),'projected_polygon_valid':bool(poly.is_valid),'projected_polygon_reason':is_valid_reason(poly),'original_boundary_ids':used[loop].tolist()}
  if not poly.is_valid:record.append(info);continue
  triangles=constrained_delaunay_triangles(poly);tree=cKDTree(xy);area=0.;start=len(caps);signed_area=.5*sum(np.cross(xy[(i+1)%len(xy)]-xy[i],xy[(i+2)%len(xy)]-xy[i]) for i in []) if False else .5*np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1));expected=np.sign(signed_area)
  for tri in triangles.geoms:
   coords=np.asarray(tri.exterior.coords)[:3];dist,idx=tree.query(coords)
   if max(dist)>1e-10:raise ValueError('unexpected Steiner vertex; lifting rule must be explicit')
   area+=tri.area;idx=loop[idx];local=(V[idx]-origin)@basis;orient=np.cross(local[1]-local[0],local[2]-local[0]);
   if orient*expected<0:idx=idx[::-1]
   caps.append(idx.tolist())
  info.update(cap_faces=len(caps)-start,projected_area_m2=poly.area,triangle_area_sum_m2=area,relative_area_error=abs(area-poly.area)/poly.area);record.append(info)
 allF=np.vstack((F,np.array(caps,dtype=np.int32))) if caps else F;tags=np.r_[np.ones(len(F),np.int32),np.full(len(caps),2,np.int32)];out={'asset':str(a.asset),'asset_sha256':sha(a.asset),'skin_vertex_displacement_max_m':0.,'skin_face_count_preserved':len(F),'original_skin_bbox':[V.min(0).tolist(),V.max(0).tolist()],'cap_projection':'SVD only as parameterization; all original boundary xyz untouched; constrained Delaunay no Steiner vertices','loops':record,'combined_topology':surface_stats(V,allF),'all_loops_capped':len(record)==2 and all(r.get('cap_faces',0)>0 for r in record)}
 np.savez_compressed(H/(a.tag+'_cap_mesh.npz'),vertices=V,faces=allF,face_markers=tags,source_skin_vertex_ids=used,source_skin_faces=F,cap_faces=np.array(caps,dtype=np.int32));(H/(a.tag+'_cap_metrics.json')).write_text(json.dumps(out,indent=2));print(a.tag,json.dumps({k:v for k,v in out.items() if k not in ['combined_topology','loops']},indent=2));print(out['loops'])
if __name__=='__main__':main()
