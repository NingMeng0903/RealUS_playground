from pathlib import Path
import argparse,json,time,numpy as np
from igl.copyleft.cgal import remesh_self_intersections
from audit import sha
H=Path(__file__).resolve().parent

def check(V,F,tags):
 start=time.perf_counter();out=remesh_self_intersections(np.asarray(V,dtype=np.float64),np.asarray(F,dtype=np.int64),detect_only=True,first_only=False,cutoff=10000000);pairs=np.asarray(out[2]);counts={};examples=[]
 for i,j in pairs:
  common=len(set(F[i]).intersection(F[j]));key=f'tag{tags[i]}_tag{tags[j]}_shared{common}';counts[key]=counts.get(key,0)+1
  if len(examples)<64:examples.append({'faces':[int(i),int(j)],'tags':[int(tags[i]),int(tags[j])],'shared_vertices':common,'vertices_a':V[F[i]].tolist(),'vertices_b':V[F[j]].tolist()})
 return {'complete_detect_only':True,'cgal_intersecting_face_pairs':len(pairs),'categories':counts,'examples':examples,'seconds':time.perf_counter()-start}

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--tag',required=True);a=p.parse_args();d=np.load(a.input,allow_pickle=True)
 if 'skin_vertices' in d:V,F=d['skin_vertices'],d['skin_faces'];tags=np.ones(len(F),int)
 else:V,F=d['vertices'],d['faces'];tags=d['face_markers']
 result=check(V,F,tags);result.update(input=str(a.input),input_sha256=sha(a.input));(H/(a.tag+'_skin_intersections.json')).write_text(json.dumps(result,indent=2));print(a.tag,result['cgal_intersecting_face_pairs'],result['categories'],result['seconds'],flush=True)
if __name__=='__main__':main()
