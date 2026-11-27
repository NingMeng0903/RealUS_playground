from pathlib import Path
import json,numpy as np
from audit import sha,SOURCE
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.harmonic import _distance_to_skin_surface
H=Path(__file__).resolve().parent;WORK=H.parent/'harmonic/rebuild_work'
rows=[]
for side in ['left','right']:
 p=WORK/'owned_base'/f'atlas_{side}.npz';q=WORK/'owned_butterfly'/f'atlas_{side}.npz';base=np.load(p,allow_pickle=True);fine=np.load(q,allow_pickle=True);V,F=base['skin_vertices'],base['skin_faces'];W,G=fine['skin_vertices'],fine['skin_faces'];direct=np.linalg.norm(W[:len(V)].astype(float)-V.astype(float),axis=1)
 x=np.vstack((W,W[G].mean(1)));y=np.vstack((V,V[F].mean(1)));dx=_distance_to_skin_surface(x,V,F);dy=_distance_to_skin_surface(y,W,G);rows.append({'side':side,'base_sha256':sha(p),'butterfly_sha256':sha(q),'native_vertex_count':len(V),'subdivision_vertices':len(W),'subdivision_faces':len(G),'max_preserved_native_vertex_displacement_m':float(max(direct)),'all_subdivision_vertices_and_face_centers_to_native_triangles':{'count':len(x),'mean_m':float(np.mean(dx)),'p95_m':float(np.quantile(dx,.95)),'max_m':float(max(dx)),'max_xyz':x[np.argmax(dx)].tolist()},'all_native_vertices_and_face_centers_to_subdivision_triangles':{'count':len(y),'mean_m':float(np.mean(dy)),'p95_m':float(np.quantile(dy,.95)),'max_m':float(max(dy))},'limit':'finite dense vertex/centroid samples, not a certified surface Hausdorff bound; interpolatory subdivision is not exact native triangle geometry'})
(H/'subdivision_distance_results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
