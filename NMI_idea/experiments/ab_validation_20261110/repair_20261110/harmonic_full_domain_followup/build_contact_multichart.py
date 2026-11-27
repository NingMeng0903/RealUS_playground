from pathlib import Path
import sys,json,numpy as np,time,hashlib
ROOT=Path('/media/camp/EXT_DRIVE/Among_US');BASE=Path(__file__).resolve().parent;PRIOR=BASE.parent/'harmonic';sys.path.insert(0,str(ROOT/'src'))
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.shaft_rebuild import SurfaceMaterialAtlas
plan=dict(candidate='intrinsic_vertex_star_contact_surface_atlas',geometry='all unchanged ipsilateral original production skin triangles',coordinates='(chart_id,u_m,v_m); d=0 only',transition='shared complete source triangle and barycentric identity, fixed shared LBS nodes',selection='no template path used in construction or chart validity; template paths evaluated afterward',unresolved='not a full medial-depth 3D volume chart')
(BASE/'frozen_contact_multichart_plan.json').write_text(json.dumps(plan,indent=2));results={}
for side in ['right','left']:
 start=time.perf_counter();a=load_leg_volume_atlas(PRIOR/f'ownership_smooth/owned_smooth_{side}_skin.npz');w=np.load(PRIOR/f'ownership_smooth/{side}_surface_lbs_weights.npz')['weights'];m=SurfaceMaterialAtlas.from_surface(a,w);path=BASE/f'contact_multichart_{side}.npz';m.save(path);b=SurfaceMaterialAtlas.load(path);assert np.array_equal(a.skin_vertices,b.vertices) and np.array_equal(a.skin_faces,b.faces);results[side]=dict(vertices=len(b.vertices),faces=len(b.faces),charts=len(b.chart_centers),split_vertex_fans=b.metadata['split_vertex_fans'],asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),unchanged_source_geometry=True,wall_s=time.perf_counter()-start);print(side,results[side],flush=True);(BASE/'contact_multichart_generation.json').write_text(json.dumps(results,indent=2))
