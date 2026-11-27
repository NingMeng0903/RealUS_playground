from pathlib import Path
import sys,numpy as np,json
from dataclasses import replace
from scipy.sparse import csr_matrix
ROOT=Path('/media/camp/EXT_DRIVE/Among_US');sys.path.insert(0,str(ROOT/'src'))
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas,save_leg_volume_atlas,_side_face_ownership,_remap_faces,_largest_skin_component
OUT=Path(__file__).resolve().parent/'ownership_smooth';OUT.mkdir(exist_ok=True);PROD=ROOT/'dataset/processed/anatomy_retarget/leg_volume_coordinates/production';CANON=ROOT/'outputs/anatomy_retarget/recovered_smplx_20261108';native=np.load(CANON/'smpl_canonical_weights.npz');skeleton=json.loads((CANON/'smpl_canonical_skeleton.json').read_text());results={}
for side in ['right','left']:
 a=load_leg_volume_atlas(PROD/'atlas'/f'atlas_{side}.npz');p=np.load(PROD/'material'/f'surface_stencil_{side}.npz');stencil=csr_matrix((p['stencil_weights'],p['stencil_indices'],p['stencil_indptr']),shape=(len(a.skin_vertices),len(p['source_full_vertex_indices'])));w=stencil@native['lbs_weights'][p['source_full_vertex_indices']];keep,meta=_side_face_ownership(a.skin_faces,w,skeleton,side);ids,faces=_remap_faces(a.skin_faces[keep],np.ones(len(a.skin_vertices),bool));ids,faces=_largest_skin_component(a.skin_vertices,ids,faces);meta.update(candidate_only=True,volume_arrays_are_old_and_invalid=True)
 b=replace(a,skin_vertices=a.skin_vertices[ids],skin_faces=faces,skin_theta=a.skin_theta[ids],skin_h=a.skin_h[ids],skin_d=a.skin_d[ids],skin_normals=a.skin_normals[ids],full_vertex_indices=a.full_vertex_indices[ids],metadata={**a.metadata,'side_face_ownership':meta});save_leg_volume_atlas(OUT/f'owned_smooth_{side}_skin.npz',b);np.savez_compressed(OUT/f'{side}_surface_lbs_weights.npz',weights=w[ids],source_skin_vertex_indices=ids,source_surface_face_mask=keep)
 results[side]=dict(original_vertices=len(a.skin_vertices),original_faces=len(a.skin_faces),retained_vertices=len(ids),retained_faces=len(faces),removed_faces=int(sum(~keep)),ownership=meta);print(side,results[side])
(OUT/'ownership_metrics.json').write_text(json.dumps(results,indent=2))
