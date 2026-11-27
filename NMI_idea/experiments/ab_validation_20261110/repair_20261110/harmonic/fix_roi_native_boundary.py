from pathlib import Path
from dataclasses import replace
import numpy as np,json,hashlib,shutil,sys
ROOT=Path('/media/camp/EXT_DRIVE/Among_US');OUT=Path(__file__).resolve().parent;TARGET=OUT/'shaft_roi_v2';ARCHIVE=OUT/'shaft_roi_v2_before_boundary_fix';sys.path.insert(0,str(ROOT/'src'))
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas,save_leg_volume_atlas,_compute_vertex_normals
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.shaft_rebuild import native_boundary_surface
if not ARCHIVE.exists():
 ARCHIVE.mkdir()
 for file in TARGET.iterdir():
  if file.is_file():shutil.copy2(file,ARCHIVE/file.name)
records={}
for side in ['right','left']:
 a=load_leg_volume_atlas(TARGET/f'atlas_{side}.npz');shape=tuple(np.load(TARGET/f'attachments_{side}.npz')['grid_shape']);ids=np.arange(len(a.harmonic_vertices)).reshape(shape)[:,-1].ravel();faces=native_boundary_surface(a.harmonic_tets,ids,len(a.harmonic_vertices));b=replace(a,skin_faces=faces,skin_normals=_compute_vertex_normals(a.skin_vertices,faces),metadata={**a.metadata,'asset_revision':'native_boundary_and_float64_queries','surface_connectivity_source':'oriented_outer_boundary_faces_of_native_tets'})
 save_leg_volume_atlas(TARGET/f'atlas_{side}.npz',b);loaded=load_leg_volume_atlas(TARGET/f'atlas_{side}.npz');names=['harmonic_vertices','harmonic_tets','harmonic_h','harmonic_theta','harmonic_d','volume_points','volume_xi','skin_vertices'];equal={name:bool(np.array_equal(getattr(a,name),getattr(loaded,name))) for name in names};assert all(equal.values());records[side]=dict(unchanged_volume_and_vertex_arrays=equal,old_skin_faces=len(a.skin_faces),new_native_boundary_faces=len(faces),prior_asset_sha256=hashlib.sha256((ARCHIVE/f'atlas_{side}.npz').read_bytes()).hexdigest(),new_asset_sha256=hashlib.sha256((TARGET/f'atlas_{side}.npz').read_bytes()).hexdigest())
hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in TARGET.glob('*.npz') if f.name.startswith(('atlas_','attachments_'))}
(TARGET/'asset_hashes.json').write_text(json.dumps(hashes,indent=2));(TARGET/'boundary_revision.json').write_text(json.dumps(records,indent=2));print(json.dumps(hashes,indent=2))
