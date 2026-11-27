from pathlib import Path
from dataclasses import replace
import numpy as np,json,hashlib,shutil,sys
ROOT=Path('/media/camp/EXT_DRIVE/Among_US');OUT=Path(__file__).resolve().parent;TARGET=OUT/'shaft_roi_v2';ARCHIVE=OUT/'shaft_roi_v2_before_metadata_normalization';sys.path.insert(0,str(ROOT/'src'))
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas,save_leg_volume_atlas
if not ARCHIVE.exists():
 ARCHIVE.mkdir()
 for file in TARGET.iterdir():
  if file.is_file():shutil.copy2(file,ARCHIVE/file.name)
keys='method mesh_type coordinate_domain default_depth_mode domain_h_range station_count theta_count radial_count inner_fraction centerline_method frame_method surface_method depth_method h_method theta_method volume_vertex_count volume_tet_count volume_sample_count skin_vertex_count skin_face_count depth_one_boundary material_coordinate_orientation full_vertex_indices_semantics scope side_face_ownership roi_source_surface centerline_source asset_revision surface_connectivity_source'.split()
records={}
for side in ['right','left']:
 a=load_leg_volume_atlas(TARGET/f'atlas_{side}.npz');metadata={k:a.metadata[k] for k in keys if k in a.metadata};metadata['source_surface_metadata']=a.metadata;metadata['asset_revision']='native_boundary_float64_queries_metadata_normalized';b=replace(a,metadata=metadata);save_leg_volume_atlas(TARGET/f'atlas_{side}.npz',b);loaded=load_leg_volume_atlas(TARGET/f'atlas_{side}.npz');same={}
 for key in a.__dataclass_fields__:
  value=getattr(a,key)
  if isinstance(value,np.ndarray):same[key]=bool(np.array_equal(value,getattr(loaded,key)))
 assert all(same.values());records[side]=dict(all_array_values_unchanged=same,prior_asset_sha256=hashlib.sha256((ARCHIVE/f'atlas_{side}.npz').read_bytes()).hexdigest(),new_asset_sha256=hashlib.sha256((TARGET/f'atlas_{side}.npz').read_bytes()).hexdigest())
hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in TARGET.glob('*.npz') if f.name.startswith(('atlas_','attachments_'))};(TARGET/'asset_hashes.json').write_text(json.dumps(hashes,indent=2));(TARGET/'metadata_revision.json').write_text(json.dumps(records,indent=2));print(json.dumps(hashes,indent=2))
