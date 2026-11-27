from pathlib import Path
import sys,time,json,numpy as np
ROOT=Path('/media/camp/EXT_DRIVE/Among_US');sys.path.insert(0,str(ROOT/'src'))
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import bake_leg_volume_atlases,save_leg_volume_atlas,LegVolumeConfig
from projects.genesis_ue_sync.anatomy_retarget.cli.run_bake_leg_volume_ultimate import _with_butterfly_surface
OUT=Path(__file__).resolve().parent/'rebuild_work';(OUT/'owned_base').mkdir(parents=True,exist_ok=True);(OUT/'owned_butterfly').mkdir(exist_ok=True)
t=time.perf_counter();atlases=bake_leg_volume_atlases(ROOT/'outputs/anatomy_retarget/recovered_smplx_20261108',config=LegVolumeConfig())
for side,a in atlases.items():
 save_leg_volume_atlas(OUT/'owned_base'/f'atlas_{side}.npz',a)
 b=_with_butterfly_surface(a,level=3);save_leg_volume_atlas(OUT/'owned_butterfly'/f'atlas_{side}.npz',b)
 print(side,a.skin_vertices.shape,a.skin_faces.shape,b.skin_vertices.shape,b.skin_faces.shape,a.metadata['side_face_ownership'],flush=True)
print('wall_s',time.perf_counter()-t)
