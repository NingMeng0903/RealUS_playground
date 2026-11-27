from pathlib import Path
from dataclasses import replace
import sys,numpy as np,json,time
ROOT=Path('/media/camp/EXT_DRIVE/Among_US');OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas,save_leg_volume_atlas
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.shaft_rebuild import build_shaft_roi,tet_determinants
TARGET=OUT/'shaft_roi_v2';TARGET.mkdir(exist_ok=True)
results={}
for side in ['right','left']:
    started=time.perf_counter()
    clean_base=load_leg_volume_atlas(OUT/f'rebuild_work/owned_base/atlas_{side}.npz')
    original_owned=load_leg_volume_atlas(OUT/f'ownership_smooth/owned_smooth_{side}_skin.npz')
    source_attachments=np.load(OUT/f'ownership_smooth/{side}_surface_lbs_weights.npz')
    weights=source_attachments['weights']
    source=replace(original_owned,core_points=clean_base.core_points,core_h=clean_base.core_h,
        metadata={**original_owned.metadata,'roi_source_surface':'ipsilateral_faces_of_original_production_loop_surface',
            'centerline_source':'recomputed_ipsilateral_canonical_base'})
    rebuilt,attachments,metrics=build_shaft_roi(source,weights,station_count=72,theta_count=108)
    metrics['surface_rebinding_distance_m']={'mean':0.,'p95':0.,'max':0.,'meaning':'original retained source vertices and original stencil weights are identical; no cross-surface rebinding'}
    save_leg_volume_atlas(TARGET/f'atlas_{side}.npz',rebuilt)
    np.savez_compressed(TARGET/f'attachments_{side}.npz',**attachments,
        source_owned_surface_weights=weights,source_original_skin_vertex_indices=source_attachments['source_skin_vertex_indices'])
    loaded=load_leg_volume_atlas(TARGET/f'atlas_{side}.npz')
    determinant=tet_determinants(loaded.harmonic_vertices,loaded.harmonic_tets)
    assert np.all(determinant>0)
    metrics['after_save_load_det_min']=float(determinant.min());metrics['wall_s']=time.perf_counter()-started;results[side]=metrics
    (TARGET/'generation_metrics.json').write_text(json.dumps(results,indent=2));print(side,metrics,flush=True)
