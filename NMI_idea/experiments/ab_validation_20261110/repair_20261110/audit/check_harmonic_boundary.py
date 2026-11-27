"""Audit-only finite triangle-interior boundary checks, independent seed."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
sys.path[:0] = ['/media/camp/EXT_DRIVE/Among_US', '/media/camp/EXT_DRIVE/Among_US/src']
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas, query_atlas_coordinates
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.volume_refine import VolumeTetRefiner
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.harmonic import _BakedTetInterpolator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'harmonic' / 'shaft_roi_v2'
tag = sys.argv[1] if len(sys.argv) > 1 else 'pre_boundary_fix'
out, arrays = {}, {}
for side in ('left', 'right'):
    path = ROOT / f'atlas_{side}.npz'
    a = load_leg_volume_atlas(path)
    mapper = VolumeTetRefiner.from_atlas(a)
    interp = _BakedTetInterpolator(mapper.fields)
    w = np.load(ROOT / f'attachments_{side}.npz')
    grid = np.arange(len(a.harmonic_vertices)).reshape(tuple(w['grid_shape']))
    face_indices = np.concatenate([a.harmonic_tets[:, trip] for trip in ([1,2,3],[0,3,2],[0,1,3],[0,2,1])])
    native_outer = face_indices[np.all(a.harmonic_d[face_indices] == 0, axis=1)]
    native_inner = face_indices[np.all(a.harmonic_d[face_indices] == 1, axis=1)]
    exposed = grid[:, -1].ravel()[a.skin_faces]
    native_set = set(map(tuple, np.sort(native_outer, axis=1)))
    missing = np.array([tuple(sorted(face)) not in native_set for face in exposed])
    item = dict(asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                skin_faces=len(exposed), native_outer_faces=len(native_outer),
                skin_faces_absent_from_native_boundary=int(missing.sum()))
    for label, faces, target in [('advertised_skin', exposed, 0.), ('actual_outer', native_outer, 0.), ('actual_inner', native_inner, 1.)]:
        rng = np.random.default_rng(19954)
        ids = rng.choice(len(faces), 128, replace=False)
        bary = rng.dirichlet([1,1,1], len(ids))
        points = np.einsum('ni,nij->nj', bary, a.harmonic_vertices[faces[ids]].astype(float))
        arrays[f'{side}_{label}_points_f64'] = points
        arrays[f'{side}_{label}_faces'] = faces[ids]
        arrays[f'{side}_{label}_bary'] = bary
        result = {}
        for mode in ('public_api', 'native_f64'):
            errors, failures = [], []
            for index, point in enumerate(points):
                try:
                    if mode == 'public_api':
                        value = query_atlas_coordinates(a, point)[0][0, 2]
                    else:
                        value = interp.query(point[None], outside='raise')[2][0]
                    errors.append(abs(float(value)-target))
                except ValueError as e:
                    failures.append(dict(index=index, message=str(e)))
            result[mode] = dict(count=len(points), rejected=len(failures), failures=failures,
                                accepted_max_depth_error=max(errors) if errors else None)
        item[label] = result
    out[side] = item
np.savez(HERE / f'harmonic_boundary_{tag}.npz', **arrays)
(HERE / f'harmonic_boundary_{tag}.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
