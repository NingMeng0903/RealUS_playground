"""Offline check of anatomy representation and first target geometry in the mode ablation."""

from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

from engine import Scanner, prior_run
from acquire import CONDITIONS, fit_anatomy
from train import OUT, read_rows, save_json, save_rows
from view_geometry import REGIONS, measure_view


def analyze(row):
    scanner = Scanner(row['seed'], row['profile'], 120+CONDITIONS.index(row['profile']))
    scene = scanner.scene
    reference = np.stack([scene.truth.landmarks[region] for region in REGIONS])
    landmarks = scene.landmarks
    weights = scene.gall_weights(scanner.prefix)
    fitted, _, _ = fit_anatomy(scene, scanner.prefix, scanner.initial.history, None)
    continuous = scene.body.anatomy(fitted)

    def rmse(points):
        return np.sqrt(np.mean(np.sum((points-reference)**2, axis=-1), axis=-1))

    metrics = {'seed': row['seed'], 'profile': row['profile'],
               'continuous_fit_rms_marker_mm': float(rmse(np.stack([continuous.landmarks[r] for r in REGIONS]))),
               'posterior_weighted_mean_rms_marker_mm': float(rmse(np.einsum('m,mrc->rc', weights, landmarks))),
               'map_particle_rms_marker_mm': float(rmse(landmarks[int(np.argmax(weights))])),
               'nearest_particle_rms_marker_mm': float(np.min(rmse(landmarks))),
               'posterior_effective_particles': float(1. / (weights @ weights))}
    for arm in ('stationary', 'multimodal'):
        stations = row['arms'][arm]['station_log']
        if stations:
            q = np.asarray(stations[0]['q'])
            m = measure_view(scene.body, q, scene.truth, scene.case.rib)
            metrics[arm] = {'strict_first_pose': m.qualifies, 'center_abs_mm': abs(m.center),
                            'axis_plane_deg': m.angle, 'same_lumen_visible_score': min(m.visibility),
                            'area': m.area}
    return metrics


if __name__ == '__main__':
    arms = read_rows(OUT/'heldout.jsonl.gz')
    with ProcessPoolExecutor(max_workers=6, initializer=prior_run.initialize) as pool:
        rows = list(pool.map(analyze, arms))
    summary = {'n': len(rows)}
    for name in rows[0]:
        if name.endswith('mm') or name.endswith('particles'):
            summary['median_'+name] = float(np.median([r[name] for r in rows]))
    for arm in ('stationary', 'multimodal'):
        frame = [r[arm] for r in rows if arm in r]
        summary[arm] = {'n': len(frame), 'strict_first_pose': sum(r['strict_first_pose'] for r in frame),
                        'median_abs_center_mm': float(np.median([r['center_abs_mm'] for r in frame])),
                        'median_axis_deg': float(np.median([r['axis_plane_deg'] for r in frame]))}
    save_rows(OUT/'belief_audit_heldout.jsonl.gz', rows)
    save_json(OUT/'belief_audit_summary.json', summary)
    print(json.dumps(summary), flush=True)
