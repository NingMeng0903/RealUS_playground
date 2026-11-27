#!/usr/bin/env python3
"""RGB-only tracking ablation: calibrated pelvis + rotation consensus + SMPL-X.

No DWPose landmarks enter tracking. A fixed-beta run deliberately shares the
previous fitted identity for a controlled comparison; --shape-source rgb uses
only the first RGB frame's adult beta median. Neither mode reads GT until all
predictions have been saved and evaluation starts.
"""
from pathlib import Path
import argparse
import contextlib
import io
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from perception.benchmark.hybrik_backend import load_official_layer
from perception.benchmark.hybrik_multiview import crop_uv_to_pixels, rotation_medoid, triangulate_root
from perception.benchmark.rgb_pose_fusion import fuse_local_rotation_predictions
from perception.benchmark.evaluation import evaluate_predictions
from perception.apps.render_mocap_benchmark_overlays import load_calibration


def main():
    import cv2
    import torch
    from scipy.spatial.transform import Rotation, Slerp
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb-cache-dir', type=Path, required=True)
    parser.add_argument('--recording-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--shape-source', choices=['fixed', 'rgb'], default='fixed')
    parser.add_argument('--init-npz', type=Path)
    parser.add_argument('--rotation-alpha', type=float, default=.7)
    args = parser.parse_args()
    if not 0 < args.rotation_alpha <= 1:
        parser.error('rotation-alpha must be in (0,1]')
    if args.shape_source == 'fixed' and args.init_npz is None:
        parser.error('fixed shape requires --init-npz (only shapes are read)')
    cache = json.loads((args.rgb_cache_dir / 'index.json').read_text())
    if Path(cache['recording_dir']).resolve() != args.recording_dir.resolve():
        raise ValueError('RGB cache recording mismatch')
    cameras = load_calibration(args.recording_dir)
    rows = sorted(cache['index'], key=lambda r: int(r['frame_index']))
    if any(meta.get('point_fallback') for r in rows for meta in r.get('bbox_metadata', [])):
        raise ValueError('RGB-only ablation forbids DWPose-point-derived crops')
    if args.shape_source == 'fixed':
        with np.load(args.init_npz) as d:
            beta = np.asarray(d['shapes'], np.float32).reshape(10)
    else:
        first = next(r for r in rows if r.get('npz_path'))
        with np.load(args.rgb_cache_dir / first['npz_path']) as d:
            beta = np.median(d['pred_beta'][:, :10], axis=0).astype(np.float32)
    asset = ROOT / 'ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_NEUTRAL.pkl'
    model = load_official_layer()(str(asset), gender='neutral', ext='pkl', num_betas=10,
                                  use_pca=False, flat_hand_mean=True).eval()
    betas = torch.from_numpy(beta[None])
    expression = torch.zeros((1, 10))
    previous, previous_root = None, None
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions, exported = [], []
    for row in rows:
        frame = int(row['frame_index'])
        if not row.get('npz_path'):
            predictions.append(dict(frame_index=frame, status='rejected', reason='rgb_cache_missing'))
            continue
        with np.load(args.rgb_cache_dir / row['npz_path']) as data:
            ids = data['camera_ids'].tolist()
            uv = crop_uv_to_pixels(data['pred_uvd_jts'].reshape(-1,71,3), data['bbox_used'])[:,0]
            perview = data['pred_rot_mats'].copy()
            rgb_elapsed = sum(float(data[k].sum()) for k in ('rgb_decode_elapsed_s','crop_elapsed_s','rgb_forward_elapsed_s'))
        started = time.perf_counter()
        try:
            ps, world_roots = [], []
            for v, camera_id in enumerate(ids):
                cam = cameras[camera_id]
                uv[v] = cv2.undistortPoints(uv[v].reshape(1,1,2), cam.K, cam.distortion, P=cam.K).reshape(2)
                ps.append(cam.K @ cam.camera_from_world[:3])
                world_roots.append(cam.world_from_camera[:3,:3] @ perview[v,0])
            root, root_inliers, reprojection = triangulate_root(uv, np.asarray(ps))
            rotations, accepted, agreement = fuse_local_rotation_predictions(perview)
            root_rotation, root_rotation_accepted = rotation_medoid(world_roots)
            rotations[0] = root_rotation
            accepted[0] = root_rotation_accepted
            for j in list(range(22)) + list(range(25,55)):
                if not accepted[j]:
                    rotations[j] = (previous[j] if previous is not None else
                                    (root_rotation if j == 0 else rotation_medoid(perview[:,j])[0]))
            # Time-continuous per-frame SO(3) interpolation, not Euler averaging.
            if previous is not None and args.rotation_alpha < 1:
                for j in list(range(22)) + list(range(25,55)):
                    rotations[j] = Slerp([0,1], Rotation.from_matrix(np.stack([previous[j],rotations[j]])))([args.rotation_alpha]).as_matrix()[0]
                root = args.rotation_alpha*root + (1-args.rotation_alpha)*previous_root
            with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
                output = model.forward_simple(betas=betas, expression=expression,
                                              full_pose=torch.from_numpy(rotations.astype(np.float32)[None]))
            local_joints = output.joints_55[0].numpy()
            offset = root - local_joints[0]
            vertices = output.vertices[0].numpy() + offset
            joints = local_joints + offset
            elapsed = time.perf_counter()-started
            target = args.output_dir / f'moment_{len(exported):04d}'
            target.mkdir()
            pose = Rotation.from_matrix(rotations).as_rotvec().astype(np.float32)
            np.savez_compressed(target/'smplx_result.npz', poses=pose.reshape(1,165), shapes=beta[None],
                                Rh=pose[0:1], pelvis_world_m=root, vertices=vertices, faces=model.faces,
                                joints_world_m=joints, rgb_pose_accepted=accepted, root_inlier_views=root_inliers,
                                root_reprojection_px=reprojection, gender=np.asarray('neutral'))
            exported.append(dict(frame_index=frame, timestamp_ns=row['timestamp_ns'], moment_dir=str(target.resolve()), fit_ok=True))
            predictions.append(dict(frame_index=frame, status='ok', timestamp_ns=row['timestamp_ns'],
                                    joints_world_m=joints.tolist(), latency_elapsed_s=None,
                                    rgb_elapsed_s=rgb_elapsed, backend_elapsed_s=elapsed,
                                    rgb_plus_backend_elapsed_s=rgb_elapsed+elapsed,
                                    accepted_pose_joints=int(accepted.sum()), root_inlier_views=int(root_inliers.sum()),
                                    root_reprojection_mean_px=float(reprojection[root_inliers].mean())))
            previous, previous_root = rotations.copy(), root.copy()
        except ValueError as exc:
            predictions.append(dict(frame_index=frame, status='rejected', reason=str(exc)))
    (args.output_dir/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    (args.output_dir/'index.json').write_text(json.dumps(dict(index=exported),indent=2)+'\n')
    # GT is opened for the first time here, after inference has completed.
    summary = evaluate_predictions(args.recording_dir, predictions, method='HybrIK-X RGB-only multiview rotation/pelvis tracking')
    summary.update(shape_source=args.shape_source, beta=beta.tolist(), rotation_alpha=args.rotation_alpha,
                   rgb_cache_dir=str(args.rgb_cache_dir.resolve()), complete_pipeline_latency_available=False,
                   input_policy='Cached RGB predictions and calibrated cameras; no DWPose points. Saved detector/temporal boxes reused, point-derived bbox fallback forbidden.',
                   shape_policy='First RGB adult10 median' if args.shape_source=='rgb' else 'Shared prior silhouette identity; only beta read from init NPZ, no pose/root seed.',
                   fusion_policy='Expanded-bbox UVD inverse; 15px >=3-view pelvis RANSAC; camera-to-world root rotations; 45deg local rotation majority; disagreements retain previous accepted rotations; first-frame medoid fallback; jaw/eyes fixed identity.',
                   latency_policy='RGB decode/crop/forward cached; CPU fusion/SMPL-X timed here. Detector timing unavailable, so no full pipeline speed claim.')
    successful = [r for r in predictions if r['status']=='ok']
    summary['rgb_plus_backend_latency_mean_s'] = float(np.mean([r['rgb_plus_backend_elapsed_s'] for r in successful])) if successful else None
    summary['backend_latency_mean_s'] = float(np.mean([r['backend_elapsed_s'] for r in successful])) if successful else None
    (args.output_dir/'benchmark_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('per_frame','rejected')},indent=2))


if __name__ == '__main__':
    main()
