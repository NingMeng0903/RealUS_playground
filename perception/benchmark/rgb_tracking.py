"""One persistent, batched official RGB pose-prior provider for live/replay tracking."""
import time
import numpy as np
from perception.benchmark.hybrik_rgb import OfficialHybrIKXRGB, select_rgb_bbox
from perception.benchmark.rgb_pose_fusion import fuse_local_rotation_predictions


class RGBPosePrior:
    def __init__(self, *, device='cuda', warmup_views=5):
        if warmup_views < 2:
            raise ValueError('RGB multiview prior requires at least two cameras')
        self.model = OfficialHybrIKXRGB(device=device, flip_test=True)
        self.rgb_metadata = dict(self.model.metadata)
        self.rgb_metadata['tracking_view_mode'] = 'one full official batch, float32, flip test'
        started = time.perf_counter()
        # Kernel initialization only: synthetic crops produce no tracking hints.
        crops = [self.model.torch.zeros((3,256,256)) for _ in range(warmup_views)]
        boxes = np.tile(np.array([0,0,256,256],np.float32), (warmup_views,1))
        centers = np.tile(np.array([128,128],np.float32), (warmup_views,1))
        self.model.forward_prepared_batch(crops, boxes, centers)
        self.warmup_elapsed_s = time.perf_counter()-started

    def infer(self, views_rgb, detection_meta, camera_ids, *, projection_matrices=None):
        started = time.perf_counter()
        crops, boxes, centers, ids, failed, bbox_meta = [], [], [], [], [], []
        crop_elapsed = 0.
        for cid in camera_ids:
            try:
                rgb = views_rgb[cid]
                bbox, metadata = select_rgb_bbox(detection_meta.get(cid, {}),
                                                image_size=(rgb.shape[1],rgb.shape[0]))
                crop, expanded, center, elapsed = self.model.prepare(rgb,bbox)
                crops.append(crop)
                boxes.append(expanded)
                centers.append(center)
                ids.append(cid)
                crop_elapsed += elapsed
                bbox_meta.append(dict(camera_id=cid, **metadata))
            except (ValueError, KeyError) as exc:
                failed.append(dict(camera_id=cid, reason=str(exc)))
        if len(ids) < 2:
            raise ValueError(f'RGB pose prior requires two valid image crops: {failed}')
        output, network_elapsed = self.model.forward_prepared_batch(crops,np.asarray(boxes),np.asarray(centers))
        fusion_start = time.perf_counter()
        rotations, accepted, agreement = fuse_local_rotation_predictions(output['pred_rot_mats'])
        fusion_elapsed = time.perf_counter()-fusion_start
        hints = dict(rgb_rotations=rotations, rgb_agreement=agreement)
        root_meta = None
        root_elapsed = 0.
        if projection_matrices is not None:
            root_started = time.perf_counter()
            from perception.benchmark.hybrik_multiview import crop_uv_to_pixels, triangulate_root
            try:
                projections = np.asarray(projection_matrices)
                if projections.shape != (len(camera_ids), 3, 4):
                    raise ValueError('RGB root projection matrix order/shape mismatch')
                pelvis_pixels = crop_uv_to_pixels(output['pred_uvd_jts'].reshape(len(ids), -1, 3),
                                                  np.asarray(boxes))[:, 0]
                selected_projections = projections[[camera_ids.index(cid) for cid in ids]]
                pelvis, inliers, errors = triangulate_root(pelvis_pixels, selected_projections)
                hints['rgb_pelvis_world_m'] = pelvis
                root_meta = dict(accepted=True, pelvis_world_m=pelvis.tolist(),
                                 used_cameras=[cid for cid, ok in zip(ids, inliers) if ok],
                                 reprojection_errors_px=[float(e) if np.isfinite(e) else None for e in errors],
                                 min_views=3, max_error_px=15.0)
            except (ValueError, KeyError) as exc:
                root_meta = dict(accepted=False, reason=str(exc))
            root_elapsed = time.perf_counter() - root_started
        metadata = dict(camera_ids=ids, failed_views=failed, bbox_metadata=bbox_meta,
                        accepted_joint_mask=accepted.tolist(), joint_agreement=agreement.tolist(),
                        crop_elapsed_s=crop_elapsed, batch_forward_elapsed_s=float(network_elapsed),
                        fusion_elapsed_s=fusion_elapsed, wall_elapsed_s=time.perf_counter()-started,
                        root_triangulation_elapsed_s=root_elapsed,
                        timing_policy='RGB wall includes crop, tensor transfers, full batch forward, output conversion, SO(3) fusion and optional calibrated root triangulation; no image decode, DWPose or SMPL-X refinement',
                        beta_policy='RGB beta does not replace the frozen initialization')
        if root_meta is not None:
            metadata['root_triangulation'] = root_meta
        return hints, metadata
