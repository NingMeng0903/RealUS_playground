from __future__ import annotations

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.easymocap.bodyhandface_viz import (
    draw_keypoints3d_repro,
)
from projects.genesis_ue_sync.multiview_realtime.easymocap.delayed_smplx import (
    _LinearConfidenceRobustBody25Loss,
    _initialize_roots_from_body25,
    body25_long_bone_diagnostics,
    estimate_body25_root_offsets,
    fusion_to_smplx_joint_diagnostics,
    mask_body25_2d_to_triangulation_inliers,
    select_pose_frame_arrays,
    stack_bodyhand_keypoints3d,
)
from projects.genesis_ue_sync.multiview_realtime.easymocap.moment_pipeline import (
    _body25_annots_from_temporal_inliers,
    _bodyhand_annots_from_selected_inliers,
    _burst_frame_score,
    _passes_final_publication_gate,
    _passes_reprojection_gate,
    _temporal_complete_body25,
    _temporal_select_body25_hypotheses,
)


def test_reprojection_publication_boundary_is_inclusive() -> None:
    assert _passes_reprojection_gate([49.9], 50.0)
    assert _passes_reprojection_gate([50.0], 50.0)
    assert not _passes_reprojection_gate([50.1], 50.0)
    assert not _passes_reprojection_gate([], 50.0)


def test_pose_extract_uses_one_synced_frame() -> None:
    kp3ds = np.zeros((12, 25, 4), dtype=np.float32)
    kp2ds = np.zeros((12, 4, 25, 3), dtype=np.float32)
    bboxes = np.zeros((12, 4, 5), dtype=np.float32)
    kp3ds[7, 0, :3] = (1.0, 2.0, 3.0)
    sliced, *_ = select_pose_frame_arrays(kp3ds, kp2ds, bboxes, None, 7)
    assert sliced.shape[0] == 1
    np.testing.assert_allclose(sliced[0, 0, :3], (1.0, 2.0, 3.0))
    all_frames, *_ = select_pose_frame_arrays(kp3ds, kp2ds, bboxes, None, None)
    assert all_frames.shape[0] == 12


def test_bed_penetration_count_is_not_a_publication_gate() -> None:
    kwargs = {"core_ok": True, "foot_ok": True, "reprojection_ok": True}
    assert _passes_final_publication_gate(**kwargs, bed_penetrating_verts=0)
    assert _passes_final_publication_gate(**kwargs, bed_penetrating_verts=9999)
    assert not _passes_final_publication_gate(
        core_ok=True,
        foot_ok=True,
        reprojection_ok=False,
        bed_penetrating_verts=0,
    )


def test_body25_root_offset_uses_robust_core_median() -> None:
    predicted = np.zeros((2, 25, 3), dtype=np.float32)
    target = np.zeros((2, 25, 4), dtype=np.float32)
    target[..., 3] = 1.0
    expected = np.asarray([[0.02, -0.03, -0.08], [-0.01, 0.04, 0.06]], dtype=np.float32)
    target[..., :3] = expected[:, None, :]
    target[0, 2, :3] += np.asarray([1.0, 1.0, 1.0], dtype=np.float32)

    offsets, diagnostics = estimate_body25_root_offsets(predicted, target)

    np.testing.assert_allclose(offsets, expected, atol=1.0e-6)
    assert diagnostics["applied_frames"] == 2
    assert diagnostics["per_frame"][0]["valid_core_joints"] == 11


class _TranslationBodyModel:
    shapedirs = None
    expr_dirs = None

    def __call__(
        self,
        *,
        Rh: np.ndarray,
        Th: np.ndarray,
        poses: np.ndarray,
        shapes: np.ndarray,
        return_verts: bool,
        return_tensor: bool,
    ) -> np.ndarray:
        del Rh, poses, shapes, return_verts, return_tensor
        base = np.zeros((len(Th), 25, 3), dtype=np.float32)
        return base + np.asarray(Th, dtype=np.float32)[:, None, :]


def test_root_initialization_applies_each_frame_offset() -> None:
    params = {
        "Rh": np.zeros((2, 3), dtype=np.float32),
        "Th": np.zeros((2, 3), dtype=np.float32),
        "poses": np.zeros((2, 6), dtype=np.float32),
        "shapes": np.zeros((1, 10), dtype=np.float32),
    }
    target = np.zeros((2, 25, 4), dtype=np.float32)
    target[..., 3] = 1.0
    expected = np.asarray([[0.01, 0.02, -0.07], [-0.02, 0.01, 0.05]], dtype=np.float32)
    target[..., :3] = expected[:, None, :]

    initialized, diagnostics = _initialize_roots_from_body25(
        _TranslationBodyModel(), params, target
    )

    np.testing.assert_allclose(initialized["Th"], expected, atol=1.0e-6)
    np.testing.assert_allclose(
        diagnostics["after_initialization"]["median_offset_m"],
        np.zeros(3),
        atol=1.0e-6,
    )


def test_green_reprojection_contains_body_and_both_hands() -> None:
    body = np.zeros((25, 4), dtype=np.float32)
    handl = np.zeros((21, 4), dtype=np.float32)
    handr = np.zeros((21, 4), dtype=np.float32)
    for index, hand in enumerate((handl, handr)):
        hand[:, 0] = np.linspace(-0.8 + index * 1.2, -0.25 + index * 1.2, 21)
        hand[:, 1] = np.linspace(-0.3, 0.3, 21)
        hand[:, 2] = 2.0
        hand[:, 3] = 1.0
    stacked = stack_bodyhand_keypoints3d(
        {"keypoints3d": body, "handl3d": handl, "handr3d": handr},
        pad_face_for_smplx=False,
    )
    assert stacked.shape == (67, 4)

    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    projection = np.asarray(
        [[80.0, 0.0, 64.0, 0.0], [0.0, 80.0, 64.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    rendered = draw_keypoints3d_repro(rgb, stacked, projection)

    green = (rendered[..., 1] > 0) & (rendered[..., 0] == 0) & (rendered[..., 2] == 0)
    assert int(np.sum(green[:, :64])) > 0
    assert int(np.sum(green[:, 64:])) > 0


def test_body25_robust_loss_applies_confidence_linearly() -> None:
    import torch
    from types import SimpleNamespace

    target_full = np.zeros((1, 25, 4), dtype=np.float32)
    target_full[0, 0, 3] = 1.0
    target_half = target_full.copy()
    target_half[0, 0, 3] = 0.5
    predicted = torch.zeros((1, 25, 3), dtype=torch.float32)
    predicted[0, 0, 0] = 0.02
    cfg = SimpleNamespace(device=torch.device("cpu"))

    full = _LinearConfidenceRobustBody25Loss(target_full, cfg, sigma_m=0.1).body(predicted)
    half = _LinearConfidenceRobustBody25Loss(target_half, cfg, sigma_m=0.1).body(predicted)

    assert float(full) > 0.0
    np.testing.assert_allclose(float(half), 0.5 * float(full), rtol=1.0e-6)


def test_hand_robust_loss_also_applies_confidence_linearly() -> None:
    import torch
    from types import SimpleNamespace

    target_full = np.zeros((1, 67, 4), dtype=np.float32)
    target_full[0, 25, 3] = 1.0
    target_half = target_full.copy()
    target_half[0, 25, 3] = 0.5
    predicted = torch.zeros((1, 67, 3), dtype=torch.float32)
    predicted[0, 25, 0] = 0.02
    cfg = SimpleNamespace(device=torch.device("cpu"))

    full = _LinearConfidenceRobustBody25Loss(target_full, cfg, sigma_m=0.1).hand(predicted)
    half = _LinearConfidenceRobustBody25Loss(target_half, cfg, sigma_m=0.1).hand(predicted)

    assert float(full) > 0.0
    np.testing.assert_allclose(float(half), 0.5 * float(full), rtol=1.0e-6)


def test_long_bone_diagnostics_requests_retry_over_two_cm() -> None:
    target = np.zeros((3, 25, 4), dtype=np.float32)
    predicted = np.zeros((3, 25, 3), dtype=np.float32)
    target[..., 3] = 1.0
    for src, dst in ((9, 10), (10, 11), (12, 13), (13, 14)):
        target[:, dst, 0] = target[:, src, 0] + 0.40
        predicted[:, dst, 0] = predicted[:, src, 0] + 0.37

    diagnostics = body25_long_bone_diagnostics(predicted, target, threshold_m=0.02)

    assert diagnostics["requires_shape_retry"]
    assert diagnostics["systematic_bones"]
    assert diagnostics["max_median_absolute_error_m"] > 0.02


def test_fusion_to_smplx_diagnostics_reports_per_joint_3d_and_2d() -> None:
    predicted = np.zeros((1, 25, 3), dtype=np.float32)
    predicted[..., 2] = 2.0
    target = np.zeros((1, 25, 4), dtype=np.float32)
    target[..., :3] = predicted
    target[..., 3] = 1.0
    target[0, 10, 0] += 0.03
    projection = np.asarray(
        [[[100.0, 0.0, 0.0, 0.0], [0.0, 100.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
        dtype=np.float64,
    )
    observed2d = np.zeros((1, 1, 25, 3), dtype=np.float32)
    observed2d[..., 2] = 1.0

    diagnostics = fusion_to_smplx_joint_diagnostics(predicted, target, observed2d, projection)

    assert len(diagnostics["per_joint"]) == 25
    np.testing.assert_allclose(diagnostics["per_joint"][10]["mean_3d_m"], 0.03, atol=1.0e-6)
    np.testing.assert_allclose(diagnostics["mean_2d_px"], 0.0, atol=1.0e-6)


def _candidate(
    hypothesis_id: int,
    xyz: tuple[float, float, float],
    cost: float,
    used_views: tuple[int, ...],
    *,
    selected: bool = False,
) -> dict[str, object]:
    return {
        "hypothesis_id": hypothesis_id,
        "xyz": list(xyz),
        "confidence": 0.8,
        "used_views": list(used_views),
        "inlier_mask": [view in used_views for view in range(4)],
        "candidate_ranks": [0 if view in used_views else -1 for view in range(4)],
        "mean_reprojection_error_px": cost,
        "max_reprojection_error_px": cost,
        "robust_cost": cost,
        "min_ray_angle_deg": 8.0,
        "geometry_ok": True,
        "selected": selected,
    }


def _burst_joint_fixture(
    middle_candidates: list[dict[str, object]],
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, object]], list[int]]:
    good_positions = ((0.0, 0.0, 2.0), (0.003, 0.0, 2.0), (0.006, 0.0, 2.0))
    parts: list[dict[str, np.ndarray]] = []
    diagnostics: list[dict[str, object]] = []
    for frame, good in enumerate(good_positions):
        body = np.zeros((25, 4), dtype=np.float32)
        candidates = (
            [_candidate(frame, good, 1.0, (2, 3), selected=True)]
            if frame != 1 else middle_candidates
        )
        selected = next(candidate for candidate in candidates if bool(candidate.get("selected")))
        body[21, :3] = np.asarray(selected["xyz"], dtype=np.float32)
        body[21, 3] = 0.8
        parts.append({"keypoints3d": body})
        diagnostics.append({
            "body25": {
                "joint_details": [{
                    "joint_index": 21,
                    "observed_views": [0, 1, 2, 3],
                    "used_views": list(selected["used_views"]),
                    "selected_hypothesis_id": selected["hypothesis_id"],
                    "candidate_hypotheses": candidates,
                    "reprojection_error_px": selected["mean_reprojection_error_px"],
                    "geometry_ok": True,
                    "status": "observed_high",
                }]
            }
        })
    return parts, diagnostics, [0, 33_333_333, 66_666_666]


def test_temporal_path_reselects_continuous_heel_candidate_and_rebuilds_2d_mask() -> None:
    jumping = _candidate(10, (0.13, 0.0, 2.0), 1.0, (0, 1, 3), selected=True)
    continuous = _candidate(11, (0.003, 0.0, 2.0), 2.0, (2, 3))
    parts, diagnostics, timestamps = _burst_joint_fixture([jumping, continuous])

    report = _temporal_select_body25_hypotheses(parts, diagnostics, timestamps)

    np.testing.assert_allclose(parts[1]["keypoints3d"][21, :3], continuous["xyz"])
    detail = diagnostics[1]["body25"]["joint_details"][0]
    assert detail["status"] == "temporal_reselected"
    assert detail["used_views"] == [2, 3]
    assert report == {"reselected_joints": 1, "rejected_joints": 0}

    raw = {
        f"cam{view + 1}": {"keypoints": np.ones((25, 3), dtype=np.float32)}
        for view in range(4)
    }
    masked = mask_body25_2d_to_triangulation_inliers(
        raw, list(raw), diagnostics[1]["body25"]
    )
    assert masked["cam1"]["keypoints"][21, 2] == 0.0
    assert masked["cam2"]["keypoints"][21, 2] == 0.0
    assert masked["cam3"]["keypoints"][21, 2] > 0.0
    assert masked["cam4"]["keypoints"][21, 2] > 0.0


def test_final_2d_inliers_use_selected_simcc_mode_pixels() -> None:
    camera_ids = ["cam1", "cam2"]
    raw = {
        cid: {"keypoints": np.ones((25, 3), dtype=np.float32)}
        for cid in camera_ids
    }
    candidate_xy = np.zeros((25, 2, 2), dtype=np.float32)
    candidate_xy[21, 0] = (100.0, 200.0)
    candidate_xy[21, 1] = (123.0, 234.0)
    detection_meta = {
        cid: {"simcc_easymocap": {"keypoints": {"candidate_xy": candidate_xy}}}
        for cid in camera_ids
    }
    body_diag = {"joint_details": [{
        "joint_index": 21,
        "used_views": [1],
        "selected_candidate_ranks": [-1, 1],
    }]}

    fitted = _body25_annots_from_temporal_inliers(
        raw, camera_ids, body_diag, detection_meta
    )

    assert fitted["cam1"]["keypoints"][21, 2] == 0.0
    np.testing.assert_allclose(fitted["cam2"]["keypoints"][21, :2], [123.0, 234.0])
    assert fitted["cam2"]["keypoints"][21, 2] == 1.0


def test_hand_2d_outlier_views_are_masked_before_smplx_fit() -> None:
    camera_ids = ["cam1", "cam2"]
    raw = {
        cid: {
            "keypoints": np.ones((25, 3), dtype=np.float32),
            "handl2d": np.ones((21, 3), dtype=np.float32),
            "handr2d": np.ones((21, 3), dtype=np.float32),
        }
        for cid in camera_ids
    }
    diagnostics = {
        "handl": {
            "joint_details": [{
                "joint_index": 5,
                "used_views": [1],
                "selected_candidate_ranks": [-1, 0],
                "selected_observations_xy": [None, [123.0, 234.0]],
            }]
        }
    }

    fitted = _bodyhand_annots_from_selected_inliers(
        raw,
        camera_ids,
        diagnostics,
        detection_meta={},
    )

    assert fitted["cam1"]["handl2d"][5, 2] == 0.0
    np.testing.assert_allclose(fitted["cam2"]["handl2d"][5, :2], [123.0, 234.0])
    assert fitted["cam2"]["handl2d"][5, 2] > 0.0


def test_temporal_path_rejects_unrecoverable_heel_jump_without_completion() -> None:
    jumping = _candidate(10, (0.13, 0.0, 2.0), 1.0, (0, 1, 3), selected=True)
    parts, diagnostics, timestamps = _burst_joint_fixture([jumping])

    report = _temporal_select_body25_hypotheses(parts, diagnostics, timestamps)
    completed = _temporal_complete_body25(parts, diagnostics, timestamps)

    assert report == {"reselected_joints": 0, "rejected_joints": 1}
    assert completed == 0
    assert np.all(parts[1]["keypoints3d"][21] == 0.0)
    detail = diagnostics[1]["body25"]["joint_details"][0]
    assert detail["status"] == "temporal_rejected"
    assert detail["used_views"] == []


def test_reference_score_prioritizes_foot_geometry_before_center_distance() -> None:
    indices = (0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24)

    def diagnostics(foot_error: float) -> dict[str, object]:
        return {"joint_details": [{
            "joint_index": joint,
            "geometry_ok": True,
            "used_views": [0, 1, 2],
            "reprojection_error_px": foot_error if joint in (11, 14, 19, 20, 21, 22, 23, 24) else 1.0,
        } for joint in indices]}

    off_center_good = _burst_frame_score(diagnostics(1.0), 0, 7)
    center_bad = _burst_frame_score(diagnostics(20.0), 7, 7)

    assert off_center_good > center_bad
