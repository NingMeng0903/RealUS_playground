from __future__ import annotations

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.easymocap.bodyhandface_viz import (
    draw_keypoints3d_repro,
)
from projects.genesis_ue_sync.multiview_realtime.easymocap.delayed_smplx import (
    _initialize_roots_from_body25,
    estimate_body25_root_offsets,
    stack_bodyhand_keypoints3d,
)
from projects.genesis_ue_sync.multiview_realtime.easymocap.moment_pipeline import (
    _passes_reprojection_gate,
)


def test_reprojection_publication_boundary_is_inclusive() -> None:
    assert _passes_reprojection_gate([49.9], 50.0)
    assert _passes_reprojection_gate([50.0], 50.0)
    assert not _passes_reprojection_gate([50.1], 50.0)
    assert not _passes_reprojection_gate([], 50.0)


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

