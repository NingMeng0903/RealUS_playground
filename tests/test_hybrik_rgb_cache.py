from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from perception.benchmark.hybrik_rgb import (
    OfficialHybrIKXRGB, cache_output_aliases, checkpoint_state_dict,
    official_working_directory, select_rgb_bbox,
)


def test_selected_pose_box_preserves_temporal_bbox_before_detector_or_points():
    metadata = {"selected_person": 1, "pose_boxes": [[0, 0, 10, 10], [20, 30, 120, 180]],
                "detector_boxes": [[1, 1, 11, 11], [40, 40, 100, 100]],
                "bbox_source": "temporal_cached",
                "easymocap": {"keypoints": [[50, 50, 1], [60, 60, 1], [70, 50, 1], [80, 80, 1]]}}
    box, policy = select_rgb_bbox(metadata, image_size=(1920, 1080))
    np.testing.assert_array_equal(box, [20, 30, 120, 180])
    assert policy["bbox_selection"] == "detection_meta.pose_boxes[selected_person]"
    assert policy["frontend_bbox_source"] == "temporal_cached"
    assert not policy["point_fallback"]


def test_rgb_landmark_fallback_filters_low_confidence_and_outside_points():
    metadata = {"easymocap": {"keypoints": [[10, 20, 1], [20, 30, 0.8], [30, 15, 1], [25, 35, 1],
                                           [90, 90, 0.1], [-50, 40, 1], [float("nan"), 20, 1]]},
                "gt_bbox": [0, 0, 1000, 1000]}
    box, policy = select_rgb_bbox(metadata, image_size=(100, 100))
    np.testing.assert_array_equal(box, [10, 15, 30, 35])
    assert policy["point_fallback"] and policy["fallback_point_count"] == 4
    with pytest.raises(ValueError, match="No selected RGB bbox"):
        select_rgb_bbox({"gt_bbox": [0, 0, 100, 100]}, image_size=(100, 100))


def test_checkpoint_unwrap_does_not_filter_keys_or_partially_strip_prefix():
    state = {"module.backbone.weight": 1, "module.smplx.buffer": 2}
    assert checkpoint_state_dict({"model": state}) == {"backbone.weight": 1, "smplx.buffer": 2}
    mixed = {"module.a": 1, "b": 2}
    assert checkpoint_state_dict(mixed) is mixed
    with pytest.raises(ValueError):
        checkpoint_state_dict({})


def test_official_constructor_cwd_restored_on_exception(tmp_path):
    original = Path.cwd()
    with pytest.raises(RuntimeError):
        with official_working_directory(tmp_path):
            assert Path.cwd() == tmp_path
            raise RuntimeError("construction failed")
    assert Path.cwd() == original


def test_cache_aliases_preserve_view_identity_xyz_units_and_coordinate_sigma():
    phi = np.zeros((2, 54, 2), np.float32)
    phi[0, :, 0], phi[1, :, 1] = 2, 3
    rotations = np.tile(np.eye(3, dtype=np.float32), (2, 55, 1, 1))
    xyz = np.arange(2 * 71 * 3, dtype=np.float32).reshape(2, 71 * 3)
    sigma = np.full((2, 71, 1), 0.2, np.float32)
    raw = {"pred_phi": phi, "pred_theta_mat": rotations.reshape(2, -1),
           "pred_xyz_hybrik_struct": xyz, "pred_sigma": sigma}
    cached = cache_output_aliases(raw, depth_factor_m=2.2)
    np.testing.assert_array_equal(cached["pred_phi"], phi)
    np.testing.assert_array_equal(cached["pred_phi_unit"][0, :, 0], 1)
    np.testing.assert_array_equal(cached["pred_phi_unit"][1, :, 1], 1)
    np.testing.assert_array_equal(cached["pred_rot_mats"], rotations)
    np.testing.assert_allclose(cached["pred_xyz"], xyz.reshape(2, 71, 3)[:, :55] * 2.2)
    np.testing.assert_array_equal(cached["sigma"], sigma)
    assert "phi_confidence" not in cached
    assert cached["pred_phi"].shape == (2, 54, 2)


class _BatchModel:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def __call__(self, image, *, flip_test, bboxes, img_center):
        self.calls.append((image, flip_test, bboxes, img_center))
        return self.output


def _prepared_batch_fixture(batch_size: int = 5):
    crops = [torch.full((3, 256, 256), index + 1, dtype=torch.float64)
             for index in range(batch_size)]
    boxes = np.arange(batch_size * 4, dtype=np.float64).reshape(batch_size, 4)
    centers = np.arange(batch_size * 2, dtype=np.float64).reshape(batch_size, 2)

    phi = torch.zeros((batch_size, 54, 2), dtype=torch.float32)
    phi[:, :, 0] = torch.arange(1, batch_size + 1, dtype=torch.float32)[:, None]
    rotations = torch.eye(3, dtype=torch.float32).repeat(batch_size, 55, 1, 1)
    structured = torch.arange(batch_size * 71 * 3, dtype=torch.float32).reshape(
        batch_size, 71 * 3)
    sigma = torch.full((batch_size, 71, 1), 0.2, dtype=torch.float32)
    output = {
        "pred_phi": phi,
        "pred_theta_mat": rotations.reshape(batch_size, -1),
        "pred_xyz_hybrik_struct": structured,
        "pred_sigma": sigma,
        "pred_uvd_jts": torch.arange(batch_size * 71 * 3, dtype=torch.float32).reshape(
            batch_size, 71, 3),
        "gt_output": 0,
    }
    return crops, boxes, centers, output


def test_forward_prepared_batch_calls_official_model_once_and_keeps_view_alignment():
    batch_size = 5
    crops, boxes, centers, output = _prepared_batch_fixture(batch_size)
    model = _BatchModel(output)
    wrapper = OfficialHybrIKXRGB.__new__(OfficialHybrIKXRGB)
    wrapper.torch = torch
    wrapper.device = torch.device("cpu")
    wrapper.flip_test = True
    wrapper.model = model
    wrapper.depth_factor_m = 2.2

    cached, elapsed = wrapper.forward_prepared_batch(crops, boxes, centers)

    assert isinstance(elapsed, float) and elapsed >= 0
    assert len(model.calls) == 1
    image, flip_test, called_boxes, called_centers = model.calls[0]
    assert flip_test is True
    assert image.dtype is torch.float32 and image.device.type == "cpu"
    assert called_boxes.dtype is torch.float32 and called_boxes.device.type == "cpu"
    assert called_centers.dtype is torch.float32 and called_centers.device.type == "cpu"
    for index, crop in enumerate(crops):
        torch.testing.assert_close(image[index], crop.to(torch.float32))
    torch.testing.assert_close(called_boxes, torch.as_tensor(boxes, dtype=torch.float32))
    torch.testing.assert_close(called_centers, torch.as_tensor(centers, dtype=torch.float32))

    expected_shapes = {
        "pred_phi": (batch_size, 54, 2),
        "pred_phi_unit": (batch_size, 54, 2),
        "pred_phi_nonzero_norm": (batch_size, 54),
        "pred_theta_mat": (batch_size, 55 * 9),
        "pred_rot_mats": (batch_size, 55, 3, 3),
        "pred_xyz_hybrik_struct": (batch_size, 71 * 3),
        "pred_xyz": (batch_size, 55, 3),
        "pred_xyz_camera_m": (batch_size, 55, 3),
        "pred_sigma": (batch_size, 71, 1),
        "pred_uvd_jts": (batch_size, 71, 3),
        "gt_output": (batch_size,),
    }
    assert set(expected_shapes) <= set(cached)
    for key, shape in expected_shapes.items():
        assert cached[key].shape == shape
    for value in cached.values():
        assert value.shape[0] == batch_size
    np.testing.assert_array_equal(cached["gt_output"], np.zeros(batch_size))
    np.testing.assert_array_equal(cached["pred_phi"][:, 0, 0], np.arange(1, batch_size + 1))


def test_forward_prepared_batch_rejects_empty_and_mismatched_batches():
    crops, boxes, centers, output = _prepared_batch_fixture()
    model = _BatchModel(output)
    wrapper = OfficialHybrIKXRGB.__new__(OfficialHybrIKXRGB)
    wrapper.torch = torch
    wrapper.device = torch.device("cpu")
    wrapper.flip_test = False
    wrapper.model = model
    wrapper.depth_factor_m = 2.2

    with pytest.raises(ValueError):
        wrapper.forward_prepared_batch([], np.empty((0, 4)), np.empty((0, 2)))
    with pytest.raises(ValueError):
        wrapper.forward_prepared_batch(crops, boxes[:-1], centers)
    with pytest.raises(ValueError):
        wrapper.forward_prepared_batch(crops, boxes, centers[:-1])
    assert model.calls == []
