from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from projects.genesis_ue_sync.multiview_realtime.triangulation import dlt as dlt_module
from projects.genesis_ue_sync.multiview_realtime.triangulation.dlt import (
    TriangulationConfig,
    triangulate_multiview,
)


def _cameras(n_views: int = 5) -> np.ndarray:
    intrinsic = np.diag([1000.0, 1000.0, 1.0])
    centers = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )[:n_views]
    return np.asarray([intrinsic @ np.c_[np.eye(3), -center] for center in centers])


def _project(point: np.ndarray, projections: np.ndarray) -> np.ndarray:
    homogeneous = np.r_[np.asarray(point, dtype=np.float64), 1.0]
    projected = np.asarray([projection @ homogeneous for projection in projections])
    return projected[:, :2] / projected[:, 2:3]


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        assert isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        np.testing.assert_array_equal(left, right, strict=True)
        return
    assert type(left) is type(right)
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right):
            _assert_nested_equal(left_value, right_value)
        return
    assert left == right


def _synthetic_multimodal_fixture() -> tuple[
    np.ndarray, np.ndarray, TriangulationConfig, dict[str, np.ndarray]
]:
    projections = _cameras()
    points = np.asarray(
        [
            [0.20, 0.30, 5.00],
            [-0.10, 0.15, 4.50],
            [0.25, -0.20, 5.50],
            [0.05, 0.05, 4.80],
        ],
        dtype=np.float64,
    )
    image_points = np.asarray([_project(point, projections) for point in points])
    keypoints = np.full((len(projections), 25, 3), np.nan, dtype=np.float32)
    keypoints[..., 2] = 0.0
    candidate_xy = np.zeros((len(projections), 25, 2, 2), dtype=np.float32)
    candidate_probabilities = np.zeros((len(projections), 25, 2), dtype=np.float32)
    variance_px2 = np.zeros((len(projections), 25, 2, 2), dtype=np.float32)

    # Every joint starts with a clean mode.  The first joint has a misleading
    # primary peak and a correct secondary peak, exercising mode rescue.
    for view in range(len(projections)):
        keypoints[view, :4, :2] = image_points[:, view, :]
        keypoints[view, :4, 2] = 0.95
        candidate_xy[view, :4, 0] = image_points[:, view, :]
        candidate_xy[view, :4, 1] = image_points[:, view, :]
        candidate_probabilities[view, :4] = 0.5
    candidate_xy[:, 0, 0] += np.asarray([18.0, -14.0], dtype=np.float32)
    candidate_probabilities[:, 0] = np.asarray([0.2, 0.8], dtype=np.float32)

    # Joint 1 has one occluded camera.  Its other four views remain valid.
    keypoints[4, 1, :2] = np.nan
    keypoints[4, 1, 2] = 0.0
    candidate_xy[4, 1] = np.nan
    candidate_probabilities[4, 1] = 0.0

    # Joint 2 is a core BODY25 joint with exactly three usable views.
    keypoints[3:, 2, :2] = np.nan
    keypoints[3:, 2, 2] = 0.0
    candidate_xy[3:, 2] = np.nan
    candidate_probabilities[3:, 2] = 0.0

    # Joint 3 has equal-probability tied modes at the same location.  The
    # selected rank is implementation-determined, but must remain identical.
    variance_px2[:, :4] = 1.0
    config = TriangulationConfig(
        confidence_threshold=0.2,
        min_conf=0.1,
        dist_max_px=30.0,
        two_view_max_reproj_px=12.0,
        two_view_min_ray_angle_deg=1.0,
        simcc_pair_hypotheses=2,
    )
    return keypoints, projections, config, {
        "candidate_xy": candidate_xy,
        "candidate_probabilities": candidate_probabilities,
        "variance_px2": variance_px2,
    }


def _run_with_counts(
    keypoints: np.ndarray,
    projections: np.ndarray,
    config: TriangulationConfig,
    metadata: dict[str, np.ndarray],
    *,
    cache_enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[np.ndarray, dict[str, Any], dict[str, int]]:
    counts = {"weights": 0, "dlt": 0, "rows": 0, "static_cost": 0}
    original_weight = dlt_module._observation_weight
    original_dlt = dlt_module._triangulate_linear
    original_rows = dlt_module._weighted_dlt_row_block
    original_static = dlt_module._hypothesis_cost_static_components

    def counting_weight(*args: Any, **kwargs: Any) -> float:
        counts["weights"] += 1
        return original_weight(*args, **kwargs)

    def counting_dlt(*args: Any, **kwargs: Any) -> np.ndarray | None:
        counts["dlt"] += 1
        return original_dlt(*args, **kwargs)

    def counting_rows(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray] | None:
        counts["rows"] += 1
        return original_rows(*args, **kwargs)

    def counting_static(*args: Any, **kwargs: Any) -> tuple[float, float, float]:
        counts["static_cost"] += 1
        return original_static(*args, **kwargs)

    monkeypatch.setattr(dlt_module, "_observation_weight", counting_weight)
    monkeypatch.setattr(dlt_module, "_triangulate_linear", counting_dlt)
    monkeypatch.setattr(dlt_module, "_weighted_dlt_row_block", counting_rows)
    monkeypatch.setattr(
        dlt_module, "_hypothesis_cost_static_components", counting_static
    )
    result, diagnostics = triangulate_multiview(
        keypoints,
        projections,
        config,
        observation_meta=metadata,
        _cache_enabled=cache_enabled,
    )
    return result, diagnostics, counts


def test_cached_hypotheses_are_bitwise_identical_and_reduce_repeated_work(monkeypatch) -> None:
    keypoints, projections, config, metadata = _synthetic_multimodal_fixture()
    uncached, uncached_diag, uncached_counts = _run_with_counts(
        keypoints,
        projections,
        config,
        metadata,
        cache_enabled=False,
        monkeypatch=monkeypatch,
    )
    cached, cached_diag, cached_counts = _run_with_counts(
        keypoints,
        projections,
        config,
        metadata,
        cache_enabled=True,
        monkeypatch=monkeypatch,
    )
    np.testing.assert_array_equal(cached, uncached, strict=True)
    _assert_nested_equal(cached_diag, uncached_diag)
    assert cached_counts["weights"] < uncached_counts["weights"]
    assert cached_counts["dlt"] < uncached_counts["dlt"]
    assert cached_counts["rows"] < uncached_counts["rows"]
    assert cached_counts["static_cost"] < uncached_counts["static_cost"]

    # Check the fixture's intended edge cases explicitly while the complete
    # diagnostics comparison above protects every hypothesis and gate.
    details = cached_diag["joint_details"]
    assert details[0]["selected_candidate_ranks"] == [1, 1, 1, 1, 1]
    assert len(details[1]["used_views"]) >= 3
    assert details[1]["observed_views"] == [0, 1, 2, 3]
    assert details[2]["observed_views"] == [0, 1, 2]
    assert len(details[2]["used_views"]) == 3
    assert all(rank in (0, 1) for rank in details[3]["selected_candidate_ranks"])


def _saved_five_camera_cases() -> list[tuple[np.ndarray, np.ndarray, TriangulationConfig]]:
    root = Path(__file__).resolve().parents[1]
    run = root / "outputs/realus_sync/tests/diagnostics_cameras/pose/optimized5_neutral_rgb_init"
    calibration_path = root / "outputs/realus_sync/tests/diagnostics_cameras/actual_optimized5_corrected/calibration.json"
    index_path = run / "index.json"
    moment_path = run / "moment_0000/moment.json"
    if not all(path.is_file() for path in (run / "index.json", calibration_path, moment_path)):
        return []
    index = json.loads(index_path.read_text(encoding="utf-8"))
    moment = json.loads(moment_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    camera_ids = list(index["camera_ids"])
    cameras = calibration["cameras"]
    projections = np.stack(
        [
            np.asarray(cameras[camera_id]["K"], dtype=np.float64)
            @ np.column_stack(
                (
                    np.asarray(cameras[camera_id]["R_world_to_camera"], dtype=np.float64),
                    np.asarray(cameras[camera_id]["t_world_to_camera_m"], dtype=np.float64),
                )
            )
            for camera_id in camera_ids
        ]
    )
    fields = ("keypoints", "handl2d", "handr2d")
    arrays = [
        np.asarray(
            [moment["detection_meta"][camera_id]["easymocap"][field] for camera_id in camera_ids],
            dtype=np.float64,
        )
        for field in fields
    ]
    config_fields = {field.name for field in dataclasses.fields(TriangulationConfig)}
    body25_config = {
        key: value
        for key, value in moment["triangulation"]["body25"].items()
        if key in config_fields
    }
    body25_config["simcc_pair_hypotheses"] = int(index["triangulation_pair_hypotheses"])
    config = TriangulationConfig(**body25_config)
    return [(array, projections, config) for array in arrays]


def test_saved_five_camera_decoded_annotations_have_exact_cpu_cache_parity() -> None:
    cases = _saved_five_camera_cases()
    if not cases:
        pytest.skip("saved five-camera CPU parity fixture is unavailable")
    for keypoints, projections, config in cases:
        before, before_diag = triangulate_multiview(
            keypoints, projections, config, _cache_enabled=False
        )
        after, after_diag = triangulate_multiview(
            keypoints, projections, config, _cache_enabled=True
        )
        np.testing.assert_array_equal(after, before, strict=True)
        _assert_nested_equal(after_diag, before_diag)


def _full_simcc_fixture_cases() -> list[
    tuple[int, dict[str, np.ndarray], TriangulationConfig]
]:
    root = (
        Path(__file__).resolve().parents[1]
        / "outputs/realus_sync/tests/diagnostics_cameras/hybrik_rate_optimization/full_simcc_profile"
    )
    cases: list[tuple[int, dict[str, np.ndarray], TriangulationConfig]] = []
    for frame in (15, 120):
        input_path = root / f"frame_{frame:04d}_inputs.npz"
        capture_path = root / f"frame_{frame:04d}_capture.json"
        if not input_path.is_file() or not capture_path.is_file():
            continue
        with np.load(input_path) as loaded:
            arrays = {name: np.asarray(value) for name, value in loaded.items()}
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        cases.append(
            (
                frame,
                arrays,
                TriangulationConfig(**capture["triangulation_config"]),
            )
        )
    return cases


def test_full_simcc_profile_fixtures_have_exact_cpu_cache_parity() -> None:
    cases = _full_simcc_fixture_cases()
    if not cases:
        pytest.skip("full-SimCC profile fixtures are unavailable")
    for frame, arrays, config in cases:
        for part in ("keypoints", "handl2d", "handr2d"):
            metadata = {
                name.split("__", 1)[1]: value
                for name, value in arrays.items()
                if name.startswith(part + "__")
            }
            before, before_diag = triangulate_multiview(
                arrays[part],
                arrays["P"],
                config,
                observation_meta=metadata,
                _cache_enabled=False,
            )
            after, after_diag = triangulate_multiview(
                arrays[part],
                arrays["P"],
                config,
                observation_meta=metadata,
                _cache_enabled=True,
            )
            np.testing.assert_array_equal(after, before, strict=True)
            _assert_nested_equal(after_diag, before_diag)
