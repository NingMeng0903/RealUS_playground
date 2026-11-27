from __future__ import annotations

import json
import time
from concurrent.futures import Future
from typing import Any
from pathlib import Path

import numpy as np
import pytest

from projects.genesis_ue_sync.multiview_realtime.easymocap.delayed_smplx import (
    triangulate_bodyhand_keypoints3d,
)
from projects.genesis_ue_sync.multiview_realtime.triangulation.dlt import (
    TriangulationConfig,
)
from projects.genesis_ue_sync.multiview_realtime.triangulation.parallel import (
    BodyHandTriangulator,
)
from projects.genesis_ue_sync.multiview_realtime.triangulation import parallel as parallel_module


def _cameras() -> np.ndarray:
    focal = 500.0
    cx, cy = 320.0, 240.0
    matrices = []
    for center_x in (-1.0, 0.0, 1.0):
        matrices.append(
            np.asarray(
                [
                    [focal, 0.0, cx, -focal * center_x],
                    [0.0, focal, cy, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            )
        )
    return np.stack(matrices, axis=0)


def _project(points: np.ndarray, P_all: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([points, np.ones((len(points), 1), dtype=np.float64)])
    projected = np.einsum("vij,nj->vni", P_all, homogeneous)
    xy = projected[:, :, :2] / projected[:, :, 2:3]
    return np.concatenate([xy, np.ones((*xy.shape[:2], 1), dtype=np.float64)], axis=2)


def _synthetic_inputs() -> tuple[dict[str, dict[str, np.ndarray]], list[str], np.ndarray, TriangulationConfig]:
    P_all = _cameras()
    rng = np.random.default_rng(44)
    body = np.column_stack(
        [rng.uniform(-0.45, 0.45, 25), rng.uniform(-0.65, 0.65, 25), rng.uniform(3.5, 5.0, 25)]
    )
    handl = np.column_stack(
        [rng.uniform(-0.9, -0.2, 21), rng.uniform(-0.4, 0.4, 21), rng.uniform(3.8, 4.8, 21)]
    )
    handr = np.column_stack(
        [rng.uniform(0.2, 0.9, 21), rng.uniform(-0.4, 0.4, 21), rng.uniform(3.8, 4.8, 21)]
    )
    projected = {
        "keypoints": _project(body, P_all),
        "handl2d": _project(handl, P_all),
        "handr2d": _project(handr, P_all),
    }
    # One occluded hand view should still be solvable from the other two.
    projected["handl2d"][1, 4, 2] = 0.0
    annots = {
        f"cam{index + 1}": {field: values[index].astype(np.float32) for field, values in projected.items()}
        for index in range(3)
    }
    camera_ids = ["cam1", "cam2", "cam3"]
    cfg = TriangulationConfig(
        confidence_threshold=0.2,
        min_conf=0.1,
        min_view=2,
        core_min_view=3,
        two_view_max_reproj_px=12.0,
        dist_max_px=25.0,
    )
    return annots, camera_ids, P_all, cfg


def _assert_nested_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_equal(left, right)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, (float, np.floating)):
        assert float(actual) == float(expected)
    else:
        assert actual == expected


def test_spawned_bodyhand_parts_match_serial_outputs_and_diagnostics():
    annots, camera_ids, P_all, cfg = _synthetic_inputs()
    serial_diag: dict[str, Any] = {}
    serial = triangulate_bodyhand_keypoints3d(
        annots,
        camera_ids,
        P_all,
        tri_cfg=cfg,
        include_hands=True,
        diagnostics=serial_diag,
    )

    parallel_diag: dict[str, Any] = {}
    with BodyHandTriangulator(workers=3) as parallel:
        parallel.preload()
        actual = parallel.triangulate(
            annots,
            camera_ids,
            P_all,
            cfg,
            include_hands=True,
            diagnostics=parallel_diag,
        )

    assert actual.keys() == serial.keys()
    for key in serial:
        np.testing.assert_array_equal(actual[key], serial[key])
    _assert_nested_equal(parallel_diag, serial_diag)


def test_disabled_hands_keep_zero_aliases_and_diagnostics():
    annots, camera_ids, P_all, cfg = _synthetic_inputs()
    serial_diag: dict[str, Any] = {}
    serial = triangulate_bodyhand_keypoints3d(
        annots,
        camera_ids,
        P_all,
        tri_cfg=cfg,
        include_hands=False,
        diagnostics=serial_diag,
    )
    parallel_diag: dict[str, Any] = {}
    with BodyHandTriangulator(workers=1) as parallel:
        actual = parallel.triangulate(
            annots,
            camera_ids,
            P_all,
            cfg,
            include_hands=False,
            diagnostics=parallel_diag,
        )
    for key in serial:
        np.testing.assert_array_equal(actual[key], serial[key])
    _assert_nested_equal(parallel_diag, serial_diag)
    np.testing.assert_array_equal(actual["handl3d"], np.zeros((21, 4), dtype=np.float32))
    np.testing.assert_array_equal(actual["handr3d"], np.zeros((21, 4), dtype=np.float32))


def test_pool_rejects_invalid_worker_count_and_closes_after_worker_error():
    with pytest.raises(ValueError, match="at least 1"):
        BodyHandTriangulator(workers=0)

    annots, camera_ids, P_all, cfg = _synthetic_inputs()
    del annots["cam1"]["handr2d"]
    parallel = BodyHandTriangulator(workers=2)
    job = parallel.submit(annots, camera_ids, P_all, cfg, include_hands=True)
    with pytest.raises(KeyError):
        job.result()
    parallel.close()
    parallel.close()
    assert parallel._executor is None
    assert job.elapsed_s is not None


def test_submit_does_not_wait_and_elapsed_excludes_late_result_collection(monkeypatch):
    annots, camera_ids, P_all, cfg = _synthetic_inputs()
    futures = []

    class DeferredExecutor:
        def submit(self, *args):
            future = Future()
            futures.append(future)
            return future

    parallel = BodyHandTriangulator(workers=3)
    executor = DeferredExecutor()
    monkeypatch.setattr(parallel, "_ensure_executor", lambda: executor)
    clock = [100.0]
    monkeypatch.setattr(parallel_module.time, "perf_counter", lambda: clock[0])
    diagnostics = {}
    job = parallel.submit(annots, camera_ids, P_all, cfg, diagnostics=diagnostics)
    assert len(futures) == 3
    assert all(not future.done() for future in futures)
    assert job.elapsed_s is None
    assert diagnostics == {}
    for index, (future, joints) in enumerate(zip(futures, (25, 21, 21))):
        clock[0] = 101.0 + index
        future.set_result((np.full((joints, 4), index, dtype=np.float32), {"part": index}))
    assert job.elapsed_s == 3.0
    clock[0] = 150.0  # Stand in for independent RGB work before collecting CPU results.
    result = job.result()
    assert job.elapsed_s == 3.0
    assert result["keypoints3d"].shape == (25, 4)
    np.testing.assert_array_equal(result["handr3d"], np.full((21, 4), 2, dtype=np.float32))
    assert diagnostics == {"body25": {"part": 0}, "handl": {"part": 1}, "handr": {"part": 2}}


def test_async_compact_diagnostics_match_persistence_without_changing_outputs():
    from projects.genesis_ue_sync.multiview_realtime.easymocap.moment_pipeline import (
        _compact_triangulation_diagnostics,
    )

    annots, camera_ids, P_all, cfg = _synthetic_inputs()
    # Exercise the no-selected-candidate branch as well as selected hypotheses.
    for camera_id in camera_ids:
        annots[camera_id]["handl2d"][0, 2] = 0.0
    serial_diag = {}
    expected = triangulate_bodyhand_keypoints3d(
        annots, camera_ids, P_all, tri_cfg=cfg, diagnostics=serial_diag,
    )
    compact_diag = {}
    with BodyHandTriangulator(workers=3) as parallel:
        parallel.preload()
        job = parallel.submit(
            annots, camera_ids, P_all, cfg, diagnostics=compact_diag,
            compact_diagnostics=True,
        )
        actual = job.result()
        assert job.elapsed_s is not None and job.elapsed_s > 0.0
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    expected_diag = {key: _compact_triangulation_diagnostics(value)
                     for key, value in serial_diag.items()}
    _assert_nested_equal(compact_diag, expected_diag)
    assert "candidate_hypotheses" not in compact_diag["handl"]["joint_details"][0]


def test_actual_full_simcc_fixture_parallel_microbenchmark():
    fixture_root = Path(
        "outputs/realus_sync/tests/diagnostics_cameras/"
        "hybrik_rate_optimization/full_simcc_profile"
    )
    npz_path = fixture_root / "frame_0120_inputs.npz"
    capture_path = fixture_root / "frame_0120_capture.json"
    if not npz_path.is_file() or not capture_path.is_file():
        pytest.skip("local full-SimCC fixture is unavailable")

    arrays = dict(np.load(npz_path))
    capture = json.loads(capture_path.read_text())
    cfg = TriangulationConfig(**capture["triangulation_config"])
    camera_ids = list(capture["camera_ids"])
    annots = {}
    metadata = {}
    for index, camera_id in enumerate(camera_ids):
        annots[camera_id] = {
            field: arrays[field][index]
            for field in ("keypoints", "handl2d", "handr2d")
        }
        metadata[camera_id] = {}
        for field in ("keypoints", "handl2d", "handr2d"):
            metadata[camera_id][field] = {
                name.split("__", 1)[1]: arrays[f"{field}__{name.split('__', 1)[1]}"][index]
                for name in (
                    f"{field}__candidate_xy",
                    f"{field}__candidate_probabilities",
                    f"{field}__variance_px2",
                )
                if name in arrays
            }

    serial_diag: dict[str, Any] = {}
    serial_times = []
    for _ in range(3):
        serial_diag.clear()
        serial_start = time.perf_counter()
        serial = triangulate_bodyhand_keypoints3d(
            annots,
            camera_ids,
            arrays["P"],
            tri_cfg=cfg,
            diagnostics=serial_diag,
            observation_meta_by_cam=metadata,
        )
        serial_times.append(time.perf_counter() - serial_start)
    parallel_diag: dict[str, Any] = {}
    with BodyHandTriangulator(workers=3) as parallel:
        parallel.preload()
        times = []
        for _ in range(3):
            parallel_diag.clear()
            started = time.perf_counter()
            actual = parallel.triangulate(
                annots,
                camera_ids,
                arrays["P"],
                cfg,
                diagnostics=parallel_diag,
                observation_meta_by_cam=metadata,
            )
            times.append(time.perf_counter() - started)
    for key in serial:
        np.testing.assert_array_equal(actual[key], serial[key])
    _assert_nested_equal(parallel_diag, serial_diag)
    print(
        "full_simcc_frame_0120_cpu_s",
        {"serial_median": float(np.median(serial_times)), "parallel_median": float(np.median(times))},
    )
