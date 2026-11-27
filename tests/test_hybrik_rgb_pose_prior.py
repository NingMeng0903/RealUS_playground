from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from projects.genesis_ue_sync.multiview_realtime.easymocap import moment_pipeline as mp
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import SyncedMultiviewFrame


CAMERA_IDS = ["cam1", "cam2"]


def _frontend_parts() -> dict[str, np.ndarray]:
    parts: dict[str, np.ndarray] = {}
    for name, count in (("keypoints3d", 25), ("handl3d", 21), ("handr3d", 21)):
        block = np.zeros((count, 4), dtype=np.float32)
        block[:, 3] = 1.0
        parts[name] = block
    return parts


def _backend_result() -> dict[str, np.ndarray]:
    return {
        "Rh": np.zeros((3,), dtype=np.float32),
        "Th": np.zeros((3,), dtype=np.float32),
        "poses": np.zeros((165,), dtype=np.float32),
        "shapes": np.zeros((10,), dtype=np.float32),
        "expression": np.zeros((10,), dtype=np.float32),
        "joints_world_m": np.zeros((55, 3), dtype=np.float32),
        "vertices": np.zeros((3, 3), dtype=np.float32),
        "faces": np.zeros((1, 3), dtype=np.int32),
        "pelvis_world_m": np.zeros((3,), dtype=np.float32),
    }


def _patch_frontend(
    monkeypatch,
    *,
    quality2d: bool = True,
    quality3d: bool = True,
    quality3d_report: dict | None = None,
) -> None:
    views = {camera_id: np.zeros((8, 8, 3), dtype=np.uint8) for camera_id in CAMERA_IDS}

    monkeypatch.setattr(
        mp,
        "camera_arrays",
        lambda *_args, **_kwargs: ({"P": np.zeros((2, 3, 4), dtype=np.float64)}, None),
    )
    monkeypatch.setattr(
        mp,
        "easymocap_fit_runtime_from_pose_backend",
        lambda _pose_backend: ({}, {}, False, "raw_inlier_2d"),
    )
    monkeypatch.setattr(mp, "triangulate_bodyhand_keypoints3d", lambda *_args, **_kwargs: _frontend_parts())
    monkeypatch.setattr(
        mp,
        "evaluate_easymocap_annot_quality",
        lambda *_args, **_kwargs: (quality2d, {"ok": quality2d}),
    )
    monkeypatch.setattr(
        mp,
        "evaluate_bodyhand3d_quality",
        lambda *_args, **_kwargs: (
            quality3d,
            dict(quality3d_report)
            if quality3d_report is not None
            else {"ok": quality3d},
        ),
    )

    class Detector:
        def infer_easymocap_annot_multiview(self, _views_rgb, _camera_ids):
            annots = {camera_id: {} for camera_id in CAMERA_IDS}
            metadata = {camera_id: {} for camera_id in CAMERA_IDS}
            return annots, metadata, {}

    _patch_frontend.views = views  # type: ignore[attr-defined]
    _patch_frontend.detector = Detector()  # type: ignore[attr-defined]


def _call(
    monkeypatch,
    tmp_path,
    *,
    prior,
    backend,
    quality3d: bool = True,
    quality3d_report: dict | None = None,
    triangulator=None,
):
    _patch_frontend(
        monkeypatch,
        quality3d=quality3d,
        quality3d_report=quality3d_report,
    )
    synced = SyncedMultiviewFrame(
        frame_index=12,
        views_rgb=_patch_frontend.views,  # type: ignore[attr-defined]
        metadata_by_camera={},
        timestamp_ns=987654,
    )
    cfg = SimpleNamespace(pose_backend={})
    return mp.process_one_hybrik_moment(
        moment_dir=tmp_path / "moment",
        synced=synced,
        cfg=cfg,
        calibration=SimpleNamespace(),
        detector=_patch_frontend.detector,  # type: ignore[attr-defined]
        camera_ids=list(CAMERA_IDS),
        hybrik_backend=backend,
        thres2d=0.15,
        max_repro_error=50.0,
        keep_rejected=True,
        write_debug_images=False,
        rgb_pose_prior=prior,
        triangulator=triangulator,
    )


def test_rgb_pose_prior_hints_reach_backend_and_keep_timestamp(monkeypatch, tmp_path) -> None:
    class Backend:
        gender = "male"

        def __init__(self):
            self.calls = []

        def step(self, _points, *, timestamp_ns=None, rgb_rotations=None, rgb_agreement=None):
            self.calls.append((timestamp_ns, rgb_rotations, rgb_agreement))
            return _backend_result()

    class Prior:
        def __init__(self):
            self.calls = 0

        def infer(self, views_rgb, detection_meta, camera_ids):
            self.calls += 1
            assert set(views_rgb) == set(CAMERA_IDS)
            assert set(detection_meta) == set(CAMERA_IDS)
            assert camera_ids == CAMERA_IDS
            return (
                {
                    "rgb_rotations": np.tile(np.eye(3, dtype=np.float32), (55, 1, 1)),
                    "rgb_agreement": np.ones((55,), dtype=np.float32),
                },
                {"accepted": True, "source": "fake"},
            )

    backend = Backend()
    prior = Prior()
    summary = _call(monkeypatch, tmp_path, prior=prior, backend=backend)

    assert summary["fit_ok"] is True
    assert prior.calls == 1
    assert summary["hybrik_rgb"] == {"accepted": True, "source": "fake"}
    assert backend.calls[0][0] == 987654
    assert backend.calls[0][1].shape == (55, 3, 3)
    assert backend.calls[0][2].shape == (55,)


def test_observation_policy_metadata_is_written_to_result_and_summary(
    monkeypatch, tmp_path
) -> None:
    geometry_metadata = {
        "rejected_mask": [False, True, False],
        "hands": {"left": {"rejected_hand_count": 1}},
        "rejected_count": 1,
    }

    class Backend:
        gender = "male"

        def step(self, _points, **_kwargs):
            result = _backend_result()
            result.update(
                {
                    "observation_geometry_filter": True,
                    "observation_geometry_metadata": json.dumps(geometry_metadata),
                    "occlusion_rotation_weight": 0.35,
                    "leg_occlusion_weights": np.asarray(
                        [1.0, 0.0, 0.5, 0.0], dtype=np.float32
                    ),
                }
            )
            return result

    summary = _call(monkeypatch, tmp_path, prior=None, backend=Backend())
    assert summary["observation_geometry_filter"] is True
    assert summary["observation_geometry_metadata_parsed"] == geometry_metadata
    assert summary["observation_geometry_rejected_mask"] == [False, True, False]
    assert summary["observation_geometry_rejected_count"] == 1
    assert summary["observation_geometry_hand_metadata"] == geometry_metadata["hands"]
    assert summary["occlusion_rotation_weight"] == 0.35
    assert summary["leg_occlusion_weights"] == [1.0, 0.0, 0.5, 0.0]
    assert summary["leg_occlusion_mask"] == [True, False, True, False]

    with np.load(tmp_path / "moment" / "smplx_result.npz") as data:
        assert bool(data["observation_geometry_filter"].item()) is True
        assert json.loads(str(data["observation_geometry_metadata"].item())) == geometry_metadata
        np.testing.assert_allclose(data["occlusion_rotation_weight"].item(), 0.35)
        np.testing.assert_allclose(
            data["leg_occlusion_weights"], [1.0, 0.0, 0.5, 0.0]
        )


_PARTIAL_BODY_QUALITY = {
    "ok": False,
    "reasons": ["min_core_body25"],
    "body3d": {
        "valid_body25": 21,
        "valid_core_body25": 3,
        "valid_shape_limbs": 10,
        "reasons": ["min_core_body25"],
    },
}


def _rgb_hints(root=None):
    hints = {
        "rgb_rotations": np.tile(np.eye(3, dtype=np.float32), (55, 1, 1)),
        "rgb_agreement": np.ones((55,), dtype=np.float32),
    }
    if root is not None:
        hints["rgb_pelvis_world_m"] = np.asarray(root, dtype=np.float32)
    return hints


def test_rgb_root_fallback_is_strictly_disabled_by_default(monkeypatch, tmp_path) -> None:
    class Backend:
        gender = "male"
        rgb_root_fallback = False
        calls = 0

        def step(self, _points, **_kwargs):
            self.calls += 1
            return _backend_result()

    class Prior:
        calls = 0

        def infer(self, *_args, **_kwargs):
            self.calls += 1
            return _rgb_hints([1.0, 2.0, 3.0]), {"source": "fake"}

    backend = Backend()
    prior = Prior()
    summary = _call(
        monkeypatch,
        tmp_path,
        prior=prior,
        backend=backend,
        quality3d=False,
        quality3d_report=_PARTIAL_BODY_QUALITY,
    )
    assert summary["fit_ok"] is False
    assert summary["skip_reason"] == "bodyhand3d_quality"
    assert backend.calls == 0
    assert prior.calls == 0
    assert summary["bodyhand3d_quality"] == _PARTIAL_BODY_QUALITY


def test_rgb_root_fallback_allows_only_partial_body_with_measured_root(
    monkeypatch, tmp_path
) -> None:
    original_points = np.zeros((67, 4), dtype=np.float32)
    original_points[:, 3] = 1.0

    class Backend:
        gender = "male"
        rgb_root_fallback = True

        def __init__(self):
            self.calls = []

        def step(self, points, *, rgb_pelvis_world_m=None, **_kwargs):
            self.calls.append(
                (np.asarray(points).copy(), None if rgb_pelvis_world_m is None else np.asarray(rgb_pelvis_world_m).copy())
            )
            result = _backend_result()
            result.update({"root_source": "rgb_triangulated_pelvis", "rgb_root_used": True})
            return result

    class Prior:
        calls = 0

        def infer(self, _views_rgb, _detection_meta, _camera_ids, *, projection_matrices=None):
            self.calls += 1
            assert projection_matrices is not None
            assert np.asarray(projection_matrices).shape == (2, 3, 4)
            return _rgb_hints([1.0, 2.0, 3.0]), {"source": "fake"}

    backend = Backend()
    prior = Prior()
    summary = _call(
        monkeypatch,
        tmp_path,
        prior=prior,
        backend=backend,
        quality3d=False,
        quality3d_report=_PARTIAL_BODY_QUALITY,
    )
    assert summary["fit_ok"] is True
    assert summary["tracking_quality_mode"] == "partial_body_with_triangulated_rgb_pelvis"
    assert summary["bodyhand3d_quality"] == _PARTIAL_BODY_QUALITY
    assert prior.calls == 1
    np.testing.assert_allclose(backend.calls[0][0], original_points)
    np.testing.assert_allclose(backend.calls[0][1], [1.0, 2.0, 3.0])
    assert summary["root_source"] == "rgb_triangulated_pelvis"
    assert summary["rgb_root_used"] is True
    with np.load(tmp_path / "moment" / "smplx_result.npz") as data:
        assert str(data["root_source"].item()) == "rgb_triangulated_pelvis"
        assert bool(data["rgb_root_used"].item()) is True


def test_rgb_root_fallback_without_root_keeps_original_quality_rejection(
    monkeypatch, tmp_path
) -> None:
    class Backend:
        gender = "male"
        rgb_root_fallback = True
        calls = 0

        def step(self, _points, **_kwargs):
            self.calls += 1
            return _backend_result()

    class Prior:
        def infer(self, _views_rgb, _detection_meta, _camera_ids, *, projection_matrices=None):
            assert projection_matrices is not None
            return _rgb_hints(), {"source": "fake"}

    backend = Backend()
    summary = _call(
        monkeypatch,
        tmp_path,
        prior=Prior(),
        backend=backend,
        quality3d=False,
        quality3d_report=_PARTIAL_BODY_QUALITY,
    )
    assert summary["fit_ok"] is False
    assert summary["skip_reason"] == "bodyhand3d_quality"
    assert summary["rgb_root_rejection"] == "missing_rgb_pelvis_world_m"
    assert backend.calls == 0
    assert summary["bodyhand3d_quality"] == _PARTIAL_BODY_QUALITY


@pytest.mark.parametrize(
    "quality_report",
    [
        {
            "ok": False,
            "reasons": ["min_core_body25"],
            "body3d": {"valid_body25": 11},
        },
        {
            "ok": False,
            "reasons": ["min_core_body25", "min_valid_shape_limbs"],
            "body3d": {"valid_body25": 21},
        },
    ],
)
def test_rgb_root_fallback_does_not_relax_low_count_or_other_quality_failures(
    monkeypatch, tmp_path, quality_report
) -> None:
    class Backend:
        gender = "male"
        rgb_root_fallback = True
        calls = 0

        def step(self, _points, **_kwargs):
            self.calls += 1
            return _backend_result()

    class Prior:
        calls = 0

        def infer(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("RGB root provider must not run for ineligible quality")

    backend = Backend()
    prior = Prior()
    summary = _call(
        monkeypatch,
        tmp_path,
        prior=prior,
        backend=backend,
        quality3d=False,
        quality3d_report=quality_report,
    )
    assert summary["fit_ok"] is False
    assert summary["skip_reason"] == "bodyhand3d_quality"
    assert backend.calls == 0
    assert prior.calls == 0
    assert summary["bodyhand3d_quality"] == quality_report


def test_quality_rejection_does_not_call_rgb_pose_prior(monkeypatch, tmp_path) -> None:
    _patch_frontend(monkeypatch, quality2d=False)
    synced = SyncedMultiviewFrame(
        frame_index=12,
        views_rgb=_patch_frontend.views,  # type: ignore[attr-defined]
        metadata_by_camera={},
        timestamp_ns=987654,
    )

    class Backend:
        gender = "male"
        calls = 0

        def step(self, _points, **_kwargs):
            self.calls += 1
            return _backend_result()

    class Prior:
        calls = 0

        def infer(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("RGB prior must run after the frontend quality gate")

    backend = Backend()
    prior = Prior()
    summary = mp.process_one_hybrik_moment(
        moment_dir=tmp_path / "rejected",
        synced=synced,
        cfg=SimpleNamespace(pose_backend={}),
        calibration=SimpleNamespace(),
        detector=_patch_frontend.detector,  # type: ignore[attr-defined]
        camera_ids=list(CAMERA_IDS),
        hybrik_backend=backend,
        thres2d=0.15,
        max_repro_error=50.0,
        keep_rejected=True,
        write_debug_images=False,
        rgb_pose_prior=prior,
    )

    assert summary["fit_ok"] is False
    assert summary["skip_reason"] == "pose2d_quality"
    assert prior.calls == 0
    assert backend.calls == 0


def test_rgb_pose_prior_value_error_is_explicit_rejection(monkeypatch, tmp_path) -> None:
    class Backend:
        gender = "male"
        calls = 0

        def step(self, _points, **_kwargs):
            self.calls += 1
            return _backend_result()

    class Prior:
        def infer(self, *_args, **_kwargs):
            raise ValueError("only one valid crop")

    backend = Backend()
    summary = _call(monkeypatch, tmp_path, prior=Prior(), backend=backend)

    assert summary["fit_ok"] is False
    assert summary["skip_reason"] == "hybrik_rgb_pose"
    assert summary["rgb_rejection"] == "only one valid crop"
    assert summary["hybrik_rgb"]["accepted"] is False
    assert backend.calls == 0


@pytest.mark.parametrize("quality3d,rgb_error", [(True, False), (True, True), (False, True)])
def test_parallel_rgb_waits_for_geometry_gate_before_stateful_fit(
    monkeypatch, tmp_path, quality3d, rgb_error,
):
    events = []

    class Job:
        elapsed_s = .125

        def result(self):
            events.append("collect")
            return _frontend_parts()

    class Triangulator:
        def triangulate(self, *args, **kwargs):
            raise AssertionError("blocking method must not be used for overlap")

        def submit(self, *args, **kwargs):
            events.append("submit")
            assert kwargs["compact_diagnostics"] is True
            return Job()

    class Prior:
        def infer(self, *args, **kwargs):
            events.append("rgb")
            if rgb_error:
                raise ValueError("bad RGB crop")
            return dict(rgb_rotations=np.tile(np.eye(3), (55, 1, 1)),
                        rgb_agreement=np.ones(55)), dict(wall_elapsed_s=.1)

    class Backend:
        gender = "neutral"

        def step(self, points, *, timestamp_ns, rgb_rotations, rgb_agreement):
            events.append("fit")
            assert timestamp_ns == 987654
            assert points.shape == (67, 4)
            return _backend_result()

    summary = _call(monkeypatch, tmp_path, prior=Prior(), backend=Backend(),
                    quality3d=quality3d, triangulator=Triangulator())
    assert events[:3] == ["submit", "rgb", "collect"]
    assert summary["frontend_includes_rgb"] is True
    assert summary["triangulation_elapsed_s"] == .125
    if not quality3d:
        assert summary["skip_reason"] == "bodyhand3d_quality"
        assert "fit" not in events
    elif rgb_error:
        assert summary["skip_reason"] == "hybrik_rgb_pose"
        assert "fit" not in events
    else:
        assert summary["fit_ok"] is True
        assert events == ["submit", "rgb", "collect", "fit"]
