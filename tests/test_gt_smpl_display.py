from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence
from projects.genesis_ue_sync.sim_platform.human_runtime import gt_smpl_display as display


class _OffsetRootModel:
    """Small SMPL-X-shaped stub whose local pelvis is deliberately non-zero."""

    faces = np.asarray([[0, 1, 2]], dtype=np.int32)

    def __init__(self) -> None:
        self.translations: list[np.ndarray] = []
        self._vertices0 = torch.tensor(
            [[[0.23, -0.11, 0.07], [1.23, -0.11, 0.07], [0.23, 0.89, 0.07]]],
            dtype=torch.float32,
        )
        self._joints0 = torch.zeros((1, 55, 3), dtype=torch.float32)
        self._joints0[:, 0] = torch.tensor([0.23, -0.11, 0.07])
        self._joints0[:, 1] = torch.tensor([0.5, 0.0, 0.0])

    def __call__(self, **kwargs):
        transl = kwargs["transl"].detach().cpu()
        self.translations.append(transl.numpy().copy())
        return SimpleNamespace(
            vertices=self._vertices0 + transl[:, None, :],
            joints=self._joints0 + transl[:, None, :],
        )


def _sequence() -> HumanMotionSequence:
    return HumanMotionSequence(
        source_dataset="test",
        sequence_name="gt-renderer",
        source_path="",
        model_type="smplx",
        fps=30.0,
        gender="neutral",
        betas=np.zeros(16, dtype=np.float32),
        poses=np.zeros((2, 165), dtype=np.float32),
        trans=np.zeros((2, 3), dtype=np.float32),
        metadata={"pose_layout": "smplx55"},
    )


def test_cached_and_uncached_gt_meshes_share_pelvis_alignment(monkeypatch):
    model = _OffsetRootModel()
    monkeypatch.setattr(display, "_create_smpl_model", lambda sequence, device: model)

    renderer = display.GtSmplFrameRenderer(_sequence(), color=(1, 2, 3, 255), device="cpu")
    poses = renderer.sequence.poses
    renderer.prewarm_pose_frames([0], poses)
    target = np.asarray([1.4, -0.5, 2.0], dtype=np.float32)

    cached_mesh, cached_joints = renderer.frame_mesh_and_joints_world(
        0, poses[0], target, want_mesh=True
    )
    uncached_mesh, uncached_joints = renderer.frame_mesh_and_joints_world(
        1, poses[1], target, want_mesh=True
    )

    assert cached_mesh is not None
    assert uncached_mesh is not None
    np.testing.assert_allclose(cached_mesh.vertices, uncached_mesh.vertices, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(cached_joints, uncached_joints, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(cached_joints[0], target, rtol=0.0, atol=1.0e-7)
    # Both forwards used the local zero-translation frame; the target is
    # applied once from the returned root joint below the model call.
    assert len(model.translations) == 2
    for translation in model.translations:
        np.testing.assert_array_equal(translation, np.zeros((1, 3), dtype=np.float32))

