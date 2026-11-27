"""Render identity changes must preserve selected motion and its provenance."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from projects.genesis_ue_sync.sim_platform.datasets.body_identity import with_smplx_body_identity
from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence


def test_shape_override_preserves_motion_and_carries_all_16_betas(tmp_path):
    source = HumanMotionSequence(
        source_dataset="amass", sequence_name="test", source_path="original.npz",
        model_type="smpl", fps=120, gender="female", betas=np.ones(10),
        poses=np.arange(2 * 156, dtype=np.float32).reshape(2, 156) * 0.001,
        trans=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
    )
    identity = tmp_path / "identity.npz"
    np.savez(identity, betas=np.arange(16), gender="neutral")
    model = tmp_path / "model.npz"
    model.write_bytes(b"model exists; no model inference needed for conversion")
    updated = with_smplx_body_identity(source, identity_npz=identity, body_model_npz=model)
    assert updated.gender == "neutral" and updated.model_type == "smplx"
    assert updated.fps == source.fps and updated.source_path == source.source_path
    np.testing.assert_array_equal(updated.trans, source.trans)
    np.testing.assert_array_equal(updated.poses[:, :66], source.poses[:, :66])
    np.testing.assert_array_equal(updated.poses[:, 75:120], source.poses[:, 66:111])
    np.testing.assert_array_equal(updated.poses[:, 120:165], source.poses[:, 111:156])
    np.testing.assert_array_equal(updated.betas, np.arange(16))
    assert updated.metadata["pose_layout"] == "smplx55"
    assert updated.metadata["original_motion_gender"] == "female"
    assert source.gender == "female" and source.poses.shape == (2, 156)


def test_shape_override_requires_real_model_asset(tmp_path):
    with pytest.raises(FileNotFoundError, match="body model"):
        with_smplx_body_identity(None, identity_npz=tmp_path / "missing",
                                body_model_npz=tmp_path / "missing-model.npz")
