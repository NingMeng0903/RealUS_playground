from __future__ import annotations

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import AnatomyRiggedAsset
from projects.genesis_ue_sync.anatomy_retarget.vessel_priors import (
    _restore_edge_lengths,
    _vessel_prior_config,
    apply_vessel_priors,
)


def test_vessel_prior_config_defaults() -> None:
    cfg = _vessel_prior_config({})
    assert cfg["enable"] is True
    assert abs(cfg["bone_anchor_blend"] - 0.2) < 1.0e-8
    assert cfg["edge_length_iters"] == 2
    assert abs(cfg["max_stretch_ratio"] - 1.15) < 1.0e-8


def test_edge_length_restore_caps_stretch() -> None:
    rest = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    edges = np.asarray([[0, 1]], dtype=np.int64)
    rest_lengths = np.asarray([1.0], dtype=np.float64)
    out = _restore_edge_lengths(
        pos,
        edges,
        rest_lengths,
        iterations=4,
        max_stretch_ratio=1.15,
    )
    final_len = float(np.linalg.norm(out[1] - out[0]))
    assert final_len <= 1.15 + 1.0e-6


def test_apply_vessel_priors_disabled() -> None:
    asset = AnatomyRiggedAsset(
        vertices_rest=np.zeros((2, 3), dtype=np.float32),
        faces=np.zeros((0, 3), dtype=np.int32),
        lbs_weights=np.ones((2, 1), dtype=np.float32),
        joint_names=["root"],
        parents=np.asarray([-1], dtype=np.int32),
        rest_joints=np.zeros((1, 3), dtype=np.float32),
        inverse_bind=np.eye(4, dtype=np.float32)[None],
        source_mesh_names=["Vessel"],
    )
    posed = np.ones((2, 3), dtype=np.float32)
    out, report = apply_vessel_priors(asset, posed, np.zeros((55, 3), dtype=np.float32), config={"vessel_priors": {"enable": False}})
    assert report["applied"] is False
    assert np.allclose(out, posed)
