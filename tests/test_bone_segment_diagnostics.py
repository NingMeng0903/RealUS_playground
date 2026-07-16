from types import SimpleNamespace

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.bone_segment_diagnostics import (
    _head_orientation_diagnostic,
    _joint_chain_diagnostic,
)


def _transform(rotation: np.ndarray | None = None, translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if rotation is not None:
        result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def test_joint_diagnostic_measures_shared_anchor_not_only_rigidity() -> None:
    asset = SimpleNamespace(
        joint_names=["left_shoulder", "left_elbow", "left_wrist"],
        rest_joints=np.asarray(((0, 0, 0), (0, 1, 0), (0, 2, 0)), dtype=np.float64),
        source_bone_names=["Upper", "Lower"],
        source_bone_parents=np.asarray((-1, 0), dtype=np.int64),
    )
    heads = np.asarray(((0, 0, 0), (0, 1, 0)), dtype=np.float64)
    tails = np.asarray(((0, 1, 0), (0, 2, 0)), dtype=np.float64)
    rotation = np.asarray(((0, -1, 0), (1, 0, 0), (0, 0, 1)), dtype=np.float64)
    lower = _transform(rotation, (1, 1, 0))
    transforms = np.stack((_transform(), lower))
    posed_joints = np.asarray(((0, 0, 0), (0, 1, 0), (-1, 1, 0)), dtype=np.float64)
    spec = {
        "joint": "left_elbow",
        "proximal": ("Upper",),
        "distal": ("Lower",),
        "axes": (
            ("Upper", "left_shoulder", "left_elbow"),
            ("Lower", "left_elbow", "left_wrist"),
        ),
    }

    good = _joint_chain_diagnostic(
        asset,
        spec=spec,
        source_transforms=transforms,
        rest_heads=heads,
        rest_tails=tails,
        posed_smplx_joints=posed_joints,
        translation=np.zeros(3),
        endpoint_fallback=False,
    )
    assert good["pass"]
    assert good["source_shared_anchor_error_m"] < 1.0e-10
    assert good["gap_change_m"] < 1.0e-10
    assert good["axis_error_deg"] < 1.0e-8
    assert good["connected_in_source"]

    # The distal component remains perfectly rigid, but is disconnected by
    # 10 mm.  The old dominant-bone metric could not distinguish this case.
    disconnected = transforms.copy()
    disconnected[1, 0, 3] += 0.010
    bad = _joint_chain_diagnostic(
        asset,
        spec=spec,
        source_transforms=disconnected,
        rest_heads=heads,
        rest_tails=tails,
        posed_smplx_joints=posed_joints,
        translation=np.zeros(3),
        endpoint_fallback=False,
    )
    assert not bad["pass"]
    assert np.isclose(bad["gap_change_m"], 0.010)


def test_head_orientation_compares_runtime_motion_to_smplx_global_motion() -> None:
    angle = np.deg2rad(28.0)
    rotation = np.asarray(
        ((1, 0, 0), (0, np.cos(angle), -np.sin(angle)), (0, np.sin(angle), np.cos(angle))),
        dtype=np.float64,
    )
    asset = SimpleNamespace(
        source_bone_names=["Head_Bone"],
        source_rest_global=np.eye(4, dtype=np.float64)[None],
        joint_names=["head"],
    )
    source_transforms = rotation[None]
    source_transforms = np.pad(source_transforms, ((0, 0), (0, 1), (0, 1)))
    source_transforms[:, 3, 3] = 1.0
    rest_global = np.eye(4, dtype=np.float64)[None]
    pose_global = rest_global.copy()
    pose_global[0, :3, :3] = rotation

    result = _head_orientation_diagnostic(
        asset,
        source_transforms=source_transforms,
        pose_global=pose_global,
        rest_global=rest_global,
    )
    assert result["pass"]
    assert result["orientation_error_deg"] < 1.0e-8
