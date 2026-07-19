from types import SimpleNamespace

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.bone_segment_diagnostics import (
    _geometry_landmark_diagnostic,
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


def test_virtual_controller_endpoints_cannot_make_joint_pass() -> None:
    asset = SimpleNamespace(
        joint_names=["left_shoulder", "left_elbow", "left_wrist"],
        rest_joints=np.asarray(((0, 0, 0), (0, 1, 0), (0, 2, 0)), dtype=np.float64),
        source_bone_names=["Upper", "Lower"],
        source_bone_parents=np.asarray((-1, 0), dtype=np.int64),
    )
    heads = np.asarray(((0, 0, 0), (0, 1, 0)), dtype=np.float64)
    tails = np.asarray(((0, 1, 0), (0, 2, 0)), dtype=np.float64)
    spec = {
        "joint": "left_elbow",
        "proximal": ("Upper",),
        "distal": ("Lower",),
        "axes": (
            ("Upper", "left_shoulder", "left_elbow"),
            ("Lower", "left_elbow", "left_wrist"),
        ),
    }

    report = _joint_chain_diagnostic(
        asset,
        spec=spec,
        source_transforms=np.stack((_transform(), _transform())),
        rest_heads=heads,
        rest_tails=tails,
        posed_smplx_joints=asset.rest_joints,
        translation=np.zeros(3),
        endpoint_fallback=True,
    )

    assert not report["pass"]
    assert report["endpoint_source"] == "derived_legacy_bind"


def test_missing_hip_geometry_is_unavailable_not_controller_success() -> None:
    asset = SimpleNamespace(
        vertices_rest=np.zeros((2, 3), dtype=np.float64),
        rest_joints=np.asarray(((0, 0, 0),), dtype=np.float64),
        joint_names=["left_hip"],
        source_mesh_names=["Femur_L"],
        source_vertex_ranges=np.asarray(((0, 2),), dtype=np.int64),
        source_tissues=["bone"],
    )

    report = _geometry_landmark_diagnostic(
        asset,
        label="hip_left",
        joint_name="left_hip",
        posed_vertices=asset.vertices_rest,
        posed_smplx_joints=asset.rest_joints,
        translation=np.zeros(3),
    )

    assert not report["available"]
    assert not report["pass"]
    assert not report["roles"]["proximal"]["available"]


def test_posed_surface_gap_recomputes_contact_inside_frozen_joint_domains() -> None:
    proximal = np.asarray(
        [
            (0.000, 0.000, 0.000),
            (0.010, 0.000, 0.000),
            (0.030, 0.000, 0.000),
            (0.040, 0.000, 0.000),
            (0.050, 0.000, 0.000),
            (0.060, 0.000, 0.000),
            (0.070, 0.000, 0.000),
            (0.080, 0.000, 0.000),
        ],
        dtype=np.float64,
    )
    distal = proximal + np.asarray((0.0, 0.001, 0.0))
    rest = np.concatenate((proximal, distal), axis=0)
    posed = rest.copy()
    # The original closest pair separates, but the adjacent frozen-domain pair
    # remains in contact after the hinge slides along the joint surface.
    posed[0] += np.asarray((0.0, 0.050, 0.0))
    asset = SimpleNamespace(
        vertices_rest=rest,
        registration_reference=rest.copy(),
        rest_joints=np.asarray(((0.0, 0.0, 0.0),), dtype=np.float64),
        joint_names=["left_hip"],
        source_mesh_names=["Ilium_L", "Femur_L"],
        source_vertex_ranges=np.asarray(((0, 8), (8, 16)), dtype=np.int64),
        source_tissues=["bone", "bone"],
    )

    report = _geometry_landmark_diagnostic(
        asset,
        label="hip_left",
        joint_name="left_hip",
        posed_vertices=posed,
        posed_smplx_joints=asset.rest_joints,
        translation=np.zeros(3),
    )

    assert report["pass"]
    assert np.isclose(report["surface_gap_m"], 0.001)
    assert np.isclose(report["posed_surface_gap_m"], 0.001)
    assert report["surface_landmarks"]["proximal_vertex"] == 0
    assert report["surface_landmarks"]["posed_proximal_vertex"] == 1
