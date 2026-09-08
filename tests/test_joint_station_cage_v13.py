from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.chain_rest_fit_v1 import ChainRestFitSubjectV1
from projects.genesis_ue_sync.anatomy_retarget.joint_station_cage_v13 import (
    apply_knee_station_cage_v13,
    station_field,
)


def _section(center: np.ndarray, t: float, radius: float = 0.08) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    return center + np.column_stack((radius * np.cos(angles), np.zeros(8), radius * np.sin(angles))) + np.asarray((0.0, t, 0.0))


def _fixture() -> tuple[ChainRestFitSubjectV1, SimpleNamespace, dict[str, np.ndarray]]:
    controller_names = [f"unused_{index}" for index in range(235)]
    controller_names[0:5] = ["Femur_Rot_L", "Knee_Rotate_L", "Tibia_Bone_L", "Ankle_Rot_L", "Patella_Rotate_L"]
    controller_names[5] = "Tibia_Twist_L"
    controller_names[6:11] = ["Femur_Rot_R", "Knee_Rotate_R", "Tibia_Bone_R", "Ankle_Rot_R", "Patella_Rotate_R"]
    controller_names[11] = "Tibia_Twist_R"
    meshes: list[np.ndarray] = []
    mesh_names: list[str] = []
    tissues: list[str] = []
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for side, x in (("L", 0.0), ("R", 1.0)):
        for name, origin, length in (
            (f"Femur_{side}", np.asarray((x, 0.0, 0.0)), 1.0),
            (f"Tibia_{side}", np.asarray((x, 1.0, 0.0)), 1.0),
            (f"Patella_{side}", np.asarray((x + 0.12, 1.0, 0.0)), 0.0),
        ):
            if length:
                ts = np.asarray((0.0, 0.10, 0.20, 0.50, 0.90, 1.0))
                points = np.concatenate([_section(origin, float(t)) for t in ts])
            else:
                points = origin + np.asarray(
                    ((0.0, 0.0, -0.05), (0.05, 0.0, 0.0), (0.0, 0.0, 0.05)),
                    dtype=np.float64,
                )
            meshes.append(points)
            mesh_names.append(name)
            tissues.append("bone")
            ranges.append((cursor, cursor + len(points)))
            cursor += len(points)
    vertices = np.concatenate(meshes)
    parents = np.full(235, -1, dtype=np.int32)
    b_prefit = np.tile(np.eye(4, dtype=np.float64), (235, 1, 1))
    b_final = b_prefit.copy()
    anchors = {
        "Femur_Rot_L": (0.0, 0.0, 0.0), "Knee_Rotate_L": (0.0, 1.0, 0.0),
        "Tibia_Bone_L": (0.0, 1.0, 0.0), "Ankle_Rot_L": (0.0, 2.0, 0.0),
        "Patella_Rotate_L": (0.0, 1.0, 0.0),
        "Femur_Rot_R": (1.0, 0.0, 0.0), "Knee_Rotate_R": (1.0, 1.0, 0.0),
        "Tibia_Bone_R": (1.0, 1.0, 0.0), "Ankle_Rot_R": (1.0, 2.0, 0.0),
        "Patella_Rotate_R": (1.0, 1.0, 0.0),
    }
    for name, point in anchors.items():
        b_final[controller_names.index(name), :3, 3] = point
    value = ChainRestFitSubjectV1(
        source_operator_digest="operator",
        calibration_digest="calibration",
        source_subject_digest="subject",
        smplx_model_sha256="model",
        capture_sha256="capture",
        subject_label="fixture",
        betas=np.zeros(10),
        vertices_prefit=vertices.astype(np.float32),
        vertices_final=vertices.astype(np.float32),
        faces=np.empty((0, 3), dtype=np.int32),
        bone_parents=parents,
        B_prefit=b_prefit,
        B_final=b_final,
        C_bone=b_final.copy(),
        target_local_bind=b_final.copy(),
        inverse_bind=np.linalg.inv(b_final),
        prefit_anatomical_frames=np.tile(np.eye(4), (6, 1, 1)),
        final_anatomical_frames=np.tile(np.eye(4), (6, 1, 1)),
        smplx_joints_tpose=np.zeros((55, 3)),
        station_frame_translation=np.zeros(3),
        centerline_points=np.zeros((2, 3, 3)),
        mesh_policy=np.asarray(["copy_142_prefit"] * len(mesh_names)),
        moved_vertex_ids=np.empty(0, dtype=np.int32),
        build_report={},
    )
    asset = SimpleNamespace(
        source_bone_names=controller_names,
        source_mesh_names=mesh_names,
        source_tissues=tissues,
        source_vertex_ranges=np.asarray(ranges, dtype=np.int32),
    )
    ids = {name: np.arange(start, stop) for name, (start, stop) in zip(mesh_names, ranges)}
    return value, asset, ids


def test_zero_shortening_is_bit_identity() -> None:
    value, asset, _ids = _fixture()
    result, report = apply_knee_station_cage_v13(value, asset, {"L": 0.0, "R": 0.0})
    assert result is value
    assert report["zero_identity"] is True


def test_station_cage_keeps_caps_together_and_rebuilds_bind() -> None:
    value, asset, ids = _fixture()
    result, report = apply_knee_station_cage_v13(value, asset, {"L": 0.03, "R": 0.0})
    delta = np.asarray((0.0, -0.03, 0.0))
    femur = ids["Femur_L"]
    tibia = ids["Tibia_L"]
    patella = ids["Patella_L"]
    # Distal femur, proximal tibia, and every patella point form one rigid
    # articular station plateau.
    np.testing.assert_allclose(result.vertices_final[femur[-8:]] - value.vertices_final[femur[-8:]], np.broadcast_to(delta, (8, 3)), atol=2.0e-7)
    np.testing.assert_allclose(result.vertices_final[tibia[:8]] - value.vertices_final[tibia[:8]], np.broadcast_to(delta, (8, 3)), atol=2.0e-7)
    np.testing.assert_allclose(result.vertices_final[patella] - value.vertices_final[patella], np.broadcast_to(delta, (len(patella), 3)), atol=2.0e-7)
    np.testing.assert_allclose(result.vertices_final[tibia[-8:]], value.vertices_final[tibia[-8:]], atol=2.0e-7)
    assert report["min_jacobian_axial"] > 0.0
    assert report["sides"]["L"]["hip_knee_length_after_m"] == pytest.approx(0.97)
    assert report["sides"]["L"]["knee_ankle_length_after_m"] == pytest.approx(1.03)
    assert report["station_displacement_policy"].startswith("smooth_axial_station_translation")
    names = asset.source_bone_names
    for name in ("Knee_Rotate_L", "Tibia_Bone_L", "Patella_Rotate_L"):
        index = names.index(name)
        np.testing.assert_allclose(result.B_final[index, :3, 3] - value.B_final[index, :3, 3], delta)
    for name in ("Femur_Rot_L", "Ankle_Rot_L", "Tibia_Twist_L", "Knee_Rotate_R"):
        index = names.index(name)
        np.testing.assert_array_equal(result.B_final[index], value.B_final[index])
    np.testing.assert_allclose(result.C_bone, result.B_final @ np.linalg.inv(result.B_prefit), atol=1.0e-12)
    np.testing.assert_allclose(result.target_local_bind, result.B_final, atol=1.0e-12)
    assert report["sides"]["L"]["femur_slab_deformation_max_m"] < 2.0e-7
    assert report["sides"]["L"]["shank_slab_deformation_max_m"] < 2.0e-7


def test_station_field_rejects_non_positive_axial_jacobian() -> None:
    # Neither sample lies at the steepest transition point; the analytic
    # Jacobian gate must still catch the fold.
    points = np.asarray(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float64)
    with pytest.raises(ValueError, match="Jacobian"):
        station_field(
            points,
            np.asarray((0.0, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
            np.asarray((0.0, -1.0, 0.0)),
            start_weight=0.0,
            end_weight=1.0,
        )


def test_station_field_reports_expanding_direction_minimum_as_one() -> None:
    _mapped, report = station_field(
        np.asarray(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float64),
        np.asarray((0.0, 0.0, 0.0)),
        np.asarray((0.0, 1.0, 0.0)),
        np.asarray((0.0, 1.0, 0.0)),
        start_weight=0.0,
        end_weight=1.0,
        return_report=True,
    )
    assert report["min_jacobian_axial"] == pytest.approx(1.0)
