"""Bounded left-elbow forearm joint-lock ablation.

This is a diagnostic prototype for the 213328 subject.  V11 remains the
reference and V12e remains the baseline.  The ablation replaces V12e's two
independent left forearm mesh-only reseats (Radius_L and Ulna_L) with one
common rigid rotation about the frozen V11 Elbow_Rot_L origin.  The bind,
weights, source helper, and all non-forearm vertices stay unchanged.  Only
the three rotation parameters are searched; translation, axial scale, and
bone bending are forbidden.

The search uses tpose and the two frozen capture poses, then reports the same
metrics on held-out sitting and kicking poses.  A numerical winner is an
ablation result, not an anatomical or publication claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from projects.genesis_ue_sync.anatomy_retarget.absolute_poke_v12 import bone_mesh_group_v12
from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    _calibration_content_digest,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.blender_link_oracle_v7 import (
    EXPECTED_OPERATOR_RUNTIME_DIGEST,
    EXPECTED_ORACLE_SHA256,
)
from projects.genesis_ue_sync.anatomy_retarget.chain_containment_v1 import (
    _signed_distance,
    _vertex_areas,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.run_material_matrix_v13 import (
    CAPTURE_SHA256,
    _load_capture,
    _load_subject_v11,
)
from projects.genesis_ue_sync.anatomy_retarget.material_runtime_v13 import (
    recover_shared_weight_rest,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v10 import (
    build_pose_map_v10,
    pose_whole_chain_vertices_v10,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import smplx_pose_hash
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    _smplx_joint_kinematics_v7,
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
    smplx_body_surface_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.validation_poses_v13 import (
    DEFAULT_AMASS_ROOT,
    load_held_out_poses_v13,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OPERATOR = ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
DEFAULT_CALIBRATION = ROOT / (
    "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006/"
    "anatomical_calibration_v1"
)
DEFAULT_ORACLE = ROOT / (
    "outputs/anatomy_retarget/v7_candidates/blender_link_oracle_v7_full_001/"
    "blender_link_oracle_v7.npz"
)
DEFAULT_MODEL = ROOT / "ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl"
DEFAULT_CAPTURE = ROOT / "smplx_outputs/20260713_213328/moment_0000/smplx_result.npz"
DEFAULT_V11 = ROOT / "outputs/anatomy_retarget/v11_candidates/chain_retarget_v11_anchored_001"
DEFAULT_V12E = ROOT / "outputs/anatomy_retarget/v12_candidates/chain_retarget_v12e_forearm_mesh_001"
DEFAULT_OUTPUT = ROOT / "outputs/anatomy_retarget/v13_forearm_joint_lock_20260908_001"

SUBJECT = "213328"
FIT_POSES = ("tpose", "pose_213328", "pose_213712")
HELDOUT_POSES = ("heldout_sitting", "heldout_kicking")
POSES = (*FIT_POSES, *HELDOUT_POSES)
GRID_DEGREES = (-5.0, 0.0, 5.0)
FOREARM_MESHES = ("Radius_L", "Ulna_L")
ELBOW_PAIRS = (("humerus", "radius"), ("humerus", "ulna"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _apply(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    return np.asarray(points, dtype=np.float64) @ matrix[:3, :3].T + matrix[:3, 3]


def _rigid_about_pivot(rotation: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
    matrix[:3, 3] = np.asarray(pivot, dtype=np.float64) - matrix[:3, :3] @ np.asarray(pivot, dtype=np.float64)
    return matrix


def _pose_skin_joints(
    model: Mapping[str, np.ndarray], *, betas: np.ndarray, pose: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    skin, skin_faces = smplx_body_surface_v7(model, betas=betas, pose_axis_angle=pose)
    _rest, globals_, _rest_to_pose = _smplx_joint_kinematics_v7(
        model, betas=betas, pose_axis_angle=pose
    )
    return (
        np.asarray(skin, dtype=np.float64),
        np.asarray(skin_faces, dtype=np.int32),
        np.asarray(globals_, dtype=np.float64)[:, :3, 3],
    )


def _mesh_vertex_ids(asset: Any, mesh_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    names = [str(name) for name in asset.source_mesh_names]
    tissues = [str(value).strip().lower() for value in asset.source_tissues]
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    result: dict[str, np.ndarray] = {}
    for mesh_name in mesh_names:
        try:
            index = names.index(mesh_name)
        except ValueError as exc:
            raise ValueError(f"source mesh {mesh_name!r} is missing") from exc
        if tissues[index] != "bone":
            raise ValueError(f"source mesh {mesh_name!r} is not a bone mesh")
        start, stop = ranges[index]
        result[mesh_name] = np.arange(int(start), int(stop), dtype=np.int64)
    return result


def _domain_ids(calibration: Any, side: str, part: str) -> np.ndarray:
    key = f"elbow/{side}/{part}.validation"
    if key not in calibration.domains:
        raise KeyError(f"missing calibration elbow domain {key}")
    return np.asarray(calibration.domains[key], dtype=np.int64).reshape(-1)


def _elbow_center(calibration: Any, vertices: np.ndarray, side: str = "left") -> np.ndarray:
    chunks = [_domain_ids(calibration, side, part) for part in ("humerus", "radius", "ulna")]
    return np.mean(np.asarray(vertices, dtype=np.float64)[np.unique(np.concatenate(chunks))], axis=0)


def _wrist_center(calibration: Any, vertices: np.ndarray, side: str = "left") -> np.ndarray:
    chunks = []
    for part in ("radius", "ulna"):
        key = f"calibration/{side}/wrist/{part}.validation"
        if key in calibration.domains:
            chunks.append(np.asarray(calibration.domains[key], dtype=np.int64).reshape(-1))
    if not chunks:
        raise KeyError(f"missing calibration wrist domains for {side}")
    return np.mean(np.asarray(vertices, dtype=np.float64)[np.unique(np.concatenate(chunks))], axis=0)


def _pair_gap(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    a = np.asarray(first, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(second, dtype=np.float64).reshape(-1, 3)
    if not len(a) or not len(b) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("joint gap domain is empty or non-finite")
    distances = np.concatenate((cKDTree(b).query(a, k=1)[0], cKDTree(a).query(b, k=1)[0]))
    return {
        "minimum_m": float(np.min(distances)),
        "mean_m": float(np.mean(distances)),
        "p95_m": float(np.quantile(distances, 0.95)),
        "sample_count": int(len(distances)),
    }


def _joint_gaps(calibration: Any, vertices: np.ndarray) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for first, second in ELBOW_PAIRS:
        values[f"{first}_{second}"] = _pair_gap(
            np.asarray(vertices)[_domain_ids(calibration, "left", first)],
            np.asarray(vertices)[_domain_ids(calibration, "left", second)],
        )
    return values


def _bone_skin_stats(
    vertices: np.ndarray,
    *,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    asset: Any,
    ids_by_bone: Mapping[str, np.ndarray],
    area_reference: np.ndarray,
) -> dict[str, Any]:
    ordered = list(ids_by_bone)
    all_ids = np.unique(np.concatenate([np.asarray(ids_by_bone[name], dtype=np.int64) for name in ordered]))
    signed = np.asarray(
        _signed_distance(np.asarray(vertices, dtype=np.float64)[all_ids], skin, skin_faces),
        dtype=np.float64,
    )
    if signed.shape != (len(all_ids),) or not np.all(np.isfinite(signed)):
        raise ValueError("bone skin signed distance is invalid")
    lookup = {int(vertex): index for index, vertex in enumerate(all_ids.tolist())}
    areas = _vertex_areas(np.asarray(area_reference, dtype=np.float64), np.asarray(asset.faces))
    result: dict[str, Any] = {}
    for name in ordered:
        ids = np.asarray(ids_by_bone[name], dtype=np.int64)
        local = np.asarray([lookup[int(value)] for value in ids], dtype=np.int64)
        values = signed[local]
        positive = np.maximum(values, 0.0)
        outside = values > 0.0
        weights = np.asarray(areas[ids], dtype=np.float64)
        total_area = float(np.sum(weights))
        result[name] = {
            "vertex_count": int(len(values)),
            "outside_count": int(np.count_nonzero(outside)),
            "outside_area_fraction": float(np.sum(weights[outside]) / total_area),
            "max_outside_m": float(np.max(positive)),
            "outside_p95_m": float(np.quantile(values[outside], 0.95)) if np.any(outside) else 0.0,
            "poke_p95_all_m": float(np.quantile(positive, 0.95)),
            "area_weighted_outside_depth_m": float(np.sum(weights * positive) / total_area),
        }
    return result


def _all_forearm_ids(ids_by_mesh: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.unique(np.concatenate([np.asarray(ids_by_mesh[name], dtype=np.int64) for name in FOREARM_MESHES]))


def _rest_reseat_diagnostics(
    v11: Any,
    v12e: Any,
    *,
    asset: Any,
    calibration: Any,
    ids_by_mesh: Mapping[str, np.ndarray],
    rigid_report: Mapping[str, Any],
) -> dict[str, Any]:
    pivot = _elbow_center(calibration, v11.vertices_final)
    wrist = _wrist_center(calibration, v11.vertices_final)
    axis = wrist - pivot
    axis_norm = float(np.linalg.norm(axis))
    if not axis_norm > 1.0e-8:
        raise ValueError("left elbow-to-wrist diagnostic axis is degenerate")
    axis /= axis_norm
    all_ids = _all_forearm_ids(ids_by_mesh)
    axial = (np.asarray(v11.vertices_final, dtype=np.float64)[all_ids] - pivot) @ axis
    low, high = np.quantile(axial, (0.20, 0.80))
    diagnostics: dict[str, Any] = {
        "pivot_v11_elbow_material_center_m": pivot.tolist(),
        "wrist_reference_center_m": wrist.tolist(),
        "axial_axis_elbow_to_wrist": axis.tolist(),
        "proximal_distal_partition_quantiles": [float(low), float(high)],
        "per_bone": {},
        "rigid_transforms_v11_to_v12e": {},
    }
    largest: tuple[float, str, str] | None = None
    for mesh_name, ids in ids_by_mesh.items():
        points_before = np.asarray(v11.vertices_final, dtype=np.float64)[ids]
        points_after = np.asarray(v12e.vertices_final, dtype=np.float64)[ids]
        displacement = np.linalg.norm(points_after - points_before, axis=1)
        t = (points_before - pivot) @ axis
        regions = {
            "all": np.ones(len(ids), dtype=bool),
            "proximal_20pct": t <= low,
            "distal_20pct": t >= high,
        }
        stats = {
            region: {
                "vertex_count": int(np.count_nonzero(mask)),
                "mean_displacement_m": float(np.mean(displacement[mask])) if np.any(mask) else 0.0,
                "p95_displacement_m": float(np.quantile(displacement[mask], 0.95)) if np.any(mask) else 0.0,
                "max_displacement_m": float(np.max(displacement[mask])) if np.any(mask) else 0.0,
            }
            for region, mask in regions.items()
        }
        diagnostics["per_bone"][mesh_name] = stats
        for region in ("proximal_20pct", "distal_20pct"):
            value = stats[region]["max_displacement_m"]
            if largest is None or value > largest[0]:
                largest = (value, mesh_name, region)
        controller = "Forearm_Bone_L" if mesh_name == "Ulna_L" else "Forearm_Twist_L"
        transform = np.asarray(rigid_report[controller]["transform"], dtype=np.float64)
        diagnostics["rigid_transforms_v11_to_v12e"][mesh_name] = {
            "controller": controller,
            "transform": transform.tolist(),
            "reconstruction_max_m": float(rigid_report[controller]["reconstruction_max_m"]),
            "rotation_deg": float(np.degrees(np.linalg.norm(Rotation.from_matrix(transform[:3, :3]).as_rotvec()))),
            "translation_norm_m": float(np.linalg.norm(transform[:3, 3])),
            "v11_pivot_shift_m": float(np.linalg.norm(_apply(transform, pivot[None, :])[0] - pivot)),
        }
    diagnostics["largest_displacement_region"] = (
        {"mesh": largest[1], "region": largest[2], "max_displacement_m": largest[0]}
        if largest is not None
        else None
    )
    diagnostics["bone_group_mapping"] = {
        name: bone_mesh_group_v12(name) for name in FOREARM_MESHES
    }
    return diagnostics


def _candidate_rest_from_v11(
    v11: Any,
    v12e: Any,
    *,
    ids_by_mesh: Mapping[str, np.ndarray],
    pivot: np.ndarray,
    initial_rotation: np.ndarray,
    total_rotvec_deg: np.ndarray,
) -> tuple[Any, np.ndarray]:
    rotation = Rotation.from_rotvec(
        np.deg2rad(np.asarray(total_rotvec_deg, dtype=np.float64))
    ).as_matrix() @ np.asarray(initial_rotation, dtype=np.float64)
    transform = _rigid_about_pivot(rotation, pivot)
    vertices = np.asarray(v12e.vertices_final, dtype=np.float64).copy()
    for mesh_name in FOREARM_MESHES:
        ids = np.asarray(ids_by_mesh[mesh_name], dtype=np.int64)
        vertices[ids] = _apply(transform, np.asarray(v11.vertices_final, dtype=np.float64)[ids])
    return dataclasses.replace(v12e, vertices_final=vertices.astype(np.float32)), transform


def _evaluate_pose(
    value: Any,
    pose_map: Any,
    *,
    pose: np.ndarray,
    pose_name: str,
    calibration: Any,
    asset: Any,
    model: Mapping[str, np.ndarray],
    betas: np.ndarray,
    ids_by_mesh: Mapping[str, np.ndarray],
    area_reference: np.ndarray,
) -> dict[str, Any]:
    vertices, _globals = pose_whole_chain_vertices_v10(
        value,
        pose_map,
        source_asset=asset,
        pose_axis_angle=pose,
    )
    skin, skin_faces, joints = _pose_skin_joints(model, betas=betas, pose=pose)
    ids: dict[str, np.ndarray] = {}
    names = [str(name) for name in asset.source_mesh_names]
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    for mesh_name in ("Humerus_L", "Radius_L", "Ulna_L"):
        index = names.index(mesh_name)
        start, stop = ranges[index]
        ids[mesh_name] = np.arange(int(start), int(stop), dtype=np.int64)
    ids["forearm_L"] = _all_forearm_ids(ids_by_mesh)
    return {
        "pose_name": pose_name,
        "pose55_sha256": _array_sha256(pose),
        "vertices": vertices,
        "skin": skin,
        "skin_faces": skin_faces,
        "smplx_joints": joints,
        "bone_skin": _bone_skin_stats(
            vertices,
            skin=skin,
            skin_faces=skin_faces,
            asset=asset,
            ids_by_bone=ids,
            area_reference=area_reference,
        ),
        "joint_gaps": _joint_gaps(calibration, vertices),
    }


def _objective(candidate_cells: Mapping[str, Any], v11_cells: Mapping[str, Any]) -> float:
    total = 0.0
    for pose_name in FIT_POSES:
        candidate = candidate_cells[pose_name]
        reference = v11_cells[pose_name]
        forearm = candidate["bone_skin"]["forearm_L"]
        skin_score = (
            float(forearm["max_outside_m"])
            + float(forearm["outside_p95_m"])
            + 0.01 * float(forearm["outside_area_fraction"])
        )
        gap_score = 0.0
        for pair in ("humerus_radius", "humerus_ulna"):
            for field, weight in (("minimum_m", 1.0), ("p95_m", 0.25)):
                gap_score += weight * abs(
                    float(candidate["joint_gaps"][pair][field])
                    - float(reference["joint_gaps"][pair][field])
                )
        total += skin_score + gap_score
    return float(total / len(FIT_POSES))


def _compact_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pose_name": cell["pose_name"],
        "pose55_sha256": cell["pose55_sha256"],
        "bone_skin": cell["bone_skin"],
        "joint_gaps": cell["joint_gaps"],
    }


def _save_comparison(
    path: Path,
    *,
    subject: str,
    pose_name: str,
    pose: np.ndarray,
    source_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    skin: np.ndarray,
    smplx_joints: np.ndarray,
    faces: np.ndarray,
    skin_faces: np.ndarray,
    vertex_tissue: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    np.savez_compressed(
        path,
        faces=np.asarray(faces, dtype=np.int32),
        skin_faces=np.asarray(skin_faces, dtype=np.int32),
        source_vertices=np.asarray(source_vertices, dtype=np.float32),
        candidate_vertices=np.asarray(candidate_vertices, dtype=np.float32),
        skin_vertices=np.asarray(skin, dtype=np.float32),
        smplx_joints=np.asarray(smplx_joints, dtype=np.float32),
        pose=np.asarray(pose, dtype=np.float32),
        vertex_tissue=np.asarray(vertex_tissue, dtype=np.int8),
        subject=np.asarray(subject),
        pose_name=np.asarray(pose_name),
        metadata_json=np.asarray(json.dumps(_json_ready(metadata), sort_keys=True, allow_nan=False)),
    )


def _tissue_codes(asset: Any) -> np.ndarray:
    codes = {"bone": 0, "vessel": 1, "nerve": 2, "organ": 3, "heart": 4, "connective": 5, "connective_tissue": 5}
    result = np.full(len(asset.vertices_rest), -1, dtype=np.int8)
    for tissue, (start, stop) in zip(asset.source_tissues, np.asarray(asset.source_vertex_ranges, dtype=np.int64)):
        label = str(tissue).strip().lower()
        if label not in codes:
            raise ValueError(f"unrecognised tissue {tissue!r}")
        result[int(start):int(stop)] = codes[label]
    if np.any(result < 0):
        raise ValueError("unlabelled source tissue vertices")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--smplx-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--capture-213712", type=Path, default=ROOT / "smplx_outputs/20260713_213712/moment_0000/smplx_result.npz")
    parser.add_argument("--v11", type=Path, default=DEFAULT_V11)
    parser.add_argument("--v12e", type=Path, default=DEFAULT_V12E)
    parser.add_argument("--amass-root", type=Path, default=DEFAULT_AMASS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite forearm joint-lock output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 13,
        "artifact_kind": "ForearmJointLockAblationV13",
        "subject": SUBJECT,
        "publishable": False,
        "production_solver_modified": False,
        "constraints": {
            "bones": ["Radius_L", "Ulna_L"],
            "rotation_only": True,
            "pivot": "V11 Elbow_Rot_L frozen material center",
            "translation_allowed": False,
            "axial_scale_allowed": False,
            "bending_allowed": False,
            "soft_tissue_transport_applied": False,
            "total_rotation_bounds_deg_per_axis": [-5.0, 5.0],
        },
        "status": "running",
    }
    try:
        operator = load_source_operator(Path(args.operator).expanduser().resolve(), mmap=True)
        operator_digest = operator.runtime_digest(validate=False)
        if operator_digest != EXPECTED_OPERATOR_RUNTIME_DIGEST:
            raise ValueError("forearm lock requires frozen rebuild_012 source operator")
        oracle = Path(args.oracle).expanduser().resolve()
        oracle_sha = _sha256(oracle)
        if oracle_sha != EXPECTED_ORACLE_SHA256:
            raise ValueError("forearm lock requires frozen Blender oracle")
        calibration = load_anatomical_calibration_v1(
            Path(args.calibration).expanduser().resolve(),
            operator=operator,
            required_scope="full_main_chain",
        )
        calibration_digest = _calibration_content_digest(calibration)
        model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
        model = load_smplx_model_v7(model_path)
        capture_betas_a, pose_213328, capture_sha_a = _load_capture(args.capture, model_path=model_path)
        capture_betas_b, pose_213712, capture_sha_b = _load_capture(args.capture_213712, model_path=model_path)
        if capture_sha_a != CAPTURE_SHA256["213328"] or capture_sha_b != CAPTURE_SHA256["213712"]:
            raise ValueError("capture SHA differs from the frozen 213328/213712 pair")
        heldout, heldout_provenance = load_held_out_poses_v13(args.amass_root)
        poses = {
            "tpose": np.zeros((55, 3), dtype=np.float32),
            "pose_213328": np.asarray(pose_213328, dtype=np.float32),
            "pose_213712": np.asarray(pose_213712, dtype=np.float32),
            "heldout_sitting": np.asarray(heldout["heldout_sitting"], dtype=np.float32),
            "heldout_kicking": np.asarray(heldout["heldout_kicking"], dtype=np.float32),
        }
        v11 = _load_subject_v11(Path(args.v11).resolve() / f"subject_{SUBJECT}")
        v12e = _load_subject_v11(Path(args.v12e).resolve() / f"subject_{SUBJECT}")
        for label, value in (("v11", v11), ("v12e", v12e)):
            if value.source_operator_digest != operator_digest or value.calibration_digest != calibration_digest:
                raise ValueError(f"{label}: frozen operator/calibration digest mismatch")
            if not np.allclose(np.asarray(value.betas), capture_betas_a, atol=1.0e-7, rtol=0.0):
                raise ValueError(f"{label}: 213328 beta mismatch")
        if not np.array_equal(v11.faces, v12e.faces):
            raise ValueError("V11 and V12e topology differs")
        materialized = materialize_subject(operator, betas=np.asarray(v12e.betas), gender="male")
        asset = materialized.rigged_asset
        if not np.array_equal(asset.faces, v12e.faces):
            raise ValueError("V12e topology differs from materialized source asset")
        ids_by_mesh = _mesh_vertex_ids(asset, FOREARM_MESHES)
        rigid_target_rest, rigid_report = recover_shared_weight_rest(v11, v12e, asset)
        del rigid_target_rest  # only the persisted rigid reports are needed here
        rest_diag = _rest_reseat_diagnostics(
            v11, v12e, asset=asset, calibration=calibration,
            ids_by_mesh=ids_by_mesh, rigid_report=rigid_report,
        )
        pivot = np.asarray(rest_diag["pivot_v11_elbow_material_center_m"], dtype=np.float64)
        controller_names = [str(name) for name in asset.source_bone_names]
        bfinal_pivot = np.asarray(
            v11.B_final[controller_names.index("Elbow_Rot_L"), :3, 3], dtype=np.float64
        )
        rest_diag.update({
            "pivot_source": "V11 vertices_final calibration elbow validation-domain centroid (humerus+radius+ulna)",
            "pivot_b_final_elbow_rot_m": bfinal_pivot.tolist(),
            "pivot_material_minus_b_final_m": (pivot - bfinal_pivot).tolist(),
            "pivot_material_vs_b_final_distance_m": float(np.linalg.norm(pivot - bfinal_pivot)),
        })
        rotations = []
        for controller in ("Forearm_Twist_L", "Forearm_Bone_L"):
            rotations.append(np.asarray(rigid_report[controller]["transform"], dtype=np.float64)[:3, :3])
        v12e_mean_rotation = Rotation.from_matrix(np.asarray(rotations)).mean().as_matrix()
        v12e_mean_rotvec_deg = np.degrees(
            Rotation.from_matrix(v12e_mean_rotation).as_rotvec()
        )
        # The authorized search domain is measured from V11 identity about the
        # frozen pivot.  The V12e mean is diagnostic only and is never used as
        # the zero point of the candidate angles.
        initial_rotation = np.eye(3, dtype=np.float64)
        initial_rotvec_deg = np.zeros(3, dtype=np.float64)
        pose_map_v11 = build_pose_map_v10(
            v11, asset=asset, calibration=calibration, oracle_path=oracle,
            source_operator_digest=operator_digest,
        )
        pose_map_v12e = build_pose_map_v10(
            v12e, asset=asset, calibration=calibration, oracle_path=oracle,
            source_operator_digest=operator_digest,
        )
        area_reference = np.asarray(asset.vertices_rest, dtype=np.float64)
        v11_cells: dict[str, Any] = {}
        v12e_cells: dict[str, Any] = {}
        for pose_name in POSES:
            v11_cells[pose_name] = _evaluate_pose(
                v11, pose_map_v11, pose=poses[pose_name], pose_name=pose_name,
                calibration=calibration, asset=asset, model=model,
                betas=np.asarray(v11.betas), ids_by_mesh=ids_by_mesh,
                area_reference=area_reference,
            )
            v12e_cells[pose_name] = _evaluate_pose(
                v12e, pose_map_v12e, pose=poses[pose_name], pose_name=pose_name,
                calibration=calibration, asset=asset, model=model,
                betas=np.asarray(v12e.betas), ids_by_mesh=ids_by_mesh,
                area_reference=area_reference,
            )
        tissue = _tissue_codes(asset)
        # Stage the actual V11 -> V12e tpose and own-capture comparison before
        # the bounded grid starts, so Genesis review does not wait on fitting.
        staged_files: dict[str, str] = {}
        for pose_name in ("tpose", "pose_213328"):
            staged = output / f"subject_{SUBJECT}_v12e_before_{pose_name}.npz"
            _save_comparison(
                staged,
                subject=SUBJECT,
                pose_name=pose_name,
                pose=poses[pose_name],
                source_vertices=v11_cells[pose_name]["vertices"],
                candidate_vertices=v12e_cells[pose_name]["vertices"],
                skin=v12e_cells[pose_name]["skin"],
                smplx_joints=v12e_cells[pose_name]["smplx_joints"],
                faces=asset.faces,
                skin_faces=v12e_cells[pose_name]["skin_faces"],
                vertex_tissue=tissue,
                metadata={
                    "schema_version": 13,
                    "artifact_kind": "ForearmJointLockBeforeV13",
                    "subject": SUBJECT,
                    "pose": pose_name,
                    "source_label": "V11",
                    "candidate_label": "V12e_independent_forearm_mesh_reseat",
                    "publishable": False,
                    "soft_tissue_transport_applied": False,
                    "stage_export_before_grid": True,
                },
            )
            staged_files[pose_name] = str(staged)
        _write_json(output / "progress.json", {
            "schema_version": 13,
            "artifact_kind": "ForearmJointLockAblationV13Progress",
            "subject": SUBJECT,
            "status": "baseline_ready",
            "staged_files": staged_files,
            "v11": {name: _compact_cell(v11_cells[name]) for name in POSES},
            "v12e_before": {name: _compact_cell(v12e_cells[name]) for name in POSES},
            "search_grid_count": len(GRID_DEGREES) ** 3,
        })
        trials: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        identity_value: Any | None = None
        identity_fit_cells: dict[str, Any] | None = None
        for total in itertools.product(GRID_DEGREES, repeat=3):
            total_array = np.asarray(total, dtype=np.float64)
            candidate, transform = _candidate_rest_from_v11(
                v11, v12e, ids_by_mesh=ids_by_mesh, pivot=pivot,
                initial_rotation=initial_rotation, total_rotvec_deg=total_array,
            )
            cells: dict[str, Any] = {}
            for pose_name in FIT_POSES:
                cells[pose_name] = _evaluate_pose(
                    candidate, pose_map_v12e, pose=poses[pose_name], pose_name=pose_name,
                    calibration=calibration, asset=asset, model=model,
                    betas=np.asarray(v12e.betas), ids_by_mesh=ids_by_mesh,
                    area_reference=area_reference,
                )
            objective = _objective(cells, v11_cells)
            trial = {
                "total_rotvec_deg": total_array.tolist(),
                "objective": objective,
                "fixed_pivot_transform": transform.tolist(),
                "fit_cells": {name: _compact_cell(cells[name]) for name in FIT_POSES},
            }
            trials.append(trial)
            if np.array_equal(total_array, np.zeros(3)):
                identity_value = candidate
                identity_fit_cells = cells
            key = (objective, float(np.linalg.norm(total_array)), tuple(total_array.tolist()))
            if best is None or key < best["key"]:
                best = {"key": key, "total": total_array, "transform": transform, "value": candidate, "fit_cells": cells}
        if best is None:
            raise ValueError("rotation grid produced no candidate")
        if identity_value is None or identity_fit_cells is None:
            raise ValueError("rotation grid did not include the V11 identity baseline")
        best_value = best["value"]
        best_cells: dict[str, Any] = {}
        for pose_name in POSES:
            best_cells[pose_name] = _evaluate_pose(
                best_value, pose_map_v12e, pose=poses[pose_name], pose_name=pose_name,
                calibration=calibration, asset=asset, model=model,
                betas=np.asarray(v12e.betas), ids_by_mesh=ids_by_mesh,
                area_reference=area_reference,
            )
        files: dict[str, str] = {}
        for pose_name in POSES:
            cell = best_cells[pose_name]
            base = output / f"subject_{SUBJECT}_joint_lock_best_{pose_name}.npz"
            _save_comparison(
                base, subject=SUBJECT, pose_name=pose_name, pose=poses[pose_name],
                source_vertices=v12e_cells[pose_name]["vertices"],
                candidate_vertices=cell["vertices"], skin=cell["skin"],
                smplx_joints=cell["smplx_joints"], faces=asset.faces,
                skin_faces=cell["skin_faces"], vertex_tissue=tissue,
                metadata={
                    "schema_version": 13,
                    "artifact_kind": "ForearmJointLockComparisonV13",
                    "subject": SUBJECT,
                    "pose": pose_name,
                    "source_label": "V12e_before_common_lock",
                    "candidate_label": "common_Radius_Ulna_fixed_V11_elbow_pivot_rotation",
                    "total_rotvec_deg": best["total"].tolist(),
                    "fixed_pivot_transform": np.asarray(best["transform"]).tolist(),
                    "search_origin_rotvec_deg": initial_rotvec_deg.tolist(),
                    "v12e_mean_rotation_rotvec_deg": v12e_mean_rotvec_deg.tolist(),
                    "translation_allowed": False,
                    "bind_frozen": True,
                    "publishable": False,
                    "soft_tissue_transport_applied": False,
                },
            )
            files[f"best/{pose_name}"] = str(base)
            before = output / f"subject_{SUBJECT}_v12e_before_{pose_name}.npz"
            _save_comparison(
                before, subject=SUBJECT, pose_name=pose_name, pose=poses[pose_name],
                source_vertices=v11_cells[pose_name]["vertices"],
                candidate_vertices=v12e_cells[pose_name]["vertices"],
                skin=v12e_cells[pose_name]["skin"],
                smplx_joints=v12e_cells[pose_name]["smplx_joints"], faces=asset.faces,
                skin_faces=v12e_cells[pose_name]["skin_faces"], vertex_tissue=tissue,
                metadata={
                    "schema_version": 13,
                    "artifact_kind": "ForearmJointLockBeforeV13",
                    "subject": SUBJECT,
                    "pose": pose_name,
                    "source_label": "V11",
                    "candidate_label": "V12e_independent_forearm_mesh_reseat",
                    "publishable": False,
                    "soft_tissue_transport_applied": False,
                },
            )
            files[f"v11_to_v12e/{pose_name}"] = str(before)
        identity_cells: dict[str, Any] = dict(identity_fit_cells)
        for pose_name in HELDOUT_POSES:
            identity_cells[pose_name] = _evaluate_pose(
                identity_value, pose_map_v12e, pose=poses[pose_name], pose_name=pose_name,
                calibration=calibration, asset=asset, model=model,
                betas=np.asarray(v12e.betas), ids_by_mesh=ids_by_mesh,
                area_reference=area_reference,
            )
        identity_files: dict[str, str] = {}
        for pose_name in ("tpose", "pose_213328"):
            identity_path = output / f"subject_{SUBJECT}_joint_lock_identity_{pose_name}.npz"
            _save_comparison(
                identity_path,
                subject=SUBJECT,
                pose_name=pose_name,
                pose=poses[pose_name],
                source_vertices=v12e_cells[pose_name]["vertices"],
                candidate_vertices=identity_cells[pose_name]["vertices"],
                skin=identity_cells[pose_name]["skin"],
                smplx_joints=identity_cells[pose_name]["smplx_joints"],
                faces=asset.faces,
                skin_faces=identity_cells[pose_name]["skin_faces"],
                vertex_tissue=tissue,
                metadata={
                    "schema_version": 13,
                    "artifact_kind": "ForearmJointLockIdentityV13",
                    "subject": SUBJECT,
                    "pose": pose_name,
                    "source_label": "V12e_before_common_lock",
                    "candidate_label": "V11_forearm_identity_about_frozen_elbow_pivot",
                    "total_rotvec_deg": [0.0, 0.0, 0.0],
                    "fixed_pivot_transform": np.eye(4).tolist(),
                    "publishable": False,
                    "soft_tissue_transport_applied": False,
                },
            )
            identity_files[pose_name] = str(identity_path)
        rollback_by_pose: dict[str, Any] = {}
        for pose_name in POSES:
            before = v12e_cells[pose_name]
            after = best_cells[pose_name]
            before_forearm = before["bone_skin"]["forearm_L"]
            after_forearm = after["bone_skin"]["forearm_L"]
            containment = {
                field: float(after_forearm[field]) - float(before_forearm[field])
                for field in ("max_outside_m", "outside_p95_m", "outside_area_fraction")
            }
            gaps = {
                pair: {
                    field: float(after["joint_gaps"][pair][field])
                    - float(before["joint_gaps"][pair][field])
                    for field in ("minimum_m", "p95_m")
                }
                for pair in ("humerus_radius", "humerus_ulna")
            }
            rollback_by_pose[pose_name] = {
                "v12e_before_forearm": before_forearm,
                "best_forearm": after_forearm,
                "best_minus_v12e_containment": containment,
                "best_minus_v12e_joint_gap": gaps,
                "rollback_recommended": bool(
                    containment["max_outside_m"] > 1.0e-9
                    or containment["outside_p95_m"] > 1.0e-9
                    or containment["outside_area_fraction"] > 1.0e-9
                ),
            }
        bind_fields = ("B_prefit", "B_final", "C_bone", "target_local_bind", "inverse_bind")
        bind_unchanged = {
            field: bool(np.array_equal(getattr(v12e, field), getattr(best_value, field)))
            for field in bind_fields
        }
        report.update({
            "status": "complete",
            "passed": False,
            "operational_evaluation_completed": True,
            "anatomical_passed": False,
            "rejected": True,
            "execution_complete": True,
            "candidate_acceptance": {
                "passed": False,
                "reason": "diagnostic ablation only; held-out containment and joint-gap trade-offs remain",
            },
            "soft_tissue_transport_applied": False,
            "provenance": {
                "operator": str(Path(args.operator).resolve()),
                "operator_runtime_digest": operator_digest,
                "calibration": str(Path(args.calibration).resolve()),
                "calibration_digest": calibration_digest,
                "calibration_fixed_domain_digest": str(calibration.fixed_domain_digest),
                "oracle": str(oracle),
                "oracle_sha256": oracle_sha,
                "smplx_model": str(model_path),
                "smplx_model_sha256": model_sha,
                "v11_subject": str(Path(args.v11).resolve() / f"subject_{SUBJECT}"),
                "v12e_subject": str(Path(args.v12e).resolve() / f"subject_{SUBJECT}"),
                "capture_213328": {"path": str(Path(args.capture).resolve()), "sha256": capture_sha_a, "pose55_hash": smplx_pose_hash(pose_213328)},
                "capture_213712": {"path": str(Path(args.capture_213712).resolve()), "sha256": capture_sha_b, "pose55_hash": smplx_pose_hash(pose_213712)},
                "heldout_poses": heldout_provenance,
            },
            "rest_reseat_diagnostics": rest_diag,
            "search": {
                "fit_poses": list(FIT_POSES),
                "heldout_test_poses": list(HELDOUT_POSES),
                "grid_values_deg": list(GRID_DEGREES),
                "trial_count": len(trials),
                "search_origin_rotvec_deg": initial_rotvec_deg.tolist(),
                "best_total_rotvec_deg": best["total"].tolist(),
                "identity_total_rotvec_deg": [0.0, 0.0, 0.0],
                "identity_fixed_pivot_transform": np.eye(4).tolist(),
                "v12e_mean_rotation_rotvec_deg": v12e_mean_rotvec_deg.tolist(),
                "best_fixed_pivot_transform": np.asarray(best["transform"]).tolist(),
                "best_objective": float(best["key"][0]),
                "trials": trials,
            },
            "metrics": {
                "v11": {name: _compact_cell(v11_cells[name]) for name in POSES},
                "v12e_before": {name: _compact_cell(v12e_cells[name]) for name in POSES},
                "best_common_lock": {name: _compact_cell(best_cells[name]) for name in POSES},
                "v11_forearm_identity": {name: _compact_cell(identity_cells[name]) for name in POSES},
            },
            "rollback_by_pose": rollback_by_pose,
            "bind_and_helper_frozen": {
                "candidate_vs_v12e_fields_unchanged": bind_unchanged,
                "candidate_changed_only_vertices_final": True,
                "source_operator_and_weights_reused": True,
                "runtime_pose_evaluator": "pose_whole_chain_vertices_v10",
            },
            "files": files,
            "identity_files": identity_files,
            "decision_file": str(output / "decision.json"),
            "publishability_note": "bounded mesh-rest ablation only; no anatomical success claim",
            "elapsed_seconds": float(time.perf_counter() - started),
        })
        _write_json(output / "decision.json", {
            "schema_version": 13,
            "artifact_kind": "ForearmJointLockDecisionV13",
            "subject": SUBJECT,
            "operational_evaluation_completed": True,
            "anatomical_passed": False,
            "rejected": True,
            "soft_tissue_transport_applied": False,
            "best_total_rotvec_deg": best["total"].tolist(),
            "rollback_by_pose": rollback_by_pose,
            "reason": "best 27-point common pivot rotation improves the own capture joint gap but regresses containment on multiple fit/held-out poses; retain V12e and do not promote",
        })
        _write_json(output / "report.json", report)
        print(
            f"forearm_joint_lock subject={SUBJECT} trials={len(trials)} "
            f"best_total_deg={best['total'].tolist()} publishable=false",
            flush=True,
        )
        return 0
    except Exception as exc:
        report.update({
            "status": "evaluation_error",
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": float(time.perf_counter() - started),
        })
        _write_json(output / "report.json", report)
        print(f"forearm_joint_lock status=evaluation_error error={type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
