"""Probe local rotation transport after a V14 rest-frame change.

This is an evidence-only ablation.  It loads an already compiled V14 subject
and its existing pose exports, evaluates the current local-delta response, and
compares it with the rotation-only conjugation

``D'_i.R = Q_i.T @ D_i.R @ Q_i``

where ``Q_i = Rreference_bind_i.T @ Rrootmap.T @ Rtarget_bind_i``.  The
translation response is kept unchanged for the primary variant.  A second
variant conjugates the translation vector as an explicitly labelled
diagnostic; it is never called the candidate.  No fit, bind, weight, vessel,
or runtime file is changed by this command.

Each output NPZ keeps the ordinary Genesis fields.  ``candidate_vertices`` is
the rotation-conjugated geometry so it can be rendered as the probe variant;
``before_vertices`` and ``current_vertices`` retain the existing compiled
response, and both variants are also stored under explicit names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import igl
import numpy as np
from scipy.spatial.transform import Rotation

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    _calibration_content_digest,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.chain_rest_fit_v1 import (
    _global_to_local,
    _weighted_rest_correction,
)
from projects.genesis_ue_sync.anatomy_retarget.consistent_runtime_v14 import (
    load_compiled_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import _fk
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import load_source_operator


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_COMPILED = ROOT / (
    "outputs/anatomy_retarget/v14_original_shape_restfit_213328_20260908_002/compiled"
)
DEFAULT_INPUT = DEFAULT_COMPILED.parent
DEFAULT_OPERATOR = ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
DEFAULT_CALIBRATION = ROOT / (
    "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006/"
    "anatomical_calibration_v1"
)
DEFAULT_OUTPUT = ROOT / "outputs/anatomy_retarget/v14_rotation_transport_probe_20260908_001"
DEFAULT_SUBJECT = "213328"
POSES = (
    "tpose",
    "pose_213328",
    "pose_213712",
    "elbow_L_30",
    "elbow_L_60",
    "elbow_L_90",
    "elbow_L_120",
    "heldout_sitting",
    "heldout_kicking",
)
PRIMARY_BONES = {"humerus": "Humerus", "radius": "Radius", "ulna": "Ulna"}
PARTITIONS = ("fit", "validation")
NEGATIVE_SIGNED_THRESHOLD_M = 1.0e-4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


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


def _mesh_face_subset(faces: np.ndarray, *, start: int, stop: int) -> np.ndarray:
    triangles = np.asarray(faces, dtype=np.int64)
    selected = np.all((triangles >= int(start)) & (triangles < int(stop)), axis=1)
    local = triangles[selected] - int(start)
    if len(local) == 0:
        raise ValueError(f"mesh range [{start}, {stop}) has no complete faces")
    return local.astype(np.int32)


def _mesh_layout(asset: Any, faces: np.ndarray) -> dict[str, dict[str, Any]]:
    names = [str(value) for value in asset.source_mesh_names]
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    result: dict[str, dict[str, Any]] = {}
    for name in ("Humerus_L", "Radius_L", "Ulna_L", "Humerus_R", "Radius_R", "Ulna_R"):
        if name not in names:
            raise ValueError(f"compiled source asset is missing {name}")
        index = names.index(name)
        start, stop = map(int, ranges[index])
        local_faces = _mesh_face_subset(faces, start=start, stop=stop)
        result[name] = {
            "vertex_start": start,
            "vertex_stop": stop,
            "faces_local": local_faces,
        }
    return result


def _left_arm_mesh_names(asset: Any) -> list[str]:
    """Reproduce the complete connected left-arm bone population."""

    names = [str(value) for value in asset.source_mesh_names]
    tissues = [str(value).strip().lower() for value in asset.source_tissues]
    owners = np.asarray(asset.source_mesh_controller_bones, dtype=np.int64).reshape(-1)
    bone_names = [str(value) for value in asset.source_bone_names]
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64).reshape(-1)
    wrist = bone_names.index("Wrist_Rotate_L")
    hand_owners: set[int] = set()
    for index in range(len(bone_names)):
        parent = index
        while parent >= 0 and parent != wrist:
            parent = int(parents[parent])
        if parent == wrist:
            hand_owners.add(index)
    selected: list[str] = []
    for name, tissue, owner in zip(names, tissues, owners.tolist()):
        if name in {"Humerus_L", "Radius_L", "Ulna_L"} or (
            tissue == "bone" and int(owner) in hand_owners
        ):
            selected.append(name)
    if not {"Humerus_L", "Radius_L", "Ulna_L"}.issubset(selected):
        raise ValueError("left arm population is missing a primary forearm mesh")
    return selected


def _load_pose_cell(path: Path, *, vertex_count: int, faces: np.ndarray) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        required = {
            "source_vertices",
            "candidate_vertices",
            "faces",
            "skin_vertices",
            "skin_faces",
            "smplx_joints",
            "pose",
            "vertex_tissue",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path}: missing fields {missing}")
        source = np.asarray(data["source_vertices"], dtype=np.float64)
        before = np.asarray(data["candidate_vertices"], dtype=np.float64)
        file_faces = np.asarray(data["faces"], dtype=np.int32)
        skin = np.asarray(data["skin_vertices"], dtype=np.float64)
        skin_faces = np.asarray(data["skin_faces"], dtype=np.int32)
        joints = np.asarray(data["smplx_joints"], dtype=np.float64)
        pose = np.asarray(data["pose"], dtype=np.float32).reshape(-1)
        tissue = np.asarray(data["vertex_tissue"], dtype=np.int8).reshape(-1)
    if source.shape != (vertex_count, 3) or before.shape != source.shape:
        raise ValueError(f"{path}: source/candidate geometry shape mismatch")
    if not np.array_equal(file_faces, faces):
        raise ValueError(f"{path}: faces differ from compiled source topology")
    if skin.ndim != 2 or skin.shape[1] != 3 or skin_faces.ndim != 2 or skin_faces.shape[1] != 3:
        raise ValueError(f"{path}: invalid skin arrays")
    if joints.shape != (55, 3) or pose.size != 165 or tissue.shape != (vertex_count,):
        raise ValueError(f"{path}: invalid joints, pose, or tissue arrays")
    for name, values in (("source", source), ("before", before), ("skin", skin), ("joints", joints)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: {name} contains non-finite values")
    return {
        "path": path.resolve(),
        "sha256": _sha256(path),
        "source_vertices": source,
        "before_vertices": before,
        "faces": file_faces,
        "skin_vertices": skin,
        "skin_faces": skin_faces,
        "smplx_joints": joints,
        "pose": pose.reshape(55, 3),
        "vertex_tissue": tissue,
        "array_sha256": {
            "source_vertices": _array_sha256(source),
            "candidate_vertices": _array_sha256(before),
            "faces": _array_sha256(file_faces),
            "skin_vertices": _array_sha256(skin),
            "skin_faces": _array_sha256(skin_faces),
            "smplx_joints": _array_sha256(joints),
            "pose": _array_sha256(pose),
            "vertex_tissue": _array_sha256(tissue),
        },
    }


def _signed_distances(points: np.ndarray, target_vertices: np.ndarray, target_faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    vertices = np.asarray(target_vertices, dtype=np.float64)
    faces = np.asarray(target_faces, dtype=np.int32)
    squared, _face_ids, _closest = igl.point_mesh_squared_distance(query, vertices, faces)
    squared = np.asarray(squared, dtype=np.float64).reshape(-1)
    unsigned = np.sqrt(np.maximum(0.0, squared))
    winding = np.asarray(igl.winding_number(vertices, faces, query), dtype=np.float64).reshape(-1)
    if len(unsigned) != len(query) or len(winding) != len(query):
        raise ValueError("libigl returned an invalid distance result")
    if not np.all(np.isfinite(unsigned)) or not np.all(np.isfinite(winding)):
        raise ValueError("libigl returned non-finite distances")
    signed = np.where(np.abs(winding) >= 0.5, -unsigned, unsigned)
    return unsigned, signed


def _distance_row(unsigned: np.ndarray, signed: np.ndarray) -> dict[str, Any]:
    return {
        "query_vertex_count": int(len(unsigned)),
        "nearest_gap_m": float(np.min(unsigned)),
        "unsigned_p95_m": float(np.quantile(unsigned, 0.95)),
        "max_unsigned_m": float(np.max(unsigned)),
        "minimum_signed_m": float(np.min(signed)),
        "signed_p05_m": float(np.quantile(signed, 0.05)),
        "signed_median_m": float(np.median(signed)),
        "signed_p95_m": float(np.quantile(signed, 0.95)),
        "maximum_signed_m": float(np.max(signed)),
        "negative_signed_gt_0.1mm_count": int(np.count_nonzero(signed < -NEGATIVE_SIGNED_THRESHOLD_M)),
    }


def _full_left_arm_skin_metric(
    vertices: np.ndarray,
    *,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    asset: Any,
    mesh_names: list[str],
) -> dict[str, Any]:
    names = [str(value) for value in asset.source_mesh_names]
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    all_signed: list[np.ndarray] = []
    per_mesh: dict[str, Any] = {}
    for mesh_name in mesh_names:
        index = names.index(mesh_name)
        start, stop = map(int, ranges[index])
        _unsigned, signed = _signed_distances(vertices[start:stop], skin, skin_faces)
        all_signed.append(signed)
        outside = np.maximum(signed, 0.0)
        per_mesh[mesh_name] = {
            "vertex_range": [start, stop],
            "vertex_count": int(len(signed)),
            "outside_vertex_count_gt_0": int(np.count_nonzero(signed > 0.0)),
            "outside_vertex_count_gt_1mm": int(np.count_nonzero(signed > 0.001)),
            "max_outside_m": float(np.max(outside)),
            "minimum_signed_m": float(np.min(signed)),
        }
    values = np.concatenate(all_signed)
    outside = np.maximum(values, 0.0)
    return {
        "population": "complete_connected_left_arm_bone_meshes",
        "mesh_names": list(mesh_names),
        "vertex_count": int(len(values)),
        "outside_vertex_count_gt_0": int(np.count_nonzero(values > 0.0)),
        "outside_vertex_count_gt_1mm": int(np.count_nonzero(values > 0.001)),
        "max_outside_m": float(np.max(outside)),
        "outside_p95_m": float(np.quantile(outside, 0.95)),
        "minimum_signed_m": float(np.min(values)),
        "signed_median_m": float(np.median(values)),
        "per_mesh": per_mesh,
        "uses_full_mesh_vertices": True,
        "optimizer_fit_ids_used": False,
    }


def _elbow_gap_metrics(
    vertices: np.ndarray,
    *,
    calibration: Any,
    layout: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for partition in PARTITIONS:
        result[partition] = {}
        for side, suffix in (("left", "L"), ("right", "R")):
            side_rows: dict[str, Any] = {}
            for query_bone, query_label in PRIMARY_BONES.items():
                domain_key = f"elbow/{side}/{query_bone}.{partition}"
                if domain_key not in calibration.domains:
                    raise KeyError(f"missing calibration domain {domain_key}")
                query_ids = np.asarray(calibration.domains[domain_key], dtype=np.int64).reshape(-1)
                if len(query_ids) < 3 or np.any(query_ids < 0) or np.any(query_ids >= len(vertices)):
                    raise ValueError(f"invalid calibration domain {domain_key}")
                for target_bone, target_label in PRIMARY_BONES.items():
                    if target_bone == query_bone:
                        continue
                    target_name = f"{target_label}_{suffix}"
                    info = layout[target_name]
                    start, stop = int(info["vertex_start"]), int(info["vertex_stop"])
                    unsigned, signed = _signed_distances(
                        vertices[query_ids],
                        vertices[start:stop],
                        info["faces_local"],
                    )
                    key = f"{side}_{query_bone}_to_{target_bone}"
                    side_rows[key] = {
                        "query_domain": domain_key,
                        "query_vertex_ids_global": query_ids.tolist(),
                        "target_mesh": target_name,
                        "target_mesh_vertex_range": [start, stop],
                        **_distance_row(unsigned, signed),
                    }
            result[partition][side] = side_rows
    return result


def _project_so3(matrices: np.ndarray) -> np.ndarray:
    """Project numerically drifted 3x3 frames to proper rotations."""

    values = np.asarray(matrices, dtype=np.float64).reshape(-1, 3, 3)
    u, _singular, vh = np.linalg.svd(values)
    result = u @ vh
    reflected = np.linalg.det(result) < 0.0
    if np.any(reflected):
        u = u.copy()
        u[reflected, :, -1] *= -1.0
        result[reflected] = u[reflected] @ vh[reflected]
    return result


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return SO(3) angles using ``actual @ inverse(expected)``.

    FK products carry small non-orthogonal roundoff after a 235-controller
    chain.  Projecting each input before constructing the relative rotation
    prevents trace/acos from turning that drift into a false anatomical angle.
    """

    actual = _project_so3(first)
    expected = _project_so3(second)
    relative = actual @ expected.transpose(0, 2, 1)
    return np.degrees(Rotation.from_matrix(relative).magnitude())


def _matrix_error_per_controller(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.max(
        np.abs(np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)),
        axis=(1, 2),
    )


def _fk_error(global_matrices: np.ndarray, parents: np.ndarray) -> float:
    local = _global_to_local(global_matrices, parents)
    reconstructed = _fk(local, parents)
    return float(np.max(np.abs(reconstructed - global_matrices)))


def _pose_frame(compiled: Any, pose: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    parents = np.asarray(compiled.parents, dtype=np.int64)
    source_global = np.asarray(compiled.source_globals(pose), dtype=np.float64)
    source_local = _global_to_local(source_global, parents)
    delta_raw = np.asarray(compiled.reference_local_inverse, dtype=np.float64) @ source_local
    delta = delta_raw.copy()
    delta[:, :3, 3] = np.einsum(
        "bij,bj->bi", np.asarray(compiled.translation_maps, dtype=np.float64), delta_raw[:, :3, 3]
    )
    target_local = np.asarray(compiled.target_local, dtype=np.float64)
    current_local = target_local @ delta
    current_global = _fk(current_local, parents)
    runtime_global = np.asarray(compiled.globals_from_source(source_global), dtype=np.float64)
    current_runtime_max_abs = float(np.max(np.abs(current_global - runtime_global)))

    delta_rot = delta.copy()
    delta_rot[:, :3, :3] = np.einsum(
        "bij,bjk,bkl->bil", q.transpose(0, 2, 1), delta[:, :3, :3], q
    )
    rotation_local = target_local @ delta_rot
    rotation_global = _fk(rotation_local, parents)

    delta_rot_trans = delta_rot.copy()
    delta_rot_trans[:, :3, 3] = np.einsum(
        "bij,bj->bi", q.transpose(0, 2, 1), delta[:, :3, 3]
    )
    rotation_translation_local = target_local @ delta_rot_trans
    rotation_translation_global = _fk(rotation_translation_local, parents)

    zero = not bool(np.any(np.asarray(pose)))
    if zero:
        target_bind = np.asarray(compiled.target_bind, dtype=np.float64)
        current_global = target_bind.copy()
        rotation_global = target_bind.copy()
        rotation_translation_global = target_bind.copy()
        current_vertices = np.asarray(compiled.target_rest, dtype=np.float64).copy()
        rotation_vertices = current_vertices.copy()
        rotation_translation_vertices = current_vertices.copy()
    else:
        target_inverse = np.asarray(compiled.target_inverse, dtype=np.float64)
        current_vertices = _weighted_rest_correction(
            np.asarray(compiled.target_rest, dtype=np.float64),
            np.asarray(compiled.indices, dtype=np.int64),
            np.asarray(compiled.weights, dtype=np.float64),
            current_global @ target_inverse,
        )
        rotation_vertices = _weighted_rest_correction(
            np.asarray(compiled.target_rest, dtype=np.float64),
            np.asarray(compiled.indices, dtype=np.int64),
            np.asarray(compiled.weights, dtype=np.float64),
            rotation_global @ target_inverse,
        )
        rotation_translation_vertices = _weighted_rest_correction(
            np.asarray(compiled.target_rest, dtype=np.float64),
            np.asarray(compiled.indices, dtype=np.int64),
            np.asarray(compiled.weights, dtype=np.float64),
            rotation_translation_global @ target_inverse,
        )
    return {
        "source_global": source_global,
        "delta_raw": delta_raw,
        "delta": delta,
        "current_global": current_global,
        "rotation_global": rotation_global,
        "rotation_translation_global": rotation_translation_global,
        "current_vertices": current_vertices,
        "rotation_vertices": rotation_vertices,
        "rotation_translation_vertices": rotation_translation_vertices,
        "zero_pose": zero,
        "current_runtime_max_abs_matrices": current_runtime_max_abs,
    }


def _variant_metrics(
    vertices: np.ndarray,
    *,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    asset: Any,
    left_arm_mesh_names: list[str],
    calibration: Any,
    layout: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "full_left_arm_bone_skin": _full_left_arm_skin_metric(
            vertices,
            skin=skin,
            skin_faces=skin_faces,
            asset=asset,
            mesh_names=left_arm_mesh_names,
        ),
        "elbow_gaps": _elbow_gap_metrics(vertices, calibration=calibration, layout=layout),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", type=Path, default=DEFAULT_COMPILED)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    compiled_path = Path(args.compiled).expanduser().resolve()
    input_root = Path(args.input).expanduser().resolve()
    operator_path = Path(args.operator).expanduser().resolve()
    calibration_path = Path(args.calibration).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not compiled_path.is_dir():
        raise FileNotFoundError(compiled_path)
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite probe output: {output}")

    started = time.perf_counter()
    compiled = load_compiled_subject(compiled_path)
    operator = load_source_operator(operator_path, validate=True, mmap=True)
    operator_digest = operator.runtime_digest(validate=False)
    source_pack_digest = str(compiled.provenance.get("shape_operator_runtime_digest", ""))
    if source_pack_digest and operator_digest != source_pack_digest:
        raise ValueError(
            "operator digest differs from compiled shape reference: "
            f"{operator_digest} != {source_pack_digest}"
        )
    calibration = load_anatomical_calibration_v1(
        calibration_path,
        operator=operator,
        required_scope="full_main_chain",
    )
    asset = compiled.source_asset
    faces = np.asarray(asset.faces, dtype=np.int32)
    if np.asarray(compiled.target_rest).shape != np.asarray(asset.vertices_rest).shape:
        raise ValueError("compiled target rest and source asset topology have different vertex counts")
    layout = _mesh_layout(asset, faces)
    left_arm_mesh_names = _left_arm_mesh_names(asset)
    names = [str(value) for value in asset.source_bone_names]
    parents = np.asarray(compiled.parents, dtype=np.int64)
    reference_bind = np.asarray(compiled.reference_bind, dtype=np.float64)
    target_bind = np.asarray(compiled.target_bind, dtype=np.float64)
    reference_rot = reference_bind[:, :3, :3]
    target_rot = target_bind[:, :3, :3]
    root_map = target_rot[0] @ reference_rot[0].T
    q = np.einsum(
        "bij,bjk,bkl->bil",
        reference_rot.transpose(0, 2, 1),
        np.broadcast_to(root_map.T, (len(names), 3, 3)),
        target_rot,
    )
    q_orth_error = float(np.max(np.abs(q @ q.transpose(0, 2, 1) - np.eye(3))))
    q_determinants = np.linalg.det(q)
    if not np.all(np.isfinite(q)) or np.any(q_determinants <= 0.0):
        raise ValueError("computed Q contains an invalid rotation")

    report: dict[str, Any] = {
        "schema_version": 14,
        "artifact_kind": "LocalRotationTransportProbeV14",
        "status": "running",
        "probe_only": True,
        "fit_or_tuning_performed": False,
        "production_candidate": False,
        "anatomical_passed": False,
        "publishable": False,
        "compiled": str(compiled_path),
        "compiled_manifest_sha256": _sha256(compiled_path / "manifest.json"),
        "input_root": str(input_root),
        "operator": str(operator_path),
        "operator_runtime_digest": operator_digest,
        "calibration": str(calibration_path),
        "calibration_content_digest": _calibration_content_digest(calibration),
        "calibration_fixed_domain_digest": str(getattr(calibration, "fixed_domain_digest", "")),
        "subject": str(args.subject),
        "pose_names": list(POSES),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "composition": {
            "matrix_convention": "column_vector_left_multiply; parent-local FK",
            "current_delta": "D_i = inv(L0bind_i) @ L0pose_i",
            "root_rotation_map": "Rrootmap = Rtarget_bind_root @ Rreference_bind_root.T",
            "Q": "Q_i = Rreference_bind_i.T @ Rrootmap.T @ Rtarget_bind_i",
            "primary": "Dprime_i.R = Q_i.T @ D_i.R @ Q_i; Dprime_i.t = existing translation_maps(D_i.t)",
            "translation_diagnostic": "Dprime_i.t = Q_i.T @ existing translation_maps(D_i.t), reported separately",
            "position_authority": "all 235 transforms use FK(target_local @ Dprime, parents); target-local bind translations are retained; no terminal override",
            "runtime_oracle": "compiled.source_globals(pose55) from the existing V14 source motion oracle",
        },
        "orientation_metric": {
            "angular_distance": "degrees(Rotation.from_matrix(project_so3(actual) @ project_so3(expected).T).magnitude())",
            "matrix_error": "max(abs(actual - expected)) per controller before SO(3) projection",
            "q_arrays": "raw bind-derived Q is retained in q_controller and used by the primary ablation",
        },
        "q_summary": {
            "controller_count": int(len(names)),
            "orthogonality_max_abs_error": q_orth_error,
            "minimum_determinant": float(np.min(q_determinants)),
            "maximum_determinant": float(np.max(q_determinants)),
            "max_abs_error_to_so3_projection": float(
                np.max(np.abs(q - _project_so3(q).reshape(q.shape)))
            ),
            "max_rotation_angle_to_so3_projection_deg": float(
                np.max(
                    _rotation_distance_deg(
                        q,
                        _project_so3(q).reshape(q.shape),
                    )
                )
            ),
            "rotation_angle_deg_min": float(np.min(_rotation_distance_deg(np.tile(np.eye(3), (len(names), 1, 1)), q))),
            "rotation_angle_deg_median": float(np.median(_rotation_distance_deg(np.tile(np.eye(3), (len(names), 1, 1)), q))),
            "rotation_angle_deg_max": float(np.max(_rotation_distance_deg(np.tile(np.eye(3), (len(names), 1, 1)), q))),
        },
        "cells": {},
        "failures": [],
    }
    output.mkdir(parents=True, exist_ok=False)
    try:
        for pose_name in POSES:
            cell_started = time.perf_counter()
            try:
                path = input_root / f"subject_{args.subject}_{pose_name}.npz"
                cell = _load_pose_cell(path, vertex_count=len(asset.vertices_rest), faces=faces)
                frame = _pose_frame(compiled, cell["pose"], q)
                current = np.asarray(frame["current_vertices"], dtype=np.float64)
                rotation = np.asarray(frame["rotation_vertices"], dtype=np.float64)
                rotation_translation = np.asarray(frame["rotation_translation_vertices"], dtype=np.float64)
                input_current_diff = float(
                    np.max(np.linalg.norm(current - cell["before_vertices"], axis=1))
                )
                variant_vertices = {
                    "current": current,
                    "rotation_conjugated": rotation,
                    "rotation_conjugated_translation_diagnostic": rotation_translation,
                }
                metrics = {
                    label: _variant_metrics(
                        values,
                        skin=cell["skin_vertices"],
                        skin_faces=cell["skin_faces"],
                        asset=asset,
                        left_arm_mesh_names=left_arm_mesh_names,
                        calibration=calibration,
                        layout=layout,
                    )
                    for label, values in variant_vertices.items()
                }
                current_rot = np.asarray(frame["current_global"][:, :3, :3], dtype=np.float64)
                rotation_rot = np.asarray(frame["rotation_global"][:, :3, :3], dtype=np.float64)
                rotation_translation_rot = np.asarray(
                    frame["rotation_translation_global"][:, :3, :3], dtype=np.float64
                )
                source_rot = np.asarray(frame["source_global"][:, :3, :3], dtype=np.float64)
                expected_rot = np.einsum(
                    "ij,bjk,bkl->bil", root_map, source_rot, q
                )
                expected_current_rot = np.einsum("ij,bjk->bik", root_map, source_rot)
                current_to_rotation = _rotation_distance_deg(current_rot, rotation_rot)
                rotation_to_expected = _rotation_distance_deg(rotation_rot, expected_rot)
                current_to_root_source = _rotation_distance_deg(current_rot, expected_current_rot)
                translation_current_to_rotation = np.linalg.norm(
                    frame["current_global"][:, :3, 3] - frame["rotation_global"][:, :3, 3], axis=1
                )
                translation_current_to_diagnostic = np.linalg.norm(
                    frame["current_global"][:, :3, 3]
                    - frame["rotation_translation_global"][:, :3, 3],
                    axis=1,
                )
                max_index = int(np.argmax(current_to_rotation))
                neutral = bool(frame["zero_pose"])
                neutral_checks = {
                    "current_global_target_bind_max_abs": float(
                        np.max(np.abs(frame["current_global"] - compiled.target_bind))
                    ) if neutral else None,
                    "rotation_global_target_bind_max_abs": float(
                        np.max(np.abs(frame["rotation_global"] - compiled.target_bind))
                    ) if neutral else None,
                    "translation_diagnostic_global_target_bind_max_abs": float(
                        np.max(np.abs(frame["rotation_translation_global"] - compiled.target_bind))
                    ) if neutral else None,
                    "current_vertices_target_rest_max_abs": float(
                        np.max(np.abs(current - compiled.target_rest))
                    ) if neutral else None,
                    "rotation_vertices_target_rest_max_abs": float(
                        np.max(np.abs(rotation - compiled.target_rest))
                    ) if neutral else None,
                    "translation_diagnostic_vertices_target_rest_max_abs": float(
                        np.max(np.abs(rotation_translation - compiled.target_rest))
                    ) if neutral else None,
                }
                output_path = output / f"subject_{args.subject}_{pose_name}.npz"
                metadata = {
                    "schema_version": 14,
                    "artifact_kind": "LocalRotationTransportProbeGeometryV14",
                    "subject": str(args.subject),
                    "pose": pose_name,
                    "source_label": "existing_raw142_source_vertices",
                    "before_label": "existing_compiled_current_candidate_vertices",
                    "current_label": "recomputed_current_compiled_response",
                    "candidate_label": "rotation_conjugated",
                    "translation_diagnostic_label": "rotation_conjugated_with_QT_translation_diagnostic",
                    "probe_only": True,
                    "production_candidate": False,
                    "anatomical_passed": False,
                    "publishable": False,
                    "compiled": str(compiled_path),
                    "composition": report["composition"],
                    "input_sha256": cell["sha256"],
                    "input_array_sha256": cell["array_sha256"],
                    "current_input_max_displacement_m": input_current_diff,
                }
                np.savez_compressed(
                    output_path,
                    faces=faces,
                    skin_faces=cell["skin_faces"],
                    source_vertices=np.asarray(cell["source_vertices"], dtype=np.float32),
                    before_vertices=np.asarray(cell["before_vertices"], dtype=np.float32),
                    current_vertices=np.asarray(current, dtype=np.float32),
                    candidate_vertices=np.asarray(rotation, dtype=np.float32),
                    rotation_conjugated_vertices=np.asarray(rotation, dtype=np.float32),
                    rotation_conjugated_translation_diagnostic_vertices=np.asarray(
                        rotation_translation, dtype=np.float32
                    ),
                    skin_vertices=np.asarray(cell["skin_vertices"], dtype=np.float64),
                    smplx_joints=np.asarray(cell["smplx_joints"], dtype=np.float64),
                    pose=np.asarray(cell["pose"], dtype=np.float32),
                    vertex_tissue=np.asarray(cell["vertex_tissue"], dtype=np.int8),
                    source_global=np.asarray(frame["source_global"], dtype=np.float64),
                    current_global=np.asarray(frame["current_global"], dtype=np.float64),
                    rotation_conjugated_global=np.asarray(frame["rotation_global"], dtype=np.float64),
                    rotation_conjugated_translation_diagnostic_global=np.asarray(
                        frame["rotation_translation_global"], dtype=np.float64
                    ),
                    q_controller=np.asarray(q, dtype=np.float64),
                    subject=np.asarray(str(args.subject)),
                    pose_name=np.asarray(pose_name),
                    metadata_json=np.asarray(json.dumps(_json_ready(metadata), sort_keys=True)),
                )
                report["cells"][pose_name] = {
                    "status": "complete",
                    "input": {
                        "path": str(cell["path"]),
                        "sha256": cell["sha256"],
                        "array_sha256": cell["array_sha256"],
                    },
                    "output": {
                        "path": str(output_path),
                        "sha256": _sha256(output_path),
                        "size_bytes": int(output_path.stat().st_size),
                    },
                    "pose_array_sha256": _array_sha256(cell["pose"]),
                    "current_reconstruction": {
                        "max_abs_matrix_difference_from_compiled_runtime": frame[
                            "current_runtime_max_abs_matrices"
                        ],
                        "max_input_candidate_displacement_m": input_current_diff,
                        "fk_reconstruction_max_abs_current": _fk_error(frame["current_global"], parents),
                        "fk_reconstruction_max_abs_rotation_conjugated": _fk_error(
                            frame["rotation_global"], parents
                        ),
                        "fk_reconstruction_max_abs_translation_diagnostic": _fk_error(
                            frame["rotation_translation_global"], parents
                        ),
                    },
                    "neutral_exact_checks": neutral_checks,
                    "global_orientation": {
                        "max_current_vs_rotation_conjugated_deg": float(np.max(current_to_rotation)),
                        "max_current_vs_rotation_conjugated_abs_matrix_error": float(
                            np.max(_matrix_error_per_controller(current_rot, rotation_rot))
                        ),
                        "controller_at_max_current_vs_rotation_conjugated": names[max_index],
                        "max_rotation_conjugated_vs_expected_deg": float(np.max(rotation_to_expected)),
                        "max_rotation_conjugated_vs_expected_abs_matrix_error": float(
                            np.max(_matrix_error_per_controller(rotation_rot, expected_rot))
                        ),
                        "controller_at_max_rotation_conjugated_vs_expected": names[int(np.argmax(rotation_to_expected))],
                        "max_current_vs_rootmapped_source_deg": float(np.max(current_to_root_source)),
                        "max_current_vs_rootmapped_source_abs_matrix_error": float(
                            np.max(_matrix_error_per_controller(current_rot, expected_current_rot))
                        ),
                        "controller_at_max_current_vs_rootmapped_source": names[int(np.argmax(current_to_root_source))],
                        "q_rotation_angle_deg_at_max_difference": float(
                            _rotation_distance_deg(
                                np.eye(3, dtype=np.float64)[None].repeat(len(names), axis=0),
                                q,
                            )[max_index]
                        ),
                        "max_global_translation_difference_current_to_rotation_m": float(
                            np.max(translation_current_to_rotation)
                        ),
                        "max_global_translation_difference_current_to_translation_diagnostic_m": float(
                            np.max(translation_current_to_diagnostic)
                        ),
                        "expected_rotation_formula": "Rrootmap @ RsourceG_i @ Q_i",
                    },
                    "variants": metrics,
                    "elapsed_seconds": float(time.perf_counter() - cell_started),
                }
                _write_json(output / "progress.json", report)
                print(
                    f"pose={pose_name} status=complete "
                    f"max_orientation_delta_deg={float(np.max(current_to_rotation)):.6f}",
                    flush=True,
                )
            except Exception as exc:
                failure = {
                    "pose": pose_name,
                    "status": "evaluation_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                report["cells"][pose_name] = failure
                report["failures"].append(failure)
                _write_json(output / "progress.json", report)
                print(f"pose={pose_name} status=evaluation_error error={type(exc).__name__}", flush=True)
    finally:
        report["status"] = "complete" if not report["failures"] else "evaluation_error"
        report["operational_evaluation_completed"] = bool(not report["failures"])
        report["cell_count"] = len(report["cells"])
        report["elapsed_seconds"] = float(time.perf_counter() - started)
        report["anatomical_passed"] = False
        report["publishable"] = False
        _write_json(output / "report.json", report)
        _write_json(output / "progress.json", report)
    print(
        f"local_rotation_transport_probe cells={report['cell_count']} "
        f"failures={len(report['failures'])} anatomical_passed=false output={output}",
        flush=True,
    )
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
