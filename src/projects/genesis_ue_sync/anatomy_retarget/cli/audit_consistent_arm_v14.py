"""Independently audit a connected V14 arm geometry export.

The input is a geometry NPZ (or a directory of such NPZs) containing the
source and candidate vertices for one or more poses.  This command is an
evidence pass only.  It never fits a map, changes an asset, calls Blender, or
uses an optimization sample to make a validation decision.

The audit deliberately keeps two kinds of evidence separate:

* frozen ``anatomical_calibration_v1`` ``validation`` domains are queried
  against complete opposing Humerus/Radius/Ulna meshes; and
* independent VTK triangle-pair collision checks inspect the complete left
  Humerus--Radius, Humerus--Ulna, and Radius--Ulna mesh pairs.  This catches
  thin-surface crossings that point samples can miss.

``full_arm_skin_boundary`` uses every vertex in the connected left-arm
  selection (Humerus, Radius, Ulna, and all wrist-descendant hand bones),
  matching the V14 fit report's ``builder.all_ids`` population.  It is kept
distinct from the optimizer's ``fit_ids`` subset so an optimization sample
cannot be mistaken for full-mesh validation.

For non-T poses, ``posed_cap_rigidity_against_own_tpose`` compares each
variant with its own T-pose cap.  Its Procrustes residual is gated at 0.1 mm;
the fitted rotation and translation describe rigid pose motion and are not
penalized. A rest cap pass cannot bypass this posed deformation gate.

The T-pose cap check is deliberately split into two references.  The normal
gated check is incremental candidate-versus-the-subject's materialized
``raw142`` geometry.  If the input's compiled provenance explicitly declares
``shape_reference_kind=frozen_operator_template``, the candidate-versus-frozen
operator-template comparison is gated as the total authored-142 shape check;
the raw subject-materialization comparison remains a separately reported
diagnostic.  Inputs without that provenance keep the legacy ungated template
reference, because their upstream shape basis is not authenticated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import igl
import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    _calibration_content_digest,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.surface_validation_v14 import audit_bone_pair
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import load_source_operator
from projects.genesis_ue_sync.anatomy_retarget.cli.audit_capture_articular_surfaces_v13 import (
    _mesh_face_subset,
    _mesh_quality,
    _query_distances,
)


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_INPUT = ROOT / "outputs/anatomy_retarget/v14_arm_fit_20260908_001"
DEFAULT_OUTPUT = ROOT / "outputs/anatomy_retarget/v14_arm_audit_20260908_001"
DEFAULT_OPERATOR = ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
DEFAULT_CALIBRATION = ROOT / (
    "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006/"
    "anatomical_calibration_v1"
)

DEFAULT_SUBJECT = "213328"
DEFAULT_POSES = ("tpose", "pose_213328", "heldout_sitting")
PARTITION = "validation"
NEGATIVE_SIGNED_THRESHOLD_M = 1.0e-4
TRIANGLE_DEPTH_TOLERANCE_M = 5.0e-4
OUTSIDE_REPORT_THRESHOLD_M = 1.0e-3
CAP_SHAPE_RMS_GATE_M = 1.0e-4
CAP_SHAPE_MAX_GATE_M = 1.0e-4
CAP_PAIRWISE_P95_GATE_M = 1.0e-4
PRIMARY_BONES = {"humerus": "Humerus", "radius": "Radius", "ulna": "Ulna"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


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


def _parse_csv(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _load_cell(path: Path) -> dict[str, Any]:
    """Load and authenticate one exported source/candidate geometry cell."""

    with np.load(path, allow_pickle=False) as data:
        required = {
            "faces",
            "source_vertices",
            "candidate_vertices",
            "skin_vertices",
            "skin_faces",
            "pose",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path}: missing geometry fields {missing}")
        faces = np.asarray(data["faces"], dtype=np.int32)
        source = np.asarray(data["source_vertices"], dtype=np.float64)
        candidate = np.asarray(data["candidate_vertices"], dtype=np.float64)
        skin = np.asarray(data["skin_vertices"], dtype=np.float64)
        skin_faces = np.asarray(data["skin_faces"], dtype=np.int32)
        pose = np.asarray(data["pose"], dtype=np.float32).reshape(-1)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"{path}: faces must have shape [F,3]")
        if source.ndim != 2 or source.shape[1] != 3 or candidate.shape != source.shape:
            raise ValueError(f"{path}: source/candidate vertices have incompatible shapes")
        if skin.ndim != 2 or skin.shape[1] != 3:
            raise ValueError(f"{path}: skin_vertices must have shape [N,3]")
        if skin_faces.ndim != 2 or skin_faces.shape[1] != 3:
            raise ValueError(f"{path}: skin_faces must have shape [F,3]")
        if len(faces) == 0 or np.any(faces < 0) or np.any(faces >= len(source)):
            raise ValueError(f"{path}: anatomy faces reference invalid vertices")
        if len(skin_faces) == 0 or np.any(skin_faces < 0) or np.any(skin_faces >= len(skin)):
            raise ValueError(f"{path}: skin faces reference invalid vertices")
        if pose.size != 165:
            raise ValueError(f"{path}: pose must contain 55*3 values")
        for name, values in (
            ("source_vertices", source),
            ("candidate_vertices", candidate),
            ("skin_vertices", skin),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{path}: {name} contains non-finite values")
        tissue_hash = _array_sha256(data["vertex_tissue"]) if "vertex_tissue" in data.files else None
    return {
        "path": path.resolve(),
        "sha256": _sha256(path),
        "faces": faces,
        "source_vertices": source,
        "candidate_vertices": candidate,
        "skin_vertices": skin,
        "skin_faces": skin_faces,
        "pose": pose.reshape(55, 3),
        "array_hashes": {
            "faces": _array_sha256(faces),
            "source_vertices": _array_sha256(source),
            "candidate_vertices": _array_sha256(candidate),
            "skin_vertices": _array_sha256(skin),
            "skin_faces": _array_sha256(skin_faces),
            "vertex_tissue": tissue_hash,
        },
    }


def _mesh_layout(asset: Any, faces: np.ndarray) -> dict[str, dict[str, Any]]:
    names = [str(value) for value in asset.source_mesh_names]
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    result: dict[str, dict[str, Any]] = {}
    for name in ("Humerus_L", "Radius_L", "Ulna_L", "Humerus_R", "Radius_R", "Ulna_R"):
        if name not in names:
            raise ValueError(f"source asset is missing required mesh {name!r}")
        index = names.index(name)
        start, stop = map(int, ranges[index])
        face_ids, local_faces = _mesh_face_subset(faces, start=start, stop=stop)
        if len(face_ids) == 0:
            raise ValueError(f"source mesh {name!r} has no complete local triangles")
        result[name] = {
            "mesh_index": int(index),
            "vertex_start": start,
            "vertex_stop": stop,
            "vertex_count": stop - start,
            "face_global_ids": face_ids,
            "faces_local": local_faces,
            "face_count": int(len(local_faces)),
        }
    return result


def _left_arm_mesh_names(asset: Any) -> list[str]:
    """Reproduce ArmMap.all_ids' complete connected left-arm population."""

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
        raise ValueError("left arm selection does not contain all three primary bones")
    return selected


def _summary(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(data) or not np.all(np.isfinite(data)):
        raise ValueError("empty or non-finite metric sample")
    return {
        "min_m": float(np.min(data)),
        "p05_m": float(np.quantile(data, 0.05)),
        "median_m": float(np.median(data)),
        "p95_m": float(np.quantile(data, 0.95)),
        "max_m": float(np.max(data)),
    }


def _full_arm_skin_metric(
    vertices: np.ndarray,
    *,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    asset: Any,
    mesh_names: list[str],
) -> dict[str, Any]:
    names = [str(value) for value in asset.source_mesh_names]
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    faces = np.asarray(asset.faces, dtype=np.int64)
    per_mesh: dict[str, Any] = {}
    all_signed: list[np.ndarray] = []
    for mesh_name in mesh_names:
        index = names.index(mesh_name)
        start, stop = map(int, ranges[index])
        values = np.asarray(
            igl.signed_distance(
                np.ascontiguousarray(vertices[start:stop], dtype=np.float64),
                np.ascontiguousarray(skin, dtype=np.float64),
                np.ascontiguousarray(skin_faces, dtype=np.int64),
            )[0],
            dtype=np.float64,
        ).reshape(-1)
        if len(values) != stop - start or not np.all(np.isfinite(values)):
            raise ValueError(f"skin signed distance failed for mesh {mesh_name}")
        all_signed.append(values)
        outside = np.maximum(values, 0.0)
        per_mesh[mesh_name] = {
            "vertex_range": [start, stop],
            "vertex_count": int(len(values)),
            "outside_vertex_count_gt_0": int(np.count_nonzero(values > 0.0)),
            "outside_vertex_count_gt_1mm": int(np.count_nonzero(values > OUTSIDE_REPORT_THRESHOLD_M)),
            "max_signed_distance_m": float(np.max(values)),
            "max_outside_m": float(np.max(outside)),
            "outside_p95_m": float(np.quantile(outside, 0.95)),
            "signed_p05_m": float(np.quantile(values, 0.05)),
            "signed_median_m": float(np.median(values)),
            "signed_p95_m": float(np.quantile(values, 0.95)),
            "minimum_signed_m": float(np.min(values)),
        }
    aggregate = np.concatenate(all_signed) if all_signed else np.empty(0, dtype=np.float64)
    outside = np.maximum(aggregate, 0.0)
    return {
        "population": "complete_connected_left_arm_meshes",
        "mesh_names": list(mesh_names),
        "vertex_count": int(len(aggregate)),
        "outside_vertex_count_gt_0": int(np.count_nonzero(aggregate > 0.0)),
        "outside_vertex_count_gt_1mm": int(np.count_nonzero(aggregate > OUTSIDE_REPORT_THRESHOLD_M)),
        "max_signed_distance_m": float(np.max(aggregate)),
        "max_outside_m": float(np.max(outside)),
        "outside_p95_m": float(np.quantile(outside, 0.95)),
        "signed_p05_m": float(np.quantile(aggregate, 0.05)),
        "signed_median_m": float(np.median(aggregate)),
        "signed_p95_m": float(np.quantile(aggregate, 0.95)),
        "minimum_signed_m": float(np.min(aggregate)),
        "per_mesh": per_mesh,
        "signed_convention": "negative_inside_SMPLX_skin; positive is outside",
        "uses_full_mesh_vertices": True,
        "optimizer_fit_ids_used": False,
    }


def _cap_shape_metrics(
    source: np.ndarray,
    candidate: np.ndarray,
    *,
    apply_gate: bool = True,
) -> dict[str, Any]:
    """Rigid Procrustes plus pairwise distance preservation for one cap correspondence."""

    x = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    y = np.asarray(candidate, dtype=np.float64).reshape(-1, 3)
    if x.shape != y.shape or len(x) < 3:
        raise ValueError("cap shape arrays must have matching [N,3] shapes with N>=3")
    cx = np.mean(x, axis=0)
    cy = np.mean(y, axis=0)
    xx = x - cx
    yy = y - cy
    u, _singular, vh = np.linalg.svd(xx.T @ yy)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vh[-1, :] *= -1.0
        rotation = vh.T @ u.T
    translation = cy - rotation @ cx
    aligned = xx @ rotation.T + cy
    residual = np.linalg.norm(aligned - y, axis=1)
    source_pair = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    candidate_pair = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=2)
    upper = np.triu_indices(len(x), k=1)
    pair_source = source_pair[upper]
    pair_candidate = candidate_pair[upper]
    valid = pair_source > 1.0e-9
    pair_abs = np.abs(pair_candidate[valid] - pair_source[valid])
    pair_rel = pair_abs / pair_source[valid]
    rms = float(np.sqrt(np.mean(residual**2)))
    max_residual = float(np.max(residual))
    pairwise_p95 = float(np.quantile(pair_abs, 0.95))
    # The trace of a proper rotation is clamped because tiny floating point
    # noise can otherwise produce acos(1+epsilon).  The angle and translation
    # describe pose motion; they are not shape errors.
    rotation_trace = float(np.trace(rotation))
    rotation_angle = float(
        np.arccos(np.clip((rotation_trace - 1.0) * 0.5, -1.0, 1.0))
    )
    passed = bool(
        apply_gate
        and rms <= CAP_SHAPE_RMS_GATE_M
        and max_residual <= CAP_SHAPE_MAX_GATE_M
        and pairwise_p95 <= CAP_PAIRWISE_P95_GATE_M
    )
    return {
        "vertex_count": int(len(x)),
        "rigid_procrustes_rms_m": rms,
        "rigid_procrustes_p95_m": float(np.quantile(residual, 0.95)),
        "rigid_procrustes_max_m": max_residual,
        "rigid_rotation_determinant": float(np.linalg.det(rotation)),
        "rigid_motion_rotation_angle_rad": rotation_angle,
        "rigid_motion_rotation_angle_deg": float(np.degrees(rotation_angle)),
        "rigid_translation_m": translation.tolist(),
        "rigid_motion_translation_norm_m": float(np.linalg.norm(translation)),
        "pair_count": int(np.count_nonzero(valid)),
        "pairwise_absolute_p95_m": pairwise_p95,
        "pairwise_absolute_max_m": float(np.max(pair_abs)),
        "pairwise_relative_p95": float(np.quantile(pair_rel, 0.95)),
        "pairwise_relative_max": float(np.max(pair_rel)),
        "gates": {
            "rigid_procrustes_rms_m": CAP_SHAPE_RMS_GATE_M,
            "rigid_procrustes_max_m": CAP_SHAPE_MAX_GATE_M,
            "pairwise_absolute_p95_m": CAP_PAIRWISE_P95_GATE_M,
        },
        "shape_preservation_passed": passed if apply_gate else None,
        "shape_gate_applied": bool(apply_gate),
        "interpretation": "corresponding cap points are compared after one rigid transform; translation is intentionally not penalized",
    }


def _triangle_pair_summary(
    source_or_candidate: np.ndarray,
    *,
    layout: Mapping[str, Mapping[str, Any]],
    first: str,
    second: str,
) -> dict[str, Any]:
    a_info = layout[first]
    b_info = layout[second]
    a_start, a_stop = int(a_info["vertex_start"]), int(a_info["vertex_stop"])
    b_start, b_stop = int(b_info["vertex_start"]), int(b_info["vertex_stop"])
    result = audit_bone_pair(
        source_or_candidate[a_start:a_stop],
        a_info["faces_local"],
        source_or_candidate[b_start:b_stop],
        b_info["faces_local"],
        depth_tolerance_m=TRIANGLE_DEPTH_TOLERANCE_M,
    )
    # The full triangle-pair list is useful interactively but can be very
    # large.  Keep counts, quality, and all signed depth summaries in this
    # reusable report, with a short reproducible sample of pair IDs.
    pairs = np.asarray(result.get("triangle_pairs", []), dtype=np.int64).reshape(-1, 2)
    return {
        "pair": f"{first}--{second}",
        "first_mesh": first,
        "second_mesh": second,
        "mesh_quality": result.get("mesh_quality"),
        "triangle_pair_count": int(result.get("triangle_pair_count", len(pairs))),
        "triangle_pairs_sample": pairs[:20].tolist(),
        "signed_samples": result.get("signed_samples"),
        "depth_tolerance_m": float(result.get("depth_tolerance_m", TRIANGLE_DEPTH_TOLERANCE_M)),
        "max_depth_is_sampled_lower_bound": bool(result.get("max_depth_is_sampled_lower_bound", True)),
        "passed": bool(result.get("passed", False)),
        "reason": str(result.get("reason", "unknown")),
        "method": str(result.get("method", "unknown")),
        "vtk_version": str(result.get("vtk_version", "unknown")),
        "separate_from_calibration_domain_queries": True,
    }


def _penetration_metrics(
    vertices: np.ndarray,
    *,
    calibration: Any,
    layout: Mapping[str, Mapping[str, Any]],
    sides: tuple[str, ...] = ("left", "right"),
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for side in sides:
        suffix = "L" if side == "left" else "R"
        result[side] = {}
        for query_bone, query_label in PRIMARY_BONES.items():
            domain_key = f"elbow/{side}/{query_bone}.{PARTITION}"
            if domain_key not in calibration.domains:
                raise KeyError(f"missing frozen calibration domain {domain_key!r}")
            query_ids = np.asarray(calibration.domains[domain_key], dtype=np.int64).reshape(-1)
            if len(query_ids) < 3 or np.any(query_ids < 0) or np.any(query_ids >= len(vertices)):
                raise ValueError(f"invalid calibration domain {domain_key!r}")
            query = vertices[query_ids]
            for target_bone, target_label in PRIMARY_BONES.items():
                if target_bone == query_bone:
                    continue
                target_name = f"{target_label}_{suffix}"
                target = layout[target_name]
                target_start, target_stop = int(target["vertex_start"]), int(target["vertex_stop"])
                target_vertices = vertices[target_start:target_stop]
                quality = _mesh_quality(target_vertices, target["faces_local"])
                key = f"{side}_{query_bone}_to_{target_bone}"
                summary = _query_distances(
                    query,
                    target_vertices,
                    target["faces_local"],
                    target["face_global_ids"],
                    quality,
                )
                summary.update(
                    {
                        "query_domain": domain_key,
                        "query_vertex_ids_global": query_ids.tolist(),
                        "target_mesh": target_name,
                        "target_mesh_vertex_range": [target_start, target_stop],
                        "target_mesh_quality": quality,
                        "validation_partition_only": True,
                    }
                )
                result[side][key] = summary
    return result


def _audit_cell(
    cell: Mapping[str, Any],
    *,
    subject: str,
    pose: str,
    calibration: Any,
    asset: Any,
    layout: Mapping[str, Mapping[str, Any]],
    arm_mesh_names: list[str],
) -> dict[str, Any]:
    source = np.asarray(cell["source_vertices"], dtype=np.float64)
    candidate = np.asarray(cell["candidate_vertices"], dtype=np.float64)
    faces = np.asarray(cell["faces"], dtype=np.int32)
    if len(source) != len(asset.vertices_rest):
        raise ValueError(
            f"{cell['path']}: vertex count {len(source)} does not match source asset {len(asset.vertices_rest)}"
        )
    if not np.array_equal(faces, np.asarray(asset.faces, dtype=np.int32)):
        raise ValueError(f"{cell['path']}: input faces differ from frozen source topology")
    skin = np.asarray(cell["skin_vertices"], dtype=np.float64)
    skin_faces = np.asarray(cell["skin_faces"], dtype=np.int32)
    variants: dict[str, Any] = {}
    for label, vertices in (("raw142", source), ("candidate", candidate)):
        variants[label] = {
            "full_arm_skin_boundary": _full_arm_skin_metric(
                vertices,
                skin=skin,
                skin_faces=skin_faces,
                asset=asset,
                mesh_names=arm_mesh_names,
            ),
            "validation_domain_signed_penetration": _penetration_metrics(
                vertices,
                calibration=calibration,
                layout=layout,
            ),
            "triangle_pair_checks": {
                "Humerus_L--Radius_L": _triangle_pair_summary(
                    vertices,
                    layout=layout,
                    first="Humerus_L",
                    second="Radius_L",
                ),
                "Humerus_L--Ulna_L": _triangle_pair_summary(
                    vertices,
                    layout=layout,
                    first="Humerus_L",
                    second="Ulna_L",
                ),
                "Radius_L--Ulna_L": _triangle_pair_summary(
                    vertices,
                    layout=layout,
                    first="Radius_L",
                    second="Ulna_L",
                ),
            },
        }
    return {
        "status": "complete",
        "subject": subject,
        "pose": pose,
        "input": {
            "path": str(cell["path"]),
            "sha256": str(cell["sha256"]),
            "array_hashes": dict(cell["array_hashes"]),
        },
        "pose_array_hash": _array_sha256(cell["pose"]),
        "partition": PARTITION,
        "variants": variants,
        "validation_domain_policy": "original anatomical_calibration_v1 validation IDs; no fit IDs are used",
        "triangle_pair_policy": "complete left-arm triangle surfaces independently checked with VTK OBB collision plus bidirectional vertex/face-centroid signed samples",
    }


def _find_input(input_root: Path, subject: str, pose: str) -> Path:
    if input_root.is_file():
        return input_root.resolve()
    candidates = (
        input_root / f"subject_{subject}_{pose}.npz",
        input_root / subject / f"subject_{subject}_{pose}.npz",
        input_root / f"subject_{subject}" / "comparisons" / f"{pose}.npz",
        input_root / "subjects" / f"subject_{subject}" / "comparisons" / f"{pose}.npz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"no geometry NPZ for subject={subject} pose={pose}; searched "
        + ", ".join(str(value) for value in candidates)
    )


def _infer_file_identity(path: Path) -> tuple[str, str]:
    match = re.match(r"subject_(?P<subject>[^_]+)_(?P<pose>.+)\.npz$", path.name)
    if not match:
        return "unknown", path.stem
    return match.group("subject"), match.group("pose")


def _candidate_provenance(input_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not input_root.is_dir():
        return result
    for name in ("report.json", "progress.json", "compiled/manifest.json"):
        path = input_root / name
        if not path.is_file():
            continue
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            parsed = None
        entry: dict[str, Any] = {"path": str(path.resolve()), "sha256": _sha256(path)}
        if isinstance(parsed, dict):
            if name == "report.json":
                for key in (
                    "status",
                    "subject",
                    "fit_poses",
                    "regression_poses",
                    "best_parameters_mm",
                    "optimizer_success",
                    "optimizer_message",
                    "independent_validation_complete",
                    "anatomical_passed",
                    "publishable",
                    "shape_reference_kind",
                ):
                    if key in parsed:
                        entry[key] = parsed[key]
            elif name == "compiled/manifest.json":
                for key in (
                    "artifact_kind",
                    "schema_version",
                    "composition",
                    "publishable",
                    "anatomical_passed",
                    "shape_reference_kind",
                ):
                    if key in parsed:
                        entry[key] = parsed[key]
            # Compiled manifests keep shape provenance under ``provenance``;
            # accepting a top-level field as well makes this reader robust to
            # the equivalent report form without weakening the exact-value
            # gate below.
            nested = parsed.get("provenance")
            if isinstance(nested, Mapping) and "shape_reference_kind" in nested:
                entry["shape_reference_kind"] = nested["shape_reference_kind"]
        result[name.replace("/", "_")] = entry
    return result


def _shape_reference_kind(provenance: Mapping[str, Any]) -> str | None:
    """Return one authenticated shape reference kind, or an explicit conflict."""

    values: set[str] = set()
    for entry in provenance.values():
        if not isinstance(entry, Mapping):
            continue
        value = entry.get("shape_reference_kind")
        if value is not None and str(value).strip():
            values.add(str(value).strip())
    if len(values) == 1:
        return next(iter(values))
    if len(values) > 1:
        return "conflicting"
    return None


def _tpose_cap_template_metrics(
    source: np.ndarray,
    candidate: np.ndarray,
    *,
    template: np.ndarray,
    calibration: Any,
    template_gate_applied: bool = False,
) -> dict[str, Any]:
    """Compare subject materialization and candidate caps with the frozen template.

    The candidate row applies the 0.1 mm cap gate only when the caller has an
    authenticated ``frozen_operator_template`` provenance.  The raw row uses
    the same gate as a useful upstream diagnostic in that mode, but it does
    not decide the candidate's total authored-142 result: subject
    materialization can alter the raw shape before the map is applied.
    """

    frozen = np.asarray(template, dtype=np.float64)
    raw = np.asarray(source, dtype=np.float64)
    mapped = np.asarray(candidate, dtype=np.float64)
    if frozen.shape != raw.shape or mapped.shape != raw.shape:
        raise ValueError("template, source, and candidate vertices must have the same shape")
    rows: dict[str, Any] = {}
    for side in ("left", "right"):
        for bone in PRIMARY_BONES:
            domain_key = f"elbow/{side}/{bone}.{PARTITION}"
            ids = np.asarray(calibration.domains[domain_key], dtype=np.int64).reshape(-1)
            if len(ids) < 3 or np.any(ids < 0) or np.any(ids >= len(frozen)):
                raise ValueError(f"invalid frozen calibration domain {domain_key!r}")
            raw_metrics = _cap_shape_metrics(
                frozen[ids],
                raw[ids],
                apply_gate=template_gate_applied,
            )
            candidate_metrics = _cap_shape_metrics(
                frozen[ids],
                mapped[ids],
                apply_gate=template_gate_applied,
            )
            rows[f"{side}_{bone}"] = {
                "query_domain": domain_key,
                "query_vertex_ids_global": ids.tolist(),
                "reference_geometry": "operator.template_asset.vertices_rest",
                "reference_geometry_digest": _array_sha256(frozen),
                "gate_applied": bool(template_gate_applied),
                "total_authored_142_gate_demonstrated": bool(
                    template_gate_applied and candidate_metrics["shape_preservation_passed"]
                ),
                "raw_subject_materialization_gate_passed": (
                    raw_metrics["shape_preservation_passed"]
                    if template_gate_applied
                    else None
                ),
                "raw142_subject_materialized_vs_template": {
                    "geometry": "subject-materialized raw142 T-pose",
                    "comparison": "operator.template_asset.vertices_rest -> raw142 subject geometry",
                    **raw_metrics,
                },
                "candidate_subject_materialized_vs_template": {
                    "geometry": "candidate T-pose",
                    "comparison": "operator.template_asset.vertices_rest -> candidate geometry",
                    **candidate_metrics,
                },
                "interpretation": (
                    "candidate-versus-frozen operator template is the total authored-142 cap gate; "
                    "raw subject materialization is retained as a diagnostic"
                    if template_gate_applied
                    else
                    "materialize_subject(beta) may change cap shape before the candidate map; "
                    "these ungated rows are a reference limitation, not a user-gate pass/fail"
                ),
            }
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="geometry NPZ or directory containing subject_<subject>_<pose>.npz files",
    )
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR, help="source operator used for mesh ranges")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subjects", help="comma-separated subject IDs; directory default is 213328")
    parser.add_argument("--poses", help="comma-separated pose names; directory default is tpose,pose_213328,heldout_sitting")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    input_root = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    operator_path = Path(args.operator).expanduser().resolve()
    calibration_path = Path(args.calibration).expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(input_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite arm audit output: {output}")
    if input_root.is_file():
        inferred_subject, inferred_pose = _infer_file_identity(input_root)
        subjects = _parse_csv(args.subjects) or (inferred_subject,)
        poses = _parse_csv(args.poses) or (inferred_pose,)
        if len(subjects) != 1 or len(poses) != 1:
            raise ValueError("a direct --input NPZ accepts exactly one subject and one pose")
    else:
        subjects = _parse_csv(args.subjects) or (DEFAULT_SUBJECT,)
        poses = _parse_csv(args.poses) or DEFAULT_POSES
    if not subjects or not poses:
        raise ValueError("at least one subject and one pose are required")

    started = time.perf_counter()
    operator = load_source_operator(operator_path, validate=True, mmap=True)
    operator_digest = operator.runtime_digest(validate=False)
    calibration = load_anatomical_calibration_v1(
        calibration_path,
        operator=operator,
        required_scope="full_main_chain",
    )
    calibration_digest = _calibration_content_digest(calibration)
    asset = operator.template_asset
    faces_reference: np.ndarray | None = None
    layout: dict[str, dict[str, Any]] | None = None
    arm_mesh_names = _left_arm_mesh_names(asset)
    candidate_provenance = _candidate_provenance(input_root)
    shape_reference_kind = _shape_reference_kind(candidate_provenance)
    template_reference_gate_applied = shape_reference_kind == "frozen_operator_template"
    report: dict[str, Any] = {
        "schema_version": 14,
        "artifact_kind": "ConsistentArmV14IndependentAudit",
        "status": "running",
        "audit_status": "running",
        "anatomical_passed": False,
        "publishable": False,
        "operational_evaluation_completed": False,
        "evidence_only": True,
        "fit_or_tuning_performed": False,
        "partition": PARTITION,
        "signed_convention": "negative means query point is inside a watertight target bone; positive means outside",
        "negative_signed_threshold_m": NEGATIVE_SIGNED_THRESHOLD_M,
        "outside_report_threshold_m": OUTSIDE_REPORT_THRESHOLD_M,
        "triangle_depth_tolerance_m": TRIANGLE_DEPTH_TOLERANCE_M,
        "cap_shape_gates": {
            "rigid_procrustes_rms_m": CAP_SHAPE_RMS_GATE_M,
            "rigid_procrustes_max_m": CAP_SHAPE_MAX_GATE_M,
            "pairwise_absolute_p95_m": CAP_PAIRWISE_P95_GATE_M,
        },
        "cap_shape_scope": {
            "gated_comparison": "candidate T-pose cap versus subject-materialized raw142 T-pose cap",
            "ungated_reference_comparison": "subject-materialized raw142 and candidate T-pose caps versus operator.template_asset.vertices_rest when frozen-template provenance is absent",
            "gate_scope": (
                "0.1 mm cap gates apply to the incremental candidate-versus-materialized-raw142 comparison; "
                "with authenticated frozen_operator_template provenance they also apply to the candidate-versus-frozen-template total authored-142 comparison"
            ),
            "shape_reference_kind": shape_reference_kind,
            "template_reference_gate_applied": template_reference_gate_applied,
            "total_authored_142_gate_demonstrated": False,
            "reason": (
                "candidate-versus-frozen-template is gated only when the input compiled provenance explicitly "
                "declares shape_reference_kind=frozen_operator_template; raw subject materialization remains "
                "a separate diagnostic"
            ),
        },
        "input_root": str(input_root),
        "operator": str(operator_path),
        "operator_runtime_digest": operator_digest,
        "calibration": str(calibration_path),
        "calibration_content_digest": calibration_digest,
        "calibration_fixed_domain_digest": str(getattr(calibration, "fixed_domain_digest", "")),
        "calibration_partition_policy": "validation only; fit domains are never queried by this audit",
        "script_sha256": _sha256(Path(__file__).resolve()),
        "subjects_requested": list(subjects),
        "poses_requested": list(poses),
        "left_arm_mesh_names": arm_mesh_names,
        "left_arm_mesh_population_policy": "Humerus_L/Radius_L/Ulna_L plus every bone mesh owned by the Wrist_Rotate_L descendant subtree",
        "candidate_provenance": candidate_provenance,
        "cells": {},
        "tpose_cap_shape_preservation": {},
        "tpose_cap_shape_vs_frozen_template": {},
        "failures": [],
        "candidate_flags": [],
    }
    output.mkdir(parents=True, exist_ok=False)
    try:
        for subject in subjects:
            report["cells"].setdefault(subject, {})
            for pose in poses:
                cell_started = time.perf_counter()
                try:
                    path = _find_input(input_root, subject, pose)
                    cell = _load_cell(path)
                    if faces_reference is None:
                        faces_reference = np.asarray(cell["faces"], dtype=np.int32)
                        layout = _mesh_layout(asset, faces_reference)
                    elif not np.array_equal(faces_reference, np.asarray(cell["faces"], dtype=np.int32)):
                        raise ValueError(f"{path}: faces differ from first input cell")
                    assert layout is not None
                    audited = _audit_cell(
                        cell,
                        subject=subject,
                        pose=pose,
                        calibration=calibration,
                        asset=asset,
                        layout=layout,
                        arm_mesh_names=arm_mesh_names,
                    )
                    audited["elapsed_seconds"] = float(time.perf_counter() - cell_started)
                    report["cells"][subject][pose] = audited
                    _write_json(output / "progress.json", report)
                    print(f"subject={subject} pose={pose} status=complete", flush=True)
                except Exception as exc:
                    failure = {
                        "subject": subject,
                        "pose": pose,
                        "status": "evaluation_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    report["cells"][subject][pose] = failure
                    report["failures"].append(failure)
                    _write_json(output / "progress.json", report)
                    print(
                        f"subject={subject} pose={pose} status=evaluation_error error={type(exc).__name__}",
                        flush=True,
                    )

        # Cap preservation is defined in T only and uses the same frozen
        # validation IDs as penetration queries.  It is intentionally a
        # separate section so moving an entire cap is not reported as a shape
        # distortion.
        for subject in subjects:
            tpose = report["cells"].get(subject, {}).get("tpose")
            if not isinstance(tpose, dict) or tpose.get("status") != "complete":
                continue
            tpose_cell = _load_cell(Path(tpose["input"]["path"]))
            source = tpose_cell["source_vertices"]
            candidate = tpose_cell["candidate_vertices"]
            cap_rows: dict[str, Any] = {}
            for side in ("left", "right"):
                for bone in PRIMARY_BONES:
                    domain_key = f"elbow/{side}/{bone}.{PARTITION}"
                    ids = np.asarray(calibration.domains[domain_key], dtype=np.int64).reshape(-1)
                    cap_rows[f"{side}_{bone}"] = {
                        "query_domain": domain_key,
                        "query_vertex_ids_global": ids.tolist(),
                        **_cap_shape_metrics(source[ids], candidate[ids]),
                    }
            report["tpose_cap_shape_preservation"][subject] = cap_rows
            report["tpose_cap_shape_vs_frozen_template"][subject] = _tpose_cap_template_metrics(
                source,
                candidate,
                template=np.asarray(asset.vertices_rest, dtype=np.float64),
                calibration=calibration,
                template_gate_applied=template_reference_gate_applied,
            )
            if template_reference_gate_applied:
                template_rows = report["tpose_cap_shape_vs_frozen_template"][subject]
                candidate_passed = all(
                    bool(row.get("total_authored_142_gate_demonstrated"))
                    for row in template_rows.values()
                )
                report["cap_shape_scope"]["total_authored_142_gate_demonstrated"] = bool(
                    candidate_passed
                )
            _write_json(output / "progress.json", report)

            # Pose rigidity is an independent deformation gate. Each
            # variant is compared with its own T-pose rest geometry so a
            # source/candidate station offset is not mistaken for deformation.
            # The fitted rigid motion is free; the residual is bounded.
            for pose, pose_row in report["cells"].get(subject, {}).items():
                if pose == "tpose" or not isinstance(pose_row, dict) or pose_row.get("status") != "complete":
                    continue
                pose_cell = _load_cell(Path(pose_row["input"]["path"]))
                rigidity_rows: dict[str, Any] = {}
                for variant_name, rest_vertices, posed_vertices in (
                    ("raw142", source, pose_cell["source_vertices"]),
                    ("candidate", candidate, pose_cell["candidate_vertices"]),
                ):
                    per_cap: dict[str, Any] = {}
                    for side in ("left", "right"):
                        for bone in PRIMARY_BONES:
                            domain_key = f"elbow/{side}/{bone}.{PARTITION}"
                            ids = np.asarray(calibration.domains[domain_key], dtype=np.int64).reshape(-1)
                            row = _cap_shape_metrics(
                                rest_vertices[ids],
                                posed_vertices[ids],
                                apply_gate=True,
                            )
                            row.update(
                                {
                                    "reference_variant": variant_name,
                                    "reference_pose": "tpose",
                                    "posed_pose": pose,
                                    "query_domain": domain_key,
                                    "query_vertex_ids_global": ids.tolist(),
                                    "pose_rigidity_gate_applied": True,
                                    "interpretation": (
                                        "same-variant posed cap versus its own T-pose cap; "
                                        "registration angle/translation describe motion, residuals describe deformation"
                                    ),
                                }
                            )
                            per_cap[f"{side}_{bone}"] = row
                    rigidity_rows[variant_name] = per_cap
                pose_row["posed_cap_rigidity_against_own_tpose"] = rigidity_rows
                _write_json(output / "progress.json", report)
    finally:
        # Keep the operational/audit status explicit.  No value here is an
        # anatomical pass or a publication decision.
        report["status"] = "complete" if not report["failures"] else "evaluation_error"
        report["audit_status"] = report["status"]
        report["operational_evaluation_completed"] = bool(not report["failures"])
        report["cell_count"] = sum(len(rows) for rows in report["cells"].values())
        report["elapsed_seconds"] = float(time.perf_counter() - started)
        if report["status"] == "complete":
            for subject, rows in report["cells"].items():
                for pose, cell in rows.items():
                    if cell.get("status") != "complete":
                        continue
                    candidate_variant = cell["variants"]["candidate"]
                    for cap_name, cap in cell.get('posed_cap_rigidity_against_own_tpose', {}).get('candidate', {}).items():
                        if not cap.get('shape_preservation_passed'):
                            report['candidate_flags'].append(dict(
                                subject=subject, pose=pose, kind='posed_cap_deformation', cap=cap_name,
                                rigid_procrustes_max_m=cap['rigid_procrustes_max_m'],
                                gate='posed_cap_0.1mm'))
                    full_arm = candidate_variant["full_arm_skin_boundary"]
                    if full_arm["max_outside_m"] > OUTSIDE_REPORT_THRESHOLD_M:
                        report["candidate_flags"].append(
                            {
                                "subject": subject,
                                "pose": pose,
                                "kind": "full_arm_skin_outside",
                                "max_outside_m": full_arm["max_outside_m"],
                                "outside_vertex_count_gt_1mm": full_arm["outside_vertex_count_gt_1mm"],
                            }
                        )
                    for side, queries in candidate_variant["validation_domain_signed_penetration"].items():
                        for name, query in queries.items():
                            signed = query.get("signed_distance_m")
                            if signed and float(signed.get("min_m", 0.0)) < -TRIANGLE_DEPTH_TOLERANCE_M:
                                report["candidate_flags"].append(
                                    {
                                        "subject": subject,
                                        "pose": pose,
                                        "kind": "validation_domain_penetration",
                                        "query": name,
                                        "min_signed_m": float(signed["min_m"]),
                                    }
                                )
                    for pair_name, pair in candidate_variant["triangle_pair_checks"].items():
                        if int(pair.get("triangle_pair_count", 0)) > 0 or not bool(pair.get("passed")):
                            report["candidate_flags"].append(
                                {
                                    "subject": subject,
                                    "pose": pose,
                                    "kind": "triangle_pair_contact_or_depth",
                                    "pair": pair_name,
                                    "triangle_pair_count": int(pair.get("triangle_pair_count", 0)),
                                    "reason": pair.get("reason"),
                                }
                            )
            if template_reference_gate_applied:
                for subject, rows in report["tpose_cap_shape_vs_frozen_template"].items():
                    for cap_name, row in rows.items():
                        if not bool(row.get("total_authored_142_gate_demonstrated")):
                            report["candidate_flags"].append(
                                {
                                    "subject": subject,
                                    "pose": "tpose",
                                    "kind": "frozen_template_cap_shape_failure",
                                    "cap": cap_name,
                                    "gate": "total_authored_142_0.1mm",
                                }
                            )
        report["candidate_flag_count"] = len(report["candidate_flags"])
        report["anatomical_passed"] = False
        report["publishable"] = False
        _write_json(output / "report.json", report)
        _write_json(output / "progress.json", report)
    print(
        f"consistent_arm_v14_audit cells={report.get('cell_count', 0)} "
        f"failures={len(report.get('failures', []))} candidate_flags={report.get('candidate_flag_count', 0)} "
        f"anatomical_passed=false output={output}",
        flush=True,
    )
    return 1 if report.get("failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
