"""Audit physical collar surfaces and nearby vessel samples after J13 pivoting.

This command is an evidence-only comparison of the already exported
``before_vertices`` and ``candidate_vertices`` cells from the V14 collar
ablation.  It never changes a mesh, fits a response, selects a new pose, or
uses the J13 point as an acceptance criterion.

The three requested bone pairs are sent through
``surface_validation_v14.audit_bone_pair`` for complete triangle contact
queries.  A vessel sample is frozen once, from the subject's T-pose vessel
vertices within 120 mm of T-pose SMPL-X J13, and those same global material
vertices are queried in every pose and in both geometry variants.  Negative
signed distance means a sample is inside the target bone.  The reported
penetration maximum is a sampled lower bound; it is not a proof about all
points of a vessel or all points of a surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import igl
import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.consistent_runtime_v14 import (
    load_compiled_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.surface_validation_v14 import (
    audit_bone_pair,
    closed_mesh_quality,
)


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT = ROOT / "outputs/anatomy_retarget/v14_collar_surface_audit_20260908_001"
DEFAULT_PIVOT_ROOT = ROOT / "outputs/anatomy_retarget"
DEFAULT_SUBJECTS = ("213328", "213712")
POSE_LABELS = ("tpose", "own", "swap", "oldsit")
TRIANGLE_PAIRS = (
    ("Clavicle_L", "Sternum"),
    ("Clavicle_L", "Scapula_L"),
    ("Humerus_L", "Scapula_L"),
)
TARGET_MESHES = ("Clavicle_L", "Sternum", "Scapula_L", "Humerus_L")
VESSEL_RADIUS_M = 0.120
PENETRATION_THRESHOLD_M = 0.0005


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(value)).tobytes()
    ).hexdigest()


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
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _pose_paths(root: Path, subject: str) -> dict[str, Path]:
    other = "213712" if subject == "213328" else "213328"
    return {
        "tpose": root / f"subject_{subject}_tpose.npz",
        "own": root / f"subject_{subject}_pose_{subject}.npz",
        "swap": root / f"subject_{subject}_pose_{other}.npz",
        "oldsit": root / f"subject_{subject}_heldout_sitting.npz",
    }


def _load_cell(path: Path) -> dict[str, Any]:
    required = {
        "before_vertices",
        "candidate_vertices",
        "faces",
        "vertex_tissue",
        "smplx_joints",
        "pose",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path}: missing required fields {missing}")
        before = np.asarray(data["before_vertices"], dtype=np.float64)
        candidate = np.asarray(data["candidate_vertices"], dtype=np.float64)
        faces = np.asarray(data["faces"], dtype=np.int64)
        tissue = np.asarray(data["vertex_tissue"], dtype=np.int64).reshape(-1)
        smplx_joints = np.asarray(data["smplx_joints"], dtype=np.float64)
        pose = np.asarray(data["pose"], dtype=np.float32).reshape(-1)
    if (
        before.ndim != 2
        or before.shape[1] != 3
        or candidate.shape != before.shape
        or len(before) == 0
    ):
        raise ValueError(f"{path}: before/candidate vertices must be matching [N,3]")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or len(faces) == 0
        or np.any(faces < 0)
        or np.any(faces >= len(before))
    ):
        raise ValueError(f"{path}: faces must be non-empty triangles in vertex range")
    if tissue.shape != (len(before),):
        raise ValueError(f"{path}: vertex_tissue must have one label per vertex")
    if smplx_joints.shape != (55, 3):
        raise ValueError(f"{path}: smplx_joints must have shape [55,3]")
    if pose.size != 165:
        raise ValueError(f"{path}: pose must contain 55*3 values")
    for name, values in (
        ("before_vertices", before),
        ("candidate_vertices", candidate),
        ("smplx_joints", smplx_joints),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: {name} contains non-finite values")
    return {
        "path": path.resolve(),
        "sha256": _sha256(path),
        "before_vertices": before,
        "candidate_vertices": candidate,
        "faces": faces,
        "vertex_tissue": tissue,
        "smplx_joints": smplx_joints,
        "pose": pose.reshape(55, 3),
    }


def _mesh_layout(asset: Any, faces: np.ndarray) -> dict[str, dict[str, Any]]:
    names = [str(value) for value in asset.source_mesh_names]
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    if len(names) != len(ranges):
        raise ValueError("source mesh names and vertex ranges have different lengths")
    layout: dict[str, dict[str, Any]] = {}
    global_faces = np.asarray(faces, dtype=np.int64)
    for name in TARGET_MESHES:
        if name not in names:
            raise ValueError(f"source asset is missing required mesh {name}")
        mesh_index = names.index(name)
        start, stop = (int(value) for value in ranges[mesh_index])
        if stop <= start or start < 0 or stop > len(asset.vertices_rest):
            raise ValueError(f"invalid vertex range for {name}: {(start, stop)}")
        inside = np.all(
            (global_faces >= start) & (global_faces < stop), axis=1
        )
        partial = np.any(
            (global_faces >= start) & (global_faces < stop), axis=1
        ) & ~inside
        if np.any(partial):
            raise ValueError(f"{name} has faces crossing its source mesh range")
        face_global_ids = np.flatnonzero(inside).astype(np.int64)
        faces_local = global_faces[face_global_ids] - start
        if len(face_global_ids) == 0:
            raise ValueError(f"{name} has no complete triangles")
        layout[name] = {
            "mesh_index": mesh_index,
            "vertex_start": start,
            "vertex_stop": stop,
            "faces_local": faces_local.astype(np.int32),
            "face_global_ids": face_global_ids,
        }
    return layout


def _freeze_vessel_selection(
    asset: Any,
    tpose: Mapping[str, Any],
    *,
    radius_m: float,
) -> dict[str, Any]:
    names = [str(value) for value in asset.source_mesh_names]
    tissues = [str(value).strip().lower() for value in (asset.source_tissues or ())]
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    materials = [
        str(value)
        for value in (asset.source_mesh_material_groups or ["" for _ in names])
    ]
    if len(tissues) != len(names) or len(materials) != len(names):
        raise ValueError("source mesh tissue/material metadata is incomplete")
    vessel_meshes = [
        (index, names[index], tuple(int(value) for value in ranges[index]), materials[index])
        for index, tissue in enumerate(tissues)
        if tissue == "vessel"
    ]
    if not vessel_meshes:
        raise ValueError("source asset has no vessel meshes")
    vessel_ids = np.concatenate(
        [np.arange(start, stop, dtype=np.int64) for _, _, (start, stop), _ in vessel_meshes]
    )
    tissue_labels = np.asarray(tpose["vertex_tissue"], dtype=np.int64)
    if np.any(vessel_ids < 0) or np.any(vessel_ids >= len(tissue_labels)):
        raise ValueError("vessel mesh range exceeds exported vertex labels")
    vessel_label_values = np.unique(tissue_labels[vessel_ids])
    if len(vessel_label_values) != 1:
        raise ValueError(
            "vessel mesh ranges do not have one frozen material/tissue label: "
            f"{vessel_label_values.tolist()}"
        )
    joint13 = np.asarray(tpose["smplx_joints"], dtype=np.float64)[13]
    distances = np.linalg.norm(tpose["before_vertices"][vessel_ids] - joint13, axis=1)
    selected = vessel_ids[distances <= float(radius_m)]
    if len(selected) == 0:
        raise ValueError("T-pose vessel selection within 120 mm of J13 is empty")
    selected_distances = distances[distances <= float(radius_m)]
    selected_mesh_counts = {
        name: int(np.count_nonzero((selected >= start) & (selected < stop)))
        for _, name, (start, stop), _ in vessel_meshes
    }
    selected_material_labels = np.asarray(tissue_labels[selected], dtype=np.int64)
    return {
        "radius_m": float(radius_m),
        "joint_id": 13,
        "joint13_m": joint13.tolist(),
        "vessel_meshes": [
            {
                "mesh_index": int(index),
                "name": name,
                "vertex_range": [int(start), int(stop)],
                "material_group": material,
            }
            for index, name, (start, stop), material in vessel_meshes
        ],
        "frozen_material_label_values": vessel_label_values.tolist(),
        "selected_count": int(len(selected)),
        "selected_count_by_mesh": selected_mesh_counts,
        "selected_id_sha256": _array_sha256(selected),
        "selected_label_sha256": _array_sha256(selected_material_labels),
        "selected_distance_min_mm": float(np.min(selected_distances) * 1000.0),
        "selected_distance_max_mm": float(np.max(selected_distances) * 1000.0),
        "selected_vertex_ids": selected,
        "selected_material_labels": selected_material_labels,
    }


def _geometry_delta(before: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    delta = np.linalg.norm(
        np.asarray(candidate, dtype=np.float64) - np.asarray(before, dtype=np.float64),
        axis=1,
    )
    return {
        "vertex_count": int(len(delta)),
        "max_mm": float(np.max(delta) * 1000.0),
        "p95_mm": float(np.quantile(delta, 0.95) * 1000.0),
        "mean_mm": float(np.mean(delta) * 1000.0),
        "changed_gt_0.001mm_count": int(np.count_nonzero(delta > 1.0e-6)),
        "delta_sha256": _array_sha256(
            np.asarray(candidate, dtype=np.float64) - np.asarray(before, dtype=np.float64)
        ),
    }


def _compact_triangle_audit(
    vertices: np.ndarray,
    layout: Mapping[str, Mapping[str, Any]],
    first: str,
    second: str,
) -> dict[str, Any]:
    first_info = layout[first]
    second_info = layout[second]
    first_start, first_stop = int(first_info["vertex_start"]), int(first_info["vertex_stop"])
    second_start, second_stop = int(second_info["vertex_start"]), int(second_info["vertex_stop"])
    result = audit_bone_pair(
        np.asarray(vertices[first_start:first_stop], dtype=np.float64),
        first_info["faces_local"],
        np.asarray(vertices[second_start:second_stop], dtype=np.float64),
        second_info["faces_local"],
        depth_tolerance_m=PENETRATION_THRESHOLD_M,
    )
    triangle_pairs = np.asarray(
        result.get("triangle_pairs", []), dtype=np.int64
    ).reshape(-1, 2)
    return {
        "pair": f"{first}--{second}",
        "first_mesh": first,
        "second_mesh": second,
        "first_face_count": int(len(first_info["faces_local"])),
        "second_face_count": int(len(second_info["faces_local"])),
        "triangle_pair_count": int(result.get("triangle_pair_count", len(triangle_pairs))),
        "triangle_pairs_sample": triangle_pairs[:20].tolist(),
        "mesh_quality": result.get("mesh_quality"),
        "signed_samples": result.get("signed_samples"),
        "depth_tolerance_m": float(result.get("depth_tolerance_m", PENETRATION_THRESHOLD_M)),
        "max_depth_is_sampled_lower_bound": bool(
            result.get("max_depth_is_sampled_lower_bound", True)
        ),
        "passed": bool(result.get("passed", False)),
        "reason": str(result.get("reason", "unknown")),
        "method": str(result.get("method", "unknown")),
        "vtk_version": str(result.get("vtk_version", "unknown")),
        "complete_triangle_pair_query": True,
    }


def _signed_depth_summary(
    points: np.ndarray,
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    quality = dict(quality)
    if not bool(quality.get("signed_distance_valid", False)):
        return {
            "sample_count": int(len(points)),
            "signed_distance_valid": False,
            "reason": "target_mesh_not_closed_winding_consistent_positive_volume",
            "max_penetration_lower_bound_mm": None,
            "penetration_gt_0.5mm_count": None,
            "signed_depth_is_sampled_lower_bound": True,
            "target_mesh_quality": quality,
        }
    signed = np.asarray(
        igl.signed_distance(
            np.ascontiguousarray(np.asarray(points, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(target_vertices, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(target_faces, dtype=np.int64)),
        )[0],
        dtype=np.float64,
    ).reshape(-1)
    if len(signed) != len(points) or not np.all(np.isfinite(signed)):
        raise ValueError("libigl returned invalid signed vessel depth values")
    penetration = np.maximum(-signed, 0.0)
    negative = signed < 0.0
    return {
        "sample_count": int(len(signed)),
        "signed_distance_valid": True,
        "signed_convention": "negative means the vessel sample is inside the target bone",
        "min_signed_mm": float(np.min(signed) * 1000.0),
        "p05_signed_mm": float(np.quantile(signed, 0.05) * 1000.0),
        "median_signed_mm": float(np.median(signed) * 1000.0),
        "p95_signed_mm": float(np.quantile(signed, 0.95) * 1000.0),
        "max_signed_mm": float(np.max(signed) * 1000.0),
        "negative_sample_count": int(np.count_nonzero(negative)),
        "penetration_gt_0.5mm_count": int(
            np.count_nonzero(signed < -PENETRATION_THRESHOLD_M)
        ),
        "max_penetration_lower_bound_mm": float(np.max(penetration) * 1000.0),
        "minimum_absolute_distance_mm": float(np.min(np.abs(signed)) * 1000.0),
        "signed_depth_is_sampled_lower_bound": True,
        "target_mesh_quality": quality,
    }


def _target_depths(
    vertices: np.ndarray,
    selected_ids: np.ndarray,
    layout: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    points = np.asarray(vertices[selected_ids], dtype=np.float64)
    for name in ("Clavicle_L", "Sternum", "Scapula_L"):
        info = layout[name]
        start, stop = int(info["vertex_start"]), int(info["vertex_stop"])
        target_vertices = np.asarray(vertices[start:stop], dtype=np.float64)
        target_faces = np.asarray(info["faces_local"], dtype=np.int64)
        quality = closed_mesh_quality(target_vertices, target_faces)
        result[name] = _signed_depth_summary(
            points, target_vertices, target_faces, quality
        )
    return result


def _validate_frozen_inputs(
    cells: Mapping[str, Mapping[str, Any]],
    *,
    tpose_faces: np.ndarray,
    tpose_labels: np.ndarray,
    selected_ids: np.ndarray,
) -> dict[str, Any]:
    pose_checks: dict[str, Any] = {}
    for label, cell in cells.items():
        if not np.array_equal(cell["faces"], tpose_faces):
            raise ValueError(f"{label}: faces differ from frozen T-pose faces")
        if not np.array_equal(cell["vertex_tissue"], tpose_labels):
            raise ValueError(f"{label}: material/tissue labels differ from frozen T-pose labels")
        if np.any(selected_ids >= len(cell["vertex_tissue"])):
            raise ValueError(f"{label}: frozen vessel IDs exceed pose vertex count")
        pose_checks[label] = {
            "faces_bitexact_to_tpose": True,
            "material_labels_bitexact_to_tpose": True,
            "selected_material_label_sha256": _array_sha256(
                cell["vertex_tissue"][selected_ids]
            ),
        }
    return pose_checks


def _subject_audit(subject: str, root: Path) -> dict[str, Any]:
    pivot_root = root / f"v14_collar_pivot_{subject}_20260908_001"
    compiled_root = pivot_root / "compiled"
    if not compiled_root.is_dir():
        raise FileNotFoundError(f"missing compiled pivot package: {compiled_root}")
    compiled = load_compiled_subject(compiled_root)
    asset = compiled.source_asset
    paths = _pose_paths(pivot_root, subject)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{subject}: missing pose cells {missing}")
    cells = {label: _load_cell(path) for label, path in paths.items()}
    tpose = cells["tpose"]
    if not np.array_equal(np.asarray(asset.faces, dtype=np.int64), tpose["faces"]):
        raise ValueError(f"{subject}: source asset faces differ from exported pose faces")
    layout = _mesh_layout(asset, tpose["faces"])
    selection = _freeze_vessel_selection(asset, tpose, radius_m=VESSEL_RADIUS_M)
    selected_ids = np.asarray(selection.pop("selected_vertex_ids"), dtype=np.int64)
    selected_labels = np.asarray(selection.pop("selected_material_labels"), dtype=np.int64)
    pose_checks = _validate_frozen_inputs(
        cells,
        tpose_faces=tpose["faces"],
        tpose_labels=tpose["vertex_tissue"],
        selected_ids=selected_ids,
    )
    poses: dict[str, Any] = {}
    for label in POSE_LABELS:
        cell = cells[label]
        before = cell["before_vertices"]
        candidate = cell["candidate_vertices"]
        triangle_audits = {
            "before": {
                f"{first}--{second}": _compact_triangle_audit(
                    before, layout, first, second
                )
                for first, second in TRIANGLE_PAIRS
            },
            "candidate": {
                f"{first}--{second}": _compact_triangle_audit(
                    candidate, layout, first, second
                )
                for first, second in TRIANGLE_PAIRS
            },
        }
        vessel_depths = {
            "before": _target_depths(before, selected_ids, layout),
            "candidate": _target_depths(candidate, selected_ids, layout),
        }
        poses[label] = {
            "input_npz": str(cell["path"]),
            "input_sha256": cell["sha256"],
            "pose_zero": bool(np.allclose(cell["pose"], 0.0, atol=1.0e-7, rtol=0.0)),
            "geometry_delta_before_to_candidate": _geometry_delta(before, candidate),
            "triangle_surfaces": triangle_audits,
            "frozen_vessel_signed_depths": vessel_depths,
        }
    return {
        "subject": subject,
        "pivot_package": str(pivot_root),
        "compiled_manifest": str(compiled_root / "manifest.json"),
        "compiled_manifest_sha256": _sha256(compiled_root / "manifest.json"),
        "mesh_layout": {
            name: {
                "mesh_index": int(info["mesh_index"]),
                "vertex_range": [int(info["vertex_start"]), int(info["vertex_stop"])],
                "face_count": int(len(info["faces_local"])),
                "faces_local_sha256": _array_sha256(info["faces_local"]),
            }
            for name, info in layout.items()
        },
        "frozen_vessel_selection": {
            **selection,
            "selected_count": int(len(selected_ids)),
            "selected_id_sha256": _array_sha256(selected_ids),
            "selected_label_sha256": _array_sha256(selected_labels),
        },
        "pose_material_identity_checks": pose_checks,
        "poses": poses,
    }


def _conclusion(subjects: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    new_contacts: list[dict[str, Any]] = []
    reduced_contacts: list[dict[str, Any]] = []
    new_penetration: list[dict[str, Any]] = []
    reduced_penetration: list[dict[str, Any]] = []
    for subject, subject_result in subjects.items():
        for pose, pose_result in subject_result["poses"].items():
            before_pairs = pose_result["triangle_surfaces"]["before"]
            candidate_pairs = pose_result["triangle_surfaces"]["candidate"]
            for pair in TRIANGLE_PAIRS:
                key = f"{pair[0]}--{pair[1]}"
                before_count = int(before_pairs[key]["triangle_pair_count"])
                candidate_count = int(candidate_pairs[key]["triangle_pair_count"])
                row = {
                    "subject": subject,
                    "pose": pose,
                    "pair": key,
                    "before": before_count,
                    "candidate": candidate_count,
                }
                if candidate_count > before_count:
                    new_contacts.append(row)
                elif candidate_count < before_count:
                    reduced_contacts.append(row)
            before_depths = pose_result["frozen_vessel_signed_depths"]["before"]
            candidate_depths = pose_result["frozen_vessel_signed_depths"]["candidate"]
            for target in ("Clavicle_L", "Sternum", "Scapula_L"):
                before_count = before_depths[target]["penetration_gt_0.5mm_count"]
                candidate_count = candidate_depths[target]["penetration_gt_0.5mm_count"]
                if before_count is None or candidate_count is None:
                    continue
                row = {
                    "subject": subject,
                    "pose": pose,
                    "target": target,
                    "before": int(before_count),
                    "candidate": int(candidate_count),
                }
                if candidate_count > before_count:
                    new_penetration.append(row)
                elif candidate_count < before_count:
                    reduced_penetration.append(row)
    if new_penetration:
        penetration_text = (
            "发现候选相对 before 新增 >0.5 mm 的血管点样本穿入骨面："
            + json.dumps(new_penetration, ensure_ascii=False, sort_keys=True)
        )
    else:
        penetration_text = "未发现候选相对 before 新增 >0.5 mm 的血管点样本穿入 Clavicle_L/Sternum/Scapula_L。"
    if new_contacts:
        contact_text = (
            "部分完整三角面接触计数增加："
            + json.dumps(new_contacts, ensure_ascii=False, sort_keys=True)
            + "；这表示接触候选变化，不能单独解释为新穿透。"
        )
    else:
        contact_text = "未发现三组完整骨面之间新增的三角面接触计数。"
    short_parts = [
        "未能由本审计证明新的物理脱离。",
        (
            "观察到固定血管样本的 >0.5 mm 穿入计数新增 "
            f"{len(new_penetration)} 个单元"
            if new_penetration
            else "未观察到固定血管样本的 >0.5 mm 穿入计数新增。"
        ),
        (
            "另有完整骨面三角接触计数增加，需结合 Genesis 图审判断是否为真实穿透"
            if new_contacts
            else "未观察到完整骨面三角接触计数增加。"
        ),
    ]
    if new_penetration:
        short_parts[1] += "。"
    return {
        "new_triangle_contact_count_observed": bool(new_contacts),
        "new_sampled_vessel_penetration_gt_0_5mm_observed": bool(new_penetration),
        "new_physical_detachment_proven": False,
        "new_triangle_contacts": new_contacts,
        "reduced_triangle_contacts": reduced_contacts,
        "new_sampled_vessel_penetration": new_penetration,
        "reduced_sampled_vessel_penetration": reduced_penetration,
        "conclusion_zh_short": "".join(short_parts),
        "conclusion_zh": (
            penetration_text
            + " "
            + contact_text
            + " 未用 J13 点或点样本把组织脱离判为通过；有符号深度和最大穿入均是固定样本的下界，"
            "是否存在连续表面脱离仍需结合 Genesis 多视角图审。"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_PIVOT_ROOT,
        help="directory containing v14_collar_pivot_<subject>_20260908_001",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        choices=DEFAULT_SUBJECTS,
        default=list(DEFAULT_SUBJECTS),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing report directory",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace: {output}")
    if output.exists():
        for child in output.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    output.mkdir(parents=True, exist_ok=True)
    subjects = {
        subject: _subject_audit(subject, args.root.resolve())
        for subject in args.subjects
    }
    report = {
        "schema": "anatomy_v14_collar_surface_audit_20260908",
        "analysis_only": True,
        "publishable": False,
        "optimization_used": False,
        "mesh_edit_used": False,
        "j13_point_acceptance_used": False,
        "triangle_validation_backend": "surface_validation_v14.audit_bone_pair",
        "triangle_pairs": [f"{first}--{second}" for first, second in TRIANGLE_PAIRS],
        "target_meshes_for_vessel_depth": list(
            ("Clavicle_L", "Sternum", "Scapula_L")
        ),
        "pose_labels": list(POSE_LABELS),
        "frozen_vessel_selection_policy": {
            "source": "each subject T-pose before_vertices",
            "material_scope": "all source meshes with source_tissues == vessel",
            "distance_target": "T-pose smplx_joints[13]",
            "radius_m": VESSEL_RADIUS_M,
            "same_global_material_vertex_ids_for_all_poses": True,
        },
        "signed_depth_policy": {
            "negative_means_inside_target_bone": True,
            "penetration_threshold_m": PENETRATION_THRESHOLD_M,
            "max_penetration_is_sampled_lower_bound": True,
            "closed_winding_required": True,
        },
        "subjects": subjects,
    }
    report["conclusion"] = _conclusion(subjects)
    _write_json(output / "report.json", report)
    readme = (
        "# V14 collar pivot surface audit\n\n"
        "此目录是只读证据审计：完整三角面对照使用 "
        "`surface_validation_v14.audit_bone_pair`；血管深度使用固定 T-pose J13 半径 120 mm "
        "顶点集合。报告中的最大穿入是采样下界，不能替代表面连续性或 Genesis 图审。\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(report["conclusion"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
