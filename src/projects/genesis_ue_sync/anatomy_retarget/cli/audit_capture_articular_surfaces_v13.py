"""Audit local articular bone geometry on the six frozen capture cells.

This is an evidence-only diagnostic.  It queries vertices from the frozen
calibration ``validation`` domains against the complete triangle mesh of the
opposing source bone.  It does not fit pivots, move vertices, optimize a pose,
or assert that a sampled point-to-triangle distance proves contact or global
non-intersection.

The exporter writes both the V12e source geometry and the V13 material
candidate.  Bone ranges are normally byte-identical between the two.  When
that is true, this CLI records the equality and computes one shared query,
labelled as the source baseline / candidate-shared bone geometry.  If a future
input changes a queried bone, source and candidate are measured separately.
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

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    _calibration_content_digest,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.blender_link_oracle_v7 import (
    EXPECTED_OPERATOR_RUNTIME_DIGEST,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import load_source_operator


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_INPUT = ROOT / "outputs/anatomy_retarget/v13_capture_joint_geometry_20260908_001"
DEFAULT_OUTPUT = ROOT / "outputs/anatomy_retarget/v13_capture_articular_surfaces_20260908_001"
DEFAULT_OPERATOR = ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
DEFAULT_CALIBRATION = ROOT / (
    "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006/"
    "anatomical_calibration_v1"
)

SUBJECTS = ("213328", "213712")
POSES = ("tpose", "pose_213328", "pose_213712")
PARTITION = "validation"
NEGATIVE_SIGNED_THRESHOLD_M = 1.0e-4
VOLUME_EPSILON_M3 = 1.0e-12

# Every target mesh is a complete authored bone object.  The query domains are
# fixed IDs from anatomical_calibration_v1, never newly selected from a pose.
MESH_NAMES = (
    "Femur_L",
    "Femur_R",
    "Ilium_L",
    "Ilium_R",
    "Tibia_L",
    "Tibia_R",
    "Humerus_L",
    "Humerus_R",
    "Radius_L",
    "Radius_R",
    "Ulna_L",
    "Ulna_R",
)


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


def _mesh_face_subset(
    faces: np.ndarray,
    *,
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (global face IDs, local faces) for one contiguous mesh range."""

    triangles = np.asarray(faces, dtype=np.int64)
    selected = np.all((triangles >= int(start)) & (triangles < int(stop)), axis=1)
    face_ids = np.flatnonzero(selected).astype(np.int64)
    local = triangles[selected] - int(start)
    return face_ids, local.astype(np.int32)


def _edge_winding_consistent(faces: np.ndarray) -> bool:
    """Check that each closed-manifold edge occurs with opposite directions."""

    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(triangles):
        return False
    directed = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        ),
        axis=0,
    )
    undirected = np.sort(directed, axis=1)
    order = np.lexsort((undirected[:, 1], undirected[:, 0]))
    sorted_edges = undirected[order]
    sorted_directed = directed[order]
    consistent = True
    start = 0
    while start < len(sorted_edges):
        stop = start + 1
        while stop < len(sorted_edges) and np.array_equal(sorted_edges[stop], sorted_edges[start]):
            stop += 1
        if stop - start != 2:
            return False
        first = sorted_directed[start]
        second = sorted_directed[start + 1]
        if not (int(first[0]) == int(second[1]) and int(first[1]) == int(second[0])):
            consistent = False
            break
        start = stop
    return consistent


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    return float(
        np.sum(
            np.einsum(
                "ij,ij->i",
                triangles[:, 0],
                np.cross(triangles[:, 1], triangles[:, 2]),
            )
        )
        / 6.0
    )


def _mesh_quality(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise ValueError("bone mesh faces must be non-empty [F,3]")
    edges = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        ),
        axis=0,
    )
    undirected = np.sort(edges, axis=1)
    _unique, counts = np.unique(undirected, axis=0, return_counts=True)
    boundary_edges = int(np.count_nonzero(counts == 1))
    nonmanifold_edges = int(np.count_nonzero(counts != 2))
    watertight = bool(boundary_edges == 0 and nonmanifold_edges == 0)
    winding = bool(_edge_winding_consistent(triangles)) if watertight else False
    volume = _signed_volume(points, triangles)
    signed_available = bool(
        watertight and winding and np.isfinite(volume) and abs(volume) > VOLUME_EPSILON_M3
    )
    return {
        "vertex_count": int(len(points)),
        "face_count": int(len(triangles)),
        "boundary_edge_count": boundary_edges,
        "nonmanifold_edge_count": nonmanifold_edges,
        "watertight": watertight,
        "winding_consistent": winding,
        "signed_volume_m3": volume,
        "volume_abs_m3": abs(volume),
        "signed_available": signed_available,
        "signed_policy": (
            "winding_number_abs_ge_0.5; negative means query point inside target bone"
            if signed_available
            else "unsigned_only_nonclosed_or_degenerate_target_mesh"
        ),
    }


def _distance_summary(values: np.ndarray) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(data) or not np.all(np.isfinite(data)):
        raise ValueError("distance array is empty or non-finite")
    return {
        "closest_m": float(np.min(data)),
        "p05_m": float(np.quantile(data, 0.05)),
        "median_m": float(np.median(data)),
        "p95_m": float(np.quantile(data, 0.95)),
        "max_m": float(np.max(data)),
    }


def _signed_distance_summary(values: np.ndarray) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(data) or not np.all(np.isfinite(data)):
        raise ValueError("signed distance array is empty or non-finite")
    return {
        "min_m": float(np.min(data)),
        "p05_m": float(np.quantile(data, 0.05)),
        "median_m": float(np.median(data)),
        "p95_m": float(np.quantile(data, 0.95)),
        "max_m": float(np.max(data)),
    }


def _query_distances(
    points: np.ndarray,
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
    target_face_global_ids: np.ndarray,
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    """Query exact point-to-triangle distances and optional winding signs."""

    import igl

    query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    vertices = np.asarray(target_vertices, dtype=np.float64)
    faces = np.asarray(target_faces, dtype=np.int32)
    squared, local_faces, closest = igl.point_mesh_squared_distance(query, vertices, faces)
    squared = np.asarray(squared, dtype=np.float64).reshape(-1)
    local_faces = np.asarray(local_faces, dtype=np.int64).reshape(-1)
    closest = np.asarray(closest, dtype=np.float64).reshape(-1, 3)
    if (
        len(squared) != len(query)
        or len(local_faces) != len(query)
        or len(closest) != len(query)
        or np.any(local_faces < 0)
        or np.any(local_faces >= len(faces))
        or not np.all(np.isfinite(squared))
        or not np.all(np.isfinite(closest))
    ):
        raise ValueError("libigl returned invalid point-to-triangle distances")
    unsigned = np.sqrt(np.maximum(0.0, squared))
    signed: np.ndarray | None = None
    if bool(quality.get("signed_available", False)):
        winding = np.asarray(igl.winding_number(vertices, faces, query), dtype=np.float64).reshape(-1)
        if len(winding) != len(query) or not np.all(np.isfinite(winding)):
            raise ValueError("libigl returned invalid winding numbers")
        inside = np.abs(winding) >= 0.5
        signed = np.where(inside, -unsigned, unsigned)
    max_index = int(np.argmax(unsigned))
    result: dict[str, Any] = {
        "query_vertex_count": int(len(query)),
        "unsigned_distance_m": _distance_summary(unsigned),
        "max_unsigned_query_index": max_index,
        "max_unsigned_closest_face_local": int(local_faces[max_index]),
        "max_unsigned_closest_face_global": int(target_face_global_ids[local_faces[max_index]]),
        "max_unsigned_query_point_m": query[max_index].tolist(),
        "max_unsigned_closest_point_m": closest[max_index].tolist(),
        "negative_signed_gt_0.1mm_count": None,
        "negative_signed_gt_0.1mm_fraction": None,
        "signed_distance_m": None,
    }
    if signed is not None:
        negative = signed < -NEGATIVE_SIGNED_THRESHOLD_M
        result["signed_distance_m"] = _signed_distance_summary(signed)
        result["negative_signed_gt_0.1mm_count"] = int(np.count_nonzero(negative))
        result["negative_signed_gt_0.1mm_fraction"] = float(np.mean(negative))
    return result


def _query_specs() -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for side, suffix in (("left", "L"), ("right", "R")):
        specs.append(
            {
                "name": f"{side}_hip_head_to_ilium",
                "joint": f"{side}_hip",
                "query_domain": f"{side}/femoral_head",
                "target_mesh": f"Ilium_{suffix}",
                "direction": "femoral_head_to_ilium",
                "primary": True,
            }
        )
        specs.append(
            {
                "name": f"{side}_hip_acetabulum_to_femur",
                "joint": f"{side}_hip",
                "query_domain": f"{side}/acetabulum",
                "target_mesh": f"Femur_{suffix}",
                "direction": "acetabulum_to_femur_reverse",
                "primary": False,
            }
        )
        for compartment in ("medial", "lateral"):
            specs.append(
                {
                    "name": f"{side}_knee_condyle_{compartment}_to_tibia",
                    "joint": f"{side}_knee",
                    "query_domain": f"{side}/femoral_condyle_{compartment}",
                    "target_mesh": f"Tibia_{suffix}",
                    "direction": "femoral_condyle_to_tibia",
                    "primary": True,
                }
            )
            specs.append(
                {
                    "name": f"{side}_knee_plateau_{compartment}_to_femur",
                    "joint": f"{side}_knee",
                    "query_domain": f"{side}/tibial_plateau_{compartment}",
                    "target_mesh": f"Femur_{suffix}",
                    "direction": "tibial_plateau_to_femur_reverse",
                    "primary": False,
                }
            )
        specs.extend(
            (
                {
                    "name": f"{side}_elbow_humerus_to_radius",
                    "joint": f"{side}_elbow",
                    "query_domain": f"elbow/{side}/humerus",
                    "target_mesh": f"Radius_{suffix}",
                    "direction": "humerus_to_radius",
                    "primary": True,
                },
                {
                    "name": f"{side}_elbow_humerus_to_ulna",
                    "joint": f"{side}_elbow",
                    "query_domain": f"elbow/{side}/humerus",
                    "target_mesh": f"Ulna_{suffix}",
                    "direction": "humerus_to_ulna",
                    "primary": True,
                },
                {
                    "name": f"{side}_elbow_radius_to_humerus",
                    "joint": f"{side}_elbow",
                    "query_domain": f"elbow/{side}/radius",
                    "target_mesh": f"Humerus_{suffix}",
                    "direction": "radius_to_humerus_reverse",
                    "primary": False,
                },
                {
                    "name": f"{side}_elbow_ulna_to_humerus",
                    "joint": f"{side}_elbow",
                    "query_domain": f"elbow/{side}/ulna",
                    "target_mesh": f"Humerus_{suffix}",
                    "direction": "ulna_to_humerus_reverse",
                    "primary": False,
                },
            )
        )
    return tuple(specs)


def _find_input(root: Path, subject: str, pose: str) -> Path:
    candidates = (
        root / f"subject_{subject}_{pose}.npz",
        root / f"subject_{subject}" / "comparisons" / f"{pose}.npz",
        root / "subjects" / f"subject_{subject}" / "comparisons" / f"{pose}.npz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"no geometry NPZ for subject={subject} pose={pose}; searched "
        + ", ".join(str(value) for value in candidates)
    )


def _load_cell(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {"faces", "source_vertices", "candidate_vertices", "pose"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path}: missing geometry fields {missing}")
        faces = np.asarray(data["faces"], dtype=np.int32)
        source = np.asarray(data["source_vertices"], dtype=np.float64)
        candidate = np.asarray(data["candidate_vertices"], dtype=np.float64)
        pose = np.asarray(data["pose"], dtype=np.float32)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"{path}: faces must be [F,3]")
        if source.ndim != 2 or source.shape[1] != 3 or candidate.shape != source.shape:
            raise ValueError(f"{path}: source/candidate vertices have incompatible shapes")
        if np.any(faces < 0) or np.any(faces >= len(source)):
            raise ValueError(f"{path}: faces reference invalid vertices")
        if pose.size != 165:
            raise ValueError(f"{path}: pose must contain 55*3 values")
        if not (np.all(np.isfinite(source)) and np.all(np.isfinite(candidate))):
            raise ValueError(f"{path}: geometry contains non-finite vertices")
        metadata = None
        if "metadata_json" in data.files:
            try:
                metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {"parse_error": True}
    return {
        "path": path,
        "sha256": _sha256(path),
        "faces": faces,
        "source_vertices": source,
        "candidate_vertices": candidate,
        "pose": pose.reshape(55, 3),
        "metadata": metadata,
    }


def _mesh_layout(asset: Any, faces: np.ndarray) -> dict[str, dict[str, Any]]:
    names = [str(name) for name in asset.source_mesh_names]
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    result: dict[str, dict[str, Any]] = {}
    for name in MESH_NAMES:
        if name not in names:
            raise ValueError(f"source asset is missing required mesh {name!r}")
        index = names.index(name)
        start, stop = map(int, ranges[index])
        global_faces, local_faces = _mesh_face_subset(faces, start=start, stop=stop)
        if not len(global_faces):
            raise ValueError(f"source mesh {name!r} has no complete local triangles")
        result[name] = {
            "mesh_index": int(index),
            "vertex_start": start,
            "vertex_stop": stop,
            "vertex_count": stop - start,
            "face_global_ids": global_faces,
            "faces_local": local_faces,
            "face_count": int(len(local_faces)),
        }
    return result


def _geometry_equal(
    first: np.ndarray,
    second: np.ndarray,
    layout: Mapping[str, Mapping[str, Any]],
) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for name, info in layout.items():
        start, stop = int(info["vertex_start"]), int(info["vertex_stop"])
        result[name] = bool(np.array_equal(first[start:stop], second[start:stop]))
    return result


def _one_query(
    spec: Mapping[str, Any],
    *,
    vertices: np.ndarray,
    calibration: Any,
    layout: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    domain_key = f"{spec['query_domain']}.{PARTITION}"
    if domain_key not in calibration.domains:
        raise KeyError(f"calibration domain {domain_key!r} is missing")
    domain_ids = np.asarray(calibration.domains[domain_key], dtype=np.int64).reshape(-1)
    if len(domain_ids) < 4 or np.any(domain_ids < 0) or np.any(domain_ids >= len(vertices)):
        raise ValueError(f"calibration domain {domain_key!r} is invalid")
    target_info = layout[str(spec["target_mesh"])]
    start, stop = int(target_info["vertex_start"]), int(target_info["vertex_stop"])
    target_vertices = np.asarray(vertices[start:stop], dtype=np.float64)
    target_faces = np.asarray(target_info["faces_local"], dtype=np.int32)
    quality = _mesh_quality(target_vertices, target_faces)
    distances = _query_distances(
        np.asarray(vertices[domain_ids], dtype=np.float64),
        target_vertices,
        target_faces,
        np.asarray(target_info["face_global_ids"], dtype=np.int64),
        quality,
    )
    distances.update(
        {
            "name": str(spec["name"]),
            "joint": str(spec["joint"]),
            "direction": str(spec["direction"]),
            "primary_direction": bool(spec["primary"]),
            "query_domain": domain_key,
            "query_vertex_ids_global": domain_ids.tolist(),
            "target_mesh": str(spec["target_mesh"]),
            "target_mesh_vertex_range": [start, stop],
            "target_mesh_face_count": int(len(target_faces)),
            "target_mesh_quality": quality,
        }
    )
    return distances


def _audit_cell(
    cell: Mapping[str, Any],
    *,
    subject: str,
    pose: str,
    calibration: Any,
    layout: Mapping[str, Mapping[str, Any]],
    specs: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    source = np.asarray(cell["source_vertices"], dtype=np.float64)
    candidate = np.asarray(cell["candidate_vertices"], dtype=np.float64)
    equality = _geometry_equal(source, candidate, layout)
    all_equal = bool(all(equality.values()))
    variants: dict[str, dict[str, Any]] = {}
    if all_equal:
        variants["source_candidate_shared_bones"] = {
            "geometry_label": "source_baseline_and_candidate_bones_exactly_equal",
            "source_baseline": True,
            "candidate": True,
            "queries": {
                str(spec["name"]): _one_query(
                    spec,
                    vertices=source,
                    calibration=calibration,
                    layout=layout,
                )
                for spec in specs
            },
        }
    else:
        variants["source"] = {
            "geometry_label": "V12e source_vertices",
            "source_baseline": True,
            "candidate": False,
            "queries": {
                str(spec["name"]): _one_query(
                    spec,
                    vertices=source,
                    calibration=calibration,
                    layout=layout,
                )
                for spec in specs
            },
        }
        variants["candidate"] = {
            "geometry_label": "V13 candidate_vertices",
            "source_baseline": False,
            "candidate": True,
            "queries": {
                str(spec["name"]): _one_query(
                    spec,
                    vertices=candidate,
                    calibration=calibration,
                    layout=layout,
                )
                for spec in specs
            },
        }
    quality_by_variant: dict[str, dict[str, Any]] = {}
    for variant_name, variant in variants.items():
        quality_by_variant[variant_name] = {}
        # Report quality for every selected source/candidate bone, not just
        # target meshes, so the reader sees which signed queries are valid.
        points = source if variant_name == "source" else candidate
        for mesh_name, info in layout.items():
            start, stop = int(info["vertex_start"]), int(info["vertex_stop"])
            quality_by_variant[variant_name][mesh_name] = _mesh_quality(
                points[start:stop], info["faces_local"]
            )
        variant["bone_mesh_quality"] = quality_by_variant[variant_name]
    return {
        "status": "complete",
        "subject": subject,
        "pose": pose,
        "input": {"path": str(cell["path"]), "sha256": str(cell["sha256"])},
        "metadata": cell.get("metadata"),
        "partition": PARTITION,
        "negative_signed_threshold_m": NEGATIVE_SIGNED_THRESHOLD_M,
        "bone_mesh_equality_source_vs_candidate": equality,
        "all_queried_bone_meshes_exactly_equal": all_equal,
        "geometry_evaluation_policy": (
            "one shared source-baseline/candidate query because all queried bone meshes are exact"
            if all_equal
            else "source and candidate queried separately because at least one queried bone mesh differs"
        ),
        "query_count": int(len(specs)),
        "queries": variants,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="flat exporter directory or old nested matrix root")
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subjects", default=",".join(SUBJECTS))
    parser.add_argument("--poses", default=",".join(POSES))
    return parser


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Capture articular surface audit V13",
        "",
        "This is a local frozen-domain vertex-to-opposing-bone-triangle diagnostic. "
        "It is not a whole-mesh intersection proof or a claim of physiological contact.",
        "",
        f"- partition: `{report.get('partition')}`",
        f"- negative signed threshold: `{float(report.get('negative_signed_threshold_m', 0.0)) * 1000.0:.3f} mm`",
        f"- publishable: `{report.get('publishable')}`",
        "",
        "| cell | geometry | primary query | unsigned p05–median–p95–max (mm) | signed min–p05–median–p95–max (mm) | negative signed >0.1 mm |",
        "|---|---|---|---:|---:|---:|",
    ]
    for subject, row in report.get("subjects", {}).items():
        if row.get("status") != "complete":
            continue
        for pose, cell in row.get("cells", {}).items():
            variant = next(iter(cell.get("queries", {}).values()), None)
            if variant is None:
                continue
            variant_name = next(iter(cell["queries"]))
            primary = [
                q for q in variant["queries"].values() if q.get("primary_direction")
            ]
            for query in primary:
                unsigned = query["unsigned_distance_m"]
                signed = query.get("signed_distance_m")
                u = " / ".join(f"{unsigned[key] * 1000.0:.3f}" for key in ("p05_m", "median_m", "p95_m", "max_m"))
                s = "unavailable" if signed is None else " / ".join(f"{signed[key] * 1000.0:.3f}" for key in ("min_m", "p05_m", "median_m", "p95_m", "max_m"))
                count = query.get("negative_signed_gt_0.1mm_count")
                lines.append(
                    f"| {subject}/{pose} | `{variant_name}` | `{query['name'] if 'name' in query else query['direction']}` "
                    f"({query['query_domain']} → {query['target_mesh']}) | {u} | {s} | {count if count is not None else '—'} |"
                )
    lines.extend(
        (
            "",
            "The JSON report contains every primary and reverse direction, the selected domain IDs, target mesh ranges, "
            "watertight/winding/volume checks, and exact source/candidate bone-equality policy.",
            "",
        )
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    input_root = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite articular audit output: {output}")
    subjects = tuple(part.strip() for part in str(args.subjects).split(",") if part.strip())
    poses = tuple(part.strip() for part in str(args.poses).split(",") if part.strip())
    unknown_subjects = sorted(set(subjects) - set(SUBJECTS))
    unknown_poses = sorted(set(poses) - set(POSES))
    if not subjects or unknown_subjects:
        raise ValueError(f"--subjects must select from {SUBJECTS}; unknown={unknown_subjects}")
    if not poses or unknown_poses:
        raise ValueError(f"--poses must select from {POSES}; unknown={unknown_poses}")
    started = time.perf_counter()
    operator = load_source_operator(Path(args.operator).expanduser().resolve(), validate=True, mmap=True)
    operator_digest = operator.runtime_digest(validate=False)
    if operator_digest != EXPECTED_OPERATOR_RUNTIME_DIGEST:
        raise ValueError("articular audit requires the frozen source operator")
    calibration = load_anatomical_calibration_v1(
        Path(args.calibration).expanduser().resolve(),
        operator=operator,
        required_scope="full_main_chain",
    )
    calibration_digest = _calibration_content_digest(calibration)
    specs = _query_specs()
    faces_reference: np.ndarray | None = None
    layout: dict[str, dict[str, Any]] | None = None
    report: dict[str, Any] = {
        "schema_version": 13,
        "artifact_kind": "CaptureArticularSurfaceAuditV13",
        "publishable": False,
        "status": "running",
        "partition": PARTITION,
        "negative_signed_threshold_m": NEGATIVE_SIGNED_THRESHOLD_M,
        "signed_convention": "negative means query point is inside a watertight target bone; nonclosed targets are unsigned-only",
        "scope": "frozen calibration validation-domain vertices to complete opposing source-bone triangles",
        "not_global_intersection_proof": True,
        "operator": str(Path(args.operator).expanduser().resolve()),
        "operator_runtime_digest": operator_digest,
        "calibration": str(Path(args.calibration).expanduser().resolve()),
        "calibration_digest": calibration_digest,
        "query_specs": [dict(spec) for spec in specs],
        "subjects": {},
        "failures": [],
    }
    output.mkdir(parents=True, exist_ok=False)
    try:
        for subject in subjects:
            report["subjects"].setdefault(subject, {"status": "running", "cells": {}})
            for pose in poses:
                cell_started = time.perf_counter()
                try:
                    path = _find_input(input_root, subject, pose)
                    cell = _load_cell(path)
                    if faces_reference is None:
                        faces_reference = np.asarray(cell["faces"], dtype=np.int32)
                        layout = _mesh_layout(operator.template_asset, faces_reference)
                    elif not np.array_equal(faces_reference, np.asarray(cell["faces"], dtype=np.int32)):
                        raise ValueError(f"{path}: faces differ from the first frozen topology")
                    assert layout is not None
                    result = _audit_cell(
                        cell,
                        subject=subject,
                        pose=pose,
                        calibration=calibration,
                        layout=layout,
                        specs=specs,
                    )
                    result["elapsed_seconds"] = float(time.perf_counter() - cell_started)
                    report["subjects"][subject]["cells"][pose] = result
                    _write_json(output / "progress.json", report)
                    print(
                        f"subject={subject} pose={pose} status=complete "
                        f"shared_bones={result['all_queried_bone_meshes_exactly_equal']}",
                        flush=True,
                    )
                except Exception as exc:
                    failure = {
                        "subject": subject,
                        "pose": pose,
                        "status": "evaluation_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    report["subjects"][subject]["cells"][pose] = failure
                    report["failures"].append(failure)
                    _write_json(output / "progress.json", report)
                    print(
                        f"subject={subject} pose={pose} status=evaluation_error "
                        f"error={type(exc).__name__}",
                        flush=True,
                    )
            report["subjects"][subject]["status"] = (
                "complete"
                if all(
                    report["subjects"][subject]["cells"].get(pose, {}).get("status") == "complete"
                    for pose in poses
                )
                else "evaluation_error"
            )
            _write_json(output / "progress.json", report)
    finally:
        report["status"] = "complete" if not report["failures"] else "evaluation_error"
        report["passed"] = bool(not report["failures"] and report["subjects"])
        report["pose_count"] = len(poses)
        report["subject_count"] = len(subjects)
        report["elapsed_seconds"] = float(time.perf_counter() - started)
        report["input_root"] = str(input_root)
        report["publishable"] = False
        _write_json(output / "report.json", report)
        _write_json(output / "progress.json", report)
        (output / "README.md").write_text(_markdown(report) + "\n", encoding="utf-8")
    print(
        f"articular_surface_audit subjects={len(subjects)} poses={len(poses)} "
        f"failures={len(report['failures'])} publishable=false output={output}",
        flush=True,
    )
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
