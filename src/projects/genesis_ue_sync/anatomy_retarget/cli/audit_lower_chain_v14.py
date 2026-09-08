"""Run an evidence-only lower-chain surface audit on frozen V14 cells.

The audit is deliberately bounded to the six latest combined-driver cells:
T-pose, the subject's own captured pose, and the other capture's pose for
subjects 213328 and 213712.  It reads the exported ``candidate_vertices``
without fitting, posing, rebinding, or changing any runtime artifact.

For each side it checks the actual authored lower meshes with
``surface_validation_v14.audit_bone_pair``:

* Ilium--Femur (the available side-specific hip-bone mesh),
* Femur--Tibia, and
* Tibia--Talus.

The source asset has a single central Sacrum mesh in addition to the two
Ilium meshes.  Sacrum--Femur is therefore recorded as a supplementary
central-pelvis context pair for both sides, so the central pelvic mesh is not
silently omitted.  There are no authored ``Pelvis``, ``Ischium``, or ``Pubis``
mesh objects in the frozen source asset.

``audit_bone_pair`` performs a complete VTK triangle-pair contact query and
bidirectional vertex/face-centroid signed samples.  Its sampled maximum depth
is retained as a lower bound; it is not presented as a theorem about every
point in the volume.  Closed/winding quality is reported separately.  A
contact or invalid signed target keeps the surface result failed, but this
report never converts that result into an anatomical or publication pass.
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

from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import load_source_operator
from projects.genesis_ue_sync.anatomy_retarget.surface_validation_v14 import (
    audit_bone_pair,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.audit_capture_articular_surfaces_v13 import (
    _mesh_face_subset,
)


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_INPUTS = {
    "213328": ROOT / "outputs/anatomy_retarget/v14_collar_driver_axes_213328_20260908_001",
    "213712": ROOT / "outputs/anatomy_retarget/v14_collar_driver_axes_213712_20260908_001",
}
DEFAULT_OUTPUT = ROOT / "outputs/anatomy_retarget/v14_lower_chain_audit_20260908_001"
DEFAULT_OPERATOR = ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
TRIANGLE_DEPTH_TOLERANCE_M = 5.0e-4
SUBJECTS = ("213328", "213712")


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
        number = float(value)
        return number if np.isfinite(number) else None
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
    """Load one geometry cell and validate only its immutable schema."""

    with np.load(path, allow_pickle=False) as data:
        required = {"faces", "source_vertices", "candidate_vertices", "pose"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path}: missing geometry fields {missing}")
        faces = np.asarray(data["faces"], dtype=np.int32)
        source = np.asarray(data["source_vertices"], dtype=np.float64)
        candidate = np.asarray(data["candidate_vertices"], dtype=np.float64)
        pose = np.asarray(data["pose"], dtype=np.float32).reshape(-1)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"{path}: faces must have shape [F,3]")
        if source.ndim != 2 or source.shape[1] != 3:
            raise ValueError(f"{path}: source_vertices must have shape [N,3]")
        if candidate.shape != source.shape:
            raise ValueError(f"{path}: candidate_vertices shape differs from source_vertices")
        if len(faces) == 0 or np.any(faces < 0) or np.any(faces >= len(source)):
            raise ValueError(f"{path}: faces reference invalid vertices")
        if pose.size != 165:
            raise ValueError(f"{path}: pose must contain 55*3 values")
        if not np.all(np.isfinite(source)) or not np.all(np.isfinite(candidate)):
            raise ValueError(f"{path}: source/candidate vertices contain non-finite values")
    return {
        "path": path.resolve(),
        "sha256": _sha256(path),
        "faces": faces,
        "source_vertices": source,
        "candidate_vertices": candidate,
        "pose": pose.reshape(55, 3),
        "array_hashes": {
            "faces": _array_sha256(faces),
            "source_vertices": _array_sha256(source),
            "candidate_vertices": _array_sha256(candidate),
            "pose": _array_sha256(pose),
        },
    }


def _mesh_layout(asset: Any, faces: np.ndarray) -> dict[str, dict[str, Any]]:
    """Resolve lower mesh ranges and complete local triangles from the asset."""

    names = [str(name) for name in asset.source_mesh_names]
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    lower_names = (
        "Ilium_L",
        "Ilium_R",
        "Sacrum",
        "Femur_L",
        "Femur_R",
        "Tibia_L",
        "Tibia_R",
        "Talus_L",
        "Talus_R",
    )
    result: dict[str, dict[str, Any]] = {}
    for name in lower_names:
        if name not in names:
            raise ValueError(f"source asset is missing required lower mesh {name!r}")
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


def _mesh_inventory(asset: Any, layout: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    names = [str(name) for name in asset.source_mesh_names]
    tissues = [str(value).strip() for value in asset.source_tissues]
    roles = [str(value).strip() for value in asset.source_mesh_roles]
    inventory: dict[str, Any] = {}
    for name, info in layout.items():
        index = int(info["mesh_index"])
        controller = np.asarray(asset.source_mesh_controller_bones, dtype=np.int64).reshape(-1)
        owner = int(controller[index])
        inventory[name] = {
            "mesh_index": index,
            "vertex_range": [int(info["vertex_start"]), int(info["vertex_stop"])],
            "vertex_count": int(info["vertex_count"]),
            "face_count": int(info["face_count"]),
            "tissue": tissues[index],
            "role": roles[index],
            "controller_bone_index": owner,
            "controller_bone_name": str(asset.source_bone_names[owner]),
        }
    return {
        "meshes": inventory,
        "pelvic_meshes_present_in_audited_asset": [
            name
            for name in names
            if tissues[names.index(name)].lower() == "bone"
            and any(token in name.lower() for token in ("ilium", "sacrum", "isch", "pub", "pelv"))
        ],
        "pelvis_mesh_names_absent": ["Pelvis_L", "Pelvis_R", "Ischium_L", "Ischium_R", "Pubis_L", "Pubis_R"],
        "hipbone_interpretation": "Ilium_L/R are the only side-specific authored hip/pelvic meshes and are used for the required ipsilateral hip pair",
    }


def _pair_specs() -> tuple[dict[str, str], ...]:
    pairs: list[dict[str, str]] = []
    for side, suffix in (("left", "L"), ("right", "R")):
        pairs.extend(
            (
                {
                    "side": side,
                    "joint": "hip",
                    "category": "required",
                    "first": f"Ilium_{suffix}",
                    "second": f"Femur_{suffix}",
                    "interpretation": "side-specific hipbone/ilium to femoral head region",
                },
                {
                    "side": side,
                    "joint": "hip_central_pelvis_context",
                    "category": "supplementary",
                    "first": "Sacrum",
                    "second": f"Femur_{suffix}",
                    "interpretation": "central sacrum to side femur context; supplementary to the ipsilateral Ilium pair",
                },
                {
                    "side": side,
                    "joint": "knee",
                    "category": "required",
                    "first": f"Femur_{suffix}",
                    "second": f"Tibia_{suffix}",
                    "interpretation": "femoral condyle to tibial plateau pair",
                },
                {
                    "side": side,
                    "joint": "ankle",
                    "category": "required",
                    "first": f"Tibia_{suffix}",
                    "second": f"Talus_{suffix}",
                    "interpretation": "distal tibia to talus pair",
                },
            )
        )
    return tuple(pairs)


def _mm(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) * 1000.0


def _pair_summary(
    vertices: np.ndarray,
    *,
    layout: Mapping[str, Mapping[str, Any]],
    spec: Mapping[str, str],
) -> dict[str, Any]:
    first = str(spec["first"])
    second = str(spec["second"])
    first_info = layout[first]
    second_info = layout[second]
    first_start, first_stop = int(first_info["vertex_start"]), int(first_info["vertex_stop"])
    second_start, second_stop = int(second_info["vertex_start"]), int(second_info["vertex_stop"])
    result = audit_bone_pair(
        vertices[first_start:first_stop],
        first_info["faces_local"],
        vertices[second_start:second_stop],
        second_info["faces_local"],
        depth_tolerance_m=TRIANGLE_DEPTH_TOLERANCE_M,
    )
    quality_rows: dict[str, Any] = {}
    quality = result.get("mesh_quality") or []
    for name, row in zip((first, second), quality):
        quality_rows[name] = {
            "watertight": bool(row.get("watertight", False)),
            "winding_consistent": bool(row.get("winding_consistent", False)),
            "signed_volume_m3": row.get("signed_volume_m3"),
            "degenerate_triangle_count": int(row.get("degenerate_triangle_count", -1)),
            "signed_distance_valid": bool(row.get("signed_distance_valid", False)),
        }
    signed_rows: list[dict[str, Any]] | None = None
    if result.get("signed_samples") is not None:
        signed_rows = []
        for direction, row in zip((f"{first}_to_{second}", f"{second}_to_{first}"), result["signed_samples"]):
            signed_rows.append(
                {
                    "direction": direction,
                    "sample_count": int(row.get("sample_count", 0)),
                    "max_depth_sampled_lower_bound_mm": _mm(row.get("max_depth_m")),
                    "above_tolerance_count": int(row.get("above_tolerance_count", 0)),
                    "min_absolute_distance_mm": _mm(row.get("min_absolute_distance_m")),
                }
            )
    pairs = np.asarray(result.get("triangle_pairs", []), dtype=np.int64).reshape(-1, 2)
    all_closed = bool(all(row.get("watertight", False) for row in quality)) if quality else False
    all_oriented = bool(all(row.get("winding_consistent", False) for row in quality)) if quality else False
    signed_valid = bool(all(row.get("signed_distance_valid", False) for row in quality)) if quality else False
    return {
        "pair": f"{first}--{second}",
        "side": str(spec["side"]),
        "joint": str(spec["joint"]),
        "category": str(spec["category"]),
        "interpretation": str(spec["interpretation"]),
        "first_mesh": first,
        "second_mesh": second,
        "mesh_vertex_ranges": {
            first: [first_start, first_stop],
            second: [second_start, second_stop],
        },
        "mesh_face_counts": {
            first: int(first_info["face_count"]),
            second: int(second_info["face_count"]),
        },
        "mesh_quality": quality_rows,
        "closed": {
            "all_watertight": all_closed,
            "all_winding_consistent": all_oriented,
            "all_signed_distance_valid": signed_valid,
            "interpretation": "closed means watertight; signed validity additionally requires consistent winding, nonzero volume, and nondegenerate triangles",
        },
        "triangle_pair_count": int(result.get("triangle_pair_count", len(pairs))),
        "triangle_pairs_sample": pairs[:20].tolist(),
        "triangle_contact_method": str(result.get("method", "unknown")),
        "vtk_version": str(result.get("vtk_version", "unknown")),
        "signed_samples": signed_rows,
        "signed_depth_tolerance_mm": float(TRIANGLE_DEPTH_TOLERANCE_M * 1000.0),
        "max_depth_is_sampled_lower_bound": bool(result.get("max_depth_is_sampled_lower_bound", True)),
        "surface_validation_passed": bool(result.get("passed", False)),
        "surface_validation_reason": str(result.get("reason", "unknown")),
        "anatomical_passed": False,
        "publishable": False,
    }


def _find_input(root: Path, subject: str, pose_file_stem: str) -> Path:
    if root.is_file():
        return root.resolve()
    candidates = (
        root / f"subject_{subject}_{pose_file_stem}.npz",
        root / subject / f"subject_{subject}_{pose_file_stem}.npz",
        root / f"subject_{subject}" / "comparisons" / f"{pose_file_stem}.npz",
        root / "subjects" / f"subject_{subject}" / "comparisons" / f"{pose_file_stem}.npz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"no geometry NPZ for subject={subject} pose_file={pose_file_stem}; searched "
        + ", ".join(str(value) for value in candidates)
    )


def _related_identity(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for relative in ("report.json", "compiled/manifest.json"):
        path = root / relative
        if not path.is_file():
            continue
        entry: dict[str, Any] = {"path": str(path.resolve()), "sha256": _sha256(path)}
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, Mapping):
            for key in (
                "subject",
                "source_pack_kind",
                "source_pack_runtime_digest",
                "source_pack_content_digest",
                "shape_reference_kind",
                "rotation_transport",
                "anatomical_passed",
                "publishable",
            ):
                if key in parsed:
                    entry[key] = parsed[key]
            provenance = parsed.get("provenance")
            if isinstance(provenance, Mapping):
                for key in (
                    "shape_reference_kind",
                    "source_pack_runtime_digest",
                    "rotation_transport",
                ):
                    if key in provenance:
                        entry[key] = provenance[key]
        result[relative.replace("/", "_")] = entry
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-213328", type=Path, default=DEFAULT_INPUTS["213328"])
    parser.add_argument("--input-213712", type=Path, default=DEFAULT_INPUTS["213712"])
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subjects", help="comma-separated subject IDs; default is 213328,213712")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    input_roots = {
        "213328": Path(args.input_213328).expanduser().resolve(),
        "213712": Path(args.input_213712).expanduser().resolve(),
    }
    subjects = _parse_csv(args.subjects) or SUBJECTS
    unknown = [subject for subject in subjects if subject not in input_roots]
    if unknown:
        raise ValueError(f"unsupported subjects {unknown}; this bounded audit supports {SUBJECTS}")
    output = Path(args.output).expanduser().resolve()
    operator_path = Path(args.operator).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite lower-chain audit output: {output}")
    for subject in subjects:
        if not input_roots[subject].exists():
            raise FileNotFoundError(input_roots[subject])
    started = time.perf_counter()
    operator = load_source_operator(operator_path, validate=True, mmap=True)
    asset = operator.template_asset
    operator_digest = operator.runtime_digest(validate=False)
    faces_reference: np.ndarray | None = None
    layout: dict[str, dict[str, Any]] | None = None
    pair_specs = _pair_specs()
    report: dict[str, Any] = {
        "schema_version": 14,
        "artifact_kind": "LowerChainSurfaceAuditV14",
        "status": "running",
        "audit_status": "running",
        "evidence_only": True,
        "fit_or_tuning_performed": False,
        "runtime_modified": False,
        "anatomical_passed": False,
        "publishable": False,
        "operational_evaluation_completed": False,
        "subject_cells_policy": "T/own/swap for each of 213328 and 213712 using the frozen latest combined-driver exports",
        "variant_audited": "candidate_vertices",
        "raw142_source_in_each_input": True,
        "raw142_source_audited": False,
        "surface_validation_module": "surface_validation_v14.audit_bone_pair",
        "surface_validation_policy": "complete VTK all-contact triangle pairs plus bidirectional vertex and face-centroid signed samples",
        "signed_depth_tolerance_m": TRIANGLE_DEPTH_TOLERANCE_M,
        "signed_depth_tolerance_mm": TRIANGLE_DEPTH_TOLERANCE_M * 1000.0,
        "signed_depth_interpretation": "max_depth is a sampled lower bound; it cannot certify non-penetration between unsampled points",
        "operator": str(operator_path),
        "operator_runtime_digest": operator_digest,
        "template_vertices_sha256": _array_sha256(asset.vertices_rest),
        "template_faces_sha256": _array_sha256(asset.faces),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "inputs": {subject: _related_identity(input_roots[subject]) for subject in subjects},
        "mesh_inventory": None,
        "pair_definitions": list(pair_specs),
        "cells": {},
        "failures": [],
        "candidate_flags": [],
    }
    output.mkdir(parents=True, exist_ok=False)
    try:
        for subject in subjects:
            report["cells"].setdefault(subject, {})
            other = "213712" if subject == "213328" else "213328"
            cell_defs = (
                ("tpose", "tpose"),
                ("own", f"pose_{subject}"),
                ("swap", f"pose_{other}"),
            )
            for cell_label, pose_stem in cell_defs:
                cell_started = time.perf_counter()
                try:
                    path = _find_input(input_roots[subject], subject, pose_stem)
                    cell = _load_cell(path)
                    if faces_reference is None:
                        faces_reference = np.asarray(cell["faces"], dtype=np.int32)
                        layout = _mesh_layout(asset, faces_reference)
                        report["mesh_inventory"] = _mesh_inventory(asset, layout)
                    elif not np.array_equal(faces_reference, cell["faces"]):
                        raise ValueError(f"{path}: faces differ from first input cell")
                    assert layout is not None
                    if len(cell["candidate_vertices"]) != len(asset.vertices_rest):
                        raise ValueError(
                            f"{path}: candidate vertex count {len(cell['candidate_vertices'])} "
                            f"does not match frozen source asset {len(asset.vertices_rest)}"
                        )
                    pair_rows = {
                        f"{row['first']}--{row['second']}": _pair_summary(
                            cell["candidate_vertices"], layout=layout, spec=row
                        )
                        for row in pair_specs
                    }
                    report["cells"][subject][cell_label] = {
                        "status": "complete",
                        "subject": subject,
                        "cell": cell_label,
                        "input_pose_stem": pose_stem,
                        "input": {
                            "path": str(cell["path"]),
                            "sha256": cell["sha256"],
                            "array_hashes": cell["array_hashes"],
                        },
                        "pose_array_hash": _array_sha256(cell["pose"]),
                        "candidate_vertices_sha256": cell["array_hashes"]["candidate_vertices"],
                        "faces_sha256": cell["array_hashes"]["faces"],
                        "pair_checks": pair_rows,
                        "anatomical_passed": False,
                        "publishable": False,
                        "elapsed_seconds": float(time.perf_counter() - cell_started),
                    }
                    _write_json(output / "progress.json", report)
                    print(f"subject={subject} cell={cell_label} status=complete", flush=True)
                except Exception as exc:
                    failure = {
                        "subject": subject,
                        "cell": cell_label,
                        "pose_stem": pose_stem,
                        "status": "evaluation_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    report["cells"][subject][cell_label] = failure
                    report["failures"].append(failure)
                    _write_json(output / "progress.json", report)
                    print(
                        f"subject={subject} cell={cell_label} status=evaluation_error "
                        f"error={type(exc).__name__}",
                        flush=True,
                    )
        if report["mesh_inventory"] is None and not report["failures"]:
            raise RuntimeError("no lower-chain cells were evaluated")
        for subject, cells in report["cells"].items():
            for cell_label, cell in cells.items():
                if not isinstance(cell, Mapping) or cell.get("status") != "complete":
                    continue
                for pair_name, pair in cell["pair_checks"].items():
                    quality = pair.get("closed", {})
                    signed_rows = pair.get("signed_samples") or []
                    depth_fail = any(
                        int(row.get("above_tolerance_count", 0)) > 0 for row in signed_rows
                    )
                    if (
                        int(pair.get("triangle_pair_count", 0)) > 0
                        or not bool(quality.get("all_watertight", False))
                        or not bool(quality.get("all_winding_consistent", False))
                        or not bool(quality.get("all_signed_distance_valid", False))
                        or depth_fail
                    ):
                        report["candidate_flags"].append(
                            {
                                "subject": subject,
                                "cell": cell_label,
                                "pair": pair_name,
                                "triangle_pair_count": int(pair.get("triangle_pair_count", 0)),
                                "all_watertight": bool(quality.get("all_watertight", False)),
                                "all_winding_consistent": bool(quality.get("all_winding_consistent", False)),
                                "all_signed_distance_valid": bool(quality.get("all_signed_distance_valid", False)),
                                "signed_depth_over_tolerance": depth_fail,
                                "surface_validation_reason": pair.get("surface_validation_reason"),
                            }
                        )
    finally:
        report["status"] = "complete" if not report["failures"] else "evaluation_error"
        report["audit_status"] = report["status"]
        report["operational_evaluation_completed"] = bool(not report["failures"])
        report["cell_count"] = sum(len(cells) for cells in report["cells"].values())
        report["pair_check_count"] = sum(
            len(cell.get("pair_checks", {}))
            for cells in report["cells"].values()
            for cell in cells.values()
            if isinstance(cell, Mapping)
        )
        report["candidate_flag_count"] = len(report["candidate_flags"])
        report["elapsed_seconds"] = float(time.perf_counter() - started)
        report["anatomical_passed"] = False
        report["publishable"] = False
        _write_json(output / "report.json", report)
        _write_json(output / "progress.json", report)
    print(
        f"lower_chain_v14_audit cells={report.get('cell_count', 0)} "
        f"pair_checks={report.get('pair_check_count', 0)} "
        f"failures={len(report.get('failures', []))} "
        f"candidate_flags={report.get('candidate_flag_count', 0)} "
        f"anatomical_passed=false output={output}",
        flush=True,
    )
    return 1 if report.get("failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
