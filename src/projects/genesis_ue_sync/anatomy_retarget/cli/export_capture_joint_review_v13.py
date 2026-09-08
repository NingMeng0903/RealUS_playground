"""Export six capture-pose geometry cells for independent joint review.

The source side is the frozen V12e subject posed with
``pose_whole_chain_vertices_v10``.  The candidate side replays the already
compiled V13 exact-replay runtime from ``v13_exact_replay_20260908_001`` after
replacing its serialized source asset with the exact materialized source rig.
No attachment compilation, fitting, optimization, or renderer code runs here.
"""

from __future__ import annotations

import argparse
import dataclasses
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
    EXPECTED_ORACLE_SHA256,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.run_material_matrix_v13 import (
    _load_capture,
    _load_subject_v11,
)
from projects.genesis_ue_sync.anatomy_retarget.material_runtime_v13 import (
    load_material_runtime_v13,
    recover_shared_weight_rest,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_fit_to_smplx55,
    smplx_pose_hash,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v10 import (
    build_pose_map_v10,
    pose_whole_chain_vertices_v10,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    _smplx_joint_kinematics_v7,
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
    smplx_body_surface_v7,
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
DEFAULT_CAPTURE_213328 = ROOT / "smplx_outputs/20260713_213328/moment_0000/smplx_result.npz"
DEFAULT_CAPTURE_213712 = ROOT / "smplx_outputs/20260713_213712/moment_0000/smplx_result.npz"
DEFAULT_V11 = ROOT / "outputs/anatomy_retarget/v11_candidates/chain_retarget_v11_anchored_001"
DEFAULT_V12E = ROOT / "outputs/anatomy_retarget/v12_candidates/chain_retarget_v12e_forearm_mesh_001"
DEFAULT_MATERIAL = ROOT / "outputs/anatomy_retarget/v13_exact_replay_20260908_001"
DEFAULT_AMONG_US = Path("/media/camp/EXT_DRIVE/Among_US")
DEFAULT_OUTPUT = ROOT / "outputs/anatomy_retarget/v13_capture_joint_geometry_20260908_001"

SUBJECTS = ("213328", "213712")
POSE_NAMES = ("tpose", "pose_213328", "pose_213712")
CAPTURE_SHA256 = {
    "213328": "c7a6c3783dc7b764e1f8013ab0a8a45d0380b81c97ac929f67c7a5a526eecbc1",
    "213712": "9887848b7b086d71a875beea50b1d7c7819a11c7b67996fe0d83f451da79b689",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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


def _pose_skin_joints(
    model: Mapping[str, np.ndarray], *, betas: np.ndarray, pose: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    skin, skin_faces = smplx_body_surface_v7(model, betas=betas, pose_axis_angle=pose)
    _rest_joints, globals_, _rest_to_pose = _smplx_joint_kinematics_v7(
        model, betas=betas, pose_axis_angle=pose
    )
    return (
        np.asarray(skin, dtype=np.float64),
        np.asarray(skin_faces, dtype=np.int32),
        np.asarray(globals_, dtype=np.float64)[:, :3, 3],
    )


def _tissue_codes(asset: Any) -> np.ndarray:
    codes_by_name = {
        "bone": 0,
        "vessel": 1,
        "nerve": 2,
        "organ": 3,
        "heart": 4,
        "connective": 5,
        "connective_tissue": 5,
    }
    values = np.full(len(asset.vertices_rest), -1, dtype=np.int8)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    for tissue, (start, stop) in zip(asset.source_tissues, ranges.tolist()):
        name = str(tissue).strip().lower()
        if name not in codes_by_name:
            raise ValueError(f"unrecognised tissue label {tissue!r}")
        values[int(start) : int(stop)] = codes_by_name[name]
    if np.any(values < 0):
        raise ValueError("materialized source asset contains unlabelled vertices")
    return values


def _asset_exact_equal(first: Any, second: Any) -> dict[str, Any]:
    """Compare every serialized ndarray field before runtime source rebinding."""

    mismatches: list[str] = []
    checked = 0
    for field in dataclasses.fields(first):
        left, right = getattr(first, field.name), getattr(second, field.name)
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            checked += 1
            if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)):
                mismatches.append(field.name)
            elif left.shape != right.shape or left.dtype != right.dtype or not np.array_equal(left, right):
                mismatches.append(field.name)
    return {"checked_array_fields": checked, "exact": not mismatches, "mismatches": mismatches}


def _resolve_runtime_root(material_root: Path, subject: str) -> Path:
    """Resolve an existing subject runtime package without compiling it."""

    material_root = Path(material_root).expanduser().resolve()
    candidates = (
        material_root / subject,
        material_root / "subjects" / f"subject_{subject}" / "runtime",
    )
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate
    joined = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"no precompiled MaterialRuntimeV13 package for {subject}; checked {joined}"
    )


def _capture_record(path: Path, *, model_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    digest = _sha256(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"Rh", "Th", "poses", "shapes"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path}: capture is missing {missing}")
        rh = np.asarray(data["Rh"], dtype=np.float32).reshape(3)
        th = np.asarray(data["Th"], dtype=np.float32).reshape(3)
        root_align = np.asarray(data["root_align_offset"], dtype=np.float32).reshape(3) if "root_align_offset" in data.files else np.zeros(3, dtype=np.float32)
        raw_pose = np.asarray(data["poses"], dtype=np.float32).reshape(-1)
        betas = np.asarray(data["shapes"], dtype=np.float32).reshape(-1)[:10]
        pose = easymocap_fit_to_smplx55(rh, raw_pose, model_path=model_path)
        record = {
            "path": str(path),
            "sha256": digest,
            "keys": list(data.files),
            "rh_mapped_root_axis_angle": rh.tolist(),
            "th_raw_m": th.tolist(),
            "root_align_offset_m": root_align.tolist(),
            "poses_raw_shape": list(np.asarray(data["poses"]).shape),
            "shapes_shape": list(np.asarray(data["shapes"]).shape),
            "betas_sha256_float32": _array_sha256(betas),
            "pose55_sha256_float32": _array_sha256(pose),
            "pose55_hash": smplx_pose_hash(pose, th + root_align),
            "translation_applied_to_exported_geometry": False,
        }
    summary = path.parent.parent / "capture_summary.json"
    if summary.is_file():
        try:
            summary_payload = json.loads(summary.read_text(encoding="utf-8"))
            capture = summary_payload.get("capture", {})
            moment = capture.get("moment", {}) if isinstance(capture, Mapping) else {}
            record["capture_summary"] = {
                "path": str(summary),
                "sha256": _sha256(summary),
                "run_name": summary_payload.get("run_name"),
                "smplx_fit_2d_source": summary_payload.get("smplx_fit_2d_source"),
                "fit_ok": capture.get("ok"),
                "frame_index": moment.get("frame_index"),
                "timestamp_ns": moment.get("timestamp_ns"),
            }
        except (OSError, ValueError, TypeError) as exc:
            record["capture_summary"] = {"path": str(summary), "read_error": f"{type(exc).__name__}: {exc}"}
    return np.asarray(pose, dtype=np.float32), record


def _scan_among_us(root: Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    patterns = (
        "outputs/offline_capture/*/moment_0000/smplx_result.npz",
        "outputs/offline_compare/**/smplx_result.npz",
    )
    artifacts: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            path = path.resolve()
            if path in seen:
                continue
            seen.add(path)
            try:
                digest = _sha256(path)
                with np.load(path, allow_pickle=False) as data:
                    keys = list(data.files)
                    shapes = {name: list(np.asarray(data[name]).shape) for name in data.files}
                artifacts.append({"path": str(path), "sha256": digest, "keys": keys, "shapes": shapes})
            except (OSError, ValueError) as exc:
                artifacts.append({"path": str(path), "status": "unreadable", "error": f"{type(exc).__name__}: {exc}"})
    matching = [row for row in artifacts if row.get("sha256") in set(CAPTURE_SHA256.values())]
    return {
        "root": str(root),
        "patterns": list(patterns),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "matching_repo_capture_sha256": matching,
        "status": "matched" if matching else "no_direct_sha_match",
        "interpretation": (
            "No Among_US smplx_result.npz under the scanned outputs has either frozen repo capture SHA; "
            "the canonical subject inputs below remain the RealUS_playground capture paths."
            if not matching
            else "At least one Among_US artifact has a frozen repo capture SHA."
        ),
    }


def _save_cell(
    path: Path,
    *,
    pose_name: str,
    subject: str,
    pose: np.ndarray,
    source_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    skin_vertices: np.ndarray,
    smplx_joints: np.ndarray,
    faces: np.ndarray,
    skin_faces: np.ndarray,
    vertex_tissue: np.ndarray,
    capture_record: Mapping[str, Any] | None,
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    capture_record = capture_record or {}
    np.savez_compressed(
        path,
        faces=np.asarray(faces, dtype=np.int32),
        skin_faces=np.asarray(skin_faces, dtype=np.int32),
        source_vertices=np.asarray(source_vertices, dtype=np.float32),
        candidate_vertices=np.asarray(candidate_vertices, dtype=np.float32),
        skin_vertices=np.asarray(skin_vertices, dtype=np.float32),
        smplx_joints=np.asarray(smplx_joints, dtype=np.float32),
        pose=np.asarray(pose, dtype=np.float32),
        vertex_tissue=np.asarray(vertex_tissue, dtype=np.int8),
        capture_Rh=np.asarray(capture_record.get("rh_mapped_root_axis_angle", np.zeros(3)), dtype=np.float32),
        capture_Th=np.asarray(capture_record.get("th_raw_m", np.zeros(3)), dtype=np.float32),
        capture_root_align_offset=np.asarray(capture_record.get("root_align_offset_m", np.zeros(3)), dtype=np.float32),
        capture_translation_applied=np.asarray(bool(capture_record.get("translation_applied_to_exported_geometry", False))),
        pose_name=np.asarray(str(pose_name)),
        subject=np.asarray(str(subject)),
        metadata_json=np.asarray(json.dumps(_json_ready(metadata), sort_keys=True, allow_nan=False)),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--smplx-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--capture-213328", type=Path, default=DEFAULT_CAPTURE_213328)
    parser.add_argument("--capture-213712", type=Path, default=DEFAULT_CAPTURE_213712)
    parser.add_argument("--v11", type=Path, default=DEFAULT_V11)
    parser.add_argument("--v12e", type=Path, default=DEFAULT_V12E)
    parser.add_argument("--material-runtime", type=Path, default=DEFAULT_MATERIAL)
    parser.add_argument("--among-us-root", type=Path, default=DEFAULT_AMONG_US)
    parser.add_argument("--subjects", default=",".join(SUBJECTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an existing output directory and append requested subjects",
    )
    return parser


def _run_subject(
    subject: str,
    *,
    args: argparse.Namespace,
    operator: Any,
    calibration: Any,
    model: Mapping[str, np.ndarray],
    model_path: Path,
    operator_digest: str,
    calibration_digest: str,
    oracle: Path,
    capture_poses: Mapping[str, np.ndarray],
    capture_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    started = time.perf_counter()
    v11 = _load_subject_v11(Path(args.v11).expanduser().resolve() / f"subject_{subject}")
    v12e = _load_subject_v11(Path(args.v12e).expanduser().resolve() / f"subject_{subject}")
    for label, value in (("v11", v11), ("v12e", v12e)):
        if value.source_operator_digest != operator_digest:
            raise ValueError(f"{subject} {label}: source operator digest mismatch")
        if value.calibration_digest != calibration_digest:
            raise ValueError(f"{subject} {label}: calibration digest mismatch")
        if not np.array_equal(np.asarray(value.betas), np.asarray(v12e.betas)):
            raise ValueError(f"{subject}: V11/V12e betas differ")
    materialized = materialize_subject(operator, betas=np.asarray(v12e.betas), gender="male")
    asset = materialized.rigged_asset
    runtime_root = _resolve_runtime_root(Path(args.material_runtime), subject)
    material_root = Path(args.material_runtime).expanduser().resolve()
    runtime_layout = (
        "exact_replay_subject_package"
        if runtime_root.parent == material_root
        else "legacy_matrix_subject_package"
    )
    runtime = load_material_runtime_v13(runtime_root)
    asset_equal = _asset_exact_equal(runtime.source_asset, asset)
    if not asset_equal["exact"]:
        raise ValueError(f"{subject}: compiled runtime source asset differs from exact materialized asset: {asset_equal['mismatches']}")
    rebound_runtime = dataclasses.replace(runtime, source_asset=asset)
    target_rest_recovered, recovery_report = recover_shared_weight_rest(v11, v12e, asset)
    target_rest_error = float(np.max(np.abs(np.asarray(rebound_runtime.target_rest, dtype=np.float64) - target_rest_recovered)))
    if target_rest_error > 2.0e-7:
        raise ValueError(f"{subject}: runtime target rest differs from persisted recovery by {target_rest_error} m")
    source_pose_map = build_pose_map_v10(
        v12e,
        asset=asset,
        calibration=calibration,
        oracle_path=oracle,
        source_operator_digest=operator_digest,
    )
    poses = {
        "tpose": np.zeros((55, 3), dtype=np.float32),
        "pose_213328": np.asarray(capture_poses["213328"], dtype=np.float32),
        "pose_213712": np.asarray(capture_poses["213712"], dtype=np.float32),
    }
    tissue = _tissue_codes(asset)
    cells: dict[str, Any] = {}
    files: dict[str, str] = {}
    for pose_name in POSE_NAMES:
        pose_started = time.perf_counter()
        pose = poses[pose_name]
        source_vertices, _source_globals = pose_whole_chain_vertices_v10(
            v12e,
            source_pose_map,
            source_asset=asset,
            pose_axis_angle=pose,
        )
        candidate_vertices = rebound_runtime.apply_pose(pose, mode="attachments")
        skin, skin_faces, smplx_joints = _pose_skin_joints(
            model,
            betas=np.asarray(v12e.betas, dtype=np.float64),
            pose=pose,
        )
        capture_label = pose_name.removeprefix("pose_") if pose_name.startswith("pose_") else None
        capture_record = capture_records.get(capture_label) if capture_label else None
        filename = f"subject_{subject}_{pose_name}.npz"
        output_path = Path(args.output).expanduser().resolve() / filename
        metadata = {
            "schema_version": 13,
            "artifact_kind": "CaptureJointGeometryComparisonV13",
            "subject": subject,
            "pose": pose_name,
            "source_label": "V12e pose_whole_chain_vertices_v10",
            "candidate_label": "compiled V13 material runtime attachments",
            "source_subject_artifact": str(Path(args.v12e).expanduser().resolve() / f"subject_{subject}"),
            "material_runtime_root": str(runtime_root),
            "material_runtime_manifest_sha256": _sha256(runtime_root / "manifest.json"),
            "material_runtime_layout": runtime_layout,
            "material_runtime_rebound_to_exact_source_asset": True,
            "material_attachment_recompiled": False,
            "soft_or_vessel_optimization": False,
            "skin_forward": "numpy SMPL-X male frozen model, subject betas and pose55, canonical translation frame",
            "capture_translation_applied": False,
            "capture_provenance": capture_record,
            "publishable": False,
        }
        _save_cell(
            output_path,
            pose_name=pose_name,
            subject=subject,
            pose=pose,
            source_vertices=source_vertices,
            candidate_vertices=candidate_vertices,
            skin_vertices=skin,
            smplx_joints=smplx_joints,
            faces=asset.faces,
            skin_faces=skin_faces,
            vertex_tissue=tissue,
            capture_record=capture_record,
            metadata=metadata,
        )
        files[pose_name] = str(output_path)
        cells[pose_name] = {
            "status": "complete",
            "available": True,
            "file": str(output_path),
            "source_vertex_count": int(len(source_vertices)),
            "candidate_vertex_count": int(len(candidate_vertices)),
            "skin_vertex_count": int(len(skin)),
            "joint_count": int(len(smplx_joints)),
            "source_candidate_max_displacement_m": float(np.max(np.linalg.norm(np.asarray(candidate_vertices, dtype=np.float64) - np.asarray(source_vertices, dtype=np.float64), axis=1))),
            "source_candidate_rms_displacement_m": float(np.sqrt(np.mean((np.asarray(candidate_vertices, dtype=np.float64) - np.asarray(source_vertices, dtype=np.float64)) ** 2))),
            "pose55_sha256_float32": _array_sha256(pose),
            "elapsed_seconds": float(time.perf_counter() - pose_started),
        }
    return {
        "subject": subject,
        "status": "complete",
        "publishable": False,
        "v11_subject": str(Path(args.v11).expanduser().resolve() / f"subject_{subject}"),
        "v12e_subject": str(Path(args.v12e).expanduser().resolve() / f"subject_{subject}"),
        "material_runtime": str(runtime_root),
        "material_runtime_layout": runtime_layout,
        "material_runtime_manifest_sha256": _sha256(runtime_root / "manifest.json"),
        "asset_exact_rebind": asset_equal,
        "target_rest_recovery": recovery_report,
        "runtime_target_rest_recovery_max_abs_m": target_rest_error,
        "cells": cells,
        "files": files,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite capture-joint output: {output}")
        manifest_path = output / "matrix_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"cannot resume output without matrix_manifest.json: {output}")
        report = json.loads(manifest_path.read_text(encoding="utf-8"))
        if report.get("artifact_kind") != "CaptureJointGeometryReviewV13":
            raise ValueError(f"cannot resume unrelated output artifact: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
        report = {
            "schema_version": 13,
            "artifact_kind": "CaptureJointGeometryReviewV13",
            "publishable": False,
            "trusted_latest_updated": False,
            "source_contract": "V12e pose_whole_chain_vertices_v10",
            "candidate_contract": "precompiled V13 material runtime attachments; source asset exact rebind",
            "no_reoptimization": True,
            "subjects": {},
            "failures": [],
        }
    started = time.perf_counter()
    report["failures"] = []
    try:
        operator = load_source_operator(Path(args.operator).expanduser().resolve(), mmap=True)
        operator_digest = operator.runtime_digest(validate=False)
        if operator_digest != EXPECTED_OPERATOR_RUNTIME_DIGEST:
            raise ValueError("capture joint export requires the frozen 142 source operator")
        calibration = load_anatomical_calibration_v1(
            Path(args.calibration).expanduser().resolve(),
            operator=operator,
            required_scope="full_main_chain",
        )
        calibration_digest = _calibration_content_digest(calibration)
        oracle = Path(args.oracle).expanduser().resolve()
        oracle_sha = _sha256(oracle)
        if oracle_sha != EXPECTED_ORACLE_SHA256:
            raise ValueError("capture joint export requires the frozen Blender oracle")
        model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
        model = load_smplx_model_v7(model_path)
        capture_paths = {"213328": Path(args.capture_213328), "213712": Path(args.capture_213712)}
        capture_poses: dict[str, np.ndarray] = {}
        capture_records: dict[str, dict[str, Any]] = {}
        for label, path in capture_paths.items():
            pose, record = _capture_record(path, model_path=model_path)
            if record["sha256"] != CAPTURE_SHA256[label]:
                raise ValueError(f"capture {label} differs from frozen SHA")
            capture_poses[label] = pose
            capture_records[label] = record
        among_us_scan = _scan_among_us(Path(args.among_us_root))
        report["provenance"] = {
            "operator": str(Path(args.operator).expanduser().resolve()),
            "operator_runtime_digest": operator_digest,
            "calibration": str(Path(args.calibration).expanduser().resolve()),
            "calibration_digest": calibration_digest,
            "oracle": str(oracle),
            "oracle_sha256": oracle_sha,
            "smplx_model": str(model_path),
            "smplx_model_sha256": model_sha,
            "v11_root": str(Path(args.v11).expanduser().resolve()),
            "v12e_root": str(Path(args.v12e).expanduser().resolve()),
            "material_runtime_root": str(Path(args.material_runtime).expanduser().resolve()),
            "repo_captures": capture_records,
            "among_us_capture_scan": among_us_scan,
            "capture_identity_policy": "repo captures are accepted only after exact frozen SHA match; Among_US scan is provenance and is never substituted silently",
        }
        subjects = [part.strip() for part in str(args.subjects).split(",") if part.strip()]
        unknown = sorted(set(subjects) - set(SUBJECTS))
        if unknown:
            raise ValueError(f"unknown subjects: {unknown}")
        for subject in subjects:
            try:
                row = _run_subject(
                    subject,
                    args=args,
                    operator=operator,
                    calibration=calibration,
                    model=model,
                    model_path=model_path,
                    operator_digest=operator_digest,
                    calibration_digest=calibration_digest,
                    oracle=oracle,
                    capture_poses=capture_poses,
                    capture_records=capture_records,
                )
                report["subjects"][subject] = row
                _write_json(output / "progress.json", report)
                print(f"subject={subject} status=complete", flush=True)
            except Exception as exc:
                failure = {"subject": subject, "status": "evaluation_error", "error": f"{type(exc).__name__}: {exc}"}
                report["subjects"][subject] = failure
                report["failures"].append(failure)
                _write_json(output / "progress.json", report)
                print(f"subject={subject} status=evaluation_error error={type(exc).__name__}", flush=True)
    except Exception as exc:
        report["status"] = "evaluation_error"
        report["failures"].append({"status": "evaluation_error", "error": f"{type(exc).__name__}: {exc}"})
        report["elapsed_seconds"] = float(time.perf_counter() - started)
        _write_json(output / "matrix_manifest.json", report)
        return 1
    report["status"] = "complete"
    report["passed"] = bool(report["subjects"] and not report["failures"] and all(row.get("status") == "complete" for row in report["subjects"].values()))
    report["elapsed_seconds"] = float(time.perf_counter() - started)
    _write_json(output / "matrix_manifest.json", report)
    _write_json(output / "progress.json", report)
    print(f"capture_joint subjects={len(report['subjects'])} failures={len(report['failures'])} publishable=false", flush=True)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
