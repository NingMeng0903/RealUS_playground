"""Export a small, authenticated V7-versus-raw-142 capture baseline.

This entry point is deliberately an exporter, not a solver.  It reads the
already-frozen V7 subject bundles, evaluates the raw 142 source runtime and
the V7 ``PoseMapV1`` right-multiply replay for two subjects and three poses,
and writes six Genesis-compatible geometry cells.  A missing or mismatched
V7 bundle is a hard error; this command never rebuilds it.

The resulting cells have the same fields consumed by the existing Genesis
renderer and articular-surface audit:

``faces, skin_faces, source_vertices, candidate_vertices, skin_vertices,
smplx_joints, pose, vertex_tissue, metadata_json``.

Translations from the EasyMocap capture are retained as provenance only.  All
geometry stays in the canonical anatomy frame used by the existing audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
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
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_fit_to_smplx55,
    smplx_pose_hash,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import (
    build_pose_map_v1,
    pose_map_content_digest,
    pose_whole_chain_vertices,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    _smplx_joint_kinematics_v7,
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
    smplx_body_surface_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    ResidentPoseEvaluatorV8,
    load_source_operator,
    materialize_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.v7_artifacts import rigged_asset_digest
from projects.genesis_ue_sync.anatomy_retarget.whole_chain_rest_fit_v1 import (
    BASELINE_COMMIT,
    FROZEN_CAPTURE_SHA256,
    load_whole_chain_rest_fit_v1,
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
DEFAULT_V7 = ROOT / "outputs/anatomy_retarget/v8_candidates/chain_retarget_v7_node2_001"
DEFAULT_AMONG_US = Path("/media/camp/EXT_DRIVE/Among_US")
DEFAULT_OUTPUT = ROOT / "outputs/anatomy_retarget/v14_baselines_20260908_001"

SUBJECTS = ("213328", "213712")
POSES = ("tpose", "pose_213328", "pose_213712")

# These are the only two capture files accepted by this frozen comparison.
# The paths are deliberately local to RealUS_playground: an Among_US scan is
# recorded for provenance but is never substituted when its SHA differs.
CAPTURE_SHA256 = dict(FROZEN_CAPTURE_SHA256)


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


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


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


def _capture_pose(path: Path, *, model_path: Path, subject: str) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"capture {subject} is missing: {path}")
    digest = _sha256(path)
    expected = CAPTURE_SHA256[subject]
    if digest != expected:
        raise ValueError(f"capture {subject} SHA mismatch: expected {expected}, got {digest}")
    with np.load(path, allow_pickle=False) as data:
        required = {"Rh", "Th", "poses", "shapes"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"capture {subject} is missing fields {missing}")
        rh = np.asarray(data["Rh"], dtype=np.float32).reshape(3)
        th = np.asarray(data["Th"], dtype=np.float32).reshape(3)
        root_align = (
            np.asarray(data["root_align_offset"], dtype=np.float32).reshape(3)
            if "root_align_offset" in data.files
            else np.zeros(3, dtype=np.float32)
        )
        raw_pose = np.asarray(data["poses"], dtype=np.float32)
        shapes = np.asarray(data["shapes"], dtype=np.float32)
        betas = shapes.reshape(-1)[:10]
        pose = easymocap_fit_to_smplx55(
            rh,
            raw_pose,
            model_path=model_path,
        )
        record = {
            "subject": subject,
            "path": str(path),
            "sha256": digest,
            "keys": list(data.files),
            "Rh_shape": list(np.asarray(data["Rh"]).shape),
            "Th_shape": list(np.asarray(data["Th"]).shape),
            "poses_shape": list(raw_pose.shape),
            "shapes_shape": list(shapes.shape),
            "rh_mapped_root_axis_angle": rh.tolist(),
            "th_raw_m": th.tolist(),
            "root_align_offset_m": root_align.tolist(),
            "betas_float32_sha256": _array_sha256(betas),
            "pose55_float32_sha256": _array_sha256(pose),
            "pose55_hash": smplx_pose_hash(pose, th + root_align),
            "translation_applied_to_exported_geometry": False,
        }
    return np.asarray(pose, dtype=np.float32), record


def _scan_among_us(root: Path) -> dict[str, Any]:
    """Record candidate capture files without using them as inputs."""

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
                artifacts.append({
                    "path": str(path),
                    "sha256": _sha256(path),
                })
            except OSError as exc:
                artifacts.append({
                    "path": str(path),
                    "status": "unreadable",
                    "error": f"{type(exc).__name__}: {exc}",
                })
    matching = [row for row in artifacts if row.get("sha256") in set(CAPTURE_SHA256.values())]
    return {
        "root": str(root),
        "patterns": list(patterns),
        "artifact_count": len(artifacts),
        "matching_frozen_capture_sha256": matching,
        "status": "matched" if matching else "no_direct_sha_match",
        "substitution_policy": "never_substitute_Among_US_artifact_for_frozen_repo_capture",
    }


def _assert_v7_subject_manifest(
    root: Path,
    *,
    subject: str,
    operator_digest: str,
    calibration_digest: str,
    oracle_sha: str,
    model_sha: str,
) -> tuple[dict[str, Any], Path]:
    """Fail closed on the existing V7 bundle before invoking its loader."""

    root = Path(root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    npz_path = root / "whole_chain_rest_fit_subject_v1.npz"
    if not manifest_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError(
            f"frozen V7 subject {subject} is incomplete: expected {manifest_path} and {npz_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    build = manifest.get("build_report", {})
    checks = {
        "artifact_kind": manifest.get("artifact_kind") == "WholeChainRestFitSubjectV1",
        "schema_version": int(manifest.get("schema_version", -1)) == 1,
        "baseline_commit": manifest.get("baseline_commit") == BASELINE_COMMIT,
        "operator_digest": manifest.get("source_operator_digest") == operator_digest,
        "calibration_digest": manifest.get("calibration_digest") == calibration_digest,
        "oracle_sha256": manifest.get("blender_oracle_sha256") == oracle_sha,
        "model_sha256": manifest.get("smplx_model_sha256") == model_sha,
        "capture_sha256": manifest.get("capture_sha256") == CAPTURE_SHA256[subject],
        "capture_sha256s": manifest.get("capture_sha256s") == CAPTURE_SHA256,
        "accepted_scope": manifest.get("accepted_scope") == "full_main_chain_shadow",
        "complete": manifest.get("complete") is True,
        "publishable": manifest.get("publishable") is False,
        "trusted_latest_updated": manifest.get("trusted_latest_updated") is False,
        "vessel_repair_started": manifest.get("vessel_repair_started") is False,
        "subject_label": str(manifest.get("subject_label")) == subject,
        "method_is_v7": "v7" in str(build.get("method", "")).lower(),
        "pose_map_is_right_multiply": "right_multiply" in str(build.get("pose_map_composition", "")).lower(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"frozen V7 subject {subject} manifest mismatch: {failed}")
    npz_sha = _sha256(npz_path)
    if npz_sha != manifest.get("npz_sha256"):
        raise ValueError(
            f"frozen V7 subject {subject} NPZ SHA mismatch: "
            f"expected {manifest.get('npz_sha256')}, got {npz_sha}"
        )
    return manifest, npz_path


def _pose_skin_joints(
    model: Mapping[str, np.ndarray], *, betas: np.ndarray, pose: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    skin, skin_faces = smplx_body_surface_v7(
        model,
        betas=np.asarray(betas, dtype=np.float64),
        pose_axis_angle=np.asarray(pose, dtype=np.float32),
    )
    _rest_joints, globals_, _rest_to_pose = _smplx_joint_kinematics_v7(
        model,
        betas=np.asarray(betas, dtype=np.float64),
        pose_axis_angle=np.asarray(pose, dtype=np.float32),
    )
    return (
        np.asarray(skin, dtype=np.float32),
        np.asarray(skin_faces, dtype=np.int32),
        np.asarray(globals_, dtype=np.float32)[:, :3, 3],
    )


def _raw142_source_pose(subject_runtime: Any, pose: np.ndarray) -> np.ndarray:
    """Evaluate the materialized 142 source runtime exactly once."""

    evaluator = ResidentPoseEvaluatorV8(subject_runtime, validate=False)
    result = evaluator.apply_pose(np.asarray(pose, dtype=np.float32))
    result = np.asarray(result, dtype=np.float32)
    if result.shape != subject_runtime.rigged_asset.vertices_rest.shape:
        raise ValueError("raw 142 source runtime returned an unexpected vertex shape")
    if not np.all(np.isfinite(result)):
        raise ValueError("raw 142 source runtime returned non-finite vertices")
    return result


def _save_cell(
    path: Path,
    *,
    subject: str,
    pose_name: str,
    pose: np.ndarray,
    source_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    skin_vertices: np.ndarray,
    smplx_joints: np.ndarray,
    faces: np.ndarray,
    skin_faces: np.ndarray,
    vertex_tissue: np.ndarray,
    capture_record: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        capture_translation_applied=np.asarray(False),
        pose_name=np.asarray(str(pose_name)),
        subject=np.asarray(str(subject)),
        metadata_json=np.asarray(json.dumps(_json_ready(metadata), sort_keys=True, allow_nan=False)),
    )
    source = np.asarray(source_vertices, dtype=np.float32)
    candidate = np.asarray(candidate_vertices, dtype=np.float32)
    delta = np.linalg.norm(candidate.astype(np.float64) - source.astype(np.float64), axis=1)
    return {
        "file": str(path),
        "sha256": _sha256(path),
        "array_sha256": {
            "faces": _array_sha256(faces),
            "skin_faces": _array_sha256(skin_faces),
            "source_vertices": _array_sha256(source),
            "candidate_vertices": _array_sha256(candidate),
            "skin_vertices": _array_sha256(skin_vertices),
            "smplx_joints": _array_sha256(smplx_joints),
            "pose": _array_sha256(pose),
            "vertex_tissue": _array_sha256(vertex_tissue),
        },
        "source_candidate_max_displacement_m": float(np.max(delta)),
        "source_candidate_rms_displacement_m": float(np.sqrt(np.mean(delta**2))),
        "source_candidate_vertex_count": int(len(source)),
        "publishable": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--smplx-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--capture-213328", type=Path, default=DEFAULT_CAPTURE_213328)
    parser.add_argument("--capture-213712", type=Path, default=DEFAULT_CAPTURE_213712)
    parser.add_argument("--v7", type=Path, default=DEFAULT_V7)
    parser.add_argument("--among-us-root", type=Path, default=DEFAULT_AMONG_US)
    parser.add_argument("--subjects", default=",".join(SUBJECTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
    captures: Mapping[str, Mapping[str, Any]],
    capture_poses: Mapping[str, np.ndarray],
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    v7_root = Path(args.v7).expanduser().resolve() / f"subject_{subject}"
    v7_manifest, v7_npz = _assert_v7_subject_manifest(
        v7_root,
        subject=subject,
        operator_digest=operator_digest,
        calibration_digest=calibration_digest,
        oracle_sha=_sha256(oracle),
        model_sha=str(require_frozen_smplx_male_v7(model_path)[1]),
    )
    with np.load(Path(args.capture_213328 if subject == "213328" else args.capture_213712).expanduser().resolve(), allow_pickle=False) as data:
        capture_betas = np.asarray(data["shapes"], dtype=np.float64).reshape(-1)[:10]
    v7_value = load_whole_chain_rest_fit_v1(
        v7_root,
        operator=operator,
        calibration=calibration,
        smplx_model=model,
        smplx_model_sha256=str(require_frozen_smplx_male_v7(model_path)[1]),
        recheck=False,
    )
    if not np.array_equal(np.asarray(v7_value.betas), capture_betas):
        raise ValueError(f"V7 subject {subject} betas differ from frozen capture")
    materialized = materialize_subject(operator, betas=capture_betas, gender="male")
    asset = materialized.rigged_asset
    asset_digest = rigged_asset_digest(asset)
    pose_map = build_pose_map_v1(
        v7_value,
        asset=asset,
        calibration=calibration,
        oracle_path=oracle,
        source_operator_digest=operator_digest,
    )
    map_digest = pose_map_content_digest(pose_map)
    tissue = _tissue_codes(asset)
    poses = {
        "tpose": np.zeros((55, 3), dtype=np.float32),
        "pose_213328": np.asarray(capture_poses["213328"], dtype=np.float32),
        "pose_213712": np.asarray(capture_poses["213712"], dtype=np.float32),
    }
    files: dict[str, Any] = {}
    for pose_name, pose in poses.items():
        source = _raw142_source_pose(materialized, pose)
        candidate, _candidate_globals = pose_whole_chain_vertices(
            v7_value,
            pose_map,
            source_asset=asset,
            pose_axis_angle=pose,
        )
        skin, skin_faces, joints = _pose_skin_joints(
            model,
            betas=capture_betas,
            pose=pose,
        )
        metadata = {
            "schema_version": 14,
            "artifact_kind": "ConsistentBaselineGeometryComparisonV14",
            "subject": subject,
            "pose": pose_name,
            "source_label": "raw142_source_runtime",
            "source_pose_authority": "ResidentPoseEvaluatorV8(materialize_subject(SourceOperatorV8))",
            "candidate_label": "frozen_V7_pose_map_v1_right_multiply",
            "candidate_pose_authority": "pose_whole_chain_vertices(PoseMapV1)",
            "pose_map_composition": "right_multiply_bind_v6: G_prime=G_source @ inv(B_source) @ B_target",
            "pose_map_version": "PoseMapV1",
            "pose_map_content_digest": map_digest,
            "v7_method": v7_manifest["build_report"].get("method"),
            "v7_femur_axial_scale_policy": v7_manifest["build_report"].get("femur_axial_scale_policy"),
            "v7_terminal_policy": v7_manifest["build_report"].get("terminal_hand_policy"),
            "v7_subject_root": str(v7_root),
            "v7_subject_manifest_sha256": _sha256(v7_root / "manifest.json"),
            "v7_subject_npz_sha256": _sha256(v7_npz),
            "v7_subject_content_digest": v7_manifest.get("content_digest"),
            "source_operator_digest": operator_digest,
            "calibration_digest": calibration_digest,
            "blender_oracle_sha256": _sha256(oracle),
            "smplx_model_sha256": str(require_frozen_smplx_male_v7(model_path)[1]),
            "materialized_asset_digest": asset_digest,
            "capture_provenance": captures[subject],
            "capture_translation_applied": False,
            "v10_pose_map_used": False,
            "v10_runtime_used": False,
            "reoptimized": False,
            "anatomical_pass": False,
            "publishable": False,
            "audit_reuse": {
                "articular_surface_cli": "audit_capture_articular_surfaces_v13.py",
                "signed_bone_skin_metrics": "joint_plausibility_metrics_v12 plus existing signed-distance surface metrics",
                "query_focus": [
                    "left/right shoulder elbow wrist",
                    "left/right hip knee",
                ],
            },
        }
        filename = f"subject_{subject}_{pose_name}.npz"
        files[pose_name] = _save_cell(
            output / filename,
            subject=subject,
            pose_name=pose_name,
            pose=pose,
            source_vertices=source,
            candidate_vertices=np.asarray(candidate, dtype=np.float32),
            skin_vertices=skin,
            smplx_joints=joints,
            faces=asset.faces,
            skin_faces=skin_faces,
            vertex_tissue=tissue,
            capture_record=captures[subject],
            metadata=metadata,
        )
        print(
            f"subject={subject} pose={pose_name} exported raw142_vs_v7 "
            f"delta_max={files[pose_name]['source_candidate_max_displacement_m'] * 1000.0:.3f}mm",
            flush=True,
        )
    return {
        "subject": subject,
        "status": "complete",
        "publishable": False,
        "anatomical_pass": False,
        "v7_subject_root": str(v7_root),
        "v7_subject_manifest_sha256": _sha256(v7_root / "manifest.json"),
        "v7_subject_npz_sha256": _sha256(v7_npz),
        "v7_subject_content_digest": v7_manifest.get("content_digest"),
        "v7_method": v7_manifest["build_report"].get("method"),
        "pose_map_composition": "right_multiply_bind_v6",
        "pose_map_content_digest": map_digest,
        "materialized_asset_digest": asset_digest,
        "cells": files,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    subjects = tuple(part.strip() for part in str(args.subjects).split(",") if part.strip())
    unknown = sorted(set(subjects) - set(SUBJECTS))
    if not subjects or unknown:
        raise ValueError(f"--subjects must select from {SUBJECTS}; unknown={unknown}")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite baseline export: {output}")
    started = time.perf_counter()
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        operator = load_source_operator(Path(args.operator).expanduser().resolve(), validate=True, mmap=True)
        operator_digest = operator.runtime_digest(validate=False)
        if operator_digest != EXPECTED_OPERATOR_RUNTIME_DIGEST:
            raise ValueError(
                f"baseline export requires frozen 142 operator {EXPECTED_OPERATOR_RUNTIME_DIGEST}, got {operator_digest}"
            )
        calibration = load_anatomical_calibration_v1(
            Path(args.calibration).expanduser().resolve(),
            operator=operator,
            required_scope="full_main_chain",
        )
        calibration_digest = _calibration_content_digest(calibration)
        oracle = Path(args.oracle).expanduser().resolve()
        oracle_sha = _sha256(oracle)
        if oracle_sha != EXPECTED_ORACLE_SHA256:
            raise ValueError(f"frozen Blender oracle SHA mismatch: expected {EXPECTED_ORACLE_SHA256}, got {oracle_sha}")
        model_path, model_sha = require_frozen_smplx_male_v7(Path(args.smplx_model).expanduser().resolve())
        model = load_smplx_model_v7(model_path)
        capture_paths = {
            "213328": Path(args.capture_213328),
            "213712": Path(args.capture_213712),
        }
        capture_poses: dict[str, np.ndarray] = {}
        captures: dict[str, dict[str, Any]] = {}
        for label, path in capture_paths.items():
            pose, record = _capture_pose(path, model_path=model_path, subject=label)
            capture_poses[label] = pose
            captures[label] = record
        report: dict[str, Any] = {
            "schema_version": 14,
            "artifact_kind": "ConsistentBaselineGeometryMatrixV14",
            "status": "running",
            "publishable": False,
            "anatomical_pass": False,
            "trusted_latest_updated": False,
            "source_contract": "raw142_source_runtime",
            "candidate_contract": "frozen_V7_pose_map_v1_right_multiply",
            "pose_map_composition": "right_multiply_bind_v6",
            "no_reoptimization": True,
            "operator": str(Path(args.operator).expanduser().resolve()),
            "operator_runtime_digest": operator_digest,
            "calibration": str(Path(args.calibration).expanduser().resolve()),
            "calibration_digest": calibration_digest,
            "oracle": str(oracle),
            "oracle_sha256": oracle_sha,
            "smplx_model": str(model_path),
            "smplx_model_sha256": model_sha,
            "baseline_commit": BASELINE_COMMIT,
            "v7_root": str(Path(args.v7).expanduser().resolve()),
            "repo_captures": captures,
            "among_us_capture_scan": _scan_among_us(Path(args.among_us_root)),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "git_head": _git_head(),
            "subjects": {},
            "failures": [],
        }
        for subject in subjects:
            report["subjects"][subject] = _run_subject(
                subject,
                args=args,
                operator=operator,
                calibration=calibration,
                model=model,
                model_path=model_path,
                operator_digest=operator_digest,
                calibration_digest=calibration_digest,
                oracle=oracle,
                captures=captures,
                capture_poses=capture_poses,
                output=temporary,
            )
            _write_json(temporary / "progress.json", report)
        report["status"] = "complete"
        report["passed"] = bool(
            report["subjects"]
            and not report["failures"]
            and all(row.get("status") == "complete" for row in report["subjects"].values())
        )
        report["elapsed_seconds"] = float(time.perf_counter() - started)
        _write_json(temporary / "matrix_manifest.json", report)
        _write_json(temporary / "progress.json", report)
        os.replace(temporary, output)
    except Exception as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        print(
            json.dumps(
                {
                    "artifact_kind": "ConsistentBaselineGeometryMatrixV14",
                    "status": "preflight_or_export_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "publishable": False,
                    "anatomical_pass": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2
    print(f"ConsistentBaselineGeometryMatrixV14 passed=true subjects={len(subjects)} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
